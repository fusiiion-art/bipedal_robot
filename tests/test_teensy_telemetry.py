import struct

import numpy as np

from real.real_io import TeensySpineIO


class FakeSerial:
    def __init__(self, response):
        self.response = response

    def write(self, _data):
        pass

    def read(self, _length):
        return self.response


def make_packet(gyro=(1.0, 2.0, 3.0), accel=(0.0, 0.0, 9.80665), fsr=None):
    if fsr is None:
        fsr = (1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0)
    payload = struct.pack("<4f", 1.0, 0.0, 0.0, 0.0)
    payload += struct.pack("<3f", *gyro)
    payload += struct.pack("<3f", *accel)
    payload += struct.pack("<8f", *fsr)
    return bytes([TeensySpineIO.TELEMETRY_START_BYTE]) + payload


def make_io(response):
    io = TeensySpineIO.__new__(TeensySpineIO)
    io.num_servos = 20
    io.ser = FakeSerial(response)
    io.dummy_mode = False
    io.last_imu_data = {
        "quat": np.array([1.0, 0.0, 0.0, 0.0]),
        "gyro": np.zeros(3),
        "lin_accel": np.zeros(3),
    }
    io.last_fsr_contacts = np.zeros(8, dtype=np.float32)
    io.servo_temps = np.full(20, 25.0)
    io.servo_voltages = np.full(20, 11.1)
    io.telemetry_timeout_flag = False
    io._consecutive_timeouts = 0
    return io


def test_valid_telemetry_updates_values():
    io = make_io(make_packet())

    io.communicate(np.zeros(20))

    np.testing.assert_allclose(io.last_imu_data["gyro"], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(io.last_imu_data["lin_accel"], [0.0, 0.0, 9.80665])
    np.testing.assert_array_equal(io.last_fsr_contacts, [1, 0, 1, 0, 1, 0, 1, 0])
    assert io._consecutive_timeouts == 0


def test_out_of_range_telemetry_keeps_last_good_values():
    io = make_io(make_packet())
    io.communicate(np.zeros(20))
    last_imu = {key: value.copy() for key, value in io.last_imu_data.items()}
    last_fsr = io.last_fsr_contacts.copy()

    io.ser.response = make_packet(gyro=(51.0, 0.0, 0.0))
    io.communicate(np.zeros(20))

    for key in last_imu:
        np.testing.assert_array_equal(io.last_imu_data[key], last_imu[key])
    np.testing.assert_array_equal(io.last_fsr_contacts, last_fsr)
    assert io._consecutive_timeouts == 1
    assert not io.telemetry_timeout_flag


def test_three_rejected_packets_raise_timeout_flag():
    io = make_io(make_packet(gyro=(51.0, 0.0, 0.0)))

    for _ in range(3):
        io.communicate(np.zeros(20))

    assert io.telemetry_timeout_flag
