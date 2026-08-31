#!/usr/bin/env python3
"""
Validation script for improved reward system
テスト：改善された報酬システムの動作確認
"""

import sys
sys.path.insert(0, '/c/bipedal_robot')

import jax
import jax.numpy as jp
from robot.config import RobotConfig
from envs.stability_metrics import StabilityMetrics

def test_curriculum_learning():
    """Test curriculum learning schedule"""
    print("=" * 60)
    print("TEST 1: Curriculum Learning Schedule")
    print("=" * 60)
    
    schedule = RobotConfig.CURRICULUM_SCHEDULE
    test_steps = [0, 50000, 100000, 300000, 500000, 1000000, 2000000, 5000000]
    
    for step in test_steps:
        # スケジュール内のキーをソート
        keys = sorted(schedule.keys())
        scale = schedule[keys[0]]
        for key in keys:
            if step >= key:
                scale = schedule[key]
        
        force = RobotConfig.RANDOM_PUSH_MAX_FORCE * scale
        print(f"Step {step:8d}: scale={scale:.2f}, max_force={force:.2f}N")
    
    print("✓ Curriculum learning schedule validated\n")

def test_stability_metrics():
    """Test stability metrics computation"""
    print("=" * 60)
    print("TEST 2: Stability Metrics")
    print("=" * 60)
    
    # Initialize metrics
    metrics = StabilityMetrics(left_foot_id=0, right_foot_id=1, com_height=0.28)
    
    # Stable state
    com_pos = jp.array([0.0, 0.0, 0.28])
    com_vel = jp.array([0.0, 0.0, 0.0])
    com_accel = jp.array([0.0, 0.0, -9.81])
    rpy = jp.array([0.0, 0.0, 0.0])
    base_ang_vel = jp.array([0.0, 0.0, 0.0])
    left_foot_pos = jp.array([-0.05, 0.0, 0.0])
    right_foot_pos = jp.array([0.05, 0.0, 0.0])
    left_foot_force = jp.array(50.0)
    right_foot_force = jp.array(50.0)
    
    stability_index, metrics_dict = metrics.compute_unified_stability_index(
        com_pos, com_vel, com_accel, rpy, base_ang_vel,
        left_foot_pos, right_foot_pos,
        left_foot_force, right_foot_force,
        gait_phase=0.5
    )
    
    print(f"Stable state:")
    print(f"  Stability Index: {float(stability_index):.4f}")
    print(f"  CP Margin:       {float(metrics_dict['cp_margin']):.4f}")
    print(f"  ZMP Margin:      {float(metrics_dict['zmp_margin']):.4f}")
    print(f"  Foot Balance:    {float(metrics_dict['foot_balance']):.4f}")
    print(f"  Orient Margin:   {float(metrics_dict['orient_margin']):.4f}")
    
    # Tilted state
    rpy_tilted = jp.array([0.2, 0.0, 0.0])
    stability_index_tilted, metrics_tilted = metrics.compute_unified_stability_index(
        com_pos, com_vel, com_accel, rpy_tilted, base_ang_vel,
        left_foot_pos, right_foot_pos,
        left_foot_force, right_foot_force,
        gait_phase=0.5
    )
    
    print(f"\nTilted state (roll=0.2):")
    print(f"  Stability Index: {float(stability_index_tilted):.4f}")
    print(f"  CP Margin:       {float(metrics_tilted['cp_margin']):.4f}")
    print(f"  Orient Margin:   {float(metrics_tilted['orient_margin']):.4f}")
    
    assert float(stability_index) > float(stability_index_tilted), \
        "Tilted state should have lower stability"
    
    print("✓ Stability metrics validated\n")

def test_adaptive_scaling():
    """Test adaptive reward scaling"""
    print("=" * 60)
    print("TEST 3: Adaptive Reward Scaling")
    print("=" * 60)
    
    from envs.mjx_rewards import MJXRewardSystem
    
    # Create dummy reward system to test scaling
    class DummyModel:
        nq = 7
        nu = 6
    
    reward_system = MJXRewardSystem(
        DummyModel(), 
        RobotConfig.REWARD_WEIGHTS,
        left_foot_id=0,
        right_foot_id=1
    )
    
    # Normal conditions
    servo_temp_normal = jp.full(6, 40.0)
    volt_normal = 11.1
    scaling_normal = reward_system._compute_adaptive_reward_scaling(servo_temp_normal, volt_normal)
    
    print(f"Normal conditions (T=40°C, V=11.1V):")
    print(f"  Recovery scale:   {float(scaling_normal['recovery']):.4f}")
    print(f"  Energy scale:     {float(scaling_normal['energy']):.4f}")
    print(f"  Smoothness scale: {float(scaling_normal['smoothness']):.4f}")
    
    # High temperature
    servo_temp_hot = jp.full(6, 75.0)
    scaling_hot = reward_system._compute_adaptive_reward_scaling(servo_temp_hot, volt_normal)
    
    print(f"\nHigh temperature (T=75°C, V=11.1V):")
    print(f"  Recovery scale:   {float(scaling_hot['recovery']):.4f}")
    print(f"  Energy scale:     {float(scaling_hot['energy']):.4f}")
    print(f"  Smoothness scale: {float(scaling_hot['smoothness']):.4f}")
    
    # Low voltage
    servo_temp_normal = jp.full(6, 40.0)
    volt_low = 9.2
    scaling_low = reward_system._compute_adaptive_reward_scaling(servo_temp_normal, volt_low)
    
    print(f"\nLow voltage (T=40°C, V=9.2V):")
    print(f"  Recovery scale:   {float(scaling_low['recovery']):.4f}")
    print(f"  Energy scale:     {float(scaling_low['energy']):.4f}")
    print(f"  Smoothness scale: {float(scaling_low['smoothness']):.4f}")
    
    # Verify scaling directions
    assert float(scaling_hot['recovery']) > 1.0, "Recovery should be boosted under high temp"
    assert float(scaling_hot['energy']) < 1.0, "Energy penalty should be reduced under high temp"
    
    print("✓ Adaptive reward scaling validated\n")


def test_stance_penalty_discourages_wide_foot_spacing():
    """広い足幅のまま停止する局所最適を避けるため、足幅の広がりにペナルティが付くことを確認する。"""
    from envs.mjx_rewards import MJXRewardSystem

    class DummyModel:
        nq = 7
        nu = 6

    reward_system = MJXRewardSystem(
        DummyModel(),
        RobotConfig.REWARD_WEIGHTS,
        left_foot_id=0,
        right_foot_id=1,
    )

    class DummyData:
        def __init__(self, left_pos, right_pos):
            self.qpos = jp.array([0.0, 0.0, 0.28, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            self.qvel = jp.zeros(6)
            self.actuator_force = jp.zeros(6)
            self.qacc = jp.array([0.0, 0.0, 0.0])
            self.sensordata = jp.zeros(8)
            self.xpos = jp.array([left_pos, right_pos])
            self.subtree_com = jp.array([[0.0, 0.0, 0.28]])

    narrow_data = DummyData(jp.array([-0.05, 0.0, 0.0]), jp.array([0.05, 0.0, 0.0]))
    wide_data = DummyData(jp.array([-0.12, 0.0, 0.0]), jp.array([0.12, 0.0, 0.0]))

    narrow_reward, _, _, _ = reward_system.compute(
        narrow_data,
        action=jp.zeros(6),
        last_action=jp.zeros(6),
        double_last_action=jp.zeros(6),
        triple_last_action=jp.zeros(6),
        cbf_penalty=jp.array(0.0),
        last_potential=jp.array(0.0),
        step=jp.array(10),
        reference_action=jp.zeros(6),
        servo_temp=jp.full(6, 40.0),
        supply_volt=11.1,
        global_step=jp.array(1000),
        gait_phase=0.5,
        was_disturbed=jp.array(False),
        disturbance_recovery_steps=jp.array(1000),
        training_progress=jp.array(0.5),
    )

    wide_reward, _, _, _ = reward_system.compute(
        wide_data,
        action=jp.zeros(6),
        last_action=jp.zeros(6),
        double_last_action=jp.zeros(6),
        triple_last_action=jp.zeros(6),
        cbf_penalty=jp.array(0.0),
        last_potential=jp.array(0.0),
        step=jp.array(10),
        reference_action=jp.zeros(6),
        servo_temp=jp.full(6, 40.0),
        supply_volt=11.1,
        global_step=jp.array(1000),
        gait_phase=0.5,
        was_disturbed=jp.array(False),
        disturbance_recovery_steps=jp.array(1000),
        training_progress=jp.array(0.5),
    )

    print(f"Narrow stance reward: {float(narrow_reward):.4f}")
    print(f"Wide stance reward:   {float(wide_reward):.4f}")

    assert float(wide_reward) < float(narrow_reward), "Wide stance should be penalized"
    print("✓ Wide-stance penalty validated\n")

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("REWARD SYSTEM IMPROVEMENT VALIDATION")
    print("=" * 60 + "\n")
    
    try:
        test_curriculum_learning()
        test_stability_metrics()
        test_adaptive_scaling()
        test_stance_penalty_discourages_wide_foot_spacing()
        
        print("=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
    except Exception as e:
        print(f"✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
