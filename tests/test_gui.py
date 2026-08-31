"""tests/test_gui.py - MuJoCo GUIビジュアルテスト (旧 train/test_gui.py)"""
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from envs.base_env import MuJoCoSim

print("--- STEP 1: MuJoCo GUI起動試行 ---")
try:
    sim = MuJoCoSim(render=True)
    print(">>> 成功: MuJoCoビューアが開きました")
except Exception as e:
    print(f"XXX 失敗: {e}")
    exit()

print("--- STEP 2: シミュレーションリセット ---")
sim.reset()
state = sim.get_state_dict()
print(f"  Base Position: {state['base_pos']}")
print(f"  RPY: {state['rpy']}")
print(f"  Joint Positions: {state['joint_positions']}")

print("--- STEP 3: ループ開始 (Ctrl+Cで停止) ---")
try:
    while True:
        sim.apply_action(sim.data.ctrl * 0)
        time.sleep(1./240.)
except KeyboardInterrupt:
    print("\n停止しました")
finally:
    sim.close()
