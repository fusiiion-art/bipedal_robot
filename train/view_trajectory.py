import os
import sys
import time
import numpy as np

# Windowsでmujocoのプラグイン読込時にDLLブロックエラーが出るのを回避するパッチ
try:
    import ctypes
    import mujoco._structs
    # プラグインロードを安全に無効化
    def dummy_load_plugins():
        pass
    import mujoco
    mujoco._load_all_bundled_plugins = dummy_load_plugins
except Exception:
    pass

import mujoco
import mujoco.viewer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from robot.config import RobotConfig

def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    traj_path = os.path.join(root_dir, "trajectory.npy")
    
    if not os.path.exists(traj_path):
        print("エラー: 軌跡データ(trajectory.npy)が見つかりません。")
        print("先にWSL側で 'python3 train/export_trajectory.py' を実行してください。")
        return
        
    print(f"軌跡データを読み込み中: {traj_path}")
    traj = np.load(traj_path)
    
    # ネイティブのMuJoCoはWindowsでも全く問題なく動く (JAXが不要だから)
    mj_model_native = mujoco.MjModel.from_xml_path(str(RobotConfig.MUJOCO_MODEL_PATH))
    mj_data_native = mujoco.MjData(mj_model_native)
    
    print("Windowsネイティブビューワーを起動します！")
    with mujoco.viewer.launch_passive(mj_model_native, mj_data_native) as viewer:
        # 無限ループでリプレイ再生
        while viewer.is_running():
            print("リプレイ再生を開始...")
            for qpos in traj:
                if not viewer.is_running():
                    break
                
                step_start = time.time()
                
                # 状態をセットして順運動学を計算（画面描画の更新）
                mj_data_native.qpos[:] = qpos
                mujoco.mj_forward(mj_model_native, mj_data_native)
                viewer.sync()
                
                # スピード調整 (100Hz)
                time_until_next = RobotConfig.CONTROL_DT - (time.time() - step_start)
                if time_until_next > 0:
                    time.sleep(time_until_next)
            time.sleep(1) # 再生終了後に1秒待って最初から

if __name__ == "__main__":
    main()
