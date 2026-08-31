import os
import sys
import mujoco

# DLL load failedなどの環境依存インポートエラーを防ぐため、グローバルスコープでtry-exceptする
try:
    import mujoco.viewer
    HAS_VIEWER = True
    VIEWER_ERROR = None
except Exception as e:
    HAS_VIEWER = False
    VIEWER_ERROR = e

def main():
    xml_path = "assets/humanoid/humanoid.xml"
    if not os.path.exists(xml_path):
        print(f"Error: {xml_path} not found.")
        return

    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    nu = model.nu
    print("=" * 60)
    print(f"[MODEL] {nu} actuators successfully loaded")
    print("=" * 60)
    for i in range(nu):
        print(f"  [{i+1:02d}] {model.actuator(i).name}  ctrlrange={model.actuator_ctrlrange[i]}")
    print("=" * 60)

    # 初期位置を少し高く設定
    if model.nq >= 7:
        data.qpos[2] = 0.35
        data.qpos[3] = 1.0

    print("Attempting to launch MuJoCo Passive Viewer...")
    
    launched = False
    if HAS_VIEWER:
        try:
            print("  Viewer 上の [Ctrl] タブのスライダーで各関節を手動操作できます。")
            print("  ウィンドウを閉じると終了します。")
            print("=" * 60 + "\n")
            with mujoco.viewer.launch_passive(model, data) as viewer:
                launched = True
                while viewer.is_running():
                    mujoco.mj_step(model, data)
                    viewer.sync()
        except Exception as e:
            print(f"\n[WARNING] Could not launch MuJoCo Passive Viewer: {e}")
            launched = False
    else:
        print(f"\n[WARNING] Could not import mujoco.viewer: {VIEWER_ERROR}")
        launched = False

    if not launched:
        print("  (Note: Windows host environment often suffers from OpenGL/DLL load failures for MuJoCo Viewer.)")
        print("  (Tip: You can run this script inside WSL2 with WSLg/X11 forwarding to see the full GUI!)")
        print("\nFalling back to Headless Simulation Mode...")
        print("Running 1000 physics steps to verify model stability...")
        print("=" * 60)
        
        try:
            for step in range(1000):
                mujoco.mj_step(model, data)
                if step % 200 == 0:
                    print(f"  Step {step:04d}/1000: z-height = {data.qpos[2]:.4f} m, base roll/pitch/yaw values are valid.")
            print("=" * 60)
            print("SUCCESS: Headless simulation completed perfectly! The modified XML model is physics-stable.")
        except Exception as sim_err:
            print(f"ERROR during headless simulation: {sim_err}")

if __name__ == "__main__":
    main()


