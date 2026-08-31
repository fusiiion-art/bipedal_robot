import os
import mujoco
import sys

def main():
    # 新しく作成した humanoid.xml のパス
    model_path = os.path.join(os.path.dirname(__file__), "..", "assets", "humanoid", "humanoid.xml")
    model_path = os.path.abspath(model_path)
    
    print(f"Loading and validating MuJoCo model: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        sys.exit(1)
        
    try:
        # MuJoCoパーサーにXMLを読み込ませる
        model = mujoco.MjModel.from_xml_path(model_path)
        print("\n=== SUCCESS: MuJoCo Model Loaded Perfectly! ===")
        print(f"Model Name        : {model.names}")
        print(f"Total Joints (nq) : {model.nq} (includes freejoint 7DoF + 20 hinge joints)")
        print(f"Total Actuators   : {model.nu}")
        print(f"Total Sensors     : {model.nsensor}")
        print(f"Total Bodies      : {model.nbody}")
        print(f"Total Geoms       : {model.ngeom}")
        print("===============================================")
    except Exception as e:
        print(f"\nERROR validating model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
