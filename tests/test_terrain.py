"""
MuJoCo 不整地テスト

ハイトフィールドを生成して、その上にロボットを配置するテスト。
MuJoCoでは hfield アセットとしてハイトフィールドを定義する。
"""
import mujoco
import mujoco.viewer
import numpy as np
import time


def create_terrain_model_xml():
    """不整地ありのMuJoCoモデルXMLを動的に生成する"""
    # ハイトフィールドデータの生成
    nrow, ncol = 64, 64
    heightfield = np.random.uniform(0, 0.05, size=(nrow, ncol)).astype(np.float32)
    # 中央を平坦にする (ロボット初期位置)
    center = nrow // 2
    radius = 8
    heightfield[center-radius:center+radius, center-radius:center+radius] = 0.0

    xml = f"""
<mujoco model="terrain_test">
  <option timestep="0.00416667" gravity="0 0 -9.8"/>

  <asset>
    <hfield name="terrain" nrow="{nrow}" ncol="{ncol}"
            size="5 5 0.1 0.01"/>
  </asset>

  <worldbody>
    <!-- 不整地 -->
    <geom name="floor" type="hfield" hfield="terrain"
          rgba="0.6 0.8 0.6 1" friction="1.0 0.5 0.5"/>

    <!-- 簡易ロボット (humanoid風) -->
    <body name="torso" pos="0 0 0.6">
      <freejoint name="root"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.2" size="0.05" mass="2.0"
            rgba="0.2 0.6 1.0 1"/>

      <body name="right_thigh" pos="0.05 0 0">
        <joint name="right_hip" type="hinge" axis="0 1 0"
               range="-1.57 1.57" damping="0.5"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.15" size="0.03" mass="0.5"
              rgba="1.0 0.4 0.2 1"/>
        <body name="right_shin" pos="0 0 -0.15">
          <joint name="right_knee" type="hinge" axis="0 1 0"
                 range="-2.0 0" damping="0.5"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.15" size="0.025" mass="0.3"
                rgba="1.0 0.6 0.3 1"/>
        </body>
      </body>

      <body name="left_thigh" pos="-0.05 0 0">
        <joint name="left_hip" type="hinge" axis="0 1 0"
               range="-1.57 1.57" damping="0.5"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.15" size="0.03" mass="0.5"
              rgba="0.2 1.0 0.4 1"/>
        <body name="left_shin" pos="0 0 -0.15">
          <joint name="left_knee" type="hinge" axis="0 1 0"
                 range="-2.0 0" damping="0.5"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.15" size="0.025" mass="0.3"
                rgba="0.4 1.0 0.6 1"/>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <position name="right_hip_act" joint="right_hip"
              kp="20" ctrlrange="-1.57 1.57"/>
    <position name="right_knee_act" joint="right_knee"
              kp="20" ctrlrange="-2.0 0"/>
    <position name="left_hip_act" joint="left_hip"
              kp="20" ctrlrange="-1.57 1.57"/>
    <position name="left_knee_act" joint="left_knee"
              kp="20" ctrlrange="-2.0 0"/>
  </actuator>
</mujoco>
"""
    return xml, heightfield, nrow, ncol


def main():
    print("=== MuJoCo Terrain Test ===")

    # 1. モデル生成
    xml_str, heightfield, nrow, ncol = create_terrain_model_xml()
    model = mujoco.MjModel.from_xml_string(xml_str)

    # 2. ハイトフィールドデータを設定
    model.hfield_data[:] = heightfield.ravel()

    # 3. データオブジェクト生成
    data = mujoco.MjData(model)

    print(f"Heightfield: {nrow}x{ncol}, range=[0, 0.05]m")
    print(f"Joints: {model.njnt}, Actuators: {model.nu}")

    # 4. ビューア起動
    print("Launching viewer... (Close window to stop)")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        t = 0
        while viewer.is_running():
            # 簡易歩行動作 (サイン波)
            target = 0.3 * np.sin(t * 5.0)
            data.ctrl[0] = target    # right_hip
            data.ctrl[1] = -target   # right_knee
            data.ctrl[2] = -target   # left_hip
            data.ctrl[3] = target    # left_knee

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(1./240.)
            t += 1./240.

    print("Test finished.")


if __name__ == "__main__":
    main()