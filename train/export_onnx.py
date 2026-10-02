import os
import sys
import pickle
import argparse
import shutil

import jax
import tensorflow as tf
from jax.experimental import jax2tf

# append project root to sys path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from robot.config import RobotConfig
from robot.policy_network import POLICY_OBS_KEY, load_checkpoint, make_inference_fn_from_params


def load_brax_inference_fn(pkl_path):
    """Brax(Flax)の保存済みパラメータから、actor観測(625次元)→行動 の推論関数を作る。

    観測正規化(学習時の running statistics)と平均クリップ付きの分布は
    make_inference_fn_from_params() が学習時と同じ構成で組み込む。
    critic 用の特権観測は actor が読まないため、ONNX の入力には含めない。"""
    params = load_checkpoint(pkl_path)
    print("[INFO] Params successfully loaded from pickle.")
    inf_fn = make_inference_fn_from_params(params, deterministic=True)

    def predict(obs):
        # deterministic なので rng は使われない
        action, _ = inf_fn({POLICY_OBS_KEY: obs}, jax.random.PRNGKey(0))
        return action

    return predict

def export_jax_to_onnx(predict_fn, obs_dim, onnx_path):
    """JAX関数 -> TensorFlow SavedModel -> ONNX の公式ツールチェーンで変換"""
    import tf2onnx
    
    print("[INFO] 1. Converting pure JAX function to TensorFlow...")
    tf_predict = jax2tf.convert(predict_fn, enable_xla=False)
    
    print("[INFO] 2. Wrapping with tf.function (fixing input signature)...")
    @tf.function(
        autograph=False,
        input_signature=[tf.TensorSpec(shape=[None, obs_dim], dtype=tf.float32, name="observation")]
    )
    def tf_func(obs):
        return tf_predict(obs)
    
    print("[INFO] 3. Saving temporary TensorFlow SavedModel...")
    saved_model_dir = "/tmp/brax_saved_model"
    if os.path.exists(saved_model_dir):
        shutil.rmtree(saved_model_dir)
        
    module = tf.Module()
    module.predict = tf_func
    tf.saved_model.save(module, saved_model_dir, signatures={'serving_default': module.predict})
    
    print("[INFO] 4. Converting SavedModel to ONNX via tf2onnx...")
    model_proto, _ = tf2onnx.convert.from_saved_model(
        saved_model_dir, output_path=onnx_path, opset=14
    )
    
    print("[INFO] 5. Cleaning up temporary files...")
    shutil.rmtree(saved_model_dir)
    print(f"\n✅ ONNX Model successfully exported to: {onnx_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="mjx_params.pkl へのパス")
    parser.add_argument("--output", type=str, default="brax_policy.onnx")
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"[Error] Parameter file not found: {args.model}")
        sys.exit(1)
        
    obs_dim = RobotConfig.OBS_DIM
    action_dim = RobotConfig.NUM_JOINTS
    
    print(f"--- ONNX Export Pipeline ---")
    print(f"Observation Dim: {obs_dim}")
    print(f"Action Dim: {action_dim}")
    print(f"Target Output: {args.output}")
    
    try:
        predict_fn = load_brax_inference_fn(args.model)
        export_jax_to_onnx(predict_fn, obs_dim, args.output)
    except Exception as e:
        print(f"\n[Error] Export failed: {e}")
        print("💡 Hint: Ensure you have `tensorflow` and `tf2onnx` installed (`pip install tensorflow-cpu tf2onnx`)")
        sys.exit(1)

if __name__ == "__main__":
    main()
