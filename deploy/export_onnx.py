import os
import sys
import pickle
import argparse
import shutil

import jax
import jax.numpy as jp
import tensorflow as tf
from jax.experimental import jax2tf
from brax.training.agents.ppo import networks as ppo_networks

# append project root to sys path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from robot.config import RobotConfig
from train.train_mjx import make_rma_network_factory

def load_brax_inference_fn(pkl_path, obs_dim, action_dim):
    """
    Brax(Flax)の保存済みパラメータファイルから、
    JAXネイティブな推論関数(predict)を復元する
    """
    with open(pkl_path, "rb") as f:
        params = pickle.load(f)
        
    print("[INFO] Params successfully loaded from pickle.")
    
    # train_mjx.py と同一のネットワーク構成を使用（アーキテクチャ不一致を防止）
    ppo_network = make_rma_network_factory(
        observation_size=obs_dim,
        action_size=action_dim,
    )
    
    make_inference_fn = ppo_networks.make_inference_fn(ppo_network)
    inf_fn = make_inference_fn(params)
    
    def predict(obs):
        dummy_rng = jax.random.PRNGKey(0)
        action, _ = inf_fn(obs, dummy_rng)
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
        predict_fn = load_brax_inference_fn(args.model, obs_dim, action_dim)
        export_jax_to_onnx(predict_fn, obs_dim, args.output)
    except Exception as e:
        print(f"\n[Error] Export failed: {e}")
        print("💡 Hint: Ensure you have `tensorflow` and `tf2onnx` installed (`pip install tensorflow-cpu tf2onnx`)")
        sys.exit(1)

if __name__ == "__main__":
    main()
