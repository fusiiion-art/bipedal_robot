import os
import sys
import pickle
import argparse

import jax
import jax.numpy as jnp
import numpy as np

# sysモジュールのパッチ (Windows/WSL上のbrax/orbax依存対策)
if not hasattr(sys.modules.get("uvloop", None), "__name__"):
    sys.modules["uvloop"] = type(sys)("uvloop")

from brax import envs
from brax.training.agents.ppo import networks as ppo_networks

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from robot.config import RobotConfig
from envs.mjx_env import SenpuuMaruMJXEnv  # noqa: F401
from train.train_mjx import make_rma_network_factory

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, default=8)
    parser.add_argument("--model", type=str, default="best_params.pkl")
    args = parser.parse_args()

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(root_dir, "log", f"version_{args.version}", args.model)
    if not os.path.exists(model_path):
        model_path = os.path.join(root_dir, "log", "mjx_ppo_rma_100hz", f"version_{args.version}", args.model)
    
    print(f"モデルをロード中: {model_path}")
    with open(model_path, "rb") as f:
        params = pickle.load(f)
        
    jax.config.update('jax_platform_name', 'cpu')
    env = envs.get_environment('senpuu_maru_mjx')
    
    ppo_network = make_rma_network_factory(observation_size=env.observation_size, action_size=env.action_size)
    make_policy = ppo_networks.make_inference_fn(ppo_network)
    
    normalizer_params, policy_params, value_params = params
    policy = make_policy((normalizer_params, policy_params, value_params), deterministic=True)
    
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    
    rng = jax.random.PRNGKey(0)
    rng, key_reset = jax.random.split(rng)
    state = jit_reset(key_reset)
    
    qpos_list = []
    
    print("AIが生成する軌跡を計算中 (WSLで実行中)...")
    for _ in range(800):  # 8秒間分シミュレーション
        rng, key_act = jax.random.split(rng)
        act_params = policy(state.obs, key_act)
        state = jit_step(state, act_params[0])
        qpos_list.append(np.array(state.pipeline_state.qpos))
        
        if state.done:
            break
            
    traj = np.array(qpos_list)
    out_path = os.path.join(root_dir, "trajectory.npy")
    np.save(out_path, traj)
    print(f"軌跡データを保存しました: {out_path}")

if __name__ == "__main__":
    main()
