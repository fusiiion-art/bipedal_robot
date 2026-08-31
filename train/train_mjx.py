import os
import sys
import argparse
import time
from datetime import datetime

# sysモジュールのパッチ (Windows上のbrax/orbax依存対策)
if not hasattr(sys.modules.get("uvloop", None), "__name__"):
    sys.modules["uvloop"] = type(sys)("uvloop")

import jax
import jax.numpy as jnp

# Reuse XLA executables across repeated WSL validation/training runs.
jax.config.update("jax_compilation_cache_dir", "/mnt/c/bipedal_robot/.jax_cache")
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

if not hasattr(jax, "device_put_replicated"):
    def _device_put_replicated(x, devices):
        return jax.tree_util.tree_map(lambda leaf: jax.device_put(jnp.expand_dims(leaf, 0)), x)
    jax.device_put_replicated = _device_put_replicated

from brax import envs
from brax.envs import training as brax_training
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.training import distribution as brax_distribution

POLICY_MEAN_CLIP_SCALE = 3.0
POLICY_MIN_STD = 0.05
POLICY_MAX_STD = 3.0


def _install_policy_std_cap():
    """Cap tanh-normal scale while preserving Brax's existing distribution API."""
    original_create_dist = brax_distribution.NormalTanhDistribution.create_dist

    def clipped_create_dist(self, parameters):
        loc, scale = jnp.split(parameters, 2, axis=-1)
        loc = POLICY_MEAN_CLIP_SCALE * (loc / (1.0 + jnp.abs(loc)))
        scale = (jax.nn.softplus(scale) + self._min_std) * self._var_scale
        scale = jnp.clip(scale, POLICY_MIN_STD, POLICY_MAX_STD)
        return brax_distribution._NormalDistribution(loc=loc, scale=scale)

    if not hasattr(brax_distribution, '_NormalDistribution'):
        raise RuntimeError('Brax distribution API changed: _NormalDistribution is unavailable')
    brax_distribution.NormalTanhDistribution.create_dist = clipped_create_dist
    return original_create_dist


_install_policy_std_cap()

# Patch brax _unpmap for JAX 0.4+ Multi-GPU safety
def _safe_unpmap(v):
    def _unpmap_leaf(x):
        if hasattr(x, "addressable_shards"):
            d = x.addressable_shards[0].data
            if d.ndim > 0 and d.shape[0] == 1:
                return d.squeeze(0)
            return d
        if hasattr(x, "device_buffers"):
            return x[0]
        return x
    return jax.tree_util.tree_map(_unpmap_leaf, v)

ppo._unpmap = _safe_unpmap

# Safe wrapper for make_inference_fn to handle leading pmap dimension in params
_orig_make_inference_fn = ppo_networks.make_inference_fn
def _safe_make_inference_fn(ppo_networks_tuple, **make_kwargs):
    orig_fn = _orig_make_inference_fn(ppo_networks_tuple, **make_kwargs)
    def safe_inference_fn(params, *args, **kwargs):
        def _strip_leading_dim(leaf):
            if hasattr(leaf, "shape") and getattr(leaf, "ndim", 0) > 1 and leaf.shape[0] == 1:
                return leaf.squeeze(0)
            return leaf
        params_stripped = jax.tree_util.tree_map(_strip_leading_dim, params)
        return orig_fn(params_stripped, *args, **kwargs)
    return safe_inference_fn

ppo_networks.make_inference_fn = _safe_make_inference_fn






sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from robot.config import RobotConfig
from train.networks import AdaptationModule, BasePolicy, TeacherPolicy
from envs.mjx_env import SenpuuMaruMJXEnv  # noqa: F401 (Brax環境登録のため)
from envs.training_wrapper import TrainingProgressWrapper

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="", help="Experiment subfolder under log. Leave empty to write directly to log/version_x.")
    parser.add_argument("--num_envs", type=int, default=None, help="並列環境数 (GPUなら2048〜4096推奨, CPU自動設定)")
    parser.add_argument("--steps", type=int, default=None, help="総学習ステップ数")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--unroll_length", type=int, default=10, help="PPOのアクションアンロール長")
    parser.add_argument("--episode_length", type=int, default=None, help="1エピソードのステップ数")
    parser.add_argument("--num_evals", type=int, default=None, help="評価回数")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_minibatches", type=int, default=None)
    parser.add_argument("--num_updates_per_batch", type=int, default=4)
    return parser.parse_args()

def make_rma_network_factory(
    observation_size: int,
    action_size: int,
    preprocess_observations_fn=lambda x, _=None: x,
):
    """
    RMA Architecture Network Factory:
    
    観測空間の構成 (OBS_DIM):
      - Base Obs (現在の状態)
      - Obs/Action History (遅延補償用の履歴バッファ)
      - Servo Temperature (各関節の温度)
      - Supply Voltage (電源電圧)
      
    学習フェーズ1: Teacher Policy (特権情報あり) で学習
    学習フェーズ2: Adaptation Module で特権情報を履歴から推定する蒸留
    デプロイ: Base Policy + Adaptation Module (Teacher不要)
    
    現在はBrax PPOの標準MLPを拡張OBS_DIMに合わせて使用。
    """
    return ppo_networks.make_ppo_networks(
        observation_size=observation_size,
        action_size=action_size,
        preprocess_observations_fn=preprocess_observations_fn,
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        mean_clip_scale=POLICY_MEAN_CLIP_SCALE,
    )

def main():
    args = parse_args()
    
    print("=== MJX GPU Training Pipeline (RMA Enabled) ===")
    devices = jax.devices()
    print(f"JAX Devices: {devices}")
    is_cpu = devices[0].platform == 'cpu'
    
    if is_cpu:
        print("[Warning] JAX is running on CPU. GPU未使用 - パラメータをCPU向けに自動縮小します。")
        print("[Info] GPU使用にはWSL2 + JAX CUDA版が必要です。")
        # CPUモード: コンパイル時間を最小化する小さなパラメータ
        num_envs       = args.num_envs       or 32
        steps          = args.steps          or 100_000
        episode_length = args.episode_length or 50
        num_evals      = args.num_evals      or 3
        batch_size     = args.batch_size     or 32
        num_minibatches = args.num_minibatches or 1
    else:
        print(f"[GPU] {devices[0]} で学習開始！")
        # GPUモード: RTX 4060 (8GB) でのコンパイルハングを避けるため、パラメータを軽量化
        num_envs       = args.num_envs       or 256
        steps          = args.steps          or 10_000_000
        episode_length = args.episode_length or RobotConfig.MAX_EPISODE_STEPS
        num_evals      = args.num_evals      or 20
        batch_size     = args.batch_size     or 256
        num_minibatches = args.num_minibatches or 16

    # --- 複数GPU環境向けの自動最適化 (Divisibilityの担保) ---
    num_devices = len(devices)
    if num_envs % num_devices != 0:
        old_num_envs = num_envs
        num_envs = (num_envs // num_devices) * num_devices
        print(f"[Auto-Tune] num_envs をGPU数({num_devices})で割り切れる {num_envs} に自動調整しました (元: {old_num_envs})")
    
    if batch_size % num_devices != 0:
        old_batch_size = batch_size
        batch_size = max(1, batch_size // num_devices) * num_devices
        print(f"[Auto-Tune] batch_size をGPU数({num_devices})で割り切れる {batch_size} に自動調整しました (元: {old_batch_size})")

    # batch_size * num_minibatches は num_envs で割り切れる必要がある
    if (batch_size * num_minibatches) % num_envs != 0:
        # 割り切れるように num_minibatches を自動調整
        import math
        old_minibatches = num_minibatches
        # 必要な最小の倍数を探す
        target_total_batch = math.ceil((batch_size * num_minibatches) / num_envs) * num_envs
        num_minibatches = target_total_batch // batch_size
        print(f"[Auto-Tune] (batch_size * num_minibatches) % num_envs == 0 を満たすため、num_minibatches を {num_minibatches} に自動調整しました (元: {old_minibatches})")

    # 1. 環境生成
    env = envs.get_environment('senpuu_maru_mjx')
    
    # 2. ログディレクトリとバージョン管理
    from pathlib import Path
    import json
    
    root_path = Path(__file__).resolve().parent.parent
    base_log_dir = root_path / "log"
    if args.exp_name:
        base_log_dir = base_log_dir / args.exp_name
    base_log_dir.mkdir(parents=True, exist_ok=True)
    
    version = 0
    while (base_log_dir / f"version_{version}").exists():
        version += 1
    run_dir = base_log_dir / f"version_{version}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Info] Logging to {run_dir}", flush=True)
    
    # 状態トラッキング用変数
    best_reward = -float('inf')
    worst_reward = float('inf')
    current_params = None
    metrics_history = []
    
    def policy_params_callback(current_step, make_policy, params):
        import pickle
        nonlocal current_params
        current_params = params
        
        # 毎回ラストのモデルを保存
        with open(run_dir / "last_params.pkl", "wb") as f:
            pickle.dump(params, f)
            
    # コールバック関数（プログレス表示・ログ保存用）
    def progress_callback(num_steps, metrics):
        import pickle
        nonlocal best_reward, worst_reward
        reward = metrics.get('eval/episode_reward', metrics.get('training/total_reward', float('nan')))

        # 学習進捗率の計算と表示。ここでは総乱数ステップではなく、
        # 各環境の累積ステップを単調に増やす構造を優先し、
        # 1.0 を超えないようにする。
        training_progress = min(num_steps / max(steps, 1), 1.0) if steps > 0 else 0.0
        print(f"Step: {num_steps:10d} | Reward: {reward:.4f} | Progress: {training_progress:.2%}", flush=True)

        # JSONログ用の辞書作成
        metrics_dict = {
            "step": int(num_steps),
            "reward": float(reward),
            "training_progress": float(training_progress),
            "num_envs": int(num_envs),
            "episode_length": int(episode_length),
        }
        for k, v in metrics.items():
            if k == "training_progress":
                continue
            metrics_dict[k] = float(v.item() if hasattr(v, 'item') else v)
        metrics_history.append(metrics_dict)
        
        with open(run_dir / "log.json", "w") as f:
            json.dump(metrics_history, f, indent=2)
            
        # 最高のモデルと最低のモデルを保存
        if current_params is not None:
            if reward > best_reward:
                best_reward = reward
                with open(run_dir / "best_params.pkl", "wb") as f:
                    pickle.dump(current_params, f)
                print(f"  >>> Best Model Saved! (Reward: {reward:.4f})", flush=True)
                
            if reward < worst_reward:
                worst_reward = reward
                with open(run_dir / "worst_params.pkl", "wb") as f:
                    pickle.dump(current_params, f)
                print(f"  >>> Worst Model Saved! (Reward: {reward:.4f})", flush=True)

    print(f"Starting training: num_envs={num_envs}, steps={steps}, episode_length={episode_length}")
    start_time = time.time()
    
    # Learning Rate: Brax内蔵のAdaptive KL LRスケジュールを使用。
    # KL爆発時に自動的に学習率を下げ、KLが低すぎる場合は上げる。
    # 以前のoptax.warmup_cosine_decay_scheduleはBrax PPOの内部optimizerには
    # 渡されておらず機能していなかったため削除。
    
    # --- TrainingProgressWrapper の注入 ---
    # brax.envs.training.wrap を一時的に差し替え、AutoResetWrapper の
    # 外側に TrainingProgressWrapper を配置する。
    # これにより training_progress がエピソード境界を跨いで単調増加する。
    steps_per_env = max(steps // num_envs, 1)
    _original_wrap = brax_training.wrap

    def _wrap_with_progress(env, **kwargs):
        wrapped = _original_wrap(env, **kwargs)
        return TrainingProgressWrapper(wrapped, total_steps_per_env=steps_per_env)

    brax_training.wrap = _wrap_with_progress

    # 3. PPO学習実行 (RMA Network Architecture)
    try:
        make_inference_fn, params, metrics = ppo.train(
            environment=env,
            network_factory=make_rma_network_factory,
            num_timesteps=steps,
            num_evals=num_evals,
            reward_scaling=0.01,  # 報酬クリップ後の値をPPOの更新量に合わせる
            episode_length=episode_length,
            normalize_observations=True,
            action_repeat=1,
            unroll_length=args.unroll_length,
            num_minibatches=num_minibatches,
            num_updates_per_batch=args.num_updates_per_batch,
            discounting=0.99,
            bootstrap_on_timeout=True,
            learning_rate=args.learning_rate,
            entropy_cost=1e-3,
            # --- KLダイバージェンス制御 ---
            clipping_epsilon=0.2,           # 0.3(Braxデフォルト)→0.2に縮小
            max_grad_norm=1.0,              # 勾配クリッピングで勾配爆発を防止
            learning_rate_schedule='ADAPTIVE_KL',  # Brax内蔵Adaptive KL LR
            desired_kl=0.02,                # ターゲットKL (20DOF高次元行動空間のためやや緩め)
            learning_rate_schedule_min_lr=1e-5,   # KL爆発時のフロア（1e-6では低すぎてLRがstuckする）
            learning_rate_schedule_max_lr=5e-4,   # KL安定時の天井

            num_envs=num_envs,
            batch_size=batch_size,
            seed=42,
            progress_fn=progress_callback,
            policy_params_fn=policy_params_callback
        )
    finally:
        # 他のモジュールに影響しないよう必ず復元
        brax_training.wrap = _original_wrap

    elapsed_time = time.time() - start_time
    print(f"Training finished in {elapsed_time/60:.1f} minutes!")
    
    # 4. パラメータ保存
    import pickle
    model_path = os.path.join(run_dir, "final_params.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(params, f)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
