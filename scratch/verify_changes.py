"""Verify the reward weight changes and import consistency."""
import sys
sys.path.insert(0, '.')

# 1. Config verification
from robot.config import RobotConfig
w = RobotConfig.REWARD_WEIGHTS
print("=== Config Verification ===")
print(f"  alive weight:        {w['alive']}  (expected: 2.0)")
print(f"  fall_penalty weight: {w['fall_penalty']}  (expected: -20.0)")
assert w['alive'] == 2.0, f"alive should be 2.0, got {w['alive']}"
assert w['fall_penalty'] == -20.0, f"fall_penalty should be -20.0, got {w['fall_penalty']}"
print("  [OK] Config OK")

# 2. Reward system import
print("\n=== Reward System Import ===")
from envs.mjx_rewards import MJXRewardSystem
import inspect
src = inspect.getsource(MJXRewardSystem.compute)
assert 'reward_per_step' in src, "reward_per_step metric missing"
assert 'total_penalty' in src, "total_penalty metric missing"
print("  [OK] reward_per_step metric present")
print("  [OK] total_penalty metric present")

# 3. Env import
print("\n=== Environment Import ===")
from envs.mjx_env import SenpuuMaruMJXEnv
src_env = inspect.getsource(SenpuuMaruMJXEnv.reset)
assert 'reward_per_step' in src_env, "reward_per_step init missing in reset()"
assert 'total_penalty' in src_env, "total_penalty init missing in reset()"
print("  [OK] reset() metrics init OK")

# 4. Training script import check
print("\n=== Training Script Check ===")
with open('train/train_mjx.py', 'r') as f:
    train_src = f.read()
assert 'clipping_epsilon=0.2' in train_src, "clipping_epsilon not set"
assert "learning_rate_schedule='ADAPTIVE_KL'" in train_src, "ADAPTIVE_KL not set"
assert 'desired_kl=0.02' in train_src, "desired_kl not set"
assert 'max_grad_norm=1.0' in train_src, "max_grad_norm not set"
assert 'import optax' not in train_src, "unused optax import still present"
print("  [OK] clipping_epsilon=0.2")
print("  [OK] learning_rate_schedule='ADAPTIVE_KL'")
print("  [OK] desired_kl=0.02")
print("  [OK] max_grad_norm=1.0")
print("  [OK] unused optax import removed")

print("\n=== Lambda Phase Check ===")
with open('envs/mjx_rewards.py', 'r') as f:
    reward_src = f.read()
assert 'lambda_phase = jp.clip(' in reward_src, "lambda_phase is not explicitly clipped"
print("  [OK] lambda_phase is clipped to [0, 1]")

# 5. Default learning rate
assert 'default=1e-4' in train_src, "default learning rate not 1e-4"
print("  [OK] default learning_rate=1e-4")

print("\n=== ALL CHECKS PASSED ===")

