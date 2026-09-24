import jax

from envs.mjx_env import SenpuuMaruMJXEnv


def test_reset_training_progress_is_full_for_direct_eval():
    env = SenpuuMaruMJXEnv()
    state = env.reset(jax.random.PRNGKey(0))
    assert float(state.info["training_progress"]) == 1.0


def test_direct_eval_policies_are_not_zeroed_by_progress_scale():
    env = SenpuuMaruMJXEnv()
    state = env.reset(jax.random.PRNGKey(1))
    # Direct evaluation should not carry the training-only zero progress that suppresses all soft penalties.
    assert float(state.info["training_progress"]) >= 0.99


def test_disturbed_eval_cli_args():
    from train.train_mjx import parse_args

    # デフォルト値の確認
    defaults = parse_args([])
    assert defaults.eval_curriculum_progress == 1.0
    assert defaults.disturbed_num_eval_envs == 64
    assert defaults.disable_disturbed_eval is False

    # カスタム指定の確認
    custom = parse_args([
        "--eval_curriculum_progress", "0.3",
        "--disturbed_num_eval_envs", "32",
        "--disable_disturbed_eval",
    ])
    assert custom.eval_curriculum_progress == 0.3
    assert custom.disturbed_num_eval_envs == 32
    assert custom.disable_disturbed_eval is True

