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
