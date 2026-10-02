"""[T5] 方策の決定論行動が全経路で数値一致すること (max|Δa| < 1e-5)。

  1. 学習時の Brax make_policy (train/train_mjx.py → ppo.train の内部と同じ構成)
  2. scratch/phase0_eval_diagnostics.py / scratch/gate0_eval.py の推論 (make_inference_fn_from_params)
  3. train/visualize_rl.py の推論 (build_inference_fn)
  4. train/export_onnx.py が書き出す ONNX グラフの式 (同じ重みから NumPy で再現。
     onnx/onnxruntime がある環境では tests/test_env_quality_contract.py が実グラフも検証する)
"""

import os

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

jax = pytest.importorskip("jax")
jp = pytest.importorskip("jax.numpy")
pytest.importorskip("brax")

from brax.training.acme import running_statistics, specs
from brax.training.agents.ppo import networks as ppo_networks

from robot.config import RobotConfig
from robot.policy_network import (
    POLICY_MEAN_CLIP_SCALE,
    POLICY_OBS_KEY,
    default_observation_size,
    make_inference_fn_from_params,
    make_policy_network_factory,
)

NU = RobotConfig.NUM_JOINTS
TOL = 1e-5


@pytest.fixture(scope="module")
def params_and_obs():
    obs_size = default_observation_size()
    net = make_policy_network_factory(obs_size, NU, preprocess_observations_fn=running_statistics.normalize)
    normalizer = running_statistics.init_state(
        {k: specs.Array(v, jp.dtype("float32")) for k, v in obs_size.items()}, std_eps=1e-4)
    batch = {k: jax.random.normal(jax.random.PRNGKey(i), (256,) + v) * 2.0 + 0.5
             for i, (k, v) in enumerate(obs_size.items())}
    normalizer = running_statistics.update(normalizer, batch)
    # 初期化直後の重みは出力がほぼ0で経路差が出にくいため、スケールを少し上げて tanh の非線形域を使う。
    # (上げすぎると中間層が数十〜数百になり、float32 の加算順の違いだけで 1e-5 を超える)
    policy = jax.tree_util.tree_map(lambda w: w * 1.5, net.policy_network.init(jax.random.PRNGKey(0)))
    params = (normalizer, policy, net.value_network.init(jax.random.PRNGKey(1)))
    obs = {k: jax.random.normal(jax.random.PRNGKey(10 + i), (32,) + v) * 2.0 + 0.5
           for i, (k, v) in enumerate(obs_size.items())}
    return params, obs, net


def _batched(policy, obs):
    # GPU では float32 の行列積が既定で TF32 相当になり、厳密な float32 計算(実機の onnxruntime/CPU)と
    # 行動が最大 5e-3 程度ずれる。ここでは式の一致を検査するため最高精度で計算する。
    with jax.default_matmul_precision("highest"):
        return np.asarray(jax.jit(jax.vmap(lambda o: policy(o, jax.random.PRNGKey(0))[0]))(obs))


def test_all_inference_paths_agree(params_and_obs):
    params, obs, net = params_and_obs

    training = ppo_networks.make_inference_fn(net)(params, deterministic=True)
    a_train = _batched(training, obs)

    a_eval = _batched(make_inference_fn_from_params(params, deterministic=True), obs)

    from train.visualize_rl import build_inference_fn
    a_vis = _batched(build_inference_fn(params), obs)

    from train.export_onnx import _policy_arrays
    mean, std, dense = _policy_arrays(params)
    h = (np.asarray(obs[POLICY_OBS_KEY], np.float64) - mean) / std
    for i, (kernel, bias) in enumerate(dense):
        h = h @ kernel + bias
        if i < len(dense) - 1:
            h = h / (1.0 + np.exp(-h))           # swish
    loc_raw = h[:, :NU]
    a_onnx = np.tanh(POLICY_MEAN_CLIP_SCALE * loc_raw / (1.0 + np.abs(loc_raw)))

    assert np.abs(a_train).max() > 0.1, "test inputs should exercise the nonlinear range"
    for name, a in (("eval", a_eval), ("visualize", a_vis), ("onnx_formula", a_onnx)):
        diff = float(np.abs(a - a_train).max())
        assert diff < TOL, f"{name}: max|Δa|={diff:.2e}"


def test_actor_ignores_privileged_state(params_and_obs):
    params, obs, _ = params_and_obs
    policy = make_inference_fn_from_params(params, deterministic=True)
    perturbed = dict(obs, privileged_state=obs["privileged_state"] + 100.0)
    assert np.abs(_batched(policy, obs) - _batched(policy, perturbed)).max() == 0.0
