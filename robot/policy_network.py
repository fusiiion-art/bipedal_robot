"""Shared PPO policy network configuration for training, evaluation and deployment.

- 方策分布: BoundedNormalTanhDistribution を make_policy_network_factory() の戻り値に
  明示的に注入する (brax 本体のクラスは書き換えない)。分布はパラメータを持たない。
- asymmetric actor-critic (docs/master_plan.md §1.3): 観測は dict で、
  actor は 'state' (実機で取得できる観測) のみ、critic は 'privileged_state'
  (シミュレータの真値を含む) を読む。
- checkpoint の読み込みと推論関数の構築は load_checkpoint() /
  make_inference_fn_from_params() に一本化する (評価・可視化・ONNX出力で共通)。
"""

import pickle
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
from brax.training import distribution as brax_distribution
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

from robot.config import RobotConfig


POLICY_MEAN_CLIP_SCALE = 3.0
POLICY_MIN_STD = 0.15
POLICY_MAX_STD = 3.0

POLICY_OBS_KEY = 'state'
VALUE_OBS_KEY = 'privileged_state'

LOG_ROOT = Path(__file__).resolve().parents[1] / "log"


class BoundedNormalTanhDistribution(brax_distribution.NormalTanhDistribution):
    """平均を softsign で ±POLICY_MEAN_CLIP_SCALE に、std を
    [POLICY_MIN_STD, POLICY_MAX_STD] に制限した tanh-normal 分布。

    deterministic 行動 (mode) は tanh(POLICY_MEAN_CLIP_SCALE * softsign(loc_raw))。
    """

    def create_dist(self, parameters):
        loc, scale = jnp.split(parameters, 2, axis=-1)
        loc = POLICY_MEAN_CLIP_SCALE * (loc / (1.0 + jnp.abs(loc)))
        scale = (jax.nn.softplus(scale) + self._min_std) * self._var_scale
        scale = jnp.clip(scale, POLICY_MIN_STD, POLICY_MAX_STD)
        return brax_distribution._NormalDistribution(loc=loc, scale=scale)


def make_policy_network_factory(
    observation_size,
    action_size: int,
    preprocess_observations_fn=lambda x, _=None: x,
):
    """Brax PPO の network_factory。observation_size は dict (env.observation_size) でも
    単一の次元数でもよい (単一配列の観測なら actor/critic とも同じ観測を読む)。"""
    networks = ppo_networks.make_ppo_networks(
        observation_size=observation_size,
        action_size=action_size,
        preprocess_observations_fn=preprocess_observations_fn,
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        policy_obs_key=POLICY_OBS_KEY,
        value_obs_key=VALUE_OBS_KEY,
    )
    return networks.replace(
        parametric_action_distribution=BoundedNormalTanhDistribution(event_size=action_size)
    )


def default_observation_size():
    return {
        POLICY_OBS_KEY: (RobotConfig.OBS_DIM,),
        VALUE_OBS_KEY: (RobotConfig.PRIVILEGED_OBS_DIM,),
    }


class _CompatibilityUnpickler(pickle.Unpickler):
    """numpy 2 で保存された pickle を numpy 1 環境でも読めるようにする。"""

    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)


def load_checkpoint(path):
    with open(path, "rb") as f:
        return _CompatibilityUnpickler(f).load()


def find_checkpoint(exp_name: str = "", version: Optional[int] = None,
                    model_name: str = "best_params.pkl") -> Optional[Path]:
    """log/<exp_name>/version_<version>/<model_name> を返す。version 省略時は最新の version。"""
    root = LOG_ROOT / exp_name if exp_name else LOG_ROOT
    if not root.exists():
        return None
    if version is not None:
        candidate = root / f"version_{version}" / model_name
        return candidate if candidate.exists() else None
    versions = sorted((d for d in root.glob("version_*") if d.is_dir()),
                      key=lambda d: int(d.name.split("_")[-1]))
    for v in reversed(versions):
        if (v / model_name).exists():
            return v / model_name
    return None


def strip_device_dim(params):
    """pmap 由来の先頭の device 次元 (サイズ1) を取り除く。"""
    def _strip(leaf):
        if getattr(leaf, "ndim", 0) > 0 and leaf.shape[0] == 1:
            return leaf.squeeze(0)
        return leaf
    return jax.tree_util.tree_map(_strip, params)


def make_inference_fn_from_params(params, deterministic: bool = True, observation_size=None):
    """学習済み params (normalizer, policy, value) から推論関数 policy(obs, rng) を作る。

    obs は env が返す dict ({'state': ..., 'privileged_state': ...})。actor は 'state'
    しか読まないため、'privileged_state' は形状さえ合っていれば値は使われない。"""
    network = make_policy_network_factory(
        observation_size or default_observation_size(),
        RobotConfig.NUM_JOINTS,
        preprocess_observations_fn=running_statistics.normalize,
    )
    make_policy = ppo_networks.make_inference_fn(network)
    return make_policy(strip_device_dim(params), deterministic=deterministic)
