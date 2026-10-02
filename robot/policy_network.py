"""Shared PPO policy network configuration for training and deployment.

[2026-10-02] 旧実装は import 時に brax の NormalTanhDistribution.create_dist を
グローバルに書き換えていた(モンキーパッチ)。そのため「このモジュールを import
したプロセスでだけ」平均クリップが効き、import しない経路(別スクリプト・
将来の推論コード)では deterministic 行動が tanh(3*softsign(loc)) ではなく
tanh(loc) になる、という import 順依存の不整合があり得た。
現在はサブクラス BoundedNormalTanhDistribution を make_policy_network_factory()
の戻り値に明示的に注入する。brax 側のクラスは一切変更しない。
分布はパラメータを持たないため、既存checkpointとの互換性は保たれる。
"""

import jax
import jax.numpy as jnp
from brax.training import distribution as brax_distribution
from brax.training.agents.ppo import networks as ppo_networks


POLICY_MEAN_CLIP_SCALE = 3.0
POLICY_MIN_STD = 0.15
POLICY_MAX_STD = 3.0


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
    observation_size: int,
    action_size: int,
    preprocess_observations_fn=lambda x, _=None: x,
):
    if not hasattr(brax_distribution, "_NormalDistribution"):
        raise RuntimeError(
            "Brax distribution API changed: _NormalDistribution is unavailable"
        )
    networks = ppo_networks.make_ppo_networks(
        observation_size=observation_size,
        action_size=action_size,
        preprocess_observations_fn=preprocess_observations_fn,
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
    )
    return networks.replace(
        parametric_action_distribution=BoundedNormalTanhDistribution(
            event_size=action_size
        )
    )
