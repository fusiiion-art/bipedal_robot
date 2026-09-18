"""Shared PPO policy network configuration for training and deployment."""

import jax
import jax.numpy as jnp
from brax.training import distribution as brax_distribution
from brax.training.agents.ppo import networks as ppo_networks


POLICY_MEAN_CLIP_SCALE = 3.0
POLICY_MIN_STD = 0.15
POLICY_MAX_STD = 3.0

_PATCH_INSTALLED = False


def install_policy_std_cap() -> None:
    """Install the bounded tanh-normal distribution patch once."""
    global _PATCH_INSTALLED
    if _PATCH_INSTALLED:
        return
    if not hasattr(brax_distribution, "_NormalDistribution"):
        raise RuntimeError(
            "Brax distribution API changed: _NormalDistribution is unavailable"
        )

    def clipped_create_dist(self, parameters):
        loc, scale = jnp.split(parameters, 2, axis=-1)
        loc = POLICY_MEAN_CLIP_SCALE * (loc / (1.0 + jnp.abs(loc)))
        scale = (jax.nn.softplus(scale) + self._min_std) * self._var_scale
        scale = jnp.clip(scale, POLICY_MIN_STD, POLICY_MAX_STD)
        return brax_distribution._NormalDistribution(loc=loc, scale=scale)

    brax_distribution.NormalTanhDistribution.create_dist = clipped_create_dist
    _PATCH_INSTALLED = True


def make_policy_network_factory(
    observation_size: int,
    action_size: int,
    preprocess_observations_fn=lambda x, _=None: x,
):
    return ppo_networks.make_ppo_networks(
        observation_size=observation_size,
        action_size=action_size,
        preprocess_observations_fn=preprocess_observations_fn,
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
    )


install_policy_std_cap()
