"""Compatibility shim for brax 0.14.2 on JAX 0.10.0+.

JAX 0.10.0 removed jax.device_put_replicated. Brax 0.14.2 calls it in two
places (pmap.bcast_local_devices and ppo/train.py). Both access it as an
attribute on the jax module, so patching it once here covers both.

Also monkey-patches brax's PPO loss to emit policy-distribution quantiles
(p05/p25/p50/p75/p95 over `loc` and `scale`) in place of the original
min/max/mean scalars, so we can render the policy distribution as a
candle/box plot over training.

Import this module before any brax import.
"""
from typing import Any, Tuple

import jax
import jax.numpy as jnp


def _device_put_replicated(value, devices):
    n = len(devices)
    return jax.tree_util.tree_map(
        lambda v: jnp.broadcast_to(jnp.asarray(v)[None], (n,) + jnp.asarray(v).shape),
        value,
    )


if not hasattr(jax, "device_put_replicated"):
    jax.device_put_replicated = _device_put_replicated


# ----------------------------------------------------------------------
# PPO loss metric replacement
# ----------------------------------------------------------------------

POLICY_DIST_QUANTILES = (0.25, 0.5, 0.75)
_QUANTILE_NAMES = tuple(f"p{int(q * 100):02d}" for q in POLICY_DIST_QUANTILES)


def _patch_ppo_losses() -> None:
    """Replace brax.training.agents.ppo.losses.compute_ppo_loss.

    The replacement is a verbatim copy of brax 0.14.2's compute_ppo_loss
    except for the final metrics dict, which drops the six
    `policy_dist_{mean,max,min}_{loc,std}` scalars and emits ten quantile
    scalars keyed `policy_dist_{loc,std}/p{05,25,50,75,95}` instead.
    """
    from brax.training import types
    from brax.training.agents.ppo import losses as ppo_losses
    from brax.training.agents.ppo import networks as ppo_networks

    quantile_levels = jnp.array(POLICY_DIST_QUANTILES)
    compute_gae = ppo_losses.compute_gae
    quantile_huber_loss = ppo_losses.quantile_huber_loss
    PPONetworkParams = ppo_losses.PPONetworkParams

    def compute_ppo_loss(
        params: PPONetworkParams,
        normalizer_params: Any,
        data: types.Transition,
        rng: jnp.ndarray,
        ppo_network: ppo_networks.PPONetworks,
        entropy_cost: float = 1e-4,
        discounting: float = 0.9,
        reward_scaling: float = 1.0,
        gae_lambda: float = 0.95,
        clipping_epsilon: float = 0.3,
        normalize_advantage: bool = True,
        vf_coefficient: float = 0.5,
        clipping_epsilon_value: float | None = None,
        use_distributional_critic: bool = False,
    ) -> Tuple[jnp.ndarray, types.Metrics]:
        parametric_action_distribution = ppo_network.parametric_action_distribution
        policy_apply = ppo_network.policy_network.apply
        value_apply = ppo_network.value_network.apply

        data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)
        policy_logits = policy_apply(
            normalizer_params, params.policy, data.observation
        )

        if use_distributional_critic:
            baseline, baseline_quantiles = value_apply(
                normalizer_params, params.value, data.observation
            )
            terminal_obs = jax.tree_util.tree_map(
                lambda x: x[-1], data.next_observation
            )
            bootstrap_value, _ = value_apply(
                normalizer_params, params.value, terminal_obs
            )
        else:
            baseline = value_apply(normalizer_params, params.value, data.observation)
            terminal_obs = jax.tree_util.tree_map(
                lambda x: x[-1], data.next_observation
            )
            bootstrap_value = value_apply(
                normalizer_params, params.value, terminal_obs
            )
            baseline_quantiles = None

        rewards = data.reward * reward_scaling
        truncation = data.extras["state_extras"]["truncation"]
        termination = (1 - data.discount) * (1 - truncation)

        target_action_log_probs = parametric_action_distribution.log_prob(
            policy_logits, data.extras["policy_extras"]["raw_action"]
        )
        behaviour_action_log_probs = data.extras["policy_extras"]["log_prob"]

        vs, advantages = compute_gae(
            truncation=truncation,
            termination=termination,
            rewards=rewards,
            values=baseline,
            bootstrap_value=bootstrap_value,
            lambda_=gae_lambda,
            discount=discounting,
        )
        gae_returns = jax.lax.stop_gradient(
            jnp.add(advantages, jax.lax.stop_gradient(baseline))
        )
        if normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        rho_s = jnp.exp(target_action_log_probs - behaviour_action_log_probs)

        surrogate_loss1 = rho_s * advantages
        surrogate_loss2 = (
            jnp.clip(rho_s, 1 - clipping_epsilon, 1 + clipping_epsilon) * advantages
        )

        policy_loss = -jnp.mean(jnp.minimum(surrogate_loss1, surrogate_loss2))

        if use_distributional_critic:
            v_loss = (
                quantile_huber_loss(
                    baseline_quantiles,
                    gae_returns,
                    kappa=clipping_epsilon_value,
                )
                * vf_coefficient
            )
        else:
            v_error = vs - baseline
            v_loss = v_error * v_error
            if clipping_epsilon_value is not None:
                old_values = data.extras["policy_extras"]["value"]
                v_clipped = old_values + jnp.clip(
                    baseline - old_values,
                    -clipping_epsilon_value,
                    clipping_epsilon_value,
                )
                v_loss_clipped = (vs - v_clipped) ** 2
                v_loss = jnp.maximum(v_loss, v_loss_clipped)
            v_loss = jnp.mean(v_loss) * 0.5 * vf_coefficient

        entropy = jnp.mean(
            parametric_action_distribution.entropy(policy_logits, rng)
        )
        entropy_loss = entropy_cost * -entropy

        total_loss = policy_loss + v_loss + entropy_loss

        new_dist = parametric_action_distribution.create_dist(policy_logits)
        if hasattr(new_dist, "kl_divergence"):
            old_dist_params = data.extras["policy_extras"]["distribution_params"]
            old_dist = parametric_action_distribution.create_dist(old_dist_params)
            kl = jnp.mean(new_dist.kl_divergence(old_dist))
        else:
            kl = jnp.array(0.0)

        loc_flat = new_dist.loc.reshape(-1)
        std_flat = new_dist.scale.reshape(-1)
        loc_q = jnp.quantile(loc_flat, quantile_levels)
        std_q = jnp.quantile(std_flat, quantile_levels)

        metrics = {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "v_loss": v_loss,
            "entropy_loss": entropy_loss,
            "kl_mean": kl,
        }
        for name, q in zip(_QUANTILE_NAMES, loc_q):
            metrics[f"policy_dist_loc/{name}"] = q
        for name, q in zip(_QUANTILE_NAMES, std_q):
            metrics[f"policy_dist_std/{name}"] = q
        metrics["policy_dist_loc/mean"] = jnp.mean(loc_flat)
        metrics["policy_dist_loc/min"] = jnp.min(loc_flat)
        metrics["policy_dist_loc/max"] = jnp.max(loc_flat)
        metrics["policy_dist_std/mean"] = jnp.mean(std_flat)
        metrics["policy_dist_std/min"] = jnp.min(std_flat)
        metrics["policy_dist_std/max"] = jnp.max(std_flat)

        return total_loss, metrics

    ppo_losses.compute_ppo_loss = compute_ppo_loss


_patch_ppo_losses()
