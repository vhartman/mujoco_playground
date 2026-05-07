"""Compatibility shim for brax 0.14.2 on JAX 0.10.0+.

JAX 0.10.0 removed jax.device_put_replicated. Brax 0.14.2 calls it in two
places (pmap.bcast_local_devices and ppo/train.py). Both access it as an
attribute on the jax module, so patching it once here covers both.

Import this module before any brax import.
"""
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
