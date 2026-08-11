"""Compatibility shim for brax on JAX 0.10+.

JAX 0.10 removed jax.device_put_replicated. brax 0.14.2 still calls it, in
pmap.bcast_local_devices and in ppo/train.py, so training dies immediately
without this. Both reach it as an attribute on the jax module, so patching it
once here covers them.

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
