from functools import partial

import jax
import jax.numpy as jnp


@partial(
    jax.jit,
    static_argnames=("n_couplings",),
)
def _inners(coups_x, phases, n_couplings):
    coups = coups_x[..., :n_couplings]  # (n_couplings)

    spins = coups_x[..., n_couplings:]  # (N)

    coups_phases = phases[..., :n_couplings]  # (n_systems, n_couplings)

    phases = phases[..., n_couplings:]  # (n_systems, n_couplings)

    dist = jnp.linalg.norm(coups[None, :] - coups_phases, axis=-1) ** 2  # (n_systems)

    idx = jnp.argmin(dist)  # (n_systems)

    spins_01 = (spins + 1) / 2  # (N)

    out = jnp.dot(spins_01, phases[idx])  # (n_systems)

    return out
