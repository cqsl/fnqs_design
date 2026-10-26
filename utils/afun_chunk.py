from functools import partial

import jax
import jax.numpy as jnp
from netket.jax.sharding import shard_along_axis


@partial(
    jax.jit,
    static_argnames=(
        "afun",
        "n_chunks",
        "dtype",
    ),
)
def afun_chunk(afun, variables, xp, n_chunks, dtype):
    chunk_size = xp.shape[0] // n_chunks

    logpsi_xp = jnp.zeros(xp.shape[:-1], dtype=dtype)

    def eval_chunk(idx, carry):
        logpsi_xp = carry

        start = idx * chunk_size

        xp_batch = jax.lax.dynamic_slice(
            xp,
            (start, 0),
            (chunk_size, xp.shape[-1]),
        )

        xp_batch = shard_along_axis(xp_batch, axis=0)

        logpsi_xp_batch = afun(variables, xp_batch)

        logpsi_xp = jax.lax.dynamic_update_slice(logpsi_xp, logpsi_xp_batch, (start,))

        return logpsi_xp

    logpsi_xp = jax.lax.fori_loop(0, n_chunks, eval_chunk, logpsi_xp)

    return logpsi_xp
