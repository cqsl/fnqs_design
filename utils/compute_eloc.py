from functools import partial

import jax
import jax.numpy as jnp
from netket.jax.sharding import (
    shard_along_axis,
)

from utils.afun_chunk import afun_chunk


@partial(jax.jit, static_argnames=("afun", "n_couplings", "n_chunks"))
def compute_eloc(operator, coups_x, n_couplings, afun, variables, n_chunks):
    coups_x = coups_x.reshape(-1, coups_x.shape[-1])

    coups = coups_x[..., :n_couplings]  # (N_mc, n_couplings)
    x = coups_x[..., n_couplings:]  # (N_mc, N)

    xp, mels = operator.get_conn_padded(x)

    coups = coups[:, None, :]
    coups = jnp.tile(coups, (1, xp.shape[-2], 1))
    coups_xp = jnp.concatenate((coups, xp), axis=-1)

    coups_x = shard_along_axis(coups_x, axis=0)

    logpsi_x = afun(variables, coups_x)

    dtype = logpsi_x.dtype

    shape_conn = coups_xp.shape[:-1]
    coups_xp = coups_xp.reshape(-1, coups_xp.shape[-1])
    assert coups_xp.shape[0] % n_chunks == 0, "n_chunks must be changed"
    logpsi_xp = afun_chunk(afun, variables, coups_xp, n_chunks, dtype)
    logpsi_xp = logpsi_xp.reshape(shape_conn)

    logHpsi_x = jax.scipy.special.logsumexp(logpsi_xp, b=mels, axis=-1)  # (Ns,)
    Eloc = jnp.exp(logHpsi_x - logpsi_x)  # (Ns,)

    return Eloc
