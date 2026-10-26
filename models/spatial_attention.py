from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from einops import rearrange
from flax import linen as nn


def pbc_distance_grid(L: int, i0: int = 0, j0: int = 0) -> jnp.ndarray:
    I, J = jnp.indices((L, L))
    dx = jnp.abs(I - i0)
    dy = jnp.abs(J - j0)
    dx = jnp.minimum(dx, L - dx)
    dy = jnp.minimum(dy, L - dy)
    return jnp.hypot(dx, dy)


@partial(jax.vmap, in_axes=(None, 0, None), out_axes=1)
@partial(jax.vmap, in_axes=(None, None, 0), out_axes=1)
def roll2d(spins, i, j):
    side = int(spins.shape[-1] ** 0.5)
    spins = spins.reshape(spins.shape[0], side, side)
    spins = jnp.roll(jnp.roll(spins, i, axis=-2), j, axis=-1)
    return spins.reshape(spins.shape[0], -1)


class SpatialAttention(nn.Module):
    d_model: int
    h: int
    L_eff: int
    dtype: Any
    transl_invariant: bool = True
    two_dimensional: bool = False
    alpha0: float = 1.0

    def setup(self):
        self.v = nn.Dense(self.d_model, kernel_init=nn.initializers.xavier_uniform(), param_dtype=self.dtype)
        assert self.two_dimensional

        sq_L_eff = int(self.L_eff**0.5)
        distances = pbc_distance_grid(sq_L_eff, 0, 0).flatten()

        if self.transl_invariant:
            self.J = self.param("J", nn.initializers.xavier_uniform(), (self.h, self.L_eff), self.dtype)
            self.alpha = self.param("alpha", nn.initializers.constant(self.alpha0), (self.h, 1), self.dtype)
            self.J *= jax.nn.softmax(-self.alpha * distances)
            self.J = roll2d(self.J, jnp.arange(sq_L_eff), jnp.arange(sq_L_eff))
            self.J = self.J.reshape(self.h, -1, self.L_eff)
        else:
            self.J = self.param("J", nn.initializers.xavier_uniform(), (self.h, self.L_eff, self.L_eff), self.dtype)
            self.alpha = self.param("alpha", nn.initializers.constant(self.alpha0), (self.h, 1, 1), self.dtype)
            distances = roll2d(distances[None], jnp.arange(sq_L_eff), jnp.arange(sq_L_eff))
            distances = distances.reshape(1, -1, self.L_eff)
            self.J *= jax.nn.softmax(-self.alpha * distances)

        self.W = nn.Dense(self.d_model, kernel_init=nn.initializers.xavier_uniform(), param_dtype=self.dtype)

    def __call__(self, x):
        v = self.v(x)
        v = rearrange(v, "batch L_eff (h d_eff) -> batch L_eff h d_eff", h=self.h)
        v = rearrange(v, "batch L_eff h d_eff -> batch h L_eff d_eff")
        x = jnp.matmul(self.J, v).astype(v.dtype)
        x = rearrange(x, "batch h L_eff d_eff  -> batch L_eff h d_eff")
        x = rearrange(x, "batch L_eff h d_eff ->  batch L_eff (h d_eff)")

        x = self.W(x)

        return x
