"""Foundation Vision Transformer (fViT) wave function conditioned on the couplings.
"""

from typing import Any

import jax
import jax.numpy as jnp
from einops import rearrange
from flax import linen as nn

from models.encoder import Encoder
from models.inner_prods import _inners
from models.output_head import OutputHead

inners = jax.vmap(_inners, in_axes=(0, None, None), out_axes=0)


def extract_patches1d(x, b):
    return rearrange(x, "batch (L_eff b) -> batch L_eff b", b=b)


def extract_patches2d(x, b):
    batch = x.shape[0]
    L_eff = int((x.shape[1] // b**2) ** 0.5)
    x = x.reshape(batch, L_eff, b, L_eff, b)  # [L_eff, b, L_eff, b]
    x = x.transpose(0, 1, 3, 2, 4)  # [L_eff, L_eff, b, b]
    # flatten the patches
    x = x.reshape(batch, L_eff, L_eff, -1)  # [L_eff, L_eff, b*b]
    x = x.reshape(batch, L_eff * L_eff, -1)  # [L_eff*L_eff, b*b]
    return x


def attach_couplings(x, coups):
    coups = jnp.expand_dims(coups, axis=-2)  # (Ns, 1, n_couplings)

    coups = jnp.tile(coups, reps=(1, x.shape[-2], 1))  # (Ns, Np, n_couplings)

    out = jnp.concatenate((x, coups), axis=-1)  # (Ns, Np, b+n_couplings)

    return out


class Embed(nn.Module):
    d_model: int
    b: int
    two_dimensional: bool = False

    def setup(self):
        if self.two_dimensional:
            self.extract_patches = extract_patches2d
        else:
            self.extract_patches = extract_patches1d

        self.embed = nn.Dense(
            self.d_model,
            kernel_init=nn.initializers.xavier_uniform(),
            param_dtype=jnp.float64,
            dtype=jnp.float64,
        )

    def __call__(self, x, coups):
        x = self.extract_patches(x, self.b)  # (Ns, Np, b)

        x = attach_couplings(x, coups)  # (Ns, Np, b+n_couplings)

        x = self.embed(x)

        return x


class fViTSignNoTranslInv(nn.Module):
    num_layers: int
    d_model: int
    heads: int
    L: int
    L_eff: int
    b: int
    n_couplings: int = 1
    transl_invariant: bool = True
    two_dimensional: bool = False
    complex: bool = False
    max_coups: Any = None
    alpha0: float = 1.0
    time_reversal: bool = False
    phases: Any = None
    dtype = jnp.float64

    def setup(self):
        self.patches_and_embed = Embed(self.d_model, self.b, two_dimensional=self.two_dimensional)

        self.encoder = Encoder(
            num_layers=self.num_layers[0],
            d_model=self.d_model,
            h=self.heads,
            L_eff=self.L_eff,
            transl_invariant=True,
            two_dimensional=self.two_dimensional,
            dtype=self.dtype,
            alpha0=self.alpha0,
        )

        self.encoder2 = Encoder(
            num_layers=self.num_layers[1],
            d_model=self.d_model,
            h=self.heads,
            L_eff=self.L_eff,
            transl_invariant=False,
            two_dimensional=self.two_dimensional,
            dtype=self.dtype,
            alpha0=self.alpha0,
        )

        self.output = OutputHead(self.d_model, self.complex)

        if self.max_coups:
            self.max_coups_array = jnp.atleast_2d(self.max_coups.wrapped)
        else:
            self.max_coups_array = 1.0

    def __call__(self, coups_x):
        logsigns = 1j * inners(coups_x, self.phases.wrapped, self.n_couplings)

        coups = coups_x[..., : self.n_couplings]  # (Ns, n_couplings)

        coups = coups / self.max_coups_array

        spins = coups_x[..., self.n_couplings :]  # (Ns, N)

        x = self.patches_and_embed(spins, coups)  # (Ns, L_eff, d_model)

        x = self.encoder(x)

        x = x + self.encoder2(x)

        z = self.output(x)

        z = z + logsigns

        if self.time_reversal:
            z = z + jnp.log(1.0 + jnp.exp(-2j * z.imag))

        return z
