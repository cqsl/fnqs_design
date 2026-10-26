from typing import Any

from flax import linen as nn

from models.spatial_attention import SpatialAttention


class EncoderBlock(nn.Module):
    d_model: int
    h: int
    L_eff: int
    dtype: Any
    transl_invariant: bool = True
    two_dimensional: bool = False
    alpha0: float = 1.0

    def setup(self):
        self.attn = SpatialAttention(
            **{
                "d_model": self.d_model,
                "h": self.h,
                "L_eff": self.L_eff,
                "transl_invariant": self.transl_invariant,
                "two_dimensional": self.two_dimensional,
                "dtype": self.dtype,
                "alpha0": self.alpha0,
            }
        )

        self.layer_norm_1 = nn.LayerNorm(param_dtype=self.dtype)
        self.layer_norm_2 = nn.LayerNorm(param_dtype=self.dtype)

        self.ff = nn.Sequential(
            [
                nn.Dense(4 * self.d_model, kernel_init=nn.initializers.xavier_uniform(), param_dtype=self.dtype),
                nn.gelu,
                nn.Dense(self.d_model, kernel_init=nn.initializers.xavier_uniform(), param_dtype=self.dtype),
            ]
        )

    def __call__(self, x):
        x = x + self.attn(self.layer_norm_1(x))

        x = x + self.ff(self.layer_norm_2(x))
        return x


class Encoder(nn.Module):
    num_layers: int
    d_model: int
    h: int
    L_eff: int
    dtype: Any
    transl_invariant: bool = True
    two_dimensional: bool = False
    alpha0: float = 1.0

    def setup(self):
        self.layers = [
            EncoderBlock(
                d_model=self.d_model,
                h=self.h,
                L_eff=self.L_eff,
                transl_invariant=self.transl_invariant,
                two_dimensional=self.two_dimensional,
                dtype=self.dtype,
                alpha0=self.alpha0,
            )
            for _ in range(self.num_layers)
        ]

    def __call__(self, x):
        for l in self.layers:
            x = l(x)

        return x
