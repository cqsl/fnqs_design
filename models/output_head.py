import jax
import jax.numpy as jnp
from flax import linen as nn


def log_cosh(x):
    """
    Logarithm of the hyperbolic cosine, implemented in a more stable way.
    """
    sgn_x = -2 * jnp.signbit(x.real) + 1
    x = x * sgn_x
    return x + jnp.log1p(jnp.exp(-2.0 * x)) - jnp.log(2.0)


class OutputHead(nn.Module):
    d_model: int
    complex: bool = False

    def setup(self):
        self.out_layer_norm = nn.LayerNorm(dtype=jnp.float64, param_dtype=jnp.float64)

        self.norm2 = nn.LayerNorm(use_scale=True, use_bias=True, dtype=jnp.float64, param_dtype=jnp.float64)

        self.output_layer0 = nn.Dense(
            self.d_model,
            param_dtype=jnp.float64,
            dtype=jnp.float64,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=jax.nn.initializers.zeros,
        )

        if self.complex:
            self.norm3 = nn.LayerNorm(use_scale=True, use_bias=True, dtype=jnp.float64, param_dtype=jnp.float64)

            self.output_layer1 = nn.Dense(
                self.d_model,
                param_dtype=jnp.float64,
                dtype=jnp.float64,
                kernel_init=nn.initializers.xavier_uniform(),
                bias_init=jax.nn.initializers.zeros,
            )

    def __call__(self, x):
        x = self.out_layer_norm(x.sum(axis=1))

        amp = self.norm2(self.output_layer0(x))

        z = amp

        if self.complex:
            sign = self.norm3(self.output_layer1(x))
            z = z + 1j * sign

        return jnp.sum(log_cosh(z), axis=-1)
