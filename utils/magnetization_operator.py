import jax
import jax.numpy as jnp


@jax.jit
def get_ea(corr):
    order_parameter = jnp.mean(corr**2)

    return order_parameter


@jax.jit
def get_grad_ea(corr, grad_corr):
    n_couplings = grad_corr.shape[-1]
    corr = jnp.expand_dims(corr, axis=-1).repeat(n_couplings, axis=-1)
    grad_order_parameter = jnp.mean(2 * corr * grad_corr, axis=0)

    return grad_order_parameter
