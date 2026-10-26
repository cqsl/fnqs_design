from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnames=("n_points", "min_value"))
def hypercube_random_points(key, center, side_length, n_points, min_value=None):
    center = jnp.asarray(center, dtype=float)
    d = center.shape[0]

    half_side = side_length / 2
    offsets = jax.random.uniform(key, shape=(n_points, d), minval=-half_side, maxval=half_side)

    points = center + offsets
    return points if min_value is None else jnp.maximum(points, min_value)
