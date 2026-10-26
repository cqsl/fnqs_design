import jax.numpy as jnp
from netket.jax.sharding import shard_along_axis


def inject_close_samples(vstate, n_samples_per_system, n_systems, n_couplings, target_coupling, couplings):
    samples = vstate.samples
    samples = samples.reshape(-1, samples.shape[-1])

    dist = jnp.linalg.norm(couplings - target_coupling, axis=-1) ** 2

    idx = jnp.argmin(dist)

    samples = samples.at[idx * n_samples_per_system : (idx + 1) * n_samples_per_system].get()

    target_coupling = jnp.repeat(jnp.expand_dims(target_coupling, axis=0), repeats=n_samples_per_system, axis=0)

    samples = samples.at[..., :n_couplings].set(target_coupling)

    samples = jnp.repeat(samples, repeats=n_systems, axis=0)

    samples = shard_along_axis(samples, axis=0)
    samples = jnp.expand_dims(samples, axis=-2)
    vstate._samples = samples
    vstate.sampler_state = vstate.sampler_state.replace(
        σ=samples[:, -1],
        log_prob=2 * vstate._apply_fun(vstate.variables, samples[:, -1]).real,
    )

    return vstate


def thermalize(vstate, n_therm):
    for _ in range(n_therm):
        vstate.sample()
    return vstate
