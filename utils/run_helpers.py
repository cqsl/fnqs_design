import jax.numpy as jnp
import netket as nk
from netket.jax.sharding import shard_along_axis

from models.fvit_sign_notranslinv import fViTSignNoTranslInv
from utils.save_sharded import save_global_array_once


def build_model(cfg, N, n_couplings, max_coups, phases):
    """Build the transformer described by `cfg.transformer` with the sign structure given by `phases`."""
    b = cfg.transformer.b
    return fViTSignNoTranslInv(
        num_layers=(cfg.transformer.num_layers - 2, 2),
        d_model=cfg.transformer.d_model,
        heads=cfg.transformer.heads,
        L=cfg.lattice.L,
        L_eff=N // b**2,
        b=b,
        n_couplings=n_couplings,
        transl_invariant=cfg.transformer.tras_inv,
        two_dimensional=cfg.transformer.two_dim,
        complex=cfg.transformer.complex,
        max_coups=max_coups,
        alpha0=cfg.transformer.alpha0,
        time_reversal=cfg.transformer.time_reversal,
        phases=phases,
    )


def set_samples(vstate, samples):
    """Overwrite the samples and the sampler state of `vstate` with `samples` of shape (n_samples, n_couplings + N)."""
    samples = shard_along_axis(samples, axis=0)
    samples = jnp.expand_dims(samples, axis=-2)
    vstate._samples = samples
    vstate.sampler_state = vstate.sampler_state.replace(
        σ=samples[:, -1],
        log_prob=2 * vstate._apply_fun(vstate.variables, samples[:, -1]).real,
    )


def rebuild_vstate(vstate, model, sampler, n_samples, n_discard_per_chain, sampler_seed):
    """Return a new MCState with `model`, keeping the variables and samples of `vstate`."""
    samples = vstate.samples
    samples = samples.reshape(-1, samples.shape[-1])
    new_vstate = nk.vqs.MCState(
        sampler=sampler,
        model=model,
        n_samples=n_samples,
        n_discard_per_chain=n_discard_per_chain,
        sampler_seed=sampler_seed,
        variables=vstate.variables,
    )
    set_samples(new_vstate, samples)
    return new_vstate


def save_array(folder_out, seed_coupling, name, array):
    """Save `array` in `folder_out`, tagged with the coupling seed."""
    save_global_array_once(folder_out + f"{name}_ngd_march_seed={seed_coupling}.npy", jnp.array(array))
