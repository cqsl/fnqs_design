import os
import sys

os.environ["NETKET_EXPERIMENTAL_SHARDING"] = "1"

# Make the repo root importable (models, utils) regardless of the working directory
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import jax

# Multi-GPU cluster run (e.g. SLURM). For a single GPU or a laptop, remove this line and reduce
# hamiltonian.n_systems and sampling.n_samples/n_chains in vmc_config.yaml.
jax.distributed.initialize()

import flax
import jax.numpy as jnp
import netket as nk
import optax
import wandb
from jsonargparse import namespace_to_dict

from config import cfg
from utils.callback_savestate import SaveState
from utils.compute_phases_8coups_singlenode import compute_phases_8coups_singlenode
from utils.coupling_hilbert_space_arbitrary import CouplingHilbert
from utils.early_stop import make_vmc_callback
from utils.foundation_heisenberg_hamiltonian_8coups import FoundationHeisenbergJ1J2
from utils.get_close_samples import inject_close_samples, thermalize
from utils.ngd_march_couplings_resample import ngd_march_onestep
from utils.run_helpers import build_model, rebuild_vstate, save_array, set_samples
from utils.sampler import FixedRule
from utils.sampling_hypercube import hypercube_random_points
from utils.spin_correlations_supercell import spin_corr
from utils.vmc_sr_spin_multiple import VMC_SR
from utils.wandb_logger import WandbLogger

cfg = cfg(os.path.join(HERE, "vmc_config.yaml"))

# Dimension and the Hilbert space
L = cfg.lattice.L
graph = nk.graph.Square(length=L, pbc=cfg.lattice.pbc)
N = graph.n_nodes

# Hilbert space
hi = nk.hilbert.Spin(s=0.5, N=N, total_sz=0)

# Couplings
n_couplings = 8  # Fixed: the Hamiltonian is written for exactly 8 couplings
coup = jnp.array(cfg.hamiltonian.J2s_initial)
side_length = cfg.hamiltonian.side_length
seed_coupling = cfg.optimization_couplings.seed_coupling
couplings = hypercube_random_points(
    jax.random.PRNGKey(seed_coupling), coup, side_length, cfg.hamiltonian.n_systems, min_value=0.0
)

couplings = jnp.array(couplings, dtype=jnp.float64)  # (n_systems, n_couplings)
n_systems = couplings.shape[0]

# Hilbert spaces
hilbert_coups = CouplingHilbert(couplings)

hit = nk.hilbert.TensorHilbert(hilbert_coups, hi)

# Samplers
n_samples = cfg.sampling.n_samples
n_chains = cfg.sampling.n_chains
assert n_chains == n_samples, "For now, we require n_samples to be equal to n_chains"
assert n_samples % n_systems == 0, "n_samples must be divisible by n_systems"
n_chains_per_system = n_chains // n_systems
n_samples_per_system = n_chains_per_system
n_discard_per_chain = cfg.sampling.n_discard_per_chain
chunk_size = cfg.sampling.chunk_size
n_chunks = cfg.sampling.n_chunks

# Sampler
local_rule = nk.sampler.rules.ExchangeRule(
    d_max=cfg.sampling.d_max,
    graph=graph,
)
fixed_rule = FixedRule()
rule = nk.sampler.rules.TensorRule(hilbert=hit, rules=(fixed_rule, local_rule))
sampler = nk.sampler.MetropolisSampler(
    hit,
    rule=rule,
    n_chains=n_chains,
)

# Define the model
max_coups = nk.utils.HashableArray(jnp.array([jnp.max(jnp.abs(couplings[:, i])) for i in range(n_couplings)]))

# Compute phases
phases = compute_phases_8coups_singlenode(None, L, cfg.hamiltonian.J1, couplings, graph)
model = build_model(cfg, N, n_couplings, max_coups, phases)

# Key
key = jax.random.PRNGKey(cfg.seed)
key, subkey = jax.random.split(key, num=2)
params = model.init(subkey, jnp.zeros((1, hit.size)))

# Define the old state
vstate = nk.vqs.MCState(
    sampler=sampler,
    model=model,
    n_samples=n_samples,
    n_discard_per_chain=n_discard_per_chain,
    sampler_seed=key,
    variables=params,
)

# Inject the samples
samples = hi.random_state(key, (n_samples,))
couplings_samples = jnp.repeat(couplings, repeats=n_samples_per_system, axis=0)
samples = jnp.concatenate((couplings_samples, samples), axis=-1)
set_samples(vstate, samples)

# Number of parameters
if jax.process_index() == 0:
    print(f"Number of parameters: {vstate.n_parameters}", flush=True)

# Hamiltonian
ham = FoundationHeisenbergJ1J2(hi, N, n_systems, J1=cfg.hamiltonian.J1, marshall=False)

# Learning rate
min_value = cfg.optimization.lr / 5
cos_dec = optax.cosine_decay_schedule(init_value=cfg.optimization.lr - min_value, decay_steps=cfg.optimization.n_iter)


def vmc_lr(t):
    return cos_dec(t) + min_value


# Define the optimizer
opt = nk.optimizer.Sgd(learning_rate=vmc_lr)

# Load on wandb
log = WandbLogger(
    project=cfg.wandb.project,
    config=namespace_to_dict(cfg),
    log=cfg.wandb.log,
)

# Output folder: the W&B run folder when logging, otherwise output_dir (relative paths start from the repo root)
if cfg.wandb.log:
    folder_out = wandb.run.dir + "/"
else:
    folder_out = os.path.join(os.path.dirname(HERE), cfg.output_dir) + "/"
    os.makedirs(folder_out, exist_ok=True)

# Callback
cb = SaveState(
    save_every=cfg.optimization.save_every, folder_out=folder_out, starting_idx=cfg.optimization.starting_idx
)

# Create the VMC (Variational Monte Carlo) driver
gs = VMC_SR(
    ham,
    opt,
    diag_shift=cfg.optimization.diag_shift,
    variational_state=vstate,
    mode=cfg.optimization.mode,
    N=N,
    n_systems=n_systems,
    n_couplings=n_couplings,
    wandb_log_every=cfg.wandb.log_every,
    chunk_size_bwd=chunk_size,
    n_chunks=n_chunks,
)

# Run the optimization
gs.run(n_iter=cfg.optimization.n_iter, callback=cb)

# Save the optimized state (serialized on all processes, written by one)
state_bytes = flax.serialization.to_bytes(vstate)
if jax.process_index() == 0:
    with open(
        folder_out + "optimized_state.mpack",
        "wb",
    ) as file:
        file.write(state_bytes)

# Take the observable list
corr_list = spin_corr(hi, L, translations=False)

# Initial coupling
coups = []
coups.append(coup)

# Store the loss
losses = []
errors = []

# Number of steps
n_steps = cfg.optimization_couplings.n_steps
use_cosine_schedule = cfg.optimization_couplings.use_cosine_schedule


def make_lr_schedule(cfg):
    alpha = cfg.optimization_couplings.min_lr / cfg.optimization_couplings.lr
    return optax.cosine_decay_schedule(
        init_value=cfg.optimization_couplings.lr,
        decay_steps=cfg.optimization_couplings.decay_steps,
        alpha=alpha,
    )


coupling_lr = make_lr_schedule(cfg)

# Adaptive solution
ema_beta = cfg.optimization_couplings.ema_beta
ema_loss = None
best_loss = jnp.inf
best_coup = coup

# Number of thermalization steps
n_therm = cfg.optimization_couplings.n_therm

# Number of iterations
n_iter = cfg.optimization_couplings.n_iter

# March parameters
march_state = None

# Number of repetitions for the gradient estimation
n_reps_grad = cfg.optimization_couplings.n_reps_grad

for step in range(1, n_steps + 1):
    # Update seed
    seed_coupling += 1

    # Previous couplings
    couplings_prev = jnp.copy(couplings)

    if use_cosine_schedule:
        lr = coupling_lr(step - 1)
    else:
        lr = cfg.optimization_couplings.lr

    # Take all the different samples
    coups_x_all = vstate.samples
    coups_x_all = coups_x_all.reshape(-1, coups_x_all.shape[-1])

    # Compute phases
    phases = compute_phases_8coups_singlenode(None, L, cfg.hamiltonian.J1, jnp.expand_dims(coup, axis=0), graph)

    # Switch to the model with the updated phases, keeping variables and samples
    model = build_model(cfg, N, n_couplings, max_coups, phases)
    vstate = rebuild_vstate(vstate, model, sampler, n_samples, n_discard_per_chain, key)

    # Duplicate closer samples
    vstate = inject_close_samples(vstate, n_samples_per_system, n_systems, n_couplings, coup, couplings)
    vstate = thermalize(vstate, n_therm=n_therm)

    # One step of March NGD
    coup, loss, error, best_loss, best_coup, ema_loss, march_state = ngd_march_onestep(
        vstate,
        vstate._apply_fun,
        vstate.variables,
        coup,
        n_reps_grad,
        corr_list,
        n_couplings,
        lr,
        best_loss,
        best_coup,
        ema_loss,
        ema_beta,
        n_chunks,
        march_state=march_state,
        mu=cfg.optimization_couplings.mu,
        beta=cfg.optimization_couplings.beta,
        march_eps=cfg.optimization_couplings.march_eps,
        diag_shift=cfg.optimization_couplings.diag_shift,
    )

    # Store the losses and the couplings
    losses.append(loss)
    errors.append(error)
    coups.append(coup)

    # Save the couplings and losses
    seed_tag = cfg.optimization_couplings.seed_coupling
    save_array(folder_out, seed_tag, "coups", coups)
    save_array(folder_out, seed_tag, "losses", losses)
    save_array(folder_out, seed_tag, "errors", errors)
    save_array(folder_out, seed_tag, "phi", march_state[0])
    save_array(folder_out, seed_tag, "velocity", march_state[1])
    save_array(folder_out, seed_tag, "previous_update", march_state[2])
    save_array(folder_out, seed_tag, "t", march_state[3])

    # Generate couplings in a hypercube around the new coupling
    couplings = hypercube_random_points(
        jax.random.PRNGKey(seed_coupling), coup, side_length, cfg.hamiltonian.n_systems, min_value=0.0
    )

    # Compute phases
    phases = compute_phases_8coups_singlenode(None, L, cfg.hamiltonian.J1, couplings, graph)

    # Switch to the model with the updated phases, keeping variables and samples
    model = build_model(cfg, N, n_couplings, max_coups, phases)
    vstate = rebuild_vstate(vstate, model, sampler, n_samples, n_discard_per_chain, key)

    # Inject the samples starting from the previous samples with the new couplings
    samples = coups_x_all.at[..., n_couplings:].get()
    samples_new = jnp.copy(samples)
    for k, coupling in enumerate(couplings):
        dist = jnp.linalg.norm(coupling - couplings_prev, axis=-1) ** 2
        idx = jnp.argmin(dist)
        samples_selected = samples.at[idx * n_samples_per_system : (idx + 1) * n_samples_per_system].get()
        samples_new = samples_new.at[k * n_samples_per_system : (k + 1) * n_samples_per_system].set(samples_selected)
    couplings_samples = jnp.repeat(couplings, repeats=n_samples_per_system, axis=0)
    samples_new = jnp.concatenate((couplings_samples, samples_new), axis=-1)
    set_samples(vstate, samples_new)

    # Thermalize
    vstate = thermalize(vstate, n_therm=n_therm)

    # Save the state (serialized on all processes, written by one)
    state_bytes = flax.serialization.to_bytes(vstate)
    if jax.process_index() == 0:
        with open(
            folder_out + f"optimized_state_step={step}_ngd_march_seed={cfg.optimization_couplings.seed_coupling}.mpack",
            "wb",
        ) as file:
            file.write(state_bytes)

    # Create the VMC (Variational Monte Carlo) driver
    gs = VMC_SR(
        ham,
        opt,
        diag_shift=cfg.optimization.diag_shift,
        variational_state=vstate,
        mode=cfg.optimization.mode,
        N=N,
        n_systems=n_systems,
        n_couplings=n_couplings,
        wandb_log_every=cfg.wandb.log_every,
        chunk_size_bwd=chunk_size,
        n_chunks=n_chunks,
        offset=cfg.optimization.n_iter + (step - 1) * n_iter,  # Continue the W&B steps after the first optimization
    )

    # Rerun the optimization for at most n_iter steps.
    gs.run(
        n_iter=n_iter,
        callback=make_vmc_callback(
            cb,
            cfg.optimization.early_stopping,
            cfg.optimization.early_stopping_min_steps,
            cfg.optimization.early_stopping_patience,
            cfg.optimization.early_stopping_min_absolute_delta,
        ),
    )

if jax.process_index() == 0:
    print(f"Best coupling: {best_coup}, Best loss: {best_loss}", flush=True)
