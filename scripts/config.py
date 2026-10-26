from jsonargparse import ActionYesNo, ArgumentParser


def cfg(config_file: str = "./vmc_config.yaml"):
    parser = ArgumentParser(default_config_files=[config_file])

    parser.add_argument("--seed", type=int)
    parser.add_argument("--output_dir", type=str)

    parser.add_argument("--wandb.log", action=ActionYesNo)
    parser.add_argument("--wandb.log_every", type=int)
    parser.add_argument("--wandb.project", type=str)

    parser.add_argument("--lattice.L", type=int)
    parser.add_argument("--lattice.pbc", action=ActionYesNo)

    parser.add_argument("--hamiltonian.J1", type=float)
    parser.add_argument("--hamiltonian.J2s_initial", nargs="+", type=float)
    parser.add_argument("--hamiltonian.n_systems", type=int)
    parser.add_argument("--hamiltonian.side_length", type=float)

    parser.add_argument("--sampling.n_samples", type=int)
    parser.add_argument("--sampling.n_chains", type=int)
    parser.add_argument("--sampling.n_discard_per_chain", type=int)
    parser.add_argument("--sampling.chunk_size", type=int)
    parser.add_argument("--sampling.n_chunks", type=int)
    parser.add_argument("--sampling.d_max", type=int)

    parser.add_argument("--transformer.num_layers", type=int)
    parser.add_argument("--transformer.d_model", type=int)
    parser.add_argument("--transformer.heads", type=int)
    parser.add_argument("--transformer.b", type=int)
    parser.add_argument("--transformer.tras_inv", action=ActionYesNo)
    parser.add_argument("--transformer.two_dim", action=ActionYesNo)
    parser.add_argument("--transformer.complex", action=ActionYesNo)
    parser.add_argument("--transformer.alpha0", type=float)
    parser.add_argument("--transformer.time_reversal", action=ActionYesNo)

    parser.add_argument("--optimization.lr", type=float)
    parser.add_argument("--optimization.n_iter", type=int)
    parser.add_argument("--optimization.diag_shift", type=float)
    parser.add_argument("--optimization.mode", type=str)
    parser.add_argument("--optimization.save_every", type=int)
    parser.add_argument("--optimization.starting_idx", type=int)
    parser.add_argument("--optimization.early_stopping", action=ActionYesNo)
    parser.add_argument("--optimization.early_stopping_min_steps", type=int)
    parser.add_argument("--optimization.early_stopping_patience", type=int)
    parser.add_argument("--optimization.early_stopping_min_absolute_delta", type=float)

    parser.add_argument("--optimization_couplings.n_steps", type=int)
    parser.add_argument("--optimization_couplings.lr", type=float)
    parser.add_argument("--optimization_couplings.use_cosine_schedule", action=ActionYesNo)
    parser.add_argument("--optimization_couplings.min_lr", type=float)
    parser.add_argument("--optimization_couplings.decay_steps", type=int)
    parser.add_argument("--optimization_couplings.ema_beta", type=float)
    parser.add_argument("--optimization_couplings.n_iter", type=int)
    parser.add_argument("--optimization_couplings.n_reps_grad", type=int)
    parser.add_argument("--optimization_couplings.diag_shift", type=float)
    parser.add_argument("--optimization_couplings.seed_coupling", type=int)
    parser.add_argument("--optimization_couplings.mu", type=float)
    parser.add_argument("--optimization_couplings.beta", type=float)
    parser.add_argument("--optimization_couplings.march_eps", type=float)
    parser.add_argument("--optimization_couplings.n_therm", type=int)

    cfg = parser.parse_args()

    return cfg
