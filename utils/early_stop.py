import jax
import numpy as np


def make_vmc_callback(
    save_callback,
    enabled,
    min_steps,
    patience,
    min_absolute_delta,
):
    best_energy = np.inf
    non_improving_steps = 0

    def callback(vmc_step, log_data, driver):
        nonlocal best_energy, non_improving_steps

        if not save_callback(vmc_step, log_data, driver):
            return False

        if not enabled:
            return True

        energy = float(jax.device_get(driver._loss_stats.mean.real))
        if not np.isfinite(best_energy):
            is_improved = True
        else:
            absolute_improvement = best_energy - energy
            is_improved = absolute_improvement > min_absolute_delta

        if is_improved:
            best_energy = energy
            non_improving_steps = 0
        else:
            non_improving_steps += 1

        if vmc_step + 1 >= min_steps and non_improving_steps >= patience:
            if jax.process_index() == 0:
                print(
                    f"Stopping VMC after {vmc_step + 1} steps: no absolute energy improvement "
                    f"greater than {min_absolute_delta} for "
                    f"{non_improving_steps} steps.",
                    flush=True,
                )
            return False

        return True

    return callback
