from copy import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
from netket.utils import struct


# Callback
class SaveState(struct.Pytree, mutable=True):
    save_every: Any
    folder_out: Any
    previous_vstate: Any
    starting_idx: int = 0

    def __init__(self, save_every, folder_out, starting_idx):
        self.save_every = save_every
        self.folder_out = folder_out
        self.starting_idx = starting_idx
        self.previous_vstate = None

    def __call__(self, step, log_data, driver):
        if step % self.save_every == 0 and step > 0:
            # Serialized on all processes, written by one
            state_bytes = flax.serialization.to_bytes(driver.state)
            if jax.process_index() == 0:
                with open(
                    self.folder_out + f"optimized_state_{self.starting_idx + step}.mpack",
                    "wb",
                ) as file:
                    file.write(state_bytes)

        loss_mean = driver._loss_stats.mean
        if jnp.isnan(loss_mean) or jnp.isinf(loss_mean):
            return False

        self.previous_vstate = copy(driver.state)

        return True
