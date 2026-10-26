import os
import time

import jax
import jax.numpy as jnp
import netket as nk
import numpy as np

from utils.compute_phases_8coups import compute_phases_8coups

compute_phases_8coups_vmap = jax.vmap(compute_phases_8coups, in_axes=(None, None, 0, None), out_axes=0)


def compute_phases_8coups_singlenode(folder_in, L, J1, coups, graph):
    folder_in = folder_in or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".phase_tmp")
    broadcast_id = os.environ.get("SLURM_JOB_ID", "local")
    phases_path = os.path.join(folder_in, f".phases_broadcast_{broadcast_id}.npy")
    temporary_path = f"{phases_path}.{jax.process_index()}.tmp.npy"
    expected_shape = (coups.shape[0], 8 + graph.n_nodes)

    if jax.process_index() == 0:
        os.makedirs(folder_in, exist_ok=True)
        # Remove a file left behind by an interrupted attempt before workers look.
        try:
            os.remove(phases_path)
        except FileNotFoundError:
            pass

    jax.experimental.multihost_utils.sync_global_devices("phases_file_cleared")

    if jax.process_index() == 0:
        phases_computed = compute_phases_8coups_vmap(L, J1, coups, graph.positions)
        phases_np = np.asarray(phases_computed)
        if phases_np.shape != expected_shape:
            raise RuntimeError(f"Computed phase table has shape {phases_np.shape}; expected {expected_shape}.")
        # Publish atomically: workers only ever observe a complete file.
        np.save(temporary_path, phases_np)
        os.replace(temporary_path, phases_path)
    else:
        max_wait = 300
        start_time = time.time()
        while not os.path.exists(phases_path) and (time.time() - start_time) < max_wait:
            time.sleep(0.1)
        if not os.path.exists(phases_path):
            raise RuntimeError(f"Master rank failed to write phases to {phases_path}")
        phases_np = np.load(phases_path)

    if phases_np.shape != expected_shape:
        raise RuntimeError(
            f"Loaded phase table has shape {phases_np.shape}; expected {expected_shape}. "
            "Remove the stale broadcast file and retry."
        )

    phases = nk.utils.HashableArray(jnp.array(phases_np))

    # Synchronize all ranks after loading
    jax.experimental.multihost_utils.sync_global_devices("phases_loaded")

    # Clean up temporary file (only on master rank)
    if jax.process_index() == 0:
        try:
            os.remove(phases_path)
        except FileNotFoundError:
            pass

    return phases
