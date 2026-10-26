import os

import jax
import numpy as np

# Saving from many sharded devices
from jax.experimental import multihost_utils


def save_global_array_once(path, arr):
    # Collect global distributed jax.Array into a regular numpy array.
    arr_np = multihost_utils.process_allgather(arr, tiled=True)

    # Only one process writes the final file.
    if jax.process_index() == 0:
        tmp = path + ".tmp"
        np.save(tmp, np.asarray(arr_np))
        os.replace(tmp + ".npy", path)

    # Make sure host 0 finished writing before others continue.
    jax.experimental.multihost_utils.sync_global_devices("save:" + path)
