import os

os.environ["WANDB_SILENT"] = "true"
import jax
import wandb
from netket.logging import RuntimeLog
from netket.utils import mpi


class WandbLoggerBase(RuntimeLog):
    def __init__(
        self,
        project,
        n_iters=None,
        log=True,
        config=None,
    ):
        self.n_iters = n_iters

        super().__init__()

        if hasattr(self, "_is_master_process"):
            is_master_process = self._is_master_process
        else:
            is_master_process = mpi.rank == 0 and jax.process_index() == 0

        if not log or not is_master_process:
            os.environ["WANDB_MODE"] = "disabled"

        wandb.init(config=config, project=project)


class WandbLogger(WandbLoggerBase):
    def __call__(self, step, item, variational_state=None):
        super().__call__(step, item, variational_state)
