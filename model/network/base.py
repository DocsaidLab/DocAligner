from pathlib import Path
from typing import Any, Dict

import docsaidkit.torch as D


class BaseMixin:

    def apply_solver_config(
        self,
        optimizer: Dict[str, Any],
        lr_scheduler: Dict[str, Any]
    ) -> None:
        self.optimizer_name, self.optimizer_opts = optimizer.values()
        self.sche_name, self.sche_opts, self.sche_pl_opts = lr_scheduler.values()

    def configure_optimizers(self):
        optimizer = D.build_optimizer(
            name=self.optimizer_name,
            model_params=self.parameters(),
            **self.optimizer_opts
        )
        scheduler = D.build_lr_scheduler(
            name=self.sche_name,
            optimizer=optimizer,
            **self.sche_opts
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                **self.sche_pl_opts
            }
        }

    def get_lr(self):
        return self.trainer.optimizers[0].param_groups[0]['lr']
