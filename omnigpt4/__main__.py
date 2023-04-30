"""Entry point for omnigpt4."""
import os

import lightning.pytorch as pl
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from lightning.fabric.utilities.cloud_io import get_filesystem


class WandbSaveConfigCallback(SaveConfigCallback):
    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        if self.already_saved:
            return

        if trainer.global_rank == 0:
            for logger in trainer.loggers:
                if isinstance(logger, WandbLogger):
                    log_dir = logger.experiment.dir
                elif isinstance(logger, CSVLogger):
                    log_dir = logger.log_dir
        else:
            log_dir = None
        log_dir = trainer.strategy.broadcast(log_dir)

        assert log_dir is not None
        config_path = os.path.join(log_dir, self.config_filename)
        fs = get_filesystem(log_dir)

        if not self.overwrite:
            # check if the file exists on rank 0
            file_exists = fs.isfile(config_path) if trainer.is_global_zero else False
            # broadcast whether to fail to all ranks
            file_exists = trainer.strategy.broadcast(file_exists)
            if file_exists:
                raise RuntimeError(
                    f"{self.__class__.__name__} expected {config_path} to NOT exist. Aborting to avoid overwriting"
                    " results of a previous run. You can delete the previous config file,"
                    " set `LightningCLI(save_config_callback=None)` to disable config saving,"
                    ' or set `LightningCLI(save_config_kwargs={"overwrite": True})` to overwrite the config file.'
                )

        # save the file on rank 0
        if trainer.is_global_zero:
            # save only on rank zero to avoid race conditions.
            # the `log_dir` needs to be created as we rely on the logger to do it usually
            # but it hasn't logged anything at this point
            fs.makedirs(log_dir, exist_ok=True)
            self.parser.save(
                self.config, config_path, skip_none=False, overwrite=self.overwrite, multifile=self.multifile
            )
            self.already_saved = True

        # broadcast so that all ranks are in sync on future calls to .setup()
        self.already_saved = trainer.strategy.broadcast(self.already_saved)


def main():
    LightningCLI(
        pl.LightningModule, pl.LightningDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_callback=WandbSaveConfigCallback,
        save_config_kwargs={"config_filename": "pl_config.yaml"},
    )


if __name__ == "__main__":
    main()
