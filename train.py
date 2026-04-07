import os
import random
import hydra
import torch
import uuid
import sys
import os
import numpy as np
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from termcolor import colored
from pytorch_lightning.loggers import WandbLogger
from typing import Optional, Any, Dict, List, Union
from pathlib import Path
from datetime import timedelta
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from datamodule.base import create_datamodule
from models.base import create_model
from utils.misc import compute_model_dim, timestamp_str

OmegaConf.register_new_resolver("eval", eval, replace=True)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, PROJECT_ROOT)
timestamp = timestamp_str()


def setup_trainer(
    gpus: Union[int, List[int]], 
    save_checkpoint: bool, 
    logger: Optional[WandbLogger], 
    checkpoint_interval: int, 
    experiment_name: str, 
    validation_interval: float, 
    training_epoch: int,
    checkpoint_dir: Optional[str]=None,
) -> pl.Trainer:
    """ Creates the Pytorch Lightning trainer object.

    Argus:
        gpus[Union[int, List[int]]]: The number of GPUs (if more than 1, uses DDP).
        test [bool]: Whether to use a test dataset.
        save_checkpoint [bool]: Whether to save checkpoints.
        logger [Optional[WandbLogger]]: The logger object, set to None if logging is disabled.
        checkpoint_interval [int]: The number of minutes between checkpoints.
        checkpoint_dir [str]: The directory in which to save checkpoints (a subdirectory will be created according to the experiment ID).
        validation_interval [float]: How often to run the validation step, either as a proportion of the training epoch or as a number of batches.
    
    Returns:
        pl.Trainer. The trainer object.
    """
    args: Dict[str, Any] = {}

    if (isinstance(gpus, list) and len(gpus) > 1) or (
        isinstance(gpus, int) and gpus > 1
    ):
        args = {
            **args,
            "strategy": DDPStrategy(find_unused_parameters=False),
        }
    
    if validation_interval is not None:
        args = {**args, "val_check_interval": validation_interval}
    callbacks: List[Callback] = []
    if logger is not None:
        experiment_id = timestamp
    else:
        experiment_id = str(uuid.uuid1()) # Create unique identifiers
    if save_checkpoint:
        if checkpoint_dir is not None:
            dirpath = Path(checkpoint_dir).resolve()
        else:
            dirpath = PROJECT_ROOT / "checkpoints" / experiment_name / experiment_id
        pl.utilities.rank_zero_info(f"Saving checkpoints to {dirpath}")

        every_n_checkpoint = ModelCheckpoint(
            monitor="val_loss",
            save_last=True,
            dirpath=dirpath,
            train_time_interval=timedelta(minutes=checkpoint_interval)
        )
        epoch_end_checkpoint = ModelCheckpoint(
            monitor="val_loss",
            save_last=True,
            dirpath=dirpath,
            save_on_train_epoch_end=True
        )
        epoch_end_checkpoint.CHECKPOINT_NAME_LAST = "epoch-{epoch}-end"
        callbacks.extend([every_n_checkpoint, epoch_end_checkpoint])

    trainer = pl.Trainer(
        enable_checkpointing=save_checkpoint,
        callbacks=callbacks,
        max_epochs=training_epoch,
        gradient_clip_val=1.0, # To prevent the gradient from being too large, crop the gradient
        accelerator='gpu',
        devices=gpus, 
        precision="16-mixed",
        limit_val_batches=0,
        logger=False if logger is None else logger,
        **args, 
    )
    return trainer


def setup_logger(is_log: bool, experiment_name: str, project_name: str, config_values: Dict[str, Any]) -> Optional[WandbLogger]:
    """ Setup the logger, log the data during the experiment.
    """
    if not is_log:
        pl.utilities.rank_zero_info("Disabling all logs")
        return None
    
    logger = WandbLogger(
        name=experiment_name, 
        project=project_name, 
        log_model=True,
    )
    logger.log_hyperparams(config_values)
    return logger


@hydra.main(version_base=None, config_path="./configs", config_name="default")
def run_training(config: DictConfig) -> None:
    ## compute modeling dimension according to task
    config.model.d_x = compute_model_dim(config.task)
    if os.environ.get('SLURM') is not None:
        config.slurm = True # update slurm config

    torch.set_float32_matmul_precision('high')
    color_name = colored(config["exp_name"], "green")
    pl.utilities.rank_zero_info(f"Experiment name: {color_name}")

    ## create wandb logger to log training process
    logger = setup_logger(
        not config["no_logging"],
        config["exp_name"],
        config["task"]["name"],
        config
    )

    ## create trainer to control training process
    trainer = setup_trainer(
        config["gpus"],
        save_checkpoint=not config["no_checkpointing"],
        logger=logger,
        checkpoint_interval=config["task"]["train"]["checkpoint_interval"],
        experiment_name = config["exp_name"],
        checkpoint_dir=config["exp_dir"],
        validation_interval=None,
        training_epoch=config["task"]["train"]["num_epochs"]
    )

    ## prepare data module for train and val
    dm = create_datamodule(cfg=config.task.datamodule, slurm=config.slurm)

    ## dataloader length (used by transformer)
    train_dataloader_len = len(dm.get_train_dataloader())

    ## create model and optimizer
    mdl = create_model(config, slurm=config.slurm)
    mdl.train_dataloader_len = train_dataloader_len

    if logger is not None:
        logger.watch(mdl, log="gradients", log_freq=100)
    trainer.fit(model=mdl, datamodule=dm)


if __name__ == '__main__':
    ## set random seed
    seed = 2024
    torch.backends.cudnn.benchmark = False     
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    run_training()