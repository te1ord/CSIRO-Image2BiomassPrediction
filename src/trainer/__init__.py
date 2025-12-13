# Legacy PyTorch trainer
from src.trainer.pytorch_trainer import Trainer, train_kfold as train_kfold_pytorch

# Lightning trainer
from src.trainer.lightning_module import BiomassLightningModule
from src.trainer.lightning_datamodule import BiomassDataModule
from src.trainer.lightning_trainer import (
    train_fold,
    train_kfold,
    create_callbacks,
    create_loggers,
    WandbGradientWatchCallback,
    WandbPredictionLoggerCallback,
)

__all__ = [
    # Legacy
    "Trainer",
    "train_kfold_pytorch",
    # Lightning
    "BiomassLightningModule",
    "BiomassDataModule",
    "train_fold",
    "train_kfold",
    "create_callbacks",
    "create_loggers",
    # Callbacks
    "WandbGradientWatchCallback",
    "WandbPredictionLoggerCallback",
]
