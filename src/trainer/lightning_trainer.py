"""
Lightning Trainer utilities and K-Fold training function

Uses pre-defined fold assignments (no on-the-fly splitting)
"""
import os
from typing import Dict, List, Callable, Optional, Any, Tuple
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar,
    Callback,
)
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from omegaconf import DictConfig, OmegaConf

from src.callbacks.gradient_callback import WandbGradientWatchCallback
from src.callbacks.prediction_callback import WandbPredictionLoggerCallback

from src.trainer.lightning_module import BiomassLightningModule
from src.trainer.lightning_datamodule import BiomassDataModule


def create_callbacks(
    checkpoint_dir: str,
    fold: int,
    early_stopping: bool = False,
    patience: int = 10,
) -> List[pl.Callback]:
    """Create Lightning callbacks"""
    callbacks = []
    
    # Model checkpoint - save best by val/score (R²)
    callbacks.append(
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=f"best_model_fold{fold}",
            monitor="val/score",
            mode="max",
            save_top_k=1,
            save_last=False,
            verbose=True,
        )
    )
    
    # Learning rate monitor
    callbacks.append(LearningRateMonitor(logging_interval="step"))
    
    # Rich progress bar
    callbacks.append(RichProgressBar())
    
    # Early stopping (optional)
    if early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor="val/score",
                mode="max",
                patience=patience,
                verbose=True,
            )
        )
    
    return callbacks


def create_loggers(
    experiment_name: str,
    fold: int,
    use_wandb: bool = True,
    wandb_project: str = "csiro-biomass",
    wandb_entity: Optional[str] = None,
    log_dir: str = "logs",
    config: Optional[Dict[str, Any]] = None,
    log_model: bool = False,
) -> List:
    """Create Lightning loggers"""
    loggers = []
    
    # CSV logger (always enabled)
    csv_logger = CSVLogger(
        save_dir=log_dir,
        name=experiment_name,
        version=f"fold{fold}",
    )
    loggers.append(csv_logger)
    
    # W&B logger (optional)
    if use_wandb:
        # Determine log_model setting
        # Options: True, False, "all", or "checkpoint"
        if log_model:
            log_model_setting = "all"  # Log all checkpoints
        else:
            log_model_setting = False
        
        wandb_logger = WandbLogger(
            project=wandb_project,
            entity=wandb_entity,
            name=f"{experiment_name}_fold{fold}",
            save_dir=log_dir,
            log_model=log_model_setting,  # This controls checkpoint uploading
            config=config,  # Log all hyperparameters
            tags=[f"fold{fold}", experiment_name],
        )
        loggers.append(wandb_logger)
    
    return loggers


def train_fold(
    model_fn: Callable,
    fold_df: pd.DataFrame,
    image_dir: str,
    fold: int,
    stream_mode: str = "two_stream",
    # Training settings
    img_size: int = 768,
    batch_size: int = 8,
    num_workers: int = 4,
    n_folds: int = 5,
    # Stage settings
    freeze_epochs: int = 5,
    unfreeze_epochs: int = 15,
    freeze_lr: float = 1e-4,
    unfreeze_lr: float = 1e-5,
    # Trainer settings
    accelerator: str = "auto",
    precision: str = "16-mixed",
    checkpoint_dir: str = "checkpoints",
    # Logging settings
    use_wandb: bool = True,
    wandb_project: str = "csiro-biomass",
    wandb_entity: Optional[str] = None,
    experiment_name: str = "experiment",
    log_dir: str = "logs",
    # W&B specific logging
    log_model: bool = False,
    log_gradients: bool = False,
    log_predictions: bool = False,
    # Full Hydra config for logging
    hydra_config: Optional[DictConfig] = None,
    # Other settings
    early_stopping: bool = False,
    patience: int = 10,
    seed: int = 42,
    # Full image resize before stream splitting
    full_image_size: Optional[Tuple[int, int]] = None,
) -> Dict:
    """
    Train a single fold using PyTorch Lightning.
    
    Args:
        model_fn: Function that returns a fresh model instance
        fold_df: DataFrame with pre-defined 'fold' column (0-indexed)
        image_dir: Path to images
        fold: Fold index (0-indexed) to use as validation
        hydra_config: Full Hydra config for W&B logging
        ... other args
        
    Returns:
        Dictionary with training results
    """
    # Set seed
    pl.seed_everything(seed + fold)
    
    total_epochs = freeze_epochs + unfreeze_epochs
    
    print(f"\n{'#'*60}")
    print(f"# FOLD {fold}/{n_folds - 1} (0-indexed)")
    print(f"{'#'*60}")
    
    # Create model
    model = model_fn()
    
    # Create Lightning module
    lightning_module = BiomassLightningModule(
        model=model,
        stream_mode=stream_mode,
        freeze_epochs=freeze_epochs,
        freeze_lr=freeze_lr,
        unfreeze_lr=unfreeze_lr,
    )
    
    # Create data module with pre-defined folds
    data_module = BiomassDataModule(
        fold_df=fold_df,
        image_dir=image_dir,
        stream_mode=stream_mode,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
        fold_idx=fold,  # 0-indexed fold
        full_image_size=full_image_size,
    )
    
    # Prepare config for logging
    # If hydra_config provided, use it; otherwise create minimal config
    if hydra_config is not None:
        # Convert OmegaConf to plain dict for W&B
        config = OmegaConf.to_container(hydra_config, resolve=True, throw_on_missing=True)
    else:
        config = {
            "fold": fold,
            "n_folds": n_folds,
            "stream_mode": stream_mode,
            "img_size": img_size,
            "batch_size": batch_size,
            "freeze_epochs": freeze_epochs,
            "unfreeze_epochs": unfreeze_epochs,
            "freeze_lr": freeze_lr,
            "unfreeze_lr": unfreeze_lr,
            "total_epochs": total_epochs,
        }
    
    # Add fold info to config
    config["fold"] = fold
    config["n_folds"] = n_folds
    
    # Create callbacks
    callbacks = create_callbacks(
        checkpoint_dir=checkpoint_dir,
        fold=fold,
        early_stopping=early_stopping,
        patience=patience,
    )
    
    # Add gradient logging callback if enabled
    if use_wandb and log_gradients:
        callbacks.append(WandbGradientWatchCallback(log_freq=100, log_graph=True))
    
    # Add prediction logging callback if enabled
    if use_wandb and log_predictions:
        callbacks.append(WandbPredictionLoggerCallback(num_samples=8, log_every_n_epochs=5))
    
    # Create loggers
    loggers = create_loggers(
        experiment_name=experiment_name,
        fold=fold,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        log_dir=log_dir,
        config=config,
        log_model=log_model,
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=total_epochs,
        accelerator=accelerator,
        precision=precision,
        callbacks=callbacks,
        logger=loggers,
        enable_progress_bar=True,
        log_every_n_steps=10,
        deterministic=True,
    )
    
    # Train
    trainer.fit(lightning_module, data_module)
    
    # Get best score from checkpoint callback
    best_score = trainer.checkpoint_callback.best_model_score
    best_score = best_score.item() if best_score is not None else 0.0
    
    # Close wandb run
    if use_wandb:
        import wandb
        wandb.finish()
    
    print(f"\nFold {fold} complete. Best Score: {best_score:.4f}")
    
    return {
        "fold": fold,
        "best_score": best_score,
        "checkpoint_path": trainer.checkpoint_callback.best_model_path,
    }


def train_kfold(
    model_fn: Callable,
    fold_df: pd.DataFrame,
    image_dir: str,
    n_folds: int = 5,
    folds_to_train: Optional[List[int]] = None,
    **kwargs,
) -> List[Dict]:
    """
    Train with K-Fold cross-validation using pre-defined folds.
    
    Args:
        model_fn: Function that returns a fresh model instance
        fold_df: DataFrame with pre-defined 'fold' column (0-indexed)
        image_dir: Path to images
        n_folds: Number of folds (for display/logging)
        folds_to_train: Optional list of fold indices to train (0-indexed).
                       If None, trains all folds [0, 1, ..., n_folds-1]
        **kwargs: Additional arguments for train_fold
        
    Returns:
        List of fold results
    """
    # Determine which folds to train
    if folds_to_train is None:
        # Get unique folds from the dataframe
        available_folds = sorted(fold_df["fold"].unique())
        folds_to_train = available_folds
    
    n_folds_actual = len(folds_to_train)
    
    print(f"\n{'='*60}")
    print(f"Training {n_folds_actual} folds: {folds_to_train}")
    print(f"{'='*60}")
    
    results = []
    
    for fold in folds_to_train:
        fold_result = train_fold(
            model_fn=model_fn,
            fold_df=fold_df,
            image_dir=image_dir,
            fold=fold,
            n_folds=n_folds,
            **kwargs,
        )
        results.append(fold_result)
    
    # Summary
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    
    scores = [r["best_score"] for r in results]
    print(f"Mean Score: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    for r in results:
        print(f"  Fold {r['fold']}: {r['best_score']:.4f}")
    
    return results
