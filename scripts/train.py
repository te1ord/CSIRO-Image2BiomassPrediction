"""
Training script for biomass prediction models using PyTorch Lightning

Uses pre-defined fold assignments (created by scripts/data_split.py)
Supports both single-stream (full image) and two-stream (left/right split) modes.
Includes W&B logging for experiment tracking.

Usage:
    # Default training with all folds
    python scripts/train.py
    
    # Train specific folds only
    python scripts/train.py training.folds_to_train=[0,1]
    
    # Disable W&B logging
    python scripts/train.py logging.enabled=false
    
    # Single-stream model
    python scripts/train.py model=single_stream
    
    # DINOv2 with tiling
    python scripts/train.py model=dinov2_tiled
    
    # Custom experiment name
    python scripts/train.py logging.experiment_name=my_experiment
"""
import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
import pytorch_lightning as pl

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.two_stream import build_model, get_stream_mode
from src.trainer.lightning_trainer import train_kfold
from src.trainer.lightning_datamodule import load_fold_assignments


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Main training function"""
    
    print("=" * 70)
    print("CSIRO Biomass Prediction - Lightning Training")
    print("=" * 70)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Set seed
    pl.seed_everything(cfg.seed)
    
    # Determine stream mode from model type
    stream_mode = get_stream_mode(cfg.model.model_type)
    print(f"\nModel type: {cfg.model.model_type}")
    print(f"Stream mode: {stream_mode}")
    print(f"Backbone: {cfg.model.backbone.name}")
    
    # W&B settings
    use_wandb = cfg.logging.enabled
    print(f"\nW&B logging: {'enabled' if use_wandb else 'disabled'}")
    if use_wandb:
        print(f"  Project: {cfg.logging.project}")
        print(f"  Entity: {cfg.logging.entity}")
        print(f"  Experiment: {cfg.logging.experiment_name}")
        print(f"  Log model: {cfg.logging.log_model}")
        print(f"  Log gradients: {cfg.logging.log_gradients}")
        print(f"  Log predictions: {cfg.logging.log_predictions}")
    
    # Load pre-defined fold assignments
    print("\n[1/3] Loading fold assignments...")
    fold_df = load_fold_assignments(cfg.data.fold_assignments_csv)
    print(f"✓ Loaded {len(fold_df)} samples with pre-defined folds")
    
    # Model factory function
    def model_fn():
        grid = tuple(cfg.model.tiled.grid) if "tiled" in cfg.model.model_type else None
        return build_model(
            model_type=cfg.model.model_type,
            backbone_name=cfg.model.backbone.name,
            pretrained=cfg.model.backbone.pretrained,
            dropout=cfg.model.heads.dropout,
            hidden_ratio=cfg.model.heads.hidden_ratio,
            grid=grid,
        )
    
    print(f"\n[2/3] Model: {cfg.model.model_type} with {cfg.model.backbone.name}")
    
    # Determine which folds to train
    folds_to_train = cfg.training.get("folds_to_train", None)
    if folds_to_train is not None:
        # Convert OmegaConf ListConfig to Python list
        if isinstance(folds_to_train, ListConfig):
            folds_to_train = list(folds_to_train)
        print(f"  Training specific folds: {folds_to_train}")
    else:
        print(f"  Training all {cfg.data.n_folds} folds")
    
    # Training
    print(f"\n[3/3] Starting training...")
    print(f"  Stage 1: {cfg.training.stage1.epochs} epochs (frozen), LR={cfg.training.stage1.lr}")
    print(f"  Stage 2: {cfg.training.stage2.epochs} epochs (fine-tune), LR={cfg.training.stage2.lr}")
    
    results = train_kfold(
        model_fn=model_fn,
        fold_df=fold_df,
        image_dir=cfg.data.train_image_dir,
        n_folds=cfg.data.n_folds,
        folds_to_train=folds_to_train,
        stream_mode=stream_mode,
        # Training settings
        img_size=cfg.data.img_size,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        # Stage settings
        freeze_epochs=cfg.training.stage1.epochs,
        unfreeze_epochs=cfg.training.stage2.epochs,
        freeze_lr=cfg.training.stage1.lr,
        unfreeze_lr=cfg.training.stage2.lr,
        # Trainer settings
        precision=cfg.training.precision,
        checkpoint_dir=cfg.checkpoint_dir,
        # Logging settings
        use_wandb=cfg.logging.enabled,
        wandb_project=cfg.logging.project,
        wandb_entity=cfg.logging.entity,
        experiment_name=cfg.logging.experiment_name,
        log_dir=cfg.logging.log_dir,
        # W&B specific logging
        log_model=cfg.logging.log_model,
        log_gradients=cfg.logging.log_gradients,
        log_predictions=cfg.logging.log_predictions,
        # Pass full Hydra config for complete logging
        hydra_config=cfg,
        # Other settings
        early_stopping=cfg.training.early_stopping.enabled,
        patience=cfg.training.early_stopping.patience,
        seed=cfg.seed,
    )
    
    print("\n✓ Training complete!")
    print(f"Checkpoints saved to: {cfg.checkpoint_dir}")
    
    return results


if __name__ == "__main__":
    main()
