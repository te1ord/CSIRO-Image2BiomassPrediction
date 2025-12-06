"""
Training script for Two-Stream Multi-Head model

Usage:
    python scripts/train.py
    python scripts/train.py model=dinov2_tiled
    python scripts/train.py training.batch_size=16 data.img_size=512
"""
import os
import sys
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.two_stream import build_model
from src.trainer.pytorch_trainer import train_kfold
from src.datasets.biomass_dataset import prepare_train_df


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Main training function"""
    
    print("=" * 70)
    print("CSIRO Biomass Prediction - Two-Stream Multi-Head Training")
    print("=" * 70)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Set seed
    set_seed(cfg.seed)
    
    # Device
    device = cfg.device if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    # Load and prepare training data
    print("\n[1/3] Loading training data...")
    train_df = prepare_train_df(cfg.data.train_csv)
    print(f"✓ Loaded {len(train_df)} training samples")
    
    # Model factory function
    def model_fn():
        return build_model(
            model_type=cfg.model.model_type,
            backbone_name=cfg.model.backbone.name,
            pretrained=cfg.model.backbone.pretrained,
            dropout=cfg.model.heads.dropout,
            hidden_ratio=cfg.model.heads.hidden_ratio,
            grid=tuple(cfg.model.tiled.grid) if "tiled" in cfg.model.model_type else None,
        )
    
    print(f"\n[2/3] Model: {cfg.model.model_type} with {cfg.model.backbone.name}")
    
    # Training
    print(f"\n[3/3] Starting {cfg.data.n_folds}-fold cross-validation...")
    print(f"  Stage 1: {cfg.training.stage1.epochs} epochs (frozen), LR={cfg.training.stage1.lr}")
    print(f"  Stage 2: {cfg.training.stage2.epochs} epochs (fine-tune), LR={cfg.training.stage2.lr}")
    
    results = train_kfold(
        model_fn=model_fn,
        train_df=train_df,
        image_dir=cfg.data.train_image_dir,
        n_splits=cfg.data.n_folds,
        img_size=cfg.data.img_size,
        # Trainer kwargs
        device=device,
        freeze_epochs=cfg.training.stage1.epochs,
        freeze_lr=cfg.training.stage1.lr,
        unfreeze_epochs=cfg.training.stage2.epochs,
        unfreeze_lr=cfg.training.stage2.lr,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        use_amp=cfg.training.use_amp,
        checkpoint_dir=cfg.checkpoint_dir,
        save_best_only=cfg.training.save_best_only,
    )
    
    print("\n✓ Training complete!")
    print(f"Checkpoints saved to: {cfg.checkpoint_dir}")
    
    return results


if __name__ == "__main__":
    main()
