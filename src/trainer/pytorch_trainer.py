"""
PyTorch Trainer with Two-Stage Fine-Tuning Strategy

Training Strategy:
- Stage 1 (Freeze): Epochs 1-5, backbone frozen, LR=1e-4
- Stage 2 (Unfreeze): Epochs 6+, backbone unfrozen, LR=1e-5
- Model checkpoint based on best R² score
- GroupKFold by Sampling_Date to prevent data leakage
"""
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import GroupKFold
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from src.models.losses import WeightedSmoothL1Loss, CompetitionMetric
from src.datasets.biomass_dataset import TwoStreamBiomassDataset, get_groups_for_kfold
from src.augmentations.transforms import get_train_transforms, get_val_transforms


class Trainer:
    """
    PyTorch trainer with two-stage fine-tuning.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        # Stage 1 settings
        freeze_epochs: int = 5,
        freeze_lr: float = 1e-4,
        # Stage 2 settings
        unfreeze_epochs: int = 15,
        unfreeze_lr: float = 1e-5,
        # Training settings
        batch_size: int = 8,
        num_workers: int = 4,
        use_amp: bool = True,
        # Checkpoint settings
        checkpoint_dir: str = "checkpoints",
        save_best_only: bool = True,
    ):
        self.model = model.to(device)
        self.device = device
        
        self.freeze_epochs = freeze_epochs
        self.freeze_lr = freeze_lr
        self.unfreeze_epochs = unfreeze_epochs
        self.unfreeze_lr = unfreeze_lr
        self.total_epochs = freeze_epochs + unfreeze_epochs
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_amp = use_amp and device == "cuda"
        
        self.checkpoint_dir = checkpoint_dir
        self.save_best_only = save_best_only
        
        # Loss and metric
        self.criterion = WeightedSmoothL1Loss()
        self.metric = CompetitionMetric()
        
        # Mixed precision scaler
        self.scaler = GradScaler() if self.use_amp else None
        
    def _get_optimizer(self, stage: int) -> torch.optim.Optimizer:
        """Get optimizer for current stage"""
        if stage == 1:
            # Only train heads
            params = list(self.model.head_total.parameters()) + \
                     list(self.model.head_gdm.parameters()) + \
                     list(self.model.head_green.parameters())
            return torch.optim.AdamW(params, lr=self.freeze_lr)
        else:
            # Train entire model with lower LR
            return torch.optim.AdamW(self.model.parameters(), lr=self.unfreeze_lr)
    
    def train_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        all_losses = {"loss_total": 0.0, "loss_gdm": 0.0, "loss_green": 0.0}
        
        pbar = tqdm(loader, desc="Training")
        for batch_idx, (x_left, x_right, targets) in enumerate(pbar):
            x_left = x_left.to(self.device)
            x_right = x_right.to(self.device)
            
            y_total = targets[:, 0].to(self.device)
            y_gdm = targets[:, 1].to(self.device)
            y_green = targets[:, 2].to(self.device)
            
            optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    pred_total, pred_gdm, pred_green = self.model(x_left, x_right)
                    loss, losses = self.criterion(
                        pred_total, pred_gdm, pred_green,
                        y_total, y_gdm, y_green,
                    )
                
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                pred_total, pred_gdm, pred_green = self.model(x_left, x_right)
                loss, losses = self.criterion(
                    pred_total, pred_gdm, pred_green,
                    y_total, y_gdm, y_green,
                )
                loss.backward()
                optimizer.step()
            
            total_loss += losses["loss"]
            for k, v in losses.items():
                if k in all_losses:
                    all_losses[k] += v
            
            pbar.set_postfix(loss=f"{losses['loss']:.4f}")
        
        n_batches = len(loader)
        return {
            "loss": total_loss / n_batches,
            "loss_total": all_losses["loss_total"] / n_batches,
            "loss_gdm": all_losses["loss_gdm"] / n_batches,
            "loss_green": all_losses["loss_green"] / n_batches,
        }
    
    @torch.no_grad()
    def validate(self, loader: DataLoader) -> Dict[str, float]:
        """Validate and compute R² scores"""
        self.model.eval()
        
        all_pred_total, all_pred_gdm, all_pred_green = [], [], []
        all_y_total, all_y_gdm, all_y_green = [], [], []
        
        for x_left, x_right, targets in tqdm(loader, desc="Validating"):
            x_left = x_left.to(self.device)
            x_right = x_right.to(self.device)
            
            if self.use_amp:
                with autocast():
                    pred_total, pred_gdm, pred_green = self.model(x_left, x_right)
            else:
                pred_total, pred_gdm, pred_green = self.model(x_left, x_right)
            
            all_pred_total.append(pred_total.cpu())
            all_pred_gdm.append(pred_gdm.cpu())
            all_pred_green.append(pred_green.cpu())
            
            all_y_total.append(targets[:, 0])
            all_y_gdm.append(targets[:, 1])
            all_y_green.append(targets[:, 2])
        
        # Concatenate all predictions
        pred_total = torch.cat(all_pred_total)
        pred_gdm = torch.cat(all_pred_gdm)
        pred_green = torch.cat(all_pred_green)
        
        y_total = torch.cat(all_y_total)
        y_gdm = torch.cat(all_y_gdm)
        y_green = torch.cat(all_y_green)
        
        # Compute competition metric
        score, scores = self.metric(
            pred_total, pred_gdm, pred_green,
            y_total, y_gdm, y_green,
        )
        
        return scores
    
    def train_fold(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        image_dir: str,
        fold: int,
        img_size: int = 768,
    ) -> Dict[str, float]:
        """Train one fold"""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Create datasets
        train_transform = get_train_transforms(img_size)
        val_transform = get_val_transforms(img_size)
        
        train_dataset = TwoStreamBiomassDataset(
            train_df, image_dir, train_transform, mode="train"
        )
        val_dataset = TwoStreamBiomassDataset(
            val_df, image_dir, val_transform, mode="train"
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        
        best_score = -float("inf")
        best_epoch = 0
        
        print(f"\n{'='*60}")
        print(f"FOLD {fold}")
        print(f"{'='*60}")
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        
        for epoch in range(1, self.total_epochs + 1):
            # Determine stage
            if epoch <= self.freeze_epochs:
                stage = 1
                if epoch == 1:
                    print(f"\n--- Stage 1: Backbone Frozen (LR={self.freeze_lr}) ---")
                    self.model.freeze_backbone()
                    optimizer = self._get_optimizer(stage=1)
            else:
                stage = 2
                if epoch == self.freeze_epochs + 1:
                    print(f"\n--- Stage 2: Fine-Tuning (LR={self.unfreeze_lr}) ---")
                    self.model.unfreeze_backbone()
                    optimizer = self._get_optimizer(stage=2)
            
            print(f"\nEpoch {epoch}/{self.total_epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader, optimizer)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Val Score (R²): {val_metrics['score']:.4f}")
            print(f"    R² Total: {val_metrics['r2_total']:.4f}")
            print(f"    R² GDM: {val_metrics['r2_gdm']:.4f}")
            print(f"    R² Green: {val_metrics['r2_green']:.4f}")
            
            # Save best model
            if val_metrics["score"] > best_score:
                best_score = val_metrics["score"]
                best_epoch = epoch
                
                checkpoint_path = os.path.join(
                    self.checkpoint_dir,
                    f"best_model_fold{fold}.pth"
                )
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"  ✓ New best! Saved to {checkpoint_path}")
        
        print(f"\nFold {fold} complete. Best Score: {best_score:.4f} (Epoch {best_epoch})")
        
        return {
            "fold": fold,
            "best_score": best_score,
            "best_epoch": best_epoch,
        }


def train_kfold(
    model_fn,
    train_df: pd.DataFrame,
    image_dir: str,
    n_splits: int = 5,
    img_size: int = 768,
    **trainer_kwargs,
) -> List[Dict]:
    """
    Train with K-Fold cross-validation using GroupKFold.
    
    Args:
        model_fn: Function that returns a fresh model instance
        train_df: Training dataframe with all samples
        image_dir: Path to image directory
        n_splits: Number of folds
        img_size: Image size for training
        **trainer_kwargs: Arguments for Trainer
        
    Returns:
        List of fold results
    """
    groups = get_groups_for_kfold(train_df)
    kfold = GroupKFold(n_splits=n_splits)
    
    results = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_df, groups=groups), 1):
        print(f"\n{'#'*60}")
        print(f"# FOLD {fold}/{n_splits}")
        print(f"{'#'*60}")
        
        # Create fresh model for each fold
        model = model_fn()
        
        # Split data
        fold_train_df = train_df.iloc[train_idx].reset_index(drop=True)
        fold_val_df = train_df.iloc[val_idx].reset_index(drop=True)
        
        # Train
        trainer = Trainer(model, **trainer_kwargs)
        fold_result = trainer.train_fold(
            fold_train_df,
            fold_val_df,
            image_dir,
            fold,
            img_size,
        )
        
        results.append(fold_result)
    
    # Summary
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    
    scores = [r["best_score"] for r in results]
    print(f"Mean Score: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    for r in results:
        print(f"  Fold {r['fold']}: {r['best_score']:.4f} (Epoch {r['best_epoch']})")
    
    return results

