"""
Inference module for Two-Stream model predictions

Supports:
- Single model inference
- Multi-fold ensemble
- Test-Time Augmentation (TTA)
"""
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm

from src.datasets.biomass_dataset import TestBiomassDataset
from src.augmentations.transforms import get_val_transforms, get_tta_transforms


class Predictor:
    """
    Predictor for Two-Stream biomass models.
    
    Supports ensemble of multiple folds and TTA.
    """
    
    # Target column order for submission
    ALL_TARGET_COLS = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]
    
    def __init__(
        self,
        models: List[nn.Module],
        device: str = "cuda",
        use_amp: bool = True,
        use_tta: bool = True,
    ):
        """
        Args:
            models: List of trained models (one per fold)
            device: Device for inference
            use_amp: Use automatic mixed precision
            use_tta: Use Test-Time Augmentation
        """
        self.models = [m.to(device).eval() for m in models]
        self.device = device
        self.use_amp = use_amp and device == "cuda"
        self.use_tta = use_tta
        
    @torch.no_grad()
    def predict_batch(
        self,
        x_left: torch.Tensor,
        x_right: torch.Tensor,
    ) -> np.ndarray:
        """
        Predict for a batch, averaging across all models.
        
        Returns:
            Array [B, 5] with [Green, Dead, Clover, GDM, Total]
        """
        x_left = x_left.to(self.device)
        x_right = x_right.to(self.device)
        
        per_model_preds = []
        
        for model in self.models:
            if self.use_amp:
                with autocast():
                    total, gdm, green = model(x_left, x_right)
            else:
                total, gdm, green = model(x_left, x_right)
            
            # Calculate derived targets
            dead = torch.clamp(total - gdm, min=0)
            clover = torch.clamp(gdm - green, min=0)
            
            # Stack: [Green, Dead, Clover, GDM, Total]
            five = torch.cat([green, dead, clover, gdm, total], dim=1)
            five = torch.clamp(five, min=0.0)
            per_model_preds.append(five.float().cpu())
        
        # Average across models
        stacked = torch.mean(torch.stack(per_model_preds, dim=0), dim=0)
        return stacked.numpy()
    
    def predict_loader(
        self,
        loader: DataLoader,
    ) -> np.ndarray:
        """
        Predict for entire dataloader.
        
        Returns:
            Array [N, 5] with predictions
        """
        all_preds = []
        
        for x_left, x_right in tqdm(loader, desc="Predicting"):
            batch_preds = self.predict_batch(x_left, x_right)
            all_preds.append(batch_preds)
        
        return np.concatenate(all_preds, axis=0)
    
    def predict_with_tta(
        self,
        test_df: pd.DataFrame,
        image_dir: str,
        img_size: int = 768,
        batch_size: int = 4,
        num_workers: int = 4,
    ) -> np.ndarray:
        """
        Predict with Test-Time Augmentation.
        
        Args:
            test_df: Test dataframe (unique images)
            image_dir: Path to test images
            img_size: Image size
            batch_size: Batch size for inference
            num_workers: Number of data loading workers
            
        Returns:
            Array [N, 5] with averaged TTA predictions
        """
        if self.use_tta:
            transforms_list = get_tta_transforms(img_size)
        else:
            transforms_list = [get_val_transforms(img_size)]
        
        per_view_preds = []
        
        for i, transform in enumerate(transforms_list):
            print(f"TTA View {i+1}/{len(transforms_list)}")
            
            dataset = TestBiomassDataset(test_df, image_dir, transform)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )
            
            view_preds = self.predict_loader(loader)
            per_view_preds.append(view_preds)
        
        # Average TTA predictions
        return np.mean(per_view_preds, axis=0)
    
    def create_submission(
        self,
        predictions: np.ndarray,
        test_long_df: pd.DataFrame,
        test_unique_df: pd.DataFrame,
        output_path: str = "submission.csv",
    ) -> pd.DataFrame:
        """
        Create submission file from predictions.
        
        Args:
            predictions: Array [N, 5] with [Green, Dead, Clover, GDM, Total]
            test_long_df: Original test CSV (long format with sample_id)
            test_unique_df: Unique images dataframe
            output_path: Path to save submission
            
        Returns:
            Submission DataFrame
        """
        # Extract individual predictions
        green = predictions[:, 0]
        dead = predictions[:, 1]
        clover = predictions[:, 2]
        gdm = predictions[:, 3]
        total = predictions[:, 4]
        
        # Ensure non-negative and handle NaN/Inf
        def clean(x):
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            return np.maximum(0, x)
        
        green, dead, clover, gdm, total = map(clean, [green, dead, clover, gdm, total])
        
        # Create wide format dataframe
        wide = pd.DataFrame({
            "image_path": test_unique_df["image_path"],
            "Dry_Green_g": green,
            "Dry_Dead_g": dead,
            "Dry_Clover_g": clover,
            "GDM_g": gdm,
            "Dry_Total_g": total,
        })
        
        # Melt to long format
        long_preds = wide.melt(
            id_vars=["image_path"],
            value_vars=self.ALL_TARGET_COLS,
            var_name="target_name",
            value_name="target",
        )
        
        # Merge with original test to get sample_id
        submission = pd.merge(
            test_long_df[["sample_id", "image_path", "target_name"]],
            long_preds,
            on=["image_path", "target_name"],
            how="left",
        )[["sample_id", "target"]]
        
        # Final cleanup
        submission["target"] = np.nan_to_num(
            submission["target"], nan=0.0, posinf=0.0, neginf=0.0
        )
        
        # Save
        submission.to_csv(output_path, index=False)
        print(f"✓ Submission saved to {output_path}")
        print(submission.head())
        
        return submission


def load_fold_models(
    model_fn,
    checkpoint_dir: str,
    n_folds: int = 5,
    device: str = "cuda",
) -> List[nn.Module]:
    """
    Load trained models from checkpoints.
    
    Args:
        model_fn: Function that returns a fresh model instance
        checkpoint_dir: Directory containing checkpoint files
        n_folds: Number of folds
        device: Device to load models on
        
    Returns:
        List of loaded models
    """
    models = []
    
    for fold in range(1, n_folds + 1):
        checkpoint_path = os.path.join(checkpoint_dir, f"best_model_fold{fold}.pth")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        model = model_fn()
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        print(f"✓ Loaded fold {fold} from {checkpoint_path}")
        models.append(model)
    
    return models
