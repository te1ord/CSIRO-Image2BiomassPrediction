"""
PyTorch Lightning DataModule for Biomass Dataset
"""
import pytorch_lightning as pl
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
from typing import Optional, List, Tuple

from src.datasets.biomass_dataset import BiomassDataset, get_groups_for_kfold
from src.augmentations.transforms import get_train_transforms, get_val_transforms


class BiomassDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for biomass dataset.
    
    Handles:
    - Data loading and preprocessing
    - Train/val splits with GroupKFold
    - Data augmentation
    - DataLoader creation
    """
    
    def __init__(
        self,
        train_df: pd.DataFrame,
        image_dir: str,
        stream_mode: str = "two_stream",
        img_size: int = 768,
        batch_size: int = 8,
        num_workers: int = 4,
        # K-Fold settings
        n_folds: int = 5,
        fold_idx: int = 0,  # 0-indexed fold to use
        # Augmentation settings
        use_augmentation: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["train_df"])
        
        self.train_df = train_df
        self.image_dir = image_dir
        self.stream_mode = stream_mode
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.n_folds = n_folds
        self.fold_idx = fold_idx
        self.use_augmentation = use_augmentation
        
        # Will be set in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.train_indices = None
        self.val_indices = None
        
    def setup(self, stage: Optional[str] = None):
        """Setup train/val datasets"""
        # Create k-fold splits
        groups = get_groups_for_kfold(self.train_df)
        kfold = GroupKFold(n_splits=self.n_folds)
        
        # Get the specific fold
        splits = list(kfold.split(self.train_df, groups=groups))
        train_idx, val_idx = splits[self.fold_idx]
        
        self.train_indices = train_idx
        self.val_indices = val_idx
        
        # Create dataframes for this fold
        train_fold_df = self.train_df.iloc[train_idx].reset_index(drop=True)
        val_fold_df = self.train_df.iloc[val_idx].reset_index(drop=True)
        
        # Create transforms
        train_transform = get_train_transforms(self.img_size) if self.use_augmentation else get_val_transforms(self.img_size)
        val_transform = get_val_transforms(self.img_size)
        
        # Create datasets
        self.train_dataset = BiomassDataset(
            train_fold_df,
            self.image_dir,
            train_transform,
            mode="train",
            stream_mode=self.stream_mode,
        )
        
        self.val_dataset = BiomassDataset(
            val_fold_df,
            self.image_dir,
            val_transform,
            mode="train",
            stream_mode=self.stream_mode,
        )
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
    
    def get_fold_info(self) -> dict:
        """Get information about current fold"""
        return {
            "fold": self.fold_idx + 1,
            "n_folds": self.n_folds,
            "train_samples": len(self.train_indices) if self.train_indices is not None else 0,
            "val_samples": len(self.val_indices) if self.val_indices is not None else 0,
        }

