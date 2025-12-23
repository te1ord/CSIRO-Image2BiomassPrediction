"""
PyTorch Lightning DataModule for Biomass Dataset

Uses pre-defined fold assignments from CSV (created by scripts/data_split.py)
"""
import pytorch_lightning as pl
import pandas as pd
from torch.utils.data import DataLoader
from typing import Optional, Tuple

from src.datasets.biomass_dataset import BiomassDataset
from src.augmentations.transforms import get_train_transforms, get_val_transforms


class BiomassDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for biomass dataset.
    
    Uses pre-defined fold assignments from a CSV file with 'fold' column.
    
    Handles:
    - Data loading and preprocessing
    - Train/val splits based on pre-defined folds
    - Data augmentation
    - DataLoader creation
    """
    
    def __init__(
        self,
        fold_df: pd.DataFrame,
        image_dir: str,
        stream_mode: str = "two_stream",
        img_size: int = 768,
        batch_size: int = 8,
        num_workers: int = 4,
        # Fold settings
        fold_idx: int = 0,  # 0-indexed fold to use as validation
        # Augmentation settings
        use_augmentation: bool = True,
        # Full image resize before stream splitting
        full_image_size: Optional[Tuple[int, int]] = None,
    ):
        """
        Args:
            fold_df: DataFrame with pre-defined 'fold' column (0-indexed)
            image_dir: Path to image directory
            stream_mode: "single_stream" or "two_stream"
            img_size: Image size for transforms
            batch_size: Batch size
            num_workers: Number of data loading workers
            fold_idx: Which fold to use as validation (0-indexed)
            use_augmentation: Whether to use training augmentations
            full_image_size: Optional (width, height) to resize full image before stream splitting
        """
        super().__init__()
        self.save_hyperparameters(ignore=["fold_df"])
        
        self.fold_df = fold_df
        self.image_dir = image_dir
        self.stream_mode = stream_mode
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.fold_idx = fold_idx
        self.use_augmentation = use_augmentation
        self.full_image_size = full_image_size
        
        # Will be set in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.train_df = None
        self.val_df = None
        
    def setup(self, stage: Optional[str] = None):
        """Setup train/val datasets based on pre-defined folds"""
        
        # Validate fold column exists
        if "fold" not in self.fold_df.columns:
            raise ValueError(
                "DataFrame must have 'fold' column. "
                "Use scripts/data_split.py to create fold assignments."
            )
        
        # Split based on pre-defined fold assignments
        # val = samples where fold == fold_idx
        # train = all other samples
        self.val_df = self.fold_df[self.fold_df["fold"] == self.fold_idx].reset_index(drop=True)
        self.train_df = self.fold_df[self.fold_df["fold"] != self.fold_idx].reset_index(drop=True)
        
        print(f"Fold {self.fold_idx}: Train={len(self.train_df)}, Val={len(self.val_df)}")
        
        # Create transforms
        train_transform = (
            get_train_transforms(self.img_size) 
            if self.use_augmentation 
            else get_val_transforms(self.img_size)
        )
        val_transform = get_val_transforms(self.img_size)
        
        # Create datasets
        self.train_dataset = BiomassDataset(
            self.train_df,
            self.image_dir,
            train_transform,
            mode="train",
            stream_mode=self.stream_mode,
            full_image_size=self.full_image_size,
        )
        
        self.val_dataset = BiomassDataset(
            self.val_df,
            self.image_dir,
            val_transform,
            mode="train",
            stream_mode=self.stream_mode,
            full_image_size=self.full_image_size,
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
        n_folds = self.fold_df["fold"].nunique() if self.fold_df is not None else 0
        return {
            "fold": self.fold_idx,
            "n_folds": n_folds,
            "train_samples": len(self.train_df) if self.train_df is not None else 0,
            "val_samples": len(self.val_df) if self.val_df is not None else 0,
        }


def load_fold_assignments(csv_path: str) -> pd.DataFrame:
    """
    Load pre-defined fold assignments from CSV.
    
    Args:
        csv_path: Path to fold assignments CSV (created by scripts/data_split.py)
        
    Returns:
        DataFrame with 'fold' column
    """
    df = pd.read_csv(csv_path)
    
    if "fold" not in df.columns:
        raise ValueError(f"CSV must have 'fold' column. Found: {df.columns.tolist()}")
    
    n_folds = df["fold"].nunique()
    samples_per_fold = df["fold"].value_counts().sort_index().to_dict()
    
    print(f"Loaded fold assignments from: {csv_path}")
    print(f"  Total samples: {len(df)}")
    print(f"  Number of folds: {n_folds}")
    print(f"  Samples per fold: {samples_per_fold}")
    
    return df
