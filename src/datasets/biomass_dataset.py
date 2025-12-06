"""
Two-Stream Biomass Dataset for CSIRO competition
Crops each 2000x1000 image into two 1000x1000 patches (left/right)
"""
import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Optional, Callable, Tuple, List


class TwoStreamBiomassDataset(Dataset):
    """
    Dataset that crops each image into left/right halves for two-stream processing.
    
    Strategy:
    - Each 2000x1000 image is split into two 1000x1000 patches
    - Augmentations are applied independently to each patch
    - Returns left_img, right_img, and targets (3 primary targets)
    """
    
    # Target columns: we predict only 3, calculate the other 2
    PRIMARY_TARGETS = ["Dry_Total_g", "GDM_g", "Dry_Green_g"]
    ALL_TARGETS = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]
    
    # Competition weights
    TARGET_WEIGHTS = {
        "Dry_Total_g": 0.50,
        "GDM_g": 0.20,
        "Dry_Green_g": 0.10,
        "Dry_Dead_g": 0.10,
        "Dry_Clover_g": 0.10,
    }
    
    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str,
        transform: Optional[Callable] = None,
        mode: str = "train",
    ):
        """
        Args:
            df: DataFrame with image_path and target columns
            image_dir: Root directory containing images
            transform: Albumentations transform to apply
            mode: 'train' or 'test'
        """
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.mode = mode
        
    def __len__(self) -> int:
        return len(self.df)
    
    def _load_image(self, path: str) -> np.ndarray:
        """Load image and convert to RGB"""
        full_path = os.path.join(self.image_dir, os.path.basename(path))
        img = cv2.imread(full_path)
        if img is None:
            # Fallback: create black image if loading fails
            img = np.zeros((1000, 2000, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def _split_image(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Split image into left and right halves"""
        h, w, _ = img.shape
        mid = w // 2
        left = img[:, :mid]
        right = img[:, mid:]
        return left, right
    
    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = self._load_image(row["image_path"])
        left, right = self._split_image(img)
        
        # Apply transforms independently to each patch
        if self.transform:
            left = self.transform(image=left)["image"]
            right = self.transform(image=right)["image"]
        
        if self.mode == "train":
            # Get the 3 primary targets
            targets = torch.tensor([
                row["Dry_Total_g"],
                row["GDM_g"],
                row["Dry_Green_g"],
            ], dtype=torch.float32)
            return left, right, targets
        else:
            # Test mode: return sample_id for submission
            return left, right, row.get("sample_id", idx)


class TestBiomassDataset(Dataset):
    """
    Test dataset for inference.
    Similar to TwoStreamBiomassDataset but optimized for inference.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str,
        transform: Optional[Callable] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.paths = self.df["image_path"].values
        
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int):
        filename = os.path.basename(self.paths[idx])
        full_path = os.path.join(self.image_dir, filename)
        
        img = cv2.imread(full_path)
        if img is None:
            img = np.zeros((1000, 2000, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Split into left/right halves
        h, w, _ = img.shape
        mid = w // 2
        left = img[:, :mid]
        right = img[:, mid:]
        
        if self.transform:
            left = self.transform(image=left)["image"]
            right = self.transform(image=right)["image"]
        
        return left, right


def prepare_train_df(csv_path: str) -> pd.DataFrame:
    """
    Prepare training DataFrame by pivoting targets.
    
    The CSV has multiple rows per image (one per target).
    This function pivots to have one row per image with all targets as columns.
    """
    df = pd.read_csv(csv_path)
    
    # Check if already pivoted (has target columns)
    if "Dry_Total_g" in df.columns:
        return df
    
    # Pivot from long to wide format
    # Expected columns: sample_id, image_path, target_name, target
    if "target_name" in df.columns:
        # Extract base image path
        df["base_path"] = df["image_path"]
        
        # Pivot
        pivoted = df.pivot_table(
            index=["base_path", "Sampling_Date"] if "Sampling_Date" in df.columns else ["base_path"],
            columns="target_name",
            values="target",
            aggfunc="first"
        ).reset_index()
        
        pivoted = pivoted.rename(columns={"base_path": "image_path"})
        return pivoted
    
    return df


def get_groups_for_kfold(df: pd.DataFrame) -> np.ndarray:
    """
    Get grouping array for GroupKFold based on Sampling_Date.
    Prevents data leakage from similar images on the same day.
    """
    if "Sampling_Date" in df.columns:
        # Use date as group
        return df["Sampling_Date"].values
    else:
        # Fallback: use index (no grouping)
        return np.arange(len(df))
