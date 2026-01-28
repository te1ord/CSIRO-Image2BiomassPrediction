"""
Biomass Dataset for CSIRO competition
Supports both single-stream (full image) and two-stream (left/right split) modes
"""
import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Optional, Callable, Tuple, Union, Dict


class BiomassDataset(Dataset):
    """
    Flexible dataset supporting both single-stream and two-stream processing.
    
    Modes:
    - single_stream: Returns full image
    - two_stream: Crops each 2000x1000 image into two 1000x1000 patches (left/right)
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
        stream_mode: str = "two_stream",  # "single_stream" or "two_stream"
        full_image_size: Optional[Tuple[int, int]] = None,  # (width, height) to resize full image before processing
        use_log_transform: bool = False, # If True, apply log1p to targets
        return_aux_targets: bool = False,
        state_to_idx: Optional[Dict[str, int]] = None,
        species_to_idx: Optional[Dict[str, int]] = None,
    ):
        """
        Args:
            df: DataFrame with image_path and target columns
            image_dir: Root directory containing images
            transform: Albumentations transform to apply
            mode: 'train' or 'test'
            stream_mode: 'single_stream' (full image) or 'two_stream' (left/right split)
            full_image_size: Optional (width, height) to resize full image before stream splitting.
                            None = keep original size. Use to test effect of early downscaling.
            use_log_transform: If True, applies log(1+x) to target variables during training.
        """
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.mode = mode
        self.stream_mode = stream_mode
        self.full_image_size = full_image_size
        self.use_log_transform = use_log_transform
        self.return_aux_targets = return_aux_targets

        self.state_to_idx = state_to_idx
        self.species_to_idx = species_to_idx
        if self.return_aux_targets:
            if self.state_to_idx is None:
                states = sorted(self.df["State"].dropna().unique().tolist())
                self.state_to_idx = {s: i for i, s in enumerate(states)}
            if self.species_to_idx is None:
                species = sorted(self.df["Species"].dropna().unique().tolist())
                self.species_to_idx = {s: i for i, s in enumerate(species)}
        
        if stream_mode not in ["single_stream", "two_stream"]:
            raise ValueError(f"stream_mode must be 'single_stream' or 'two_stream', got {stream_mode}")
        
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
    
    def _resize_full_image(self, img: np.ndarray) -> np.ndarray:
        """Resize full image before stream processing (if configured)"""
        if self.full_image_size is None:
            return img
        target_w, target_h = self.full_image_size
        return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    
    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = self._load_image(row["image_path"])
        
        # Optionally resize full image before stream splitting
        img = self._resize_full_image(img)
        
        if self.stream_mode == "single_stream":
            # Single stream: use full image
            if self.transform:
                img = self.transform(image=img)["image"]
            
            if self.mode == "train":
                targets_np = np.array([
                    row["Dry_Total_g"],
                    row["GDM_g"],
                    row["Dry_Green_g"],
                ], dtype=np.float32)
                
                # Apply log transform if enabled
                if self.use_log_transform:
                    targets_np = np.log1p(targets_np)
                    
                targets = torch.from_numpy(targets_np)
                if self.return_aux_targets:
                    aux_targets = {
                        "height_cm": torch.tensor([float(row["Height_Ave_cm"])], dtype=torch.float32),
                        "ndvi": torch.tensor([float(row["Pre_GSHH_NDVI"])], dtype=torch.float32),
                        "state": torch.tensor(self.state_to_idx[str(row["State"])], dtype=torch.long),
                        "species": torch.tensor(self.species_to_idx[str(row["Species"])], dtype=torch.long),
                    }
                    return img, targets, aux_targets

                return img, targets
            else:
                return img, row.get("sample_id", idx)
        
        else:
            # Two stream: split into left/right
            left, right = self._split_image(img)
            
            if self.transform:
                left = self.transform(image=left)["image"]
                right = self.transform(image=right)["image"]
            
            if self.mode == "train":
                targets_np = np.array([
                    row["Dry_Total_g"],
                    row["GDM_g"],
                    row["Dry_Green_g"],
                ], dtype=np.float32)
                
                # Apply log transform if enabled
                if self.use_log_transform:
                    targets_np = np.log1p(targets_np)
                
                targets = torch.from_numpy(targets_np)
                if self.return_aux_targets:
                    aux_targets = {
                        "height_cm": torch.tensor([float(row["Height_Ave_cm"])], dtype=torch.float32),
                        "ndvi": torch.tensor([float(row["Pre_GSHH_NDVI"])], dtype=torch.float32),
                        "state": torch.tensor(self.state_to_idx[str(row["State"])], dtype=torch.long),
                        "species": torch.tensor(self.species_to_idx[str(row["Species"])], dtype=torch.long),
                    }
                    return left, right, targets, aux_targets

                return left, right, targets
            else:
                return left, right, row.get("sample_id", idx)


# Backward compatibility aliases
TwoStreamBiomassDataset = BiomassDataset


class TestBiomassDataset(Dataset):
    """
    Test dataset for inference.
    Supports both single-stream and two-stream modes.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str,
        transform: Optional[Callable] = None,
        stream_mode: str = "two_stream"
    ):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.stream_mode = stream_mode
        self.paths = self.df["image_path"].values
        
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int):
        filename = os.path.basename(self.paths[idx])
        full_path = os.path.join(self.image_dir, filename)
        
        img = cv2.imread(full_path)
        if img is None:
            # img = np.zeros((1000, 2000, 3), dtype=np.uint8)
            raise ValueError(f"Image not found: {full_path}")
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.stream_mode == "single_stream":
            # Single stream: use full image
            if self.transform:
                img = self.transform(image=img)["image"]
            return img
        
        else:
            # Two stream: split into left/right halves
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
