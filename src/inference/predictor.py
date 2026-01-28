"""
Inference module for model predictions

Supports:
- Single-stream and two-stream models
- Multi-fold ensemble
- Test-Time Augmentation (TTA)
- Post-processing constraint reconciliation for physical consistency
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
import glob

from src.datasets.biomass_dataset import TestBiomassDataset
from src.augmentations.transforms import get_val_transforms, get_tta_transforms


def reconcile_biomass_predictions(predictions: np.ndarray) -> np.ndarray:
    """
    Apply constraint reconciliation to ensure physical consistency.
    
    Projects predictions onto the subspace satisfying:
    - GDM = Dry_Green + Dry_Clover
    - Dry_Total = GDM + Dry_Dead
    
    Uses orthogonal projection: P = I - C^T(CC^T)^{-1}C
    
    Args:
        predictions: Array [N, 5] with [Green, Dead, Clover, GDM, Total]
                    (order matching ALL_TARGET_COLS)
    
    Returns:
        Reconciled predictions [N, 5] satisfying the constraints
    
    Reference:
        This is a form of "reconciliation" or "coherent adjustment" 
        from hierarchical forecasting literature.
    """
    # Column order: [Green, Dead, Clover, GDM, Total]
    # Constraints:
    #   Green + Clover - GDM = 0       → [1, 0, 1, -1, 0]
    #   Dead + GDM - Total = 0         → [0, 1, 0, 1, -1]
    
    C = np.array([
        [1, 0, 1, -1, 0],   # Green + Clover = GDM
        [0, 1, 0, 1, -1],   # Dead + GDM = Total
    ], dtype=np.float64)
    
    # Projection matrix: P = I - C^T @ (C @ C^T)^{-1} @ C
    C_T = C.T
    CCT_inv = np.linalg.inv(C @ C_T)
    P = np.eye(5) - C_T @ CCT_inv @ C
    
    # Apply projection: Y_reconciled = P @ Y
    # predictions is [N, 5], we need to transpose for matrix multiplication
    Y = predictions.T  # [5, N]
    Y_reconciled = P @ Y  # [5, N]
    Y_reconciled = Y_reconciled.T  # [N, 5]
    
    # Clip to non-negative (biomass can't be negative)
    Y_reconciled = np.clip(Y_reconciled, 0, None)
    
    return Y_reconciled


def compute_constraint_violation(predictions: np.ndarray) -> Dict[str, float]:
    """
    Compute how much predictions violate the physical constraints.
    
    Args:
        predictions: Array [N, 5] with [Green, Dead, Clover, GDM, Total]
    
    Returns:
        Dictionary with violation statistics
    """
    green, dead, clover, gdm, total = predictions.T
    
    # Constraint violations
    gdm_violation = np.abs(green + clover - gdm)  # Should be 0
    total_violation = np.abs(dead + gdm - total)  # Should be 0
    
    return {
        "gdm_violation_mean": float(np.mean(gdm_violation)),
        "gdm_violation_max": float(np.max(gdm_violation)),
        "total_violation_mean": float(np.mean(total_violation)),
        "total_violation_max": float(np.max(total_violation)),
    }


class Predictor:
    """
    Predictor for biomass models.
    
    Supports ensemble of multiple folds, TTA, and both stream modes.
    Includes optional constraint reconciliation for physical consistency.
    """
    
    # Target column order for submission
    ALL_TARGET_COLS = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]
    
    def __init__(
        self,
        models: List[nn.Module],
        stream_mode: str = "two_stream",
        device: str = "cuda",
        use_amp: bool = True,
        use_tta: bool = True,
        use_reconciliation: bool = True,  # NEW: constraint reconciliation
        use_log_transform: bool = False, # Whether model was trained on log-transformed targets
        ensemble_weights: Optional[List[float]] = None,  # Weights for ensemble averaging
    ):
        """
        Args:
            models: List of trained models (one per fold)
            stream_mode: "single_stream" or "two_stream"
            device: Device for inference
            use_amp: Use automatic mixed precision
            use_tta: Use Test-Time Augmentation
            use_reconciliation: Apply post-processing to ensure GDM=Green+Clover, Total=GDM+Dead
            use_log_transform: If True, apply expm1 to model outputs.
            ensemble_weights: Optional weights for weighted averaging. If None, uses equal weights.
                             Weights are automatically normalized to sum to 1.
                             Example: [0.3, 0.2, 0.2, 0.15, 0.15] for 5 folds
        """
        self.models = [m.to(device).eval() for m in models]
        self.stream_mode = stream_mode
        self.device = device
        self.use_amp = use_amp and device == "cuda"
        self.use_tta = use_tta
        self.use_reconciliation = use_reconciliation
        self.use_log_transform = use_log_transform
        
        # Setup ensemble weights
        if ensemble_weights is not None:
            if len(ensemble_weights) != len(models):
                raise ValueError(
                    f"ensemble_weights length ({len(ensemble_weights)}) must match "
                    f"number of models ({len(models)})"
                )
            # Normalize weights to sum to 1
            total = sum(ensemble_weights)
            self.ensemble_weights = torch.tensor(
                [w / total for w in ensemble_weights], 
                dtype=torch.float32
            )
            print(f"Using weighted ensemble: {[f'{w:.3f}' for w in self.ensemble_weights.tolist()]}")
        else:
            self.ensemble_weights = None  # Will use simple mean
        
    @torch.no_grad()
    def predict_batch(self, batch) -> np.ndarray:
        """
        Predict for a batch, averaging across all models.
        
        Returns:
            Array [B, 5] with [Green, Dead, Clover, GDM, Total]
        """
        if self.stream_mode == "single_stream":
            x = batch.to(self.device)
        else:
            x_left, x_right = batch
            x_left = x_left.to(self.device)
            x_right = x_right.to(self.device)
        
        per_model_preds = []
        
        for model in self.models:
            if self.use_amp:
                with autocast():
                    if self.stream_mode == "single_stream":
                        total, gdm, green = model(x)
                    else:
                        total, gdm, green = model(x_left, x_right)
            else:
                if self.stream_mode == "single_stream":
                    total, gdm, green = model(x)
                else:
                    total, gdm, green = model(x_left, x_right)
            
            # Inverse transform if model was trained on log targets
            if self.use_log_transform:
                total = torch.expm1(total)
                gdm = torch.expm1(gdm)
                green = torch.expm1(green)
            
            # Calculate derived targets
            dead = torch.clamp(total - gdm, min=0)
            clover = torch.clamp(gdm - green, min=0)
            
            # Stack: [Green, Dead, Clover, GDM, Total]
            five = torch.cat([green, dead, clover, gdm, total], dim=1)
            five = torch.clamp(five, min=0.0)
            per_model_preds.append(five.float().cpu())
        
        # Aggregate across models (weighted or simple average)
        stacked = torch.stack(per_model_preds, dim=0)  # [num_models, B, 5]
        
        if self.ensemble_weights is not None:
            # Weighted average: sum(weight_i * pred_i)
            # weights: [num_models] -> [num_models, 1, 1] for broadcasting
            weights = self.ensemble_weights.view(-1, 1, 1)
            result = (stacked * weights).sum(dim=0)
        else:
            # Simple mean
            result = stacked.mean(dim=0)
        
        return result.numpy()
    
    def predict_loader(self, loader: DataLoader) -> np.ndarray:
        """
        Predict for entire dataloader.
        
        Returns:
            Array [N, 5] with predictions
        """
        all_preds = []
        
        for batch in tqdm(loader, desc="Predicting"):
            batch_preds = self.predict_batch(batch)
            all_preds.append(batch_preds)
        
        return np.concatenate(all_preds, axis=0)
    
    def predict_with_tta(
        self,
        test_df: pd.DataFrame,
        image_dir: str,
        img_size: Optional[int] = 768,
        batch_size: int = 4,
        num_workers: int = 4,
        use_log_transform: bool = False, # Passed to dataset
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
            Array [N, 5] with averaged TTA predictions (reconciled if enabled)
        """
        if self.use_tta:
            transforms_list = get_tta_transforms(img_size)
        else:
            transforms_list = [get_val_transforms(img_size)]
        
        per_view_preds = []
        
        for i, transform in enumerate(transforms_list):
            print(f"TTA View {i+1}/{len(transforms_list)}")
            
            dataset = TestBiomassDataset(
                test_df, image_dir, transform, 
                stream_mode=self.stream_mode,
            )
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
        predictions = np.mean(per_view_preds, axis=0)
        
        # Apply constraint reconciliation if enabled
        if self.use_reconciliation:
            print("Applying constraint reconciliation...")
            violations_before = compute_constraint_violation(predictions)
            print(f"  Before: GDM violation={violations_before['gdm_violation_mean']:.4f}, "
                  f"Total violation={violations_before['total_violation_mean']:.4f}")
            
            predictions = reconcile_biomass_predictions(predictions)
            
            violations_after = compute_constraint_violation(predictions)
            print(f"  After:  GDM violation={violations_after['gdm_violation_mean']:.6f}, "
                  f"Total violation={violations_after['total_violation_mean']:.6f}")
        
        return predictions
    
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


def _load_checkpoint_state_dict(checkpoint_path: str, device: str = "cuda") -> dict:
    """
    Load state_dict from a checkpoint file, handling both Lightning and raw formats.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load on
        
    Returns:
        Model state_dict
    """
    # weights_only=False needed for PyTorch 2.6+ (checkpoints contain numpy scalars)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle Lightning checkpoints
    if "state_dict" in checkpoint:
        state_dict = {
            k.replace("model.", "", 1): v
            for k, v in checkpoint["state_dict"].items()
            if k.startswith("model.")
        }
    else:
        # Raw torch checkpoint
        state_dict = checkpoint
    
    return state_dict


def _filter_state_dict_for_heads(state_dict: dict) -> dict:
    """
    Filter state_dict to only include trainable parts (heads, pooling, etc.).
    Excludes frozen backbone and semantic_branch weights.
    
    Args:
        state_dict: Full model state_dict
        
    Returns:
        Filtered state_dict with only trainable components
    """
    # Prefixes to EXCLUDE (frozen during training)
    frozen_prefixes = (
        "backbone.",
        "semantic_branch.",
    )
    
    return {
        k: v for k, v in state_dict.items()
        if not k.startswith(frozen_prefixes)
    }


def share_frozen_modules(models: List[nn.Module]) -> List[nn.Module]:
    """
    Make all models share the same frozen modules (backbone, semantic_branch).
    
    This significantly reduces memory usage during ensemble inference because:
    - DINOv2 backbone: ~300MB-1.2GB depending on size
    - SigLIP semantic: ~400MB
    - MLP heads: ~few MB each
    
    Instead of loading 5x backbone + 5x semantic, we load 1x each and share.
    
    Args:
        models: List of loaded model instances
        
    Returns:
        Same models with shared frozen modules (modified in place)
    """
    import gc
    
    if len(models) <= 1:
        return models
    
    # Use first model's frozen modules as the shared ones
    shared_backbone = models[0].backbone
    shared_semantic = getattr(models[0], 'semantic_branch', None)
    shared_patch_norm = getattr(models[0], 'patch_norm', None)
    
    # Get memory before
    if torch.cuda.is_available():
        mem_before = torch.cuda.memory_allocated() / 1024**3
    
    # Point all other models to the shared modules
    for i, model in enumerate(models[1:], start=1):
        # Replace backbone reference (the old one will be garbage collected)
        model.backbone = shared_backbone
        
        # Replace semantic branch if it exists
        if shared_semantic is not None and hasattr(model, 'semantic_branch') and model.semantic_branch is not None:
            model.semantic_branch = shared_semantic
        
        # patch_norm is usually small but let's share it too for consistency
        if shared_patch_norm is not None and hasattr(model, 'patch_norm'):
            model.patch_norm = shared_patch_norm
    
    # Force garbage collection to free the old modules
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        mem_after = torch.cuda.memory_allocated() / 1024**3
        print(f"✓ Shared frozen modules across {len(models)} models")
        print(f"  Memory: {mem_before:.2f}GB → {mem_after:.2f}GB (saved ~{mem_before - mem_after:.2f}GB)")
    else:
        print(f"✓ Shared frozen modules across {len(models)} models")
    
    return models


def load_models(
    model_fn,
    checkpoint_dir: str,
    folds: Optional[List[int]] = None,
    device: str = "cuda",
    checkpoint_template: str = "best_model_fold{fold}.ckpt",
    share_frozen: bool = False,
) -> List[nn.Module]:
    """
    Load model checkpoints from a directory.
    
    Supports:
    - Loading specific folds by number (e.g., folds=[0, 2, 4])
    - Loading all available checkpoints (folds=None)
    - Memory-efficient loading with shared frozen modules (share_frozen=True)
    
    Args:
        model_fn: Function returning a fresh model instance
        checkpoint_dir: Folder with checkpoint files
        folds: List of fold numbers to load. If None, loads all matching checkpoints.
        device: Device to place models on
        checkpoint_template: Template for checkpoint filename with {fold} placeholder
        share_frozen: If True, share backbone and semantic_branch across all models
                     to significantly reduce memory usage during ensemble inference.
                     Only the MLP heads and learnable pooling modules are loaded per-fold.
        
    Returns:
        List of loaded models
        
    Examples:
        # Load all checkpoints (memory-intensive)
        models = load_models(model_fn, "logs/exp", folds=None)
        
        # Load with shared backbone (memory-efficient)
        models = load_models(model_fn, "logs/exp", folds=None, share_frozen=True)
        
        # Load only folds 0, 2, 4
        models = load_models(model_fn, "logs/exp", folds=[0, 2, 4])
        
        # Load single fold
        models = load_models(model_fn, "logs/exp", folds=[1])
    """
    checkpoint_paths = []
    
    if folds is not None:
        # Load specific folds
        for fold_idx in folds:
            filename = checkpoint_template.format(fold=fold_idx)
            ckpt_path = os.path.join(checkpoint_dir, filename)
            if os.path.exists(ckpt_path):
                checkpoint_paths.append(ckpt_path)
            else:
                print(f"⚠ Warning: Checkpoint not found: {ckpt_path}")
    else:
        # Load all checkpoints matching the pattern
        # Convert template to glob pattern: "best_model_fold{fold}.ckpt" -> "best_model_fold*.ckpt"
        glob_pattern = checkpoint_template.replace("{fold}", "*")
        checkpoint_paths = sorted(glob.glob(os.path.join(checkpoint_dir, glob_pattern)))
        
        # Fallback: try generic *.ckpt if no matches
        if len(checkpoint_paths) == 0:
            checkpoint_paths = sorted(glob.glob(os.path.join(checkpoint_dir, "*.ckpt")))
    
    if len(checkpoint_paths) == 0:
        raise FileNotFoundError(
            f"No checkpoints found in {checkpoint_dir}. "
            f"Looked for: {checkpoint_template.format(fold='*') if folds is None else [checkpoint_template.format(fold=f) for f in folds]}"
        )
    
    folds_desc = str(folds) if folds else "all"
    share_desc = " (with shared frozen modules)" if share_frozen else ""
    print(f"Loading {len(checkpoint_paths)} checkpoint(s) (folds: {folds_desc}){share_desc}")
    
    models = []
    shared_backbone = None
    shared_semantic = None
    
    for i, ckpt_path in enumerate(checkpoint_paths):
        print(f"→ Loading: {os.path.basename(ckpt_path)}")
        
        model = model_fn()
        state_dict = _load_checkpoint_state_dict(ckpt_path, device)
        
        if share_frozen and i > 0:
            # For subsequent folds, only load trainable parts
            # The frozen backbone/semantic will be shared from fold 0
            heads_state_dict = _filter_state_dict_for_heads(state_dict)
            model.load_state_dict(heads_state_dict, strict=False)
            
            # Share frozen modules from first model
            model.backbone = shared_backbone
            if shared_semantic is not None and hasattr(model, 'semantic_branch'):
                model.semantic_branch = shared_semantic
            
            # IMPORTANT: Re-register feature hooks on the shared backbone
            # The original hooks point to the first model's _intermediate_features dict
            # We need hooks that point to THIS model's dict
            if hasattr(model, 'reregister_hooks'):
                model.reregister_hooks()
        else:
            # First model (or share_frozen=False): load everything
            model.load_state_dict(state_dict, strict=False)
            
            if share_frozen:
                # Store references to frozen modules for sharing
                shared_backbone = model.backbone
                shared_semantic = getattr(model, 'semantic_branch', None)
        
        model.to(device)
        model.eval()
        models.append(model)
    
    # Force garbage collection if sharing
    if share_frozen and len(models) > 1:
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print(f"✓ Loaded {len(models)} model(s) successfully")
    return models


# Legacy aliases for backwards compatibility
def load_single_model(model_fn, checkpoint_path: str, device: str = "cuda") -> nn.Module:
    """Load a single model (legacy wrapper)."""
    checkpoint_dir = os.path.dirname(checkpoint_path)
    filename = os.path.basename(checkpoint_path)
    # Create a dummy template that matches the exact filename
    models = load_models(model_fn, checkpoint_dir, folds=[0], device=device, checkpoint_template=filename.replace("0", "{fold}") if "0" in filename else filename)
    return models[0] if models else load_models(model_fn, checkpoint_dir, folds=None, device=device, checkpoint_template=filename)[0]


def load_models_from_dir(model_fn, checkpoint_dir: str, device: str = "cuda", pattern: str = "*.ckpt") -> List[nn.Module]:
    """Load all models from directory (legacy wrapper)."""
    return load_models(model_fn, checkpoint_dir, folds=None, device=device)


load_fold_models = load_models_from_dir
