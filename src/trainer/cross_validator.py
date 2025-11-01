"""
Cross-validation trainer for Lasso models
"""
import random
import numpy as np
import torch
from typing import List, Tuple
from src.models.dinov2_lasso import LassoEnsemble


class CrossValidator:
    """Handles cross-validation training"""
    
    def __init__(
        self,
        n_splits: int = 5,
        train_ratio: float = 0.8,
        random_seed: int = 42
    ):
        """
        Args:
            n_splits: Number of cross-validation splits
            train_ratio: Ratio of training data in each split
            random_seed: Random seed for reproducibility
        """
        self.n_splits = n_splits
        self.train_ratio = train_ratio
        self.random_seed = random_seed
        
    def create_splits(self, n_samples: int) -> List[Tuple[List[int], List[int]]]:
        """
        Create random train/val splits
        
        Args:
            n_samples: Total number of samples
            
        Returns:
            List of (train_indices, val_indices) tuples
        """
        lst = list(range(n_samples))
        random.seed(self.random_seed)
        random.shuffle(lst)
        
        splits = []
        for i in range(self.n_splits):
            temp_lst = lst.copy()
            random.seed(self.random_seed + i)
            random.shuffle(temp_lst)
            
            split_point = int(len(temp_lst) * self.train_ratio)
            train_idxs = temp_lst[:split_point]
            val_idxs = temp_lst[split_point:]
            splits.append((train_idxs, val_idxs))
            
        return splits
    
    def train(
        self,
        embeds: List[torch.Tensor],
        targets: List[List[torch.Tensor]],
        lasso_ensemble: LassoEnsemble
    ) -> dict:
        """
        Train Lasso ensemble with cross-validation
        
        Args:
            embeds: List of embedding tensors
            targets: List of target tensors for each of 5 targets
            lasso_ensemble: LassoEnsemble object to train
            
        Returns:
            Dictionary with training metrics
        """
        # Convert to numpy arrays
        embeds_np = np.array(torch.cat(embeds))
        
        # Create splits
        splits = self.create_splits(len(embeds))
        
        results = {}
        
        # Train each target
        for target_idx in range(5):
            print(f"\n=== Target {target_idx + 1} ===")
            targets_np = np.array(torch.cat(targets[target_idx]))
            
            split_scores = []
            
            for fold_idx, (train_idxs, val_idxs) in enumerate(splits):
                print(f"Fold {fold_idx + 1}:")
                
                X_train, y_train = embeds_np[train_idxs], targets_np[train_idxs]
                X_val, y_val = embeds_np[val_idxs], targets_np[val_idxs]
                
                train_r2, val_r2 = lasso_ensemble.train_fold(
                    X_train, y_train, X_val, y_val, target_idx, fold_idx
                )
                
                print(f"  Train R²: {train_r2:.4f}")
                print(f"  Val R²: {val_r2:.4f}")
                
                split_scores.append((train_r2, val_r2))
            
            # Calculate averages
            avg_train_r2 = np.mean([score[0] for score in split_scores])
            avg_val_r2 = np.mean([score[1] for score in split_scores])
            
            print(f"\nTarget {target_idx + 1} Average:")
            print(f"  Avg Train R²: {avg_train_r2:.4f}")
            print(f"  Avg Val R²: {avg_val_r2:.4f}")
            
            results[f"target_{target_idx + 1}"] = {
                "avg_train_r2": avg_train_r2,
                "avg_val_r2": avg_val_r2,
                "fold_scores": split_scores
            }
        
        return results

