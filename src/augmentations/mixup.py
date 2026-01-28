"""
Embedding-level MixUp Augmentation

MixUp is applied after feature extraction but before MLP heads:
1. Extract features for batch
2. Mix features: mixed_feat = λ * feat_a + (1 - λ) * feat_b
3. Mix labels: mixed_labels = λ * labels_a + (1 - λ) * labels_b
4. Train on mixed features/labels

Reference: https://arxiv.org/abs/1710.09412
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional


class EmbeddingMixUp(nn.Module):
    """
    Embedding-level MixUp augmentation.
    
    Samples lambda from Beta(alpha, alpha) distribution.
    Higher alpha = more mixing (lambda closer to 0.5)
    Lower alpha = less mixing (lambda closer to 0 or 1)
    """
    
    def __init__(
        self,
        alpha: float = 0.4,
        p: float = 0.5,  # probability of applying mixup
    ):
        """
        Args:
            alpha: Beta distribution parameter. 
                   alpha=0.4 is common, alpha=1.0 gives uniform distribution
            p: Probability of applying mixup per batch
        """
        super().__init__()
        self.alpha = alpha
        self.p = p
        
    def forward(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        return_index: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Apply MixUp to features and targets.
        
        Args:
            features: Feature tensor [B, D]
            targets: Target tensor [B, num_targets]
            
        Returns:
            Tuple of (mixed_features, mixed_targets, lambda_value)
            If return_index=True, also returns the permutation index used for mixing.
        """
        if not self.training:
            if return_index:
                index = torch.arange(features.size(0), device=features.device)
                return features, targets, 1.0, index
            return features, targets, 1.0
            
        # Decide whether to apply mixup this batch
        if torch.rand(1).item() > self.p:
            if return_index:
                index = torch.arange(features.size(0), device=features.device)
                return features, targets, 1.0, index
            return features, targets, 1.0
            
        batch_size = features.size(0)
        
        # Sample lambda from Beta distribution
        if self.alpha > 0:
            lam = torch.distributions.Beta(self.alpha, self.alpha).sample().item()
        else:
            lam = 1.0
            
        # For stability, ensure lambda >= 0.5 (so first sample dominates)
        lam = max(lam, 1 - lam)
        
        # Random permutation for mixing pairs
        index = torch.randperm(batch_size, device=features.device)
        
        # Mix features and targets
        mixed_features = lam * features + (1 - lam) * features[index]
        mixed_targets = lam * targets + (1 - lam) * targets[index]
        
        if return_index:
            return mixed_features, mixed_targets, lam, index
        return mixed_features, mixed_targets, lam


class CutMixEmbedding(nn.Module):
    """
    CutMix variant for embeddings.
    
    Instead of spatial cutting, we cut along the feature dimension.
    This is less common but can work for embedding augmentation.
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        p: float = 0.5,
    ):
        super().__init__()
        self.alpha = alpha
        self.p = p
        
    def forward(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Apply CutMix to features"""
        if not self.training:
            return features, targets, 1.0
            
        if torch.rand(1).item() > self.p:
            return features, targets, 1.0
            
        batch_size, feat_dim = features.shape
        
        # Sample lambda
        if self.alpha > 0:
            lam = torch.distributions.Beta(self.alpha, self.alpha).sample().item()
        else:
            lam = 1.0
            
        lam = max(lam, 1 - lam)
        
        # Cut size
        cut_size = int(feat_dim * (1 - lam))
        cut_start = torch.randint(0, feat_dim - cut_size + 1, (1,)).item()
        cut_end = cut_start + cut_size
        
        # Random permutation
        index = torch.randperm(batch_size, device=features.device)
        
        # Cut and paste features
        mixed_features = features.clone()
        mixed_features[:, cut_start:cut_end] = features[index, cut_start:cut_end]
        
        # Actual lambda (based on cut proportion)
        actual_lam = 1 - cut_size / feat_dim
        
        # Mix targets
        mixed_targets = actual_lam * targets + (1 - actual_lam) * targets[index]
        
        return mixed_features, mixed_targets, actual_lam


class ManifoldMixUp(nn.Module):
    """
    Manifold MixUp wrapper.
    
    Designed to be applied at intermediate layers (embeddings).
    Same as MixUp but specifically for manifold/embedding space.
    
    Reference: https://arxiv.org/abs/1806.05236
    """
    
    def __init__(
        self,
        alpha: float = 0.4,
        p: float = 0.5,
    ):
        super().__init__()
        self.mixup = EmbeddingMixUp(alpha=alpha, p=p)
        
    def forward(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Apply Manifold MixUp"""
        return self.mixup(features, targets)
