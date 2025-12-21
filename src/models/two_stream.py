"""
Multi-Head Model Architectures for CSIRO Biomass Prediction

Supports:
- Single-stream: Full image processing (with optional tiling)
- Two-stream: Left/right image patches processed separately

Architecture:
- Shared backbone (timm model, e.g., convnext_tiny or DINOv2 ViT)
- Optional tiling: Split image into grid and average features
- Three separate MLP heads for each target (Total, GDM, Green)
- Derived targets (Dead, Clover) are calculated from predictions
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Tuple, Optional, List


def _make_edges(length: int, parts: int) -> List[Tuple[int, int]]:
    """Create edge indices for tiling"""
    step = length // parts
    edges = []
    start = 0
    for _ in range(parts - 1):
        edges.append((start, start + step))
        start += step
    edges.append((start, length))
    return edges


class BaseMultiHead(nn.Module):
    """
    Base class for multi-head biomass prediction models.
    
    Predicts 3 primary targets: Dry_Total_g, GDM_g, Dry_Green_g
    Derived targets: Dry_Dead_g = Total - GDM, Dry_Clover_g = GDM - Green
    """
    
    def __init__(
        self,
        backbone_name: str = "convnext_tiny",
        pretrained: bool = True,
        dropout: float = 0.3,
        hidden_ratio: float = 0.25,
        feature_multiplier: int = 1,  # 1 for single-stream, 2 for two-stream
        tile_size: Optional[int] = None,  # Resolution to resize tiles to (None = infer from backbone)
    ):
        super().__init__()
        
        # Create backbone
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
        )
        
        # Get feature dimension
        self.feat_dim = self.backbone.num_features
        self.combined_dim = self.feat_dim * feature_multiplier
        
        # Input resolution: use config tile_size if provided, otherwise infer from backbone
        if tile_size is not None:
            self.input_res = tile_size
        else:
            self.input_res = self._infer_input_res()
        print(f"Model input_res (tile size): {self.input_res}")
        
        # Calculate hidden size
        hidden_size = max(8, int(self.combined_dim * hidden_ratio))
        
        # Create three separate heads for each target
        self.head_total = self._make_head(hidden_size, dropout)
        self.head_gdm = self._make_head(hidden_size, dropout)
        self.head_green = self._make_head(hidden_size, dropout)
        
        # Softplus for non-negative outputs
        self.softplus = nn.Softplus(beta=1.0)
        
    def _infer_input_res(self) -> int:
        """Infer input resolution from backbone"""
        if hasattr(self.backbone, "patch_embed") and hasattr(self.backbone.patch_embed, "img_size"):
            isz = self.backbone.patch_embed.img_size
            return int(isz if isinstance(isz, (int, float)) else isz[0])
        if hasattr(self.backbone, "img_size"):
            isz = self.backbone.img_size
            return int(isz if isinstance(isz, (int, float)) else isz[0])
        dc = getattr(self.backbone, "default_cfg", {}) or {}
        ins = dc.get("input_size", None)
        if ins:
            if isinstance(ins, (tuple, list)) and len(ins) >= 2:
                return int(ins[1])
            return int(ins if isinstance(ins, (int, float)) else 224)
        name = dc.get("architecture", "") or str(type(self.backbone))
        return 518 if "dinov2" in name.lower() else 224
        
    def _make_head(self, hidden_size: int, dropout: float) -> nn.Sequential:
        """Create an MLP head for one target"""
        return nn.Sequential(
            nn.Linear(self.combined_dim, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
    
    def freeze_backbone(self):
        """Freeze backbone parameters (Stage 1 training)"""
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters (Stage 2 fine-tuning)"""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def _predict_from_features(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict targets from combined features"""
        total = self.softplus(self.head_total(features))
        gdm = self.softplus(self.head_gdm(features))
        green = self.softplus(self.head_green(features))
        return total, gdm, green
    
    def predict_all_targets(self, *args) -> torch.Tensor:
        """
        Predict all 5 targets including derived ones.
        
        Returns:
            Tensor [B, 5] with [Green, Dead, Clover, GDM, Total]
        """
        total, gdm, green = self.forward(*args)
        
        # Calculate derived targets
        dead = torch.clamp(total - gdm, min=0)
        clover = torch.clamp(gdm - green, min=0)
        
        # Stack in order: [Green, Dead, Clover, GDM, Total]
        all_targets = torch.cat([green, dead, clover, gdm, total], dim=1)
        return all_targets


# =============================================================================
# SINGLE-STREAM MODELS (Full image processing)
# =============================================================================

class SingleStreamMultiHead(BaseMultiHead):
    """
    Single-Stream Multi-Head model.
    Processes the full image without splitting.
    """
    
    def __init__(self, backbone_name: str = "convnext_tiny", **kwargs):
        super().__init__(backbone_name, feature_multiplier=1, **kwargs)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Full image [B, C, H, W]
            
        Returns:
            Tuple of (total, gdm, green) predictions, each [B, 1]
        """
        features = self.backbone(x)
        return self._predict_from_features(features)


class SingleStreamTiled(BaseMultiHead):
    """
    Single-Stream model with tiled processing.
    Splits full image into a grid and averages features.
    """
    
    def __init__(
        self,
        backbone_name: str = "vit_base_patch14_dinov2",
        grid: Tuple[int, int] = (2, 2),
        **kwargs,
    ):
        super().__init__(backbone_name, feature_multiplier=1, **kwargs)
        self.grid = grid
        
    def _encode_tiles(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image by processing tiles and averaging"""
        B, C, H, W = x.shape
        r, c = self.grid
        rows = _make_edges(H, r)
        cols = _make_edges(W, c)
        
        feats = []
        for (rs, re) in rows:
            for (cs, ce) in cols:
                tile = x[:, :, rs:re, cs:ce]
                if tile.shape[-2:] != (self.input_res, self.input_res):
                    tile = F.interpolate(
                        tile,
                        size=(self.input_res, self.input_res),
                        mode="bilinear",
                        align_corners=False,
                    )
                feat = self.backbone(tile)
                feats.append(feat)
        
        # Average tile features
        feats = torch.stack(feats, dim=0).permute(1, 0, 2)  # [B, T, F]
        return feats.mean(dim=1)  # [B, F]
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self._encode_tiles(x)
        return self._predict_from_features(features)


class FiLM(nn.Module):
    """Feature-wise Linear Modulation layer"""
    
    def __init__(self, in_dim: int):
        super().__init__()
        hidden = max(32, in_dim // 2)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_dim * 2),
        )
        
    def forward(self, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gb = self.mlp(context)
        gamma, beta = torch.chunk(gb, 2, dim=1)
        return gamma, beta


class SingleStreamTiledFiLM(SingleStreamTiled):
    """
    Single-Stream Tiled model with FiLM modulation.
    """
    
    def __init__(self, backbone_name: str = "vit_base_patch14_dinov2", **kwargs):
        super().__init__(backbone_name, **kwargs)
        self.film = FiLM(self.feat_dim)
        
    def _tiles_backbone(self, x: torch.Tensor) -> torch.Tensor:
        """Get tile features without averaging"""
        B, C, H, W = x.shape
        r, c = self.grid
        rows = _make_edges(H, r)
        cols = _make_edges(W, c)
        
        feats = []
        for (rs, re) in rows:
            for (cs, ce) in cols:
                tile = x[:, :, rs:re, cs:ce]
                if tile.shape[-2:] != (self.input_res, self.input_res):
                    tile = F.interpolate(
                        tile,
                        size=(self.input_res, self.input_res),
                        mode="bilinear",
                        align_corners=False,
                    )
                feat = self.backbone(tile)
                feats.append(feat)
        
        return torch.stack(feats, dim=0).permute(1, 0, 2)  # [B, T, F]
    
    def _encode_with_film(self, x: torch.Tensor) -> torch.Tensor:
        """Encode with FiLM modulation"""
        tiles = self._tiles_backbone(x)  # [B, T, F]
        context = tiles.mean(dim=1)  # [B, F]
        gamma, beta = self.film(context)  # [B, F]
        
        # Apply FiLM
        tiles = tiles * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        return tiles.mean(dim=1)  # [B, F]
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self._encode_with_film(x)
        return self._predict_from_features(features)


# =============================================================================
# TWO-STREAM MODELS (Left/Right split processing)
# =============================================================================

class TwoStreamMultiHead(BaseMultiHead):
    """
    Two-Stream Multi-Head model for biomass prediction.
    Processes left and right image halves separately, then concatenates features.
    """
    
    def __init__(self, backbone_name: str = "convnext_tiny", **kwargs):
        super().__init__(backbone_name, feature_multiplier=2, **kwargs)
    
    def forward(
        self,
        x_left: torch.Tensor,
        x_right: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x_left: Left image patch [B, C, H, W]
            x_right: Right image patch [B, C, H, W]
            
        Returns:
            Tuple of (total, gdm, green) predictions, each [B, 1]
        """
        feat_left = self.backbone(x_left)
        feat_right = self.backbone(x_right)
        combined = torch.cat([feat_left, feat_right], dim=1)
        return self._predict_from_features(combined)


class TwoStreamTiled(BaseMultiHead):
    """
    Two-Stream model with tiled processing.
    Splits each patch into a grid and averages features.
    """
    
    def __init__(
        self,
        backbone_name: str = "vit_base_patch14_dinov2",
        grid: Tuple[int, int] = (2, 2),
        **kwargs,
    ):
        super().__init__(backbone_name, feature_multiplier=2, **kwargs)
        self.grid = grid
        
    def _encode_tiles(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image by processing tiles and averaging"""
        B, C, H, W = x.shape
        r, c = self.grid
        rows = _make_edges(H, r)
        cols = _make_edges(W, c)
        
        feats = []
        for (rs, re) in rows:
            for (cs, ce) in cols:
                tile = x[:, :, rs:re, cs:ce]
                if tile.shape[-2:] != (self.input_res, self.input_res):
                    tile = F.interpolate(
                        tile,
                        size=(self.input_res, self.input_res),
                        mode="bilinear",
                        align_corners=False,
                    )
                feat = self.backbone(tile)
                feats.append(feat)
        
        feats = torch.stack(feats, dim=0).permute(1, 0, 2)  # [B, T, F]
        return feats.mean(dim=1)  # [B, F]
    
    def forward(
        self,
        x_left: torch.Tensor,
        x_right: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat_left = self._encode_tiles(x_left)
        feat_right = self._encode_tiles(x_right)
        combined = torch.cat([feat_left, feat_right], dim=1)
        return self._predict_from_features(combined)


class TwoStreamTiledFiLM(TwoStreamTiled):
    """
    Two-Stream Tiled model with FiLM modulation.
    Uses Feature-wise Linear Modulation for better feature fusion.
    """
    
    def __init__(self, backbone_name: str = "vit_base_patch14_dinov2", **kwargs):
        super().__init__(backbone_name, **kwargs)
        self.film_left = FiLM(self.feat_dim)
        self.film_right = FiLM(self.feat_dim)
        
    def _tiles_backbone(self, x: torch.Tensor) -> torch.Tensor:
        """Get tile features without averaging"""
        B, C, H, W = x.shape
        r, c = self.grid
        rows = _make_edges(H, r)
        cols = _make_edges(W, c)
        
        feats = []
        for (rs, re) in rows:
            for (cs, ce) in cols:
                tile = x[:, :, rs:re, cs:ce]
                if tile.shape[-2:] != (self.input_res, self.input_res):
                    tile = F.interpolate(
                        tile,
                        size=(self.input_res, self.input_res),
                        mode="bilinear",
                        align_corners=False,
                    )
                feat = self.backbone(tile)
                feats.append(feat)
        
        return torch.stack(feats, dim=0).permute(1, 0, 2)  # [B, T, F]
    
    def _encode_stream(self, x: torch.Tensor, film: FiLM) -> torch.Tensor:
        """Encode with FiLM modulation"""
        tiles = self._tiles_backbone(x)  # [B, T, F]
        context = tiles.mean(dim=1)  # [B, F]
        gamma, beta = film(context)  # [B, F]
        
        # Apply FiLM
        tiles = tiles * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        return tiles.mean(dim=1)  # [B, F]
    
    def forward(
        self,
        x_left: torch.Tensor,
        x_right: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat_left = self._encode_stream(x_left, self.film_left)
        feat_right = self._encode_stream(x_right, self.film_right)
        combined = torch.cat([feat_left, feat_right], dim=1)
        return self._predict_from_features(combined)


# =============================================================================
# MODEL FACTORY
# =============================================================================

# Model registry for easy lookup
MODEL_REGISTRY = {
    # Single-stream models
    "single_stream": SingleStreamMultiHead,
    "single_stream_tiled": SingleStreamTiled,
    "single_stream_tiled_film": SingleStreamTiledFiLM,
    # Two-stream models
    "two_stream": TwoStreamMultiHead,
    "two_stream_tiled": TwoStreamTiled,
    "two_stream_tiled_film": TwoStreamTiledFiLM,
}


def build_model(
    model_type: str = "two_stream",
    backbone_name: str = "convnext_tiny",
    pretrained: bool = True,
    grid: Optional[Tuple[int, int]] = None,
    tile_size: Optional[int] = None,
    **kwargs,
) -> nn.Module:
    """
    Factory function to build models.
    
    Args:
        model_type: One of:
            - 'single_stream': Full image, no tiling
            - 'single_stream_tiled': Full image with tiling
            - 'single_stream_tiled_film': Full image with tiling and FiLM
            - 'two_stream': Left/right split, no tiling
            - 'two_stream_tiled': Left/right split with tiling
            - 'two_stream_tiled_film': Left/right split with tiling and FiLM
        backbone_name: Name of timm backbone
        pretrained: Whether to use pretrained weights
        grid: Tuple (rows, cols) for tiling (only for tiled models)
        tile_size: Resolution to resize tiles to before backbone (None = infer from backbone)
        **kwargs: Additional arguments (dropout, hidden_ratio)
        
    Returns:
        Model instance
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    
    model_class = MODEL_REGISTRY[model_type]
    
    # Build common kwargs
    common_kwargs = {"pretrained": pretrained, "tile_size": tile_size, **kwargs}
    
    # Add grid parameter for tiled models
    if "tiled" in model_type and grid is not None:
        return model_class(backbone_name, grid=grid, **common_kwargs)
    else:
        return model_class(backbone_name, **common_kwargs)


def get_stream_mode(model_type: str) -> str:
    """Get stream mode from model type"""
    if model_type.startswith("single_stream"):
        return "single_stream"
    else:
        return "two_stream"
