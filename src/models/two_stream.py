"""
Multi-Head Model Architectures for CSIRO Biomass Prediction

Supports:
- Single-stream: Full image processing (with optional tiling)
- Two-stream: Left/right image patches processed separately
- Custom HuggingFace backbones with safetensors weights

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
from typing import Tuple, Optional, List, Dict


def _load_hf_safetensors(
    hf_repo: str,
    hf_filename: str,
    base_model_name: str,
) -> nn.Module:
    """
    Load a timm model with weights from HuggingFace safetensors.
    
    Args:
        hf_repo: HuggingFace repository (e.g., "vincent-espitalier/dino-v2-reg4-with-plantclef2024-weights")
        hf_filename: Safetensors filename in the repo
        base_model_name: Base timm model name (e.g., "vit_base_patch14_reg4_dinov2.lvd142m")
        
    Returns:
        Loaded model
    """
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    
    print(f"Loading HuggingFace backbone:")
    print(f"  Repo: {hf_repo}")
    print(f"  File: {hf_filename}")
    print(f"  Base model: {base_model_name}")
    
    # Download safetensors file
    ckpt_path = hf_hub_download(repo_id=hf_repo, filename=hf_filename)
    sd = load_file(ckpt_path)
    
    # Create base timm model (without pretrained weights)
    model = timm.create_model(base_model_name, pretrained=False, num_classes=0)
    
    # Clean state dict
    cleaned_sd = {}
    for k, v in sd.items():
        nk = k
        # Remove common prefixes
        if nk.startswith("module."):
            nk = nk[len("module."):]
        cleaned_sd[nk] = v
    
    # Drop classifier head keys (we use num_classes=0)
    drop_prefixes = ("head.", "fc.", "classifier.", "cls_head.", "linear_head.")
    cleaned_sd = {k: v for k, v in cleaned_sd.items() if not k.startswith(drop_prefixes)}
    
    # Load weights
    missing, unexpected = model.load_state_dict(cleaned_sd, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)} (expected for num_classes=0)")
    if unexpected:
        print(f"  Unexpected keys: {unexpected[:5]}...")
    
    print(f"✓ Loaded HuggingFace backbone successfully")
    return model


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
    
    Supports:
    - Standard timm backbones (backbone_name only)
    - HuggingFace safetensors backbones (hf_repo + hf_filename + backbone_name as base)
    """
    
    def __init__(
        self,
        backbone_name: str = "convnext_tiny",
        pretrained: bool = True,
        dropout: float = 0.3,
        hidden_ratio: float = 0.25,
        feature_multiplier: int = 1,  # 1 for single-stream, 2 for two-stream
        tile_size: Optional[int] = None,  # Resolution to resize tiles to (None = infer from backbone)
        feature_layers: Optional[List[int]] = None,  # Layers to extract (None = last only, e.g., [8,9,10,11])
        feature_pooling: str = "cls",  # "cls", "mean", or "cls_mean" (concat both)
        hf_repo: Optional[str] = None,  # HuggingFace repo for custom weights
        hf_filename: Optional[str] = None,  # Safetensors filename in HF repo
    ):
        super().__init__()
        
        # Store pooling config
        self.feature_pooling = feature_pooling
        if feature_pooling not in ("cls", "mean", "cls_mean"):
            raise ValueError(f"feature_pooling must be 'cls', 'mean', or 'cls_mean', got: {feature_pooling}")
        
        # Create backbone - either from HuggingFace or standard timm
        if hf_repo is not None and hf_filename is not None:
            # Load from HuggingFace safetensors
            self.backbone = _load_hf_safetensors(
                hf_repo=hf_repo,
                hf_filename=hf_filename,
                base_model_name=backbone_name,
            )
        else:
            # Standard timm model
            self.backbone = timm.create_model(
                backbone_name,
                pretrained=pretrained,
                num_classes=0,
            )
        
        # Store feature layer config
        self.feature_layers = feature_layers
        
        # Get feature dimension (per layer)
        self.feat_dim_per_layer = self.backbone.num_features
        
        # Pooling multiplier: cls_mean doubles the feature dim (concat CLS + mean)
        pooling_mult = 2 if feature_pooling == "cls_mean" else 1
        
        # Calculate total feature dimension based on layer concatenation
        if feature_layers is not None:
            n_layers = len(feature_layers)
            self.feat_dim = self.feat_dim_per_layer * n_layers * pooling_mult
            print(f"Multi-layer features: layers {feature_layers}, dim={self.feat_dim_per_layer}x{n_layers}x{pooling_mult}={self.feat_dim}, pooling={feature_pooling}")
        else:
            self.feat_dim = self.feat_dim_per_layer * pooling_mult
            print(f"Single layer features (last): dim={self.feat_dim}, pooling={feature_pooling}")
        
        self.combined_dim = self.feat_dim * feature_multiplier
        
        # Setup hooks for intermediate layer extraction if needed
        self._intermediate_features = {}
        self._hooks = []
        if feature_layers is not None:
            self._register_feature_hooks()
        
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
    
    def _get_backbone_blocks(self) -> Optional[nn.ModuleList]:
        """Get the sequential blocks/stages from the backbone."""
        # ViT, DINOv2, Swin
        if hasattr(self.backbone, "blocks"):
            return self.backbone.blocks
        # ConvNeXT
        if hasattr(self.backbone, "stages"):
            return self.backbone.stages
        # ResNet
        if hasattr(self.backbone, "layer4"):
            layers = nn.ModuleList()
            for name in ["layer1", "layer2", "layer3", "layer4"]:
                if hasattr(self.backbone, name):
                    layers.append(getattr(self.backbone, name))
            return layers
        return None
    
    def _register_feature_hooks(self):
        """Register forward hooks on specified layers to capture intermediate features."""
        blocks = self._get_backbone_blocks()
        if blocks is None:
            print("Warning: Could not find backbone blocks, using last layer only")
            self.feature_layers = None
            return
        
        n_blocks = len(blocks)
        print(f"Backbone has {n_blocks} blocks, registering hooks on layers: {self.feature_layers}")
        
        for layer_idx in self.feature_layers:
            if layer_idx < 0:
                # Support negative indexing
                layer_idx = n_blocks + layer_idx
            
            if 0 <= layer_idx < n_blocks:
                hook = blocks[layer_idx].register_forward_hook(
                    self._make_hook(layer_idx)
                )
                self._hooks.append(hook)
            else:
                print(f"Warning: Layer {layer_idx} out of range [0, {n_blocks})")
    
    def _make_hook(self, layer_idx: int):
        """Create a hook function for a specific layer."""
        def hook(module, input, output):
            # For ViT: output is [B, N, D] where N = num_patches + 1 (CLS token at index 0)
            # For ViT with registers: output is [B, N, D] where N = num_patches + 1 + num_registers
            if isinstance(output, torch.Tensor):
                if output.dim() == 3:
                    # ViT-like: [B, N, D]
                    if self.feature_pooling == "cls":
                        # Take CLS token only (first token)
                        self._intermediate_features[layer_idx] = output[:, 0]
                    elif self.feature_pooling == "mean":
                        # Mean pooling over all patch tokens (exclude CLS at index 0)
                        self._intermediate_features[layer_idx] = output[:, 1:].mean(dim=1)
                    else:  # cls_mean
                        # Concatenate CLS token + mean of patches
                        cls_feat = output[:, 0]  # [B, D]
                        mean_feat = output[:, 1:].mean(dim=1)  # [B, D]
                        self._intermediate_features[layer_idx] = torch.cat([cls_feat, mean_feat], dim=1)  # [B, 2D]
                elif output.dim() == 2:
                    # Already pooled [B, D] - duplicate for cls_mean to match expected dim
                    if self.feature_pooling == "cls_mean":
                        self._intermediate_features[layer_idx] = torch.cat([output, output], dim=1)
                    else:
                        self._intermediate_features[layer_idx] = output
                else:
                    # CNN-like: [B, C, H, W] -> global avg pool
                    pooled = output.mean(dim=[2, 3])
                    if self.feature_pooling == "cls_mean":
                        self._intermediate_features[layer_idx] = torch.cat([pooled, pooled], dim=1)
                    else:
                        self._intermediate_features[layer_idx] = pooled
            elif isinstance(output, tuple):
                # Some blocks return tuple, take first element
                out = output[0]
                if out.dim() == 3:
                    if self.feature_pooling == "cls":
                        self._intermediate_features[layer_idx] = out[:, 0]
                    elif self.feature_pooling == "mean":
                        self._intermediate_features[layer_idx] = out[:, 1:].mean(dim=1)
                    else:  # cls_mean
                        cls_feat = out[:, 0]
                        mean_feat = out[:, 1:].mean(dim=1)
                        self._intermediate_features[layer_idx] = torch.cat([cls_feat, mean_feat], dim=1)
                else:
                    pooled = out.mean(dim=[2, 3]) if out.dim() == 4 else out
                    if self.feature_pooling == "cls_mean":
                        self._intermediate_features[layer_idx] = torch.cat([pooled, pooled], dim=1)
                    else:
                        self._intermediate_features[layer_idx] = pooled
        return hook
    
    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from backbone, optionally from multiple layers.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Features [B, feat_dim] - concatenated if multiple layers
        """
        # Clear previous features
        self._intermediate_features.clear()
        
        # Forward pass (hooks will capture intermediate features)
        last_features = self.backbone(x)
        
        if self.feature_layers is None:
            # Use last layer only
            return last_features
        
        # Concatenate features from specified layers
        layer_features = []
        for layer_idx in self.feature_layers:
            if layer_idx < 0:
                layer_idx = len(self._get_backbone_blocks()) + layer_idx
            if layer_idx in self._intermediate_features:
                layer_features.append(self._intermediate_features[layer_idx])
            else:
                print(f"Warning: Layer {layer_idx} features not found, using last layer")
                layer_features.append(last_features)
        
        return torch.cat(layer_features, dim=1)
        
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
        features = self._extract_features(x)
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
                feat = self._extract_features(tile)
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
                feat = self._extract_features(tile)
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
        feat_left = self._extract_features(x_left)
        feat_right = self._extract_features(x_right)
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
                feat = self._extract_features(tile)
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
                feat = self._extract_features(tile)
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
