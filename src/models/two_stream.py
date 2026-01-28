"""
Multi-Head Model Architectures for CSIRO Biomass Prediction

Supports:
- Single-stream: Full image processing (with optional tiling)
- Two-stream: Left/right image patches processed separately
- Custom HuggingFace backbones with safetensors weights
- Optional semantic features from SigLIP for enhanced predictions

Architecture:
- Shared backbone (timm model, e.g., convnext_tiny or DINOv2 ViT)
- Optional tiling: Split image into grid and average features
- Optional semantic branch (SigLIP) for concept similarity features
- Three separate MLP heads for each target (Total, GDM, Green)
- Derived targets (Dead, Clover) are calculated from predictions

Feature Pooling Options:
- cls: Use CLS token only
- mean: Mean pooling over patch tokens
- gem: GeM (Generalized Mean) pooling over patch tokens
- cls_mean: Concatenate CLS + mean pooled patches
- cls_gem: Concatenate CLS + GeM pooled patches
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Tuple, Optional, List, Dict, Union, Any


class TileEmbedFusion(nn.Module):
    """
    Fuse 8 tile embeddings (4 left + 4 right) with learned half+tile embeddings
    and a small MambaFusion over the tile-token sequence.

    Input:
      feats_left:  [B, 4, F]
      feats_right: [B, 4, F]

    Output:
      pooled_left:  [B, F]
      pooled_right: [B, F]
    """
    def __init__(
        self,
        dim: int,
        num_tiles_per_half: int = 4,
        num_layers: int = 2,
        kernel_size: int = 5,
        dropout: float = 0.1,
        pool: str = "mean",  # keep simple; can add "attn" later if needed
    ):
        super().__init__()
        self.dim = dim
        self.num_tiles = num_tiles_per_half
        self.pool = pool

        # Identity embeddings (learned)
        self.half_emb = nn.Embedding(2, dim)         # 0=left, 1=right
        self.tile_emb = nn.Embedding(self.num_tiles, dim)  # tile index 0..3

        # Sequence fusion over 8 tile tokens
        self.fusion = MambaFusion(
            dim=dim,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        # (Optional) learned pooling could be added later.
        # Keeping it mean makes this a "tiny fusion" that tends to generalize better.

    def forward(self, feats_left: torch.Tensor, feats_right: torch.Tensor):
        # feats_left/right: [B, 4, F]
        if feats_left.dim() != 3 or feats_right.dim() != 3:
            raise ValueError("feats_left and feats_right must be [B, T, F] tensors")

        B, T, F = feats_left.shape
        if T != self.num_tiles or feats_right.shape[1] != self.num_tiles:
            raise ValueError(f"Expected {self.num_tiles} tiles per half, got {T} and {feats_right.shape[1]}")
        if F != self.dim:
            raise ValueError(f"Feature dim mismatch: expected {self.dim}, got {F}")

        device = feats_left.device

        # Tile ids: [0,1,2,3]
        tile_ids = torch.arange(self.num_tiles, device=device, dtype=torch.long)  # [T]
        e_tile = self.tile_emb(tile_ids)[None, :, :]  # [1, T, F]

        # Half embeddings
        e_left = self.half_emb(torch.tensor([0], device=device, dtype=torch.long))[:, None, :]   # [1, 1, F]
        e_right = self.half_emb(torch.tensor([1], device=device, dtype=torch.long))[:, None, :]  # [1, 1, F]

        # Add identity embeddings
        x_left = feats_left + e_tile + e_left      # [B, T, F]
        x_right = feats_right + e_tile + e_right   # [B, T, F]

        # Concatenate into 8-token sequence
        x = torch.cat([x_left, x_right], dim=1)    # [B, 2T, F] = [B, 8, F]

        # Fuse across tiles (and across halves) with Mamba
        y = self.fusion(x)                         # [B, 8, F]

        # Pool back to two half vectors
        y_left = y[:, :T, :]
        y_right = y[:, T:, :]

        if self.pool == "mean":
            pooled_left = y_left.mean(dim=1)       # [B, F]
            pooled_right = y_right.mean(dim=1)     # [B, F]
        else:
            # Keep simple; you can add a learned pooling later if desired
            pooled_left = y_left.mean(dim=1)
            pooled_right = y_right.mean(dim=1)

        return pooled_left, pooled_right



class GeMPooling(nn.Module):
    """
    Generalized Mean (GeM) Pooling.
    
    GeM(x) = (1/N * Σ x_i^p)^(1/p)
    
    When p=1 → mean pooling
    When p→∞ → max pooling
    Typical p=3 highlights stronger activations while considering all patches.
    
    Reference: https://arxiv.org/abs/1711.02512
    """
    
    def __init__(self, p: float = 3.0, eps: float = 1e-6, learnable: bool = True):
        """
        Args:
            p: Initial power parameter (default 3.0)
            eps: Small value for numerical stability
            learnable: If True, p is a learnable parameter
        """
        super().__init__()
        self.eps = eps
        if learnable:
            self.p = nn.Parameter(torch.tensor(p))
        else:
            self.register_buffer("p", torch.tensor(p))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply GeM pooling.
        
        Args:
            x: Input tensor [B, N, D] where N is number of tokens
            
        Returns:
            Pooled tensor [B, D]
        """
        # Clamp to avoid numerical issues with negative values
        x_clamped = x.clamp(min=self.eps)
        # GeM: (mean(x^p))^(1/p)
        return x_clamped.pow(self.p).mean(dim=1).pow(1.0 / self.p)
    
    def extra_repr(self) -> str:
        return f"p={self.p.item():.4f}, eps={self.eps}"


def gem_pool(x: torch.Tensor, p: float = 3.0, eps: float = 1e-6) -> torch.Tensor:
    """
    Functional GeM pooling (non-learnable).
    
    Args:
        x: Input tensor [B, N, D]
        p: Power parameter
        eps: Epsilon for numerical stability
        
    Returns:
        Pooled tensor [B, D]
    """
    x_clamped = x.clamp(min=eps)
    return x_clamped.pow(p).mean(dim=1).pow(1.0 / p)


class AttnPool(nn.Module):
    """
    Attention-based tile pooling.
    
    Learns which tiles are most important via a small MLP that predicts
    per-tile scores. Softmax converts scores to weights, then tiles are
    weighted-averaged. This preserves important tile signal (like max)
    while being smooth and learnable.
    """
    
    def __init__(self, dim: int, hidden: int = 256, dropout: float = 0.0):
        """
        Args:
            dim: Feature dimension of each tile
            hidden: Hidden dimension for attention MLP
            dropout: Dropout in attention MLP
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, tiles: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply attention pooling.
        
        Args:
            tiles: Tile features [B, T, F]
            
        Returns:
            Tuple of:
                - pooled: Weighted average of tiles [B, F]
                - weights: Attention weights [B, T] (for visualization/debugging)
        """
        scores = self.net(tiles).squeeze(-1)      # [B, T]
        weights = torch.softmax(scores, dim=1)    # [B, T]
        pooled = (tiles * weights.unsqueeze(-1)).sum(dim=1)  # [B, F]
        return pooled, weights


class LocalMambaBlock(nn.Module):
    """
    Lightweight Mamba-style block (Gated CNN) for token fusion.
    
    This allows tokens from different tiles/halves to interact via:
    1. Gating: sigmoid(gate(x)) * x - learns which tokens to emphasize
    2. Depthwise 1D conv: local mixing across the sequence dimension
    3. Residual connection: preserves original information
    
    Key insight: When you concatenate tokens from LEFT and RIGHT image halves
    (or from different tiles), the depthwise convolution allows cross-image
    token interaction with linear O(n) complexity (vs O(n²) for attention).
    
    Reference: Inspired by Mamba (S4/SSM) but uses simpler gated CNN.
    """
    
    def __init__(self, dim: int, kernel_size: int = 5, dropout: float = 0.0):
        """
        Args:
            dim: Token feature dimension
            kernel_size: Kernel size for depthwise conv (controls local receptive field)
            dropout: Dropout rate after projection
        """
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        # Depthwise conv mixes tokens locally across sequence dimension
        self.dwconv = nn.Conv1d(
            dim, dim, 
            kernel_size=kernel_size, 
            padding=kernel_size // 2, 
            groups=dim  # Depthwise = each channel processed independently
        )
        self.gate = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Mamba-style token mixing.
        
        Args:
            x: Token features [B, N, D] where N is sequence length
            
        Returns:
            Mixed tokens [B, N, D]
        """
        # x: (Batch, Tokens, Dim)
        shortcut = x
        x = self.norm(x)
        
        # Gating mechanism - learns which tokens to emphasize
        g = torch.sigmoid(self.gate(x))
        x = x * g
        
        # Spatial mixing via 1D Conv (requires transpose: B,N,D -> B,D,N)
        x = x.transpose(1, 2)  # -> (B, D, N)
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # -> (B, N, D)
        
        # Projection and dropout
        x = self.proj(x)
        x = self.drop(x)
        
        # Residual connection
        return shortcut + x


class MambaFusion(nn.Module):
    """
    Multi-layer Mamba fusion module.
    
    Stacks multiple LocalMambaBlocks for deeper token interaction.
    """
    
    def __init__(
        self, 
        dim: int, 
        num_layers: int = 2, 
        kernel_size: int = 5, 
        dropout: float = 0.1
    ):
        """
        Args:
            dim: Token feature dimension
            num_layers: Number of LocalMambaBlocks to stack
            kernel_size: Kernel size for depthwise conv
            dropout: Dropout rate
        """
        super().__init__()
        self.layers = nn.Sequential(*[
            LocalMambaBlock(dim, kernel_size=kernel_size, dropout=dropout)
            for _ in range(num_layers)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-layer Mamba fusion.
        
        Args:
            x: Token features [B, N, D]
            
        Returns:
            Fused tokens [B, N, D]
        """
        return self.layers(x)


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
    - Optional semantic features from SigLIP for enhanced predictions
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
        feature_pooling: str = "cls",  # "cls", "mean", "gem", "cls_mean", "cls_gem"
        gem_p: float = 3.0,  # GeM pooling power parameter (higher = closer to max pooling)
        gem_learnable: bool = True,  # Whether GeM p parameter is learnable
        hf_repo: Optional[str] = None,  # HuggingFace repo for custom weights
        hf_filename: Optional[str] = None,  # Safetensors filename in HF repo
        # Semantic features configuration
        use_semantic: bool = False,  # Whether to use semantic features from SigLIP
        semantic_model_path: Optional[str] = None,  # HuggingFace path for SigLIP model
        semantic_freeze: bool = True,  # Whether to freeze SigLIP weights
        semantic_multiplier: int = 1,  # 1 for single-stream, 2 for two-stream (avg or concat)
        # Auxiliary prediction heads (train-only supervision)
        aux_tasks: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        super().__init__()

        self.aux_tasks: Dict[str, Dict[str, Any]] = aux_tasks or {}
        self.aux_task_types: Dict[str, str] = {}
        self.aux_heads = nn.ModuleDict()
        
        # Store pooling config
        self.feature_pooling = feature_pooling
        valid_pooling = ("cls", "mean", "gem", "cls_mean", "cls_gem")
        if feature_pooling not in valid_pooling:
            raise ValueError(f"feature_pooling must be one of {valid_pooling}, got: {feature_pooling}")
        
        # Store GeM parameters
        self.gem_p = gem_p
        self.gem_learnable = gem_learnable
        
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

       ##########################################################################################
        ##########################################################################################
        ##### Custom Dinov2 backbone from HuggingFace #####

        # ckpt_path = "/home/ib/CSIRO-Image2BiomassPrediction/notebooks/dinov2_grassclover_out_hfv0/best_dinov2_grassclover.pth"
        # ckpt = torch.load(ckpt_path, map_location="cpu")
        # sd = ckpt["model"]
        # bb_sd = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}

        # timm_sd = convert_hf_dinov2_to_timm_vit(bb_sd, num_layers=24)

        # m = timm.create_model("vit_large_patch14_dinov2.lvd142m", pretrained=False, num_classes=0, global_pool='')
        # self.backbone = m
        # missing, unexpected = m.load_state_dict(timm_sd, strict=False)
        # print("missing:", len(missing))
        # print("missing:", missing[:10])
        # print("unexpected:", unexpected[:10])

        ##### Custom Dinov2 backbone from HuggingFace #####
        ##########################################################################################
        ##########################################################################################
        
        # Store feature layer config
        # If feature_layers is None but cls_mean/cls_gem pooling is requested,
        # we need to use hooks to extract patch tokens. Auto-select last block.
        if feature_layers is None and feature_pooling in ("cls_mean", "cls_gem"):
            # Get number of blocks to auto-select last one
            blocks = None
            if hasattr(self.backbone, "blocks"):
                blocks = self.backbone.blocks
            elif hasattr(self.backbone, "stages"):
                blocks = self.backbone.stages
            
            if blocks is not None:
                last_block_idx = len(blocks) - 1
                feature_layers = [last_block_idx]
                print(f"Auto-selecting last block [{last_block_idx}] for {feature_pooling} pooling")
            else:
                print(f"Warning: Cannot find backbone blocks for {feature_pooling} pooling, falling back to backbone default")
        
        self.feature_layers = feature_layers
        
        # Get feature dimension (per layer)
        self.feat_dim_per_layer = self.backbone.num_features
        
        # Pooling multiplier: cls_mean/cls_gem double the feature dim (concat CLS + pooled patches)
        pooling_mult = 2 if feature_pooling in ("cls_mean", "cls_gem") else 1
        
        # Calculate total feature dimension based on layer concatenation
        if feature_layers is not None:
            n_layers = len(feature_layers)
            self.feat_dim = self.feat_dim_per_layer * n_layers * pooling_mult
            print(f"Multi-layer features: layers {feature_layers}, dim={self.feat_dim_per_layer}x{n_layers}x{pooling_mult}={self.feat_dim}, pooling={feature_pooling}")
        else:
            # When feature_layers is None, we use the backbone's pooled output directly
            # This is typically just CLS token or global avg pool, NOT cls_mean/cls_gem
            # So we don't apply pooling_mult here - the backbone already pooled
            self.feat_dim = self.feat_dim_per_layer
            print(f"Single layer features (last): dim={self.feat_dim}, pooling=backbone_default")
        
        self.combined_dim = self.feat_dim * feature_multiplier
        
        # Setup semantic branch if enabled
        self.use_semantic = use_semantic
        self.semantic_dim = 0
        self.semantic_branch = None
        
        if use_semantic:
            from src.models.semantic_features import SemanticBranch
            model_path = semantic_model_path or "google/siglip-so400m-patch14-384"
            self.semantic_branch = SemanticBranch(
                model_path=model_path,
                freeze=semantic_freeze,
                input_size=384,
                # During inference (pretrained=False), skip loading from HuggingFace
                # Weights will come from checkpoint instead
                load_pretrained=pretrained,
            )
            # Semantic features are averaged for left/right, so multiplier is 1
            self.semantic_dim = self.semantic_branch.output_dim * semantic_multiplier
            self.combined_dim += self.semantic_dim
            print(f"Semantic features enabled: model={model_path}, dim={self.semantic_dim}, "
                  f"total combined_dim={self.combined_dim}")
        
        # Create LayerNorm for patch tokens before pooling
        # This normalizes magnitudes so pooling isn't dominated by high-magnitude patches
        self.patch_norm = nn.LayerNorm(self.feat_dim_per_layer)
        
        # Create GeM pooling layer if needed (for gem or cls_gem pooling)
        if feature_pooling in ("gem", "cls_gem"):
            self.gem_pool = GeMPooling(p=gem_p, learnable=gem_learnable)
            print(f"GeM pooling enabled: p={gem_p}, learnable={gem_learnable}, with LayerNorm")
        else:
            self.gem_pool = None
        
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
        
        # Calculate hidden size based on final combined_dim (including semantic)
        hidden_size = max(8, int(self.combined_dim * hidden_ratio))
        
        # Create three separate heads for each target
        self.head_total = self._make_head(hidden_size, dropout)
        self.head_gdm = self._make_head(hidden_size, dropout)
        self.head_green = self._make_head(hidden_size, dropout)
        self._rebuild_aux_heads(hidden_size, dropout)

        self._rebuild_aux_heads(hidden_size, dropout)
        
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
    
    def _clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            try:
                hook.remove()
            except Exception:
                # Hook may be invalid if backbone was replaced/deleted
                pass
        self._hooks = []
        self._intermediate_features = {}
    
    def reregister_hooks(self):
        """
        Clear and re-register feature hooks on the current backbone.
        
        This is needed after sharing backbones during inference, because
        the original hooks point to the first model's _intermediate_features dict.
        Call this after assigning a shared backbone to ensure hooks store
        features to this model's dict.
        """
        if self.feature_layers is None:
            return
        
        self._clear_hooks()
        self._register_feature_hooks(verbose=False)
    
    def _register_feature_hooks(self, verbose: bool = True):
        """Register forward hooks on specified layers to capture intermediate features."""
        blocks = self._get_backbone_blocks()
        if blocks is None:
            if verbose:
                print("Warning: Could not find backbone blocks, using last layer only")
            self.feature_layers = None
            return
        
        n_blocks = len(blocks)
        if verbose:
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
    
    def _pool_patches(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Pool patch tokens using configured pooling method.
        
        For GeM pooling, applies LayerNorm before pooling to prevent 
        magnitude differences from dominating the aggregation.
        
        Args:
            patches: Patch tokens [B, N, D] (without CLS token)
            
        Returns:
            Pooled features [B, D]
        """
        if self.feature_pooling in ("mean", "cls_mean"):
            return patches.mean(dim=1)
        elif self.feature_pooling in ("gem", "cls_gem"):
            # Apply LayerNorm before GeM to normalize magnitudes
            # This ensures pooling weights are based on relative activation patterns,
            # not raw magnitude differences
            
            # patches = self.patch_norm(patches)
            
            # Use GeM pooling
            if self.gem_pool is not None:
                return self.gem_pool(patches)
            else:
                # Fallback to functional GeM
                return gem_pool(patches, p=self.gem_p)
        else:
            # Default to mean
            return patches.mean(dim=1)
    
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
                    elif self.feature_pooling in ("mean", "gem"):
                        # Pooling over all patch tokens (exclude CLS at index 0)
                        patches = output[:, 1:]
                        self._intermediate_features[layer_idx] = self._pool_patches(patches)
                    else:  # cls_mean or cls_gem
                        # Concatenate CLS token + pooled patches
                        cls_feat = output[:, 0]  # [B, D]
                        patches = output[:, 1:]  # [B, N-1, D]
                        pooled_feat = self._pool_patches(patches)  # [B, D]
                        self._intermediate_features[layer_idx] = torch.cat([cls_feat, pooled_feat], dim=1)  # [B, 2D]
                elif output.dim() == 2:
                    # Already pooled [B, D] - duplicate for cls_* to match expected dim
                    if self.feature_pooling in ("cls_mean", "cls_gem"):
                        self._intermediate_features[layer_idx] = torch.cat([output, output], dim=1)
                    else:
                        self._intermediate_features[layer_idx] = output
                else:
                    # CNN-like: [B, C, H, W] -> global avg pool
                    pooled = output.mean(dim=[2, 3])
                    if self.feature_pooling in ("cls_mean", "cls_gem"):
                        self._intermediate_features[layer_idx] = torch.cat([pooled, pooled], dim=1)
                    else:
                        self._intermediate_features[layer_idx] = pooled
            elif isinstance(output, tuple):
                # Some blocks return tuple, take first element
                out = output[0]
                if out.dim() == 3:
                    if self.feature_pooling == "cls":
                        self._intermediate_features[layer_idx] = out[:, 0]
                    elif self.feature_pooling in ("mean", "gem"):
                        patches = out[:, 1:]
                        self._intermediate_features[layer_idx] = self._pool_patches(patches)
                    else:  # cls_mean or cls_gem
                        cls_feat = out[:, 0]
                        patches = out[:, 1:]
                        pooled_feat = self._pool_patches(patches)
                        self._intermediate_features[layer_idx] = torch.cat([cls_feat, pooled_feat], dim=1)
                else:
                    pooled = out.mean(dim=[2, 3]) if out.dim() == 4 else out
                    if self.feature_pooling in ("cls_mean", "cls_gem"):
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
        
    def _make_head(self, hidden_size: int, dropout: float, out_dim: int = 1) -> nn.Sequential:
        """Create an MLP head."""
        return nn.Sequential(
            nn.Linear(self.combined_dim, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, out_dim),
        )

    def _rebuild_aux_heads(self, hidden_size: int, dropout: float):
        if not self.aux_tasks:
            self.aux_heads = nn.ModuleDict()
            self.aux_task_types = {}
            return

        aux_heads = nn.ModuleDict()
        aux_types: Dict[str, str] = {}
        for name, cfg in self.aux_tasks.items():
            t = str(cfg.get("type", "regression"))
            out_dim = int(cfg.get("out_dim", 1))
            aux_types[name] = t
            aux_heads[name] = self._make_head(hidden_size, dropout, out_dim=out_dim)

        self.aux_heads = aux_heads
        self.aux_task_types = aux_types
    
    def freeze_backbone(self):
        """Freeze backbone parameters (Stage 1 training)"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        # Semantic branch is typically frozen, but ensure it stays frozen
        if self.semantic_branch is not None:
            for param in self.semantic_branch.parameters():
                param.requires_grad = False
            
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters (Stage 2 fine-tuning)"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        # Note: Semantic branch typically stays frozen even during fine-tuning
    
    def unfreeze_last_n_blocks(self, n: int) -> int:
        """
        Unfreeze only the last N blocks of the backbone.
        
        This enables gradual/partial unfreezing for more stable fine-tuning.
        Earlier blocks learn low-level features that often don't need task-specific tuning.
        Later blocks learn higher-level features that may benefit from fine-tuning.
        
        Args:
            n: Number of blocks to unfreeze from the end (e.g., 6 for last 6 blocks)
        
        Returns:
            Number of blocks actually unfrozen (may be less than n if backbone has fewer blocks)
            
        Example:
            For ViT-Large (24 blocks), unfreeze_last_n_blocks(6) unfreezes blocks 18-23
        """
        blocks = self._get_backbone_blocks()
        
        if blocks is None:
            # Backbone doesn't have clear block structure, fall back to full unfreeze
            print(f"Warning: Backbone has no recognizable block structure, unfreezing all parameters")
            self.unfreeze_backbone()
            return -1  # Signal that we couldn't do partial unfreezing
        
        total_blocks = len(blocks)
        n_to_unfreeze = min(n, total_blocks)
        start_idx = total_blocks - n_to_unfreeze
        
        # Keep first (total_blocks - n) frozen, unfreeze last n
        unfrozen_params = 0
        for idx, block in enumerate(blocks):
            if idx >= start_idx:
                for param in block.parameters():
                    param.requires_grad = True
                    unfrozen_params += param.numel()
        
        # Also unfreeze the final norm/head projection if they exist
        if hasattr(self.backbone, "norm"):
            for param in self.backbone.norm.parameters():
                param.requires_grad = True
        if hasattr(self.backbone, "fc_norm"):
            for param in self.backbone.fc_norm.parameters():
                param.requires_grad = True
        
        print(f"Partially unfroze backbone: last {n_to_unfreeze}/{total_blocks} blocks "
              f"(blocks {start_idx}-{total_blocks-1}), ~{unfrozen_params:,} parameters")
        
        return n_to_unfreeze
    
    def get_backbone_block_count(self) -> int:
        """Return the number of blocks in the backbone, or -1 if unknown."""
        blocks = self._get_backbone_blocks()
        return len(blocks) if blocks is not None else -1
    
    def _predict_from_features(
        self,
        features: torch.Tensor,
        semantic_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict targets from combined features.
        
        Args:
            features: Visual features from backbone [B, feat_dim]
            semantic_features: Optional semantic features [B, semantic_dim]
            
        Returns:
            Tuple of (total, gdm, green) predictions
        """
        # Concatenate semantic features if provided
        if semantic_features is not None and self.use_semantic:
            features = torch.cat([features, semantic_features], dim=1)
        
        total = self.softplus(self.head_total(features))
        gdm = self.softplus(self.head_gdm(features))
        green = self.softplus(self.head_green(features))
        return total, gdm, green
    
    def predict_from_features(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Public method to predict from pre-extracted features.
        Used for embedding-level MixUp augmentation.
        
        Args:
            features: Combined features [B, combined_dim]
            
        Returns:
            Tuple of (total, gdm, green) predictions
        """
        return self._predict_from_features(features)

    def predict_aux_from_features(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict auxiliary targets from pre-extracted features.

        Returns:
            Dict mapping aux task name -> prediction tensor.
        """
        if not self.aux_heads:
            return {}
        return {name: head(features) for name, head in self.aux_heads.items()}
    
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
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features without prediction (for MixUp).
        
        Args:
            x: Full image [B, C, H, W]
            
        Returns:
            Features [B, combined_dim]
        """
        return self._extract_features(x)
    
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
    Splits full image into a grid and aggregates features.
    
    Tile pooling options:
    - "mean": Average tile features (default)
    - "max": Max pool tile features  
    - "mean_max": Concatenate mean + max (doubles feature dim)
    - "attn": Attention-weighted average (learnable, smooth but highlights important tiles)
    - "mamba": Mamba-style fusion allowing cross-tile token interaction before pooling
    """
    
    def __init__(
        self,
        backbone_name: str = "vit_base_patch14_dinov2",
        grid: Tuple[int, int] = (2, 2),
        tile_pooling: str = "mean",
        attn_hidden: int = 256,
        attn_dropout: float = 0.0,
        # Mamba fusion settings (only used when tile_pooling="mamba")
        mamba_layers: int = 2,
        mamba_kernel_size: int = 5,
        mamba_dropout: float = 0.1,
        **kwargs,
    ):
        # Tile pooling affects feature dimension
        tile_multiplier = 2 if tile_pooling == "mean_max" else 1
        super().__init__(backbone_name, feature_multiplier=tile_multiplier, **kwargs)
        self.grid = grid
        self.tile_pooling = tile_pooling
        
        # Create attention pooling module if needed
        if tile_pooling == "attn":
            self.attn_pool = AttnPool(
                dim=self.feat_dim,
                hidden=attn_hidden,
                dropout=attn_dropout,
            )
            print(f"Attention tile pooling enabled: hidden={attn_hidden}, dropout={attn_dropout}")
        
        # Create Mamba fusion module if needed
        if tile_pooling == "mamba":
            self.mamba_fusion = MambaFusion(
                dim=self.feat_dim,
                num_layers=mamba_layers,
                kernel_size=mamba_kernel_size,
                dropout=mamba_dropout,
            )
            print(f"Mamba tile fusion enabled: layers={mamba_layers}, kernel={mamba_kernel_size}, dropout={mamba_dropout}")
        
    def _encode_tiles(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image by processing tiles and aggregating with configured pooling"""
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
        
        # Aggregate tile features with configured pooling
        feats = torch.stack(feats, dim=0).permute(1, 0, 2)  # [B, T, F]
        return self._pool_tiles(feats)
    
    def _pool_tiles(self, feats: torch.Tensor) -> torch.Tensor:
        """
        Pool tile features using configured method.
        
        Args:
            feats: Tile features [B, T, F] where T is number of tiles
            
        Returns:
            Pooled features [B, F] or [B, 2*F] for mean_max
        """
        if self.tile_pooling == "mean":
            return feats.mean(dim=1)
        elif self.tile_pooling == "max":
            return feats.max(dim=1)[0]
        elif self.tile_pooling == "mean_max":
            # Concatenate mean and max for richer representation
            # Max preserves the most important tile, mean gives context
            mean_feat = feats.mean(dim=1)
            max_feat = feats.max(dim=1)[0]
            return torch.cat([mean_feat, max_feat], dim=1)
        elif self.tile_pooling == "attn":
            # Attention-weighted pooling (learnable)
            pooled, _ = self.attn_pool(feats)
            return pooled
        elif self.tile_pooling == "mamba":
            # Mamba-style fusion: allow tiles to interact via gated CNN
            # before pooling. This enables cross-tile information flow.
            # feats: [B, T, F] where T is number of tiles (e.g., 4 for 2x2 grid)
            fused = self.mamba_fusion(feats)  # [B, T, F] with cross-tile interaction
            return fused.mean(dim=1)  # Pool after fusion
        else:
            # Default to mean
            return feats.mean(dim=1)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without prediction (for MixUp)"""
        return self._encode_tiles(x)
    
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
        return self._pool_tiles(tiles)  # [B, F] or [B, 2*F]
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without prediction (for MixUp)"""
        return self._encode_with_film(x)
    
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
    
    def extract_features(
        self,
        x_left: torch.Tensor,
        x_right: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract concatenated features without prediction (for MixUp).
        
        Args:
            x_left: Left image patch [B, C, H, W]
            x_right: Right image patch [B, C, H, W]
            
        Returns:
            Combined features [B, combined_dim]
        """
        feat_left = self._extract_features(x_left)
        feat_right = self._extract_features(x_right)
        return torch.cat([feat_left, feat_right], dim=1)
    
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


from typing import Tuple, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F

# Assumes these exist in your codebase (as in your current project):
# - BaseMultiHead
# - AttnPool
# - MambaFusion
# - TileEmbedFusion
# - _make_edges


class TwoStreamTiled(BaseMultiHead):
    """
    Two-Stream model with tiled processing.
    Splits each patch into a grid and aggregates features.
    Optionally includes semantic features from SigLIP.

    Tile pooling options:
    - "mean": Average tile features (default)
    - "max": Max pool tile features
    - "mean_max": Concatenate mean + max (doubles feature dim per stream)
    - "attn": Attention-weighted average (learnable)
    - "mamba": Mamba-style fusion allowing cross-tile token interaction before pooling (per-half)
    - "tile_embed_fusion": 8-tile sequence (4 left + 4 right) with half+tile embeddings + Mamba fusion
    - "q_weighted": quality-only weighting over tiles (per-half), pooled = sum softmax(MLP(q)) * f
    - "q_f_weighting": feature+quality weighting over tiles (per-half), pooled = sum softmax(MLP([f,q])) * f

    Feature-layer fusion options (controlled by `feature_fusion`):
    - "concat": Concatenate layer features (default; matches previous behavior)
    - "weighted_sum": Learn global softmax weights over layers and sum (dim stays single-layer)
    - "gated": Learn per-channel gate between two layers (expects exactly 2 layers)

    Semantic features fusion options (controlled by `semantic_features_pooling`):
    - "concat": Concatenate SigLIP semantic features with visual features (default)
    - "gating": Use SigLIP to FiLM/gate the visual features; no concatenation is performed
    """

    def __init__(
        self,
        backbone_name: str = "vit_base_patch14_dinov2",
        grid: Tuple[int, int] = (2, 2),
        tile_pooling: str = "mean",
        attn_hidden: int = 256,
        attn_dropout: float = 0.0,
        # Feature-layer fusion
        feature_fusion: str = "concat",  # "concat", "weighted_sum", "gated"
        # Semantic fusion
        semantic_features_pooling: str = "concat",  # "concat" | "gating"
        semantic_gate_hidden_ratio: float = 0.5,    # hidden size ratio vs visual dim for gating MLP
        # Mamba fusion settings (tile-level fusion)
        mamba_layers: int = 2,
        mamba_kernel_size: int = 5,
        mamba_dropout: float = 0.1,
        # Quality-weighted pooling settings
        quality_hidden: int = 64,
        quality_dropout: float = 0.0,
        quality_temperature: float = 1.0,
        quality_detach: bool = True,   # do not backprop through image-derived q
        **kwargs,
    ):
        # Set semantic_multiplier=1 for two-stream (we average left/right semantic features)
        kwargs.setdefault("semantic_multiplier", 1)

        self.grid = grid
        self.tile_pooling = tile_pooling

        valid_tile_pooling = (
            "mean",
            "max",
            "mean_max",
            "attn",
            "mamba",
            "tile_embed_fusion",
            "q_weighted",
            "q_f_weighting",
        )
        if tile_pooling not in valid_tile_pooling:
            raise ValueError(f"tile_pooling must be one of {valid_tile_pooling}, got: {tile_pooling}")

        valid_feature_fusion = ("concat", "weighted_sum", "gated")
        if feature_fusion not in valid_feature_fusion:
            raise ValueError(f"feature_fusion must be one of {valid_feature_fusion}, got: {feature_fusion}")
        self.feature_fusion = feature_fusion

        valid_semantic_pooling = ("concat", "gating")
        if semantic_features_pooling not in valid_semantic_pooling:
            raise ValueError(
                f"semantic_features_pooling must be one of {valid_semantic_pooling}, got: {semantic_features_pooling}"
            )
        self.semantic_features_pooling = semantic_features_pooling

        # Tile pooling affects feature_multiplier only for mean_max (doubles per stream)
        tile_multiplier = 2 if tile_pooling == "mean_max" else 1

        # Build BaseMultiHead normally (sets backbone, hooks, semantic branch, etc.)
        # We'll recompute dims/heads after feature-layer fusion rules are applied.
        super().__init__(backbone_name, feature_multiplier=2 * tile_multiplier, **kwargs)

        # ---------------------------------------------------------------------
        # Feature-layer fusion setup
        # ---------------------------------------------------------------------
        pooling_mult = 2 if self.feature_pooling in ("cls_mean", "cls_gem") else 1
        self.per_layer_dim = self.feat_dim_per_layer * pooling_mult
        self.n_feature_layers = len(self.feature_layers) if self.feature_layers is not None else 0

        if self.feature_layers is not None and self.n_feature_layers > 0:
            if self.feature_fusion == "concat":
                fused_feat_dim = self.per_layer_dim * self.n_feature_layers
            else:
                fused_feat_dim = self.per_layer_dim
        else:
            fused_feat_dim = self.feat_dim  # backbone pooled output path

        self.feat_dim = fused_feat_dim

        # Fusion parameters/modules
        self.layer_logits = None
        self.layer_gate = None
        if self.feature_layers is not None and self.n_feature_layers > 0:
            if self.feature_fusion == "weighted_sum":
                self.layer_logits = nn.Parameter(torch.zeros(self.n_feature_layers))
            elif self.feature_fusion == "gated":
                if self.n_feature_layers != 2:
                    raise ValueError("feature_fusion='gated' expects exactly 2 feature_layers")
                d = self.per_layer_dim
                self.layer_gate = nn.Sequential(
                    nn.Linear(2 * d, d),
                    nn.GELU(),
                    nn.Linear(d, d),
                    nn.Sigmoid(),
                )

        # ---------------------------------------------------------------------
        # Semantic fusion: concat vs gating (FiLM)
        # ---------------------------------------------------------------------
        # Visual feature dimension before semantic:
        # - two streams, and mean_max doubles per stream
        self.visual_out_dim = self.feat_dim * (2 * tile_multiplier)

        if self.use_semantic and self.semantic_branch is not None:
            if self.semantic_features_pooling == "concat":
                self.combined_dim = self.visual_out_dim + self.semantic_dim
                self.semantic_film = None
            else:  # gating
                self.combined_dim = self.visual_out_dim
                hidden = max(32, int(self.visual_out_dim * semantic_gate_hidden_ratio))
                self.semantic_film = nn.Sequential(
                    nn.Linear(self.semantic_dim, hidden),
                    nn.GELU(),
                    nn.Dropout(0.0),
                    nn.Linear(hidden, 2 * self.visual_out_dim),
                )
        else:
            self.combined_dim = self.visual_out_dim
            self.semantic_film = None

        # Rebuild heads to match new combined_dim (keep same hidden/dropout as previously constructed)
        hidden_size = self.head_total[0].out_features
        dropout = self.head_total[2].p if isinstance(self.head_total[2], nn.Dropout) else 0.0
        self.head_total = self._make_head(hidden_size, dropout)
        self.head_gdm = self._make_head(hidden_size, dropout)
        self.head_green = self._make_head(hidden_size, dropout)

        # Prints
        if self.feature_layers is not None and self.n_feature_layers > 0:
            print(
                f"Feature-layer fusion: layers={self.feature_layers}, pooling={self.feature_pooling}, "
                f"per_layer_dim={self.per_layer_dim}, n_layers={self.n_feature_layers}, "
                f"fusion='{self.feature_fusion}', feat_dim={self.feat_dim}"
            )
        if self.use_semantic:
            print(
                f"Semantic fusion: mode='{self.semantic_features_pooling}', "
                f"semantic_dim={self.semantic_dim}, visual_out_dim={self.visual_out_dim}, combined_dim={self.combined_dim}"
            )

        # ---------------------------------------------------------------------
        # Tile pooling modules
        # ---------------------------------------------------------------------
        self.attn_pool = None
        self.mamba_fusion = None
        self.tile_embed_fusion = None

        if tile_pooling == "attn":
            self.attn_pool = AttnPool(
                dim=self.feat_dim,
                hidden=attn_hidden,
                dropout=attn_dropout,
            )
            print(f"Attention tile pooling enabled: hidden={attn_hidden}, dropout={attn_dropout}")

        if tile_pooling == "mamba":
            self.mamba_fusion = MambaFusion(
                dim=self.feat_dim,
                num_layers=mamba_layers,
                kernel_size=mamba_kernel_size,
                dropout=mamba_dropout,
            )
            print(f"Mamba tile fusion enabled: layers={mamba_layers}, kernel={mamba_kernel_size}, dropout={mamba_dropout}")

        if tile_pooling == "tile_embed_fusion":
            self.tile_embed_fusion = TileEmbedFusion(
                dim=self.feat_dim,
                num_tiles_per_half=grid[0] * grid[1],
                num_layers=mamba_layers,
                kernel_size=mamba_kernel_size,
                dropout=mamba_dropout,
                pool="mean",
            )
            print(
                f"Tile-embed fusion enabled: 8-tile sequence, "
                f"layers={mamba_layers}, kernel={mamba_kernel_size}, dropout={mamba_dropout}"
            )

        # ---------------------------------------------------------------------
        # Quality-weighted pooling modules (benchmarkable, independent of mamba fusion)
        # ---------------------------------------------------------------------
        self.quality_detach = bool(quality_detach)
        self.quality_temperature = float(quality_temperature)

        # _tile_quality returns 8 dims
        self.quality_dim = 8
        self.quality_norm = nn.LayerNorm(self.quality_dim)

        self.q_mlp = None
        self.qf_mlp = None

        if tile_pooling == "q_weighted":
            self.q_mlp = nn.Sequential(
                nn.Linear(self.quality_dim, quality_hidden),
                nn.GELU(),
                nn.Dropout(quality_dropout),
                nn.Linear(quality_hidden, 1),
            )
            print(f"Quality tile pooling enabled: mode='q_weighted', qdim={self.quality_dim}, hidden={quality_hidden}")

        if tile_pooling == "q_f_weighting":
            self.qf_mlp = nn.Sequential(
                nn.Linear(self.feat_dim + self.quality_dim, quality_hidden),
                nn.GELU(),
                nn.Dropout(quality_dropout),
                nn.Linear(quality_hidden, 1),
            )
            print(f"Quality tile pooling enabled: mode='q_f_weighting', qdim={self.quality_dim}, hidden={quality_hidden}")

    # -------------------------------------------------------------------------
    # Feature extraction override to support feature_fusion
    # -------------------------------------------------------------------------
    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        self._intermediate_features.clear()
        last_features = self.backbone(x)

        if self.feature_layers is None:
            return last_features

        layer_feats = []
        blocks = self._get_backbone_blocks()
        n_blocks = len(blocks) if blocks is not None else None

        for layer_idx in self.feature_layers:
            idx = layer_idx
            if idx < 0 and n_blocks is not None:
                idx = n_blocks + idx
            if idx in self._intermediate_features:
                layer_feats.append(self._intermediate_features[idx])
            else:
                layer_feats.append(last_features)

        if self.feature_fusion == "concat":
            return torch.cat(layer_feats, dim=1)

        X = torch.stack(layer_feats, dim=1)  # [B, L, per_layer_dim]

        if self.feature_fusion == "weighted_sum":
            w = torch.softmax(self.layer_logits, dim=0)          # [L]
            fused = (X * w[None, :, None]).sum(dim=1)            # [B, per_layer_dim]
            return fused

        if self.feature_fusion == "gated":
            f0 = X[:, 0, :]
            f1 = X[:, 1, :]
            g = self.layer_gate(torch.cat([f0, f1], dim=1))      # [B, per_layer_dim]
            fused = g * f1 + (1.0 - g) * f0
            return fused

        return torch.cat(layer_feats, dim=1)

    # -------------------------------------------------------------------------
    # Tile quality features
    # -------------------------------------------------------------------------
    def _tile_quality(self, tile: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        Compute simple, cheap quality features per tile.

        tile: [B, 3, H, W] in same scale as your pipeline.
        Assumes 0..1. If your tensors are 0..255, either normalize here or adjust thresholds.

        Returns: [B, 8]
        """
        # If your pipeline uses 0..255 tensors, normalize first:
        # tile = tile / 255.0

        r, g, b = tile[:, 0:1], tile[:, 1:2], tile[:, 2:3]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b  # [B,1,H,W]

        lum_mean = gray.mean(dim=(2, 3))
        lum_std = gray.std(dim=(2, 3))

        clip_low = (gray < 0.02).float().mean(dim=(2, 3))
        clip_high = (gray > 0.98).float().mean(dim=(2, 3))

        # Simple gradient magnitude (finite differences)
        dx = gray[:, :, :, 1:] - gray[:, :, :, :-1]
        dy = gray[:, :, 1:, :] - gray[:, :, :-1, :]
        grad_mean = (dx.abs().mean(dim=(2, 3)) + dy.abs().mean(dim=(2, 3))) / 2.0
        grad_mean = torch.log1p(grad_mean)  # stabilize

        # Laplacian-ish (finite differences)
        if gray.shape[2] >= 3 and gray.shape[3] >= 3:
            lap = (
                -4 * gray[:, :, 1:-1, 1:-1]
                + gray[:, :, 1:-1, :-2]
                + gray[:, :, 1:-1, 2:]
                + gray[:, :, :-2, 1:-1]
                + gray[:, :, 2:, 1:-1]
            )
            lap_var = lap.var(dim=(2, 3))
        else:
            lap_var = torch.zeros_like(lum_mean)
        lap_var = torch.log1p(lap_var)  # stabilize

        # Saturation proxy via chroma
        maxc = torch.max(torch.max(r, g), b)
        minc = torch.min(torch.min(r, g), b)
        sat_mean = ((maxc - minc) / (maxc + eps)).mean(dim=(2, 3))

        green_ratio = (g / (r + g + b + eps)).mean(dim=(2, 3))

        q = torch.cat(
            [lum_mean, lum_std, clip_low, clip_high, lap_var, grad_mean, sat_mean, green_ratio],
            dim=1,
        )  # [B, 8]
        return q

    # -------------------------------------------------------------------------
    # Tile encoding helpers
    # -------------------------------------------------------------------------
    def _encode_tiles_raw(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image by processing tiles and returning per-tile features [B, T, F] without pooling."""
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
                feat = self._extract_features(tile)  # [B, F]
                feats.append(feat)

        feats = torch.stack(feats, dim=0).permute(1, 0, 2)  # [B, T, F]
        return feats

    def _encode_tiles_raw_with_quality(self, x: torch.Tensor):
        """
        Encode image by processing tiles and returning both:
          feats: [B, T, F]
          quals: [B, T, Q]
        """
        B, C, H, W = x.shape
        r, c = self.grid
        rows = _make_edges(H, r)
        cols = _make_edges(W, c)

        feats, quals = [], []
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

                q = self._tile_quality(tile)  # [B, Q]
                if self.quality_detach:
                    q = q.detach()
                quals.append(q)

                feat = self._extract_features(tile)  # [B, F]
                feats.append(feat)

        feats = torch.stack(feats, dim=0).permute(1, 0, 2)   # [B, T, F]
        quals = torch.stack(quals, dim=0).permute(1, 0, 2)   # [B, T, Q]
        return feats, quals

    def _encode_tiles(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image by processing tiles and aggregating with configured pooling."""
        if self.tile_pooling in ("q_weighted", "q_f_weighting"):
            feats, quals = self._encode_tiles_raw_with_quality(x)
            return self._pool_tiles(feats, quals)

        feats = self._encode_tiles_raw(x)
        return self._pool_tiles(feats)

    # -------------------------------------------------------------------------
    # Tile pooling
    # -------------------------------------------------------------------------
    def _pool_tiles_quality(self, feats: torch.Tensor, quals: torch.Tensor) -> torch.Tensor:
        """
        Quality-weighted pooling.

        feats:  [B, T, F]
        quals:  [B, T, Q]
        return: [B, F]
        """
        q = self.quality_norm(quals)  # normalize for stability

        if self.tile_pooling == "q_weighted":
            if self.q_mlp is None:
                raise RuntimeError("q_mlp is not initialized for tile_pooling='q_weighted'")
            logits = self.q_mlp(q)  # [B, T, 1]
        elif self.tile_pooling == "q_f_weighting":
            if self.qf_mlp is None:
                raise RuntimeError("qf_mlp is not initialized for tile_pooling='q_f_weighting'")
            x = torch.cat([feats, q], dim=2)  # [B, T, F+Q]
            logits = self.qf_mlp(x)           # [B, T, 1]
        else:
            raise ValueError(f"_pool_tiles_quality called with tile_pooling={self.tile_pooling}")

        # Softmax over tiles
        temp = max(1e-6, float(self.quality_temperature))
        w = torch.softmax(logits / temp, dim=1)  # [B, T, 1]

        pooled = (w * feats).sum(dim=1)          # [B, F]
        return pooled

    def _pool_tiles(self, feats: torch.Tensor, quals: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Pool tile features using configured method.

        Args:
            feats: Tile features [B, T, F]
            quals: Optional tile quality [B, T, Q] (required for q_weighted/q_f_weighting)

        Returns:
            Pooled features [B, F] or [B, 2*F] for mean_max
        """
        if self.tile_pooling == "mean":
            return feats.mean(dim=1)

        if self.tile_pooling == "max":
            return feats.max(dim=1)[0]

        if self.tile_pooling == "mean_max":
            mean_feat = feats.mean(dim=1)
            max_feat = feats.max(dim=1)[0]
            return torch.cat([mean_feat, max_feat], dim=1)

        if self.tile_pooling == "attn":
            if self.attn_pool is None:
                raise RuntimeError("attn_pool is not initialized for tile_pooling='attn'")
            pooled, _ = self.attn_pool(feats)
            return pooled

        if self.tile_pooling == "mamba":
            if self.mamba_fusion is None:
                raise RuntimeError("mamba_fusion is not initialized for tile_pooling='mamba'")
            fused = self.mamba_fusion(feats)  # [B, T, F]
            return fused.mean(dim=1)

        if self.tile_pooling in ("q_weighted", "q_f_weighting"):
            if quals is None:
                raise ValueError("quals must be provided for tile_pooling in ('q_weighted','q_f_weighting')")
            return self._pool_tiles_quality(feats, quals)

        # tile_embed_fusion handled in forward()/extract_features()
        return feats.mean(dim=1)

    # -------------------------------------------------------------------------
    # Semantic extraction
    # -------------------------------------------------------------------------
    def _extract_semantic_features(self, x_left: torch.Tensor, x_right: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.use_semantic or self.semantic_branch is None:
            return None
        sem_left = self.semantic_branch(x_left)
        sem_right = self.semantic_branch(x_right)
        return (sem_left + sem_right) / 2

    # -------------------------------------------------------------------------
    # Apply semantic fusion (concat or gating) to visual vector
    # -------------------------------------------------------------------------
    def _apply_semantic_fusion(
        self,
        visual_features: torch.Tensor,
        semantic_features: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        visual_features: [B, visual_out_dim]
        semantic_features: [B, semantic_dim] or None
        returns: [B, combined_dim]
        """
        if semantic_features is None or not self.use_semantic or self.semantic_branch is None:
            return visual_features

        if self.semantic_features_pooling == "concat":
            return torch.cat([visual_features, semantic_features], dim=1)

        # gating / FiLM
        if self.semantic_film is None:
            # fallback
            return torch.cat([visual_features, semantic_features], dim=1)

        film = self.semantic_film(semantic_features)  # [B, 2*visual_out_dim]
        gamma, beta = film.chunk(2, dim=1)

        # stable FiLM: start near identity
        gamma = torch.tanh(gamma)
        fused = visual_features * (1.0 + gamma) + beta
        return fused

    # -------------------------------------------------------------------------
    # Public feature extraction for MixUp
    # -------------------------------------------------------------------------
    def extract_features(self, x_left: torch.Tensor, x_right: torch.Tensor) -> torch.Tensor:
        if self.tile_pooling == "tile_embed_fusion":
            if self.tile_embed_fusion is None:
                raise RuntimeError("tile_embed_fusion is not initialized for tile_pooling='tile_embed_fusion'")
            tiles_left = self._encode_tiles_raw(x_left)
            tiles_right = self._encode_tiles_raw(x_right)
            feat_left, feat_right = self.tile_embed_fusion(tiles_left, tiles_right)
        else:
            feat_left = self._encode_tiles(x_left)
            feat_right = self._encode_tiles(x_right)

        visual_features = torch.cat([feat_left, feat_right], dim=1)

        semantic_features = self._extract_semantic_features(x_left, x_right)
        return self._apply_semantic_fusion(visual_features, semantic_features)

    # -------------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------------
    def forward(self, x_left: torch.Tensor, x_right: torch.Tensor):
        if self.tile_pooling == "tile_embed_fusion":
            if self.tile_embed_fusion is None:
                raise RuntimeError("tile_embed_fusion is not initialized for tile_pooling='tile_embed_fusion'")
            tiles_left = self._encode_tiles_raw(x_left)
            tiles_right = self._encode_tiles_raw(x_right)
            feat_left, feat_right = self.tile_embed_fusion(tiles_left, tiles_right)
        else:
            feat_left = self._encode_tiles(x_left)
            feat_right = self._encode_tiles(x_right)

        visual_features = torch.cat([feat_left, feat_right], dim=1)
        semantic_features = self._extract_semantic_features(x_left, x_right)
        combined = self._apply_semantic_fusion(visual_features, semantic_features)

        total = self.softplus(self.head_total(combined))
        gdm = self.softplus(self.head_gdm(combined))
        green = self.softplus(self.head_green(combined))
        return total, gdm, green





class TwoStreamTiledFiLM(TwoStreamTiled):
    """
    Two-Stream Tiled model with FiLM modulation.
    Uses Feature-wise Linear Modulation for better feature fusion.
    Optionally includes semantic features from SigLIP.
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
        return self._pool_tiles(tiles)  # [B, F] or [B, 2*F]
    
    def extract_features(
        self,
        x_left: torch.Tensor,
        x_right: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract concatenated features without prediction (for MixUp).
        Includes semantic features if enabled.
        """
        feat_left = self._encode_stream(x_left, self.film_left)
        feat_right = self._encode_stream(x_right, self.film_right)
        visual_features = torch.cat([feat_left, feat_right], dim=1)
        
        # Add semantic features if enabled
        if self.use_semantic and self.semantic_branch is not None:
            semantic_features = self._extract_semantic_features(x_left, x_right)
            return torch.cat([visual_features, semantic_features], dim=1)
        
        return visual_features
    
    def forward(
        self,
        x_left: torch.Tensor,
        x_right: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with FiLM modulation and optional semantic features.
        """
        feat_left = self._encode_stream(x_left, self.film_left)
        feat_right = self._encode_stream(x_right, self.film_right)
        combined = torch.cat([feat_left, feat_right], dim=1)
        
        # Extract semantic features if enabled
        semantic_features = self._extract_semantic_features(x_left, x_right)
        
        return self._predict_from_features(combined, semantic_features)


class TwoStreamPatchMamba(nn.Module):
    """
    Two-Stream model with patch-level Mamba fusion.
    
    Architecture from Kaggle DINOv3 notebook:
    1. Backbone extracts patch tokens (global_pool='') -> [B, N, D]
    2. Concatenate left+right patch sequences -> [B, 2N, D]
    3. Apply Mamba fusion across all patches (cross-image interaction)
    4. Pool with AdaptiveAvgPool1d -> [B, D]
    5. Predict with MLP heads (Total, GDM, Green)
    
    Key differences from TwoStreamTiled:
    - No tiling - processes full left/right images
    - Mamba operates on patch tokens, not tile features
    - Enables cross-image token interaction before pooling
    """
    
    def __init__(
        self,
        backbone_name: str = "vit_huge_plus_patch16_dinov3.lvd1689m",
        pretrained: bool = True,
        dropout: float = 0.2,
        hidden_ratio: float = 0.5,
        # Mamba fusion settings
        mamba_layers: int = 2,
        mamba_kernel_size: int = 5,
        mamba_dropout: float = 0.1,
        # Gradient checkpointing for memory efficiency
        use_grad_checkpointing: bool = True,
        # HuggingFace custom weights (optional)
        hf_repo: Optional[str] = None,
        hf_filename: Optional[str] = None,
    ):
        super().__init__()
        
        self.backbone_name = backbone_name
        self.use_grad_checkpointing = use_grad_checkpointing
        
        # Create backbone with global_pool='' to get patch tokens [B, N, D]
        if hf_repo is not None and hf_filename is not None:
            # Load from HuggingFace safetensors
            self.backbone = _load_hf_safetensors(
                hf_repo=hf_repo,
                hf_filename=hf_filename,
                base_model_name=backbone_name,
            )
            # Modify to return patch tokens (remove global pooling)
            # Note: _load_hf_safetensors uses num_classes=0 but not global_pool=''
            # We need to recreate with proper settings
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file
            ckpt_path = hf_hub_download(repo_id=hf_repo, filename=hf_filename)
            sd = load_file(ckpt_path)
            
            # Create with global_pool='' for patch tokens
            self.backbone = timm.create_model(
                backbone_name, pretrained=False, num_classes=0, global_pool=''
            )
            # Clean and load state dict
            cleaned_sd = {}
            for k, v in sd.items():
                nk = k
                if nk.startswith("module."):
                    nk = nk[len("module."):]
                cleaned_sd[nk] = v
            drop_prefixes = ("head.", "fc.", "classifier.", "cls_head.", "linear_head.")
            cleaned_sd = {k: v for k, v in cleaned_sd.items() if not k.startswith(drop_prefixes)}
            self.backbone.load_state_dict(cleaned_sd, strict=False)
            print(f"✓ Loaded HuggingFace backbone with patch tokens (global_pool='')")
        else:
            # Standard timm model with global_pool='' for patch tokens
            self.backbone = timm.create_model(
                backbone_name,
                pretrained=pretrained,
                num_classes=0,
                global_pool='',  # Key: returns patch tokens [B, N, D]
            )
            print(f"✓ Created backbone with patch tokens (global_pool='')")
        
        
        
        # Enable gradient checkpointing for memory efficiency
        if use_grad_checkpointing and hasattr(self.backbone, 'set_grad_checkpointing'):
            self.backbone.set_grad_checkpointing(True)
            print("✓ Gradient checkpointing enabled (saves ~50% VRAM)")
        

        # Get feature dimension
        self.feat_dim = self.backbone.num_features
        print(f"Backbone feature dim: {self.feat_dim}")
        
        # Mamba fusion neck - mixes concatenated patch tokens [B, 2N, D]
        self.fusion = MambaFusion(
            dim=self.feat_dim,
            num_layers=mamba_layers,
            kernel_size=mamba_kernel_size,
            dropout=mamba_dropout,
        )
        print(f"Mamba fusion: {mamba_layers} layers, kernel={mamba_kernel_size}, dropout={mamba_dropout}")
        
        # Adaptive pooling after fusion
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Combined dim is just feat_dim (after pooling)
        self.combined_dim = self.feat_dim
        
        # Calculate hidden size for heads
        hidden_size = max(8, int(self.combined_dim * hidden_ratio))
        
        # Create three separate heads for each target (matching notebook structure)
        # Using GELU and higher dropout like notebook
        self.head_total = nn.Sequential(
            nn.Linear(self.combined_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        self.head_gdm = nn.Sequential(
            nn.Linear(self.combined_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        self.head_green = nn.Sequential(
            nn.Linear(self.combined_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        
        # Softplus for non-negative outputs
        self.softplus = nn.Softplus(beta=1.0)
        
        print(f"Heads: combined_dim={self.combined_dim}, hidden={hidden_size}, dropout={dropout}")
    
    def _get_backbone_blocks(self) -> Optional[nn.ModuleList]:
        """Get the sequential blocks from the backbone."""
        if hasattr(self.backbone, "blocks"):
            return self.backbone.blocks
        if hasattr(self.backbone, "stages"):
            return self.backbone.stages
        return None
    
    def get_backbone_block_count(self) -> int:
        """Return the number of blocks in the backbone, or -1 if unknown."""
        blocks = self._get_backbone_blocks()
        return len(blocks) if blocks is not None else -1
    
    def freeze_backbone(self):
        """Freeze backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        # Keep fusion trainable
        for param in self.fusion.parameters():
            param.requires_grad = True
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def unfreeze_last_n_blocks(self, n: int) -> int:
        """
        Unfreeze only the last N blocks of the backbone.
        
        Args:
            n: Number of blocks to unfreeze from the end
        
        Returns:
            Number of blocks actually unfrozen
        """
        blocks = self._get_backbone_blocks()
        
        if blocks is None:
            print(f"Warning: Backbone has no recognizable block structure, unfreezing all")
            self.unfreeze_backbone()
            return -1
        
        total_blocks = len(blocks)
        n_to_unfreeze = min(n, total_blocks)
        start_idx = total_blocks - n_to_unfreeze
        
        unfrozen_params = 0
        for idx, block in enumerate(blocks):
            if idx >= start_idx:
                for param in block.parameters():
                    param.requires_grad = True
                    unfrozen_params += param.numel()
        
        # Also unfreeze final norm if exists
        if hasattr(self.backbone, "norm"):
            for param in self.backbone.norm.parameters():
                param.requires_grad = True
        if hasattr(self.backbone, "fc_norm"):
            for param in self.backbone.fc_norm.parameters():
                param.requires_grad = True
        
        print(f"Partially unfroze backbone: last {n_to_unfreeze}/{total_blocks} blocks")
        return n_to_unfreeze
    
    def extract_features(
        self,
        x_left: torch.Tensor,
        x_right: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract fused features without prediction (for MixUp).
        
        Args:
            x_left: Left image [B, C, H, W]
            x_right: Right image [B, C, H, W]
            
        Returns:
            Pooled features [B, feat_dim]
        """
        # Extract patch tokens from both images
        # Output: [B, N, D] where N = num_patches + 1 (CLS token)
        tokens_left = self.backbone(x_left)
        tokens_right = self.backbone(x_right)
        
        # Concatenate along sequence dimension [B, 2N, D]
        tokens_cat = torch.cat([tokens_left, tokens_right], dim=1)
        
        # Apply Mamba fusion (cross-image token interaction)
        tokens_fused = self.fusion(tokens_cat)
        
        # Pool: [B, 2N, D] -> [B, D, 2N] -> [B, D, 1] -> [B, D]
        pooled = self.pool(tokens_fused.transpose(1, 2)).flatten(1)
        
        return pooled
    
    def predict_from_features(
        self,
        features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict from pre-extracted features (for MixUp).
        
        Args:
            features: Pooled features [B, feat_dim]
            
        Returns:
            Tuple of (total, gdm, green) predictions
        """
        total = self.softplus(self.head_total(features))
        gdm = self.softplus(self.head_gdm(features))
        green = self.softplus(self.head_green(features))
        return total, gdm, green
    
    def forward(
        self,
        x_left: torch.Tensor,
        x_right: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x_left: Left image [B, C, H, W]
            x_right: Right image [B, C, H, W]
            
        Returns:
            Tuple of (total, gdm, green) predictions, each [B, 1]
        """
        features = self.extract_features(x_left, x_right)
        return self.predict_from_features(features)
    
    def predict_all_targets(
        self,
        x_left: torch.Tensor,
        x_right: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict all 5 targets including derived ones.
        
        Returns:
            Tensor [B, 5] with [Green, Dead, Clover, GDM, Total]
        """
        total, gdm, green = self.forward(x_left, x_right)
        
        # Calculate derived targets
        dead = torch.clamp(total - gdm, min=0)
        clover = torch.clamp(gdm - green, min=0)
        
        # Stack in order: [Green, Dead, Clover, GDM, Total]
        all_targets = torch.cat([green, dead, clover, gdm, total], dim=1)
        return all_targets


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
    # Patch-level Mamba fusion (DINOv3 notebook style)
    "two_stream_patch_mamba": TwoStreamPatchMamba,
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
            - 'two_stream_patch_mamba': Left/right with patch-level Mamba fusion
        backbone_name: Name of timm backbone
        pretrained: Whether to use pretrained weights
        grid: Tuple (rows, cols) for tiling (only for tiled models)
        tile_size: Resolution to resize tiles to before backbone (None = infer from backbone)
        **kwargs: Additional arguments including:
            - dropout: Dropout rate for heads
            - hidden_ratio: Hidden layer size ratio
            - feature_layers: List of layer indices for multi-layer features
            - feature_pooling: "cls", "mean", "gem", "cls_mean", "cls_gem"
            - gem_p: GeM pooling power parameter (default 3.0)
            - gem_learnable: Wh ether GeM p is learnable
            - tile_pooling: "mean", "max", "mean_max", "attn", "mamba" - how to aggregate tile features
            - attn_hidden: Hidden dim for attention pooling MLP (default 256)
            - attn_dropout: Dropout for attention pooling MLP (default 0.0)
            - mamba_layers: Number of LocalMambaBlocks for mamba/patch_mamba (default 2)
            - mamba_kernel_size: Kernel size for mamba depthwise conv (default 5)
            - mamba_dropout: Dropout for mamba fusion (default 0.1)
            - use_grad_checkpointing: Enable gradient checkpointing (for patch_mamba)
            - use_semantic: Whether to use semantic features from SigLIP
            - semantic_model_path: HuggingFace path for SigLIP model
            - semantic_freeze: Whether to freeze SigLIP weights
        
    Returns:
        Model instance
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    
    model_class = MODEL_REGISTRY[model_type]
    
    # Handle patch_mamba model - doesn't use tiling or tile_size
    if model_type == "two_stream_patch_mamba":
        # Filter out tiling-specific kwargs
        patch_mamba_kwargs = {
            k: v for k, v in kwargs.items()
            if k in (
                "dropout", "hidden_ratio", "mamba_layers", "mamba_kernel_size",
                "mamba_dropout", "use_grad_checkpointing", "hf_repo", "hf_filename"
            )
        }
        return model_class(backbone_name, pretrained=pretrained, **patch_mamba_kwargs)
    
    # Build common kwargs for other models
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
