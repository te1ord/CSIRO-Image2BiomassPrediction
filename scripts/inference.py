"""
Inference script for biomass prediction models

Supports:
- Single-stream (full image) and two-stream (left/right split) modes
- Ensemble of multiple folds or specific fold selection
- With or without Test-Time Augmentation (TTA)

Usage:
    # All folds (default)
    python scripts/inference.py inference.checkpoint_dir=logs/exp
    
    # Specific folds only
    python scripts/inference.py inference.folds="[0, 2, 4]"
    
    # Single fold
    python scripts/inference.py inference.folds="[1]"
    
    # Without TTA
    python scripts/inference.py inference.use_tta=false
    
    # Single-stream model
    python scripts/inference.py model=single_stream
"""
import os
import sys
import torch
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.two_stream import build_model, get_stream_mode
from src.inference.predictor import Predictor, load_models


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Main inference function"""
    
    print("=" * 70)
    print("CSIRO Biomass Prediction - Inference")
    print("=" * 70)
    
    # Device
    device = cfg.device if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    # Determine stream mode from model type
    stream_mode = get_stream_mode(cfg.model.model_type)
    print(f"Model type: {cfg.model.model_type}")
    print(f"Stream mode: {stream_mode}")
    print(f"TTA: {cfg.inference.use_tta}")
    
    # Model factory function
    tile_size = cfg.data.get("tile_size", None)
    
    # Parse feature_layers from config
    feature_layers_cfg = cfg.model.get("feature_layers", None)
    feature_layers = list(feature_layers_cfg) if feature_layers_cfg is not None else None
    feature_layers_fusion = cfg.model.get("feature_layers_fusion", "concat")
    
    # Feature pooling method for intermediate layers
    feature_pooling = cfg.model.get("feature_pooling", "cls")
    
    # Parse GeM pooling config (for gem/cls_gem pooling)
    gem_cfg = cfg.model.get("gem", {})
    gem_p = gem_cfg.get("p", 3.0) if gem_cfg else 3.0
    gem_learnable = gem_cfg.get("learnable", True) if gem_cfg else True
    
    # Parse HuggingFace custom backbone config (for model architecture only - weights from checkpoint)
    hf_repo = cfg.model.backbone.get("hf_repo", None)
    hf_filename = cfg.model.backbone.get("hf_filename", None)
    
    # Parse semantic features config (must match training config for checkpoint loading)
    # Parse semantic features config (optional)
    semantic_cfg = cfg.model.get("semantic", {})
    use_semantic = semantic_cfg.get("enabled", False) if semantic_cfg else False
    semantic_model_path = semantic_cfg.get("model_path", None) if semantic_cfg else None
    semantic_freeze = semantic_cfg.get("freeze", True) if semantic_cfg else True
    semantic_features_pooling = semantic_cfg.get("semantic_features_pooling", "concat") if semantic_cfg else "concat"
    semantic_gate_hidden_ratio = semantic_cfg.get("semantic_gate_hidden_ratio", 0.5) if semantic_cfg else 0.5
    
    if use_semantic:
        print(f"Semantic features: enabled (model={semantic_model_path})")
    
    def model_fn():
        grid = tuple(cfg.model.tiled.grid) if "tiled" in cfg.model.model_type else None
        tile_pooling = cfg.model.tiled.get("pooling", "mean") if "tiled" in cfg.model.model_type else "mean"
        attn_hidden = cfg.model.tiled.get("attn_hidden", 256) if "tiled" in cfg.model.model_type else 256
        attn_dropout = cfg.model.tiled.get("attn_dropout", 0.0) if "tiled" in cfg.model.model_type else 0.0
        mamba_layers = cfg.model.tiled.get("mamba_layers", 2) if "tiled" in cfg.model.model_type else 2
        mamba_kernel_size = cfg.model.tiled.get("mamba_kernel_size", 5) if "tiled" in cfg.model.model_type else 5
        mamba_dropout = cfg.model.tiled.get("mamba_dropout", 0.1) if "tiled" in cfg.model.model_type else 0.1
        quality_hidden = cfg.model.tiled.get("quality_hidden", 64) if "tiled" in cfg.model.model_type else 64
        quality_dropout = cfg.model.tiled.get("quality_dropout", 0.0) if "tiled" in cfg.model.model_type else 0.0
        quality_temperature = cfg.model.tiled.get("quality_temperature", 1.0) if "tiled" in cfg.model.model_type else 1.0
        quality_detach = cfg.model.tiled.get("quality_detach", True) if "tiled" in cfg.model.model_type else True
        return build_model(
            model_type=cfg.model.model_type,
            backbone_name=cfg.model.backbone.name,
            pretrained=False,  # We load from checkpoint, not pretrained
            dropout=cfg.model.heads.dropout,
            hidden_ratio=cfg.model.heads.hidden_ratio,
            grid=grid,
            tile_size=tile_size,
            tile_pooling=tile_pooling,
            attn_hidden=attn_hidden,
            attn_dropout=attn_dropout,
            mamba_layers=mamba_layers,
            mamba_kernel_size=mamba_kernel_size,
            mamba_dropout=mamba_dropout,
            quality_hidden=quality_hidden,
            quality_dropout=quality_dropout,
            quality_temperature=quality_temperature,
            quality_detach=quality_detach,
            feature_layers=feature_layers,
            feature_pooling=feature_pooling,
            feature_fusion=feature_layers_fusion,
            gem_p=gem_p,
            gem_learnable=gem_learnable,
            # Don't pass HF params during inference - we load weights from checkpoint
            hf_repo=None,
            hf_filename=None,
            # Semantic features - must match training config
            use_semantic=use_semantic,
            semantic_model_path=semantic_model_path,
            semantic_freeze=semantic_freeze,
            semantic_features_pooling=semantic_features_pooling,
            semantic_gate_hidden_ratio=semantic_gate_hidden_ratio,
        )
    
    # Load model(s)
    print(f"\n[1/4] Loading model(s)...")
    
    checkpoint_dir = cfg.inference.checkpoint_dir
    checkpoint_template = cfg.inference.get("checkpoint_template", "best_model_fold{fold}.ckpt")
    
    # Parse folds from config (null = all, [0,2,4] = specific folds)
    folds_cfg = cfg.inference.get("folds", None)
    if folds_cfg is not None:
        folds = list(folds_cfg)
    else:
        folds = None
    
    # Memory-efficient loading: share frozen backbone/semantic across folds
    share_frozen = cfg.inference.get("share_frozen", True)
    
    folds_desc = str(folds) if folds else "all available"
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Folds to load: {folds_desc}")
    print(f"Share frozen modules: {share_frozen} {'(memory efficient)' if share_frozen else '(independent models)'}")
    
    models = load_models(
        model_fn=model_fn,
        checkpoint_dir=checkpoint_dir,
        folds=folds,
        device=device,
        checkpoint_template=checkpoint_template,
        share_frozen=share_frozen,
    )
    
    print(f"Using {len(models)} model(s) for inference")
    
    # Load test data
    print(f"\n[2/4] Loading test data...")
    test_long_df = pd.read_csv(cfg.data.test_csv)
    test_unique_df = test_long_df.drop_duplicates(subset=["image_path"]).reset_index(drop=True)
    print(f"✓ Found {len(test_unique_df)} unique test images")
    
    # Parse reconciliation config (default: enabled)
    use_reconciliation = cfg.inference.get("use_reconciliation", True)
    
    # Check if model was trained with log transform
    use_log_transform = cfg.augmentation.get("use_log_transform", False)
    
    # Parse ensemble weights (optional)
    ensemble_weights_cfg = cfg.inference.get("ensemble_weights", None)
    if ensemble_weights_cfg is not None:
        ensemble_weights = list(ensemble_weights_cfg)
        print(f"Ensemble weights: {ensemble_weights}")
    else:
        ensemble_weights = None
        print("Ensemble weights: equal (simple average)")
    
    # Create predictor
    predictor = Predictor(
        models=models,
        stream_mode=stream_mode,
        device=device,
        use_amp=cfg.inference.use_amp,
        use_tta=cfg.inference.use_tta,
        use_reconciliation=use_reconciliation,
        use_log_transform=use_log_transform,
        ensemble_weights=ensemble_weights,
    )
    
    # Predict
    print(f"\n[3/4] Running inference (TTA={cfg.inference.use_tta}, Reconciliation={use_reconciliation})...")
    predictions = predictor.predict_with_tta(
        test_df=test_unique_df,
        image_dir=cfg.data.test_image_dir,
        img_size=cfg.data.img_size,
        batch_size=cfg.inference.batch_size,
        num_workers=cfg.inference.num_workers,
    )
    
    # Create submission
    print(f"\n[4/4] Creating submission...")
    submission = predictor.create_submission(
        predictions=predictions,
        test_long_df=test_long_df,
        test_unique_df=test_unique_df,
        output_path=cfg.inference.submission_file,
    )
    
    print("\n✓ Inference complete!")
    
    return submission


if __name__ == "__main__":
    main()
