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
    
    # Feature pooling method for intermediate layers
    feature_pooling = cfg.model.get("feature_pooling", "cls")
    
    # Parse HuggingFace custom backbone config (for model architecture only - weights from checkpoint)
    hf_repo = cfg.model.backbone.get("hf_repo", None)
    hf_filename = cfg.model.backbone.get("hf_filename", None)
    
    def model_fn():
        grid = tuple(cfg.model.tiled.grid) if "tiled" in cfg.model.model_type else None
        return build_model(
            model_type=cfg.model.model_type,
            backbone_name=cfg.model.backbone.name,
            pretrained=False,  # We load from checkpoint, not pretrained
            dropout=cfg.model.heads.dropout,
            hidden_ratio=cfg.model.heads.hidden_ratio,
            grid=grid,
            tile_size=tile_size,
            feature_layers=feature_layers,
            feature_pooling=feature_pooling,
            # Don't pass HF params during inference - we load weights from checkpoint
            hf_repo=None,
            hf_filename=None,
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
    
    folds_desc = str(folds) if folds else "all available"
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Folds to load: {folds_desc}")
    
    models = load_models(
        model_fn=model_fn,
        checkpoint_dir=checkpoint_dir,
        folds=folds,
        device=device,
        checkpoint_template=checkpoint_template,
    )
    
    print(f"Using {len(models)} model(s) for inference")
    
    # Load test data
    print(f"\n[2/4] Loading test data...")
    test_long_df = pd.read_csv(cfg.data.test_csv)
    test_unique_df = test_long_df.drop_duplicates(subset=["image_path"]).reset_index(drop=True)
    print(f"✓ Found {len(test_unique_df)} unique test images")
    
    # Create predictor
    predictor = Predictor(
        models=models,
        stream_mode=stream_mode,
        device=device,
        use_amp=cfg.inference.use_amp,
        use_tta=cfg.inference.use_tta,
    )
    
    # Predict
    print(f"\n[3/4] Running inference (TTA={cfg.inference.use_tta})...")
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
