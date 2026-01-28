"""
Training script for biomass prediction models using PyTorch Lightning

Uses pre-defined fold assignments (created by scripts/data_split.py)
Supports both single-stream (full image) and two-stream (left/right split) modes.
Includes W&B logging for experiment tracking.

Usage:
    # Default training with all folds
    python scripts/train.py
    
    # Train specific folds only
    python scripts/train.py training.folds_to_train=[0,1]
    
    # Disable W&B logging
    python scripts/train.py logging.enabled=false
    
    # Single-stream model
    python scripts/train.py model=single_stream
    
    # DINOv2 with tiling
    python scripts/train.py model=dinov2_tiled
    
    # Custom experiment name
    python scripts/train.py logging.experiment_name=my_experiment
"""
import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
import pytorch_lightning as pl

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.two_stream import build_model, get_stream_mode
from src.trainer.lightning_trainer import train_kfold
from src.trainer.lightning_datamodule import load_fold_assignments


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Main training function"""
    
    print("=" * 70)
    print("CSIRO Biomass Prediction - Lightning Training")
    print("=" * 70)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Set seed
    pl.seed_everything(cfg.seed)
    
    # Determine stream mode from model type
    stream_mode = get_stream_mode(cfg.model.model_type)
    print(f"\nModel type: {cfg.model.model_type}")
    print(f"Stream mode: {stream_mode}")
    print(f"Backbone: {cfg.model.backbone.name}")
    
    # W&B settings
    use_wandb = cfg.logging.enabled
    print(f"\nW&B logging: {'enabled' if use_wandb else 'disabled'}")
    if use_wandb:
        print(f"  Project: {cfg.logging.project}")
        print(f"  Entity: {cfg.logging.entity}")
        print(f"  Experiment: {cfg.logging.experiment_name}")
        print(f"  Log model: {cfg.logging.log_model}")
        print(f"  Log gradients: {cfg.logging.log_gradients}")
        print(f"  Log predictions: {cfg.logging.log_predictions}")
    
    # Load pre-defined fold assignments
    print("\n[1/3] Loading fold assignments...")
    fold_df = load_fold_assignments(cfg.data.fold_assignments_csv)
    print(f"✓ Loaded {len(fold_df)} samples with pre-defined folds")
    
    # Model factory function
    # tile_size: None means infer from backbone, otherwise use config value
    tile_size = cfg.data.get("tile_size", None)
    
    # Parse feature_layers from config (null = last only, list = concat)
    feature_layers_cfg = cfg.model.get("feature_layers", None)
    if feature_layers_cfg is not None:
        feature_layers = list(feature_layers_cfg)
    else:
        feature_layers = None
    
    # Feature pooling method for intermediate layers
    feature_pooling = cfg.model.get("feature_pooling", "cls")
    feature_layers_fusion = cfg.model.get("feature_layers_fusion", "concat")
    
    # Parse GeM pooling config (optional, for gem/cls_gem pooling)
    gem_cfg = cfg.model.get("gem", {})
    gem_p = gem_cfg.get("p", 3.0) if gem_cfg else 3.0
    gem_learnable = gem_cfg.get("learnable", True) if gem_cfg else True
    
    # Parse HuggingFace custom backbone config (optional)
    hf_repo = cfg.model.backbone.get("hf_repo", None)
    hf_filename = cfg.model.backbone.get("hf_filename", None)
    
    # Parse semantic features config (optional)
    semantic_cfg = cfg.model.get("semantic", {})
    use_semantic = semantic_cfg.get("enabled", False) if semantic_cfg else False
    semantic_model_path = semantic_cfg.get("model_path", None) if semantic_cfg else None
    semantic_freeze = semantic_cfg.get("freeze", True) if semantic_cfg else True
    semantic_features_pooling = semantic_cfg.get("semantic_features_pooling", "concat") if semantic_cfg else "concat"
    semantic_gate_hidden_ratio = semantic_cfg.get("semantic_gate_hidden_ratio", 0.5) if semantic_cfg else 0.5

    # Parse auxiliary losses config (optional)
    aux_cfg = cfg.model.get("aux_losses", {})
    aux_losses_enabled = aux_cfg.get("enabled", False) if aux_cfg else False
    aux_loss_weight = aux_cfg.get("weight", 0.1) if aux_cfg else 0.1
    state_classes = aux_cfg.get("states_classes", None) if aux_cfg else None
    species_classes = aux_cfg.get("species_classes", None) if aux_cfg else None

    aux_tasks = None
    if aux_losses_enabled:
        if state_classes is None:
            state_classes = sorted(fold_df["State"].dropna().unique().tolist())
        if species_classes is None:
            species_classes = sorted(fold_df["Species"].dropna().unique().tolist())
        # Build aux task spec (names must match dataset keys)
        aux_tasks = {
            "height_cm": {"type": "regression", "out_dim": 1},
            "ndvi": {"type": "regression", "out_dim": 1},
            "state": {"type": "classification", "out_dim": len(state_classes)},
            "species": {"type": "classification", "out_dim": len(species_classes)},
        }
        print(f"  Aux losses: enabled (weight={aux_loss_weight})")
        print(f"  Aux classes: state={len(state_classes)}, species={len(species_classes)}")

    def model_fn():
        model_type = cfg.model.model_type
        
        # Handle patch_mamba model type separately (different config structure)
        if model_type == "two_stream_patch_mamba":
            mamba_cfg = cfg.model.get("mamba", {})
            return build_model(
                model_type=model_type,
                backbone_name=cfg.model.backbone.name,
                pretrained=cfg.model.backbone.pretrained,
                dropout=cfg.model.heads.dropout,
                hidden_ratio=cfg.model.heads.hidden_ratio,
                mamba_layers=mamba_cfg.get("layers", 2) if mamba_cfg else 2,
                mamba_kernel_size=mamba_cfg.get("kernel_size", 5) if mamba_cfg else 5,
                mamba_dropout=mamba_cfg.get("dropout", 0.1) if mamba_cfg else 0.1,
                use_grad_checkpointing=cfg.model.get("use_grad_checkpointing", True),
                hf_repo=hf_repo,
                hf_filename=hf_filename,
            )
        
        # Standard tiled models
        grid = tuple(cfg.model.tiled.grid) if "tiled" in model_type else None
        tile_pooling = cfg.model.tiled.get("pooling", "mean") if "tiled" in model_type else "mean"
        attn_hidden = cfg.model.tiled.get("attn_hidden", 256) if "tiled" in model_type else 256
        attn_dropout = cfg.model.tiled.get("attn_dropout", 0.0) if "tiled" in model_type else 0.0
        return build_model(
            model_type=model_type,
            backbone_name=cfg.model.backbone.name,
            pretrained=cfg.model.backbone.pretrained,
            dropout=cfg.model.heads.dropout,
            hidden_ratio=cfg.model.heads.hidden_ratio,
            grid=grid,
            tile_size=tile_size,
            tile_pooling=tile_pooling,
            attn_hidden=attn_hidden,
            attn_dropout=attn_dropout,
            feature_layers=feature_layers,
            feature_pooling=feature_pooling,
            feature_fusion=feature_layers_fusion,
            gem_p=gem_p,
            gem_learnable=gem_learnable,
            hf_repo=hf_repo,
            hf_filename=hf_filename,
            # Semantic features
            use_semantic=use_semantic,
            semantic_model_path=semantic_model_path,
            semantic_freeze=semantic_freeze,
            semantic_features_pooling=semantic_features_pooling,
            semantic_gate_hidden_ratio=semantic_gate_hidden_ratio,
            # Aux heads
            aux_tasks=aux_tasks,
        )
    
    tile_desc = f"{tile_size}px" if tile_size else "inferred from backbone"
    tile_pool_desc = cfg.model.tiled.get("pooling", "mean") if "tiled" in cfg.model.model_type else "N/A"
    layers_desc = str(feature_layers) if feature_layers else "last only"
    hf_desc = f"{hf_repo}" if hf_repo else "none (using timm)"
    semantic_desc = f"{semantic_model_path}" if use_semantic else "disabled"
    print(f"\n[2/3] Model: {cfg.model.model_type} with {cfg.model.backbone.name}")
    print(f"  Tile size: {tile_desc}, tile pooling: {tile_pool_desc}")
    print(f"  Feature layers: {layers_desc}, pooling: {feature_pooling}")
    print(f"  HuggingFace weights: {hf_desc}")
    print(f"  Semantic features: {semantic_desc}")
    
    # Determine which folds to train
    folds_to_train = cfg.training.get("folds_to_train", None)
    if folds_to_train is not None:
        # Convert OmegaConf ListConfig to Python list
        if isinstance(folds_to_train, ListConfig):
            folds_to_train = list(folds_to_train)
        print(f"  Training specific folds: {folds_to_train}")
    else:
        print(f"  Training all {cfg.data.n_folds} folds")
    
    # Training
    print(f"\n[3/3] Starting training...")
    print(f"  Stage 1: {cfg.training.stage1.epochs} epochs (frozen), LR={cfg.training.stage1.lr}")
    print(f"  Stage 2: {cfg.training.stage2.epochs} epochs (fine-tune), LR={cfg.training.stage2.lr}")
    
    # Parse full_image_size from config (can be null or [width, height])
    full_image_size_cfg = cfg.data.get("full_image_size", None)
    if full_image_size_cfg is not None:
        full_image_size = tuple(full_image_size_cfg) if hasattr(full_image_size_cfg, '__iter__') else None
    else:
        full_image_size = None
    
    print(f"  Full image size: {full_image_size}")
    
    # Parse MixUp config
    use_mixup = cfg.augmentation.mixup.get("enabled", False)
    mixup_alpha = cfg.augmentation.mixup.get("alpha", 0.4)
    mixup_p = cfg.augmentation.mixup.get("p", 0.5)
    
    if use_mixup:
        print(f"  Embedding MixUp: enabled (alpha={mixup_alpha}, p={mixup_p})")
    
    # Parse adaptive unfreezing config
    adaptive_cfg = cfg.training.get("adaptive_unfreeze", {})
    adaptive_unfreeze = adaptive_cfg.get("enabled", False) if adaptive_cfg else False
    adaptive_unfreeze_patience = adaptive_cfg.get("patience", 5) if adaptive_cfg else 5
    adaptive_unfreeze_mode = adaptive_cfg.get("mode", "partial") if adaptive_cfg else "partial"
    adaptive_unfreeze_last_n = adaptive_cfg.get("unfreeze_last_n", 6) if adaptive_cfg else 6
    adaptive_unfreeze_lr_factor = adaptive_cfg.get("lr_factor", 0.5) if adaptive_cfg else 0.5
    
    if adaptive_unfreeze:
        print(f"  Adaptive unfreeze: enabled (patience={adaptive_unfreeze_patience}, "
              f"mode={adaptive_unfreeze_mode}, last_n={adaptive_unfreeze_last_n})")
    
    # Parse warmup config
    scheduler_cfg = cfg.training.get("scheduler", {})
    warmup_cfg = scheduler_cfg.get("warmup", {}) if scheduler_cfg else {}
    use_warmup = warmup_cfg.get("enabled", False) if warmup_cfg else False
    warmup_epochs = warmup_cfg.get("epochs", 2) if warmup_cfg else 2
    warmup_start_factor = warmup_cfg.get("start_factor", 0.1) if warmup_cfg else 0.1
    
    if use_warmup:
        print(f"  LR warmup: {warmup_epochs} epochs (start_factor={warmup_start_factor})")
    
    # Parse EMA config
    ema_cfg = cfg.training.get("ema", {})
    use_ema = ema_cfg.get("enabled", False) if ema_cfg else False
    ema_decay = ema_cfg.get("decay", 0.999) if ema_cfg else 0.999
    
    if use_ema:
        print(f"  EMA: enabled (decay={ema_decay})")
    
    # Parse gradient accumulation config
    accumulate_grad_batches = cfg.training.get("accumulate_grad_batches", 1)
    if accumulate_grad_batches > 1:
        print(f"  Gradient accumulation: {accumulate_grad_batches} steps (effective batch={cfg.training.batch_size * accumulate_grad_batches})")
    
    # Parse gradient clipping config
    grad_clip_cfg = cfg.training.get("gradient_clip", {})
    gradient_clip_enabled = grad_clip_cfg.get("enabled", False) if grad_clip_cfg else False
    gradient_clip_val = grad_clip_cfg.get("max_norm", 1.0) if gradient_clip_enabled else None
    if gradient_clip_enabled:
        print(f"  Gradient clipping: max_norm={gradient_clip_val}")
        
    # Parse target transform config
    use_log_transform = cfg.augmentation.get("use_log_transform", False)
    if use_log_transform:
        print(f"  Target transform: log1p enabled")
    
    results = train_kfold(
        model_fn=model_fn,
        fold_df=fold_df,
        image_dir=cfg.data.train_image_dir,
        n_folds=cfg.data.n_folds,
        folds_to_train=folds_to_train,
        stream_mode=stream_mode,
        # Training settings
        img_size=cfg.data.img_size,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        # Stage settings
        freeze_epochs=cfg.training.stage1.epochs,
        unfreeze_epochs=cfg.training.stage2.epochs,
        freeze_lr=cfg.training.stage1.lr,
        unfreeze_lr=cfg.training.stage2.lr,
        # Trainer settings
        precision=cfg.training.precision,
        checkpoint_dir=cfg.checkpoint_dir,
        # Logging settings
        use_wandb=cfg.logging.enabled,
        wandb_project=cfg.logging.project,
        wandb_entity=cfg.logging.entity,
        experiment_name=cfg.logging.experiment_name,
        log_dir=cfg.logging.log_dir,
        # W&B specific logging
        log_model=cfg.logging.log_model,
        log_gradients=cfg.logging.log_gradients,
        log_predictions=cfg.logging.log_predictions,
        # Pass full Hydra config for complete logging
        hydra_config=cfg,
        # Other settings
        early_stopping=cfg.training.early_stopping.enabled,
        patience=cfg.training.early_stopping.patience,
        seed=cfg.seed,
        # Full image resize before stream splitting
        full_image_size=full_image_size,
        # Embedding-level MixUp
        use_mixup=use_mixup,
        mixup_alpha=mixup_alpha,
        mixup_p=mixup_p,
        # Adaptive unfreezing
        adaptive_unfreeze=adaptive_unfreeze,
        adaptive_unfreeze_patience=adaptive_unfreeze_patience,
        adaptive_unfreeze_mode=adaptive_unfreeze_mode,
        adaptive_unfreeze_last_n=adaptive_unfreeze_last_n,
        adaptive_unfreeze_lr_factor=adaptive_unfreeze_lr_factor,
        # Learning rate warmup
        use_warmup=use_warmup,
        warmup_epochs=warmup_epochs,
        warmup_start_factor=warmup_start_factor,
        # EMA
        use_ema=use_ema,
        ema_decay=ema_decay,
        # Gradient accumulation
        accumulate_grad_batches=accumulate_grad_batches,
        # Gradient clipping
        gradient_clip_val=gradient_clip_val,
        # Target transformation
        use_log_transform=use_log_transform,
        # Auxiliary losses / targets
        aux_losses_enabled=aux_losses_enabled,
        aux_loss_weight=aux_loss_weight,
        state_classes=state_classes,
        species_classes=species_classes,
    )
    
    print("\n✓ Training complete!")
    print(f"Checkpoints saved to: {cfg.checkpoint_dir}")
    
    return results


if __name__ == "__main__":
    main()
