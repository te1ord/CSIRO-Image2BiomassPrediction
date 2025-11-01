"""
Training script for DINOv2 + Lasso baseline
"""
import os
import sys
import torch
import pandas as pd
import hydra
from omegaconf import DictConfig
from transformers import AutoImageProcessor, AutoModel

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.dinov2_lasso import DINOv2FeatureExtractor, LassoEnsemble
from src.trainer.cross_validator import CrossValidator
from src.utils.embeddings import extract_train_embeddings


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Main training function"""
    
    print("=" * 80)
    print("DINOv2 + Lasso Baseline Training")
    print("=" * 80)
    
    # Set device
    device = cfg.device if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Load DINOv2 model and processor
    print("\n[1/4] Loading DINOv2 model...")
    processor = AutoImageProcessor.from_pretrained(cfg.model.dinov2.processor_path)
    dinov2_model = AutoModel.from_pretrained(cfg.model.dinov2.model_path)
    dinov2_model = dinov2_model.to(device)
    dinov2_model.eval()
    print("✓ Model loaded successfully")
    
    # Load training data
    print("\n[2/4] Loading training data...")
    train_df = pd.read_csv(cfg.data.train_csv)
    print(f"✓ Loaded {len(train_df)} training samples")
    
    # Extract embeddings
    print("\n[3/4] Extracting embeddings...")
    embeds, targets = extract_train_embeddings(
        train_df=train_df,
        root_dir=cfg.data.data_root + "/",
        model=dinov2_model,
        processor=processor,
        device=device,
        sample_every_n=cfg.data.sample_every_n
    )
    print(f"✓ Extracted {len(embeds)} embeddings")
    
    # Train Lasso ensemble with cross-validation
    print("\n[4/4] Training Lasso ensemble with cross-validation...")
    lasso_ensemble = LassoEnsemble(
        n_targets=cfg.model.lasso.n_targets,
        n_folds=cfg.model.lasso.n_folds,
        alpha=cfg.model.lasso.alpha
    )
    
    cross_validator = CrossValidator(
        n_splits=cfg.training.cross_validation.n_splits,
        train_ratio=cfg.training.cross_validation.train_ratio,
        random_seed=cfg.training.cross_validation.random_seed
    )
    
    results = cross_validator.train(embeds, targets, lasso_ensemble)
    
    # Save model
    os.makedirs(os.path.dirname(cfg.model.save_path), exist_ok=True)
    lasso_ensemble.save(cfg.model.save_path)
    print(f"\n✓ Model saved to {cfg.model.save_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("Training Summary")
    print("=" * 80)
    for target_name, metrics in results.items():
        print(f"\n{target_name}:")
        print(f"  Avg Train R²: {metrics['avg_train_r2']:.4f}")
        print(f"  Avg Val R²: {metrics['avg_val_r2']:.4f}")
    
    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()

