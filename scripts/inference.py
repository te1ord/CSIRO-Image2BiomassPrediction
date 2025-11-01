"""
Inference script for DINOv2 + Lasso baseline
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
from src.inference.predictor import BiomassPredictor


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Main inference function"""
    
    print("=" * 80)
    print("DINOv2 + Lasso Baseline Inference")
    print("=" * 80)
    
    # Set device
    device = cfg.device if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Load DINOv2 model and processor
    print("\n[1/4] Loading DINOv2 model...")
    processor = AutoImageProcessor.from_pretrained(cfg.model.dinov2.processor_path)
    feature_extractor = DINOv2FeatureExtractor(cfg.model.dinov2.model_path)
    feature_extractor = feature_extractor.to(device)
    print("✓ Model loaded successfully")
    
    # Load trained Lasso ensemble
    print("\n[2/4] Loading trained Lasso ensemble...")
    lasso_ensemble = LassoEnsemble(
        n_targets=cfg.model.lasso.n_targets,
        n_folds=cfg.model.lasso.n_folds,
        alpha=cfg.model.lasso.alpha
    )
    lasso_ensemble.load(cfg.inference.model_checkpoint)
    print("✓ Ensemble loaded successfully")
    
    # Create predictor
    predictor = BiomassPredictor(
        feature_extractor=feature_extractor,
        lasso_ensemble=lasso_ensemble,
        device=device
    )
    
    # Load test data
    print("\n[3/4] Loading test data...")
    test_df = pd.read_csv(cfg.data.test_csv)
    print(f"✓ Loaded {len(test_df)} test samples")
    
    # Extract test embeddings
    print("\n[4/4] Extracting embeddings and making predictions...")
    test_embeds = predictor.extract_embeddings(
        test_df=test_df,
        root_dir=cfg.data.data_root + "/",
        processor=processor
    )
    
    # Make predictions
    submission = predictor.predict(test_df, test_embeds)
    
    # Save submission
    submission.to_csv(cfg.inference.submission_file, index=False)
    print(f"\n✓ Submission saved to {cfg.inference.submission_file}")
    
    print("\n" + "=" * 80)
    print("Inference Summary")
    print("=" * 80)
    print(f"Total predictions: {len(submission)}")
    print(f"\nFirst 5 predictions:")
    print(submission.head())
    
    print("\n✓ Inference complete!")


if __name__ == "__main__":
    main()

