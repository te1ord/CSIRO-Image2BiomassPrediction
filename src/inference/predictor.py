"""
Inference module for making predictions
"""
import torch
import numpy as np
import pandas as pd
from typing import Dict
from src.models.dinov2_lasso import DINOv2FeatureExtractor, LassoEnsemble


class BiomassPredictor:
    """Predictor for biomass values"""
    
    def __init__(
        self,
        feature_extractor: DINOv2FeatureExtractor,
        lasso_ensemble: LassoEnsemble,
        device: str = 'cuda'
    ):
        """
        Args:
            feature_extractor: DINOv2 feature extractor
            lasso_ensemble: Trained Lasso ensemble
            device: Device to run inference on
        """
        self.feature_extractor = feature_extractor.to(device)
        self.feature_extractor.eval()
        self.lasso_ensemble = lasso_ensemble
        self.device = device
        
        # Target mapping from notebook
        self.mapping = {
            "Dry_Clover_g": 0,
            "Dry_Dead_g": 1,
            "Dry_Green_g": 2,
            "Dry_Total_g": 3,
            "GDM_g": 4
        }
        
    def extract_embeddings(self, test_df: pd.DataFrame, root_dir: str, processor) -> Dict[str, torch.Tensor]:
        """
        Extract embeddings for test images
        
        Args:
            test_df: Test dataframe
            root_dir: Root directory for images
            processor: Image processor
            
        Returns:
            Dictionary mapping sample_id to embeddings
        """
        from PIL import Image
        
        test_embeds = {}
        sample_ids = []
        counter = 0
        
        for i in range(len(test_df)):
            entry = test_df.iloc[i]
            file_path = root_dir + entry['image_path']
            sample_id = entry['sample_id']
            
            if sample_id not in sample_ids:
                img = Image.open(file_path)
                x = torch.tensor(processor(img).pixel_values)
                
                with torch.no_grad():
                    x = x.to(self.device)
                    embedding = self.feature_extractor(x).cpu()
                    test_embeds[sample_id.split("_")[0]] = embedding
                    counter += 1
                
                sample_ids.append(sample_id)
                
                if counter % 100 == 0:
                    print(f"{counter} batches processed.")
        
        return test_embeds
    
    def predict(self, test_df: pd.DataFrame, test_embeds: Dict[str, torch.Tensor]) -> pd.DataFrame:
        """
        Make predictions for test set
        
        Args:
            test_df: Test dataframe
            test_embeds: Dictionary of test embeddings
            
        Returns:
            DataFrame with sample_id and predictions
        """
        predictions = []
        sample_ids = []
        
        for i in range(len(test_df)):
            try:
                entry = test_df.iloc[i]
                sample_id = entry['sample_id']
                sample_ids.append(sample_id)
                
                # Get embedding
                base_id = sample_id.split("__")[0]
                X = np.array(test_embeds[base_id])
                
                # Get target type and make prediction
                target_name = sample_id.split("__")[1]
                target_idx = self.mapping[target_name]
                
                prediction = self.lasso_ensemble.predict(X, target_idx)
                predictions.append(prediction)
                
            except Exception as e:
                print(f"Error processing {sample_id}: {e}")
                predictions.append(0.0)
        
        return pd.DataFrame({
            'sample_id': sample_ids,
            'target': predictions
        })

