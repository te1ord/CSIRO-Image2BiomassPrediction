"""
DINOv2 + Lasso Regression Model
"""
import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from transformers import AutoImageProcessor, AutoModel
from typing import List, Optional


class DINOv2FeatureExtractor(nn.Module):
    """DINOv2 model for feature extraction"""
    
    def __init__(self, model_path: str):
        """
        Args:
            model_path: Path to pretrained DINOv2 model
        """
        super().__init__()
        self.model = AutoModel.from_pretrained(model_path)
        self.model.eval()
        
    def forward(self, pixel_values):
        """Extract features from images"""
        with torch.no_grad():
            outputs = self.model(pixel_values)
            return outputs.pooler_output


class LassoEnsemble:
    """Ensemble of Lasso regressors trained with cross-validation"""
    
    def __init__(self, n_targets: int = 5, n_folds: int = 5, alpha: float = 1.0):
        """
        Args:
            n_targets: Number of target variables
            n_folds: Number of cross-validation folds
            alpha: Lasso regularization parameter
        """
        self.n_targets = n_targets
        self.n_folds = n_folds
        self.alpha = alpha
        # Initialize regressors: [n_targets][n_folds]
        self.regressors = [[None for _ in range(n_folds)] for _ in range(n_targets)]
        
    def train_fold(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        target_idx: int,
        fold_idx: int
    ):
        """Train a single fold for a specific target"""
        reg = Lasso(alpha=self.alpha)
        reg.fit(X_train, y_train)
        
        # Store the regressor
        self.regressors[target_idx][fold_idx] = reg
        
        # Calculate metrics
        train_preds = reg.predict(X_train)
        train_preds[train_preds < 0.0] = 0.0
        train_r2 = r2_score(y_train, train_preds)
        
        val_preds = reg.predict(X_val)
        val_preds[val_preds < 0.0] = 0.0
        val_r2 = r2_score(y_val, val_preds)
        
        return train_r2, val_r2
    
    def predict(self, X: np.ndarray, target_idx: int) -> float:
        """
        Predict using ensemble of models for a specific target
        
        Args:
            X: Feature vector
            target_idx: Index of the target variable (0-4)
            
        Returns:
            Averaged prediction from all folds
        """
        prediction = 0.0
        for reg in self.regressors[target_idx]:
            single_pred = reg.predict(X)
            if single_pred < 0.0:
                single_pred = 0.0
            prediction += single_pred
        prediction = prediction / self.n_folds
        return float(prediction)
    
    def save(self, path: str):
        """Save the ensemble models"""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.regressors, f)
    
    def load(self, path: str):
        """Load the ensemble models"""
        import pickle
        with open(path, 'rb') as f:
            self.regressors = pickle.load(f)

