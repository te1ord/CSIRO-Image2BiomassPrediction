"""
PyTorch Lightning Module for Biomass Prediction

Implements two-stage fine-tuning strategy:
- Stage 1: Backbone frozen, train heads only
- Stage 2: Fine-tune entire model
"""
import torch
import pytorch_lightning as pl
from typing import Dict, Any, Optional, Tuple
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

from src.models.losses import WeightedSmoothL1Loss
from src.models.metrics import CompetitionMetric


class BiomassLightningModule(pl.LightningModule):
    """
    Lightning Module for biomass prediction with two-stage training.
    
    Training Strategy:
    - Stage 1 (epochs 1 to freeze_epochs): Backbone frozen, train heads only
    - Stage 2 (epochs > freeze_epochs): Fine-tune entire model with lower LR
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        stream_mode: str = "two_stream",
        # Stage 1 settings
        freeze_epochs: int = 5,
        freeze_lr: float = 1e-4,
        # Stage 2 settings  
        unfreeze_lr: float = 1e-5,
        # Optimizer settings
        weight_decay: float = 0.01,
        # Scheduler settings
        use_scheduler: bool = True,
        scheduler_type: str = "cosine",  # "cosine" or "onecycle"
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        
        self.model = model
        self.stream_mode = stream_mode
        
        self.freeze_epochs = freeze_epochs
        self.freeze_lr = freeze_lr
        self.unfreeze_lr = unfreeze_lr
        self.weight_decay = weight_decay
        self.use_scheduler = use_scheduler
        self.scheduler_type = scheduler_type
        
        # Loss and metric
        self.criterion = WeightedSmoothL1Loss()
        self.metric = CompetitionMetric()
        
        # Track current stage
        self._current_stage = 1
        
        # Validation predictions for epoch-end metric calculation
        self.validation_step_outputs = []
        
    def forward(self, *args):
        """Forward pass through the model"""
        return self.model(*args)
    
    def _forward_batch(self, batch) -> Tuple[torch.Tensor, ...]:
        """Forward pass handling both stream modes"""
        if self.stream_mode == "single_stream":
            x, targets = batch
            y_total = targets[:, 0]
            y_gdm = targets[:, 1]
            y_green = targets[:, 2]
            pred_total, pred_gdm, pred_green = self.model(x)
        else:
            x_left, x_right, targets = batch
            y_total = targets[:, 0]
            y_gdm = targets[:, 1]
            y_green = targets[:, 2]
            pred_total, pred_gdm, pred_green = self.model(x_left, x_right)
        
        return pred_total, pred_gdm, pred_green, y_total, y_gdm, y_green
    
    def on_train_epoch_start(self):
        """Handle stage transitions at epoch start"""
        current_epoch = self.current_epoch + 1  # 1-indexed
        
        if current_epoch == 1:
            # Stage 1: Freeze backbone
            self.model.freeze_backbone()
            self._current_stage = 1
            self.log("stage", 1, prog_bar=True)
            
        elif current_epoch == self.freeze_epochs + 1:
            # Stage 2: Unfreeze backbone
            self.model.unfreeze_backbone()
            self._current_stage = 2
            self.log("stage", 2, prog_bar=True)
            
            # Update learning rate for stage 2
            for param_group in self.trainer.optimizers[0].param_groups:
                param_group['lr'] = self.unfreeze_lr
    
    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """Training step"""
        pred_total, pred_gdm, pred_green, y_total, y_gdm, y_green = self._forward_batch(batch)
        
        loss, losses = self.criterion(
            pred_total, pred_gdm, pred_green,
            y_total, y_gdm, y_green,
        )
        
        # Log losses
        self.log("train/loss", losses["loss"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/loss_total", losses["loss_total"], on_step=False, on_epoch=True)
        self.log("train/loss_gdm", losses["loss_gdm"], on_step=False, on_epoch=True)
        self.log("train/loss_green", losses["loss_green"], on_step=False, on_epoch=True)
        self.log("train/lr", self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, on_epoch=False)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        pred_total, pred_gdm, pred_green, y_total, y_gdm, y_green = self._forward_batch(batch)
        
        loss, losses = self.criterion(
            pred_total, pred_gdm, pred_green,
            y_total, y_gdm, y_green,
        )
        
        # Log validation loss
        self.log("val/loss", losses["loss"], on_step=False, on_epoch=True, prog_bar=True)
        
        # Store predictions for epoch-end metric calculation
        self.validation_step_outputs.append({
            "pred_total": pred_total.detach(),
            "pred_gdm": pred_gdm.detach(),
            "pred_green": pred_green.detach(),
            "y_total": y_total.detach(),
            "y_gdm": y_gdm.detach(),
            "y_green": y_green.detach(),
        })
        
        return loss
    
    def on_validation_epoch_end(self):
        """Calculate R² scores at end of validation epoch"""
        if not self.validation_step_outputs:
            return
        
        # Concatenate all predictions
        pred_total = torch.cat([x["pred_total"] for x in self.validation_step_outputs])
        pred_gdm = torch.cat([x["pred_gdm"] for x in self.validation_step_outputs])
        pred_green = torch.cat([x["pred_green"] for x in self.validation_step_outputs])
        
        y_total = torch.cat([x["y_total"] for x in self.validation_step_outputs])
        y_gdm = torch.cat([x["y_gdm"] for x in self.validation_step_outputs])
        y_green = torch.cat([x["y_green"] for x in self.validation_step_outputs])
        
        # Compute competition metric
        score, scores = self.metric(
            pred_total, pred_gdm, pred_green,
            y_total, y_gdm, y_green,
        )
        
        # Log R² scores
        self.log("val/score", scores["score"], prog_bar=True)
        self.log("val/r2_total", scores["r2_total"])
        self.log("val/r2_gdm", scores["r2_gdm"])
        self.log("val/r2_green", scores["r2_green"])
        self.log("val/r2_dead", scores["r2_dead"])
        self.log("val/r2_clover", scores["r2_clover"])
        
        # Clear outputs
        self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler"""
        # Get trainable parameters based on current stage
        if self._current_stage == 1:
            # Only train heads in stage 1
            params = list(self.model.head_total.parameters()) + \
                     list(self.model.head_gdm.parameters()) + \
                     list(self.model.head_green.parameters())
            lr = self.freeze_lr
        else:
            # Train all parameters in stage 2
            params = self.model.parameters()
            lr = self.unfreeze_lr
        
        optimizer = AdamW(
            params,
            lr=lr,
            weight_decay=self.weight_decay,
        )
        
        if not self.use_scheduler:
            return optimizer
        
        # Configure scheduler
        if self.scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs,
                eta_min=lr * 0.01,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }
        elif self.scheduler_type == "onecycle":
            # OneCycleLR needs total steps
            total_steps = self.trainer.estimated_stepping_batches
            scheduler = OneCycleLR(
                optimizer,
                max_lr=lr,
                total_steps=total_steps,
                pct_start=0.1,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
        
        return optimizer

