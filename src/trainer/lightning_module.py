"""
PyTorch Lightning Module for Biomass Prediction

Implements two-stage fine-tuning strategy:
- Stage 1: Backbone frozen, train heads only
- Stage 2: Fine-tune entire model

Supports:
- Embedding-level MixUp augmentation
- Adaptive unfreezing (plateau-triggered backbone unfreezing)
- Learning rate warmup
- EMA (Exponential Moving Average) of model weights
"""
import copy
import torch
import pytorch_lightning as pl
from typing import Dict, Any, Optional, Tuple
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, LinearLR, SequentialLR

from src.models.losses import WeightedSmoothL1Loss
from src.models.metrics import CompetitionMetric
from src.augmentations.mixup import EmbeddingMixUp


class ModelEMA:
    """
    Exponential Moving Average of model weights.
    
    Maintains a smoothed copy of model weights that often generalizes better.
    The EMA model is updated each training step and can be used for validation.
    
    Based on timm's ModelEmaV2 but simplified for our use case.
    """
    
    def __init__(self, model: torch.nn.Module, decay: float = 0.999, device: Optional[torch.device] = None):
        """
        Args:
            model: Model to track
            decay: EMA decay rate (higher = slower update, more smoothing)
            device: Device for EMA model (None = same as model)
        """
        # Create a deep copy of the model
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        self.decay = decay
        self.device = device
        
        # Disable gradients for EMA model
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        
        # Move to device if specified
        if device is not None:
            self.ema_model.to(device)
    
    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        """Update EMA weights with current model weights."""
        # Get source and target state dicts
        model_state = model.state_dict()
        ema_state = self.ema_model.state_dict()
        
        for key in ema_state.keys():
            if key in model_state:
                # EMA update: ema = decay * ema + (1 - decay) * current
                ema_state[key].mul_(self.decay).add_(
                    model_state[key].to(ema_state[key].device), 
                    alpha=1.0 - self.decay
                )
    
    def state_dict(self) -> Dict[str, Any]:
        """Get EMA model state dict."""
        return self.ema_model.state_dict()
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load EMA model state dict."""
        self.ema_model.load_state_dict(state_dict)


class BiomassLightningModule(pl.LightningModule):
    """
    Lightning Module for biomass prediction with two-stage training.
    
    Training Strategy:
    - Stage 1 (epochs 1 to freeze_epochs): Backbone frozen, train heads only
    - Stage 2 (epochs > freeze_epochs): Fine-tune entire model with lower LR
    
    Optional Adaptive Unfreezing:
    - If validation score plateaus for N epochs during Stage 1, automatically
      unfreeze the backbone (fully or partially - last M blocks only)
    - This helps when frozen features are insufficient for the task
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
        # Warmup settings
        use_warmup: bool = False,
        warmup_epochs: int = 2,
        warmup_start_factor: float = 0.1,
        # EMA settings
        use_ema: bool = False,
        ema_decay: float = 0.999,
        # Embedding MixUp settings
        use_mixup: bool = False,
        mixup_alpha: float = 0.4,
        mixup_p: float = 0.5,
        # Adaptive unfreezing settings
        adaptive_unfreeze: bool = False,
        adaptive_unfreeze_patience: int = 5,
        adaptive_unfreeze_mode: str = "partial",  # "full" or "partial"
        adaptive_unfreeze_last_n: int = 6,
        adaptive_unfreeze_lr_factor: float = 0.5,
        use_log_transform: bool = False,
        # Auxiliary losses (train-only metadata supervision)
        aux_losses_enabled: bool = False,
        aux_loss_weight: float = 0.1,
    ):
        # Target transformation
        # use_log_transform: bool = False,

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
        
        # Warmup settings
        self.use_warmup = use_warmup
        self.warmup_epochs = warmup_epochs
        self.warmup_start_factor = warmup_start_factor
        
        # EMA settings
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema = None  # Initialized in on_fit_start
        
        # Embedding-level MixUp
        self.use_mixup = use_mixup
        if use_mixup:
            self.mixup = EmbeddingMixUp(alpha=mixup_alpha, p=mixup_p)
        else:
            self.mixup = None
        
        # Adaptive unfreezing settings
        self.adaptive_unfreeze = adaptive_unfreeze
        self.adaptive_unfreeze_patience = adaptive_unfreeze_patience
        self.adaptive_unfreeze_mode = adaptive_unfreeze_mode
        self.adaptive_unfreeze_last_n = adaptive_unfreeze_last_n
        self.adaptive_unfreeze_lr_factor = adaptive_unfreeze_lr_factor
        
        # Target transformation
        self.use_log_transform = use_log_transform

        # Auxiliary losses
        self.aux_losses_enabled = bool(aux_losses_enabled)
        self.aux_loss_weight = float(aux_loss_weight)
        self._aux_reg_criterion = torch.nn.SmoothL1Loss(beta=1.0)
        self._aux_cls_criterion = torch.nn.CrossEntropyLoss()
        
        # Loss and metric
        self.criterion = WeightedSmoothL1Loss()
        self.metric = CompetitionMetric()
        
        # Track current stage
        self._current_stage = 1
        
        # Adaptive unfreezing tracking
        self._best_score = float("-inf")
        self._epochs_without_improvement = 0
        self._adaptive_unfreeze_triggered = False
        
        # Validation predictions for epoch-end metric calculation
        self.validation_step_outputs = []
        
    def forward(self, *args):
        """Forward pass through the model"""
        return self.model(*args)
    
    def on_fit_start(self):
        """Initialize EMA model at the start of training."""
        if self.use_ema:
            self.ema = ModelEMA(self.model, decay=self.ema_decay, device=self.device)
            print(f"✓ EMA enabled with decay={self.ema_decay}")
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Update EMA model after each training batch."""
        if self.ema is not None:
            self.ema.update(self.model)
    
    def _forward_batch(self, batch) -> Tuple[torch.Tensor, ...]:
        """Forward pass handling both stream modes"""
        aux_targets = None

        if self.stream_mode == "single_stream":
            if len(batch) == 2:
                x, targets = batch
            else:
                x, targets, aux_targets = batch
            features = self.model.extract_features(x)
        else:
            if len(batch) == 3:
                x_left, x_right, targets = batch
            else:
                x_left, x_right, targets, aux_targets = batch
            features = self.model.extract_features(x_left, x_right)

        y_total = targets[:, 0]
        y_gdm = targets[:, 1]
        y_green = targets[:, 2]
        pred_total, pred_gdm, pred_green = self.model.predict_from_features(features)

        return pred_total, pred_gdm, pred_green, y_total, y_gdm, y_green, aux_targets, features
    
    def on_train_epoch_start(self):
        """Handle stage transitions at epoch start"""
        current_epoch = self.current_epoch + 1  # 1-indexed
        
        if current_epoch == 1:
            # Check if we should freeze backbone or train from start
            if self.freeze_epochs == 0:
                # No freeze stage - train backbone from start
                self._current_stage = 2
                self.log("stage", 2, prog_bar=True)
                print(f"✓ Training backbone from start (freeze_epochs=0)")
            else:
                # Stage 1: Freeze backbone
                self.model.freeze_backbone()
                self._current_stage = 1
                self.log("stage", 1, prog_bar=True)
                
                # Log adaptive unfreeze status if enabled
                if self.adaptive_unfreeze:
                    n_blocks = self.model.get_backbone_block_count()
                    print(f"✓ Adaptive unfreezing enabled: patience={self.adaptive_unfreeze_patience}, "
                        f"mode={self.adaptive_unfreeze_mode}, last_n={self.adaptive_unfreeze_last_n}, "
                        f"backbone has {n_blocks} blocks")
            
        elif current_epoch == self.freeze_epochs + 1 and not self._adaptive_unfreeze_triggered:
            # Stage 2: Unfreeze backbone (only if adaptive unfreeze hasn't already triggered)
            # Skip if we're already in stage 2 (e.g., freeze_epochs=0)
            if self._current_stage == 1:
                self.model.unfreeze_backbone()
                self._current_stage = 2
                self.log("stage", 2, prog_bar=True)
                
                # Update learning rate for stage 2
                for param_group in self.trainer.optimizers[0].param_groups:
                    param_group['lr'] = self.unfreeze_lr
    
    def _trigger_adaptive_unfreeze(self):
        """
        Trigger adaptive unfreezing when plateau is detected.
        
        Called during validation epoch end when improvement patience is exceeded.
        """
        if self._adaptive_unfreeze_triggered:
            return  # Already triggered, don't do it again
        
        self._adaptive_unfreeze_triggered = True
        current_epoch = self.current_epoch + 1
        
        print(f"\n{'='*60}")
        print(f"ADAPTIVE UNFREEZE TRIGGERED at epoch {current_epoch}")
        print(f"No improvement for {self._epochs_without_improvement} epochs")
        print(f"{'='*60}")
        
        # Calculate new LR for unfrozen layers (lower than current to avoid instability)
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        new_backbone_lr = current_lr * self.adaptive_unfreeze_lr_factor
        
        if self.adaptive_unfreeze_mode == "full":
            # Full unfreeze
            self.model.unfreeze_backbone()
            print(f"Mode: FULL unfreeze (all backbone layers)")
        else:
            # Partial unfreeze - only last N blocks
            unfrozen = self.model.unfreeze_last_n_blocks(self.adaptive_unfreeze_last_n)
            print(f"Mode: PARTIAL unfreeze (last {unfrozen} blocks)")
        
        # Create new optimizer with separate param groups for backbone vs heads
        # This allows different learning rates
        self._reconfigure_optimizer_after_unfreeze(new_backbone_lr, current_lr)
        
        # Reset improvement tracking since we changed the model
        self._epochs_without_improvement = 0
        self._current_stage = 1.5  # Mark as "adaptive stage"
        
        print(f"New LR: backbone={new_backbone_lr:.2e}, heads={current_lr:.2e}")
        print(f"{'='*60}\n")
    
    def _reconfigure_optimizer_after_unfreeze(self, backbone_lr: float, head_lr: float):
        """
        Reconfigure optimizer with separate learning rates for backbone and heads.
        
        Called after adaptive unfreezing to set lower LR for backbone layers.
        """
        optimizer = self.trainer.optimizers[0]
        
        # Identify backbone vs head parameters
        backbone_params = []
        head_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "backbone" in name:
                    backbone_params.append(param)
                else:
                    head_params.append(param)
        
        # Clear existing param groups and add new ones
        optimizer.param_groups.clear()
        
        if backbone_params:
            optimizer.add_param_group({
                'params': backbone_params,
                'lr': backbone_lr,
                'weight_decay': self.weight_decay,
            })
        
        if head_params:
            optimizer.add_param_group({
                'params': head_params,
                'lr': head_lr,
                'weight_decay': self.weight_decay,
            })
    
    def _forward_with_mixup(self, batch) -> Tuple[torch.Tensor, ...]:
        """Forward pass with embedding-level MixUp"""
        # Extract inputs and targets
        aux_targets = None
        if self.stream_mode == "single_stream":
            if len(batch) == 2:
                x, targets = batch
            else:
                x, targets, aux_targets = batch
            y_total = targets[:, 0]
            y_gdm = targets[:, 1]
            y_green = targets[:, 2]
            
            # Extract features
            features = self.model.extract_features(x)
        else:
            if len(batch) == 3:
                x_left, x_right, targets = batch
            else:
                x_left, x_right, targets, aux_targets = batch
            y_total = targets[:, 0]
            y_gdm = targets[:, 1]
            y_green = targets[:, 2]
            
            # Extract features
            features = self.model.extract_features(x_left, x_right)
        
        # Stack targets for MixUp
        stacked_targets = torch.stack([y_total, y_gdm, y_green], dim=1)  # [B, 3]
        
        # Apply MixUp on embeddings
        mixed_features, mixed_targets, lam, index = self.mixup(features, stacked_targets, return_index=True)
        
        # Predict from mixed features
        pred_total, pred_gdm, pred_green = self.model.predict_from_features(mixed_features)
        
        # Unstack mixed targets
        y_total_mixed = mixed_targets[:, 0]
        y_gdm_mixed = mixed_targets[:, 1]
        y_green_mixed = mixed_targets[:, 2]
        
        return pred_total, pred_gdm, pred_green, y_total_mixed, y_gdm_mixed, y_green_mixed, aux_targets, mixed_features, lam, index

    def _soft_cross_entropy(self, logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        return -(target_probs * log_probs).sum(dim=1).mean()

    def _compute_aux_losses_mixup(
        self,
        features: torch.Tensor,
        aux_targets: Optional[Dict[str, torch.Tensor]],
        lam: float,
        index: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if not self.aux_losses_enabled or aux_targets is None:
            return torch.tensor(0.0, device=self.device), {}

        if not hasattr(self.model, "predict_aux_from_features"):
            return torch.tensor(0.0, device=self.device), {}

        aux_preds: Dict[str, torch.Tensor] = self.model.predict_aux_from_features(features)
        if not aux_preds:
            return torch.tensor(0.0, device=self.device), {}

        aux_types = getattr(self.model, "aux_task_types", {}) or {}
        total = torch.tensor(0.0, device=self.device)
        logs: Dict[str, float] = {}

        lam_f = float(lam)

        for name, pred in aux_preds.items():
            if name not in aux_targets:
                continue

            t = str(aux_types.get(name, "regression"))
            if t == "classification":
                y = aux_targets[name].long().view(-1)
                num_classes = int(pred.shape[1])
                y1 = F.one_hot(y, num_classes=num_classes).float()
                y2 = y1[index]
                y_mix = lam_f * y1 + (1.0 - lam_f) * y2
                loss = self._soft_cross_entropy(pred, y_mix)
            else:
                y = aux_targets[name].float().view(-1, 1)
                y_mix = lam_f * y + (1.0 - lam_f) * y[index]
                loss = self._aux_reg_criterion(pred.view(-1, 1), y_mix)

            total = total + loss
            logs[f"aux_{name}"] = float(loss.detach().cpu().item())

        return total, logs

    def _compute_aux_losses(
        self,
        features: torch.Tensor,
        aux_targets: Optional[Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if not self.aux_losses_enabled or aux_targets is None:
            return torch.tensor(0.0, device=self.device), {}

        if not hasattr(self.model, "predict_aux_from_features"):
            return torch.tensor(0.0, device=self.device), {}

        aux_preds: Dict[str, torch.Tensor] = self.model.predict_aux_from_features(features)
        if not aux_preds:
            return torch.tensor(0.0, device=self.device), {}

        aux_types = getattr(self.model, "aux_task_types", {}) or {}

        total = torch.tensor(0.0, device=self.device)
        logs: Dict[str, float] = {}

        for name, pred in aux_preds.items():
            if name not in aux_targets:
                continue
            t = str(aux_types.get(name, "regression"))
            if t == "classification":
                loss = self._aux_cls_criterion(pred, aux_targets[name].long().view(-1))
            else:
                loss = self._aux_reg_criterion(pred.view(-1, 1), aux_targets[name].float().view(-1, 1))
            total = total + loss
            logs[f"aux_{name}"] = float(loss.detach().cpu().item())

        return total, logs
    
    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """Training step"""
        # Use MixUp if enabled
        if self.use_mixup and self.mixup is not None:
            pred_total, pred_gdm, pred_green, y_total, y_gdm, y_green, aux_targets, features, lam, index = self._forward_with_mixup(batch)
            aux_loss, aux_logs = self._compute_aux_losses_mixup(features, aux_targets, lam, index)
        else:
            pred_total, pred_gdm, pred_green, y_total, y_gdm, y_green, aux_targets, features = self._forward_batch(batch)
            aux_loss, aux_logs = self._compute_aux_losses(features, aux_targets)
        
        loss, losses = self.criterion(
            pred_total, pred_gdm, pred_green,
            y_total, y_gdm, y_green,
        )

        total_loss = loss + (self.aux_loss_weight * aux_loss)
        
        # Log losses
        self.log("train/loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/loss_total", losses["loss_total"], on_step=False, on_epoch=True)
        self.log("train/loss_gdm", losses["loss_gdm"], on_step=False, on_epoch=True)
        self.log("train/loss_green", losses["loss_green"], on_step=False, on_epoch=True)
        if self.aux_losses_enabled:
            self.log("train/aux_loss", aux_loss, on_step=False, on_epoch=True)
            for k, v in aux_logs.items():
                self.log(f"train/{k}", v, on_step=False, on_epoch=True)
        self.log("train/lr", self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, on_epoch=False)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        pred_total, pred_gdm, pred_green, y_total, y_gdm, y_green, aux_targets, features = self._forward_batch(batch)
        
        loss, losses = self.criterion(
            pred_total, pred_gdm, pred_green,
            y_total, y_gdm, y_green,
        )

        aux_loss, aux_logs = self._compute_aux_losses(features, aux_targets)
        total_loss = loss + (self.aux_loss_weight * aux_loss)
        
        # Log validation loss
        self.log("val/loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        if self.aux_losses_enabled:
            self.log("val/aux_loss", aux_loss, on_step=False, on_epoch=True)
            for k, v in aux_logs.items():
                self.log(f"val/{k}", v, on_step=False, on_epoch=True)
        
        # Store predictions for epoch-end metric calculation
        self.validation_step_outputs.append({
            "pred_total": pred_total.detach(),
            "pred_gdm": pred_gdm.detach(),
            "pred_green": pred_green.detach(),
            "y_total": y_total.detach(),
            "y_gdm": y_gdm.detach(),
            "y_green": y_green.detach(),
        })
        
        return total_loss
    
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
        
        # Inverse transform predictions if using log transform
        if self.use_log_transform:
            pred_total = torch.expm1(pred_total)
            pred_gdm = torch.expm1(pred_gdm)
            pred_green = torch.expm1(pred_green)
            
            # Also inverse transform the ground truth targets
            y_total = torch.expm1(y_total)
            y_gdm = torch.expm1(y_gdm)
            y_green = torch.expm1(y_green)
        
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
        
        # Adaptive unfreezing: track improvement and trigger if plateaued
        if self.adaptive_unfreeze and self._current_stage == 1:
            current_score = scores["score"]
            
            if current_score > self._best_score:
                self._best_score = current_score
                self._epochs_without_improvement = 0
            else:
                self._epochs_without_improvement += 1
                
                # Check if we should trigger adaptive unfreezing
                if self._epochs_without_improvement >= self.adaptive_unfreeze_patience:
                    self._trigger_adaptive_unfreeze()
        
        # Clear outputs
        self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler with optional warmup"""
        # Get trainable parameters based on current stage
        if self._current_stage == 1:
            # Only train heads in stage 1
            params = (
                list(self.model.head_total.parameters())
                + list(self.model.head_gdm.parameters())
                + list(self.model.head_green.parameters())
            )
            if hasattr(self.model, "aux_heads"):
                params = params + list(self.model.aux_heads.parameters())
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
            # Calculate epochs for main scheduler
            main_epochs = self.trainer.max_epochs
            if self.use_warmup:
                main_epochs = max(1, self.trainer.max_epochs - self.warmup_epochs)
            
            main_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=main_epochs,
                eta_min=lr * 0.01,
            )
            
            # Add warmup if enabled
            if self.use_warmup and self.warmup_epochs > 0:
                warmup_scheduler = LinearLR(
                    optimizer,
                    start_factor=self.warmup_start_factor,
                    end_factor=1.0,
                    total_iters=self.warmup_epochs,
                )
                scheduler = SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, main_scheduler],
                    milestones=[self.warmup_epochs],
                )
                print(f"✓ LR warmup enabled: {self.warmup_epochs} epochs, "
                      f"start_factor={self.warmup_start_factor}")
            else:
                scheduler = main_scheduler
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }
        elif self.scheduler_type == "onecycle":
            # OneCycleLR has built-in warmup (pct_start), so we use that instead
            total_steps = self.trainer.estimated_stepping_batches
            # If warmup is enabled, use warmup_epochs to calculate pct_start
            if self.use_warmup:
                steps_per_epoch = total_steps // self.trainer.max_epochs
                warmup_steps = self.warmup_epochs * steps_per_epoch
                pct_start = warmup_steps / total_steps
                pct_start = max(0.05, min(0.3, pct_start))  # Clamp to reasonable range
            else:
                pct_start = 0.1
            
            scheduler = OneCycleLR(
                optimizer,
                max_lr=lr,
                total_steps=total_steps,
                pct_start=pct_start,
            )
            
            if self.use_warmup:
                print(f"✓ OneCycleLR warmup: pct_start={pct_start:.2f} "
                      f"({int(pct_start * total_steps)} steps)")
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
        
        return optimizer
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]):
        """Save EMA model state in checkpoint."""
        if self.ema is not None:
            checkpoint["ema_state_dict"] = self.ema.state_dict()
    
    def on_load_checkpoint(self, checkpoint: Dict[str, Any]):
        """Load EMA model state from checkpoint."""
        if self.use_ema and "ema_state_dict" in checkpoint:
            # EMA will be initialized in on_fit_start, so we save the state for later
            self._ema_state_to_load = checkpoint["ema_state_dict"]
    
    def on_train_start(self):
        """Load saved EMA state after EMA is initialized."""
        if hasattr(self, "_ema_state_to_load") and self.ema is not None:
            self.ema.load_state_dict(self._ema_state_to_load)
            del self._ema_state_to_load
            print("✓ Loaded EMA state from checkpoint")
