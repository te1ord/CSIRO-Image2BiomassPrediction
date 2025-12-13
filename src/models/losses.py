"""
Loss functions for CSIRO Biomass Prediction

Implements weighted SmoothL1Loss aligned with competition scoring metric:
- Dry_Total_g: 50% weight
- GDM_g: 20% weight  
- Dry_Green_g: 10% weight
"""
import torch
import torch.nn as nn
from typing import Dict, Tuple


class WeightedSmoothL1Loss(nn.Module):
    """
    Weighted SmoothL1 (Huber) Loss for multi-target regression.
    
    Uses competition weights:
    - Total: 0.50
    - GDM: 0.20
    - Green: 0.10
    
    Total loss = w_total * L(pred_total, y_total) + 
                 w_gdm * L(pred_gdm, y_gdm) + 
                 w_green * L(pred_green, y_green)
    """
    
    def __init__(
        self,
        weight_total: float = 0.50,
        weight_gdm: float = 0.20,
        weight_green: float = 0.10,
        beta: float = 1.0,
    ):
        """
        Args:
            weight_total: Weight for Dry_Total_g loss
            weight_gdm: Weight for GDM_g loss
            weight_green: Weight for Dry_Green_g loss
            beta: Threshold for SmoothL1 transition
        """
        super().__init__()
        self.weight_total = weight_total
        self.weight_gdm = weight_gdm
        self.weight_green = weight_green
        
        self.smooth_l1 = nn.SmoothL1Loss(beta=beta)
        
    def forward(
        self,
        pred_total: torch.Tensor,
        pred_gdm: torch.Tensor,
        pred_green: torch.Tensor,
        y_total: torch.Tensor,
        y_gdm: torch.Tensor,
        y_green: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute weighted loss.
        
        Args:
            pred_total, pred_gdm, pred_green: Model predictions [B, 1]
            y_total, y_gdm, y_green: Ground truth targets [B]
            
        Returns:
            Tuple of (total_loss, losses_dict)
        """
        # Ensure shapes match
        y_total = y_total.view(-1, 1)
        y_gdm = y_gdm.view(-1, 1)
        y_green = y_green.view(-1, 1)
        
        # Individual losses
        loss_total = self.smooth_l1(pred_total, y_total)
        loss_gdm = self.smooth_l1(pred_gdm, y_gdm)
        loss_green = self.smooth_l1(pred_green, y_green)
        
        # Weighted sum
        total_loss = (
            self.weight_total * loss_total +
            self.weight_gdm * loss_gdm +
            self.weight_green * loss_green
        )
        
        losses_dict = {
            "loss": total_loss.item(),
            "loss_total": loss_total.item(),
            "loss_gdm": loss_gdm.item(),
            "loss_green": loss_green.item(),
        }
        
        return total_loss, losses_dict
