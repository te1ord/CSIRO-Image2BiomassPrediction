import torch
import torch.nn as nn
from typing import Tuple, Dict


class CompetitionMetric(nn.Module):
    """
    Official-style globally weighted R² metric for the CSIRO competition.

    Works with *wide* format batches, where for each sample we have:
        - Dry_Total_g   (y_total)
        - GDM_g         (y_gdm)
        - Dry_Green_g   (y_green)

    Dead and Clover are derived from predictions and targets:
        Dry_Dead_g   ≈ Dry_Total_g - GDM_g
        Dry_Clover_g ≈ GDM_g - Dry_Green_g

    The metric itself follows the competition description:
        - One global weighted R² over all (sample, target) pairs
        - Per-target weights:
            Dry_Green_g  : 0.1
            Dry_Dead_g   : 0.1
            Dry_Clover_g : 0.1
            GDM_g        : 0.2
            Dry_Total_g  : 0.5
        - Uses a global weighted mean of all ground-truth values
    """

    WEIGHTS = {
        "green": 0.10,   # Dry_Green_g
        "dead": 0.10,    # Dry_Dead_g
        "clover": 0.10,  # Dry_Clover_g
        "gdm": 0.20,     # GDM_g
        "total": 0.50,   # Dry_Total_g
    }

    def __init__(self):
        super().__init__()
        # Order: [green, dead, clover, gdm, total]
        self.register_buffer(
            "target_weights",
            torch.tensor(
                [
                    self.WEIGHTS["green"],
                    self.WEIGHTS["dead"],
                    self.WEIGHTS["clover"],
                    self.WEIGHTS["gdm"],
                    self.WEIGHTS["total"],
                ],
                dtype=torch.float32,
            ),
        )

    @staticmethod
    def _flatten(x: torch.Tensor) -> torch.Tensor:
        """Ensure tensor is 1D float [N]."""
        return x.view(-1).float()

    @staticmethod
    def _r2_unweighted(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Standard (unweighted) R² for logging per-target scores.
        Not the competition metric itself.
        """
        pred = pred.view(-1).float()
        target = target.view(-1).float()

        t_mean = target.mean()
        ss_res = ((target - pred) ** 2).sum()
        ss_tot = ((target - t_mean) ** 2).sum()

        if ss_tot == 0:
            # Edge case: no variance in target
            return torch.tensor(0.0, device=target.device)

        return 1.0 - ss_res / ss_tot

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
        Compute competition-style global weighted R².

        Args:
            pred_total: predictions for Dry_Total_g   [N] or [N, 1]
            pred_gdm:   predictions for GDM_g         [N] or [N, 1]
            pred_green: predictions for Dry_Green_g   [N] or [N, 1]
            y_total:    ground truth Dry_Total_g
            y_gdm:      ground truth GDM_g
            y_green:    ground truth Dry_Green_g

        Returns:
            (global_r2, scores_dict)
                global_r2: scalar tensor with the metric
                scores_dict: dict with:
                    - "score": global weighted R² (float)
                    - "r2_total", "r2_gdm", "r2_green",
                      "r2_dead", "r2_clover" for logging
        """
        # Flatten everything to [N]
        pred_total = self._flatten(pred_total)
        pred_gdm = self._flatten(pred_gdm)
        pred_green = self._flatten(pred_green)

        y_total = self._flatten(y_total)
        y_gdm = self._flatten(y_gdm)
        y_green = self._flatten(y_green)

        # Derive dead and clover from predictions & targets
        # (matches your existing training setup)
        pred_dead = torch.clamp(pred_total - pred_gdm, min=0.0)
        pred_clover = torch.clamp(pred_gdm - pred_green, min=0.0)

        y_dead = torch.clamp(y_total - y_gdm, min=0.0)
        y_clover = torch.clamp(y_gdm - y_green, min=0.0)

        # Stack into shape [N, 5] in consistent order:
        # [green, dead, clover, gdm, total]
        y_all = torch.stack(
            [y_green, y_dead, y_clover, y_gdm, y_total],
            dim=1,
        )
        p_all = torch.stack(
            [pred_green, pred_dead, pred_clover, pred_gdm, pred_total],
            dim=1,
        )

        # Broadcast weights to [N, 5]
        w = self.target_weights.view(1, 5).expand_as(y_all)

        # Global weighted mean of ground truth
        wsum = w.sum()
        if wsum == 0:
            global_r2 = torch.tensor(0.0, device=y_all.device)
        else:
            mu_w = (w * y_all).sum() / wsum

            # Weighted residual and total sum of squares
            ss_res = ((y_all - p_all) ** 2 * w).sum()
            ss_tot = ((y_all - mu_w) ** 2 * w).sum()

            if ss_tot == 0:
                global_r2 = torch.tensor(0.0, device=y_all.device)
            else:
                global_r2 = 1.0 - ss_res / ss_tot

        # Per-target unweighted R² for logging (not part of competition metric)
        r2_total = self._r2_unweighted(pred_total, y_total)
        r2_gdm = self._r2_unweighted(pred_gdm, y_gdm)
        r2_green = self._r2_unweighted(pred_green, y_green)
        r2_dead = self._r2_unweighted(pred_dead, y_dead)
        r2_clover = self._r2_unweighted(pred_clover, y_clover)

        scores_dict = {
            "score": float(global_r2.item()),
            "r2_total": float(r2_total.item()),
            "r2_gdm": float(r2_gdm.item()),
            "r2_green": float(r2_green.item()),
            "r2_dead": float(r2_dead.item()),
            "r2_clover": float(r2_clover.item()),
        }
        return global_r2, scores_dict
