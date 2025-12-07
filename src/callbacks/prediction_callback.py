from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger


class WandbPredictionLoggerCallback(Callback):
    """Callback to log sample predictions to W&B"""
    
    def __init__(self, num_samples: int = 8, log_every_n_epochs: int = 5):
        super().__init__()
        self.num_samples = num_samples
        self.log_every_n_epochs = log_every_n_epochs
        
    def on_validation_epoch_end(self, trainer, pl_module):
        """Log sample predictions at end of validation"""
        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0:
            return
            
        # Find wandb logger
        wandb_logger = None
        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                wandb_logger = logger
                break
        
        if wandb_logger is None:
            return
            
        import wandb
        
        # Get sample predictions from validation outputs
        if not hasattr(pl_module, 'validation_step_outputs') or not pl_module.validation_step_outputs:
            return
        
        # Take first batch
        first_batch = pl_module.validation_step_outputs[0] if pl_module.validation_step_outputs else None
        if first_batch is None:
            return
            
        # Create a table with predictions
        columns = ["pred_total", "pred_gdm", "pred_green", "y_total", "y_gdm", "y_green"]
        data = []
        
        n_samples = min(self.num_samples, len(first_batch["pred_total"]))
        for i in range(n_samples):
            row = [
                first_batch["pred_total"][i].item(),
                first_batch["pred_gdm"][i].item(),
                first_batch["pred_green"][i].item(),
                first_batch["y_total"][i].item(),
                first_batch["y_gdm"][i].item(),
                first_batch["y_green"][i].item(),
            ]
            data.append(row)
        
        table = wandb.Table(columns=columns, data=data)
        wandb_logger.experiment.log({
            "val/predictions_table": table,
            "epoch": trainer.current_epoch,
        })