from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger

class WandbGradientWatchCallback(Callback):
    """Callback to watch gradients with W&B"""
    
    def __init__(self, log_freq: int = 100, log_graph: bool = True):
        super().__init__()
        self.log_freq = log_freq
        self.log_graph = log_graph
        
    def on_train_start(self, trainer, pl_module):
        """Watch the model for gradient logging"""
        # Find wandb logger
        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                import wandb
                wandb.watch(
                    pl_module.model,
                    log="all",  # "gradients", "parameters", or "all"
                    log_freq=self.log_freq,
                    log_graph=self.log_graph,
                )
                break