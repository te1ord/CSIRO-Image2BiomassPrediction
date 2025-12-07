# CSIRO Biomass Prediction - Two-Stream Multi-Head Model

A structured PyTorch Lightning implementation for the CSIRO Pasture Biomass Prediction competition.

## 🎯 Key Features

- **PyTorch Lightning**: Clean, modular training code with built-in best practices
- **W&B Integration**: Track experiments, metrics, and hyperparameters
- **Two-Stream Architecture**: Processes left/right image halves separately
- **Multi-Head Output**: Specialized heads for each target
- **Two-Stage Fine-Tuning**: Freeze → Unfreeze training strategy
- **GroupKFold CV**: Prevents data leakage by grouping by Sampling_Date
- **TTA Support**: Test-Time Augmentation for better inference

## 📊 Methodology

### 1. Core Strategy
Predict only 3 primary targets (avoiding redundancy):
- `Dry_Total_g` (50% weight)
- `GDM_g` (20% weight)
- `Dry_Green_g` (10% weight)

Derived targets are calculated:
- `Dry_Dead_g = Dry_Total_g - GDM_g`
- `Dry_Clover_g = GDM_g - Dry_Green_g`

### 2. Two-Stream Processing
- Each 2000×1000 image → two 1000×1000 patches (left/right)
- Resize to 768×768 for high-resolution detail
- Augmentations applied independently to each patch

### 3. Model Architecture
```
img_left  → backbone → features_left  ─┐
                                        ├→ concat → head_total → Dry_Total_g
img_right → backbone → features_right ─┤          → head_gdm   → GDM_g
                                                  → head_green → Dry_Green_g
```

### 4. Training Strategy
- **Stage 1** (Epochs 1-5): Backbone frozen, train heads only, LR=1e-4
- **Stage 2** (Epochs 6-20): Fine-tune entire model, LR=1e-5
- Save checkpoint based on highest R² score

## 📁 Project Structure

```
├── configs/                    # Hydra configuration files
│   ├── augmentation/          
│   │   └── default.yaml       
│   ├── data/
│   │   └── default.yaml       
│   ├── inference/
│   │   └── default.yaml       
│   ├── logging/
│   │   └── wandb.yaml         # W&B settings
│   ├── model/
│   │   ├── two_stream.yaml    
│   │   └── dinov2_tiled.yaml  
│   ├── training/
│   │   └── two_stage.yaml     
│   └── config.yaml            
│
├── src/
│   ├── augmentations/         
│   ├── datasets/              
│   ├── inference/             
│   ├── models/                
│   └── trainer/
│       ├── lightning_module.py     # LightningModule
│       ├── lightning_datamodule.py # LightningDataModule
│       ├── lightning_trainer.py    # Training utilities
│       └── pytorch_trainer.py      # Legacy trainer
│
├── scripts/
│   ├── train.py              
│   └── inference.py          
│
└── data/csiro-biomass/       
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
poetry config virtualenvs.in-project true
poetry install --no-root
poetry env activate
```

Or with pip:
```bash
pip install torch torchvision pytorch-lightning timm albumentations hydra-core wandb rich
```

### 2. Login to W&B (Optional)

```bash
wandb login
```

### 3. Train Model

```bash
# Default: Two-stream with ConvNeXt-Tiny + W&B logging
python scripts/train.py

# DINOv2 with tiling and FiLM
python scripts/train.py model=dinov2_tiled

# Disable W&B logging
python scripts/train.py logging.enabled=false

# Custom experiment name
python scripts/train.py logging.experiment_name=my_experiment

# Custom W&B project
python scripts/train.py logging.project=my-project logging.entity=my-team
```

### 4. Run Inference

```bash
python scripts/inference.py
```

## 📊 W&B Logging

The trainer automatically logs to Weights & Biases:

### Metrics Logged
- `train/loss` - Training loss (per step and epoch)
- `train/loss_total`, `train/loss_gdm`, `train/loss_green` - Individual losses
- `train/lr` - Learning rate
- `val/loss` - Validation loss
- `val/score` - Competition weighted R² score
- `val/r2_total`, `val/r2_gdm`, `val/r2_green` - Individual R² scores
- `val/r2_dead`, `val/r2_clover` - Derived target R² scores
- `stage` - Current training stage (1=frozen, 2=fine-tuning)

### Configuration
```yaml
# configs/logging/wandb.yaml
enabled: true
project: csiro-biomass
entity: null  # Your W&B username/team
experiment_name: ${model.model_type}_${model.backbone.name}
```

### Command Line Overrides
```bash
# Disable W&B
python scripts/train.py logging.enabled=false

# Change project
python scripts/train.py logging.project=my-project

# Custom experiment name
python scripts/train.py logging.experiment_name=exp_001
```

## ⚙️ Configuration

### Model Variants

| Config | Backbone | Features |
|--------|----------|----------|
| `model=two_stream` | ConvNeXt-Tiny | Basic two-stream |
| `model=dinov2_tiled` | DINOv2 ViT-B/14 | Tiling + FiLM modulation |

### Key Parameters

```yaml
# Data
data.img_size: 768          
data.n_folds: 5             

# Training
training.stage1.epochs: 5   
training.stage2.epochs: 15  
training.batch_size: 8      
training.precision: "16-mixed"  # Mixed precision training

# Model
model.heads.dropout: 0.30   
model.heads.hidden_ratio: 0.25

# Logging
logging.enabled: true
logging.project: csiro-biomass
```

## 📈 Expected Results

With the two-stream ConvNeXt-Tiny model:

| Target | Validation R² |
|--------|---------------|
| Dry_Total_g | ~0.55-0.65 |
| GDM_g | ~0.50-0.60 |
| Dry_Green_g | ~0.60-0.70 |
| **Weighted Score** | **~0.55-0.65** |

## 🔧 Advanced Usage

### Custom Backbone

```bash
python scripts/train.py model.backbone.name=efficientnet_b0
```

### Early Stopping

```bash
python scripts/train.py training.early_stopping.enabled=true training.early_stopping.patience=10
```

### Hyperparameter Sweep

```bash
python scripts/train.py -m \
    training.stage2.lr=1e-5,5e-6,1e-6 \
    model.heads.dropout=0.2,0.3,0.4
```

### Train Single Fold

```python
from src.trainer import train_fold

result = train_fold(
    model_fn=model_fn,
    train_df=train_df,
    image_dir=image_dir,
    fold=1,
    n_folds=5,
    use_wandb=True,
)
```

## 🏗️ Architecture

### Lightning Module

The `BiomassLightningModule` handles:
- Two-stage training (freeze/unfreeze)
- Loss computation with competition weights
- R² metric calculation
- Automatic logging

### DataModule

The `BiomassDataModule` handles:
- GroupKFold splitting
- Train/val data loading
- Augmentation

### Callbacks

- `ModelCheckpoint`: Save best model by R² score
- `LearningRateMonitor`: Log learning rate
- `RichProgressBar`: Beautiful progress display
- `EarlyStopping`: Optional early stopping

## 📝 Key Differences from Legacy Trainer

| Aspect | Legacy (`pytorch_trainer.py`) | Lightning |
|--------|-------------------------------|-----------|
| Training loop | Manual | Automatic |
| Logging | Manual print | W&B + CSV |
| Checkpointing | Manual | Automatic |
| Mixed precision | Manual GradScaler | Built-in |
| Progress | tqdm | Rich |
| Multi-GPU | Not supported | Automatic |

## 📚 References

- [PyTorch Lightning](https://lightning.ai/) - Training framework
- [Weights & Biases](https://wandb.ai/) - Experiment tracking
- [timm](https://github.com/huggingface/pytorch-image-models) - PyTorch Image Models
- [Albumentations](https://albumentations.ai/) - Image augmentation
- [Hydra](https://hydra.cc/) - Configuration management
