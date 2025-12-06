# CSIRO Biomass Prediction - Two-Stream Multi-Head Model

A structured PyTorch implementation for the CSIRO Pasture Biomass Prediction competition.

## 🎯 Key Features

- **Two-Stream Architecture**: Processes left/right image halves separately
- **Multi-Head Output**: Specialized heads for each target
- **Two-Stage Fine-Tuning**: Freeze → Unfreeze training strategy
- **GroupKFold CV**: Prevents data leakage by grouping by Sampling_Date
- **Weighted Loss**: Aligned with competition scoring metric
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
│   │   └── default.yaml       # Augmentation settings
│   ├── data/
│   │   └── default.yaml       # Data paths and settings
│   ├── inference/
│   │   └── default.yaml       # Inference settings
│   ├── model/
│   │   ├── two_stream.yaml    # ConvNeXt backbone
│   │   └── dinov2_tiled.yaml  # DINOv2 with tiling + FiLM
│   ├── training/
│   │   └── two_stage.yaml     # Two-stage training config
│   └── config.yaml            # Main config
│
├── src/                       # Source code
│   ├── augmentations/         # Albumentations transforms
│   ├── datasets/              # Two-stream dataset
│   ├── inference/             # Predictor with TTA
│   ├── models/                # Two-stream multi-head models
│   ├── trainer/               # PyTorch trainer
│   └── utils/
│
├── scripts/                   # Entry points
│   ├── train.py              # Training script
│   └── inference.py          # Inference script
│
└── data/csiro-biomass/       # Data directory
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
pip install torch torchvision timm albumentations hydra-core omegaconf pandas scikit-learn tqdm
```

### 2. Train Model

```bash
# Default: ConvNeXt-Tiny backbone
python scripts/train.py

# DINOv2 with tiling and FiLM
python scripts/train.py model=dinov2_tiled

# Custom settings
python scripts/train.py \
    training.batch_size=16 \
    data.img_size=512 \
    training.stage1.epochs=3
```

### 3. Run Inference

```bash
# With TTA (default)
python scripts/inference.py

# Without TTA
python scripts/inference.py inference.use_tta=false
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
data.img_size: 768          # Image resolution
data.n_folds: 5             # Cross-validation folds

# Training
training.stage1.epochs: 5   # Frozen backbone epochs
training.stage2.epochs: 15  # Fine-tuning epochs
training.batch_size: 8      # Batch size

# Model
model.heads.dropout: 0.30   # Head dropout
model.heads.hidden_ratio: 0.25
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

### Modify Augmentations

```bash
python scripts/train.py \
    augmentation.geometric.horizontal_flip.p=0.7 \
    augmentation.color.color_jitter.enabled=false
```

### Hyperparameter Sweep

```bash
# Try different learning rates
python scripts/train.py -m \
    training.stage2.lr=1e-5,5e-6,1e-6
```

## 📝 Key Differences from Baseline

| Aspect | Old (Lasso) | New (Two-Stream) |
|--------|-------------|------------------|
| Model | DINOv2 + Lasso | CNN/ViT + MLP heads |
| Training | Sklearn | PyTorch |
| CV | Random split | GroupKFold |
| Targets | All 5 | 3 primary + 2 derived |
| Augmentation | None | Flip, Rotate, ColorJitter |
| Loss | MSE | Weighted SmoothL1 |
| TTA | No | Yes |

## 📚 References

- [timm](https://github.com/huggingface/pytorch-image-models) - PyTorch Image Models
- [Albumentations](https://albumentations.ai/) - Image augmentation
- [Hydra](https://hydra.cc/) - Configuration management
