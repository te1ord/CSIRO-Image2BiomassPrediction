# CSIRO Biomass Prediction - DINOv2 + Lasso Baseline

This repository contains a structured baseline implementation for the CSIRO Biomass Prediction competition using DINOv2 features and Lasso regression.

## 📁 Project Structure

```
CSIRO-Image2BiomassPrediction/
├── configs/                      # Hydra configuration files
│   ├── augmentation/            # Augmentation configs
│   │   └── none.yaml
│   ├── data/                    # Data configs
│   │   └── default.yaml
│   ├── inference/               # Inference configs
│   │   └── default.yaml
│   ├── model/                   # Model configs
│   │   └── dinov2_lasso.yaml
│   ├── training/                # Training configs
│   │   └── cross_validation.yaml
│   └── config.yaml              # Global config
├── data/                        # Data directory
│   └── csiro-biomass/
│       ├── train/
│       ├── test/
│       ├── train.csv
│       └── test.csv
├── src/                         # Source code
│   ├── augmentations/           # Image augmentations
│   │   └── transforms.py
│   ├── datasets/                # Dataset classes
│   │   └── biomass_dataset.py
│   ├── inference/               # Inference logic
│   │   └── predictor.py
│   ├── models/                  # Model definitions
│   │   └── dinov2_lasso.py
│   ├── trainer/                 # Training logic
│   │   └── cross_validator.py
│   └── utils/                   # Utility functions
│       └── embeddings.py
├── scripts/                     # Executable scripts
│   ├── train.py
│   └── inference.py
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🚀 Setup

1. **Install dependencies:**
```bash
poetry config virtualenvs.in-project true
poetry install --no-root
poetry env activate
```

2. **Prepare data:**
   - Place your data in `data/csiro-biomass/`
   - Ensure the following structure:
     ```
     data/csiro-biomass/
     ├── train/
     ├── test/
     ├── train.csv
     └── test.csv
     ```

3. **Configure paths:**
   - Update `configs/data/default.yaml` with correct data paths
   - Update `configs/model/dinov2_lasso.yaml` with DINOv2 model path

## 📊 Training

Train the model using Hydra configuration:

```bash
python scripts/train.py
```

### Custom Configuration

Override config parameters from command line:

```bash
python scripts/train.py \
    data.data_root=/path/to/data \
    model.lasso.alpha=0.5 \
    training.cross_validation.n_splits=10
```

Or create a new config file in `configs/` and use it:

```bash
python scripts/train.py --config-name=my_config
```

## 🔮 Inference

Run inference on test set:

```bash
python scripts/inference.py
```

This will:
1. Load the trained model from `models/lasso_ensemble.pkl`
2. Extract DINOv2 features from test images
3. Generate predictions using the Lasso ensemble
4. Save results to `submission.csv`

## 🧩 Model Architecture

The baseline model consists of:

1. **Feature Extractor:** DINOv2 (pretrained vision transformer)
2. **Regressor:** Lasso regression with 5-fold cross-validation
3. **Ensemble:** Averages predictions from all 5 folds

### Training Strategy

- Samples every 5th image during embedding extraction
- 5-fold cross-validation with 80/20 train/val split
- Separate Lasso model for each of 5 target variables:
  - Dry_Clover_g
  - Dry_Dead_g
  - Dry_Green_g
  - Dry_Total_g
  - GDM_g

## 📈 Results

The baseline achieves the following average validation R² scores:

- Target 1 (Dry_Clover_g): ~0.51
- Target 2 (Dry_Dead_g): ~0.35
- Target 3 (Dry_Green_g): ~0.66
- Target 4 (Dry_Total_g): ~0.53
- Target 5 (GDM_g): ~0.63

## 🔧 Configuration Management

This project uses [Hydra](https://hydra.cc/) for configuration management:

- **Global config:** `configs/config.yaml`
- **Data config:** `configs/data/default.yaml`
- **Model config:** `configs/model/dinov2_lasso.yaml`
- **Training config:** `configs/training/cross_validation.yaml`
- **Inference config:** `configs/inference/default.yaml`
- **Augmentation config:** `configs/augmentation/none.yaml`

## 🎯 Next Steps

This baseline provides a structured foundation. Consider these improvements:

1. **Augmentations:** Add data augmentation in `src/augmentations/`
2. **Advanced Models:** Replace Lasso with XGBoost, LightGBM, or neural networks
3. **Feature Engineering:** Add custom features or use different backbones
4. **Ensemble Methods:** Combine multiple models
5. **Hyperparameter Tuning:** Use Hydra's sweeper for grid search

## 📝 License

This is a competition baseline implementation.

## 🙏 Acknowledgments

- DINOv2 by Meta AI
- CSIRO for the biomass dataset
