# Data Preprocessing Scripts

This folder contains scripts for preprocessing the CSIRO Biomass dataset.

## Scripts Overview

### 1. `data_aggregation.py`
Transforms `train.csv` from long format (5 rows per sample) to wide format (1 row per sample with 5 target columns).

**Input:** Long format CSV with 1787 rows (357 samples × 5 targets)
**Output:** Wide format CSV with 357 rows (1 row per sample)

**Target columns created:**
- `Dry_Clover_g`
- `Dry_Dead_g`
- `Dry_Green_g`
- `Dry_Total_g`
- `GDM_g`

### 2. `data_split.py`
Creates balanced K-fold splits with no data leakage and organizes images into fold directories.

**Features:**
- ✅ Groups by `Sampling_Date` (prevents data leakage - same date samples stay together)
- ✅ Stratifies by `State` + `Species` (maintains balanced distribution)
- ✅ **Creates physical folder structure** with train/test splits for each fold
- ✅ **Reproducible splits** - fixed random_state ensures everyone gets identical splits
- ✅ Uses symlinks by default (saves disk space) or can copy files
- ✅ Verification checks included
- ✅ Comprehensive statistics and reports

**Output Structure:**
```
folds/
├── fold_assignments.csv          # CSV with fold assignments for each image
├── fold0/
│   ├── train/                    # Images from fold1, fold2, fold3, fold4
│   └── test/                     # Images from fold0
├── fold1/
│   ├── train/                    # Images from fold0, fold2, fold3, fold4
│   └── test/                     # Images from fold1
├── fold2/
│   ├── train/                    # Images from fold0, fold1, fold3, fold4
│   └── test/                     # Images from fold2
├── fold3/
│   ├── train/                    # Images from fold0, fold1, fold2, fold4
│   └── test/                     # Images from fold3
└── fold4/
    ├── train/                    # Images from fold0, fold1, fold2, fold3
    └── test/                     # Images from fold4
```

**How K-Fold Cross-Validation Works:**
```
All Images (357)
├─ Fold 0 (80 images)   ──→  fold0/test/   | fold1,2,3,4/train/
├─ Fold 1 (83 images)   ──→  fold1/test/   | fold0,2,3,4/train/
├─ Fold 2 (61 images)   ──→  fold2/test/   | fold0,1,3,4/train/
├─ Fold 3 (55 images)   ──→  fold3/test/   | fold0,1,2,4/train/
└─ Fold 4 (78 images)   ──→  fold4/test/   | fold0,1,2,3/train/

Training Fold 0:  Train on folds 1,2,3,4 (277 images)  →  Test on fold 0 (80 images)
Training Fold 1:  Train on folds 0,2,3,4 (274 images)  →  Test on fold 1 (83 images)
Training Fold 2:  Train on folds 0,1,3,4 (296 images)  →  Test on fold 2 (61 images)
Training Fold 3:  Train on folds 0,1,2,4 (302 images)  →  Test on fold 3 (55 images)
Training Fold 4:  Train on folds 0,1,2,3 (279 images)  →  Test on fold 4 (78 images)
```

### 3. `example_usage.py`
Demonstrates how to use the organized fold structure in your training code.

Run this to see examples of loading and using the folds:
```bash
python example_usage.py
```

## Usage

### Step 1: Aggregate Data

```bash
# Basic usage (default paths)
cd scripts
python data_aggregation.py

# Custom input/output paths
python data_aggregation.py \
    --input /path/to/train.csv \
    --output /path/to/train_wide.csv
```

**Default paths:**
- Input: `../data/csiro-biomass/train.csv`
- Output: `../data/csiro-biomass/train_wide.csv`

### Step 2: Create Folds and Organize Images

```bash
# Basic usage (5 folds with symlinks)
python data_split.py

# 10-fold cross-validation
python data_split.py --n-splits 10

# Copy files instead of symlinks (uses more disk space)
python data_split.py --copy-files

# Custom parameters
python data_split.py \
    --input ../data/csiro-biomass/train_wide.csv \
    --image-dir ../data/csiro-biomass \
    --output-dir ../data/csiro-biomass/folds \
    --n-splits 5 \
    --random-state 42 \
    --group-col Sampling_Date \
    --stratify-cols State Species \
    --use-symlinks
```

**Default paths:**
- Input CSV: `../data/csiro-biomass/train_wide.csv`
- Image directory: `../data/csiro-biomass`
- Output directory: `../data/csiro-biomass/folds`

**Parameters:**
- `--input`: Path to train_wide.csv (wide format)
- `--image-dir`: Directory containing the original images
- `--output-dir`: Directory where fold directories will be created
- `--n-splits`: Number of folds (default: 5)
- `--random-state`: Random seed for reproducibility (default: 42, **FIXED**)
- `--group-col`: Column to group by to prevent leakage (default: Sampling_Date)
- `--stratify-cols`: Columns to stratify by for balance (default: State Species)
- `--use-symlinks`: Use symlinks instead of copying (default: True, saves space)
- `--copy-files`: Copy files instead of symlinks (overrides --use-symlinks)

**Important Notes:**
- ⚠️ **Reproducibility**: Using `--random-state 42` (default) ensures everyone gets **identical splits**
- 💾 **Disk Space**: Symlinks (default) save disk space; use `--copy-files` if needed
- 🔒 **No Data Leakage**: Samples from the same date stay in the same fold

## Complete Pipeline

Run both scripts sequentially:

```bash
cd scripts

# Step 1: Aggregate data (long → wide format)
python data_aggregation.py

# Step 2: Create folds and organize images
python data_split.py

# Outputs:
# - ../data/csiro-biomass/train_wide.csv (aggregated data)
# - ../data/csiro-biomass/folds/ (organized fold directories)
#   - fold_assignments.csv (fold assignments for all images)
#   - fold0/, fold1/, ..., fold4/ (each with train/ and test/ subdirs)
```

### Quick Start (One-Liner)

```bash
cd scripts && python data_aggregation.py && python data_split.py
```

## Using the Folds in Your Training Code

### Option 1: Using Organized Folder Structure (Recommended)

```python
from pathlib import Path
import pandas as pd

# Load fold assignments to get targets
fold_assignments = pd.read_csv('../data/csiro-biomass/folds/fold_assignments.csv')

# 5-fold cross-validation
for fold in range(5):
    print(f"\nTraining fold {fold}...")
    
    # Get image directories
    train_dir = Path('../data/csiro-biomass/folds') / f'fold{fold}' / 'train'
    test_dir = Path('../data/csiro-biomass/folds') / f'fold{fold}' / 'test'
    
    # Get all images
    train_images = list(train_dir.glob('*.jpg'))
    test_images = list(test_dir.glob('*.jpg'))
    
    print(f'Fold {fold}: {len(train_images)} train, {len(test_images)} test')
    
    # Get targets from fold_assignments
    # Extract image_id from filename and merge with targets
    
    # Train your model
    # model = train_model(train_dir, test_dir)
```

### Option 2: Using CSV with Fold Assignments

```python
import pandas as pd

# Load data with fold assignments
df = pd.read_csv('../data/csiro-biomass/folds/fold_assignments.csv')

# 5-fold cross-validation
for fold in range(5):
    print(f"\nTraining fold {fold}...")
    
    # Split data
    train_data = df[df['fold'] != fold]
    val_data = df[df['fold'] == fold]
    
    # Access features and targets
    train_images = train_data['image_path'].values
    train_targets = train_data[['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 
                                  'Dry_Total_g', 'GDM_g']].values
    
    val_images = val_data['image_path'].values
    val_targets = val_data[['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g',
                             'Dry_Total_g', 'GDM_g']].values
    
    # Train your model
    # model = train_model(train_images, train_targets, val_images, val_targets)
```

## Data Characteristics

### Original Data (Long Format)
- **Rows:** 1,787 (357 samples × 5 targets each)
- **Format:** Each sample has 5 separate rows for different target types

### Aggregated Data (Wide Format)
- **Rows:** 357 (one per sample/image)
- **Format:** Each sample has 5 target columns

### Features Available
- `image_id`: Unique identifier for each image
- `image_path`: Path to image file
- `Sampling_Date`: Date when sample was collected
- `State`: Australian state (NSW, Tas, Vic, WA)
- `Species`: Plant species type (15 unique species)
- `Pre_GSHH_NDVI`: NDVI value
- `Height_Ave_cm`: Average height in cm
- `fold`: Fold assignment (0-4 for 5-fold CV)

### Targets
All measurements in grams (g):
- `Dry_Clover_g`: Dry clover biomass
- `Dry_Dead_g`: Dry dead biomass
- `Dry_Green_g`: Dry green biomass
- `Dry_Total_g`: Total dry biomass
- `GDM_g`: Green dry matter

## Verification

The `data_split.py` script includes automatic verification:

1. **No Data Leakage Check:** Verifies each date appears in only one fold
2. **Balance Check:** Shows State and Species distribution across folds
3. **Date Range Report:** Displays date ranges for each fold
4. **Target Statistics:** Shows mean and std of targets per fold

All verification results are printed during execution.

## Notes

### Reproducibility
- 🔒 **IMPORTANT**: The script uses `random_state=42` by default to ensure **everyone gets identical splits**
- This means Person A and Person B will have the exact same train/test split for each fold
- The splits are deterministic based on the random seed and data order
- To ensure reproducibility, the data is sorted consistently before splitting

### Performance
- 💾 **Symlinks (default)**: Uses virtually no extra disk space, instant creation
- 📁 **Copy files**: Uses 2x disk space (original + copies), slower but works on all systems
- Use `--copy-files` if your system doesn't support symlinks or you need portable copies

### Data Leakage Prevention
- ✅ All samples from the same `Sampling_Date` stay in the same fold
- ✅ Prevents temporal data leakage
- ✅ Simulates realistic deployment where you predict on unseen dates

### Balance
- ✅ State distribution maintained across folds
- ✅ Species distribution maintained across folds
- ✅ Each fold is representative of the overall dataset

### Other
- The scripts use relative paths by default, assuming execution from the `scripts/` directory
- All outputs include comprehensive verification and statistics
- Detailed reports show fold composition and verify no data leakage

