#!/usr/bin/env python3
import os
import shutil
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold

# Try to import tqdm, fall back to basic iteration if not available
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc="", leave=True):
        """Fallback when tqdm is not available"""
        return iterable


def get_season(month: int) -> str:
        if month in [9, 10, 11]:
            return "Autumn"
        elif month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Spring"
        else:
            return "Summer"

def get_month(date: str) -> int:
    return int(date.split("/")[1])


def discretize_target(
    values: pd.Series,
    n_bins: int = 5,
    strategy: str = 'quantile'
) -> pd.Series:
    """
    Discretize a continuous target variable into bins for stratification.
    
    Args:
        values: Series of continuous target values
        n_bins: Number of bins to create
        strategy: 'quantile' for equal-frequency bins, 'uniform' for equal-width bins
    
    Returns:
        Series of bin labels (integers 0 to n_bins-1)
    """
    if strategy == 'quantile':
        # Equal-frequency binning (each bin has roughly same number of samples)
        try:
            bins = pd.qcut(values, q=n_bins, labels=False, duplicates='drop')
        except ValueError:
            # If too few unique values, fall back to fewer bins
            n_unique = values.nunique()
            bins = pd.qcut(values, q=min(n_bins, n_unique), labels=False, duplicates='drop')
    elif strategy == 'uniform':
        # Equal-width binning
        bins = pd.cut(values, bins=n_bins, labels=False)
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'quantile' or 'uniform'.")
    
    return bins




def create_folds(
    train_wide: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
    group_col: str = 'Sampling_Date',
    target_col: str = 'Dry_Total_g',
    n_target_bins: int = 5,
    bin_strategy: str = 'quantile',
    stratify_cols: list = None,
    no_stratify: bool = False
) -> pd.DataFrame:
    """
    Create K-fold splits with date-based grouping and optional stratification.
    
    Args:
        train_wide: DataFrame in wide format (one row per sample)
        n_splits: Number of folds
        random_state: Random seed for reproducibility
        group_col: Column to group by (prevents data leakage)
        target_col: Regression target column to discretize for stratification
        n_target_bins: Number of bins to discretize the target into
        bin_strategy: 'quantile' for equal-frequency, 'uniform' for equal-width bins
        stratify_cols: Additional columns to stratify by (e.g., State, Species). 
                       These are combined with the discretized target.
        no_stratify: If True, use simple GroupKFold without any stratification
    
    Returns:
        DataFrame with 'fold' column added
    """
    print(f"Creating {n_splits}-fold splits...")
    print(f"Group by: {group_col}")
    
    # Create a copy to avoid modifying the original
    df = train_wide.copy()
    
    # Parse dates if not already done
    if 'Sampling_Date_parsed' not in df.columns:
        df['Sampling_Date_parsed'] = pd.to_datetime(df['Sampling_Date'])
    
    if no_stratify:
        # Simple GroupKFold without stratification
        print("Stratification: DISABLED (using simple GroupKFold)")
        
        gkf = GroupKFold(n_splits=n_splits)
        
        df['fold'] = -1
        for fold_idx, (train_idx, val_idx) in enumerate(
            gkf.split(df, groups=df[group_col])
        ):
            df.loc[val_idx, 'fold'] = fold_idx
    else:
        # StratifiedGroupKFold with discretized target
        print(f"Discretizing target '{target_col}' into {n_target_bins} bins (strategy: {bin_strategy})")
        df['target_bin'] = discretize_target(
            df[target_col], 
            n_bins=n_target_bins, 
            strategy=bin_strategy
        )
        
        # Print target bin distribution
        bin_counts = df['target_bin'].value_counts().sort_index()
        print(f"Target bin distribution:")
        for bin_idx, count in bin_counts.items():
            bin_mask = df['target_bin'] == bin_idx
            bin_min = df.loc[bin_mask, target_col].min()
            bin_max = df.loc[bin_mask, target_col].max()
            print(f"  Bin {bin_idx}: {count:3d} samples ({bin_min:.1f}g - {bin_max:.1f}g)")
        
        # Build stratification key starting with discretized target
        df['stratify_key'] = df['target_bin'].astype(str)
        stratify_components = [f'{target_col}_binned']
        
        # Add additional stratification columns if provided
        if stratify_cols is not None:
            print(f"Additional stratification by: {stratify_cols}")
            
            # Add Season column if needed for stratification
            if 'Season' in stratify_cols and 'Season' not in df.columns:
                df['Month'] = df['Sampling_Date'].apply(get_month)
                df['Season'] = df['Month'].apply(get_season)
            
            for col in stratify_cols:
                df['stratify_key'] += '_' + df[col].astype(str)
                stratify_components.append(col)
        
        print(f"Stratify key components: {stratify_components}")
        
        # Check number of unique stratify keys vs groups
        n_unique_keys = df['stratify_key'].nunique()
        n_groups = df[group_col].nunique()
        print(f"Unique stratify keys: {n_unique_keys}, Groups: {n_groups}")
        
        # Use StratifiedGroupKFold
        sgkf = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state
        )
        
        df['fold'] = -1
        for fold_idx, (train_idx, val_idx) in enumerate(
            sgkf.split(df, df['stratify_key'], groups=df[group_col])
        ):
            df.loc[val_idx, 'fold'] = fold_idx
    
    print(f"Created {n_splits} folds")
    
    return df


def verify_splits(df: pd.DataFrame, group_col: str = 'Sampling_Date', target_col: str = 'Dry_Total_g') -> dict:
    """
    Verify that splits have no data leakage and are balanced.
    
    Args:
        df: DataFrame with 'fold' column
        group_col: Column that was used for grouping
        target_col: Regression target column
    
    Returns:
        Dictionary with verification results
    """
    results = {}
    
    # Check for data leakage
    date_fold_check = df.groupby(group_col)['fold'].nunique()
    dates_in_multiple_folds = date_fold_check[date_fold_check > 1]
    
    if len(dates_in_multiple_folds) != 0:
        print(f"WARNING: {len(dates_in_multiple_folds)} dates appear in multiple folds")
    
    # Fold distribution
    print(f"\nFold distribution:")
    fold_counts = df['fold'].value_counts().sort_index()
    print(fold_counts)
    results['fold_counts'] = fold_counts.to_dict()
    
    # Dates per fold
    print(f"\nDates per fold:")
    dates_per_fold = df.groupby('fold')[group_col].nunique()
    print(dates_per_fold)
    results['dates_per_fold'] = dates_per_fold.to_dict()
    
    # Target distribution across folds (key for regression stratification)
    print(f"\n{target_col} statistics per fold:")
    target_stats = df.groupby('fold')[target_col].agg(['mean', 'std', 'min', 'max', 'median'])
    print(target_stats.round(2))
    results['target_stats'] = target_stats.to_dict()
    
    # Target bin distribution across folds
    if 'target_bin' in df.columns:
        print(f"\nTarget bin distribution across folds:")
        bin_dist = pd.crosstab(df['target_bin'], df['fold'], margins=True)
        print(bin_dist)
    
    # State distribution across folds
    print(f"\nState distribution across folds:")
    state_dist = pd.crosstab(df['State'], df['fold'], margins=True)
    print(state_dist)
    
    # Species distribution across folds
    print(f"\nSpecies distribution across folds:")
    species_dist = pd.crosstab(df['Species'], df['fold'], margins=True)
    print(species_dist)
    
    # Date ranges per fold
    print(f"\nDate ranges per fold:")
    for fold in sorted(df['fold'].unique()):
        fold_data = df[df['fold'] == fold]
        min_date = fold_data['Sampling_Date_parsed'].min()
        max_date = fold_data['Sampling_Date_parsed'].max()
        n_samples = len(fold_data)
        n_dates = fold_data[group_col].nunique()
        print(f"  Fold {fold}: {n_samples:3d} samples, {n_dates:2d} dates | {min_date.date()} to {max_date.date()}")
    
    return results


def organize_images_into_folds(
    df: pd.DataFrame,
    image_dir: Path,
    output_dir: Path,
    n_splits: int,
    use_symlinks: bool = True
) -> None:
    """
    Organize images into fold directories with train/test structure.
    
    Args:
        df: DataFrame with 'fold' column and 'image_path'
        image_dir: Directory containing the original images
        output_dir: Directory where fold directories will be created
        n_splits: Number of folds
        use_symlinks: If True, create symlinks instead of copying files (saves space)
    """
    print(f"Image source: {image_dir}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save fold assignments CSV
    csv_output = output_dir / 'fold_assignments.csv'
    df.to_csv(csv_output, index=False)
    print(f"Saved fold assignments to: {csv_output}")
    
    # Organize images for each fold
    total_operations = 0
    for fold in range(n_splits):
        # Create fold directory structure
        fold_dir = output_dir / f"fold{fold}"
        train_dir = fold_dir / "train"
        test_dir = fold_dir / "test"
        
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Get test images (current fold)
        test_images = df[df['fold'] == fold]['image_path'].values
        
        # Get train images (all other folds)
        train_images = df[df['fold'] != fold]['image_path'].values
        
        # Copy/symlink test images
        for img_path in tqdm(test_images, desc=f"  Fold {fold} test", leave=False):
            src = image_dir / img_path
            dst = test_dir / Path(img_path).name
            
            if src.exists():
                if use_symlinks:
                    # Create symlink (relative path for portability)
                    if not dst.exists():
                        dst.symlink_to(src.resolve())
                else:
                    # Copy file
                    if not dst.exists():
                        shutil.copy2(src, dst)
                total_operations += 1
            else:
                print(f"    ⚠️ Warning: Image not found: {src}")
        
        # Copy/symlink train images
        for img_path in tqdm(train_images, desc=f"  Fold {fold} train", leave=False):
            src = image_dir / img_path
            dst = train_dir / Path(img_path).name
            
            if src.exists():
                if use_symlinks:
                    # Create symlink
                    if not dst.exists():
                        dst.symlink_to(src.resolve())
                else:
                    # Copy file
                    if not dst.exists():
                        shutil.copy2(src, dst)
                total_operations += 1
            else:
                print(f"Warning: Image not found: {src}")
        

def main():
    parser = argparse.ArgumentParser(
        description='Create K-fold splits with no data leakage (groups by date). '
                    'By default, uses StratifiedGroupKFold with discretized regression target (Dry_Total_g) '
                    'to ensure balanced target distribution. Use --no-stratify for simple GroupKFold.'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='../data/csiro-biomass/train_wide.csv',
        help='Path to input train_wide.csv (wide format)'
    )
    parser.add_argument(
        '--image-dir',
        type=str,
        default='../data/csiro-biomass',
        help='Directory containing the original images (default: ../data/csiro-biomass)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='../data/csiro-biomass/folds',
        help='Output directory for fold directories (default: ../data/csiro-biomass/folds)'
    )
    parser.add_argument(
        '--n-splits',
        type=int,
        default=5,
        help='Number of folds (default: 5) - FIXED for reproducibility'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42) - FIXED to ensure everyone gets same splits'
    )
    parser.add_argument(
        '--group-col',
        type=str,
        default='Sampling_Date',
        help='Column to group by to prevent data leakage (default: Sampling_Date)'
    )
    parser.add_argument(
        '--target-col',
        type=str,
        default='Dry_Total_g',
        help='Regression target column to discretize for stratification (default: Dry_Total_g)'
    )
    parser.add_argument(
        '--n-target-bins',
        type=int,
        default=5,
        help='Number of bins to discretize the target into (default: 5)'
    )
    parser.add_argument(
        '--bin-strategy',
        type=str,
        choices=['quantile', 'uniform'],
        default='quantile',
        help='Binning strategy: quantile (equal-frequency) or uniform (equal-width) (default: quantile)'
    )
    parser.add_argument(
        '--stratify-cols',
        type=str,
        nargs='+',
        default=None,
        help='Additional columns to stratify by (combined with target bins). Example: --stratify-cols State Species'
    )
    parser.add_argument(
        '--no-stratify',
        action='store_true',
        default=False,
        help='Disable stratification entirely; use simple GroupKFold (group by date only)'
    )
    parser.add_argument(
        '--use-symlinks',
        action='store_true',
        default=True,
        help='Use symlinks instead of copying files (default: True, saves space)'
    )
    parser.add_argument(
        '--copy-files',
        action='store_true',
        help='Copy files instead of using symlinks (overrides --use-symlinks)'
    )
    
    args = parser.parse_args()
    
    # Determine whether to use symlinks or copy files
    use_symlinks = not args.copy_files  # If --copy-files is set, use_symlinks is False
    
    # Convert to absolute paths if relative
    script_dir = Path(__file__).parent
    input_path = script_dir / args.input if not os.path.isabs(args.input) else Path(args.input)
    image_dir = script_dir / args.image_dir if not os.path.isabs(args.image_dir) else Path(args.image_dir)
    output_dir = script_dir / args.output_dir if not os.path.isabs(args.output_dir) else Path(args.output_dir)
    
    print(f"\nReading data from: {input_path}")
    train_wide = pd.read_csv(input_path)
    print(f"Data shape: {train_wide.shape}")
    
    # Create folds
    train_with_folds = create_folds(
        train_wide,
        n_splits=args.n_splits,
        random_state=args.random_state,
        group_col=args.group_col,
        target_col=args.target_col,
        n_target_bins=args.n_target_bins,
        bin_strategy=args.bin_strategy,
        stratify_cols=args.stratify_cols,
        no_stratify=args.no_stratify
    )
    
    # Verify splits
    verification_results = verify_splits(
        train_with_folds, 
        group_col=args.group_col,
        target_col=args.target_col
    )
    
    # Organize images into fold directories
    organize_images_into_folds(
        df=train_with_folds,
        image_dir=image_dir,
        output_dir=output_dir,
        n_splits=args.n_splits,
        use_symlinks=use_symlinks
    )


if __name__ == "__main__":
    main()


# # Default: StratifiedGroupKFold with discretized Dry_Total_g (5 quantile bins)
# python data_split.py

# # Simple GroupKFold - just group by date, no stratification
# python data_split.py --no-stratify

# # StratifiedGroupKFold with more bins
# python data_split.py --n-target-bins 10

# # StratifiedGroupKFold + additional stratification by State/Species
# python data_split.py --stratify-cols State Species    

