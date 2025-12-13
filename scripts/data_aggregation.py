#!/usr/bin/env python3
import os
import argparse
import pandas as pd
from pathlib import Path


def aggregate_train_data(input_path: str, output_path: str = None) -> pd.DataFrame:
    """
    Aggregate train.csv from long format to wide format.
    
    Args:
        input_path: Path to the original train.csv (long format)
        output_path: Optional path to save the wide format CSV
    
    Returns:
        DataFrame in wide format with one row per image
    """
    print(f"Reading data from: {input_path}")
    train_df = pd.read_csv(input_path)
    
    print(f"Original shape (long format): {train_df.shape}")
    print(f"Number of unique samples: {train_df['sample_id'].str.split('__').str[0].nunique()}")
    
    # Extract the actual image ID from sample_id (remove the target suffix)
    train_df['image_id'] = train_df['sample_id'].str.split('__').str[0]
    
    # Pivot the dataframe so each image has one row with 5 target columns
    train_wide = train_df.pivot_table(
        index=['image_id', 'image_path', 'Sampling_Date', 'State', 'Species', 'Pre_GSHH_NDVI', 'Height_Ave_cm'],
        columns='target_name',
        values='target',
        aggfunc='first'
    ).reset_index()
    
    # Flatten the column names (remove the multi-level structure)
    train_wide.columns.name = None
    
    print(f"New shape (wide format): {train_wide.shape}")
    print(f"\nTarget columns created: {[col for col in train_wide.columns if col.endswith('_g')]}")
    
    # Verify we have all 5 target columns
    expected_targets = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
    missing_targets = set(expected_targets) - set(train_wide.columns)
    if missing_targets:
        print(f"WARNING: Missing target columns: {missing_targets}")
    
    # Save if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        train_wide.to_csv(output_path, index=False)
        print(f"Saved wide format data to: {output_path}")
    
    return train_wide


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate train.csv from long format to wide format'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='../data/csiro-biomass/train.csv',
        help='Path to input train.csv (long format)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='../data/csiro-biomass/train_wide.csv',
        help='Path to output train_wide.csv (wide format)'
    )
    
    args = parser.parse_args()
    
    # Convert to absolute paths if relative
    script_dir = Path(__file__).parent
    input_path = script_dir / args.input if not os.path.isabs(args.input) else Path(args.input)
    output_path = script_dir / args.output if not os.path.isabs(args.output) else Path(args.output)
    
    # Run aggregation
    train_wide = aggregate_train_data(str(input_path), str(output_path))

if __name__ == "__main__":
    main()

