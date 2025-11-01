#!/bin/bash
set -e

DATA_DIR="data/csiro-biomass"
COMPETITION_NAME="csiro-biomass"

mkdir -p "$DATA_DIR"
kaggle competitions download -c $COMPETITION_NAME -p "$DATA_DIR"
unzip -o "$DATA_DIR"/$COMPETITION_NAME.zip -d "$DATA_DIR"
rm "$DATA_DIR"/$COMPETITION_NAME.zip
echo "Dataset downloaded, extracted, and cleaned up in '$DATA_DIR'"