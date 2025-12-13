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