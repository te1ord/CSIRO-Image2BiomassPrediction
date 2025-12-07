#!/usr/bin/env bash

# Check if required arguments are provided
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "Usage: $0 <LOCAL_CHECKPOINT_DIR> <ENSEMBLE_NAME> [--delete-local]"
    echo "Example: $0 ./checkpoints/experiment_1 exp1_ensemble --delete-local"
    echo ""
    echo "Arguments:"
    echo "  LOCAL_CHECKPOINT_DIR  - Path to folder containing best_model_fold{1-5}.ckpt files"
    echo "  ENSEMBLE_NAME        - Name for the ensemble folder in Kaggle dataset"
    echo "  --delete-local       - Optional: delete local files after upload"
    exit 1
fi

# ─── CONFIG ────────────────────────────────────────────────────────────────
LOCAL_CHECKPOINT_DIR="$1"
ENSEMBLE_NAME="$2"
DELETE_LOCAL=false

# Check for --delete-local flag
if [ "$#" -eq 3 ] && [ "$3" == "--delete-local" ]; then
    DELETE_LOCAL=true
fi

BASE_DIR="data/kaggle_datasets"
DATASET_NAME="csiro-image2biomassprediction-models"
MAIN_FOLDER="main_folder"

# Kaggle config
KAGGLE_DATASET_NAME="ivanbashtovyi/csiro-image2biomassprediction-models"
KAGGLE_DEST_DIR="${BASE_DIR}/${DATASET_NAME}/${MAIN_FOLDER}"

# ─── 1) VALIDATE LOCAL CHECKPOINTS ────────────────────────────────────────
echo ">>> Validating local checkpoint directory"

if [ ! -d "$LOCAL_CHECKPOINT_DIR" ]; then
    echo "❌ Error: Directory '$LOCAL_CHECKPOINT_DIR' does not exist"
    exit 1
fi

# Check for all 5 fold checkpoints
N_FOLDS=5
MISSING_FOLDS=()

for fold in $(seq 1 $N_FOLDS); do
    checkpoint_path="${LOCAL_CHECKPOINT_DIR}/best_model_fold${fold}-v2.ckpt"
    if [ ! -f "$checkpoint_path" ]; then
        MISSING_FOLDS+=("fold${fold}")
    fi
done

if [ ${#MISSING_FOLDS[@]} -ne 0 ]; then
    echo "❌ Error: Missing checkpoint files for: ${MISSING_FOLDS[*]}"
    echo "Expected files: best_model_fold1.ckpt, best_model_fold2.ckpt, ..., best_model_fold5.ckpt"
    exit 1
fi

echo "✅ Found all 5 checkpoint files"

# ─── 2) DOWNLOAD EXISTING KAGGLE DATASET ──────────────────────────────────
echo ">>> Downloading existing Kaggle dataset"
rm -rf "$KAGGLE_DEST_DIR"
mkdir -p "$KAGGLE_DEST_DIR"

kaggle datasets download \
  -d "$KAGGLE_DATASET_NAME" \
  -p "$KAGGLE_DEST_DIR" \
  --unzip \
  --force

echo "✅ Downloaded existing dataset to $KAGGLE_DEST_DIR"

# ─── 3) CREATE ENSEMBLE FOLDER AND COPY CHECKPOINTS ───────────────────────
ENSEMBLE_DIR="${KAGGLE_DEST_DIR}/${ENSEMBLE_NAME}"

echo ">>> Creating ensemble folder: $ENSEMBLE_NAME"

if [ -d "$ENSEMBLE_DIR" ]; then
    echo "⚠️  Warning: Ensemble folder '$ENSEMBLE_NAME' already exists, will be overwritten"
    rm -rf "$ENSEMBLE_DIR"
fi

mkdir -p "$ENSEMBLE_DIR"

# Copy all 5 checkpoints
echo ">>> Copying checkpoints to ensemble folder"
for fold in $(seq 1 $N_FOLDS); do
    checkpoint_path="${LOCAL_CHECKPOINT_DIR}/best_model_fold${fold}-v2.ckpt"
    cp "$checkpoint_path" "$ENSEMBLE_DIR/"
    echo "  ✓ Copied best_model_fold${fold}-v2.ckpt"
done

# ─── 4) CREATE ENSEMBLE INFO FILE ─────────────────────────────────────────
echo ">>> Creating ensemble metadata"
cat > "${ENSEMBLE_DIR}/ensemble_info.txt" <<EOF
Ensemble Name: ${ENSEMBLE_NAME}
Number of Folds: ${N_FOLDS}
Source Directory: ${LOCAL_CHECKPOINT_DIR}
Upload Date: $(date '+%Y-%m-%d %H:%M:%S')

Checkpoint Files:
$(for fold in $(seq 1 $N_FOLDS); do echo "  - best_model_fold${fold}-v2.ckpt"; done)
EOF

echo "✅ Created ensemble with $N_FOLDS checkpoints"

# ─── 5) UPLOAD TO KAGGLE ──────────────────────────────────────────────────
cd "${BASE_DIR}"

echo ">>> Uploading updated dataset to Kaggle"
kaggle datasets version \
  -p "${DATASET_NAME}" \
  -m "Add ensemble: ${ENSEMBLE_NAME}" \
  --dir-mode zip

echo "✅ Dataset uploaded to Kaggle"

# ─── 6) CLEANUP ───────────────────────────────────────────────────────────
# echo ">>> Cleaning up local Kaggle directory"
# CLEAN_PATH="${DATASET_NAME}/${MAIN_FOLDER}"
# rm -rf "${CLEAN_PATH:?}/"*

# if [ "$DELETE_LOCAL" = true ]; then
#     echo ">>> Deleting local checkpoint directory (--delete-local flag set)"
#     rm -rf "${LOCAL_CHECKPOINT_DIR:?}"
#     echo "✅ Deleted $LOCAL_CHECKPOINT_DIR"
# fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "✅ Done! Ensemble '$ENSEMBLE_NAME' uploaded to Kaggle successfully"
echo "════════════════════════════════════════════════════════════════"
echo "Ensemble contains:"
echo "  - 5 checkpoint files (best_model_fold1.ckpt through best_model_fold5.ckpt)"
echo "  - ensemble_info.txt metadata file"
echo ""
if [ "$DELETE_LOCAL" = true ]; then
    echo "Local checkpoint directory has been deleted"
else
    echo "Local checkpoint directory preserved at: $LOCAL_CHECKPOINT_DIR"
    echo "To delete it next time, add --delete-local flag"
fi
echo "════════════════════════════════════════════════════════════════"