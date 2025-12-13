#!/usr/bin/env bash

# ─── USAGE ────────────────────────────────────────────────────────────────
# ./kaggle/update_models_fromlocal.sh ./checkpoints/exp1 my_ensemble
# ./kaggle/update_models_fromlocal.sh ./checkpoints/exp1 my_ensemble --delete-local
# ./kaggle/update_models_fromlocal.sh ./checkpoints/exp1 my_ensemble --clean-kaggle
# ./kaggle/update_models_fromlocal.sh ./checkpoints/exp1 my_ensemble --delete-local --clean-kaggle

# ─── ARGUMENT CHECK ───────────────────────────────────────────────────────
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <LOCAL_CHECKPOINT_DIR> <ENSEMBLE_NAME> [OPTIONS]"
    echo ""
    echo "Arguments:"
    echo "  LOCAL_CHECKPOINT_DIR  - Path to folder containing best_model_fold0-4.ckpt files"
    echo "  ENSEMBLE_NAME        - Name for the ensemble folder in Kaggle dataset"
    echo ""
    echo "Options:"
    echo "  --delete-local       - Delete local checkpoint files after upload"
    echo "  --clean-kaggle       - Clean local Kaggle dataset directory after upload"
    exit 1
fi

# ─── CONFIG ────────────────────────────────────────────────────────────────
LOCAL_CHECKPOINT_DIR="$1"
ENSEMBLE_NAME="$2"
DELETE_LOCAL=false
CLEAN_KAGGLE=false

shift 2
while [ "$#" -gt 0 ]; do
    case "$1" in
        --delete-local)
            DELETE_LOCAL=true
            ;;
        --clean-kaggle)
            CLEAN_KAGGLE=true
            ;;
        *)
            echo "❌ Error: Unknown option '$1'"
            exit 1
            ;;
    esac
    shift
done

BASE_DIR="data/kaggle_datasets"
DATASET_NAME="csiro-image2biomassprediction-models"
MAIN_FOLDER="main_folder"

KAGGLE_DATASET_NAME="ivanbashtovyi/csiro-image2biomassprediction-models"
KAGGLE_DEST_DIR="${BASE_DIR}/${DATASET_NAME}/${MAIN_FOLDER}"

# ─── FOLD CONFIG (SINGLE SOURCE OF TRUTH) ─────────────────────────────────
N_FOLDS=5
FIRST_FOLD=0
LAST_FOLD=$((N_FOLDS - 1))

# ─── 1) VALIDATE LOCAL CHECKPOINTS ────────────────────────────────────────
echo ">>> Validating local checkpoint directory"

if [ ! -d "$LOCAL_CHECKPOINT_DIR" ]; then
    echo "❌ Error: Directory '$LOCAL_CHECKPOINT_DIR' does not exist"
    exit 1
fi

MISSING_FOLDS=()

for ((fold=FIRST_FOLD; fold<=LAST_FOLD; fold++)); do
    checkpoint_path="${LOCAL_CHECKPOINT_DIR}/best_model_fold${fold}.ckpt"
    if [ ! -f "$checkpoint_path" ]; then
        MISSING_FOLDS+=("fold${fold}")
    fi
done

if [ ${#MISSING_FOLDS[@]} -ne 0 ]; then
    echo "❌ Error: Missing checkpoint files for: ${MISSING_FOLDS[*]}"
    echo "Expected files: best_model_fold0.ckpt ... best_model_fold4.ckpt"
    exit 1
fi

echo "✅ Found all ${N_FOLDS} checkpoint files"

# ─── 2) DOWNLOAD EXISTING KAGGLE DATASET (ONLY IF EMPTY) ───────────────────
echo ">>> Checking local Kaggle dataset directory"

if [ ! -d "$KAGGLE_DEST_DIR" ] || [ -z "$(ls -A "$KAGGLE_DEST_DIR" 2>/dev/null)" ]; then
    echo ">>> Local Kaggle dataset is empty, downloading from Kaggle"
    mkdir -p "$KAGGLE_DEST_DIR"

    kaggle datasets download \
        -d "$KAGGLE_DATASET_NAME" \
        -p "$KAGGLE_DEST_DIR" \
        --unzip \
        --force

    echo "✅ Downloaded existing dataset to $KAGGLE_DEST_DIR"
else
    echo "✅ Local Kaggle dataset already exists, skipping download"
fi

# ─── 3) CREATE ENSEMBLE FOLDER AND COPY CHECKPOINTS ────────────────────────
ENSEMBLE_DIR="${KAGGLE_DEST_DIR}/${ENSEMBLE_NAME}"

echo ">>> Creating ensemble folder: $ENSEMBLE_NAME"

if [ -d "$ENSEMBLE_DIR" ]; then
    echo "❌ Error: Ensemble folder '$ENSEMBLE_NAME' already exists. Aborting to prevent overwrite."
    exit 1
fi

mkdir -p "$ENSEMBLE_DIR"

echo ">>> Copying checkpoints to ensemble folder"
for ((fold=FIRST_FOLD; fold<=LAST_FOLD; fold++)); do
    cp "${LOCAL_CHECKPOINT_DIR}/best_model_fold${fold}.ckpt" "$ENSEMBLE_DIR/"
    echo "  ✓ Copied best_model_fold${fold}.ckpt"
done

# ─── 4) CREATE ENSEMBLE INFO FILE ─────────────────────────────────────────
echo ">>> Creating ensemble metadata"

cat > "${ENSEMBLE_DIR}/ensemble_info.txt" <<EOF
Ensemble Name: ${ENSEMBLE_NAME}
Number of Folds: ${N_FOLDS}
Source Directory: ${LOCAL_CHECKPOINT_DIR}
Upload Date: $(date '+%Y-%m-%d %H:%M:%S')

Checkpoint Files:
$(for ((fold=FIRST_FOLD; fold<=LAST_FOLD; fold++)); do
    echo "  - best_model_fold${fold}.ckpt"
done)
EOF

echo "✅ Created ensemble with ${N_FOLDS} checkpoints"

# ─── 5) UPLOAD TO KAGGLE ──────────────────────────────────────────────────
cd "${BASE_DIR}" || exit 1

echo ">>> Uploading updated dataset to Kaggle"
kaggle datasets version \
    -p "${DATASET_NAME}" \
    -m "Add ensemble: ${ENSEMBLE_NAME}" \
    --dir-mode zip

echo "✅ Dataset uploaded to Kaggle"

# ─── 6) CLEANUP ───────────────────────────────────────────────────────────
if [ "$CLEAN_KAGGLE" = true ]; then
    echo ">>> Cleaning up local Kaggle directory (--clean-kaggle flag set)"
    rm -rf "${DATASET_NAME}/${MAIN_FOLDER:?}/"*
    echo "✅ Cleaned local Kaggle directory"
fi

if [ "$DELETE_LOCAL" = true ]; then
    echo ">>> Deleting local checkpoint directory (--delete-local flag set)"
    rm -rf "${LOCAL_CHECKPOINT_DIR:?}"
    echo "✅ Deleted $LOCAL_CHECKPOINT_DIR"
fi

# ─── DONE ────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "✅ Done! Ensemble '$ENSEMBLE_NAME' uploaded to Kaggle successfully"
echo "════════════════════════════════════════════════════════════════"
echo "Ensemble contains:"
echo "  - ${N_FOLDS} checkpoint files (best_model_fold0.ckpt through best_model_fold4.ckpt)"
echo ""
if [ "$DELETE_LOCAL" = true ]; then
    echo "✓ Local checkpoint directory has been deleted"
else
    echo "✓ Local checkpoint directory preserved at: $LOCAL_CHECKPOINT_DIR"
fi
if [ "$CLEAN_KAGGLE" = true ]; then
    echo "✓ Local Kaggle dataset directory has been cleaned"
else
    echo "✓ Local Kaggle dataset directory preserved for reuse"
fi
echo "════════════════════════════════════════════════════════════════"