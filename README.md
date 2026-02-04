# CSIRO-Image2BiomassPrediction (Kaggle)

Our solution for the **CSIRO Image2BiomassPrediction** Kaggle competition: a **DINO-based multi-tile regressor** with **physics-aware post-processing** and an optional **ensemble** with public models + classic ML on embeddings.

**Result:** peaked at **156 / 3803 (silver zone)** during the competition; after the final evaluation shake-up finished **279 / 3803 (bronze medal)**.

---

## Key ideas

- **Leakage-resistant CV:** **Stratified-Group K-Fold** grouped by **sampling date**
- **Metric:** **Weighted R²** (per-row weights depend on target type)
- **Backbone:** **DINOv2 (ViT)** family features (consistently outperformed other backbones)
- **Spatial modeling:** split image into **2 halves**, create **4 tiles per half**, extract features per tile, then **average** tiles
- **Feature recipe:** concatenate **CLS + mean patch tokens** from **blocks 17 & 23**, then fuse left/right halves
- **Regularization:** **embedding mixup**
- **Semantics:** concatenate with **SigLIP** embeddings (optional)
- **Heads:** predict **3 primary targets** → derive the rest with constraints:
  - Predict: `Total`, `GDM`, `Green`
  - Derive: `Dead = Total - GDM`, `Clover = GDM - Green`
- **Post-processing:** enforce **physical realism** (non-negativity, ordering constraints)

---

## Training

- Loss: **Huber**
- Optimizer: **Adam**
- Scheduler: **Cosine**
- Early stopping on CV
- Two-stage training:
  1. **Heads-only** warm-up (frozen backbone)
  2. Full training (or partial unfreeze depending on experiment)
- Head MLP: `dropout=0.3`, `hidden_ratio=0.25`

Augmentations used:
- horizontal/vertical flips, rotations, color jitter
- ImageNet normalization

---

## Inference pipeline (high level)

1. image → **split into left/right halves**
2. each half → **4 tiles**
3. each tile → DINO features (**CLS + mean** from blocks **17 & 23**)
4. average features across tiles (per half)
5. concat halves → (optional) **mixup embeddings during training**
6. concat with **SigLIP** semantic features (optional)
7. 3 regression heads → derive clover/dead → **post-process**

---

## Ensemble (final boost)

Best scoring submission used an ensemble of:
- my DINO-based model
- a **public DINOv3** model
- **LightGBM / CatBoost** trained on **SigLIP embeddings**

---

## What didn’t help (selected experiments)

- domain pretraining (PlantCLEF / Irish grass)
- alternative pooling (CLS/mean/max combos, GeM across layers)
- attention between blocks (17/23), complex fusion (FiLM/gating/quality-weighted)
- auxiliary metadata losses
- sliding-window aggregation
- aggressive augmentations and large TTA sets
- adaptive unfreezing schedules beyond the simple two-stage approach

---

## Repo usage

This repo contains training/inference code and notebooks used to reproduce the approach.

Typical workflow:
- configure CV split + paths
- train folds
- run inference
- (optional) ensemble and submit

> Note: paths/config names depend on your local setup and Kaggle dataset layout.

---

## License

MIT

---

## Citation / Acknowledgements

- Kaggle CSIRO Image2BiomassPrediction competition organizers and community
- DINO / SigLIP authors and open-source ecosystem (PyTorch, timm, albumentations)
