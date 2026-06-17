# mfish-roi-classifier

v5d ROI-quality classifier for HCR cell segmentations in the mFISH coregistration pipeline.

## Install

```bash
pip install -e .              # core (inference + training)
pip install -e ".[label]"     # adds the labelling GUI (matplotlib + ipywidgets)
```

## Configuration

All paths are controlled by environment variables (see `src/roi_classifier/config.py`):

| Variable | Default | Purpose |
|---|---|---|
| `MFISH_DATA_ROOT` | `/root/capsule/data` | Root of subject data tree |
| `MFISH_CACHE_DIR` | `~/.cache/mfish-roi-classifier` | Root for regenerable caches |
| `MFISH_ROI_QUALITY_DIR` | `$MFISH_CACHE_DIR/roi_quality` | Feature parquets + contract output |
| `MFISH_TIGHT_BBOX_DIR` | `$MFISH_CACHE_DIR/hcr_cell_tight_bbox` | Per-cell tight-bbox parquets |
| `MFISH_PER_CELL_CROPS_DIR` | `$MFISH_CACHE_DIR/per_cell_crops` | 3-D crop arrays |
| `MFISH_MODELS_DIR` | `$MFISH_ROI_QUALITY_DIR` | Trained LightGBM .txt model files |

## Cross-repo data contract

This package writes the contract parquet consumed by `mfish-autocoreg`:

```
{MFISH_ROI_QUALITY_DIR}/{sid}_stage2_4class_proba_v5d_um.parquet
```

Columns: `hcr_id`, `p_bad`, `p_bad_ok`, `p_good`, `p_merged` (and `human_label` if labels exist).

`mfish-autocoreg` reads this file and keeps cells where `argmax ∈ {p_good, p_bad_ok}`.

## Usage

```bash
# Cold-start a fresh subject: build the tight-bbox cache + all 4 feature-group
# parquets (reads the level-2 segmentation_mask_orig_res.zarr), then predict:
roi-classifier build-features 790322
roi-classifier predict 790322

# Just the tight-bbox cache (the cold-start prerequisite the extractors read):
roi-classifier build-bbox 790322          # --force to rebuild

# Run inference when the feature parquets already exist:
MFISH_DATA_ROOT=/data roi-classifier predict 790322

# Train production models (LOSO cross-validation):
roi-classifier train

# Train on specific subjects with a custom label log:
roi-classifier train --label-log /path/to/roi_qc_actions.jsonl 790322 788406
```

### Cold-start data flow

`predict` only **reads** the per-group feature parquets; it does not build them.
For a subject with nothing cached, run `build-features` first. That chain is:

```
build-bbox      → {sid}_hcr_cell_tight_bbox_v1.parquet     (MFISH_TIGHT_BBOX_DIR)
  ↓ (read by the crops builder + shape/axis extractors)
build-crops     → {sid}_per_cell_crops.parquet             (MFISH_PER_CELL_CROPS_DIR)
  ↓ (read by the surface/v4 + protrusion/v5 extractors)
build-features  → {sid}_features_{v2,v3_extra,v4,v5}.parquet   (MFISH_ROI_QUALITY_DIR)
  ↓ (merged + scored)
predict         → {sid}_stage2_4class_proba_v5d_um.parquet     (contract output)
```

`build-features` runs the whole chain (bbox → crops → 4 feature groups). The
tight-bbox and crops caches are in the **level-2 voxel frame** (the `orig_res`
zarr the extractors index), per-subject, and read only during cold feature
extraction; once the feature parquets exist they are never read again.

> The shape/v2 extractor optionally reads a `{sid}_stage1_score.parquet` for its
> neighbour-adjacency features; if absent those columns are NaN (LightGBM handles
> NaN), so cold-start still completes — it is not a hard prerequisite.

### Parallel extraction

The shape (v2) and axis (v3) extractors parallelize their z-strip scan across
cores via `MFISH_FEAT_WORKERS` (default `cpu-2`). The fastest run axis when you
have a few subjects is **serialize subjects, parallelize within each** — run
`build-features` with `-j 1` (serial feature groups) and let each group's z-strips
use all the cores:

```bash
for sid in 790322 782149 767022; do
  MFISH_FEAT_WORKERS=14 roi-classifier build-features "$sid" -j 1
done
```

Do **not** combine a high `-j` (parallel feature groups) with a high
`MFISH_FEAT_WORKERS` (parallel strips) — the nested process pools over-subscribe
the CPU. Feature values are identical to the serial path (verified exact).

## Python API

```python
from roi_classifier.model import predict_subject
df = predict_subject("790322")
# → DataFrame with hcr_id, binary_score, proba_bad, proba_bad_ok,
#   proba_good, proba_merged, predicted_class
```

Lower-level entry points live in the same module: `predict(features_df)` returns
`(binary_score, four_class_proba)`, and `train(...)` runs the LOSO + production
training. Feature extraction is `from roi_classifier.features import extract_features`.
