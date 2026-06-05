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
# Run inference for one subject (reads cached feature parquets):
MFISH_DATA_ROOT=/data roi-classifier predict 790322

# Train production models (LOSO cross-validation):
roi-classifier train

# Train on specific subjects with a custom label log:
roi-classifier train --label-log /path/to/roi_qc_actions.jsonl 790322 788406
```

## Python API

```python
from roi_classifier.roi_quality_v5d import predict_subject
df = predict_subject("790322")
# → DataFrame with hcr_id, binary_score, proba_bad, proba_bad_ok, proba_good, proba_merged
```
