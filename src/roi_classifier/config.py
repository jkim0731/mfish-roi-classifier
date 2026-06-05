"""Centralised path configuration for mfish-roi-classifier.

All cache/data paths flow through here.  Override at runtime via env vars
so the package can run against any dataset tree without editing source.

Contract output (consumed by mfish-autocoreg):
    ROI_QUALITY_DIR / "{sid}_stage2_4class_proba_v5d_um.parquet"
    columns: hcr_id, p_bad, p_bad_ok, p_good, p_merged[, human_label]
"""
import os
from pathlib import Path

# Root of the subject data tree.
# Must contain per-subject coreg dirs (`{sid}*ctl-czstack-hcr-coreg_*`)
# and HCR processed dirs (`HCR_{sid}_*_processed_*`).
DATA_ROOT = Path(os.environ.get("MFISH_DATA_ROOT", "/root/capsule/data"))

# Root for all regenerable per-run caches.
CACHE_DIR = Path(
    os.environ.get(
        "MFISH_CACHE_DIR",
        str(Path.home() / ".cache" / "mfish-roi-classifier"),
    )
)

# Per-subject feature parquets, model text files, and contract output parquets.
ROI_QUALITY_DIR = Path(
    os.environ.get("MFISH_ROI_QUALITY_DIR", str(CACHE_DIR / "roi_quality"))
)

# Per-cell tight-bbox parquet cache (level-2 coordinates).
TIGHT_BBOX_DIR = Path(
    os.environ.get("MFISH_TIGHT_BBOX_DIR", str(CACHE_DIR / "hcr_cell_tight_bbox"))
)

# Per-cell 3-D crop arrays (heavy; several hundred MB for 6 subjects).
PER_CELL_CROPS_DIR = Path(
    os.environ.get("MFISH_PER_CELL_CROPS_DIR", str(CACHE_DIR / "per_cell_crops"))
)

# Directory that holds trained model text files when different from ROI_QUALITY_DIR.
# Defaults to ROI_QUALITY_DIR because that is where the train() function writes them.
MODELS_DIR = Path(
    os.environ.get("MFISH_MODELS_DIR", str(ROI_QUALITY_DIR))
)
