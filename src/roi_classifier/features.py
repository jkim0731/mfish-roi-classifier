"""Public feature-extraction entry point for the v5d _um ROI-quality classifier.

Single call: ``extract_features(sid, cache=True) -> DataFrame``

Merges four per-group feature parquets (shape/v2, axis/v3_extra, surface/v4,
protrusion/v5) into the 91-column µm-variant frame that ``model.predict`` and
``model.train`` consume.  No v6_vox features.  No percentile-rank columns.

``volume_um3_raw_v4`` in the target set is a merge-suffix artefact: both v2
and v4 contain ``volume_um3_raw``; when v4 is merged with suffix ``"_v4"`` on
collision the v4 copy becomes ``volume_um3_raw_v4`` automatically.

Cache strategy
--------------
Each group module writes its own parquet to ``config.ROI_QUALITY_DIR`` (keyed
``{sid}_features_v{2,3_extra,4,5}.parquet``).  When ``cache=True`` and those
parquets already exist the per-group computation is skipped entirely.
``extract_features`` itself does NOT write a merged parquet.

Coordinate frame
----------------
All features use level-2 voxels (``segmentation_mask_orig_res.zarr``).
``hcr_centroids`` from ``SubjectData`` are also in the level-2 frame.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config as _cfg
from .model import FEATURE_COLUMNS  # noqa: F401 — re-export for convenience

# Cache dir for reading pre-existing per-group parquets.
_CACHE_DIR = _cfg.ROI_QUALITY_DIR


def extract_features(sid: str, cache: bool = True) -> pd.DataFrame:
    """Return the 91-column µm feature matrix for one subject.

    A single unified extractor (`feat_shape.compute`) computes all feature
    families in one z-strip pass — each cell's mask, opening, and 405 crop are
    computed once and fed to the shape/axis/surface/protrusion math — and writes
    one parquet, ``{sid}_features_all.parquet``.

    Parameters
    ----------
    sid   : subject ID string (e.g. "790322").
    cache : if True (default), read the cached ``{sid}_features_all.parquet``.
            If False, run the extractor now (does not write).

    Returns
    -------
    DataFrame with columns ``["hcr_id"] + FEATURE_COLUMNS`` (92 total).
    """
    if cache:
        p = _CACHE_DIR / f"{sid}_features_all.parquet"
        if not p.exists():
            raise FileNotFoundError(
                f"[{sid}] feature parquet missing: {p}\n"
                f"Build it with `roi-classifier build-features {sid}` (or point "
                f"MFISH_ROI_QUALITY_DIR at a directory that contains it)."
            )
        out = pd.read_parquet(p)
    else:
        from .benchmark_data_loader import load_subject
        from . import feat_shape
        out = feat_shape.compute(load_subject(sid), cache=False)

    missing = [c for c in FEATURE_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"[{sid}] missing feature columns: {missing}")

    out = out[["hcr_id"] + FEATURE_COLUMNS].copy()

    nan_counts = out[FEATURE_COLUMNS].isna().sum()
    cols_with_nan = nan_counts[nan_counts > 0]
    if len(cols_with_nan) > 0:
        print(
            f"  [{sid}] NaN counts in feature columns (expected for some intensity "
            f"cols on empty masks):"
        )
        for col, cnt in cols_with_nan.items():
            print(f"    {col}: {cnt}")

    print(f"  [{sid}] extract_features: {len(out)} ROIs, {len(FEATURE_COLUMNS)} features")
    return out
