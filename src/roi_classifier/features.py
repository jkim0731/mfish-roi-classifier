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


def _read_group_parquet(sid: str, suffix: str) -> pd.DataFrame:
    """Read a per-group parquet from ROI_QUALITY_DIR; raise if missing."""
    p = _CACHE_DIR / f"{sid}_features_{suffix}.parquet"
    if not p.exists():
        raise FileNotFoundError(
            f"[{sid}] group parquet missing: {p}\n"
            f"Run the corresponding feat_* module to generate it, or point "
            f"MFISH_ROI_QUALITY_DIR at a directory that contains it."
        )
    return pd.read_parquet(p)


def extract_features(sid: str, cache: bool = True) -> pd.DataFrame:
    """Merge the four cached per-group parquets and return the 91-col _um feature matrix.

    Parameters
    ----------
    sid   : subject ID string (e.g. "790322").
    cache : if True (default), reads from the per-group parquets already on
            disk.  Pass cache=False only if you need to force re-extraction via
            the feat_* modules; the merged frame is never persisted by this
            function.

    Returns
    -------
    DataFrame with columns ``["hcr_id"] + FEATURE_COLUMNS`` (92 total).
    Row count equals the number of ROIs for the subject.

    Raises
    ------
    FileNotFoundError  if a required per-group parquet is absent.
    ValueError         if a required feature column is missing after merge.
    """
    if not cache:
        from .benchmark_data_loader import load_subject
        from . import feat_shape, feat_axis, feat_surface, feat_protrusion
        s = load_subject(sid)
        f = feat_shape.compute(s, cache=False)
        g = feat_axis.compute(s, cache=False)
        h = feat_surface.compute(s, cache=False)
        k = feat_protrusion.compute(s, cache=False)
    else:
        f = _read_group_parquet(sid, "v2")
        g = _read_group_parquet(sid, "v3_extra")
        h = _read_group_parquet(sid, "v4")
        k = _read_group_parquet(sid, "v5")

    n_ref = len(f)
    for name, df in [("v3_extra", g), ("v4", h), ("v5", k)]:
        if len(df) != n_ref:
            print(
                f"  [{sid}] WARNING: {name} has {len(df)} rows vs v2 {n_ref} rows — "
                f"merge will be on hcr_id (left join)"
            )

    # v4 also contains volume_um3_raw; the _v4 suffix produces
    # volume_um3_raw_v4, which is one of the 91 _um model features.
    out = f.merge(g, on="hcr_id", how="left", suffixes=("", "_v3"))
    out = out.merge(h, on="hcr_id", how="left", suffixes=("", "_v4"))
    out = out.merge(k, on="hcr_id", how="left", suffixes=("", "_v5d"))

    # Validate that all 91 _um feature columns are present.
    missing = [c for c in FEATURE_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"[{sid}] missing feature columns after merge: {missing}")

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
