"""Per-cell tight-bbox builder for the ROI-quality feature extractors.

The feature extractor (`feat_shape`) **reads** a per-cell
tight-bbox parquet (`{sid}_hcr_cell_tight_bbox.parquet` in
`MFISH_TIGHT_BBOX_DIR`) to drive their z-strip iteration without a full-volume
`find_objects`.  This module **builds** that parquet from the level-2
segmentation (`segmentation_mask_orig_res.zarr`).

Ported 2026-06-17 from the now-removed ``roi_quality.py``
(``extract_roi_features`` / ``_update_tight_bbox_cache``).  The refactor that
split feature extraction into the ``feat_*`` modules kept only the *readers* of
this cache; this restores the *writer* so a fresh subject cold-starts without a
hand-staged cache.

For each labelled cell (label id == ``hcr_id``) present in the subject's
centroid table, it computes — in the **level-2 voxel frame** (the same frame as
``hcr_centroids``) — the tight voxel bbox ``[zmin..zmax)`` (half-open), the voxel
count, and the voxel centroid.  The scan is z-stripped (``STRIP_Z`` slices at a
time) so the ~7 G-voxel volume never loads whole; cells that span strips are
accumulated across them.

Public API
----------
``build_tight_bbox(s, cache=True) -> pd.DataFrame``
    One row per cell found in the segmentation.
``build_tight_bbox_sid(sid, cache=True) -> pd.DataFrame``
    Convenience wrapper that loads the subject first.

Schema (exactly what ``feat_shape`` / ``feat_axis`` / ``feat_surface`` read)::

    hcr_id, zmin_vox, ymin_vox, xmin_vox, zmax_vox, ymax_vox, xmax_vox,
    volume_vox, zc_vox, yc_vox, xc_vox
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import zarr
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module="zarr")

from . import config as _cfg
from .benchmark_data_loader import SubjectData, load_subject

TIGHT_BBOX_DIR = _cfg.TIGHT_BBOX_DIR

# z-strip height in slices (matches feat_shape.STRIP_Z so the build and the
# readers stripe the volume identically).
STRIP_Z = 128

_BBOX_COLS = [
    "hcr_id",
    "zmin_vox", "ymin_vox", "xmin_vox",
    "zmax_vox", "ymax_vox", "xmax_vox",
    "volume_vox", "zc_vox", "yc_vox", "xc_vox",
]


def _orig_res_path(s: SubjectData) -> Path:
    """``segmentation_mask_orig_res.zarr`` — the level-2 (centroid) frame."""
    p = s.hcr_dir / "cell_body_segmentation" / "segmentation_mask_orig_res.zarr"
    if not p.exists():
        raise FileNotFoundError(f"orig_res zarr not found: {p}")
    return p


def tight_bbox_cache_path(sid: str) -> Path:
    return TIGHT_BBOX_DIR / f"{sid}_hcr_cell_tight_bbox.parquet"


def build_tight_bbox(s: SubjectData, cache: bool = True, force: bool = False) -> pd.DataFrame:
    """Build (or load) the per-cell tight-bbox parquet for subject ``s``.

    Parameters
    ----------
    s : SubjectData
    cache : if True (default), persist the freshly-built parquet to
        ``MFISH_TIGHT_BBOX_DIR``.  Pass ``cache=False`` to compute in memory
        without writing.
    force : if True, rebuild even when a cached parquet already exists
        (default False = return the cache if present).

    Returns
    -------
    DataFrame with columns ``_BBOX_COLS``, one row per cell found in the
    segmentation, sorted by ``hcr_id``.
    """
    sid = s.subject_id
    out_path = tight_bbox_cache_path(sid)
    if not force and out_path.exists():
        print(f"  [{sid}] loading cached tight-bbox from {out_path}")
        return pd.read_parquet(out_path)

    t0 = time.time()
    seg_orig = zarr.open(str(_orig_res_path(s)), mode="r")
    _, _, Z_seg, Y_seg, X_seg = seg_orig.shape

    if s.hcr_centroids is None or s.hcr_centroids.empty:
        raise ValueError(f"[{sid}] no hcr_centroids — cannot scope the tight-bbox scan")
    all_hids_set = set(s.hcr_centroids["hcr_id"].astype(int).tolist())

    # Scope the scan to the centroid z-range (+/- 2 px), as the original did.
    z_lo_global = max(0, int(s.hcr_centroids["z_px"].min()) - 2)
    z_hi_global = min(Z_seg, int(s.hcr_centroids["z_px"].max()) + 2)
    n_strips = (z_hi_global - z_lo_global + STRIP_Z - 1) // STRIP_Z
    print(f"[{sid}] tight-bbox build | seg (Z,Y,X)=({Z_seg},{Y_seg},{X_seg}) | "
          f"{len(all_hids_set)} centroids | z-strips [{z_lo_global},{z_hi_global}) "
          f"n={n_strips}")

    # hid -> running min/max/sum accumulators (level-2 voxel coords).
    tb_acc: Dict[int, Dict] = {}

    for z0 in tqdm(range(z_lo_global, z_hi_global, STRIP_Z),
                   desc=f"[{sid}] tbbox z-strips", unit="strip"):
        z1 = min(z0 + STRIP_Z, z_hi_global)
        seg_strip = np.asarray(seg_orig[0, 0, z0:z1, :, :])  # (dz, Y, X)

        # One C-level scan returns a tight slice per label present in the strip.
        label_slices = ndi.find_objects(seg_strip)
        for lbl_idx, sl in enumerate(label_slices):
            if sl is None:
                continue
            hid = lbl_idx + 1  # find_objects is 0-indexed for 1-indexed labels
            if hid not in all_hids_set:
                continue

            sl_z, sl_y, sl_x = sl
            mask_tight = (seg_strip[sl] == hid)
            n_pxs = int(mask_tight.sum())
            if n_pxs == 0:
                continue

            zc_local, yc_local, xc_local = np.where(mask_tight)
            z_glob = zc_local + z0 + sl_z.start  # global z (strip offset + slice)
            y_glob = yc_local + sl_y.start
            x_glob = xc_local + sl_x.start

            if hid not in tb_acc:
                tb_acc[hid] = {
                    "zmin": int(z_glob.min()), "zmax": int(z_glob.max()),
                    "ymin": int(y_glob.min()), "ymax": int(y_glob.max()),
                    "xmin": int(x_glob.min()), "xmax": int(x_glob.max()),
                    "volume_vox": n_pxs,
                    "zsum": float(z_glob.sum()),
                    "ysum": float(y_glob.sum()),
                    "xsum": float(x_glob.sum()),
                }
            else:
                tb = tb_acc[hid]
                tb["zmin"] = min(tb["zmin"], int(z_glob.min()))
                tb["zmax"] = max(tb["zmax"], int(z_glob.max()))
                tb["ymin"] = min(tb["ymin"], int(y_glob.min()))
                tb["ymax"] = max(tb["ymax"], int(y_glob.max()))
                tb["xmin"] = min(tb["xmin"], int(x_glob.min()))
                tb["xmax"] = max(tb["xmax"], int(x_glob.max()))
                tb["volume_vox"] += n_pxs
                tb["zsum"] += float(z_glob.sum())
                tb["ysum"] += float(y_glob.sum())
                tb["xsum"] += float(x_glob.sum())

    rows: List[Dict] = []
    for hid, tb in tb_acc.items():
        n = tb["volume_vox"]
        rows.append({
            "hcr_id": int(hid),
            "zmin_vox": int(tb["zmin"]),
            "ymin_vox": int(tb["ymin"]),
            "xmin_vox": int(tb["xmin"]),
            "zmax_vox": int(tb["zmax"]) + 1,  # half-open
            "ymax_vox": int(tb["ymax"]) + 1,
            "xmax_vox": int(tb["xmax"]) + 1,
            "volume_vox": int(n),
            "zc_vox": float(tb["zsum"]) / n,
            "yc_vox": float(tb["ysum"]) / n,
            "xc_vox": float(tb["xsum"]) / n,
        })
    df = (pd.DataFrame(rows, columns=_BBOX_COLS)
          .sort_values("hcr_id").reset_index(drop=True))
    print(f"  [{sid}] tight-bbox: {len(df)}/{len(all_hids_set)} cells found in seg "
          f"in {time.time() - t0:.1f}s")

    if cache:
        TIGHT_BBOX_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        print(f"  [{sid}] tight-bbox cache written → {out_path}")
    return df


def build_tight_bbox_sid(sid: str, cache: bool = True, force: bool = False) -> pd.DataFrame:
    """Load subject ``sid`` and build (or load) its tight-bbox parquet."""
    return build_tight_bbox(load_subject(sid), cache=cache, force=force)
