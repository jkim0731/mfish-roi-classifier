"""Per-ROI v4-NEW features (surface area + sphericity + calibrated core/shell).

Design:
- Reads per-cell padded mask crops from `cached_per_cell_crops/{sid}_per_cell_crops.parquet`
  (no zarr re-read for seg).
- Reads c405 zarr ONCE per z-strip (using v2 STRIP_Z=128, Z_PAD=24) and slices
  per-cell windows from the strip block — same logic as v2/v3 but with a
  cheap raw uint16 zarr instead of the labelled seg zarr.
- Recomputes binary opening per-cell on the padded crop (cheap on small crops).
- Calls `roi_v4_features.all_v4_features` on the tight crops.

Output: `cached_roi_quality/{sid}_features_v4.parquet` with columns
        `hcr_id` + `roi_v4_features.feature_columns()`.
"""
from __future__ import annotations

import json
import time
import warnings
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.ndimage as ndi
import zarr

warnings.filterwarnings("ignore", category=UserWarning, module="zarr")

from .benchmark_data_loader import load_subject
from .feat_shape import (
    OPENING_RADIUS, STRIP_Z, Z_PAD, _CROSS_3D, _ch405_l2, _orig_res_path,
)
from .roi_v4_features import all_v4_features, feature_columns, R_CORE_UM_DEFAULT
from . import config as _cfg

PER_CELL_CROPS = _cfg.PER_CELL_CROPS_DIR
ROI_QUALITY_CACHE = _cfg.ROI_QUALITY_DIR
TIGHT_BBOX_CACHE = _cfg.TIGHT_BBOX_DIR


def _features_v4_cache_path(sid: str) -> Path:
    return ROI_QUALITY_CACHE / f"{sid}_features_v4.parquet"


def _meta_v4_path(sid: str) -> Path:
    return ROI_QUALITY_CACHE / f"{sid}_meta_v4.json"


def _decode_mask(row: pd.Series) -> np.ndarray:
    n = int(row["dz"]) * int(row["dy"]) * int(row["dx"])
    bits = np.unpackbits(np.frombuffer(row["mask_packed"], dtype=np.uint8))[:n]
    return bits.astype(bool).reshape(int(row["dz"]), int(row["dy"]), int(row["dx"]))


def _extract_v4_subject(
    sid: str, r_core_um: float = R_CORE_UM_DEFAULT, force: bool = False,
) -> Dict:
    out_path = _features_v4_cache_path(sid)
    if out_path.exists() and not force:
        return {"sid": sid, "skipped": "cache_hit", "path": str(out_path)}

    crops_path = PER_CELL_CROPS / f"{sid}_per_cell_crops.parquet"
    if not crops_path.exists():
        return {"sid": sid, "error": f"missing crops: {crops_path} "
                f"(build with `roi-classifier build-crops {sid}`)"}

    tb_path = TIGHT_BBOX_CACHE / f"{sid}_hcr_cell_tight_bbox_v1.parquet"
    if not tb_path.exists():
        return {"sid": sid, "error": f"missing tight bbox cache: {tb_path} "
                f"(build with `roi-classifier build-bbox {sid}`)"}

    s = load_subject(sid)
    seg_orig = zarr.open(str(_orig_res_path(s)), mode="r")
    _, _, Z_seg, Y_seg, X_seg = seg_orig.shape
    vz = float(s.hcr_z_um)
    vxy = float(s.hcr_xy_um)

    arr405 = _ch405_l2(s)
    has_405 = arr405 is not None

    crops_df = pd.read_parquet(crops_path)
    tb_df = pd.read_parquet(tb_path)
    tb_lookup = {
        int(r.hcr_id): (
            int(r.zmin_vox), int(r.zmax_vox),
            int(r.ymin_vox), int(r.ymax_vox),
            int(r.xmin_vox), int(r.xmax_vox),
        )
        for r in tb_df.itertuples(index=False)
    }

    # Bucket cache rows by their strip bucket using crop origin z0_lvl2.
    cent = s.hcr_centroids.set_index("hcr_id")
    if not s.hcr_centroids.empty:
        z_lo_global = max(0, int(s.hcr_centroids["z_px"].min()) - 2)
    else:
        z_lo_global = 0
    cent_z = {}
    for hid in crops_df["hcr_id"].astype(int).tolist():
        if hid in cent.index:
            cent_z[hid] = int(round(float(cent.loc[hid]["z_px"])))

    crops_df["_strip"] = crops_df["hcr_id"].map(
        lambda h: z_lo_global + ((cent_z.get(int(h), 0) - z_lo_global) // STRIP_Z) * STRIP_Z
    )

    feature_rows: List[Dict] = []
    n_done = 0
    n_missing = 0
    t0 = time.time()
    strip_keys = sorted(crops_df["_strip"].unique())

    for s_idx, z0_inner in enumerate(strip_keys):
        sub = crops_df[crops_df["_strip"] == z0_inner]
        if len(sub) == 0:
            continue
        z1_inner = min(z0_inner + STRIP_Z, Z_seg)
        z0_load = max(0, z0_inner - Z_PAD)
        z1_load = min(Z_seg, z1_inner + Z_PAD)
        y_min = int(sub["y0_lvl2"].min())
        y_max = int((sub["y0_lvl2"] + sub["dy"]).max())
        x_min = int(sub["x0_lvl2"].min())
        x_max = int((sub["x0_lvl2"] + sub["dx"]).max())
        sub_y0 = max(0, y_min)
        sub_y1 = min(Y_seg, y_max)
        sub_x0 = max(0, x_min)
        sub_x1 = min(X_seg, x_max)

        if has_405:
            ch405_block = np.asarray(
                arr405[0, 0, z0_load:z1_load, sub_y0:sub_y1, sub_x0:sub_x1]
            ).astype(np.uint16)
        else:
            ch405_block = None

        for _, row in sub.iterrows():
            hid = int(row["hcr_id"])
            mask_pad = _decode_mask(row)
            if not mask_pad.any():
                n_missing += 1
                continue

            tb = tb_lookup.get(hid)
            if tb is None:
                n_missing += 1
                continue
            zmin, zmax_ex, ymin, ymax_ex, xmin, xmax_ex = tb
            tz0 = zmin - int(row["z0_lvl2"])
            ty0 = ymin - int(row["y0_lvl2"])
            tx0 = xmin - int(row["x0_lvl2"])
            tz1 = tz0 + (zmax_ex - zmin)
            ty1 = ty0 + (ymax_ex - ymin)
            tx1 = tx0 + (xmax_ex - xmin)
            if (tz0 < 0 or ty0 < 0 or tx0 < 0
                    or tz1 > mask_pad.shape[0]
                    or ty1 > mask_pad.shape[1]
                    or tx1 > mask_pad.shape[2]):
                n_missing += 1
                continue

            mask_raw_tight = mask_pad[tz0:tz1, ty0:ty1, tx0:tx1]
            mask_opened_pad = ndi.binary_opening(
                mask_pad, structure=_CROSS_3D, iterations=OPENING_RADIUS,
            )
            mask_opened_tight = mask_opened_pad[tz0:tz1, ty0:ty1, tx0:tx1]

            if has_405:
                bz0 = int(row["z0_lvl2"]) - z0_load
                bz1 = bz0 + int(row["dz"])
                by0 = int(row["y0_lvl2"]) - sub_y0
                by1 = by0 + int(row["dy"])
                bx0 = int(row["x0_lvl2"]) - sub_x0
                bx1 = bx0 + int(row["dx"])
                if (bz0 < 0 or by0 < 0 or bx0 < 0
                        or bz1 > ch405_block.shape[0]
                        or by1 > ch405_block.shape[1]
                        or bx1 > ch405_block.shape[2]):
                    img_tight = None
                else:
                    img_pad = ch405_block[bz0:bz1, by0:by1, bx0:bx1]
                    img_tight = img_pad[tz0:tz1, ty0:ty1, tx0:tx1]
            else:
                img_tight = None

            feats = all_v4_features(
                mask_raw_tight, mask_opened_tight, img_tight,
                vz, vxy, vxy, r_core_um=r_core_um,
            )
            feats["hcr_id"] = hid
            feature_rows.append(feats)
            n_done += 1

        if (s_idx + 1) % max(1, len(strip_keys) // 5) == 0:
            print(
                f"  [{sid}] strip {s_idx+1}/{len(strip_keys)}  "
                f"done={n_done}  missing={n_missing}  "
                f"elapsed={time.time()-t0:.0f}s",
                flush=True,
            )

    cols = feature_columns()
    nan_template = {c: float("nan") for c in cols}
    seen = {int(r["hcr_id"]) for r in feature_rows}
    for hid in s.hcr_centroids["hcr_id"].astype(int).tolist():
        if int(hid) in seen:
            continue
        miss = dict(nan_template)
        miss["hcr_id"] = int(hid)
        feature_rows.append(miss)

    feat_df = pd.DataFrame(feature_rows).sort_values("hcr_id").reset_index(drop=True)
    feat_df = feat_df[["hcr_id"] + cols]
    feat_df.to_parquet(out_path, index=False)

    elapsed = time.time() - t0
    meta = {
        "subject_id": sid,
        "version": "v4",
        "channels_used": ["405"] if has_405 else [],
        "total_rois": int(len(feat_df)),
        "n_extracted": int(n_done),
        "n_missing": int(n_missing),
        "extraction_timestamp": datetime.utcnow().isoformat() + "Z",
        "seg_xy_um": vxy,
        "seg_z_um": vz,
        "r_core_um": float(r_core_um),
        "opening_radius_voxels": OPENING_RADIUS,
        "strip_z": STRIP_Z,
        "z_pad": Z_PAD,
        "elapsed_seconds": float(elapsed),
    }
    with open(_meta_v4_path(sid), "w") as f:
        json.dump(meta, f, indent=2)

    sz = out_path.stat().st_size
    print(
        f"[{sid}] DONE: {n_done} cells, {n_missing} missing, "
        f"{elapsed:.0f}s, {sz/1e6:.1f} MB", flush=True,
    )
    return {
        "sid": sid,
        "n_done": int(n_done),
        "n_missing": int(n_missing),
        "elapsed_s": float(elapsed),
        "path": str(out_path),
    }


def extract_v4_all(
    subjects: List[str],
    workers: int = 6,
    r_core_um: float = R_CORE_UM_DEFAULT,
    force: bool = False,
) -> List[Dict]:
    ctx = get_context("spawn")
    args = [(sid, r_core_um, force) for sid in subjects]
    with ctx.Pool(processes=min(workers, len(subjects))) as pool:
        results = pool.starmap(_extract_v4_subject, args)
    return results


# Public entry point used by features.extract_features.
def compute(s, cache: bool = True) -> "pd.DataFrame":
    """Compute (or load from cache) surface-area + sphericity + core/shell features.

    `s` is a SubjectData instance (or any object with `.subject_id`).
    Returns the cached parquet without re-extraction if it exists and cache=True.
    """
    from .benchmark_data_loader import SubjectData  # local import avoids circular dep
    sid = s.subject_id if hasattr(s, "subject_id") else str(s)
    out_path = _features_v4_cache_path(sid)
    if cache and out_path.exists():
        return pd.read_parquet(out_path)
    result = _extract_v4_subject(sid, force=True)
    if "error" in result:
        raise RuntimeError(f"[{sid}] feat_surface extract failed: {result['error']}")
    return pd.read_parquet(out_path)
