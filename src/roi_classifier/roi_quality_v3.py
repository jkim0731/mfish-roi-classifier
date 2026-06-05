"""Per-ROI v3 extras for the HCR ROI pass/fail classifier (S11).

This module ONLY computes v3-NEW features (axis profile, 2D projections,
waist).  v2 features stay in `cached_roi_quality/{sid}_features_v2.parquet`
and the trainer concats v2 + v3-extras side-by-side.

Why a separate file: v2 took ~2 h to extract across 6 subjects; we don't
want to redo it every time we add features.

Output: `cached_roi_quality/{sid}_features_v3_extra.parquet` with columns
        `hcr_id` + the names returned by `roi_v3_axis_features.feature_columns()`.
"""
from __future__ import annotations

import json
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import zarr
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module="zarr")

from .benchmark_data_loader import SubjectData
from .roi_quality_v2 import (
    OPENING_RADIUS, STRIP_Z, Z_PAD, _CROSS_3D,
    _ch405_l2, _orig_res_path,
)
from .roi_v3_axis_features import all_v3_axis_features, feature_columns
from . import config as _cfg

ROI_QUALITY_CACHE = _cfg.ROI_QUALITY_DIR
TIGHT_BBOX_CACHE = _cfg.TIGHT_BBOX_DIR


def _features_v3_extra_cache_path(sid: str) -> Path:
    return ROI_QUALITY_CACHE / f"{sid}_features_v3_extra.parquet"


def _meta_v3_extra_path(sid: str) -> Path:
    return ROI_QUALITY_CACHE / f"{sid}_meta_v3_extra.json"


def extract_v3_extras(
    s: SubjectData,
    cache: bool = True,
    bin_um: float = 1.0,
    compute_dropped_peaks: bool = False,
) -> pd.DataFrame:
    """Compute v3-NEW per-ROI features (axis profile + 2D projections + waist)."""
    sid = s.subject_id
    out_path = _features_v3_extra_cache_path(sid)
    if cache and out_path.exists():
        print(f"  [{sid}] loading cached v3-extras from {out_path}")
        return pd.read_parquet(out_path)

    t_start = time.time()

    seg_orig = zarr.open(str(_orig_res_path(s)), mode="r")
    _, _, Z_seg, Y_seg, X_seg = seg_orig.shape
    seg_xy_um = float(s.hcr_xy_um)
    seg_z_um = float(s.hcr_z_um)
    print(f"\n[{sid}] v3-extras | seg shape Z={Z_seg} Y={Y_seg} X={X_seg} | "
          f"xy={seg_xy_um:.4f}µm z={seg_z_um:.4f}µm  bin_um={bin_um}")

    arr405 = _ch405_l2(s)
    has_405 = arr405 is not None
    if not has_405:
        print(f"  [{sid}] WARNING: 405 channel missing — intensity profiles will be NaN")

    # centroid lookup
    cent = s.hcr_centroids.set_index("hcr_id")
    all_hids = s.hcr_centroids["hcr_id"].astype(int).tolist()
    cent_z: Dict[int, int] = {}
    for hid in all_hids:
        if hid in cent.index:
            cent_z[hid] = int(round(float(cent.loc[hid]["z_px"])))

    # strip range
    if not s.hcr_centroids.empty:
        z_lo_global = max(0, int(s.hcr_centroids["z_px"].min()) - 2)
        z_hi_global = min(Z_seg, int(s.hcr_centroids["z_px"].max()) + 2)
    else:
        z_lo_global, z_hi_global = 0, Z_seg

    # tight bbox cache (level-2)
    tb_path = TIGHT_BBOX_CACHE / f"{sid}_hcr_cell_tight_bbox_v1.parquet"
    if not tb_path.exists():
        raise FileNotFoundError(
            f"tight bbox cache missing: {tb_path}. Run roi_quality.extract_roi_features first."
        )
    tb_df = pd.read_parquet(tb_path)
    tb_lookup: Dict[int, Tuple[int, int, int, int, int, int]] = {
        int(r.hcr_id): (
            int(r.zmin_vox), int(r.zmax_vox),
            int(r.ymin_vox), int(r.ymax_vox),
            int(r.xmin_vox), int(r.xmax_vox),
        )
        for r in tb_df.itertuples(index=False)
    }
    print(f"  [{sid}] tight bbox cache loaded: {len(tb_lookup)} cells")

    # bucket cells by strip
    cells_per_strip: Dict[int, List[int]] = {}
    for hid in all_hids:
        cz = cent_z.get(hid)
        if cz is None or not (z_lo_global <= cz < z_hi_global):
            continue
        bucket = z_lo_global + ((cz - z_lo_global) // STRIP_Z) * STRIP_Z
        cells_per_strip.setdefault(bucket, []).append(int(hid))

    feature_rows: List[Dict] = []
    n_owned = 0
    n_missing = 0
    z_strips = sorted(cells_per_strip.keys())
    for z0_inner in tqdm(z_strips, desc=f"[{sid}] v3-extras z-strips", unit="strip"):
        z1_inner = min(z0_inner + STRIP_Z, z_hi_global)
        z0_load = max(0, z0_inner - Z_PAD)
        z1_load = min(Z_seg, z1_inner + Z_PAD)

        owned = cells_per_strip[z0_inner]
        own_bbs = [tb_lookup.get(h) for h in owned if h in tb_lookup]
        if not own_bbs:
            continue
        y_min = min(b[2] for b in own_bbs)
        y_max = max(b[3] for b in own_bbs)
        x_min = min(b[4] for b in own_bbs)
        x_max = max(b[5] for b in own_bbs)
        pad = OPENING_RADIUS + 1
        sub_y0 = max(0, y_min - pad)
        sub_y1 = min(Y_seg, y_max + pad)
        sub_x0 = max(0, x_min - pad)
        sub_x1 = min(X_seg, x_max + pad)

        seg_block = np.asarray(
            seg_orig[0, 0, z0_load:z1_load, sub_y0:sub_y1, sub_x0:sub_x1]
        )
        if has_405:
            ch405_block = np.asarray(
                arr405[0, 0, z0_load:z1_load, sub_y0:sub_y1, sub_x0:sub_x1]
            ).astype(np.float32)
            if ch405_block.shape != seg_block.shape:
                ch405_block = ch405_block[: seg_block.shape[0],
                                           : seg_block.shape[1],
                                           : seg_block.shape[2]]
        else:
            ch405_block = None

        for hid in owned:
            tb = tb_lookup.get(hid)
            if tb is None:
                continue
            zmin, zmax_ex, ymin, ymax_ex, xmin, xmax_ex = tb
            n_owned += 1

            psz0_g = max(0, zmin - pad)
            psz1_g = min(Z_seg, zmax_ex + pad)
            psy0_g = max(0, ymin - pad)
            psy1_g = min(Y_seg, ymax_ex + pad)
            psx0_g = max(0, xmin - pad)
            psx1_g = min(X_seg, xmax_ex + pad)

            bz0 = psz0_g - z0_load
            bz1 = psz1_g - z0_load
            by0 = psy0_g - sub_y0
            by1 = psy1_g - sub_y0
            bx0 = psx0_g - sub_x0
            bx1 = psx1_g - sub_x0
            if (bz0 < 0 or bz1 > seg_block.shape[0]
                    or by0 < 0 or by1 > seg_block.shape[1]
                    or bx0 < 0 or bx1 > seg_block.shape[2]):
                n_missing += 1
                continue

            seg_crop = seg_block[bz0:bz1, by0:by1, bx0:bx1]
            mask_crop = seg_crop == hid
            if not mask_crop.any():
                n_missing += 1
                continue

            tz0 = zmin - psz0_g
            ty0 = ymin - psy0_g
            tx0 = xmin - psx0_g
            tz1 = tz0 + (zmax_ex - zmin)
            ty1 = ty0 + (ymax_ex - ymin)
            tx1 = tx0 + (xmax_ex - xmin)
            mask_raw_tight = mask_crop[tz0:tz1, ty0:ty1, tx0:tx1]
            mask_opened = ndi.binary_opening(
                mask_crop, structure=_CROSS_3D, iterations=OPENING_RADIUS,
            )
            mask_opened_tight = mask_opened[tz0:tz1, ty0:ty1, tx0:tx1]

            if ch405_block is not None:
                img_tight = ch405_block[bz0:bz1, by0:by1, bx0:bx1][tz0:tz1, ty0:ty1, tx0:tx1]
            else:
                img_tight = None

            feats = all_v3_axis_features(
                mask_raw_tight, mask_opened_tight, img_tight,
                seg_z_um, seg_xy_um, seg_xy_um, bin_um=bin_um,
                compute_dropped_peaks=compute_dropped_peaks,
            )
            feats["hcr_id"] = int(hid)
            feature_rows.append(feats)

    from .roi_v3_axis_features import _DROPPED_PEAK_COLS
    cols = feature_columns()
    nan_template = {
        c: (float("nan") if c in _DROPPED_PEAK_COLS else (0 if "n_peaks" in c else float("nan")))
        for c in cols
    }

    seen_set = {int(r["hcr_id"]) for r in feature_rows}
    for hid in all_hids:
        if int(hid) in seen_set:
            continue
        miss = dict(nan_template)
        miss["hcr_id"] = int(hid)
        feature_rows.append(miss)

    feat_df = pd.DataFrame(feature_rows).sort_values("hcr_id").reset_index(drop=True)
    feat_df = feat_df[["hcr_id"] + cols]

    elapsed = time.time() - t_start
    print(f"  [{sid}] v3-extras done: {len(feat_df)} ROIs in {elapsed:.1f}s "
          f"({elapsed/60:.1f}min)  owned={n_owned} missing={n_missing}")
    print(f"  [{sid}] v3-extras columns ({len(feat_df.columns)})")

    if cache:
        feat_df.to_parquet(out_path, index=False)
        print(f"  [{sid}] saved → {out_path}")

    meta = {
        "subject_id": sid,
        "version": "v3_extra",
        "channels_used": ["405"] if has_405 else [],
        "total_rois": int(len(feat_df)),
        "n_owned_in_strip_pass": int(n_owned),
        "n_missing": int(n_missing),
        "extraction_timestamp": datetime.utcnow().isoformat() + "Z",
        "seg_xy_um": seg_xy_um,
        "seg_z_um": seg_z_um,
        "bin_um": float(bin_um),
        "opening_radius_voxels": OPENING_RADIUS,
        "strip_z": STRIP_Z,
        "z_pad": Z_PAD,
        "elapsed_seconds": float(elapsed),
    }
    with open(_meta_v3_extra_path(sid), "w") as f:
        json.dump(meta, f, indent=2)
    return feat_df
