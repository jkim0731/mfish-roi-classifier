"""Per-ROI v6 voxel-companion features for the v5d expansion-rate test.

For every µm-derived v4/v3 feature that has no voxel/unitless companion, recompute
the same feature with identity calibration `(vz, vy, vx) = (1, 1, 1)` and
`r_core_um = 4` (= 4 voxels) / `bin_um = 1` (= 1 voxel). The outputs are renamed
from `*_um*` to `*_vox*`.

Output: `cached_roi_quality/{sid}_features_v6_vox.parquet` with columns
        `hcr_id` + the voxel-companion feature names.
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
import scipy.ndimage as ndi
import zarr
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore", category=UserWarning, module="zarr")

from .benchmark_data_loader import load_subject
from .roi_quality_v2 import (
    OPENING_RADIUS, STRIP_Z, Z_PAD, _CROSS_3D, _ch405_l2, _orig_res_path,
)
from .roi_v4_features import all_v4_features
from .roi_v3_axis_features import all_v3_axis_features
from . import config as _cfg

PER_CELL_CROPS = _cfg.PER_CELL_CROPS_DIR
ROI_QUALITY_CACHE = _cfg.ROI_QUALITY_DIR
TIGHT_BBOX_CACHE = _cfg.TIGHT_BBOX_DIR

R_CORE_VOX = 4.0
BIN_VOX = 1.0
NEIGHBOR_RADIUS_VOX = 30.0

# v4 features that are µm-typed and need a vox-renamed companion. Volume_vox*
# and sphericity_*/sav are already in the merged frame (sphericity is unitless,
# volume_vox already exists), so we only emit the truly missing ones.
_V4_RENAME = {
    "surface_area_um2_raw":             "surface_area_vox_raw",
    "surface_area_um2_opened":          "surface_area_vox_opened",
    "sa_to_vol_um_inv_raw":             "sa_to_vol_vox_inv_raw",
    "sa_to_vol_um_inv_opened":          "sa_to_vol_vox_inv_opened",
    "core4um_voxel_frac_opened":        "core4vox_voxel_frac_opened",
    "c405_core4um_p50_opened":          "c405_core4vox_p50_opened",
    "c405_shell4um_p50_opened":         "c405_shell4vox_p50_opened",
    "c405_shell_minus_core4um_p50":     "c405_shell_minus_core4vox_p50",
    "c405_shell_minus_core4um_p90":     "c405_shell_minus_core4vox_p90",
    "c405_shell_over_core4um_p50_ratio": "c405_shell_over_core4vox_p50_ratio",
}

# v3 axis/projection features that are µm-typed.
_V3_RENAME = {
    "axis3d_extent_um":              "axis3d_extent_vox",
    "axis3d_lambda1_um2":            "axis3d_lambda1_vox2",
    "axis3d_lambda2_um2":            "axis3d_lambda2_vox2",
    "axis3d_lambda3_um2":            "axis3d_lambda3_vox2",
    "axis3d_peak_sep_um_intens":     "axis3d_peak_sep_vox_intens",
    "axis3d_raw_extent_um":          "axis3d_raw_extent_vox",
    "proj_xy_main_extent_um":        "proj_xy_main_extent_vox",
    "proj_yz_main_extent_um":        "proj_yz_main_extent_vox",
    "proj_zx_main_extent_um":        "proj_zx_main_extent_vox",
    "proj_xy_orth_fwhm_um":          "proj_xy_orth_fwhm_vox",
    "proj_yz_orth_fwhm_um":          "proj_yz_orth_fwhm_vox",
    "proj_zx_orth_fwhm_um":          "proj_zx_orth_fwhm_vox",
}

# bbox extents + equivalent diameters in voxel units (computed directly).
_EXTRA_COLS = [
    "bbox_z_extent_vox",
    "bbox_y_extent_vox",
    "bbox_x_extent_vox",
    "equivalent_diameter_vox_raw",
    "equivalent_diameter_vox_opened",
    "n_neighbors_30vox",
]


def feature_columns() -> list[str]:
    return list(_V4_RENAME.values()) + list(_V3_RENAME.values()) + _EXTRA_COLS


def _features_v6_path(sid: str) -> Path:
    return ROI_QUALITY_CACHE / f"{sid}_features_v6_vox.parquet"


def _meta_v6_path(sid: str) -> Path:
    return ROI_QUALITY_CACHE / f"{sid}_meta_v6_vox.json"


def _decode_mask(row: pd.Series) -> np.ndarray:
    n = int(row["dz"]) * int(row["dy"]) * int(row["dx"])
    bits = np.unpackbits(np.frombuffer(row["mask_packed"], dtype=np.uint8))[:n]
    return bits.astype(bool).reshape(int(row["dz"]), int(row["dy"]), int(row["dx"]))


def _equivalent_diameter_vox(volume_vox: float) -> float:
    if not np.isfinite(volume_vox) or volume_vox <= 0:
        return float("nan")
    return 2.0 * (3.0 * volume_vox / (4.0 * np.pi)) ** (1.0 / 3.0)


def _extract_v6_subject(sid: str, force: bool = False) -> Dict:
    out_path = _features_v6_path(sid)
    if out_path.exists() and not force:
        return {"sid": sid, "skipped": "cache_hit", "path": str(out_path)}

    crops_path = PER_CELL_CROPS / f"{sid}_per_cell_crops.parquet"
    if not crops_path.exists():
        return {"sid": sid, "error": f"missing crops: {crops_path}"}
    tb_path = TIGHT_BBOX_CACHE / f"{sid}_hcr_cell_tight_bbox_v1.parquet"
    if not tb_path.exists():
        return {"sid": sid, "error": f"missing tight bbox cache: {tb_path}"}

    s = load_subject(sid)
    seg_orig = zarr.open(str(_orig_res_path(s)), mode="r")
    _, _, Z_seg, Y_seg, X_seg = seg_orig.shape

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

    cent = s.hcr_centroids.set_index("hcr_id")
    if not s.hcr_centroids.empty:
        z_lo_global = max(0, int(s.hcr_centroids["z_px"].min()) - 2)
    else:
        z_lo_global = 0
    cent_z: Dict[int, int] = {}
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
            ).astype(np.float32)
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

            if has_405 and ch405_block is not None:
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
                    img_tight = ch405_block[bz0:bz1, by0:by1, bx0:bx1][
                        tz0:tz1, ty0:ty1, tx0:tx1
                    ]
            else:
                img_tight = None

            v4_feats = all_v4_features(
                mask_raw_tight, mask_opened_tight, img_tight,
                vz=1.0, vy=1.0, vx=1.0, r_core_um=R_CORE_VOX,
            )
            v3_feats = all_v3_axis_features(
                mask_opened_tight, mask_raw_tight, img_tight,
                vz=1.0, vy=1.0, vx=1.0, bin_um=BIN_VOX,
            )

            row_out: Dict[str, float] = {"hcr_id": hid}
            for src, dst in _V4_RENAME.items():
                row_out[dst] = float(v4_feats.get(src, float("nan")))
            for src, dst in _V3_RENAME.items():
                row_out[dst] = float(v3_feats.get(src, float("nan")))
            row_out["bbox_z_extent_vox"] = float(zmax_ex - zmin)
            row_out["bbox_y_extent_vox"] = float(ymax_ex - ymin)
            row_out["bbox_x_extent_vox"] = float(xmax_ex - xmin)
            vol_raw = float(mask_raw_tight.sum())
            vol_op = float(mask_opened_tight.sum())
            row_out["equivalent_diameter_vox_raw"] = _equivalent_diameter_vox(vol_raw)
            row_out["equivalent_diameter_vox_opened"] = _equivalent_diameter_vox(vol_op)
            feature_rows.append(row_out)
            n_done += 1

        if (s_idx + 1) % max(1, len(strip_keys) // 5) == 0:
            print(
                f"  [{sid}] strip {s_idx+1}/{len(strip_keys)}  "
                f"done={n_done}  missing={n_missing}  "
                f"elapsed={time.time()-t0:.0f}s",
                flush=True,
            )

    # n_neighbors_30vox: count cells with centroid within 30 voxels in
    # anisotropic voxel coords (z, y, x). NOTE: this uses voxel-units of the
    # level-2 grid directly — no µm calibration. Matches v2's n_neighbors_30um
    # in shape but in voxel space.
    neigh_map: Dict[int, int] = {}
    if not s.hcr_centroids.empty:
        cent_arr = s.hcr_centroids[["z_px", "y_px", "x_px"]].to_numpy(float)
        ids = s.hcr_centroids["hcr_id"].astype(int).to_numpy()
        tree = cKDTree(cent_arr)
        for hid, p in zip(ids, cent_arr):
            if not np.isfinite(p).all():
                neigh_map[int(hid)] = 0
                continue
            counts = tree.query_ball_point(p, r=NEIGHBOR_RADIUS_VOX, return_length=True)
            neigh_map[int(hid)] = int(max(0, counts - 1))

    seen_ids = {int(r["hcr_id"]) for r in feature_rows}
    nan_template = {c: float("nan") for c in feature_columns()}
    for hid in s.hcr_centroids["hcr_id"].astype(int).tolist():
        if int(hid) in seen_ids:
            continue
        miss = dict(nan_template)
        miss["hcr_id"] = int(hid)
        feature_rows.append(miss)

    for r in feature_rows:
        r["n_neighbors_30vox"] = neigh_map.get(int(r["hcr_id"]), 0)

    cols = feature_columns()
    feat_df = pd.DataFrame(feature_rows).sort_values("hcr_id").reset_index(drop=True)
    feat_df = feat_df[["hcr_id"] + cols]
    feat_df.to_parquet(out_path, index=False)

    elapsed = time.time() - t0
    meta = {
        "subject_id": sid,
        "version": "v6_vox",
        "total_rois": int(len(feat_df)),
        "n_extracted": int(n_done),
        "n_missing": int(n_missing),
        "extraction_timestamp": datetime.utcnow().isoformat() + "Z",
        "r_core_vox": float(R_CORE_VOX),
        "bin_vox": float(BIN_VOX),
        "neighbor_radius_vox": float(NEIGHBOR_RADIUS_VOX),
        "opening_radius_voxels": OPENING_RADIUS,
        "strip_z": STRIP_Z,
        "z_pad": Z_PAD,
        "elapsed_seconds": float(elapsed),
    }
    with open(_meta_v6_path(sid), "w") as f:
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


def extract_v6_all(subjects: List[str], workers: int = 6, force: bool = False
                   ) -> List[Dict]:
    ctx = get_context("spawn")
    args = [(sid, force) for sid in subjects]
    with ctx.Pool(processes=min(workers, len(subjects))) as pool:
        results = pool.starmap(_extract_v6_subject, args)
    return results
