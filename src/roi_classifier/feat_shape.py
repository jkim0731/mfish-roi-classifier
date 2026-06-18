"""Per-ROI feature extractor v2 for the HCR ROI pass/fail classifier (S11).

Redesign (2026-04-30) based on user feedback that v1's channel
interpretation was wrong:
  - 405 is Rn28S (cytoplasmic ribosomal RNA, *bright in cytoplasm / dim in
    nucleus*).  Good cells therefore have a DIM CORE and a BRIGHT SHELL —
    the opposite polarity of v1's `c405_peak_inside`.
  - 488 is GFP, expressed in a subpopulation only.  Not a membrane marker.
    Brightness on 488 says nothing about ROI quality, so 488 / 514 / 561 /
    594 are dropped from intensity features.
  - Cell masks are post-expansion (≈2-3×) so a morphological opening with
    a 3-D ball SE of radius 3 voxels at level-2 cleans up most "process"
    bridges before shape / intensity stats are computed.
  - Adjacency to other masks (rim overlap with other hcr_ids) is informative
    — merged or fused cells tend to share large surfaces with neighbours.

Output schema (~42 features, all numeric or bool):

  Shape raw                : volume_vox_raw, volume_um3_raw,
                             bbox_{z,y,x}_extent_um,
                             aspect_{zy,zx,yx},
                             equivalent_diameter_um_raw,
                             solidity_raw, bbox_occupancy_raw,
                             boundary_touching
  Shape opened             : volume_vox_opened, frac_kept_opening,
                             solidity_opened, equivalent_diameter_um_opened,
                             n_components_after_opening
  405 raw                  : c405_raw_{mean,std,p10,p50,p90}
  405 opened               : c405_opened_{mean,std,p10,p50,p90}
  405 core vs shell        : c405_core_p50_opened, c405_shell_p50_opened,
                             c405_shell_minus_core_p50,
                             c405_shell_minus_core_p90
  405 inside vs outside    : c405_outside_p50,
                             c405_inside_minus_outside_p50,
                             c405_inside_minus_outside_p90
  Adjacency                : n_touching_neighbors,
                             surface_touching_frac,
                             top_neighbor_overlap_frac,
                             mean_touching_score_stage1,
                             min_touching_score_stage1
  Spatial                  : knn_d1, n_neighbors_30um
  Sanity                   : tight_bbox_in_pickle_bbox,
                             volume_pickle_minus_zarr_l2_eq

Coordinates / data sources are identical to v1: level-2
`segmentation_mask_orig_res.zarr` and `image_tile_fusing/fused/channel_405.zarr`
`["2"]`.  (Historical detail lived in the now-removed `roi_quality.py`.)

Strategy
--------
Z-strip pass with overlap.  Each strip [z0, z1) is loaded with `Z_PAD=20`
voxels of context above and below so morphology with r=3 ball SE is correct
for any cell whose centroid sits within [z0, z1).  Cells are "owned" by the
strip whose [z0, z1) contains their centroid; cells with z-extent half >
Z_PAD - r are flagged (rare; expect 0 in practice — the p99 cell z-extent is
about 30 voxels).

For each owned cell, we compute the raw + opened mask shape stats, the 405
intensity stats raw + opened + core/shell + inside/outside, and adjacency
(which other hcr_ids touch the cell's 1-vox dilated rim).
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
from scipy.spatial import cKDTree
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module="zarr")

from .benchmark_data_loader import SubjectData, load_subject
# Family feature math (leaf modules) — used by the unified single-pass extractor
# so each cell's mask/opening/405 crop is computed ONCE for all families.
from . import roi_v3_axis_features, roi_v4_features, roi_v5_features
from .roi_v3_axis_features import all_v3_axis_features
from .roi_v4_features import all_v4_features
from .roi_v5_features import protrusion_features
from . import config as _cfg


# ──────────────────────────────────────────────────────────────────────────────
# constants
# ──────────────────────────────────────────────────────────────────────────────

ROI_QUALITY_CACHE = _cfg.ROI_QUALITY_DIR
ROI_QUALITY_CACHE.mkdir(parents=True, exist_ok=True)

TIGHT_BBOX_CACHE = _cfg.TIGHT_BBOX_DIR

OPENING_RADIUS = 3            # SE radius (voxels at level-2). Implemented as
                              # `iterations=3` of a 6-connected cross — an
                              # octahedron of "radius" 3, ≥10× faster than a
                              # 7³ ball footprint and visually indistinguishable
                              # for r=3 process trimming.
EROSION_CORE_RADIUS = 1       # erosion of opened mask -> core
RIM_RADIUS = 1                # dilation of raw mask -> rim used for adjacency
                              # and inside-vs-outside intensity contrast

import os as _os
# z-strip height (default 128). NOTE: NOT freely tunable — lowering it puts more
# cells near strip boundaries, where cells with large z-extent overflow the
# smaller loaded block (z = STRIP_Z + 2·Z_PAD) and get skipped → different
# (NaN) features. Changing it requires raising Z_PAD accordingly to stay exact.
# The production value is 128 (what the _v5d_um model's features were built with).
STRIP_Z = int(_os.environ.get("MFISH_STRIP_Z", "128"))
Z_PAD = 24                    # half-context above/below each strip;
                              # must satisfy Z_PAD >= max half-z-extent + OPENING_RADIUS

KNN_KS = [1]
KNN_RADIUS_UM = 30.0

# Cross structuring element (6-connectivity) — used with `iterations=N` for
# an octahedral approximation of ball-N.
_CROSS_3D = ndi.generate_binary_structure(3, 1)


# ──────────────────────────────────────────────────────────────────────────────
# paths
# ──────────────────────────────────────────────────────────────────────────────

def _orig_res_path(s: SubjectData) -> Path:
    p = s.hcr_dir / "cell_body_segmentation" / "segmentation_mask_orig_res.zarr"
    if not p.exists():
        raise FileNotFoundError(f"orig_res zarr not found: {p}")
    return p



def _ch405_l2(s: SubjectData):
    p = s.hcr_dir / "image_tile_fusing" / "fused" / "channel_405.zarr"
    if not p.exists():
        return None
    try:
        z = zarr.open(str(p), mode="r")
        return z["2"] if "2" in z else None
    except Exception:
        return None


def _features_v2_cache_path(sid: str) -> Path:
    # Unified single-pass output: all 91-feature families in one parquet.
    return ROI_QUALITY_CACHE / f"{sid}_features_all.parquet"


def _meta_v2_path(sid: str) -> Path:
    return ROI_QUALITY_CACHE / f"{sid}_meta_v2.json"


def _stage1_score_path(sid: str) -> Path:
    return ROI_QUALITY_CACHE / f"{sid}_stage1_score.parquet"


# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────

def _percentile_or_nan(arr: np.ndarray, q: float) -> float:
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q))


def _shape_stats(
    binary: np.ndarray,
    seg_z_um: float,
    seg_xy_um: float,
) -> Dict[str, float]:
    """Volume (vox), aspect ratios, solidity, bbox_occupancy.

    binary is a tight crop (no padding).  µm columns and n_components are
    not produced — they were in DROP_UM_FEATURES / DROP_DEAD_FEATURES.
    """
    out = {
        "volume_vox": 0,
        "aspect_zy": float("nan"),
        "aspect_zx": float("nan"),
        "aspect_yx": float("nan"),
        "solidity": float("nan"),
        "bbox_occupancy": float("nan"),
        # µm outputs (restored for the production v5d_um model — were dropped
        # by DROP_UM_FEATURES; see the um-vs-vox decision record).
        "volume_um3": float("nan"),
        "bbox_z_extent_um": float("nan"),
        "bbox_y_extent_um": float("nan"),
        "bbox_x_extent_um": float("nan"),
        "equivalent_diameter_um": float("nan"),
    }
    n = int(binary.sum())
    if n == 0:
        return out
    out["volume_vox"] = n

    zz, yy, xx = np.where(binary)
    dz = int(zz.max() - zz.min() + 1)
    dy = int(yy.max() - yy.min() + 1)
    dx = int(xx.max() - xx.min() + 1)
    # aspect ratios use physical µm so they are scale-invariant
    bbox_z = dz * seg_z_um
    bbox_y = dy * seg_xy_um
    bbox_x = dx * seg_xy_um
    out["aspect_zy"] = bbox_z / max(bbox_y, 1e-9)
    out["aspect_zx"] = bbox_z / max(bbox_x, 1e-9)
    out["aspect_yx"] = bbox_y / max(bbox_x, 1e-9)
    bbox_vol = max(dz * dy * dx, 1)
    out["bbox_occupancy"] = n / bbox_vol

    out["bbox_z_extent_um"] = bbox_z
    out["bbox_y_extent_um"] = bbox_y
    out["bbox_x_extent_um"] = bbox_x
    volume_um3 = n * seg_z_um * seg_xy_um * seg_xy_um
    out["volume_um3"] = float(volume_um3)
    out["equivalent_diameter_um"] = float(
        2.0 * (3.0 * volume_um3 / (4.0 * np.pi)) ** (1.0 / 3.0)
    )

    if n > 7:
        try:
            from scipy.spatial import ConvexHull
            pts_um = np.column_stack([
                zz * seg_z_um,
                yy * seg_xy_um,
                xx * seg_xy_um,
            ])
            if pts_um.shape[0] >= 4:
                hull = ConvexHull(pts_um)
                hv = float(hull.volume)
                if hv > 0:
                    out["solidity"] = float(volume_um3 / hv)
        except Exception:
            pass

    return out


def _intensity_stats(pxs: np.ndarray) -> Dict[str, float]:
    if pxs.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "p10": float("nan"),
            "p50": float("nan"),
            "p90": float("nan"),
        }
    return {
        "mean": float(pxs.mean()),
        "std": float(pxs.std()),
        "p10": float(np.percentile(pxs, 10)),
        "p50": float(np.percentile(pxs, 50)),
        "p90": float(np.percentile(pxs, 90)),
    }


# ──────────────────────────────────────────────────────────────────────────────
# kNN
# ──────────────────────────────────────────────────────────────────────────────

def _build_knn_tree(s: SubjectData) -> Tuple[Optional[cKDTree], np.ndarray]:
    c = s.hcr_centroids
    if c.empty:
        return None, np.zeros((0, 3))
    arr = c[["z_px", "y_px", "x_px"]].to_numpy(float)
    pts_um = arr * np.array([s.hcr_z_um, s.hcr_xy_um, s.hcr_xy_um])
    return cKDTree(pts_um), pts_um


def _knn_features(
    tree: Optional[cKDTree],
    pts_um: np.ndarray,
    query_um: np.ndarray,
    radius: float = KNN_RADIUS_UM,
) -> pd.DataFrame:
    N = len(query_um)
    if N == 0 or tree is None:
        return pd.DataFrame({
            "knn_d1": np.full(N, float("nan")),
            "n_neighbors_30um": np.full(N, float("nan")),
        })
    valid = np.isfinite(query_um).all(axis=1)
    knn_d1 = np.full(N, float("nan"))
    n_neighbors = np.full(N, float("nan"))
    if valid.any():
        # k=2 because query points are themselves in the tree → first neighbour is self
        d, _ = tree.query(query_um[valid], k=min(2, pts_um.shape[0]), workers=1)
        if d.ndim == 1:
            d = d[:, None]
        if d.shape[1] >= 2:
            knn_d1[valid] = d[:, 1]
        # n_neighbors_30um: count of OTHER cells within `radius` µm (exclude self).
        # query points are in the tree, so subtract 1.
        counts = tree.query_ball_point(query_um[valid], radius, return_length=True)
        n_neighbors[valid] = np.asarray(counts, dtype=float) - 1.0
    return pd.DataFrame({"knn_d1": knn_d1, "n_neighbors_30um": n_neighbors})




# ──────────────────────────────────────────────────────────────────────────────
# main extractor
# ──────────────────────────────────────────────────────────────────────────────

# ── z-strip parallelism (within-mouse; max-core) ──────────────────────────────
_V2CTX: Dict = {}


def _feat_workers(n_items: int) -> int:
    """Worker count for strip-level parallelism: MFISH_FEAT_WORKERS or cpu-2."""
    import os as _os
    env = _os.environ.get("MFISH_FEAT_WORKERS")
    if env:
        try:
            w = int(env)
        except ValueError:
            w = 1
    else:
        w = max(1, (_os.cpu_count() or 2) - 2)
    return max(1, min(w, max(1, n_items)))


def _v2_worker_init(sid, ctx_light):
    s = load_subject(sid)
    _V2CTX.clear()
    _V2CTX.update(ctx_light)
    _V2CTX["seg_orig"] = zarr.open(str(_orig_res_path(s)), mode="r")
    _V2CTX["arr405"] = _ch405_l2(s)
    _V2CTX["has_405"] = _V2CTX["arr405"] is not None


def _process_strip_v2(task):
    """Process one (strip, cell-chunk) task; returns (rows, n_owned, n_oversized, n_missing)."""
    z0_inner, owned = task
    C = _V2CTX
    seg_orig = C["seg_orig"]; arr405 = C["arr405"]; has_405 = C["has_405"]
    Z_seg = C["Z_seg"]; Y_seg = C["Y_seg"]; X_seg = C["X_seg"]
    seg_z_um = C["seg_z_um"]; seg_xy_um = C["seg_xy_um"]
    tb_lookup = C["tb_lookup"]; cent_z = C["cent_z"]; stage1 = C["stage1"]
    z_hi_global = C["z_hi_global"]
    n_owned = 0
    n_oversized = 0
    n_missing_in_zarr = 0
    feature_rows: List[Dict] = []

    # Adaptive per-chunk 3-D bbox: load exactly the z/y/x sub-region this chunk's
    # cells span (± pad), on ALL THREE axes. Because every cell's padded crop is
    # contained in the chunk's bbox-union + pad by construction, no cell can
    # overflow the loaded block → ZERO boundary skips (the old fixed strip ± Z_PAD
    # z-window is gone). z-strip bucketing is kept only to group z-nearby cells so
    # the block stays compact; it no longer bounds what is loaded.
    own_bbs = [tb_lookup.get(h) for h in owned if h in tb_lookup]
    if not own_bbs:
        return feature_rows, n_owned, n_oversized, n_missing_in_zarr
    pad = OPENING_RADIUS + 1
    z_min = min(b[0] for b in own_bbs)
    z_max = max(b[1] for b in own_bbs)
    y_min = min(b[2] for b in own_bbs)
    y_max = max(b[3] for b in own_bbs)
    x_min = min(b[4] for b in own_bbs)
    x_max = max(b[5] for b in own_bbs)
    z0_load = max(0, z_min - pad)
    z1_load = min(Z_seg, z_max + pad)
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

        cz = cent_z.get(hid, (zmin + zmax_ex) // 2)
        half_extent = max(cz - zmin, zmax_ex - 1 - cz)
        if half_extent + OPENING_RADIUS + 1 > Z_PAD:
            n_oversized += 1

        # Padded crop bounds in *global* level-2 coords.
        psz0_g = max(0, zmin - pad)
        psz1_g = min(Z_seg, zmax_ex + pad)
        psy0_g = max(0, ymin - pad)
        psy1_g = min(Y_seg, ymax_ex + pad)
        psx0_g = max(0, xmin - pad)
        psx1_g = min(X_seg, xmax_ex + pad)

        # Translate into block indices.
        bz0 = psz0_g - z0_load
        bz1 = psz1_g - z0_load
        by0 = psy0_g - sub_y0
        by1 = psy1_g - sub_y0
        bx0 = psx0_g - sub_x0
        bx1 = psx1_g - sub_x0

        if (bz0 < 0 or bz1 > seg_block.shape[0]
                or by0 < 0 or by1 > seg_block.shape[1]
                or bx0 < 0 or bx1 > seg_block.shape[2]):
            # Cell padded crop overflows the loaded sub-block (rare; at
            # z-strip edges for very tall cells). Skip this cell — it'll
            # be left as missing and emit NaN below.
            n_missing_in_zarr += 1
            continue

        seg_crop = seg_block[bz0:bz1, by0:by1, bx0:bx1]
        mask_crop = seg_crop == hid
        if not mask_crop.any():
            n_missing_in_zarr += 1
            continue

        n_raw = int(mask_crop.sum())

        # tight subview within padded crop
        tz0 = zmin - psz0_g
        ty0 = ymin - psy0_g
        tx0 = xmin - psx0_g
        tz1 = tz0 + (zmax_ex - zmin)
        ty1 = ty0 + (ymax_ex - ymin)
        tx1 = tx0 + (xmax_ex - xmin)
        mask_raw_tight = mask_crop[tz0:tz1, ty0:ty1, tx0:tx1]
        shape_raw = _shape_stats(mask_raw_tight, seg_z_um, seg_xy_um)

        # ── opening: cross+iter octahedral approximation, ≥10× faster
        #    than ball-r=3 footprint.
        mask_opened = ndi.binary_opening(
            mask_crop, structure=_CROSS_3D, iterations=OPENING_RADIUS,
        )
        mask_opened_tight = mask_opened[tz0:tz1, ty0:ty1, tx0:tx1]
        n_opened = int(mask_opened_tight.sum())
        shape_opened = _shape_stats(mask_opened_tight, seg_z_um, seg_xy_um)
        frac_kept = n_opened / n_raw if n_raw > 0 else float("nan")

        # ── core / shell on opened mask ────────────────────────────────
        if n_opened > 0:
            core_full = ndi.binary_erosion(
                mask_opened, structure=_CROSS_3D, iterations=EROSION_CORE_RADIUS,
            )
            core_tight = core_full[tz0:tz1, ty0:ty1, tx0:tx1]
            shell_tight = mask_opened_tight & ~core_tight
        else:
            core_tight = np.zeros_like(mask_opened_tight)
            shell_tight = np.zeros_like(mask_opened_tight)

        # ── 405 raw / opened / core / shell ─────────────────────────────
        if ch405_block is not None:
            img_pad = ch405_block[bz0:bz1, by0:by1, bx0:bx1]
            img_tight = img_pad[tz0:tz1, ty0:ty1, tx0:tx1]
            px_raw = img_tight[mask_raw_tight]
            px_opened = img_tight[mask_opened_tight] if n_opened > 0 else np.array([], dtype=np.float32)
            px_core = img_tight[core_tight] if core_tight.any() else np.array([], dtype=np.float32)
            px_shell = img_tight[shell_tight] if shell_tight.any() else np.array([], dtype=np.float32)
        else:
            px_raw = px_opened = px_core = px_shell = np.array([], dtype=np.float32)
            img_pad = None
            img_tight = None

        i_raw = _intensity_stats(px_raw)
        i_opened = _intensity_stats(px_opened)
        core_p50 = _percentile_or_nan(px_core, 50)
        shell_p50 = _percentile_or_nan(px_shell, 50)
        shell_p90 = _percentile_or_nan(px_shell, 90)
        core_p90 = _percentile_or_nan(px_core, 90)
        shell_minus_core_p50 = (
            shell_p50 - core_p50 if np.isfinite(shell_p50) and np.isfinite(core_p50) else float("nan")
        )
        shell_minus_core_p90 = (
            shell_p90 - core_p90 if np.isfinite(shell_p90) and np.isfinite(core_p90) else float("nan")
        )

        # ── adjacency / outside (rim from raw mask, dilation r=1) ──────
        mask_rim_full = ndi.binary_dilation(
            mask_crop, structure=_CROSS_3D, iterations=RIM_RADIUS,
        ) & ~mask_crop
        n_rim = int(mask_rim_full.sum())
        if n_rim > 0:
            rim_labels = seg_crop[mask_rim_full]
            # other hcr_ids
            fg_mask_in_rim = rim_labels != 0
            fg_ids = rim_labels[fg_mask_in_rim]
            if fg_ids.size > 0:
                uniq_ids, counts = np.unique(fg_ids, return_counts=True)
                surface_touching_frac = float(fg_ids.size) / n_rim
                top_neighbor_overlap_frac = float(counts.max()) / n_rim
                n_touching = int(uniq_ids.size)
                nbr_scores = [stage1.get(int(u)) for u in uniq_ids]
                nbr_scores = [v for v in nbr_scores if v is not None and np.isfinite(v)]
                if nbr_scores:
                    mean_score = float(np.mean(nbr_scores))
                    min_score = float(np.min(nbr_scores))
                else:
                    mean_score = float("nan")
                    min_score = float("nan")
            else:
                surface_touching_frac = 0.0
                top_neighbor_overlap_frac = 0.0
                n_touching = 0
                mean_score = float("nan")
                min_score = float("nan")
        else:
            surface_touching_frac = float("nan")
            top_neighbor_overlap_frac = float("nan")
            n_touching = 0
            mean_score = float("nan")
            min_score = float("nan")

        # ── 405 inside vs outside (using rim of raw) ───────────────────
        if ch405_block is not None and n_rim > 0:
            px_outside = img_pad[mask_rim_full]
            outside_p50 = _percentile_or_nan(px_outside, 50)
            outside_p90 = _percentile_or_nan(px_outside, 90)
        else:
            outside_p50 = outside_p90 = float("nan")
        inside_minus_outside_p50 = (
            i_raw["p50"] - outside_p50
            if np.isfinite(i_raw["p50"]) and np.isfinite(outside_p50) else float("nan")
        )
        inside_minus_outside_p90 = (
            i_raw["p90"] - outside_p90
            if np.isfinite(i_raw["p90"]) and np.isfinite(outside_p90) else float("nan")
        )

        row = {
            "hcr_id": hid,
            # shape raw
            "volume_vox_raw": shape_raw["volume_vox"],
            "aspect_zy": shape_raw["aspect_zy"],
            "aspect_zx": shape_raw["aspect_zx"],
            "aspect_yx": shape_raw["aspect_yx"],
            "solidity_raw": shape_raw["solidity"],
            "bbox_occupancy_raw": shape_raw["bbox_occupancy"],
            # shape opened
            "volume_vox_opened": shape_opened["volume_vox"],
            "frac_kept_opening": frac_kept,
            "solidity_opened": shape_opened["solidity"],
            # shape µm (volume/bbox from raw mask; equiv-diam raw + opened)
            "volume_um3_raw": shape_raw["volume_um3"],
            "bbox_z_extent_um": shape_raw["bbox_z_extent_um"],
            "bbox_y_extent_um": shape_raw["bbox_y_extent_um"],
            "bbox_x_extent_um": shape_raw["bbox_x_extent_um"],
            "equivalent_diameter_um_raw": shape_raw["equivalent_diameter_um"],
            "equivalent_diameter_um_opened": shape_opened["equivalent_diameter_um"],
            # 405 raw
            "c405_raw_mean": i_raw["mean"],
            "c405_raw_std": i_raw["std"],
            "c405_raw_p10": i_raw["p10"],
            "c405_raw_p50": i_raw["p50"],
            "c405_raw_p90": i_raw["p90"],
            # 405 opened
            "c405_opened_mean": i_opened["mean"],
            "c405_opened_std": i_opened["std"],
            "c405_opened_p10": i_opened["p10"],
            "c405_opened_p50": i_opened["p50"],
            "c405_opened_p90": i_opened["p90"],
            # 405 core vs shell
            "c405_core_p50_opened": core_p50,
            "c405_shell_p50_opened": shell_p50,
            "c405_shell_minus_core_p50": shell_minus_core_p50,
            "c405_shell_minus_core_p90": shell_minus_core_p90,
            # 405 inside vs outside
            "c405_outside_p50": outside_p50,
            "c405_inside_minus_outside_p50": inside_minus_outside_p50,
            "c405_inside_minus_outside_p90": inside_minus_outside_p90,
            # adjacency
            "n_touching_neighbors": n_touching,
            "surface_touching_frac": surface_touching_frac,
            "top_neighbor_overlap_frac": top_neighbor_overlap_frac,
            "mean_touching_score_stage1": mean_score,
            "min_touching_score_stage1": min_score,
        }
        # ── unified pass: compute axis (v3) + surface (v4) + protrusion (v5)
        #    families from the SAME raw/opened mask + 405 crop computed above
        #    (no re-read of the volume, no re-opening).
        row.update(all_v3_axis_features(
            mask_raw_tight, mask_opened_tight, img_tight,
            seg_z_um, seg_xy_um, seg_xy_um, bin_um=1.0, compute_dropped_peaks=False,
        ))
        _fv4 = all_v4_features(
            mask_raw_tight, mask_opened_tight, img_tight,
            seg_z_um, seg_xy_um, seg_xy_um, r_core_um=4.0,
        )
        # v2 + v4 both emit volume_um3_raw; keep v4's as *_v4 (matches the old
        # 4-parquet merge-suffix the _v5d_um model was trained on).
        if "volume_um3_raw" in _fv4:
            _fv4["volume_um3_raw_v4"] = _fv4.pop("volume_um3_raw")
        row.update(_fv4)
        row.update(protrusion_features(mask_crop, mask_opened, seg_crop, hid))
        feature_rows.append(row)
    return feature_rows, n_owned, n_oversized, n_missing_in_zarr


def extract_roi_features_v2(
    s: SubjectData,
    cache: bool = True,
) -> pd.DataFrame:
    """Extract v2 per-ROI features (405-only intensity + opening + adjacency)."""
    sid = s.subject_id
    out_path = _features_v2_cache_path(sid)
    if cache and out_path.exists():
        print(f"  [{sid}] loading cached v2 features from {out_path}")
        return pd.read_parquet(out_path)

    t_start = time.time()

    # ── open zarrs ─────────────────────────────────────────────────────────────
    seg_orig = zarr.open(str(_orig_res_path(s)), mode="r")
    _, _, Z_seg, Y_seg, X_seg = seg_orig.shape
    seg_xy_um = float(s.hcr_xy_um)
    seg_z_um = float(s.hcr_z_um)
    print(f"\n[{sid}] v2 extract | seg shape Z={Z_seg} Y={Y_seg} X={X_seg} | "
          f"xy={seg_xy_um:.4f} µm  z={seg_z_um:.4f} µm")

    arr405 = _ch405_l2(s)
    has_405 = arr405 is not None
    if not has_405:
        print(f"  [{sid}] WARNING: 405 channel missing — intensity features will be NaN")

    # ── stage-1 scores (for adjacency neighbour scoring) ──────────────────────
    stage1: Dict[int, float] = {}
    s1p = _stage1_score_path(sid)
    if s1p.exists():
        s1df = pd.read_parquet(s1p)
        stage1 = dict(zip(s1df["hcr_id"].astype(int), s1df["score"].astype(float)))
        print(f"  [{sid}] stage-1 scores loaded: {len(stage1)} cells")
    else:
        print(f"  [{sid}] WARNING: stage-1 score parquet missing — neighbour score features will be NaN")

    # ── kNN tree + centroid lookup ────────────────────────────────────────────
    knn_tree, pts_um = _build_knn_tree(s)
    cent = s.hcr_centroids.set_index("hcr_id")
    all_hids = s.hcr_centroids["hcr_id"].astype(int).tolist()
    cent_um: Dict[int, np.ndarray] = {}
    cent_z: Dict[int, int] = {}
    for hid in all_hids:
        if hid in cent.index:
            r = cent.loc[hid]
            cent_um[hid] = np.array([
                float(r["z_px"]) * seg_z_um,
                float(r["y_px"]) * seg_xy_um,
                float(r["x_px"]) * seg_xy_um,
            ])
            cent_z[hid] = int(round(float(r["z_px"])))

    # ── strip range ───────────────────────────────────────────────────────────
    if not s.hcr_centroids.empty:
        z_lo_global = max(0, int(s.hcr_centroids["z_px"].min()) - 2)
        z_hi_global = min(Z_seg, int(s.hcr_centroids["z_px"].max()) + 2)
    else:
        z_lo_global, z_hi_global = 0, Z_seg

    # Load cached tight bboxes (level-2) — used to drive iteration without
    # find_objects on the full strip.  Required for performance.
    tb_path = TIGHT_BBOX_CACHE / f"{sid}_hcr_cell_tight_bbox_v1.parquet"
    if not tb_path.exists():
        raise FileNotFoundError(
            f"tight bbox cache missing: {tb_path}.  Build it first with "
            f"`roi-classifier build-bbox {sid}` (or feat_tight_bbox.build_tight_bbox_sid)."
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

    feature_rows: List[Dict] = []

    # Bucket owned cells by strip (centroid_z assignment).
    cells_per_strip: Dict[int, List[int]] = {}
    for hid in all_hids:
        cz = cent_z.get(hid)
        if cz is None:
            continue
        # snap into [z_lo_global, z_hi_global)
        if not (z_lo_global <= cz < z_hi_global):
            continue
        bucket = z_lo_global + ((cz - z_lo_global) // STRIP_Z) * STRIP_Z
        cells_per_strip.setdefault(bucket, []).append(int(hid))

    n_owned = 0
    n_oversized = 0
    n_missing_in_zarr = 0
    import math as _math
    _ctx_light = dict(
        Z_seg=Z_seg, Y_seg=Y_seg, X_seg=X_seg,
        seg_z_um=seg_z_um, seg_xy_um=seg_xy_um,
        tb_lookup=tb_lookup, cent_z=cent_z, stage1=stage1,
        z_hi_global=z_hi_global,
    )
    # Balanced parallel units: split each strip's cells into spatially-contiguous
    # chunks (sorted by ymin) so a dense strip spreads across workers while each
    # chunk's bbox stays compact (small block load). The strip z-context
    # (z0..z1 ± Z_PAD) is unchanged, so per-cell features are identical → exact.
    _total = sum(len(v) for v in cells_per_strip.values())
    _w = _feat_workers(max(1, _total))
    _chunk = max(200, _math.ceil(_total / (_w * 3))) if _total else 1
    tasks = []
    for _z0 in sorted(cells_per_strip.keys()):
        _cells = sorted(cells_per_strip[_z0],
                        key=lambda h: tb_lookup[h][2] if h in tb_lookup else 0)
        for _i in range(0, len(_cells), _chunk):
            tasks.append((_z0, _cells[_i:_i + _chunk]))
    workers = _feat_workers(max(1, len(tasks)))
    print(f"  [{sid}] v2 strips={len(cells_per_strip)} chunks={len(tasks)} "
          f"(~{_chunk} cells/chunk) workers={workers}")
    if workers <= 1:
        _V2CTX.clear(); _V2CTX.update(_ctx_light)
        _V2CTX["seg_orig"] = seg_orig
        _V2CTX["arr405"] = arr405
        _V2CTX["has_405"] = has_405
        results = [_process_strip_v2(t) for t in tasks]
    else:
        from concurrent.futures import ProcessPoolExecutor
        from multiprocessing import get_context
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"),
                                 initializer=_v2_worker_init,
                                 initargs=(sid, _ctx_light)) as _ex:
            results = list(_ex.map(_process_strip_v2, tasks))
    for _rows, _no, _nov, _nm in results:
        feature_rows.extend(_rows)
        n_owned += _no
        n_oversized += _nov
        n_missing_in_zarr += _nm

    # Cells in centroid table that we never owned (e.g., centroid in seg gap).
    # Emit NaN rows so output schema covers all hcr_ids.
    nan_template = {
        k: float("nan") for k in [
            "volume_vox_raw",
            "aspect_zy", "aspect_zx", "aspect_yx",
            "solidity_raw", "bbox_occupancy_raw",
            "volume_vox_opened", "frac_kept_opening", "solidity_opened",
            "volume_um3_raw", "bbox_z_extent_um", "bbox_y_extent_um",
            "bbox_x_extent_um", "equivalent_diameter_um_raw",
            "equivalent_diameter_um_opened",
            "c405_raw_mean", "c405_raw_std", "c405_raw_p10", "c405_raw_p50", "c405_raw_p90",
            "c405_opened_mean", "c405_opened_std", "c405_opened_p10", "c405_opened_p50", "c405_opened_p90",
            "c405_core_p50_opened", "c405_shell_p50_opened",
            "c405_shell_minus_core_p50", "c405_shell_minus_core_p90",
            "c405_outside_p50", "c405_inside_minus_outside_p50", "c405_inside_minus_outside_p90",
            "surface_touching_frac", "top_neighbor_overlap_frac",
            "mean_touching_score_stage1", "min_touching_score_stage1",
        ]
    }
    nan_template["n_touching_neighbors"] = 0
    # unified families: match each old extractor's nan-fill convention
    for _c in roi_v3_axis_features.feature_columns():       # v3: n_peaks→0 (unless dropped)
        nan_template[_c] = (0 if ("n_peaks" in _c and _c not in roi_v3_axis_features._DROPPED_PEAK_COLS)
                            else float("nan"))
    for _c in roi_v4_features.feature_columns():            # v4: all NaN (volume_um3_raw→_v4)
        nan_template["volume_um3_raw_v4" if _c == "volume_um3_raw" else _c] = float("nan")
    for _c in roi_v5_features.feature_columns():            # v5: all NaN
        nan_template[_c] = float("nan")

    seen_set = {int(r["hcr_id"]) for r in feature_rows}
    for hid in all_hids:
        if int(hid) in seen_set:
            continue
        miss = dict(nan_template)
        miss["hcr_id"] = int(hid)
        feature_rows.append(miss)

    feat_df = pd.DataFrame(feature_rows).sort_values("hcr_id").reset_index(drop=True)

    # ── kNN ──────────────────────────────────────────────────────────────────
    query_um_arr = np.array([cent_um.get(int(h), np.full(3, np.nan)) for h in feat_df["hcr_id"]])
    knn_df = _knn_features(knn_tree, pts_um, query_um_arr)
    feat_df = pd.concat([feat_df, knn_df], axis=1)

    elapsed = time.time() - t_start
    print(f"  [{sid}] v2 done: {len(feat_df)} ROIs in {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"  [{sid}] v2 columns ({len(feat_df.columns)}): {list(feat_df.columns)}")

    if cache:
        feat_df.to_parquet(out_path, index=False)
        print(f"  [{sid}] v2 features saved → {out_path}")

    meta = {
        "subject_id": sid,
        "version": "v2",
        "channels_used": ["405"] if has_405 else [],
        "stage1_score_loaded": bool(stage1),
        "total_rois": int(len(feat_df)),
        "n_owned_in_strip_pass": int(n_owned),
        "n_oversized": int(n_oversized),
        "extraction_timestamp": datetime.utcnow().isoformat() + "Z",
        "seg_xy_um": seg_xy_um,
        "seg_z_um": seg_z_um,
        "opening_radius_voxels": OPENING_RADIUS,
        "core_erosion_radius_voxels": EROSION_CORE_RADIUS,
        "rim_radius_voxels": RIM_RADIUS,
        "strip_z": STRIP_Z,
        "z_pad": Z_PAD,
        "elapsed_seconds": float(elapsed),
    }
    with open(_meta_v2_path(sid), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  [{sid}] v2 meta saved → {_meta_v2_path(sid)}")

    return feat_df


# Public entry point used by features.extract_features.
def compute(s: SubjectData, cache: bool = True) -> pd.DataFrame:
    """Compute (or load from cache) shape + 405 + adjacency features for subject s."""
    return extract_roi_features_v2(s, cache=cache)
