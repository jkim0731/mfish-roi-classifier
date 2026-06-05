"""Per-ROI feature extractor for the HCR ROI pass/fail classifier (S11).

Public API
----------
extract_roi_features(s, level=2, cache=True) -> pd.DataFrame
    One row per HCR ROI for subject s.  Cached to
    code/dev_code/cached_roi_quality/<sid>_features.parquet.

Coordinate conventions
----------------------
All feature computation uses `segmentation_mask_orig_res.zarr` (level-2 in
XY, same Z as level-0).  This is the "misnamed" zarr — the name says orig_res
but it is actually the level-2 version (4× coarser in XY than the raw
`segmentation_mask.zarr`).

- orig_res seg: shape (1,1,Z,Y2,X2), uint32.
  xy voxel size = s.hcr_xy_um (~0.988 µm), z = s.hcr_z_um (~1 µm).
  These are the same units as hcr_centroids (which are also level-2).
- fused channel zarrs at level-2 share the same voxel grid as orig_res.
- hcr_centroids: level-2 voxel frame (z, y, x in level-2 pixels).
- metrics.pickle global_bbox: level-0 voxel indices — divide xy by 4 to
  get orig_res / level-2 coordinates.

Strategy
--------
Use a z-strip approach: load the orig_res seg mask and all 4 fused channels
one 128-z strip at a time.  For each strip, loop over all labels found in
that strip and accumulate min/max/sum per axis plus intensity stats (mean,
std, percentiles, 405 nuclear features) from in-memory arrays.  After
processing all strips, assemble per-cell tight bboxes and finalize statistics.

Cells that span multiple z-strips are handled by keeping running accumulators
across strips.  Finalization happens after the last strip that contains each
cell (identified by the tight-bbox zmax crossing below the current strip top).

The tight-bbox cache (cached_hcr_cell_tight_bbox) is also written for spot
subjects so subsequent sessions can skip the zarr scan.
"""
from __future__ import annotations

import json
import pickle
import time
import warnings
from collections import defaultdict
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

from .benchmark_data_loader import (
    HCR_SEG_XY_DOWNSAMPLE,
    SubjectData,
)
from . import config as _cfg

# ──────────────────────────────────────────────────────────────────────────────
# paths and constants
# ──────────────────────────────────────────────────────────────────────────────

ROI_QUALITY_CACHE = _cfg.ROI_QUALITY_DIR
ROI_QUALITY_CACHE.mkdir(parents=True, exist_ok=True)

TIGHT_BBOX_CACHE = _cfg.TIGHT_BBOX_DIR
TIGHT_BBOX_CACHE.mkdir(parents=True, exist_ok=True)

# 514 for intensity subjects (755252, 767022); 561 for spot subjects.
PROBE_CHANNEL: Dict[str, str] = {"755252": "514", "767022": "514"}
PROBE_CHANNEL_DEFAULT = "561"

SPOT_SUBJECTS = frozenset({"788406", "790322", "767018", "782149"})
INTENSITY_SUBJECTS = frozenset({"755252", "767022"})

KNN_KS = [1, 3, 5]
KNN_RADIUS_UM = 30.0

# z-strip height in slices
STRIP_Z = 128

# Number of percentile bins used for p10/p50/p90
# (we accumulate these as exact values since crops are small enough)
_PERCS = [10, 50, 90]


# ──────────────────────────────────────────────────────────────────────────────
# path helpers
# ──────────────────────────────────────────────────────────────────────────────

def _orig_res_path(s: SubjectData) -> Path:
    """segmentation_mask_orig_res.zarr — level-2 XY, same Z as level-0."""
    p = s.hcr_dir / "cell_body_segmentation" / "segmentation_mask_orig_res.zarr"
    if not p.exists():
        raise FileNotFoundError(f"orig_res zarr not found: {p}")
    return p


def _seg_zarr_l0_path(s: SubjectData) -> Path:
    p = s.hcr_dir / "cell_body_segmentation" / "segmentation_mask.zarr"
    if not p.exists():
        raise FileNotFoundError(f"seg mask zarr not found: {p}")
    return p


def _metrics_path(s: SubjectData) -> Optional[Path]:
    p = s.hcr_dir / "cell_body_segmentation" / "metrics.pickle"
    return p if p.exists() else None


def _fused_l2(s: SubjectData, channel: str):
    """Open fused channel zarr at level-2.  Returns zarr array or None."""
    p = s.hcr_dir / "image_tile_fusing" / "fused" / f"channel_{channel}.zarr"
    if not p.exists():
        return None
    try:
        z = zarr.open(str(p), mode="r")
        return z["2"] if "2" in z else None
    except Exception:
        return None


def _features_cache_path(sid: str) -> Path:
    return ROI_QUALITY_CACHE / f"{sid}_features.parquet"


def _tight_bbox_cache_path(sid: str) -> Path:
    return TIGHT_BBOX_CACHE / f"{sid}_hcr_cell_tight_bbox_v1.parquet"


def _meta_path(sid: str) -> Path:
    return ROI_QUALITY_CACHE / f"{sid}_meta.json"


# ──────────────────────────────────────────────────────────────────────────────
# per-cell accumulator helpers
# ──────────────────────────────────────────────────────────────────────────────

def _empty_intens_acc() -> Dict:
    """Empty accumulator for intensity stats."""
    return {
        "px_list": [],   # list of float32 arrays (one per strip)
        "shell_list": [],  # shell pixels per strip for fg_minus_margin
    }


def _empty_c405_acc() -> Dict:
    return {
        "core_list": [],   # eroded-core 405 pixels
        "shell_list": [],  # dilated-shell 405 pixels
        "max_405_in": 0.0,   # max 405 inside ROI (for peak_inside check)
        "max_405_global": 0.0,  # max 405 in crop
    }


def _finalize_intens(acc: Dict) -> Dict:
    """Finalize intensity accumulator into scalar stats."""
    all_px = np.concatenate(acc["px_list"]) if acc["px_list"] else np.array([], dtype=np.float32)
    all_shell = np.concatenate(acc["shell_list"]) if acc["shell_list"] else np.array([], dtype=np.float32)
    if all_px.size == 0:
        return {"mean": float("nan"), "std": float("nan"),
                "p10": float("nan"), "p50": float("nan"),
                "p90": float("nan"), "fg_minus_margin_p50": float("nan")}
    mean_ = float(np.mean(all_px))
    std_ = float(np.std(all_px))
    p10, p50, p90 = [float(np.percentile(all_px, q)) for q in _PERCS]
    shell_p50 = float(np.median(all_shell)) if all_shell.size > 0 else float("nan")
    return {"mean": mean_, "std": std_,
            "p10": p10, "p50": p50, "p90": p90,
            "fg_minus_margin_p50": p50 - shell_p50}


def _finalize_c405(acc: Dict) -> Dict:
    core_px = np.concatenate(acc["core_list"]) if acc["core_list"] else np.array([], dtype=np.float32)
    shell_px = np.concatenate(acc["shell_list"]) if acc["shell_list"] else np.array([], dtype=np.float32)
    core_p90 = float(np.percentile(core_px, 90)) if core_px.size > 0 else float("nan")
    shell_p90 = float(np.percentile(shell_px, 90)) if shell_px.size > 0 else float("nan")
    contrast = core_p90 - shell_p90 if np.isfinite(core_p90) and np.isfinite(shell_p90) else float("nan")
    peak_inside = bool(acc["max_405_in"] >= acc["max_405_global"] - 1e-6)
    return {
        "c405_core_p90": core_p90,
        "c405_shell_p90": shell_p90,
        "c405_peak_inside": peak_inside,
        "c405_core_minus_shell_p90": contrast,
    }


# ──────────────────────────────────────────────────────────────────────────────
# shape features
# ──────────────────────────────────────────────────────────────────────────────

def _shape_features(
    hid: int,
    binary: np.ndarray,
    tb: Dict,
    seg_z_um: float,
    seg_xy_um: float,
    seg_ZYX: Tuple[int, int, int],
) -> Dict:
    """Compute shape features from the binary mask (tight crop) and tight bbox.

    binary: (dz,dy,dx) bool, tight crop of the cell.
    tb: tight-bbox dict with zmin_vox..zmax_vox (level-2 coords, half-open).
    seg_ZYX: full seg mask shape (for boundary test).
    """
    Z, Y, X = seg_ZYX

    vol_vox = int(tb["volume_vox"])
    dz = int(tb["zmax_vox"]) - int(tb["zmin_vox"])
    dy = int(tb["ymax_vox"]) - int(tb["ymin_vox"])
    dx = int(tb["xmax_vox"]) - int(tb["xmin_vox"])
    bbox_vol_vox = dz * dy * dx

    bbox_z_um = dz * seg_z_um
    bbox_y_um = dy * seg_xy_um
    bbox_x_um = dx * seg_xy_um
    vol_um3 = vol_vox * seg_z_um * seg_xy_um * seg_xy_um

    bbox_occ = vol_vox / max(bbox_vol_vox, 1)
    aspect_zy = bbox_z_um / max(bbox_y_um, 1e-9)
    aspect_zx = bbox_z_um / max(bbox_x_um, 1e-9)
    aspect_yx = bbox_y_um / max(bbox_x_um, 1e-9)
    equiv_diam = 2.0 * (3.0 * vol_um3 / (4.0 * np.pi)) ** (1.0 / 3.0)

    boundary = bool(
        int(tb["zmin_vox"]) == 0 or int(tb["zmax_vox"]) >= Z
        or int(tb["ymin_vox"]) == 0 or int(tb["ymax_vox"]) >= Y
        or int(tb["xmin_vox"]) == 0 or int(tb["xmax_vox"]) >= X
    )

    solidity = float("nan")
    sphericity = float("nan")
    # Solidity via scipy.spatial.ConvexHull on the voxel point cloud.
    # This is ~10× faster than skimage regionprops_table's convex_hull_image
    # approach (~0.8ms vs ~2.7ms per cell), which matters for 100k+ cells.
    # Sphericity via marching_cubes is expensive (~7.6ms per cell = 10+ min
    # total for large subjects), so it is skipped and left as NaN.
    if binary is not None and binary.any() and binary.sum() > 7:
        try:
            from scipy.spatial import ConvexHull
            # Scale voxel indices to µm so the convex hull volume is in µm³
            zz, yy, xx = np.where(binary)
            pts_um = np.column_stack([
                zz * seg_z_um,
                yy * seg_xy_um,
                xx * seg_xy_um,
            ])
            if pts_um.shape[0] >= 4:
                hull = ConvexHull(pts_um)
                hull_vol = float(hull.volume)  # convex hull volume in µm³
                if hull_vol > 0:
                    # Note: solidity computed from voxel-center ConvexHull may
                    # slightly exceed 1.0 for nearly-convex cells (the voxel
                    # centroid set convex hull is slightly smaller than the
                    # actual filled voxel convex hull image). Clip in training.
                    solidity = float(vol_um3 / hull_vol)
        except Exception:
            pass  # leave solidity = NaN for degenerate cells

    return {
        "volume_vox": vol_vox,
        "volume_um3": vol_um3,
        "bbox_z_extent": bbox_z_um,
        "bbox_y_extent": bbox_y_um,
        "bbox_x_extent": bbox_x_um,
        "equivalent_diameter_um": equiv_diam,
        "aspect_zy": aspect_zy,
        "aspect_zx": aspect_zx,
        "aspect_yx": aspect_yx,
        "solidity": solidity,
        "sphericity": sphericity,
        "boundary_touching": boundary,
        "bbox_occupancy": bbox_occ,
    }


# ──────────────────────────────────────────────────────────────────────────────
# kNN features
# ──────────────────────────────────────────────────────────────────────────────

def _build_knn_tree(s: SubjectData) -> Tuple[Optional[cKDTree], np.ndarray]:
    """cKDTree on HCR centroids in µm (z,y,x).  hcr_centroids are level-2."""
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
    ks: List[int] = KNN_KS,
    radius: float = KNN_RADIUS_UM,
) -> pd.DataFrame:
    """Batch kNN query.  query_um: (N,3) in µm (z,y,x)."""
    N = len(query_um)
    if N == 0 or tree is None:
        return pd.DataFrame(
            {f"knn_d{k}": np.full(N, float("nan")) for k in ks}
            | {"n_neighbors_30um": np.zeros(N, dtype=int)}
        )
    max_k = max(ks) + 1
    actual_k = min(max_k, pts_um.shape[0])
    valid = np.isfinite(query_um).all(axis=1)
    dists_full = np.full((N, actual_k), float("nan"))
    if valid.any():
        dists_valid, _ = tree.query(query_um[valid], k=actual_k, workers=1)
        if actual_k == 1:
            dists_valid = dists_valid[:, np.newaxis]
        dists_full[valid] = dists_valid

    rows: Dict[str, np.ndarray] = {}
    for k in ks:
        rows[f"knn_d{k}"] = dists_full[:, k] if k < actual_k else np.full(N, float("nan"))

    n_in_radius = np.zeros(N, dtype=int)
    if valid.any():
        counts = tree.query_ball_point(query_um[valid], r=radius, workers=1)
        n_in_radius[valid] = [max(0, len(c) - 1) for c in counts]
    rows["n_neighbors_30um"] = n_in_radius
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# GFP+ features
# ──────────────────────────────────────────────────────────────────────────────

def _gfp_df(s: SubjectData) -> pd.DataFrame:
    gdf = s.hcr_gfp_df
    if gdf is None or gdf.empty:
        return pd.DataFrame(columns=["hcr_id", "gfp_feature_value",
                                      "gfp_density", "gfp_counts"])
    out = pd.DataFrame({"hcr_id": gdf["hcr_id"].astype(int)})
    feat = s.gfp_feature_name
    out["gfp_feature_value"] = (
        gdf[feat].values if feat and feat in gdf.columns else float("nan")
    )
    out["gfp_density"] = gdf["density"].values if "density" in gdf.columns else float("nan")
    out["gfp_counts"] = gdf["counts"].values if "counts" in gdf.columns else float("nan")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# pickle consistency
# ──────────────────────────────────────────────────────────────────────────────

def _pickle_consistency(
    hids: np.ndarray,
    tight_bboxes: Dict[int, Dict],  # level-2 coords
    metrics: Optional[Dict],        # pickle, level-0 coords
) -> pd.DataFrame:
    """Check tight zarr bbox (level-2) vs pickle global_bbox (level-0).

    Pickle bbox: [zlo, ylo, xlo, zhi, yhi, xhi] all inclusive.
    Tight bbox (level-2): [zmin_vox..zmax_vox) half-open.
    To compare: multiply tight bbox xy by 4 to get level-0; compare with
    pickle bbox extended by +1 (inclusive → exclusive).
    """
    in_pickle = np.ones(len(hids), dtype=bool)
    vol_diff = np.zeros(len(hids), dtype=float)

    if metrics is not None:
        for i, hid in enumerate(hids):
            tb = tight_bboxes.get(int(hid))
            m = metrics.get(int(hid))
            if m is None or tb is None:
                in_pickle[i] = False
                continue
            bb = np.asarray(m["global_bbox"], dtype=int)
            pz0, py0, px0, pz1, py1, px1 = bb  # level-0, inclusive
            # tight bbox (level-2); z same, xy ×4 for level-0 comparison
            tz0 = int(tb["zmin_vox"]); tz1 = int(tb["zmax_vox"])
            ty0 = int(tb["ymin_vox"]) * HCR_SEG_XY_DOWNSAMPLE
            ty1 = int(tb["ymax_vox"]) * HCR_SEG_XY_DOWNSAMPLE
            tx0 = int(tb["xmin_vox"]) * HCR_SEG_XY_DOWNSAMPLE
            tx1 = int(tb["xmax_vox"]) * HCR_SEG_XY_DOWNSAMPLE
            in_pickle[i] = bool(
                tz0 >= pz0 and tz1 <= pz1 + 1
                and ty0 >= py0 and ty1 <= py1 + 1
                and tx0 >= px0 and tx1 <= px1 + 1
            )
            pickle_vol = float(m.get("volume", float("nan")))
            zarr_vol = float(tb["volume_vox"])
            # volume comparison: pickle vol is in level-0 voxels, zarr vol in
            # level-2 voxels → multiply zarr vol by HCR_SEG_XY_DOWNSAMPLE² to
            # convert to level-0 equivalents before taking the difference.
            # NOTE: this is approximate; the actual level-0 voxel count can
            # differ slightly from 16× the level-2 count due to rounding.
            vol_diff[i] = (
                pickle_vol - zarr_vol * (HCR_SEG_XY_DOWNSAMPLE ** 2)
                if np.isfinite(pickle_vol) else float("nan")
            )

    return pd.DataFrame({
        "hcr_id": hids.astype(int),
        "tight_bbox_in_pickle_bbox": in_pickle,
        "volume_pickle_minus_zarr": vol_diff,
    })


# ──────────────────────────────────────────────────────────────────────────────
# main extractor
# ──────────────────────────────────────────────────────────────────────────────

def extract_roi_features(
    s: SubjectData,
    level: int = 2,
    cache: bool = True,
) -> pd.DataFrame:
    """Extract per-ROI features for all HCR ROIs in subject s.

    Parameters
    ----------
    s : SubjectData
        Loaded subject.
    level : int
        Recorded in meta JSON only.  Features always use level-2 data.
    cache : bool
        Load from parquet if it exists; otherwise compute and write.

    Returns
    -------
    pd.DataFrame
        One row per HCR ROI.  hcr_id is a column (not the index).
    """
    sid = s.subject_id
    cache_path = _features_cache_path(sid)

    if cache and cache_path.exists():
        print(f"  [{sid}] loading cached features from {cache_path}")
        return pd.read_parquet(cache_path)

    t_start = time.time()
    probe_ch = PROBE_CHANNEL.get(sid, PROBE_CHANNEL_DEFAULT)
    channels = ["405", "488", probe_ch, "594"]
    has_pickle = _metrics_path(s) is not None

    print(f"\n[{sid}] extracting ROI features | "
          f"pickle={has_pickle} | probe={probe_ch} | "
          f"n_centroids={len(s.hcr_centroids)}")

    # ── open zarrs ─────────────────────────────────────────────────────────────
    seg_orig = zarr.open(str(_orig_res_path(s)), mode="r")
    # orig_res is a single array (not a group)
    _, _, Z_seg, Y_seg, X_seg = seg_orig.shape
    seg_ZYX = (Z_seg, Y_seg, X_seg)
    # orig_res xy voxel size = s.hcr_xy_um (level-2); z = s.hcr_z_um
    seg_xy_um = s.hcr_xy_um
    seg_z_um = s.hcr_z_um
    print(f"  [{sid}] orig_res shape (Z,Y,X)=({Z_seg},{Y_seg},{X_seg}), "
          f"seg_xy_um={seg_xy_um:.4f} µm, seg_z_um={seg_z_um:.4f} µm")

    ch_arrs: Dict[str, object] = {}
    for ch in channels:
        arr = _fused_l2(s, ch)
        if arr is not None:
            print(f"  [{sid}] ch {ch} level-2: shape {arr.shape}")
        else:
            print(f"  [{sid}] ch {ch}: MISSING — intensities will be NaN")
        ch_arrs[ch] = arr

    # ── load pickle ────────────────────────────────────────────────────────────
    metrics: Optional[Dict] = None
    if has_pickle:
        with open(_metrics_path(s), "rb") as f:
            metrics = pickle.load(f)
        print(f"  [{sid}] metrics.pickle: {len(metrics)} cells")

    # ── kNN tree ───────────────────────────────────────────────────────────────
    print(f"  [{sid}] building kNN tree on {len(s.hcr_centroids)} centroids...")
    knn_tree, pts_um = _build_knn_tree(s)

    # ── centroid lookup: level-2 px → µm (z,y,x) ──────────────────────────────
    cent = s.hcr_centroids.set_index("hcr_id")
    all_hids = s.hcr_centroids["hcr_id"].astype(int).tolist()
    cent_um: Dict[int, np.ndarray] = {}
    for hid in all_hids:
        if hid in cent.index:
            r = cent.loc[hid]
            cent_um[hid] = np.array([
                float(r["z_px"]) * seg_z_um,
                float(r["y_px"]) * seg_xy_um,
                float(r["x_px"]) * seg_xy_um,
            ])

    # ── determine z-strip range ────────────────────────────────────────────────
    if not s.hcr_centroids.empty:
        z_lo_global = max(0, int(s.hcr_centroids["z_px"].min()) - 2)
        z_hi_global = min(Z_seg, int(s.hcr_centroids["z_px"].max()) + 2)
    else:
        z_lo_global, z_hi_global = 0, Z_seg
    print(f"  [{sid}] z-strip range: [{z_lo_global}, {z_hi_global}), "
          f"n_strips={(z_hi_global-z_lo_global+STRIP_Z-1)//STRIP_Z}")

    # ──────────────────────────────────────────────────────────────────────────
    # Per-cell accumulators (indexed by label id)
    # ──────────────────────────────────────────────────────────────────────────
    # Shape / tight bbox: accumulated across strips
    tb_acc: Dict[int, Dict] = {}   # hid -> tight_bbox partial result
    # Binary mask strips: accumulated for solidity/sphericity.
    # Each entry is a list of (z_start_global, tight_mask_crop) tuples.
    # Kept only until the cell is fully in the past; cleared after finalization.
    binary_strips: Dict[int, List] = {}   # hid -> list of (z_start, mask_arr)
    # Intensity per channel
    intens_acc: Dict[int, Dict[str, Dict]] = {}   # hid -> {ch -> acc}
    # 405 nuclear
    c405_acc: Dict[int, Dict] = {}

    all_hids_set = set(all_hids)

    def _get_or_init_acc(hid: int):
        if hid not in intens_acc:
            intens_acc[hid] = {ch: _empty_intens_acc() for ch in channels}
            c405_acc[hid] = _empty_c405_acc()

    # ──────────────────────────────────────────────────────────────────────────
    # z-strip loop
    # ──────────────────────────────────────────────────────────────────────────
    z_strips = range(z_lo_global, z_hi_global, STRIP_Z)
    for z0 in tqdm(z_strips, desc=f"[{sid}] z-strips", unit="strip"):
        z1 = min(z0 + STRIP_Z, z_hi_global)
        dz = z1 - z0

        # Load seg strip (level-2 XY)
        seg_strip = np.asarray(seg_orig[0, 0, z0:z1, :, :])  # (dz, Y2, X2)

        # Load all channel strips
        ch_strips: Dict[str, Optional[np.ndarray]] = {}
        for ch in channels:
            if ch_arrs[ch] is not None:
                # fused level-2 shares the same z as orig_res
                arr_strip = np.asarray(ch_arrs[ch][0, 0, z0:z1, :, :]).astype(np.float32)
                # sanity: shapes should match in z; clamp if off by 1
                if arr_strip.shape[0] != dz:
                    arr_strip = arr_strip[:dz]
                if arr_strip.shape[1:] != (Y_seg, X_seg):
                    # resize to match (shouldn't happen if same pipeline)
                    arr_strip = None
                ch_strips[ch] = arr_strip
            else:
                ch_strips[ch] = None

        # find_objects: one pass over the strip returns tight slices for every
        # label.  This replaces the O(N × strip_size) np.where loop with a
        # single O(strip_size) C-level scan followed by O(crop_size) per label.
        label_slices = ndi.find_objects(seg_strip)  # list of (sz,sy,sx) or None

        for lbl_idx, sl in enumerate(label_slices):
            if sl is None:
                continue
            hid = lbl_idx + 1  # find_objects is 0-indexed for 1-indexed labels
            if hid not in all_hids_set:
                continue  # skip labels not in centroid table

            # sl = (slice_z, slice_y, slice_x) — tight bounds in strip coords
            sl_z, sl_y, sl_x = sl

            # Tight crop (no padding) — used for voxel counting and centroids
            tight_crop = seg_strip[sl]        # small (dz', dy', dx') subarray
            mask_tight = (tight_crop == hid)  # bool mask of just this label
            n_pxs = int(mask_tight.sum())
            if n_pxs == 0:
                continue

            # Extract global z-coords for the tight crop to compute centroid/bbox
            sz0 = sl_z.start; sz1 = sl_z.stop
            sy0 = sl_y.start; sy1 = sl_y.stop
            sx0 = sl_x.start; sx1 = sl_x.stop

            # Recover per-voxel indices within the tight crop for centroid sums
            zc_local, yc_local, xc_local = np.where(mask_tight)
            z_global_arr = zc_local + z0 + sz0   # global z
            y_global_arr = yc_local + sy0         # global y (same as strip)
            x_global_arr = xc_local + sx0         # global x (same as strip)

            # Add 1-voxel padding for dilation/erosion (clamped to strip bounds)
            sz0p = max(0, sz0 - 1); sz1p = min(z1 - z0, sz1 + 1)
            sy0p = max(0, sy0 - 1); sy1p = min(Y_seg, sy1 + 1)
            sx0p = max(0, sx0 - 1); sx1p = min(X_seg, sx1 + 1)

            # Padded crop (for dilation/erosion to have correct boundary pixels)
            mask_crop = (seg_strip[sz0p:sz1p, sy0p:sy1p, sx0p:sx1p] == hid)

            # ── accumulate tight bbox ──────────────────────────────────────
            if hid not in tb_acc:
                tb_acc[hid] = {
                    "zmin": int(z_global_arr.min()), "zmax": int(z_global_arr.max()),
                    "ymin": int(y_global_arr.min()), "ymax": int(y_global_arr.max()),
                    "xmin": int(x_global_arr.min()), "xmax": int(x_global_arr.max()),
                    "volume_vox": n_pxs,
                    "zsum": float(z_global_arr.sum()),
                    "ysum": float(y_global_arr.sum()),
                    "xsum": float(x_global_arr.sum()),
                }
            else:
                tb = tb_acc[hid]
                tb["zmin"] = min(tb["zmin"], int(z_global_arr.min()))
                tb["zmax"] = max(tb["zmax"], int(z_global_arr.max()))
                tb["ymin"] = min(tb["ymin"], int(y_global_arr.min()))
                tb["ymax"] = max(tb["ymax"], int(y_global_arr.max()))
                tb["xmin"] = min(tb["xmin"], int(x_global_arr.min()))
                tb["xmax"] = max(tb["xmax"], int(x_global_arr.max()))
                tb["volume_vox"] += n_pxs
                tb["zsum"] += float(z_global_arr.sum())
                tb["ysum"] += float(y_global_arr.sum())
                tb["xsum"] += float(x_global_arr.sum())

            # ── accumulate tight binary strip for shape features ──────────
            # Store the tight (unpadded) boolean mask for this strip contribution,
            # plus its global (y, x) origin so we can place it correctly in the
            # full-cell bbox volume during finalization.
            if hid not in binary_strips:
                binary_strips[hid] = []
            binary_strips[hid].append((z0 + sz0, sy0, sx0, mask_tight))

            # ── intensity accumulation (operate on tight crop only) ─────────
            _get_or_init_acc(hid)

            # Dilation/erosion on the small crop (not the full strip)
            dilated_crop = ndi.binary_dilation(mask_crop, iterations=1)
            shell_crop = dilated_crop & ~mask_crop
            eroded_crop = ndi.binary_erosion(mask_crop, iterations=1)

            for ch in channels:
                ch_strip = ch_strips[ch]
                if ch_strip is None:
                    continue
                img_crop = ch_strip[sz0p:sz1p, sy0p:sy1p, sx0p:sx1p]
                pxs = img_crop[mask_crop].astype(np.float32)
                intens_acc[hid][ch]["px_list"].append(pxs)
                shell_pxs = img_crop[shell_crop].astype(np.float32)
                intens_acc[hid][ch]["shell_list"].append(shell_pxs)

            # ── 405 nuclear accumulation ────────────────────────────────────
            ch405_strip = ch_strips.get("405")
            if ch405_strip is not None:
                img405_crop = ch405_strip[sz0p:sz1p, sy0p:sy1p, sx0p:sx1p]
                core_pxs = img405_crop[eroded_crop].astype(np.float32) if eroded_crop.any() else np.array([], dtype=np.float32)
                shell_pxs405 = img405_crop[shell_crop].astype(np.float32)
                in_roi_pxs = img405_crop[mask_crop]
                dilated_pxs = img405_crop[dilated_crop]

                acc405 = c405_acc[hid]
                acc405["core_list"].append(core_pxs)
                acc405["shell_list"].append(shell_pxs405)
                acc405["max_405_in"] = max(acc405["max_405_in"],
                                           float(in_roi_pxs.max()) if in_roi_pxs.size > 0 else 0.0)
                acc405["max_405_global"] = max(acc405["max_405_global"],
                                               float(dilated_pxs.max()) if dilated_pxs.size > 0 else 0.0)

    print(f"  [{sid}] z-strip pass done. Accumulated {len(tb_acc)} labels.")

    # ──────────────────────────────────────────────────────────────────────────
    # Finalize per-cell stats
    # ──────────────────────────────────────────────────────────────────────────
    print(f"  [{sid}] finalizing features...")

    nan_shape = {k: float("nan") for k in [
        "volume_vox", "volume_um3", "bbox_z_extent", "bbox_y_extent",
        "bbox_x_extent", "equivalent_diameter_um", "aspect_zy",
        "aspect_zx", "aspect_yx", "solidity", "sphericity", "bbox_occupancy",
    ]}
    nan_shape["boundary_touching"] = False
    nan_intens = {"mean": float("nan"), "std": float("nan"),
                   "p10": float("nan"), "p50": float("nan"),
                   "p90": float("nan"), "fg_minus_margin_p50": float("nan")}
    nan_c405 = {"c405_core_p90": float("nan"), "c405_shell_p90": float("nan"),
                 "c405_peak_inside": False, "c405_core_minus_shell_p90": float("nan")}

    shape_rows: List[Dict] = []
    intens_rows: Dict[str, List[Dict]] = {ch: [] for ch in channels}
    c405_rows: List[Dict] = []
    tight_bboxes_final: Dict[int, Dict] = {}
    query_um_list: List[np.ndarray] = []

    for hid in all_hids:
        hid = int(hid)

        # centroid µm (for kNN)
        query_um_list.append(cent_um.get(hid, np.full(3, float("nan"))))

        if hid not in tb_acc:
            # Cell not found in seg mask at all
            sf = dict(nan_shape); sf["hcr_id"] = hid
            shape_rows.append(sf)
            for ch in channels:
                r = dict(nan_intens); r["hcr_id"] = hid; intens_rows[ch].append(r)
            r = dict(nan_c405); r["hcr_id"] = hid; c405_rows.append(r)
            continue

        tb_raw = tb_acc[hid]
        n = tb_raw["volume_vox"]
        tb = {
            "zmin_vox": tb_raw["zmin"],
            "zmax_vox": tb_raw["zmax"] + 1,  # half-open
            "ymin_vox": tb_raw["ymin"],
            "ymax_vox": tb_raw["ymax"] + 1,
            "xmin_vox": tb_raw["xmin"],
            "xmax_vox": tb_raw["xmax"] + 1,
            "volume_vox": n,
            "zc_vox": tb_raw["zsum"] / n,
            "yc_vox": tb_raw["ysum"] / n,
            "xc_vox": tb_raw["xsum"] / n,
        }
        tight_bboxes_final[hid] = tb

        # Shape: reconstruct binary mask from accumulated strip masks.
        # Avoids per-cell zarr re-read (~100ms each × 100k cells = hours).
        # For cells spanning multiple strips we concatenate z-slabs.
        binary: Optional[np.ndarray] = None
        strips_for_cell = binary_strips.pop(hid, None)
        if strips_for_cell is not None:
            dz_full = int(tb["zmax_vox"]) - int(tb["zmin_vox"])
            dy_full = int(tb["ymax_vox"]) - int(tb["ymin_vox"])
            dx_full = int(tb["xmax_vox"]) - int(tb["xmin_vox"])
            if dz_full > 0 and dy_full > 0 and dx_full > 0:
                binary = np.zeros((dz_full, dy_full, dx_full), dtype=bool)
                z_off = int(tb["zmin_vox"])
                y_off = int(tb["ymin_vox"])
                x_off = int(tb["xmin_vox"])
                try:
                    for (z_start_global, y_start_global, x_start_global,
                         mask_slice) in strips_for_cell:
                        dz_s, dy_s, dx_s = mask_slice.shape
                        bz0 = z_start_global - z_off
                        bz1 = bz0 + dz_s
                        by0 = y_start_global - y_off
                        bx0 = x_start_global - x_off
                        binary[bz0:bz1, by0:by0+dy_s, bx0:bx0+dx_s] |= mask_slice
                except (IndexError, ValueError):
                    binary = None  # fall back to NaN for solidity/sphericity

        sf = _shape_features(hid, binary, tb, seg_z_um, seg_xy_um, seg_ZYX)
        sf["hcr_id"] = hid
        shape_rows.append(sf)

        # Intensity finalization
        if hid in intens_acc:
            for ch in channels:
                stats = _finalize_intens(intens_acc[hid][ch])
                stats["hcr_id"] = hid
                intens_rows[ch].append(stats)
        else:
            for ch in channels:
                r = dict(nan_intens); r["hcr_id"] = hid; intens_rows[ch].append(r)

        # 405 finalization
        if hid in c405_acc:
            nf = _finalize_c405(c405_acc[hid])
            nf["hcr_id"] = hid
            c405_rows.append(nf)
        else:
            r = dict(nan_c405); r["hcr_id"] = hid; c405_rows.append(r)

    # ── kNN features ────────────────────────────────────────────────────────
    query_um_arr = np.array(query_um_list)
    knn_df = _knn_features(knn_tree, pts_um, query_um_arr)
    knn_df["hcr_id"] = all_hids

    # ── GFP+ features ────────────────────────────────────────────────────────
    gfp_feats = _gfp_df(s)

    # ── pickle consistency ────────────────────────────────────────────────────
    hids_arr = np.array(all_hids)
    cons_df = _pickle_consistency(hids_arr, tight_bboxes_final, metrics)

    # ── assemble ────────────────────────────────────────────────────────────
    shape_df = pd.DataFrame(shape_rows)
    intens_parts = []
    for ch in channels:
        df_ch = pd.DataFrame(intens_rows[ch])
        df_ch = df_ch.rename(columns={c: f"c{ch}_{c}" for c in df_ch.columns if c != "hcr_id"})
        intens_parts.append(df_ch)
    intens_df = intens_parts[0]
    for p in intens_parts[1:]:
        intens_df = intens_df.merge(p, on="hcr_id", how="outer")

    c405_df = pd.DataFrame(c405_rows)

    result = shape_df
    for df in [intens_df, c405_df, knn_df, gfp_feats, cons_df]:
        result = result.merge(df, on="hcr_id", how="left")

    result = result.sort_values("hcr_id").reset_index(drop=True)
    for bc in ["boundary_touching", "c405_peak_inside", "tight_bbox_in_pickle_bbox"]:
        if bc in result.columns:
            result[bc] = result[bc].fillna(False).astype(bool)

    elapsed = time.time() - t_start
    print(f"  [{sid}] done: {len(result)} ROIs in {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"  [{sid}] columns ({len(result.columns)}): {list(result.columns)}")
    if len(result) > 0:
        vol_fin = result["volume_vox"].dropna()
        if len(vol_fin) > 0:
            print(f"  [{sid}] volume_vox range: [{vol_fin.min():.0f}, {vol_fin.max():.0f}]")
        print(f"  [{sid}] boundary_touching: {result['boundary_touching'].mean():.3f}")
        print(f"  [{sid}] c405_peak_inside: {result['c405_peak_inside'].mean():.3f}")
        print(f"  [{sid}] cells found in seg: {len(tight_bboxes_final)} / {len(all_hids)}")

    # ── also save/update tight-bbox cache (v1 schema) ─────────────────────────
    _update_tight_bbox_cache(sid, tight_bboxes_final)

    # ── cache features ────────────────────────────────────────────────────────
    if cache:
        result.to_parquet(cache_path, index=False)
        print(f"  [{sid}] features saved → {cache_path}")

    # ── meta JSON ─────────────────────────────────────────────────────────────
    _write_meta(sid, s, channels, probe_ch, has_pickle, len(result),
                elapsed, level, seg_ZYX, ch_arrs)

    return result


def _update_tight_bbox_cache(sid: str, tight_bboxes: Dict[int, Dict]) -> None:
    """Merge newly computed tight bboxes into the v1 parquet cache."""
    if not tight_bboxes:
        return
    cache_path = _tight_bbox_cache_path(sid)
    cols = ["hcr_id", "zmin_vox", "ymin_vox", "xmin_vox",
            "zmax_vox", "ymax_vox", "xmax_vox",
            "volume_vox", "zc_vox", "yc_vox", "xc_vox"]
    rows = []
    for hid, tb in tight_bboxes.items():
        rows.append({
            "hcr_id": int(hid),
            "zmin_vox": int(tb["zmin_vox"]), "zmax_vox": int(tb["zmax_vox"]),
            "ymin_vox": int(tb["ymin_vox"]), "ymax_vox": int(tb["ymax_vox"]),
            "xmin_vox": int(tb["xmin_vox"]), "xmax_vox": int(tb["xmax_vox"]),
            "volume_vox": int(tb["volume_vox"]),
            "zc_vox": float(tb["zc_vox"]),
            "yc_vox": float(tb["yc_vox"]),
            "xc_vox": float(tb["xc_vox"]),
        })
    new_df = pd.DataFrame(rows, columns=cols)
    if cache_path.exists():
        old_df = pd.read_parquet(cache_path)
        combined = pd.concat([old_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates("hcr_id", keep="last")
    else:
        combined = new_df
    combined = combined.sort_values("hcr_id").reset_index(drop=True)
    combined.to_parquet(cache_path, index=False)
    print(f"  [{sid}] tight-bbox cache updated: {len(combined)} entries → {cache_path}")


def _write_meta(
    sid: str,
    s: SubjectData,
    channels: List[str],
    probe_ch: str,
    has_pickle: bool,
    n_roi: int,
    elapsed: float,
    level: int,
    seg_ZYX: Tuple[int, int, int],
    ch_arrs: Dict,
) -> None:
    available = [ch for ch in channels if ch_arrs.get(ch) is not None]
    quirks = []
    if not has_pickle:
        quirks.append(
            "No metrics.pickle — tight bboxes from orig_res zarr scan. "
            "Pickle-bbox consistency columns are all True/0."
        )
    missing = [c for c in ["405", "488"] if c not in available]
    if missing:
        quirks.append(f"Missing channels {missing} — intensity NaN.")
    if sid in INTENSITY_SUBJECTS:
        quirks.append("Intensity subject: GFP+ feature is mean_minus_bg, not spot density.")
    quirks.append(
        "All features use segmentation_mask_orig_res.zarr (level-2 XY) "
        "and fused channel zarr level-2. "
        "seg_xy_um = s.hcr_xy_um (level-2 frame)."
    )
    meta = {
        "subject_id": sid,
        "channels_used": available,
        "probe_channel": probe_ch,
        "has_pickle": has_pickle,
        "total_rois": n_roi,
        "extraction_timestamp": datetime.utcnow().isoformat() + "Z",
        "level_recorded": level,
        "intensity_stats_level": 2,
        "seg_zarr_used": "segmentation_mask_orig_res.zarr (level-2 XY)",
        "seg_shape_ZYX": list(seg_ZYX),
        "seg_xy_um": float(s.hcr_xy_um),
        "seg_z_um": float(s.hcr_z_um),
        "elapsed_seconds": float(elapsed),
        "quirks": quirks,
    }
    with open(_meta_path(sid), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  [{sid}] meta saved → {_meta_path(sid)}")
