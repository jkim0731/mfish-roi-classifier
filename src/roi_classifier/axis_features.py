"""v3 axis / projection / waist features for HCR ROI quality classifier.

Pure functions; called per-cell from `feat_axis.py`.

Feature groups
--------------
1. 3D principal-axis profile (PCA on the opened mask, in physical µm coords):
   - eigenvalues (λ1 ≥ λ2 ≥ λ3) and ratios — elongation
   - extent along PC1
   - intensity profile mean(c405) per 1 µm bin along PC1: peak count in
     the inner 60 %, max prominence, 2nd prominence, peak separation,
     min/max ratio (waist signal in *intensity*)
   - cross-section *area* profile along PC1: min/median, min/max, peak
     count, max prominence (waist signal in *shape*)

2. 3D principal-axis profile on the **raw** mask (subset of group 1):
   - extent_um, section min/med, intensity n_peaks_inner

3. 2D projections of the opened mask (xy / yz / zx). For each plane:
   - find the 2D principal axis of the projected mask
   - profile mean(c405) along it: extent, peak count inner, max prom,
     min/max ratio
   - at the main-axis peak strip, take the orthogonal profile and
     report n_peaks and FWHM (dot vs cell-cell-boundary discriminator)

Bins are 1 µm wide.  All physical sizes use the level-2 voxel
spacings supplied by the caller (vz, vy, vx in µm).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


# ──────────────────────────────────────────────────────────────────────────────
# generic profile peak / waist statistics
# ──────────────────────────────────────────────────────────────────────────────

def _peak_stats(
    profile: np.ndarray,
    *,
    inner_lo: float = 0.2,
    inner_hi: float = 0.8,
    bin_um: float = 1.0,
    prom_frac: float = 0.05,
) -> Dict[str, float]:
    """Smooth a 1-D profile and return peak-count / prominence stats over
    its inner ``[inner_lo, inner_hi]`` fraction.

    Output keys (all floats / ints):
        n_peaks_inner, peak_prom_max, peak_prom_2nd, peak_sep_um,
        min_over_max_inner
    """
    out = {
        "n_peaks_inner": 0,
        "peak_prom_max": float("nan"),
        "peak_prom_2nd": float("nan"),
        "peak_sep_um": float("nan"),
        "min_over_max_inner": float("nan"),
    }
    n = len(profile)
    if n < 5 or not np.isfinite(profile).any():
        return out
    arr = np.asarray(profile, dtype=float)
    if not np.isfinite(arr).all():
        arr = np.where(np.isfinite(arr), arr, 0.0)
    sigma = max(1.0, n / 25.0)
    sm = gaussian_filter1d(arr, sigma=sigma)
    rng = float(np.ptp(sm))
    if rng <= 0:
        rng = 1e-9
    lo = max(int(inner_lo * n), 1)
    hi = min(int(inner_hi * n) + 1, n)
    if hi - lo < 3:
        return out
    inner = sm[lo:hi]
    peaks, props = find_peaks(inner, prominence=rng * prom_frac)
    proms = props.get("prominences", np.array([]))
    out["n_peaks_inner"] = int(len(peaks))
    if len(peaks) >= 1:
        order = np.argsort(-proms)
        out["peak_prom_max"] = float(proms[order[0]] / rng)
        if len(peaks) >= 2:
            out["peak_prom_2nd"] = float(proms[order[1]] / rng)
            sep_bins = abs(int(peaks[order[0]]) - int(peaks[order[1]]))
            out["peak_sep_um"] = float(sep_bins * bin_um)
    if inner.max() > 0:
        out["min_over_max_inner"] = float(inner.min() / inner.max())
    return out


def _waist_stats(area_per_bin: np.ndarray) -> Dict[str, float]:
    """Min/median, min/max ratios of cross-section areas (the *interior* min
    excludes the two end bins to avoid the natural taper at the cell tips)."""
    out = {
        "section_min_over_med_area": float("nan"),
        "section_min_over_max_area": float("nan"),
    }
    a = np.asarray(area_per_bin, dtype=float)
    if a.size < 3:
        return out
    inner = a[1:-1] if a.size >= 3 else a
    inner = inner[inner > 0] if inner.size > 0 else inner
    if inner.size == 0:
        return out
    med = float(np.median(a[a > 0])) if (a > 0).any() else 0.0
    mx = float(a.max())
    mn_inner = float(inner.min())
    if med > 0:
        out["section_min_over_med_area"] = mn_inner / med
    if mx > 0:
        out["section_min_over_max_area"] = mn_inner / mx
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 3D principal-axis profile
# ──────────────────────────────────────────────────────────────────────────────

def _principal_axis_profile_3d(
    mask: np.ndarray,
    img: Optional[np.ndarray],
    vz: float, vy: float, vx: float,
    bin_um: float = 1.0,
) -> Optional[Dict[str, np.ndarray]]:
    """Project mask voxels onto their largest-PCA axis (in µm).

    Returns dict with eigenvalues (ascending), extent_um, area_per_bin,
    mean_intens_per_bin (NaN if img is None) — or None if cell too small.
    """
    zz, yy, xx = np.where(mask)
    n = zz.size
    if n < 8:
        return None
    pts = np.column_stack([zz * vz, yy * vy, xx * vx]).astype(float)
    cen = pts.mean(0)
    X = pts - cen
    cov = X.T @ X / max(n - 1, 1)
    w, v = np.linalg.eigh(cov)  # ascending eigenvalues
    axis = v[:, -1]
    proj = X @ axis
    p_lo = float(proj.min())
    p_hi = float(proj.max())
    extent = p_hi - p_lo
    nbins = max(int(np.ceil(extent / bin_um)), 4)
    edges = np.linspace(p_lo, p_hi + 1e-6, nbins + 1)
    idx = np.clip(np.digitize(proj, edges) - 1, 0, nbins - 1)
    area_per_bin = np.bincount(idx, minlength=nbins).astype(float)
    if img is not None:
        intens = img[mask].astype(float)
        intens_sum = np.bincount(idx, weights=intens, minlength=nbins)
        with np.errstate(invalid="ignore"):
            mean_per_bin = intens_sum / np.maximum(area_per_bin, 1.0)
    else:
        mean_per_bin = np.full(nbins, np.nan, dtype=float)
    return {
        "extent_um": float(extent),
        "lambdas_um2": w,            # ascending
        "area_per_bin": area_per_bin,
        "mean_intens_per_bin": mean_per_bin,
    }


def axis_features_3d(
    mask_opened_tight: np.ndarray,
    mask_raw_tight: np.ndarray,
    img_tight: Optional[np.ndarray],
    vz: float, vy: float, vx: float,
    bin_um: float = 1.0,
) -> Dict[str, float]:
    """Compute 3D PCA + axis profile features for opened (rich) and raw (subset)
    masks.  Returns ~22 features."""
    out: Dict[str, float] = {
        # opened — eigvals
        "axis3d_extent_um": float("nan"),
        "axis3d_lambda1_um2": float("nan"),
        "axis3d_lambda2_um2": float("nan"),
        "axis3d_lambda3_um2": float("nan"),
        "axis3d_lambda_ratio_l1_l3": float("nan"),
        "axis3d_lambda_ratio_l1_l2": float("nan"),
        # opened — intensity profile peaks
        "axis3d_n_peaks_inner_intens": 0,
        "axis3d_peak_prom_max_intens": float("nan"),
        "axis3d_peak_prom_2nd_intens": float("nan"),
        "axis3d_peak_sep_um_intens": float("nan"),
        "axis3d_min_over_max_inner_intens": float("nan"),
        # opened — section area peaks/waist
        "axis3d_section_min_over_med_area": float("nan"),
        "axis3d_section_min_over_max_area": float("nan"),
        "axis3d_n_peaks_inner_area": 0,
        "axis3d_peak_prom_max_area": float("nan"),
        # raw subset
        "axis3d_raw_extent_um": float("nan"),
        "axis3d_raw_section_min_over_med_area": float("nan"),
        "axis3d_raw_n_peaks_inner_intens": 0,
        "axis3d_raw_peak_prom_max_intens": float("nan"),
    }
    res_o = _principal_axis_profile_3d(
        mask_opened_tight, img_tight, vz, vy, vx, bin_um,
    )
    if res_o is not None:
        l = res_o["lambdas_um2"]
        out["axis3d_extent_um"] = res_o["extent_um"]
        out["axis3d_lambda1_um2"] = float(l[2])
        out["axis3d_lambda2_um2"] = float(l[1])
        out["axis3d_lambda3_um2"] = float(l[0])
        out["axis3d_lambda_ratio_l1_l3"] = float(l[2] / max(l[0], 1e-9))
        out["axis3d_lambda_ratio_l1_l2"] = float(l[2] / max(l[1], 1e-9))
        ip = _peak_stats(res_o["mean_intens_per_bin"], bin_um=bin_um)
        out["axis3d_n_peaks_inner_intens"] = ip["n_peaks_inner"]
        out["axis3d_peak_prom_max_intens"] = ip["peak_prom_max"]
        out["axis3d_peak_prom_2nd_intens"] = ip["peak_prom_2nd"]
        out["axis3d_peak_sep_um_intens"] = ip["peak_sep_um"]
        out["axis3d_min_over_max_inner_intens"] = ip["min_over_max_inner"]
        ws = _waist_stats(res_o["area_per_bin"])
        out["axis3d_section_min_over_med_area"] = ws["section_min_over_med_area"]
        out["axis3d_section_min_over_max_area"] = ws["section_min_over_max_area"]
        ap = _peak_stats(res_o["area_per_bin"], bin_um=bin_um)
        # interpret area peaks as "lobes" — use interior peaks
        out["axis3d_n_peaks_inner_area"] = ap["n_peaks_inner"]
        out["axis3d_peak_prom_max_area"] = ap["peak_prom_max"]

    res_r = _principal_axis_profile_3d(
        mask_raw_tight, img_tight, vz, vy, vx, bin_um,
    )
    if res_r is not None:
        out["axis3d_raw_extent_um"] = res_r["extent_um"]
        wr = _waist_stats(res_r["area_per_bin"])
        out["axis3d_raw_section_min_over_med_area"] = wr["section_min_over_med_area"]
        ir = _peak_stats(res_r["mean_intens_per_bin"], bin_um=bin_um)
        out["axis3d_raw_n_peaks_inner_intens"] = ir["n_peaks_inner"]
        out["axis3d_raw_peak_prom_max_intens"] = ir["peak_prom_max"]
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 2D projection profiles (xy / yz / zx)
# ──────────────────────────────────────────────────────────────────────────────

def _2d_axis_features(
    mask2d: np.ndarray,
    intens2d_sum: np.ndarray,
    plane_name: str,
    voxel_um_along_axes: Tuple[float, float],
    bin_um: float = 1.0,
) -> Dict[str, float]:
    """For a 2D MIP mask + summed intensity (sum_along_projection_axis),
    compute principal-axis profile + orthogonal profile at the main peak."""
    keys = [
        f"proj_{plane_name}_main_extent_um",
        f"proj_{plane_name}_main_n_peaks_inner",
        f"proj_{plane_name}_main_peak_prom_max",
        f"proj_{plane_name}_main_min_over_max_inner",
        f"proj_{plane_name}_orth_n_peaks_at_main",
        f"proj_{plane_name}_orth_fwhm_um",
        f"proj_{plane_name}_aspect_ratio",
    ]
    out = {
        k: (0 if "n_peaks" in k else float("nan"))
        for k in keys
    }
    # orth_n_peaks_at_main is in _DROPPED_PEAK_COLS and always NaN
    out[f"proj_{plane_name}_orth_n_peaks_at_main"] = float("nan")
    yy, xx = np.where(mask2d)
    n = yy.size
    if n < 8:
        return out
    v0, v1 = voxel_um_along_axes
    pts = np.column_stack([yy * v0, xx * v1]).astype(float)
    cen = pts.mean(0)
    X = pts - cen
    cov = X.T @ X / max(n - 1, 1)
    w, v = np.linalg.eigh(cov)  # w ascending
    main = v[:, 1]; orth = v[:, 0]
    if w[0] > 0:
        out[f"proj_{plane_name}_aspect_ratio"] = float(w[1] / max(w[0], 1e-9))

    proj_main = X @ main
    proj_orth = X @ orth
    pm_lo = float(proj_main.min()); pm_hi = float(proj_main.max())
    extent_main = pm_hi - pm_lo
    out[f"proj_{plane_name}_main_extent_um"] = extent_main

    nbins = max(int(np.ceil(extent_main / bin_um)), 4)
    edges = np.linspace(pm_lo, pm_hi + 1e-6, nbins + 1)
    idx = np.clip(np.digitize(proj_main, edges) - 1, 0, nbins - 1)
    area_per_bin = np.bincount(idx, minlength=nbins).astype(float)
    intens_at_pix = intens2d_sum[mask2d].astype(float)
    intens_sum_per_bin = np.bincount(idx, weights=intens_at_pix, minlength=nbins)
    with np.errstate(invalid="ignore"):
        mean_per_bin = intens_sum_per_bin / np.maximum(area_per_bin, 1.0)

    pk = _peak_stats(mean_per_bin, bin_um=bin_um)
    out[f"proj_{plane_name}_main_n_peaks_inner"] = pk["n_peaks_inner"]
    out[f"proj_{plane_name}_main_peak_prom_max"] = pk["peak_prom_max"]
    out[f"proj_{plane_name}_main_min_over_max_inner"] = pk["min_over_max_inner"]

    # orthogonal profile at the main-peak bin (within inner 60 %)
    lo_b = max(int(0.2 * nbins), 1)
    hi_b = min(int(0.8 * nbins) + 1, nbins)
    inner_mean = mean_per_bin[lo_b:hi_b]
    if inner_mean.size > 0 and np.isfinite(inner_mean).any():
        peak_bin = lo_b + int(np.nanargmax(inner_mean))
        in_bin = (idx == peak_bin)
        if in_bin.sum() >= 8:
            orth_proj = proj_orth[in_bin]
            po_lo = float(orth_proj.min()); po_hi = float(orth_proj.max())
            extent_o = po_hi - po_lo
            nb_o = max(int(np.ceil(extent_o / bin_um)), 4)
            edges_o = np.linspace(po_lo, po_hi + 1e-6, nb_o + 1)
            idx_o = np.clip(np.digitize(orth_proj, edges_o) - 1, 0, nb_o - 1)
            yy_b = yy[in_bin]; xx_b = xx[in_bin]
            intens_b = intens2d_sum[yy_b, xx_b].astype(float)
            int_sum_o = np.bincount(idx_o, weights=intens_b, minlength=nb_o)
            ar_o = np.bincount(idx_o, minlength=nb_o).astype(float)
            with np.errstate(invalid="ignore"):
                mean_o = int_sum_o / np.maximum(ar_o, 1.0)
            if mean_o.size > 2 and np.isfinite(mean_o).any():
                mn = float(np.nanmin(mean_o))
                mx = float(np.nanmax(mean_o))
                if mx > mn:
                    half = mn + 0.5 * (mx - mn)
                    above = mean_o >= half
                    if above.any():
                        ai = np.where(above)[0]
                        out[f"proj_{plane_name}_orth_fwhm_um"] = float(
                            (ai.max() - ai.min() + 1) * bin_um
                        )
    return out


def projection_axis_features(
    mask_opened_tight: np.ndarray,
    img_tight: Optional[np.ndarray],
    vz: float, vy: float, vx: float,
    bin_um: float = 1.0,
) -> Dict[str, float]:
    """Compute 2D-projection axis features on the OPENED mask for xy/yz/zx
    planes (~21 features total).  n_peaks columns in _DROPPED_PEAK_COLS are
    always NaN (set by the caller, compute_axis_features)."""
    out: Dict[str, float] = {}
    if mask_opened_tight.size == 0 or not mask_opened_tight.any():
        for plane in ("xy", "yz", "zx"):
            for k in [
                f"proj_{plane}_main_extent_um",
                f"proj_{plane}_main_n_peaks_inner",
                f"proj_{plane}_main_peak_prom_max",
                f"proj_{plane}_main_min_over_max_inner",
                f"proj_{plane}_orth_n_peaks_at_main",
                f"proj_{plane}_orth_fwhm_um",
                f"proj_{plane}_aspect_ratio",
            ]:
                out[k] = 0 if "n_peaks" in k else float("nan")
        return out

    if img_tight is not None:
        intens_in = (img_tight.astype(np.float32) * mask_opened_tight)
    else:
        intens_in = mask_opened_tight.astype(np.float32)  # equivalent to area count

    # xy: project along z (axis 0) — pixels (y, x), µm (vy, vx)
    mask_xy = mask_opened_tight.any(axis=0)
    intens_xy = intens_in.sum(axis=0)
    out.update(_2d_axis_features(mask_xy, intens_xy, "xy", (vy, vx), bin_um))

    # yz: project along x (axis 2) — pixels (z, y), µm (vz, vy)
    mask_yz = mask_opened_tight.any(axis=2)
    intens_yz = intens_in.sum(axis=2)
    out.update(_2d_axis_features(mask_yz, intens_yz, "yz", (vz, vy), bin_um))

    # zx: project along y (axis 1) — pixels (z, x), µm (vz, vx)
    mask_zx = mask_opened_tight.any(axis=1)
    intens_zx = intens_in.sum(axis=1)
    out.update(_2d_axis_features(mask_zx, intens_zx, "zx", (vz, vx), bin_um))

    return out


# ──────────────────────────────────────────────────────────────────────────────
# top-level wrapper called per-cell from feat_axis
# ──────────────────────────────────────────────────────────────────────────────

_DROPPED_PEAK_COLS = frozenset({
    "axis3d_n_peaks_inner_intens",
    "axis3d_peak_prom_2nd_intens",
    "axis3d_n_peaks_inner_area",
    "axis3d_raw_n_peaks_inner_intens",
    "proj_xy_main_n_peaks_inner",
    "proj_yz_main_n_peaks_inner",
    "proj_zx_main_n_peaks_inner",
    "proj_xy_orth_n_peaks_at_main",
    "proj_yz_orth_n_peaks_at_main",
    "proj_zx_orth_n_peaks_at_main",
})


def compute_axis_features(
    mask_raw_tight: np.ndarray,
    mask_opened_tight: np.ndarray,
    img_tight: Optional[np.ndarray],
    vz: float, vy: float, vx: float,
    bin_um: float = 1.0,
    compute_dropped_peaks: bool = False,
) -> Dict[str, float]:
    feats = axis_features_3d(
        mask_opened_tight, mask_raw_tight, img_tight, vz, vy, vx, bin_um=bin_um,
    )
    feats.update(projection_axis_features(
        mask_opened_tight, img_tight, vz, vy, vx, bin_um=bin_um,
    ))
    # _DROPPED_PEAK_COLS are low-gain features; always set to NaN.
    for col in _DROPPED_PEAK_COLS:
        feats[col] = float("nan")
    return feats


def feature_columns() -> list:
    """Stable list of v3-NEW feature column names (used to build NaN rows for
    cells we never own)."""
    cols = [
        "axis3d_extent_um", "axis3d_lambda1_um2", "axis3d_lambda2_um2", "axis3d_lambda3_um2",
        "axis3d_lambda_ratio_l1_l3", "axis3d_lambda_ratio_l1_l2",
        "axis3d_n_peaks_inner_intens", "axis3d_peak_prom_max_intens",
        "axis3d_peak_prom_2nd_intens", "axis3d_peak_sep_um_intens",
        "axis3d_min_over_max_inner_intens",
        "axis3d_section_min_over_med_area", "axis3d_section_min_over_max_area",
        "axis3d_n_peaks_inner_area", "axis3d_peak_prom_max_area",
        "axis3d_raw_extent_um", "axis3d_raw_section_min_over_med_area",
        "axis3d_raw_n_peaks_inner_intens", "axis3d_raw_peak_prom_max_intens",
    ]
    for plane in ("xy", "yz", "zx"):
        cols += [
            f"proj_{plane}_main_extent_um",
            f"proj_{plane}_main_n_peaks_inner",
            f"proj_{plane}_main_peak_prom_max",
            f"proj_{plane}_main_min_over_max_inner",
            f"proj_{plane}_orth_n_peaks_at_main",
            f"proj_{plane}_orth_fwhm_um",
            f"proj_{plane}_aspect_ratio",
        ]
    return cols
