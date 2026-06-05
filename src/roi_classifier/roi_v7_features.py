"""Per-ROI v7 features: image-quality (intra-soma contrast + boundary sharpness).

Motivation: contrast inside the soma and across its mask boundary is the
upstream cause of a `bad_ok` call (image dim/blurry → seg slightly wrong but
the cell is still usable downstream). v4 core/shell captures *morphological*
nucleus-vs-cytoplasm contrast; v7 captures *image-quality* contrast.

All features computed on tight (mask_opened, img_405) crops:

  intra-soma stats over `mask_opened`:
    intra_soma_n_voxels, intra_soma_mean_405, intra_soma_sd_405,
    intra_soma_p10_405, intra_soma_p50_405, intra_soma_p90_405,
    intra_soma_p90_minus_p10_405, intra_soma_cv_405,
    intra_soma_iqr_over_median_405

  boundary contrast (1-vox shell just inside vs 1-vox just outside):
    inside_shell_mean_405, outside_shell_mean_405,
    inside_outside_diff_mean_405, inside_outside_ratio_mean_405,
    inside_shell_p50_405, outside_shell_p50_405,
    inside_outside_diff_p50_405,

  boundary sharpness (Sobel magnitude on inside_shell positions):
    boundary_grad_mean_405, boundary_grad_p50_405, boundary_grad_p90_405

NaN policy: if `mask_opened` is empty or `img_405` is None, every feature is
NaN. Empty inside/outside shells (e.g. mask is one voxel thick) → those
shell-derived stats are NaN, but intra-soma stats still defined.

Inputs are voxel arrays at level-2 resolution; voxel sizes (vz, vy, vx) in µm
are accepted but not currently used (intensity stats are unitless; the Sobel
kernel is applied in voxel units).
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import scipy.ndimage as ndi

_CROSS_3D = ndi.generate_binary_structure(3, 1)


def feature_columns():
    return [
        # intra-soma
        "intra_soma_n_voxels",
        "intra_soma_mean_405",
        "intra_soma_sd_405",
        "intra_soma_p10_405",
        "intra_soma_p50_405",
        "intra_soma_p90_405",
        "intra_soma_p90_minus_p10_405",
        "intra_soma_cv_405",
        "intra_soma_iqr_over_median_405",
        # boundary contrast
        "inside_shell_mean_405",
        "outside_shell_mean_405",
        "inside_outside_diff_mean_405",
        "inside_outside_ratio_mean_405",
        "inside_shell_p50_405",
        "outside_shell_p50_405",
        "inside_outside_diff_p50_405",
        # boundary sharpness
        "boundary_grad_mean_405",
        "boundary_grad_p50_405",
        "boundary_grad_p90_405",
    ]


def _nan_row():
    return {c: float("nan") for c in feature_columns()}


def _safe_cv(mean_v: float, sd_v: float) -> float:
    if not np.isfinite(mean_v) or mean_v <= 0:
        return float("nan")
    return float(sd_v) / float(mean_v)


def _safe_ratio(num: float, den: float) -> float:
    if not np.isfinite(den) or den <= 0:
        return float("nan")
    return float(num) / float(den)


def image_quality_features(
    mask_opened: np.ndarray,
    img_405: Optional[np.ndarray],
    vz: float = 1.0,
    vy: float = 1.0,
    vx: float = 1.0,
) -> Dict[str, float]:
    """Tight-crop image-quality features. Returns NaN row if inputs missing."""
    out = _nan_row()
    if img_405 is None or mask_opened is None:
        return out
    if mask_opened.shape != img_405.shape:
        return out
    if not mask_opened.any():
        return out

    img = img_405.astype(np.float32, copy=False)

    soma_vals = img[mask_opened]
    n_soma = int(soma_vals.size)
    out["intra_soma_n_voxels"] = float(n_soma)

    mean_v = float(soma_vals.mean())
    sd_v = float(soma_vals.std(ddof=0))
    p10, p25, p50, p75, p90 = np.percentile(soma_vals, [10, 25, 50, 75, 90])
    out["intra_soma_mean_405"] = mean_v
    out["intra_soma_sd_405"] = sd_v
    out["intra_soma_p10_405"] = float(p10)
    out["intra_soma_p50_405"] = float(p50)
    out["intra_soma_p90_405"] = float(p90)
    out["intra_soma_p90_minus_p10_405"] = float(p90) - float(p10)
    out["intra_soma_cv_405"] = _safe_cv(mean_v, sd_v)
    out["intra_soma_iqr_over_median_405"] = _safe_ratio(
        float(p75) - float(p25), float(p50)
    )

    eroded = ndi.binary_erosion(
        mask_opened, structure=_CROSS_3D, iterations=1, border_value=0
    )
    inside_shell = mask_opened & ~eroded
    dilated = ndi.binary_dilation(
        mask_opened, structure=_CROSS_3D, iterations=1, border_value=0
    )
    outside_shell = dilated & ~mask_opened

    if inside_shell.any():
        in_vals = img[inside_shell]
        in_mean = float(in_vals.mean())
        in_p50 = float(np.median(in_vals))
        out["inside_shell_mean_405"] = in_mean
        out["inside_shell_p50_405"] = in_p50
    else:
        in_mean = float("nan")
        in_p50 = float("nan")

    if outside_shell.any():
        out_vals = img[outside_shell]
        out_mean = float(out_vals.mean())
        out_p50 = float(np.median(out_vals))
        out["outside_shell_mean_405"] = out_mean
        out["outside_shell_p50_405"] = out_p50
    else:
        out_mean = float("nan")
        out_p50 = float("nan")

    if np.isfinite(in_mean) and np.isfinite(out_mean):
        out["inside_outside_diff_mean_405"] = in_mean - out_mean
        out["inside_outside_ratio_mean_405"] = _safe_ratio(in_mean, out_mean)
    if np.isfinite(in_p50) and np.isfinite(out_p50):
        out["inside_outside_diff_p50_405"] = in_p50 - out_p50

    if inside_shell.any():
        # Sobel gradient magnitude on the full tight crop, sampled at the
        # inside-shell voxels (one-voxel-thick surface inside the mask).
        gz = ndi.sobel(img, axis=0, mode="reflect")
        gy = ndi.sobel(img, axis=1, mode="reflect")
        gx = ndi.sobel(img, axis=2, mode="reflect")
        grad_mag = np.sqrt(gz * gz + gy * gy + gx * gx)
        gvals = grad_mag[inside_shell]
        if gvals.size > 0:
            out["boundary_grad_mean_405"] = float(gvals.mean())
            out["boundary_grad_p50_405"] = float(np.median(gvals))
            out["boundary_grad_p90_405"] = float(np.percentile(gvals, 90))

    return out
