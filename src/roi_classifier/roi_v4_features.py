"""v4-NEW per-ROI features for the HCR ROI pass/fail classifier (S11).

Adds two families to v2+v3:

1. Surface area + sphericity (per-mask geometry, no intensity needed)
   - `surface_area_um2_raw`, `surface_area_um2_opened`
   - `volume_um3_raw`,        `volume_um3_opened`
   - `sa_to_vol_um_inv_raw`,  `sa_to_vol_um_inv_opened`        (= SA/V)
   - `sphericity_raw`,        `sphericity_opened`              (in [0, 1])

2. Calibrated 4-µm core/shell intensity (replaces v2's 1-voxel erosion)
   - `c405_core4um_p50_opened`, `c405_shell4um_p50_opened`
   - `c405_shell_minus_core4um_p50`
   - `c405_shell_minus_core4um_p90`
   - `c405_shell_over_core4um_p50_ratio`
   - `core4um_voxel_frac_opened` (fraction of opened-mask voxels in calibrated core)

The 4-µm radius comes from `08_core_shell_calibration` (pooled r_thresh = 4 µm
across 6 subjects; per-subject 3-4 µm).  We use anisotropic Euclidean
distance-transform so the radius is physically µm, not voxels.

Surface area is computed on the padded crop using face-counting:
  for each axis, count XOR-flips between adjacent slabs and weight by the
  cross-section voxel-face area in µm².

Inputs to `all_v4_features`:
    mask_raw_tight    : bool[Z,Y,X] tight crop (raw seg == hid)
    mask_opened_tight : bool[Z,Y,X] tight crop after binary opening
    img_tight         : uint16[Z,Y,X] c405 in the tight crop  (None → NaN intensities)
    vz, vy, vx        : voxel size in µm
    r_core_um         : calibrated core radius (default 4.0)
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import scipy.ndimage as ndi

R_CORE_UM_DEFAULT = 4.0


def _percentile_or_nan(arr: np.ndarray, q: float) -> float:
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q))


def surface_area_um2(mask: np.ndarray, vz: float, vy: float, vx: float) -> float:
    """Total external surface area (µm²) by voxel-face counting.

    Faces are counted between True voxels and False (or out-of-bounds) voxels.
    XY-faces (perpendicular to z) have area vy*vx; XZ-faces (perp to y) have
    area vz*vx; YZ-faces (perp to x) have area vz*vy.
    """
    if not mask.any():
        return 0.0
    p = np.pad(mask, 1, constant_values=False)
    fz = int(np.logical_xor(p[1:, :, :], p[:-1, :, :]).sum())
    fy = int(np.logical_xor(p[:, 1:, :], p[:, :-1, :]).sum())
    fx = int(np.logical_xor(p[:, :, 1:], p[:, :, :-1]).sum())
    return float(fz * vy * vx + fy * vz * vx + fx * vz * vy)


def volume_um3(mask: np.ndarray, vz: float, vy: float, vx: float) -> float:
    return float(int(mask.sum()) * vz * vy * vx)


def sphericity(volume: float, surface: float) -> float:
    if surface <= 0 or volume <= 0:
        return float("nan")
    return float((np.pi ** (1.0 / 3.0) * (6.0 * volume) ** (2.0 / 3.0)) / surface)


def _calibrated_core(
    mask: np.ndarray, vz: float, vy: float, vx: float, r_core_um: float,
) -> np.ndarray:
    """Voxels of `mask` whose anisotropic distance to the boundary ≥ r_core_um."""
    if not mask.any():
        return np.zeros_like(mask)
    dist = ndi.distance_transform_edt(mask, sampling=(vz, vy, vx))
    return dist >= r_core_um


def core_shell_intensity_features(
    mask_opened: np.ndarray,
    img: Optional[np.ndarray],
    vz: float, vy: float, vx: float,
    r_core_um: float = R_CORE_UM_DEFAULT,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if mask_opened.any():
        core = _calibrated_core(mask_opened, vz, vy, vx, r_core_um)
        shell = mask_opened & ~core
        out["core4um_voxel_frac_opened"] = float(core.sum()) / float(mask_opened.sum())
    else:
        core = np.zeros_like(mask_opened)
        shell = np.zeros_like(mask_opened)
        out["core4um_voxel_frac_opened"] = float("nan")

    if img is None or not mask_opened.any():
        out["c405_core4um_p50_opened"] = float("nan")
        out["c405_shell4um_p50_opened"] = float("nan")
        out["c405_shell_minus_core4um_p50"] = float("nan")
        out["c405_shell_minus_core4um_p90"] = float("nan")
        out["c405_shell_over_core4um_p50_ratio"] = float("nan")
        return out

    px_core = img[core] if core.any() else np.array([], dtype=np.float32)
    px_shell = img[shell] if shell.any() else np.array([], dtype=np.float32)
    core_p50 = _percentile_or_nan(px_core, 50)
    shell_p50 = _percentile_or_nan(px_shell, 50)
    core_p90 = _percentile_or_nan(px_core, 90)
    shell_p90 = _percentile_or_nan(px_shell, 90)

    out["c405_core4um_p50_opened"] = core_p50
    out["c405_shell4um_p50_opened"] = shell_p50
    out["c405_shell_minus_core4um_p50"] = (
        shell_p50 - core_p50 if np.isfinite(shell_p50) and np.isfinite(core_p50) else float("nan")
    )
    out["c405_shell_minus_core4um_p90"] = (
        shell_p90 - core_p90 if np.isfinite(shell_p90) and np.isfinite(core_p90) else float("nan")
    )
    out["c405_shell_over_core4um_p50_ratio"] = (
        float(shell_p50 / core_p50) if np.isfinite(shell_p50) and np.isfinite(core_p50)
        and core_p50 > 0 else float("nan")
    )
    return out


def shape_features(
    mask: np.ndarray, vz: float, vy: float, vx: float, prefix: str,
) -> Dict[str, float]:
    # surface_area_um2 and volume_um3 are intermediate values only used for
    # sphericity; the µm outputs were in DROP_UM_FEATURES and are not returned.
    sa = surface_area_um2(mask, vz, vy, vx)
    vol = volume_um3(mask, vz, vy, vx)
    sph = sphericity(vol, sa) if sa > 0 and vol > 0 else float("nan")
    return {f"sphericity_{prefix}": sph}


def all_v4_features(
    mask_raw_tight: np.ndarray,
    mask_opened_tight: np.ndarray,
    img_tight: Optional[np.ndarray],
    vz: float, vy: float, vx: float,
    r_core_um: float = R_CORE_UM_DEFAULT,
) -> Dict[str, float]:
    # img_tight and r_core_um are accepted for API compatibility but not used;
    # the µm core/shell outputs were in DROP_UM_FEATURES.
    out: Dict[str, float] = {}
    out.update(shape_features(mask_raw_tight, vz, vy, vx, prefix="raw"))
    out.update(shape_features(mask_opened_tight, vz, vy, vx, prefix="opened"))
    return out


def feature_columns() -> list[str]:
    return ["sphericity_raw", "sphericity_opened"]
