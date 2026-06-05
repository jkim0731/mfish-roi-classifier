"""Per-ROI v5 features: protrusion-touches-other-ROI.

Definition: protrusion = mask_raw & ~mask_opened (voxels removed by binary
opening). Question: do those protrusion voxels' immediate neighbors include
other segmented ROIs (i.e., is the protrusion *into* another cell)?

Method: dilate the protrusion by 1 voxel (3D cross), intersect with the seg
label volume; count voxels whose label is neither 0 nor the host hcr_id.

Features (all defined; NaN when n_protrusion_voxels==0):
  - n_protrusion_voxels                        : int
  - protrusion_rim_voxels                      : int (1-vox dilation \ protrusion)
  - protrusion_rim_other_frac                  : float in [0, 1]
  - protrusion_rim_bg_frac                     : float in [0, 1]
  - protrusion_top_neighbor_frac               : float in [0, 1]
  - n_distinct_neighbor_ids_at_protrusion      : int
  - protrusion_touches_other                   : 0/1 (any other ROI in rim)
  - protrusion_into_neighbor_frac              : top-neighbor count / n_protrusion_voxels

The protrusion *rim* is computed *outside* the host raw mask (so we measure
what the protrusion is *adjacent to*, not what it overlaps with — by
construction the protrusion is part of mask_raw, so seg there equals hcr_id).
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import scipy.ndimage as ndi

# Re-use v2's connectivity-1 cross structure for consistency.
_CROSS_3D = ndi.generate_binary_structure(3, 1)


def feature_columns():
    return [
        "n_protrusion_voxels",
        "protrusion_voxel_frac",
        "protrusion_rim_voxels",
        "protrusion_rim_other_frac",
        "protrusion_rim_bg_frac",
        "protrusion_top_neighbor_frac",
        "n_distinct_neighbor_ids_at_protrusion",
        "protrusion_touches_other",
        "protrusion_into_neighbor_frac",
    ]


def _nan_row():
    return {c: float("nan") for c in feature_columns()}


def protrusion_features(
    mask_raw_pad: np.ndarray,
    mask_opened_pad: np.ndarray,
    seg_pad: np.ndarray,
    hcr_id: int,
) -> Dict[str, float]:
    """All inputs share the same padded shape. seg_pad is int label volume."""
    out = _nan_row()

    if mask_raw_pad.shape != mask_opened_pad.shape or mask_raw_pad.shape != seg_pad.shape:
        return out

    n_raw = int(mask_raw_pad.sum())
    if n_raw == 0:
        return out

    protrusion = mask_raw_pad & ~mask_opened_pad
    n_pro = int(protrusion.sum())
    out["n_protrusion_voxels"] = float(n_pro)
    out["protrusion_voxel_frac"] = float(n_pro) / float(n_raw)

    if n_pro == 0:
        out["protrusion_rim_voxels"] = 0.0
        out["protrusion_rim_other_frac"] = 0.0
        out["protrusion_rim_bg_frac"] = float("nan")
        out["protrusion_top_neighbor_frac"] = 0.0
        out["n_distinct_neighbor_ids_at_protrusion"] = 0.0
        out["protrusion_touches_other"] = 0.0
        out["protrusion_into_neighbor_frac"] = 0.0
        return out

    # Rim = (dilate protrusion 1) \ host_raw_mask. We exclude the host's own
    # voxels so we measure what's *adjacent* to the protrusion, not its
    # internal label (which is hcr_id by construction).
    pro_dil = ndi.binary_dilation(protrusion, structure=_CROSS_3D, iterations=1)
    rim = pro_dil & ~mask_raw_pad
    n_rim = int(rim.sum())
    out["protrusion_rim_voxels"] = float(n_rim)

    if n_rim == 0:
        out["protrusion_rim_other_frac"] = 0.0
        out["protrusion_rim_bg_frac"] = 0.0
        out["protrusion_top_neighbor_frac"] = 0.0
        out["n_distinct_neighbor_ids_at_protrusion"] = 0.0
        out["protrusion_touches_other"] = 0.0
        out["protrusion_into_neighbor_frac"] = 0.0
        return out

    rim_labels = seg_pad[rim]
    bg = int((rim_labels == 0).sum())
    fg_mask = (rim_labels != 0) & (rim_labels != hcr_id)
    fg_ids = rim_labels[fg_mask]
    out["protrusion_rim_bg_frac"] = float(bg) / float(n_rim)
    out["protrusion_rim_other_frac"] = float(fg_ids.size) / float(n_rim)

    if fg_ids.size == 0:
        out["protrusion_top_neighbor_frac"] = 0.0
        out["n_distinct_neighbor_ids_at_protrusion"] = 0.0
        out["protrusion_touches_other"] = 0.0
        out["protrusion_into_neighbor_frac"] = 0.0
        return out

    uniq, counts = np.unique(fg_ids, return_counts=True)
    top = int(counts.max())
    out["protrusion_top_neighbor_frac"] = float(top) / float(n_rim)
    out["n_distinct_neighbor_ids_at_protrusion"] = float(uniq.size)
    out["protrusion_touches_other"] = 1.0
    out["protrusion_into_neighbor_frac"] = float(top) / float(n_pro)
    return out
