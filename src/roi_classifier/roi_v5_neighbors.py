"""Per-ROI neighbor lists for stage-3 iterative prediction.

For each cell, dumps the long-format list of touching neighbors:
    (hcr_id, neighbor_id, overlap_voxels, rim_voxels)

where overlap_voxels = count of voxels in the cell's 1-vox raw-rim dilation
that carry segmentation label `neighbor_id`. Background voxels are NOT
included; rows where neighbor_id == hcr_id are dropped (host's own voxels
should not appear in the rim by construction, but we filter defensively).

Definitions match v2's `top_neighbor_overlap_frac` rim:
    rim = ndi.binary_dilation(mask_raw, cross-3D, iter=1) & ~mask_raw

Output: `cached_roi_quality/{sid}_neighbors_v1.parquet` with one row per
(hcr_id, neighbor_id) pair.
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

warnings.filterwarnings("ignore", category=UserWarning, module="zarr")

from .benchmark_data_loader import load_subject
from .roi_quality_v2 import STRIP_Z, Z_PAD, _CROSS_3D, _orig_res_path
from . import config as _cfg

PER_CELL_CROPS = _cfg.PER_CELL_CROPS_DIR
ROI_QUALITY_CACHE = _cfg.ROI_QUALITY_DIR


def _neighbors_cache_path(sid: str) -> Path:
    return ROI_QUALITY_CACHE / f"{sid}_neighbors_v1.parquet"


def _meta_path(sid: str) -> Path:
    return ROI_QUALITY_CACHE / f"{sid}_neighbors_v1_meta.json"


def _decode_mask(row: pd.Series) -> np.ndarray:
    n = int(row["dz"]) * int(row["dy"]) * int(row["dx"])
    bits = np.unpackbits(np.frombuffer(row["mask_packed"], dtype=np.uint8))[:n]
    return bits.astype(bool).reshape(int(row["dz"]), int(row["dy"]), int(row["dx"]))


def _extract_neighbors_subject(sid: str, force: bool = False) -> Dict:
    out_path = _neighbors_cache_path(sid)
    if out_path.exists() and not force:
        return {"sid": sid, "skipped": "cache_hit", "path": str(out_path)}

    crops_path = PER_CELL_CROPS / f"{sid}_per_cell_crops.parquet"
    if not crops_path.exists():
        return {"sid": sid, "error": f"missing crops: {crops_path}"}

    s = load_subject(sid)
    seg_orig = zarr.open(str(_orig_res_path(s)), mode="r")
    _, _, Z_seg, Y_seg, X_seg = seg_orig.shape

    crops_df = pd.read_parquet(crops_path)
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

    rows: List[Dict] = []
    n_done = 0
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
        sub_y0 = max(0, y_min); sub_y1 = min(Y_seg, y_max)
        sub_x0 = max(0, x_min); sub_x1 = min(X_seg, x_max)

        seg_block = np.asarray(
            seg_orig[0, 0, z0_load:z1_load, sub_y0:sub_y1, sub_x0:sub_x1]
        )

        for _, row in sub.iterrows():
            hid = int(row["hcr_id"])
            mask_pad = _decode_mask(row)
            if not mask_pad.any():
                continue
            rim = ndi.binary_dilation(mask_pad, structure=_CROSS_3D, iterations=1) & ~mask_pad
            n_rim = int(rim.sum())
            if n_rim == 0:
                continue

            bz0 = int(row["z0_lvl2"]) - z0_load
            bz1 = bz0 + int(row["dz"])
            by0 = int(row["y0_lvl2"]) - sub_y0
            by1 = by0 + int(row["dy"])
            bx0 = int(row["x0_lvl2"]) - sub_x0
            bx1 = bx0 + int(row["dx"])
            if (bz0 < 0 or by0 < 0 or bx0 < 0
                    or bz1 > seg_block.shape[0]
                    or by1 > seg_block.shape[1]
                    or bx1 > seg_block.shape[2]):
                continue
            seg_pad = seg_block[bz0:bz1, by0:by1, bx0:bx1]

            rim_labels = seg_pad[rim]
            fg = rim_labels[(rim_labels != 0) & (rim_labels != hid)]
            if fg.size == 0:
                continue
            uniq, counts = np.unique(fg, return_counts=True)
            for nb, ct in zip(uniq, counts):
                rows.append({
                    "hcr_id": int(hid),
                    "neighbor_id": int(nb),
                    "overlap_voxels": int(ct),
                    "rim_voxels": int(n_rim),
                })
            n_done += 1

        if (s_idx + 1) % max(1, len(strip_keys) // 5) == 0:
            print(
                f"  [{sid}] strip {s_idx+1}/{len(strip_keys)}  "
                f"cells_with_neighbors={n_done}  edges={len(rows)}  "
                f"elapsed={time.time()-t0:.0f}s",
                flush=True,
            )

    nb_df = pd.DataFrame(rows)
    nb_df.to_parquet(out_path, index=False)

    elapsed = time.time() - t0
    meta = {
        "subject_id": sid,
        "version": "neighbors_v1",
        "n_cells_with_neighbors": int(n_done),
        "n_edges": int(len(nb_df)),
        "rim_radius_voxels": 1,
        "extraction_timestamp": datetime.utcnow().isoformat() + "Z",
        "elapsed_seconds": float(elapsed),
    }
    with open(_meta_path(sid), "w") as f:
        json.dump(meta, f, indent=2)
    sz = out_path.stat().st_size
    print(
        f"[{sid}] DONE: {n_done} cells with neighbors, {len(nb_df)} edges, "
        f"{elapsed:.0f}s, {sz/1e6:.1f} MB", flush=True,
    )
    return {
        "sid": sid,
        "n_cells_with_neighbors": int(n_done),
        "n_edges": int(len(nb_df)),
        "elapsed_s": float(elapsed),
        "path": str(out_path),
    }


def extract_neighbors_all(subjects: List[str], workers: int = 6, force: bool = False
                          ) -> List[Dict]:
    ctx = get_context("spawn")
    args = [(sid, force) for sid in subjects]
    with ctx.Pool(processes=min(workers, len(subjects))) as pool:
        results = pool.starmap(_extract_neighbors_subject, args)
    return results
