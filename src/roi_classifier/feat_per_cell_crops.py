"""Per-cell padded mask-crop builder for the ROI-quality feature extractors.

The surface (v4), protrusion (v5) and neighbour modules **read** a per-cell crops
parquet (`{sid}_per_cell_crops.parquet` in `MFISH_PER_CELL_CROPS_DIR`) — one
packbits-encoded padded mask crop per cell — so they skip the slow seg-zarr label
decode.  This module **builds** that parquet.

Ported 2026-06-17 from the S11 session script ``02_dump_per_cell_crops.py``
(`data/claude-data_ophys-mfish-autocoreg_260503/.../v3_S11_roi_quality/`), whose
builder was dropped when feature extraction was refactored into the ``feat_*``
modules (only the *readers* survived).  Depends on the tight-bbox cache
(`feat_tight_bbox`), which must exist first.

For every cell in the subject's centroid table, the padded tight crop of the
binary mask (``seg == hid`` plus ``PAD_VOX`` voxels each side) is packbits-encoded
and written.  All coordinates are level-2 voxels (the ``orig_res`` frame).

Public API
----------
``build_per_cell_crops(s, cache=True, force=False) -> Path``
``build_per_cell_crops_sid(sid, cache=True, force=False) -> Path``

Schema::

    hcr_id int64
    z0_lvl2, y0_lvl2, x0_lvl2 int32   # global lvl2-voxel coords of crop origin
    dz, dy, dx int32                  # padded crop shape
    pad int32                         # voxel padding each side (PAD_VOX)
    mask_packed binary                # np.packbits(flat_bool).tobytes()
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import zarr

warnings.filterwarnings("ignore", category=UserWarning, module="zarr")

from . import config as _cfg
from .benchmark_data_loader import SubjectData, load_subject
from .feat_tight_bbox import _orig_res_path, tight_bbox_cache_path

PER_CELL_CROPS_DIR = _cfg.PER_CELL_CROPS_DIR

# Must match the strip-pass shape the extractors stripe with (feat_shape.STRIP_Z).
STRIP_Z = 128
Z_PAD = 24       # z half-context for morphology (>= max half-z-extent + radius)
PAD_VOX = 4      # padding each side written into the cache (covers opening r<=3 + 1)


def _arrow_schema() -> pa.Schema:
    return pa.schema([
        ("hcr_id", pa.int64()),
        ("z0_lvl2", pa.int32()),
        ("y0_lvl2", pa.int32()),
        ("x0_lvl2", pa.int32()),
        ("dz", pa.int32()),
        ("dy", pa.int32()),
        ("dx", pa.int32()),
        ("pad", pa.int32()),
        ("mask_packed", pa.binary()),
    ])


def crops_cache_path(sid: str) -> Path:
    return PER_CELL_CROPS_DIR / f"{sid}_per_cell_crops.parquet"


def build_per_cell_crops(s: SubjectData, cache: bool = True, force: bool = False) -> Path:
    """Build (or reuse) the per-cell crops parquet for subject ``s``.

    Requires the tight-bbox cache (build it first with ``feat_tight_bbox``).
    Returns the parquet path.  With ``cache=False`` the crops are computed but
    not written (rarely useful — the readers expect the file on disk).
    """
    sid = s.subject_id
    out_path = crops_cache_path(sid)
    if not force and out_path.exists():
        print(f"  [{sid}] per-cell crops already present: {out_path}")
        return out_path

    tb_path = tight_bbox_cache_path(sid)
    if not tb_path.exists():
        raise FileNotFoundError(
            f"tight bbox cache missing: {tb_path}. Build it first with "
            f"`roi-classifier build-bbox {sid}` (or feat_tight_bbox.build_tight_bbox_sid)."
        )

    t0 = time.time()
    seg = zarr.open(str(_orig_res_path(s)), mode="r")
    _, _, Z_seg, Y_seg, X_seg = seg.shape

    if s.hcr_centroids is None or s.hcr_centroids.empty:
        raise ValueError(f"[{sid}] no hcr_centroids — cannot scope the crops scan")
    cent = s.hcr_centroids.set_index("hcr_id")
    all_hids = s.hcr_centroids["hcr_id"].astype(int).tolist()
    cent_z: Dict[int, int] = {
        hid: int(round(float(cent.loc[hid]["z_px"]))) for hid in all_hids if hid in cent.index
    }
    z_lo_global = max(0, int(s.hcr_centroids["z_px"].min()) - 2)
    z_hi_global = min(Z_seg, int(s.hcr_centroids["z_px"].max()) + 2)

    tb_df = pd.read_parquet(tb_path)
    tb_lookup = {
        int(r.hcr_id): (int(r.zmin_vox), int(r.zmax_vox),
                        int(r.ymin_vox), int(r.ymax_vox),
                        int(r.xmin_vox), int(r.xmax_vox))
        for r in tb_df.itertuples(index=False)
    }

    cells_per_strip: Dict[int, List[int]] = {}
    for hid in all_hids:
        cz = cent_z.get(hid)
        if cz is None or not (z_lo_global <= cz < z_hi_global):
            continue
        bucket = z_lo_global + ((cz - z_lo_global) // STRIP_Z) * STRIP_Z
        cells_per_strip.setdefault(bucket, []).append(int(hid))

    PER_CELL_CROPS_DIR.mkdir(parents=True, exist_ok=True)
    # Write to a temp path then rename, so an interrupted build never leaves a
    # half-written cache the readers would trust.
    tmp_path = out_path.with_suffix(".parquet.tmp")
    writer = pq.ParquetWriter(str(tmp_path), _arrow_schema(),
                              compression="snappy", use_dictionary=False)
    rows: List[Dict] = []
    BATCH = 5000
    n_owned = n_missing = 0

    def _flush():
        if rows:
            writer.write_table(pa.Table.from_pylist(rows, schema=_arrow_schema()))
            rows.clear()

    z_strips = sorted(cells_per_strip.keys())
    print(f"[{sid}] per-cell crops build | seg (Z,Y,X)=({Z_seg},{Y_seg},{X_seg}) | "
          f"{len(z_strips)} strips | PAD_VOX={PAD_VOX}")
    try:
        for s_idx, z0_inner in enumerate(z_strips):
            z1_inner = min(z0_inner + STRIP_Z, z_hi_global)
            z0_load = max(0, z0_inner - Z_PAD)
            z1_load = min(Z_seg, z1_inner + Z_PAD)

            owned = cells_per_strip[z0_inner]
            own_bbs = [tb_lookup.get(h) for h in owned if h in tb_lookup]
            if not own_bbs:
                continue
            y_min = min(b[2] for b in own_bbs); y_max = max(b[3] for b in own_bbs)
            x_min = min(b[4] for b in own_bbs); x_max = max(b[5] for b in own_bbs)
            sub_y0 = max(0, y_min - PAD_VOX); sub_y1 = min(Y_seg, y_max + PAD_VOX)
            sub_x0 = max(0, x_min - PAD_VOX); sub_x1 = min(X_seg, x_max + PAD_VOX)

            seg_block = np.asarray(seg[0, 0, z0_load:z1_load, sub_y0:sub_y1, sub_x0:sub_x1])

            for hid in owned:
                tb = tb_lookup.get(hid)
                if tb is None:
                    continue
                zmin, zmax_ex, ymin, ymax_ex, xmin, xmax_ex = tb
                n_owned += 1
                psz0_g = max(0, zmin - PAD_VOX); psz1_g = min(Z_seg, zmax_ex + PAD_VOX)
                psy0_g = max(0, ymin - PAD_VOX); psy1_g = min(Y_seg, ymax_ex + PAD_VOX)
                psx0_g = max(0, xmin - PAD_VOX); psx1_g = min(X_seg, xmax_ex + PAD_VOX)
                bz0 = psz0_g - z0_load; bz1 = psz1_g - z0_load
                by0 = psy0_g - sub_y0; by1 = psy1_g - sub_y0
                bx0 = psx0_g - sub_x0; bx1 = psx1_g - sub_x0
                if (bz0 < 0 or bz1 > seg_block.shape[0]
                        or by0 < 0 or by1 > seg_block.shape[1]
                        or bx0 < 0 or bx1 > seg_block.shape[2]):
                    n_missing += 1
                    continue
                mask_crop = (seg_block[bz0:bz1, by0:by1, bx0:bx1] == hid)
                if not mask_crop.any():
                    n_missing += 1
                    continue
                rows.append({
                    "hcr_id": int(hid),
                    "z0_lvl2": int(psz0_g), "y0_lvl2": int(psy0_g), "x0_lvl2": int(psx0_g),
                    "dz": int(mask_crop.shape[0]), "dy": int(mask_crop.shape[1]),
                    "dx": int(mask_crop.shape[2]), "pad": int(PAD_VOX),
                    "mask_packed": np.packbits(mask_crop.reshape(-1)).tobytes(),
                })
                if len(rows) >= BATCH:
                    _flush()
            print(f"  [{sid}] strip {s_idx+1}/{len(z_strips)} owned={n_owned} "
                  f"missing={n_missing} elapsed={time.time()-t0:.0f}s", flush=True)
        _flush()
    finally:
        writer.close()

    tmp_path.replace(out_path)
    sz = out_path.stat().st_size
    print(f"[{sid}] per-cell crops DONE: {n_owned} cells ({n_missing} missing) "
          f"in {time.time()-t0:.0f}s, {sz/1e6:.0f} MB → {out_path}")
    return out_path


def build_per_cell_crops_sid(sid: str, cache: bool = True, force: bool = False) -> Path:
    return build_per_cell_crops(load_subject(sid), cache=cache, force=force)
