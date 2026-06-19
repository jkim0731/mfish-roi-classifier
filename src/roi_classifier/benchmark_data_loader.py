"""Data loader for benchmark coregistration subjects.

Handles the 6 benchmark subjects (755252, 767018, 767022, 782149, 788406, 790322)
and the format variations between them.

Coordinate conventions (following manual workflow/step_2):
- CZ centroids and landmarks are in CZ-pixel coordinates:
    XY: 0.78 um/pixel (all current subjects; 400 um FOV / 512 pixels)
    Z : 1.0 um/pixel
- HCR centroids and landmarks (HCR side) are in level-2 pyramid pixels:
    XY: (dimensions.x[0] in meters) * 4e6  um/pixel  (approx 0.988)
    Z : (dimensions.z[0] in meters) * 1e6  um/pixel  (approx 1.0)
  Column ordering in landmark CSVs is [cz_x, cz_y, cz_z, hcr_x, hcr_y, hcr_z].
"""
from __future__ import annotations

import glob
import json
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


# Default minimum spots per cell to define a GFP+ HCR cell. Used only when
# `gfp_threshold_method='counts_min'`. Overridable at module level.
DEFAULT_GFP_MIN_SPOTS = 5

# Default GFP+ threshold method for spot-data subjects. 'peakgauss3_density_p0.1'
# fits a 3-component GMM on log(density>0), picks the rightmost (signal) component,
# and returns its 0.1st-percentile lower tail as the density cutoff. Chosen in
# session 04 subgoal 01 v2.2 because:
#   (a) distribution-driven with no hand-picked percentile on the target quantity
#       (the 0.1 percentile is a lower-bound on the fitted signal Gaussian, not
#       a fixed fraction of the full histogram),
#   (b) tighter than yen_log_joint on every spot subject while keeping
#       coreg_coverage >= 98%, and
#   (c) the rightmost GMM component tracks the visual "signal peak" — the same
#       peak the coreg-cell distribution centres on.
# 'yen_log_joint', 'yen_log', 'counts_min', and 'fixed_frac' retained for
# diagnostics / legacy.
DEFAULT_GFP_THRESHOLD_METHOD = "peakgauss3_density_p0.1"
DEFAULT_GFP_TARGET_FRAC = 0.20  # retained for the legacy 'fixed_frac' method

# Default GFP+ threshold method for intensity-only subjects (755252, 767022).
# 'peakgauss3_mean_bg_p1' applies a 3-component GMM to log(mean - background)
# for cells with mean > background, picks the rightmost component (the coreg-
# cell signal peak), and thresholds at its 1st-percentile lower tail. Chosen in
# session 04 subgoal 02 because raw log(mean) is dominated by the autofluorescence
# bulk — subtracting the per-cell local background cleanly separates signal from
# the autofluorescence pedestal. 'none' keeps the legacy behaviour (no threshold,
# every HCR cell returned as GFP+).
DEFAULT_GFP_INTENSITY_METHOD = "peakgauss3_mean_bg_p1"


from . import config as _cfg  # noqa: E402 (below module-level constants)
DATA_DIR = _cfg.DATA_ROOT

# CZ side resolution (um/pixel). All current benchmark subjects use 400 um FOV / 512 px = 0.78125.
CZ_XY_UM = 0.78
CZ_Z_UM = 1.0

# Fallback HCR resolution when fused_ng.json is absent (e.g. 767018).
# Derived from the other subjects (0.2451-0.2474 raw um/pixel => * 4 ~= 0.988 um/level-2-px).
HCR_XY_UM_FALLBACK = 0.988
HCR_Z_UM_FALLBACK = 1.0

# HCR pyramid levels — xy downsampling factor between level-0 and the
# level-2 frame that hcr_centroids / hcr_gfp_df use.
# `cell_body_segmentation/segmentation_mask.zarr` is at LEVEL-0 (raw) in
# xy — its `metrics.pickle` global_bbox entries and voxel indices are in
# that 4×-finer grid. The sibling `segmentation_mask_orig_res.zarr` is
# the level-2 version (mis-named — "orig_res" is actually the downsampled
# one). Any code that reads from `segmentation_mask.zarr` must use
# `s.hcr_seg_xy_um` (= s.hcr_xy_um / HCR_SEG_XY_DOWNSAMPLE) rather than
# s.hcr_xy_um, or voxel extents / areas will be off by a factor of 4 / 16.
HCR_SEG_XY_DOWNSAMPLE = 4


@dataclass
class SubjectData:
    subject_id: str
    coreg_dir: Path | None   # None when only HCR is attached (classifier-only / no coreg)
    hcr_dir: Path

    # resolution in um/pixel for each modality in native pixel space
    cz_xy_um: float = CZ_XY_UM
    cz_z_um: float = CZ_Z_UM
    hcr_xy_um: float = HCR_XY_UM_FALLBACK  # level-2 (centroid frame)
    hcr_z_um: float = HCR_Z_UM_FALLBACK

    # HCR `cell_body_segmentation/segmentation_mask.zarr` is at level-0;
    # these fields expose its µm/voxel so callers don't have to remember
    # the 4× factor. Populated from hcr_xy_um / HCR_SEG_XY_DOWNSAMPLE.
    hcr_seg_xy_um: float = HCR_XY_UM_FALLBACK / HCR_SEG_XY_DOWNSAMPLE
    hcr_seg_z_um: float = HCR_Z_UM_FALLBACK

    # core dataframes
    cz_centroids: pd.DataFrame = field(default_factory=pd.DataFrame)   # [id, z, y, x] px
    hcr_centroids: pd.DataFrame = field(default_factory=pd.DataFrame)  # [id, z, y, x] px
    hcr_gfp_df: pd.DataFrame = field(default_factory=pd.DataFrame)     # GFP+ table w/ hcr_id
    landmarks_qced: pd.DataFrame = field(default_factory=pd.DataFrame) # final qc'd landmarks
    coreg_table: pd.DataFrame = field(default_factory=pd.DataFrame)    # [cz_id, hcr_id]
    landmark_iter_files: list[Path] = field(default_factory=list)

    gfp_feature_name: str = ""  # 'density' | 'mean' | 'mean_minus_bg' | 'counts'
    gfp_min_spots: int = DEFAULT_GFP_MIN_SPOTS  # *effective* threshold actually applied
    gfp_source: str = ""  # 'spot_counts_csv' | 'aggregated_spots_csv' | 'intensity_r1' | ''
    gfp_threshold_method: str = DEFAULT_GFP_THRESHOLD_METHOD
    gfp_intensity_method: str = DEFAULT_GFP_INTENSITY_METHOD
    gfp_target_frac: float | None = None  # populated when method='fixed_frac'


# -----------------------------------------------------------
# path resolution
# -----------------------------------------------------------
def _find_coreg_dir(subject_id: str, required: bool = True) -> Path | None:
    matches = sorted(DATA_DIR.glob(f"{subject_id}*ctl-czstack-hcr-coreg_*"))
    if not matches:
        if required:
            raise FileNotFoundError(f"No coreg dir for {subject_id}")
        return None   # HCR-only mode (e.g. the ROI-quality classifier capsule)
    # Use the most recent one
    return matches[-1]


def _find_hcr_dir(subject_id: str) -> Path:
    matches = sorted(DATA_DIR.glob(f"HCR_{subject_id}_*_processed_*"))
    if not matches:
        raise FileNotFoundError(f"No HCR processed dir for {subject_id}")
    return matches[-1]


def _find_czstack_reg_dir(subject_id: str) -> Path | None:
    matches = sorted(DATA_DIR.glob(f"multiplane-ophys_{subject_id}_*cortical-zstack-registration*"))
    if not matches:
        return None
    return matches[-1]


# -----------------------------------------------------------
# resolution extraction
# -----------------------------------------------------------
def _read_hcr_resolution(hcr_dir: Path) -> tuple[float, float] | None:
    """Read HCR level-2 pixel resolution (um/pixel) for (xy, z).

    `hcr_centroids` (+ `hcr_gfp_df`) use level-2 pixel indices, so this
    is the scale downstream callers want. Multiplies the level-0 value
    by 4 in xy (`dims["x"][0] * 4e6`) and leaves z unchanged.

    Note: `cell_body_segmentation/segmentation_mask.zarr` is at LEVEL-0
    in xy — callers who read it must use `SubjectData.hcr_seg_xy_um`
    (= this value / HCR_SEG_XY_DOWNSAMPLE), not this value.
    """
    for name in ("fused_ng.json", "tile_subset_corrected_ng.json"):
        f = hcr_dir / name
        if f.exists():
            with open(f) as fp:
                data = json.load(fp)
            dims = data.get("dimensions")
            if dims and "x" in dims and "z" in dims:
                xy = dims["x"][0] * 4e6
                z = dims["z"][0] * 1e6
                return float(xy), float(z)
    return None


def _read_cz_resolution(subject_id: str) -> tuple[float, float]:
    """CZ pixel resolution (xy_um, z_um). All current subjects are 0.78 um/pixel XY, 1 um/pixel Z.

    Attempts to read from session.json; falls back to the nominal 0.78.
    """
    reg_dir = _find_czstack_reg_dir(subject_id)
    if reg_dir is not None:
        session_file = reg_dir / "session.json"
        if session_file.exists():
            try:
                with open(session_file) as f:
                    data = json.load(f)
                fovs = data["data_streams"][0].get("ophys_fovs", [])
                if fovs:
                    scale = float(fovs[0]["fov_scale_factor"])
                    return scale, 1.0
            except Exception:
                pass
    return CZ_XY_UM, CZ_Z_UM


# -----------------------------------------------------------
# centroid loaders
# -----------------------------------------------------------
def _load_cz_centroids(coreg_dir: Path) -> pd.DataFrame:
    files = list(coreg_dir.glob("*czstack_cell_centroids.csv"))
    if not files:
        raise FileNotFoundError(f"No CZ centroid CSV in {coreg_dir}")
    df = pd.read_csv(files[0])
    # normalize column names
    rename = {
        "czstack_cell_id": "cz_id",
        "czstack_z": "z_px",
        "czstack_y": "y_px",
        "czstack_x": "x_px",
    }
    df = df.rename(columns=rename)[["cz_id", "z_px", "y_px", "x_px"]]
    return df


def _load_hcr_centroids(hcr_dir: Path, coreg_dir: Path | None = None) -> pd.DataFrame:
    """Load HCR cell centroids, preferring the NPY from HCR processed dir.

    For 767018 (older pipeline), use the CSV in the coreg dir (only when a coreg dir
    is present). Returns columns [hcr_id, z_px, y_px, x_px].
    """
    npy = hcr_dir / "cell_body_segmentation" / "cell_centroids.npy"
    if npy.exists():
        arr = np.load(npy)  # (N, 4) with columns z, y, x, id
        df = pd.DataFrame(
            {
                "hcr_id": arr[:, 3].astype(int),
                "z_px": arr[:, 0].astype(float),
                "y_px": arr[:, 1].astype(float),
                "x_px": arr[:, 2].astype(float),
            }
        )
        return df

    csvs = list(coreg_dir.glob("*HCR_cell_centroids.csv")) if coreg_dir is not None else []
    if csvs:
        df = pd.read_csv(csvs[0])
        df = df.rename(
            columns={
                "hcr_cell_id": "hcr_id",
                "hcr_z": "z_px",
                "hcr_y": "y_px",
                "hcr_x": "x_px",
            }
        )[["hcr_id", "z_px", "y_px", "x_px"]]
        return df

    raise FileNotFoundError(f"No HCR centroid file for {hcr_dir}")


# -----------------------------------------------------------
# GFP+ loader
# -----------------------------------------------------------
def _aggregate_spots_from_hcr(hcr_dir: Path) -> pd.DataFrame | None:
    """Fallback used when a subject does not have a pre-aggregated
    `*_spot_488_counts.csv`. Mirrors `step_2_automatic_mapping_for_qc.ipynb`:
        1. Read `image_spot_detection/channel_488_spots/spots.csv`
        2. counts = spots['SEG_ID'].value_counts()
        3. Merge with `cell_body_segmentation/metrics.pickle` for volume
        4. density = counts / volume
    Returns a DataFrame with columns [hcr_id, counts, volume, density].
    """
    spots_path = hcr_dir / "image_spot_detection" / "channel_488_spots" / "spots.csv"
    metrics_path = hcr_dir / "cell_body_segmentation" / "metrics.pickle"
    if not spots_path.exists():
        return None
    spots = pd.read_csv(spots_path, usecols=["SEG_ID"])
    counts = spots["SEG_ID"].value_counts()
    df = pd.DataFrame({"hcr_id": counts.index.astype(int), "counts": counts.values})
    if metrics_path.exists():
        with open(metrics_path, "rb") as f:
            m = pickle.load(f)
        metrics_df = pd.DataFrame(m).transpose()
        metrics_df.index = metrics_df.index.astype(int)
        metrics_df.index.name = "hcr_id"
        df = df.merge(
            metrics_df[["volume"]],
            left_on="hcr_id",
            right_index=True,
            how="left",
        )
        df["density"] = df["counts"] / df["volume"]
    return df


def _fixed_frac_count_threshold(
    spot_counts: np.ndarray, n_hcr_total: int, target_frac: float
) -> int:
    """Smallest integer count c such that `|cells with counts>=c| / n_hcr_total <= target_frac`.

    `spot_counts` contains per-cell counts for spot-bearing cells only (cells with
    0 spots are absent). Cells not present contribute 0 and are never selected.
    """
    if target_frac >= 1.0:
        return 1
    target_n = int(np.floor(target_frac * n_hcr_total))
    # cells_at_or_above(t) = (spot_counts >= t).sum(); cells with 0 spots never qualify
    # Take the (n_hcr_total - target_n)-th percentile of the union of 0-counts + spot_counts
    # Equivalently: the (target_n)-th largest value in the per-HCR-cell count vector.
    if target_n <= 0:
        return int(spot_counts.max()) + 1
    if target_n >= len(spot_counts):
        return 1
    sorted_desc = np.sort(spot_counts)[::-1]
    thr = int(np.ceil(sorted_desc[target_n - 1]))
    return max(thr, 1)


def _yen_log_count_threshold(spot_counts: np.ndarray) -> int:
    """Yen's max-entropy threshold on log(counts>=1), returned as an integer count.

    Distribution-driven: no fixed fraction or percentile, and depends only on
    the per-subject count histogram. See subgoal 04/R1/01 for the decision to
    adopt this (highest threshold that preserves >=95% coreg_table coverage on
    all spot-data subjects).
    """
    from skimage.filters import threshold_yen

    counts = spot_counts[spot_counts >= 1]
    if len(counts) == 0:
        return DEFAULT_GFP_MIN_SPOTS
    t_log = float(threshold_yen(np.log(counts), nbins=256))
    return max(int(np.ceil(np.exp(t_log))), 1)


def _yen_log_density_threshold(density: np.ndarray) -> float:
    """Yen's max-entropy threshold on log(density>0), returned as a float cutoff.

    Companion to `_yen_log_count_threshold` used by the `yen_log_joint` method.
    Subjects without a density column (metrics.pickle missing) skip the density
    leg at the caller's discretion.
    """
    from skimage.filters import threshold_yen

    d = np.asarray(density, dtype=float)
    d = d[d > 0]
    if len(d) == 0:
        return 0.0
    t_log = float(threshold_yen(np.log(d), nbins=256))
    return float(np.exp(t_log))


def _peakgauss3_log_lower_pct(
    values: np.ndarray, percentile: float, n_components: int = 3
) -> float:
    """Lower-tail percentile of the rightmost Gaussian of a GMM on log(values>0).

    Fits a ``n_components``-component GMM on ``log(values>0)``, picks the
    component with the largest mean (the rightmost Gaussian, i.e. the signal
    peak), and returns ``exp(μ + z_p · σ)`` in the *linear* domain where
    ``z_p = Φ⁻¹(percentile/100)``. With ``percentile=0.1`` the cutoff keeps
    ~99.9 % of the signal component; with ``percentile=1.0`` ~99 %.

    Subgoal 01/02 v2.2 motivation: the visual signal peak is roughly
    log-normal, and the coreg-cell histogram centres on it. The rightmost GMM
    component tracks that peak while absorbing the lower-end shoulder into
    the remaining components, so its lower percentile is a clean distribution-
    driven cutoff.
    """
    from scipy.stats import norm
    from sklearn.mixture import GaussianMixture

    v = np.asarray(values, dtype=float)
    v = v[v > 0]
    if len(v) < 100:
        return float("inf")
    log_x = np.log(v).reshape(-1, 1)
    gmm = GaussianMixture(
        n_components=n_components, random_state=0, n_init=5
    ).fit(log_x)
    mus = gmm.means_.ravel()
    sigmas = np.sqrt(gmm.covariances_.ravel())
    k = int(np.argmax(mus))
    z = float(norm.ppf(percentile / 100.0))
    return float(np.exp(float(mus[k]) + z * float(sigmas[k])))


def _apply_count_threshold(
    df: pd.DataFrame,
    n_hcr_total: int,
    method: str,
    gfp_min_spots: int,
    gfp_target_frac: float,
) -> tuple[pd.DataFrame, int, float | None]:
    """Apply a count threshold to `df` (with a 'counts' column).

    Returns (filtered_df, effective_min_spots, effective_target_frac_or_None).
    """
    if method == "counts_min":
        eff = int(gfp_min_spots)
        return df[df["counts"] >= eff].reset_index(drop=True), eff, None
    if method == "yen_log":
        counts = df["counts"].values.astype(float)
        eff = _yen_log_count_threshold(counts)
        return df[df["counts"] >= eff].reset_index(drop=True), eff, None
    if method == "yen_log_joint":
        counts = df["counts"].values.astype(float)
        t_c = _yen_log_count_threshold(counts)
        if "density" in df.columns:
            density = df["density"].values.astype(float)
            t_d = _yen_log_density_threshold(density)
            mask = (df["counts"].values.astype(float) >= t_c) & (density >= t_d)
        else:
            mask = df["counts"].values.astype(float) >= t_c
        return df[mask].reset_index(drop=True), t_c, None
    if method == "peakgauss3_density_p0.1":
        # Density-only cutoff via rightmost GMM-3 component on log(density>0),
        # 0.1st-percentile lower tail. Falls back to counts-only Yen if no
        # density column is present (e.g. aggregated spots without metrics).
        counts = df["counts"].values.astype(float)
        if "density" in df.columns:
            density = df["density"].values.astype(float)
            t_d = _peakgauss3_log_lower_pct(density, percentile=0.1, n_components=3)
            mask = density >= t_d
            return df[mask].reset_index(drop=True), 1, None
        t_c = _yen_log_count_threshold(counts)
        return df[df["counts"] >= t_c].reset_index(drop=True), t_c, None
    if method == "fixed_frac":
        counts = df["counts"].values.astype(float)
        eff = _fixed_frac_count_threshold(counts, n_hcr_total, gfp_target_frac)
        return df[df["counts"] >= eff].reset_index(drop=True), eff, gfp_target_frac
    raise ValueError(f"unknown gfp_threshold_method={method!r}")


def _apply_intensity_threshold(
    df: pd.DataFrame, method: str
) -> tuple[pd.DataFrame, str]:
    """Apply an intensity-side GFP+ threshold to a `cell_data_mean_*_R1.csv` frame.

    Returns (filtered_df, feature_name). ``df`` must have `mean` and
    `background` columns; the returned frame adds `mean_minus_bg` and is
    filtered to GFP+ cells. ``method='none'`` returns the input unchanged.
    """
    if method == "none":
        feature = "mean" if "mean" in df.columns else df.columns[-1]
        return df.reset_index(drop=True), feature
    if method == "peakgauss3_mean_bg_p1":
        if "background" not in df.columns or "mean" not in df.columns:
            raise ValueError(
                "peakgauss3_mean_bg_p1 requires 'mean' and 'background' columns"
            )
        mean = df["mean"].values.astype(float)
        bg = df["background"].values.astype(float)
        mbg_all = mean - bg
        valid = mbg_all > 0
        t = _peakgauss3_log_lower_pct(mbg_all[valid], percentile=1.0, n_components=3)
        out = df.loc[valid].copy()
        out["mean_minus_bg"] = mbg_all[valid]
        out = out[out["mean_minus_bg"] >= t].reset_index(drop=True)
        return out, "mean_minus_bg"
    raise ValueError(f"unknown gfp_intensity_method={method!r}")


def _load_gfp(
    subject_id: str,
    coreg_dir: Path,
    hcr_dir: Path,
    n_hcr_total: int,
    gfp_min_spots: int,
    gfp_threshold_method: str,
    gfp_target_frac: float,
    gfp_intensity_method: str,
) -> tuple[pd.DataFrame, str, str, int, float | None]:
    """Return (df, feature_name, source_tag, effective_min_spots, effective_target_frac).

    Priority:
        1. `{coreg_dir}/*spot_488_counts.csv` (pre-aggregated, provided for
           782149/788406/790322).  Apply the chosen count threshold.
        2. `{hcr_dir}/image_spot_detection/channel_488_spots/spots.csv`
           aggregated per cell (fallback for 767018 and any subject that was
           not pre-aggregated).  Apply the chosen count threshold.
        3. `data/cell_data_mean_{subj}_R1.csv` channel 488 mean intensity
           (755252, 767022). Apply the chosen ``gfp_intensity_method``.
    """
    # (1) pre-aggregated per-cell counts
    spot_files = list(coreg_dir.glob("*spot_488_counts.csv"))
    if spot_files:
        df = pd.read_csv(spot_files[0])
        df, eff, eff_frac = _apply_count_threshold(
            df, n_hcr_total, gfp_threshold_method, gfp_min_spots, gfp_target_frac
        )
        return df, "density", "spot_counts_csv", eff, eff_frac

    # (2) aggregate raw spots on the fly (767018-style)
    agg = _aggregate_spots_from_hcr(hcr_dir)
    if agg is not None and len(agg):
        agg, eff, eff_frac = _apply_count_threshold(
            agg, n_hcr_total, gfp_threshold_method, gfp_min_spots, gfp_target_frac
        )
        return agg, "density", "aggregated_spots_csv", eff, eff_frac

    # (3) intensity fallback (755252, 767022).
    intensity_file = DATA_DIR / f"cell_data_mean_{subject_id}_R1.csv"
    if intensity_file.exists():
        df = pd.read_csv(intensity_file)
        df = df[df["channel"] == 488] if "channel" in df.columns else df
        rename_candidates = {"cell_id": "hcr_id", "id": "hcr_id"}
        for k, v in rename_candidates.items():
            if k in df.columns:
                df = df.rename(columns={k: v})
        df, feature = _apply_intensity_threshold(df, gfp_intensity_method)
        return df, feature, "intensity_r1", gfp_min_spots, None

    return pd.DataFrame(), "", "", gfp_min_spots, None


# -----------------------------------------------------------
# landmarks
# -----------------------------------------------------------
LANDMARK_COLS = ["ids", "active", "cz_x", "cz_y", "cz_z", "hcr_x", "hcr_y", "hcr_z"]


def _read_landmark_file(path: Path) -> pd.DataFrame:
    """Read a landmark CSV (no header; values may be quoted)."""
    df = pd.read_csv(path, header=None)
    if df.shape[1] != len(LANDMARK_COLS):
        raise ValueError(
            f"{path} has {df.shape[1]} cols, expected {len(LANDMARK_COLS)}"
        )
    df.columns = LANDMARK_COLS
    # Normalize active to boolean
    if df["active"].dtype != bool:
        df["active"] = df["active"].apply(
            lambda v: str(v).strip().lower() in ("true", "1", "t")
        )
    # Numeric columns
    for c in LANDMARK_COLS[2:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _find_final_qced_landmarks(coreg_dir: Path) -> tuple[Path | None, list[Path]]:
    """Return (final_qced_landmark_path, all_iter_paths_sorted)."""
    pattern = "*landmarks_matched_ext_iter*.csv"
    all_files = sorted(coreg_dir.glob(pattern))

    def iter_num(p: Path) -> int:
        name = p.name
        # extract iter number
        try:
            frag = name.split("iter")[1]
            return int(frag.split("_")[0].split(".")[0])
        except Exception:
            return -1

    def is_qced(p: Path) -> bool:
        return "qced" in p.name and not any(
            tag in p.name for tag in ("backup", "temp", "test")
        )

    qced = [p for p in all_files if is_qced(p)]
    qced.sort(key=iter_num)
    final = qced[-1] if qced else None

    # keep only per-iteration main files (not temp/backup/test)
    iter_files = [
        p for p in all_files
        if not any(tag in p.name for tag in ("backup", "temp", "test"))
    ]
    iter_files.sort(key=iter_num)
    return final, iter_files


def _load_coreg_table(coreg_dir: Path) -> pd.DataFrame:
    files = list(coreg_dir.glob("*coreg_table.csv"))
    # Prefer the plainest filename (e.g. '{subj}_coreg_table.csv')
    files = sorted(files, key=lambda p: len(p.name))
    if not files:
        return pd.DataFrame(columns=["cz_id", "hcr_id"])
    df = pd.read_csv(files[0])
    df = df.rename(columns={"czstack_id": "cz_id"})
    return df[["cz_id", "hcr_id"]]


# -----------------------------------------------------------
# main loader
# -----------------------------------------------------------
def load_subject(
    subject_id: str,
    gfp_min_spots: int = DEFAULT_GFP_MIN_SPOTS,
    gfp_threshold_method: str = DEFAULT_GFP_THRESHOLD_METHOD,
    gfp_target_frac: float = DEFAULT_GFP_TARGET_FRAC,
    gfp_intensity_method: str = DEFAULT_GFP_INTENSITY_METHOD,
) -> SubjectData:
    """Load all data for one benchmark subject.

    Parameters
    ----------
    subject_id : str or int
        One of BENCHMARK_SUBJECTS.
    gfp_min_spots : int
        Minimum 488-spot count per HCR cell when
        ``gfp_threshold_method='counts_min'``. Ignored otherwise.
        Default ``DEFAULT_GFP_MIN_SPOTS`` = 5 (legacy behaviour).
    gfp_threshold_method : {'peakgauss3_density_p0.1', 'yen_log_joint', 'yen_log', 'counts_min', 'fixed_frac'}
        Thresholding strategy for spot-data subjects.

        - ``'peakgauss3_density_p0.1'`` (default, subgoal 01 v2.2): fit a
          3-comp GMM on ``log(density>0)``, pick the rightmost (signal)
          component, threshold at its 0.1st-percentile lower tail
          (``exp(μ + z_{0.001} · σ)``). Falls back to counts-only Yen if a
          subject has no density column.
        - ``'yen_log_joint'`` (subgoal 01 v2.1): Yen on ``log(counts>=1)``
          AND Yen on ``log(density>0)``; cell qualifies if it passes both.
          Retained as a diagnostic / rollback option.
        - ``'yen_log'``: Yen applied only to ``log(counts>=1)``. Diagnostic.
        - ``'counts_min'``: legacy fixed cutoff at ``gfp_min_spots``.
        - ``'fixed_frac'``: pick the smallest ``c`` such that
          ``(counts>=c) / n_hcr_total ≲ gfp_target_frac``. Retained only for
          diagnostics — not generalisable across subjects.

        For intensity-only subjects (755252, 767022) this argument is
        replaced by ``gfp_intensity_method``.
    gfp_target_frac : float
        Target GFP+ fraction of ``n_hcr_total`` when using ``'fixed_frac'``.
        Default 0.20.
    gfp_intensity_method : {'peakgauss3_mean_bg_p1', 'none'}
        Thresholding strategy for intensity-only subjects (755252, 767022).

        - ``'peakgauss3_mean_bg_p1'`` (default, subgoal 02): fit a 3-comp
          GMM on ``log(mean - background)`` for cells with ``mean > bg``,
          pick the rightmost (signal) component, threshold at its 1st-
          percentile lower tail. Cells with ``mean ≤ bg`` are dropped.
        - ``'none'``: legacy behaviour — return every HCR cell as GFP+.

    The returned ``SubjectData`` records both the derived ``gfp_min_spots``
    (the count threshold actually applied) and the method / target fraction
    that produced it so downstream code can be audited.
    """
    subject_id = str(subject_id)
    # Coreg is OPTIONAL: the ROI-quality classifier path uses only HCR (hcr_dir +
    # hcr_centroids). When no coreg dir is attached we load HCR-only and leave the
    # coreg-derived fields (cz_centroids, gfp, landmarks, coreg_table) empty.
    coreg_dir = _find_coreg_dir(subject_id, required=False)
    hcr_dir = _find_hcr_dir(subject_id)

    cz_xy, cz_z = _read_cz_resolution(subject_id)
    hcr_res = _read_hcr_resolution(hcr_dir)
    if hcr_res is None:
        hcr_xy, hcr_z = HCR_XY_UM_FALLBACK, HCR_Z_UM_FALLBACK
    else:
        hcr_xy, hcr_z = hcr_res

    hcr = _load_hcr_centroids(hcr_dir, coreg_dir)
    if coreg_dir is not None:
        cz = _load_cz_centroids(coreg_dir)
        gfp_df, feat, source, effective_min, effective_frac = _load_gfp(
            subject_id, coreg_dir, hcr_dir,
            n_hcr_total=len(hcr),
            gfp_min_spots=gfp_min_spots,
            gfp_threshold_method=gfp_threshold_method,
            gfp_target_frac=gfp_target_frac,
            gfp_intensity_method=gfp_intensity_method,
        )
        final_lm, all_lm = _find_final_qced_landmarks(coreg_dir)
        lm_df = _read_landmark_file(final_lm) if final_lm else pd.DataFrame(columns=LANDMARK_COLS)
        coreg = _load_coreg_table(coreg_dir)
    else:
        print(f"[load_subject] {subject_id}: no coreg dir found — HCR-only load "
              f"(classifier feature path needs only HCR; cz/gfp/landmarks/coreg_table left empty).",
              flush=True)
        cz = pd.DataFrame()
        gfp_df, feat, source, effective_min, effective_frac = (
            pd.DataFrame(), "", "", gfp_min_spots, None)
        final_lm, all_lm = None, []
        lm_df = pd.DataFrame(columns=LANDMARK_COLS)
        coreg = pd.DataFrame()

    return SubjectData(
        subject_id=subject_id,
        coreg_dir=coreg_dir,
        hcr_dir=hcr_dir,
        cz_xy_um=cz_xy,
        cz_z_um=cz_z,
        hcr_xy_um=hcr_xy,
        hcr_z_um=hcr_z,
        hcr_seg_xy_um=hcr_xy / HCR_SEG_XY_DOWNSAMPLE,
        hcr_seg_z_um=hcr_z,
        cz_centroids=cz,
        hcr_centroids=hcr,
        hcr_gfp_df=gfp_df,
        landmarks_qced=lm_df,
        coreg_table=coreg,
        landmark_iter_files=all_lm,
        gfp_feature_name=feat,
        gfp_min_spots=effective_min,
        gfp_source=source,
        gfp_threshold_method=gfp_threshold_method if source != "intensity_r1" else gfp_intensity_method,
        gfp_intensity_method=gfp_intensity_method,
        gfp_target_frac=effective_frac,
    )


# -----------------------------------------------------------
# coordinate helpers
# -----------------------------------------------------------
def cz_px_to_um(arr_px: np.ndarray, s: SubjectData) -> np.ndarray:
    """arr_px columns = (z, y, x) in pixels."""
    a = np.asarray(arr_px, dtype=float).copy()
    a[:, 0] *= s.cz_z_um
    a[:, 1] *= s.cz_xy_um
    a[:, 2] *= s.cz_xy_um
    return a


def hcr_px_to_um(arr_px: np.ndarray, s: SubjectData) -> np.ndarray:
    a = np.asarray(arr_px, dtype=float).copy()
    a[:, 0] *= s.hcr_z_um
    a[:, 1] *= s.hcr_xy_um
    a[:, 2] *= s.hcr_xy_um
    return a


def landmark_pairs_um(s: SubjectData, active_only: bool = True):
    """Return (cz_um, hcr_um) in physical microns.

    Columns are (x, y, z) -> we return arrays shaped (N, 3) with the same order as stored: x, y, z.
    """
    lm = s.landmarks_qced
    if lm.empty:
        return np.empty((0, 3)), np.empty((0, 3))
    if active_only:
        lm = lm[lm["active"]]
    cz = lm[["cz_x", "cz_y", "cz_z"]].values.astype(float)
    hcr = lm[["hcr_x", "hcr_y", "hcr_z"]].values.astype(float)
    cz_um = cz.copy()
    cz_um[:, 0] *= s.cz_xy_um
    cz_um[:, 1] *= s.cz_xy_um
    cz_um[:, 2] *= s.cz_z_um
    hcr_um = hcr.copy()
    hcr_um[:, 0] *= s.hcr_xy_um
    hcr_um[:, 1] *= s.hcr_xy_um
    hcr_um[:, 2] *= s.hcr_z_um
    return cz_um, hcr_um


BENCHMARK_SUBJECTS = ["755252", "767018", "767022", "782149", "788406", "790322"]


def load_all(
    gfp_min_spots: int = DEFAULT_GFP_MIN_SPOTS,
    gfp_threshold_method: str = DEFAULT_GFP_THRESHOLD_METHOD,
    gfp_target_frac: float = DEFAULT_GFP_TARGET_FRAC,
    gfp_intensity_method: str = DEFAULT_GFP_INTENSITY_METHOD,
) -> dict[str, SubjectData]:
    out = {}
    for sid in BENCHMARK_SUBJECTS:
        out[sid] = load_subject(
            sid,
            gfp_min_spots=gfp_min_spots,
            gfp_threshold_method=gfp_threshold_method,
            gfp_target_frac=gfp_target_frac,
            gfp_intensity_method=gfp_intensity_method,
        )
    return out


if __name__ == "__main__":
    data = load_all()
    for sid, s in data.items():
        print(
            f"{sid}: cz={len(s.cz_centroids)}, hcr={len(s.hcr_centroids)}, "
            f"gfp={len(s.hcr_gfp_df)} ({s.gfp_feature_name} via {s.gfp_source}, "
            f"min_spots={s.gfp_min_spots}, method={s.gfp_threshold_method}, "
            f"frac={s.gfp_target_frac}), "
            f"lm_active={int(s.landmarks_qced['active'].sum()) if not s.landmarks_qced.empty else 0}, "
            f"coreg={len(s.coreg_table)}; res cz={s.cz_xy_um:.3f}, hcr_xy={s.hcr_xy_um:.3f}"
        )
