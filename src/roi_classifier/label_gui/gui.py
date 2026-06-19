"""
S11 — HCR ROI pass/fail labelling GUI (3-D orthoviewer).

Single-screen ipywidgets labeller built around a 3-orthogonal-view
display (xy + xz + yz) with a z-scrubber so reviewers can verify
3-D segmentation quality without leaving the keyboard.

    +-------------------------------------------------------------+
    | sid=788406  hcr_id=14330  score=0.51  17/100   reviewer=... |
    |  +-----------+ +-----------+ +-----------+   ┌────────────┐ |
    |  | xy z=cur  | | xz y=cy   | | yz x=cx   |   │ features   │ |
    |  | + mask    | | + mask    | | + mask    |   │ panel      │ |
    |  +-----------+ +-----------+ +-----------+   │            │ |
    |  channel: (405) 488 overlay  | mip ☐         │            │ |
    |  z [<<<<<<──○─────>] (j/k or ↑/↓ steps z=±1) │            │ |
    |  [Good g] [Bad b] [Unsure u] [Skip s] [Undo z] [Quit]      │ |
    |  last: good(14328) bad(14329) ...                            │ |
    +-------------------------------------------------------------+

Crosshairs on each view mark where the other two slice; mask outline
is drawn in red for the active label only.  The full crop volume is
read once into RAM per ROI so scrubbing is instant.

Labels are appended to
    {MFISH_ROI_QUALITY_DIR}/roi_qc_actions.jsonl   (config.ROI_QUALITY_DIR)

Keyboard:
    g  good      b  bad      u  unsure
    s  skip      z  undo
    j / ↓  z − 1
    k / ↑  z + 1
    m      toggle MIP / single-slice
    1/2/3  channel = 405 / 488 / overlay

Usage in a notebook::

    from gui import launch
    launch(sid="788406", n_rois=80, score_band=(0.3, 0.7), reviewer="alice")
"""
from __future__ import annotations

import datetime as _dt
import json
import os as _os
import subprocess as _subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import ipywidgets as W
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr
from IPython.display import display

try:
    from ipyevents import Event as _IpyEvent
    _HAS_IPYEVENTS = True
except ImportError:  # pragma: no cover — environments without ipyevents
    _IpyEvent = None  # type: ignore[assignment]
    _HAS_IPYEVENTS = False

from .. import config as _cfg

# Default paths — all overridable via env vars (see config.py).
# CACHE and TIGHT_BBOX point at the feature/tight-bbox parquet cache dirs.
# LABEL_LOG is the append-only labelling log; defaults to the bundled copy
# under labels/ but callers should pass an explicit path in production.
CACHE = _cfg.ROI_QUALITY_DIR
TIGHT_BBOX = _cfg.TIGHT_BBOX_DIR
LABEL_LOG = _cfg.ROI_QUALITY_DIR / "roi_qc_actions.jsonl"

# Labels are per-session, timestamped assets. LABEL_READ_SRC is where prior labels
# are READ from (a file, or a directory of attached *.jsonl label assets); LABEL_OUT
# is where THIS session's events are WRITTEN (its own timestamped file). The GUI
# launcher (app.main) sets these from --label-assets / --label-out; both default to
# the single LABEL_LOG for back-compat.
LABEL_READ_SRC: Any = None
LABEL_OUT: Path = LABEL_LOG

SUBJECTS = ["755252", "767018", "767022", "782149", "788406", "790322"]
HCR_DATA_ROOTS: dict[str, Path] = {}  # populated lazily by `_hcr_dir`

# Display
# Display features — all members of the consolidated _um (91-col) set.
KEY_DISPLAY_FEATURES = [
    "volume_um3_raw",
    "equivalent_diameter_um_opened",
    "solidity_opened",
    "bbox_occupancy_raw",
    "sphericity_opened",
    "frac_kept_opening",
    "c405_raw_p90",
    "c405_shell_minus_core_p90",
    "surface_touching_frac",
    "n_touching_neighbors",
    "knn_d1",
    "n_neighbors_30um",
    "axis3d_lambda_ratio_l1_l3",
    "protrusion_voxel_frac",
]


def _hcr_dir(sid: str) -> Path:
    if sid in HCR_DATA_ROOTS:
        return HCR_DATA_ROOTS[sid]
    matches = sorted(_cfg.DATA_ROOT.glob(f"HCR_{sid}_*"))
    if not matches:
        raise FileNotFoundError(f"No HCR dir for {sid}")
    HCR_DATA_ROOTS[sid] = matches[0]
    return matches[0]


def _open_zarr(path: Path):
    return zarr.open(str(path), mode="r")


# ── label provenance: segmentation data-asset identity + code version ─────────
def _datasets_json_path() -> Path | None:
    """Locate the capsule's CodeOcean attached-datasets manifest."""
    cands = []
    if _os.environ.get("MFISH_DATASETS_JSON"):
        cands.append(Path(_os.environ["MFISH_DATASETS_JSON"]))
    cands += [Path("/root/capsule/.codeocean/datasets.json"),
              Path.cwd() / ".codeocean" / "datasets.json"]
    for p in cands:
        if p.exists():
            return p
    return None


_ASSET_ID_MAP: dict[str, str] | None = None


def _asset_id_for(mount_name: str) -> str | None:
    """CodeOcean data-asset id for a mounted asset folder name (None if unknown)."""
    global _ASSET_ID_MAP
    if _ASSET_ID_MAP is None:
        _ASSET_ID_MAP = {}
        p = _datasets_json_path()
        if p is not None:
            try:
                for a in json.loads(p.read_text()).get("attached_datasets", []):
                    if a.get("mount") and a.get("id"):
                        _ASSET_ID_MAP[a["mount"]] = a["id"]
            except Exception:
                pass
    return _ASSET_ID_MAP.get(mount_name)


def _segmentation_asset(sid: str) -> dict:
    """{name, id} of the HCR_*_processed asset that defines this subject's hcr_ids."""
    try:
        name = _hcr_dir(sid).name
    except Exception:
        return {"name": None, "id": None}
    return {"name": name, "id": _asset_id_for(name)}


_CODE_VERSION: str | None = None


def _code_version() -> str:
    """git short-hash of the installed mfish-roi-classifier (extractor + model version)."""
    global _CODE_VERSION
    if _CODE_VERSION is None:
        try:
            import roi_classifier as _rc
            repo = Path(_rc.__file__).resolve().parents[2]
            _CODE_VERSION = _subprocess.check_output(
                ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                text=True, stderr=_subprocess.DEVNULL).strip()
        except Exception:
            _CODE_VERSION = "unknown"
    return _CODE_VERSION


def _label_read_files() -> list[Path]:
    src = LABEL_READ_SRC if LABEL_READ_SRC is not None else LABEL_LOG
    p = Path(src)
    if p.is_dir():
        return sorted(p.glob("*.jsonl"))
    return [p] if p.exists() else []


def _load_label_log() -> pd.DataFrame:
    """Merge prior labels from LABEL_READ_SRC (a file, or a directory of
    per-session *.jsonl label assets) into one event frame."""
    rows = []
    for fp in _label_read_files():
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["ts", "sid", "hcr_id", "label"])


def _append_label(record: dict[str, Any]) -> None:
    LABEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(LABEL_OUT, "a") as f:
        f.write(json.dumps(record) + "\n")


@dataclass
class _ROI:
    sid: str
    hcr_id: int
    score: float                                     # binary positive prob
    bbox: tuple[int, int, int, int, int, int]  # zmin, zmax, ymin, ymax, xmin, xmax (level-2 vox)
    centroid: tuple[float, float, float]       # zc, yc, xc (level-2 vox)
    features: dict[str, Any]
    proba_4class: dict[str, float] = field(default_factory=dict)  # keys: bad, bad_ok, good, merged


def _load_quality_proba(sid: str) -> pd.DataFrame:
    """Per-ROI (_um) scores via the consolidated model + features.

    Cols: hcr_id, score (binary positive prob), p_bad, p_bad_ok, p_good, p_merged.
    """
    from .. import features as _features, model as _model
    feats = _features.extract_features(sid)
    binary_score, proba4 = _model.predict(feats)
    out = proba4.rename(
        columns={"bad": "p_bad", "bad_ok": "p_bad_ok",
                 "good": "p_good", "merged": "p_merged"}
    ).copy()
    # binary_score and proba4 are both built from feats in the same row order.
    out["score"] = binary_score.to_numpy()
    return out[["hcr_id", "score", "p_bad", "p_bad_ok", "p_good", "p_merged"]]


def _load_quality_features(sid: str) -> pd.DataFrame:
    """Per-ROI _um feature matrix (the consolidated production set).

    Delegates to the single source of truth `roi_classifier.features`
    (shape + axis + surface + protrusion; µm columns; no v6_vox, no
    percentile-rank columns)."""
    from .. import features as _features
    return _features.extract_features(sid)


def _load_all_features(sid: str) -> pd.DataFrame:
    """All per-ROI features for display + sampling — the consolidated
    _um feature matrix.  (The legacy v1 base parquet and the v2..v6_vox
    + pct-rank stack are no longer used.)"""
    return _load_quality_features(sid)


# Columns we strip from a merged-parquet row when copying values into
# `_ROI.features`.  Anything else numeric/bool is kept so the GUI can
# display arbitrary feature subsets (e.g. live top-N from a model file).
_RESERVED_ROI_KEYS = frozenset(
    {
        "hcr_id", "sid", "y", "label", "human_label", "pseudo_label", "source",
        # bbox + centroid (carried as their own _ROI fields)
        "zmin_vox", "zmax_vox", "ymin_vox", "ymax_vox", "xmin_vox", "xmax_vox",
        "zc_vox", "yc_vox", "xc_vox", "volume_vox",
        # model probabilities (carried as `score` and `proba_4class`)
        "score", "p_bad", "p_bad_ok", "p_good", "p_merged",
    }
)


def _make_roi_from_row(sid: str, r: pd.Series) -> _ROI:
    """Build a `_ROI` from a merged-parquet row that carries feature,
    bbox, binary and 4-class columns.  All non-reserved numeric
    cells are copied into `features` so downstream code can display
    any subset (e.g. the live top-N from the model file)."""
    features: dict[str, Any] = {}
    for k in r.index:
        if k in _RESERVED_ROI_KEYS:
            continue
        v = r[k]
        if isinstance(v, (str, bytes)):
            continue
        features[k] = v
    return _ROI(
        sid=sid,
        hcr_id=int(r["hcr_id"]),
        score=float(r["score"]),
        bbox=(
            int(r["zmin_vox"]), int(r["zmax_vox"]),
            int(r["ymin_vox"]), int(r["ymax_vox"]),
            int(r["xmin_vox"]), int(r["xmax_vox"]),
        ),
        centroid=(float(r["zc_vox"]), float(r["yc_vox"]), float(r["xc_vox"])),
        features=features,
        proba_4class={
            "bad": float(r["p_bad"]),
            "bad_ok": float(r["p_bad_ok"]),
            "good": float(r["p_good"]),
            "merged": float(r["p_merged"]),
        },
    )


def _sample_uncertain(
    sid: str,
    n_rois: int,
    score_band: tuple[float, float],
    rng: np.random.Generator,
    skip_hcr_ids: set[int],
) -> list[_ROI]:
    feats = _load_all_features(sid)
    quality_proba = _load_quality_proba(sid)
    bbox = pd.read_parquet(TIGHT_BBOX / f"{sid}_hcr_cell_tight_bbox_v1.parquet")

    df = feats.merge(quality_proba, on="hcr_id").merge(bbox, on="hcr_id", how="inner")
    lo, hi = score_band
    df = df[(df["score"] >= lo) & (df["score"] <= hi)]
    df = df[~df["hcr_id"].isin(skip_hcr_ids)]
    if df.empty:
        return []

    n = min(n_rois, len(df))
    idx = rng.choice(len(df), size=n, replace=False)
    sub = df.iloc[idx]

    return [_make_roi_from_row(sid, r) for _, r in sub.iterrows()]


def _crop_with_margin(
    arr_zarr,
    bbox: tuple[int, int, int, int, int, int],
    margin_xy: int = 8,
    margin_z: int = 2,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Read a 3-D crop from a zarr array shaped (1,1,Z,Y,X) with a margin
    around `bbox`.  Returns (crop, (z_offset, y_offset, x_offset)) where
    the offset gives the global coords of the crop[0, 0, 0] voxel.
    """
    Z, Y, X = arr_zarr.shape[-3:]
    zmin, zmax, ymin, ymax, xmin, xmax = bbox
    z0 = max(0, zmin - margin_z)
    z1 = min(Z, zmax + margin_z)
    y0 = max(0, ymin - margin_xy)
    y1 = min(Y, ymax + margin_xy)
    x0 = max(0, xmin - margin_xy)
    x1 = min(X, xmax + margin_xy)
    crop = np.asarray(arr_zarr[0, 0, z0:z1, y0:y1, x0:x1])
    return crop, (z0, y0, x0)


def _normalize_volume(vol: np.ndarray, p_lo=1.0, p_hi=99.5) -> np.ndarray:
    """Robust per-volume min-max scaling to [0, 1] for display.  Same
    contrast across all slices of a ROI so scrubbing doesn't flicker."""
    flat = vol.ravel().astype(np.float32)
    if flat.size == 0:
        return vol.astype(np.float32)
    lo = float(np.percentile(flat, p_lo))
    hi = float(np.percentile(flat, p_hi))
    if hi <= lo:
        return np.zeros_like(vol, dtype=np.float32)
    return np.clip((vol.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)


@dataclass
class _RoiCrops:
    """Cached per-ROI volumetric crops, normalised once for fast scrubbing."""

    img_405: np.ndarray  # (Z, Y, X) float32 in [0,1]
    img_488: np.ndarray
    label_mask: np.ndarray  # (Z, Y, X) bool — only the active hcr_id
    crop_origin: tuple[int, int, int]  # (z0, y0, x0) of crop in global level-2 vox
    z_anchor: int  # local z index of the centroid (clamped into bbox z range)
    y_anchor: int
    x_anchor: int

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.img_405.shape


def _load_roi_crops(roi: _ROI) -> _RoiCrops:
    hd = _hcr_dir(roi.sid)
    seg = _open_zarr(hd / "cell_body_segmentation/segmentation_mask_orig_res.zarr")
    fz_405 = _open_zarr(hd / "image_tile_fusing/fused/channel_405.zarr")["2"]
    fz_488 = _open_zarr(hd / "image_tile_fusing/fused/channel_488.zarr")["2"]

    mask_crop, (z0, y0, x0) = _crop_with_margin(seg, roi.bbox)
    img_405, _ = _crop_with_margin(fz_405, roi.bbox)
    img_488, _ = _crop_with_margin(fz_488, roi.bbox)
    label_mask = (mask_crop == roi.hcr_id)

    img_405 = _normalize_volume(img_405)
    img_488 = _normalize_volume(img_488)

    zc = int(round(roi.centroid[0])) - z0
    yc = int(round(roi.centroid[1])) - y0
    xc = int(round(roi.centroid[2])) - x0
    Z, Y, X = label_mask.shape
    zc = int(np.clip(zc, 0, Z - 1))
    yc = int(np.clip(yc, 0, Y - 1))
    xc = int(np.clip(xc, 0, X - 1))

    return _RoiCrops(
        img_405=img_405,
        img_488=img_488,
        label_mask=label_mask,
        crop_origin=(z0, y0, x0),
        z_anchor=zc,
        y_anchor=yc,
        x_anchor=xc,
    )


def _planes_for_channel(
    crops: _RoiCrops,
    channel: str,
    mip: bool,
    z_idx: int,
    y_idx: int | None = None,
    x_idx: int | None = None,
):
    """Return (xy, xz, yz) planes for the requested channel and the
    matching mask outlines.  `channel` ∈ {"405", "488", "overlay"}.
    `mip` collapses xy/xz/yz to MIPs; otherwise we slice at
    (z_idx, y_idx or y_anchor, x_idx or x_anchor)."""
    if channel == "405":
        vol = crops.img_405
    elif channel == "488":
        vol = crops.img_488
    elif channel == "overlay":
        vol = np.maximum(crops.img_405, crops.img_488 * 0.7)
    else:
        raise ValueError(f"unknown channel {channel}")

    yc = crops.y_anchor if y_idx is None else int(y_idx)
    xc = crops.x_anchor if x_idx is None else int(x_idx)

    if mip:
        xy_img = vol.max(axis=0)              # (Y, X)
        xy_mask = crops.label_mask.any(axis=0)
        xz_img = vol.max(axis=1)              # (Z, X)
        xz_mask = crops.label_mask.any(axis=1)
        yz_img = vol.max(axis=2)              # (Z, Y)
        yz_mask = crops.label_mask.any(axis=2)
    else:
        xy_img = vol[z_idx]                   # (Y, X)
        xy_mask = crops.label_mask[z_idx]
        xz_img = vol[:, yc, :]                # (Z, X)
        xz_mask = crops.label_mask[:, yc, :]
        yz_img = vol[:, :, xc]                # (Z, Y)
        yz_mask = crops.label_mask[:, :, xc]

    return (xy_img, xy_mask), (xz_img, xz_mask), (yz_img, yz_mask)


def _format_features(features: dict[str, Any]) -> str:
    lines = []
    for k in KEY_DISPLAY_FEATURES:
        if k not in features:
            continue
        v = features[k]
        if v is None or (isinstance(v, float) and np.isnan(v)):
            txt = "—"
        elif isinstance(v, (bool, np.bool_)):
            txt = "✓" if v else "✗"
        elif isinstance(v, (int, np.integer)):
            txt = f"{v}"
        else:
            txt = f"{float(v):.3g}"
        lines.append(f"<b>{k}</b>: {txt}")
    return "<br>".join(lines)


class LabelSession:
    """3-D orthoview labelling session that walks uncertain ROIs.

    Per ROI: one volumetric crop (xy/xz/yz) is loaded and held in RAM;
    the z-slider, channel toggle, and MIP checkbox redraw the three
    panels in place via `set_data` for instant scrubbing.
    """

    def __init__(
        self,
        sid: str,
        n_rois: int = 80,
        score_band: tuple[float, float] = (0.3, 0.7),
        reviewer: str = "anonymous",
        seed: int = 20260429,
    ) -> None:
        self.sid = sid
        self.reviewer = reviewer
        self.token = uuid.uuid4().hex[:8]

        prior = _load_label_log()
        skip = (
            set(prior.loc[prior["sid"] == sid, "hcr_id"].astype(int))
            if not prior.empty
            else set()
        )

        rng = np.random.default_rng(seed)
        self.rois: list[_ROI] = _sample_uncertain(sid, n_rois, score_band, rng, skip)
        self.history: list[dict[str, Any]] = []  # in-memory undo stack
        self.idx = 0
        self._crops: _RoiCrops | None = None
        self._suppress_observers = False

        # ---- widgets ----
        self.header = W.HTML()
        self.fig_out = W.Output()
        self.feat_html = W.HTML()
        self.history_html = W.HTML()

        self.channel_toggle = W.ToggleButtons(
            options=[("405", "405"), ("488", "488"), ("overlay", "overlay")],
            value="405",
            description="ch:",
            style={"button_width": "70px"},
        )
        self.mip_check = W.Checkbox(value=False, description="MIP (m)")
        self.z_slider = W.IntSlider(
            min=0, max=0, value=0, description="z",
            continuous_update=True, layout=W.Layout(width="600px"),
        )

        self.channel_toggle.observe(self._on_channel, names="value")
        self.mip_check.observe(self._on_mip, names="value")
        self.z_slider.observe(self._on_slider, names="value")

        self.btn_good = W.Button(description="Good (g)", button_style="success", layout=W.Layout(width="120px"))
        self.btn_bad = W.Button(description="Bad (b)", button_style="danger", layout=W.Layout(width="120px"))
        self.btn_unsure = W.Button(description="Unsure (u)", button_style="warning", layout=W.Layout(width="120px"))
        self.btn_skip = W.Button(description="Skip (s)", layout=W.Layout(width="120px"))
        self.btn_undo = W.Button(description="Undo (z)", layout=W.Layout(width="120px"))
        self.btn_quit = W.Button(description="Quit", layout=W.Layout(width="80px"))

        self.btn_good.on_click(lambda _b: self._record("good"))
        self.btn_bad.on_click(lambda _b: self._record("bad"))
        self.btn_unsure.on_click(lambda _b: self._record("unsure"))
        self.btn_skip.on_click(lambda _b: self._advance(record=False))
        self.btn_undo.on_click(lambda _b: self._undo())
        self.btn_quit.on_click(lambda _b: self._quit())

        # Keyboard (optional — falls back to button-only if ipyevents absent)
        if _HAS_IPYEVENTS:
            self._key_event = _IpyEvent(
                source=self.fig_out, watched_events=["keydown"], wait=80
            )
            self._key_event.on_dom_event(self._on_key)
        else:
            self._key_event = None

        # Persistent figure (one per session) — three orthogonal axes
        with self.fig_out:
            self.fig, axes = plt.subplots(1, 3, figsize=(10, 4))
            self.ax_xy, self.ax_xz, self.ax_yz = axes
            for ax, ttl in [(self.ax_xy, "xy (axial)"),
                            (self.ax_xz, "xz (coronal)"),
                            (self.ax_yz, "yz (sagittal)")]:
                ax.set_title(ttl, fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
            self._im_xy = self.ax_xy.imshow(np.zeros((1, 1)), cmap="gray", vmin=0, vmax=1)
            self._im_xz = self.ax_xz.imshow(np.zeros((1, 1)), cmap="gray", vmin=0, vmax=1, aspect="auto")
            self._im_yz = self.ax_yz.imshow(np.zeros((1, 1)), cmap="gray", vmin=0, vmax=1, aspect="auto")
            self._contour_artists: list = []
            self._crosshair_artists: list = []
            self.fig.tight_layout()
            plt.show()

        kb_status = (
            "<span style='color:#888;font-size:11px'>keyboard shortcuts active</span>"
            if _HAS_IPYEVENTS
            else "<span style='color:#cf222e;font-size:11px'>"
                 "ipyevents not installed — keyboard shortcuts disabled "
                 "(use buttons; <code>pip install ipyevents</code> + restart kernel to enable)"
                 "</span>"
        )
        kb_html = W.HTML(kb_status)
        view_controls = W.HBox([self.channel_toggle, self.mip_check, kb_html])
        slider_row = W.HBox([self.z_slider])
        action_buttons = W.HBox(
            [self.btn_good, self.btn_bad, self.btn_unsure, self.btn_skip, self.btn_undo, self.btn_quit]
        )
        body = W.HBox(
            [
                W.VBox(
                    [self.header, self.fig_out, view_controls, slider_row,
                     action_buttons, self.history_html],
                    layout=W.Layout(width="900px"),
                ),
                W.VBox(
                    [W.HTML("<h4>Features</h4>"), self.feat_html],
                    layout=W.Layout(width="320px", padding="0 0 0 12px"),
                ),
            ]
        )
        self.root = body

        # Render the first ROI
        self._render()

    # ---------- core actions ----------
    def _on_key(self, ev: dict[str, Any]) -> None:
        k = ev.get("key", "").lower()
        if k == "g":
            self._record("good")
        elif k == "b":
            self._record("bad")
        elif k == "u":
            self._record("unsure")
        elif k == "s":
            self._advance(record=False)
        elif k == "z":
            self._undo()
        elif k in ("j", "arrowdown"):
            self.z_slider.value = max(self.z_slider.min, self.z_slider.value - 1)
        elif k in ("k", "arrowup"):
            self.z_slider.value = min(self.z_slider.max, self.z_slider.value + 1)
        elif k == "m":
            self.mip_check.value = not self.mip_check.value
        elif k in ("1", "2", "3"):
            self.channel_toggle.value = ["405", "488", "overlay"][int(k) - 1]

    def _on_slider(self, _change) -> None:
        if self._suppress_observers:
            return
        self._redraw_orthoview()

    def _on_channel(self, _change) -> None:
        if self._suppress_observers:
            return
        self._redraw_orthoview()

    def _on_mip(self, _change) -> None:
        if self._suppress_observers:
            return
        # Slider is irrelevant in MIP mode, but keep its value alive
        self._redraw_orthoview()

    def _current(self) -> _ROI | None:
        if self.idx >= len(self.rois):
            return None
        return self.rois[self.idx]

    def _record(self, label: str) -> None:
        roi = self._current()
        if roi is None:
            return
        rec = {
            "ts": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "sid": roi.sid,
            "hcr_id": roi.hcr_id,
            "label": label,
            "stage1_score": roi.score,
            "reviewer": self.reviewer,
            "session_token": self.token,
        }
        _append_label(rec)
        self.history.append(rec)
        self._advance(record=True)

    def _advance(self, record: bool) -> None:
        self.idx += 1
        self._render()

    def _undo(self) -> None:
        if not self.history or self.idx == 0:
            return
        last = self.history.pop()
        tomb = {
            "ts": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "sid": last["sid"],
            "hcr_id": last["hcr_id"],
            "label": "_undone_",
            "undoes": last,
            "reviewer": self.reviewer,
            "session_token": self.token,
        }
        _append_label(tomb)
        self.idx = max(0, self.idx - 1)
        self._render()

    def _quit(self) -> None:
        self.header.value = (
            f"<b>Session ended.</b> {len(self.history)} labels written to "
            f"{LABEL_LOG}."
        )

    # ---------- rendering ----------
    def _clear_overlay(self) -> None:
        for art in self._contour_artists:
            try:
                art.remove()
            except Exception:
                pass
        self._contour_artists = []
        for art in self._crosshair_artists:
            try:
                art.remove()
            except Exception:
                pass
        self._crosshair_artists = []

    def _draw_overlay(self, xy_mask, xz_mask, yz_mask) -> None:
        for ax, m in [(self.ax_xy, xy_mask),
                      (self.ax_xz, xz_mask),
                      (self.ax_yz, yz_mask)]:
            if m.any():
                cs = ax.contour(m.astype(np.uint8), levels=[0.5],
                                colors="red", linewidths=1.0)
                self._contour_artists.append(cs)
        if self.mip_check.value or self._crops is None:
            return
        z_idx = self.z_slider.value
        yc = self._crops.y_anchor
        xc = self._crops.x_anchor
        # xy: rows=y, cols=x
        self._crosshair_artists.append(
            self.ax_xy.axvline(xc, color="cyan", lw=0.5, alpha=0.7))
        self._crosshair_artists.append(
            self.ax_xy.axhline(yc, color="cyan", lw=0.5, alpha=0.7))
        # xz: rows=z, cols=x
        self._crosshair_artists.append(
            self.ax_xz.axvline(xc, color="cyan", lw=0.5, alpha=0.7))
        self._crosshair_artists.append(
            self.ax_xz.axhline(z_idx, color="cyan", lw=0.5, alpha=0.7))
        # yz: rows=z, cols=y
        self._crosshair_artists.append(
            self.ax_yz.axvline(yc, color="cyan", lw=0.5, alpha=0.7))
        self._crosshair_artists.append(
            self.ax_yz.axhline(z_idx, color="cyan", lw=0.5, alpha=0.7))

    def _redraw_orthoview(self) -> None:
        if self._crops is None:
            return
        (xy_img, xy_mask), (xz_img, xz_mask), (yz_img, yz_mask) = _planes_for_channel(
            self._crops, self.channel_toggle.value, self.mip_check.value, self.z_slider.value
        )
        self._im_xy.set_data(xy_img)
        self._im_xz.set_data(xz_img)
        self._im_yz.set_data(yz_img)
        self._clear_overlay()
        self._draw_overlay(xy_mask, xz_mask, yz_mask)
        self.fig.canvas.draw_idle()

    def _reset_axes_for_new_crop(self) -> None:
        """Resize the three AxesImages (and axes limits) when the volume
        shape changes between ROIs."""
        Z, Y, X = self._crops.shape
        self._im_xy.set_extent((-0.5, X - 0.5, Y - 0.5, -0.5))
        self.ax_xy.set_xlim(-0.5, X - 0.5)
        self.ax_xy.set_ylim(Y - 0.5, -0.5)
        self._im_xz.set_extent((-0.5, X - 0.5, Z - 0.5, -0.5))
        self.ax_xz.set_xlim(-0.5, X - 0.5)
        self.ax_xz.set_ylim(Z - 0.5, -0.5)
        self._im_yz.set_extent((-0.5, Y - 0.5, Z - 0.5, -0.5))
        self.ax_yz.set_xlim(-0.5, Y - 0.5)
        self.ax_yz.set_ylim(Z - 0.5, -0.5)

    def _render(self) -> None:
        roi = self._current()
        if roi is None:
            self.header.value = (
                f"<b>Done.</b> {len(self.history)} labels in this session "
                f"({sum(1 for h in self.history if h['label']=='good')} good / "
                f"{sum(1 for h in self.history if h['label']=='bad')} bad / "
                f"{sum(1 for h in self.history if h['label']=='unsure')} unsure)."
            )
            self.feat_html.value = ""
            with self.fig_out:
                self._clear_overlay()
                self._im_xy.set_data(np.zeros((1, 1)))
                self._im_xz.set_data(np.zeros((1, 1)))
                self._im_yz.set_data(np.zeros((1, 1)))
                self.fig.canvas.draw_idle()
            return

        try:
            self._crops = _load_roi_crops(roi)
        except Exception as exc:  # pragma: no cover — surface in UI
            self.header.value = (
                f"<b>{roi.sid}</b> hcr_id={roi.hcr_id} — load error: {exc}"
            )
            return

        Z = self._crops.shape[0]
        self._suppress_observers = True
        try:
            self.z_slider.max = max(0, Z - 1)
            self.z_slider.min = 0
            self.z_slider.value = self._crops.z_anchor
        finally:
            self._suppress_observers = False

        self.header.value = (
            f"<b>{roi.sid}</b> &nbsp; hcr_id=<b>{roi.hcr_id}</b> &nbsp; "
            f"stage1 score=<b>{roi.score:.3f}</b> &nbsp; "
            f"shape (Z×Y×X)=<code>{Z}×{self._crops.shape[1]}×{self._crops.shape[2]}</code> &nbsp; "
            f"<span style='color:#888'>{self.idx + 1} / {len(self.rois)}</span>"
        )
        self.feat_html.value = _format_features(roi.features)

        with self.fig_out:
            self._reset_axes_for_new_crop()
            self._redraw_orthoview()

        items = []
        for h in self.history[-5:]:
            color = {"good": "#1a7f37", "bad": "#cf222e", "unsure": "#9a6700"}.get(h["label"], "#666")
            items.append(
                f"<span style='color:{color}'>{h['label']}</span>(#{h['hcr_id']})"
            )
        self.history_html.value = (
            "<div style='font-size:12px;color:#444'>last: "
            + ", ".join(items)
            + "</div>"
        )

    # ---------- public API ----------
    def display(self) -> None:
        display(self.root)


def launch(
    sid: str,
    n_rois: int = 80,
    score_band: tuple[float, float] = (0.3, 0.7),
    reviewer: str = "anonymous",
    seed: int = 20260429,
) -> LabelSession:
    """Convenience: build and display a `LabelSession`."""
    sess = LabelSession(
        sid=sid,
        n_rois=n_rois,
        score_band=score_band,
        reviewer=reviewer,
        seed=seed,
    )
    sess.display()
    return sess


def label_progress() -> pd.DataFrame:
    """Tally labelled ROIs per subject (excluding undone)."""
    df = _load_label_log()
    if df.empty:
        return pd.DataFrame(columns=["sid", "good", "bad", "unsure", "total"])
    # Drop undone
    undone_keys = set()
    if "_undone_" in df.get("label", pd.Series()).unique():
        for _, r in df[df["label"] == "_undone_"].iterrows():
            ub = r.get("undoes") or {}
            undone_keys.add((ub.get("sid"), int(ub.get("hcr_id", -1))))
    if undone_keys:
        keep = [(s, int(h)) not in undone_keys for s, h in zip(df["sid"], df["hcr_id"])]
        df = df[keep]
    df = df[df["label"].isin(["good", "bad", "unsure"])]
    out = (
        df.groupby(["sid", "label"]).size().unstack(fill_value=0).reset_index()
    )
    for c in ("good", "bad", "unsure"):
        if c not in out.columns:
            out[c] = 0
    out["total"] = out["good"] + out["bad"] + out["unsure"]
    return out[["sid", "good", "bad", "unsure", "total"]]
