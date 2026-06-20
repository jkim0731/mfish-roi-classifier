"""
S11 / Step 4b — STANDALONE HCR ROI labelling GUI (no Jupyter).

A pure matplotlib-window application.  Run from a shell with a
display::

    python -m roi_classifier.label_gui.app \
        --sid 788406 --reviewer alice

Why this exists alongside `04_label_gui.ipynb`: the notebook flavour
needs `ipyevents` + `pyarrow` inside the live kernel, which is
brittle (kernel restarts, user-site vs conda paths, etc.).  This
standalone app uses only matplotlib's bundled widgets — no extra
runtime requirements beyond the same data-loading helpers.

Both flavours share `gui.py`'s helpers, so labels written here land
in the same JSONL and feed Step 5 with no conversion.

Layout (3-D orthoview)::

    +-----------+ +-----------+ +-----------+ ┌─────────┐
    | xy axial  | | xz coronal| | yz sagit. | │ feature │
    +-----------+ +-----------+ +-----------+ │ panel   │
    | header (sid / hcr_id / score / pos z,y,x / i / N)   │
    | (405)(488)(overlay)  [☐ MIP]            │  right  │
    | [Good][Bad][Bad-OK][Merged][Unsure][Skip][Undo][Quit]  side) │
    | last: good(14328) bad(14329) ...        │         │
    +-----------------------------------------+ └─────────┘

Keyboard:
    g / b / o / e / u    label good / bad / bad_ok / merged / unsure (and advance)
    s                    skip          z   undo            q  quit
    j / down-arrow       z − 1
    k / up-arrow         z + 1
    mouse wheel          z ± 1 by default; over xz panel scrolls y;
                         over yz panel scrolls x
    m                    toggle MIP / single-slice
    1 / 2 / 3            channel = 405 / 488 / overlay
    n / p                next / previous subject (when --sid lists multiple)
"""
from __future__ import annotations

import argparse
import datetime as _dt
import sys
import uuid
from pathlib import Path

import matplotlib

# Pick a backend that supports interactive windows.  TkAgg is in the
# Python stdlib so it's the safest default; fall back to QtAgg if Tk
# isn't available, then to the matplotlib default.
for _bk in ("TkAgg", "QtAgg", "Qt5Agg"):
    try:
        matplotlib.use(_bk, force=True)
        break
    except Exception:  # pragma: no cover — backend probing
        continue

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, CheckButtons, RadioButtons, TextBox

from .gui import (
    CACHE,
    LABEL_LOG,
    TIGHT_BBOX,
    _ROI,
    _RoiCrops,
    _append_label,
    _code_version,
    _load_all_features,
    _load_label_log,
    _load_roi_crops,
    _load_quality_proba,
    _make_roi_from_row,
    _planes_for_channel,
    _sample_uncertain,
    _segmentation_asset,
)
import pandas as pd  # noqa: E402  (used by the local samplers)


# Saturated counterparts of the pastel button colors below — used to
# briefly flash the three image-panel borders when a label is recorded
# so the reviewer gets visible confirmation of the choice they just made.
_LABEL_BLINK_COLORS = {
    "good":   "#2ecc71",
    "bad":    "#e74c3c",
    "bad_ok": "#e67e22",
    "merged": "#9b59b6",
    "unsure": "#f1c40f",
}
_BLINK_SECONDS = 0.15


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------
_PROBA_ORDER = ("bad", "bad_ok", "good", "merged")

# Default production model whose feature-importance ranking drives
# the right-hand panel.  We watch its mtime; whenever it changes (e.g.
# the user reruns the trainer) the panel is rebuilt with
# the new top-N list.
from .. import config as _cfg  # noqa: E402
from ..model import FEATURE_COLUMNS  # noqa: E402  (embed these in each label record)
_QUALITY_4CLASS_MODEL = _cfg.MODELS_DIR / "roi_quality_4class.txt"
_TOP_FEATURE_N = 10


def _predicted_class(proba_4class: dict) -> tuple[str, float]:
    """Return (class_name, max_proba) from a 4-class proba dict.

    Empty dict → ("?", 0.0)."""
    if not proba_4class:
        return "?", 0.0
    items = [(c, float(proba_4class.get(c, 0.0))) for c in _PROBA_ORDER]
    cls, p = max(items, key=lambda kv: kv[1])
    return cls, p


class _TopFeaturesCache:
    """Cache top-N (name, gain) pulled from a LightGBM model file.

    Re-reads on demand whenever the file's mtime changes, so the GUI can
    surface fresh importances after a retrain without restart."""

    def __init__(self, model_path: Path, n: int = _TOP_FEATURE_N) -> None:
        self.path = model_path
        self.n = n
        self._mtime: float | None = None
        self._top: list[tuple[str, float]] = []
        self._error: str | None = None

    def get(self) -> tuple[list[tuple[str, float]], bool]:
        """Return (top_features, changed_since_last_get).

        `top_features` is the cached list (possibly empty if the model
        file is missing).  `changed_since_last_get` is True whenever the
        cache was just rebuilt on this call.
        """
        try:
            mtime = self.path.stat().st_mtime
        except FileNotFoundError:
            if self._top:
                self._top = []
                self._mtime = None
                self._error = f"model file missing: {self.path.name}"
                return self._top, True
            self._error = f"model file missing: {self.path.name}"
            return self._top, False
        if mtime == self._mtime:
            return self._top, False
        try:
            import lightgbm as lgb
            booster = lgb.Booster(model_file=str(self.path))
            names = booster.feature_name()
            gains = booster.feature_importance(importance_type="gain")
            top = sorted(
                zip(names, gains),
                key=lambda kv: -float(kv[1]),
            )[: self.n]
            self._top = [(n, float(g)) for n, g in top]
            self._mtime = mtime
            self._error = None
        except Exception as exc:  # pragma: no cover — surface in UI
            self._error = f"{type(exc).__name__}: {exc}"
        return self._top, True

    @property
    def error(self) -> str | None:
        return self._error


def _format_feature_value(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if isinstance(v, (bool, np.bool_)):
        return "True" if v else "False"
    if isinstance(v, (int, np.integer)):
        return f"{int(v)}"
    try:
        return f"{float(v):.3g}"
    except (TypeError, ValueError):
        return str(v)


def _now_iso() -> str:
    return _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _active_labels(label_log: pd.DataFrame, sid: str) -> dict[int, str]:
    """Return {hcr_id: label} for `sid` — the *latest* event per cell wins
    (a real label sets it, an `_undone_` clears it), across merged sessions/assets.
    """
    if label_log.empty:
        return {}
    sub = label_log[label_log["sid"].astype(str) == str(sid)].copy()
    if sub.empty:
        return {}
    valid = ("good", "bad", "bad_ok", "merged", "unsure")

    def _hid(r):
        if r.get("label") == "_undone_":
            ub = r.get("undoes") or {}
            try:
                return int(ub.get("hcr_id", -1))
            except (TypeError, ValueError):
                return -1
        try:
            return int(r["hcr_id"])
        except (TypeError, ValueError, KeyError):
            return -1

    sub["_hid"] = sub.apply(_hid, axis=1)
    sub = sub[(sub["_hid"] >= 0) & (sub["label"].isin((*valid, "_undone_")))]
    if sub.empty:
        return {}
    if "ts" in sub.columns:
        sub = sub.sort_values("ts", kind="stable")
    last = sub.groupby("_hid", as_index=False).last()
    last = last[last["label"].isin(valid)]
    return dict(zip(last["_hid"].astype(int), last["label"]))


def _sample_from_candidates(
    sid: str,
    hcr_ids: list[int],
    n_rois: int,
    rng: np.random.Generator,
    skip_hcr_ids: set[int],
) -> list[_ROI]:
    """Walk a fixed candidate list (e.g. top-K by margin or p_merged).

    Preserves the order in `hcr_ids` (highest-priority first), drops anything
    already in the label log, then takes the first `n_rois`. No random
    sampling — we want the model's most confident candidates first.
    """
    if not hcr_ids:
        return []
    feats = _load_all_features(sid)
    quality_proba = _load_quality_proba(sid)
    bbox = pd.read_parquet(TIGHT_BBOX / f"{sid}_hcr_cell_tight_bbox.parquet")
    df = feats.merge(quality_proba, on="hcr_id").merge(bbox, on="hcr_id", how="inner")

    keep = [h for h in hcr_ids if h not in skip_hcr_ids]
    df = df.set_index("hcr_id")
    keep = [h for h in keep if h in df.index]
    if not keep:
        return []
    sub = df.loc[keep[:n_rois]].reset_index()
    return [_make_roi_from_row(sid, r) for _, r in sub.iterrows()]


def _sample_labelled(
    sid: str,
    n_rois: int,
    rng: np.random.Generator,
    label_log: pd.DataFrame,
) -> tuple[list[_ROI], dict[int, str]]:
    """Sample previously-labelled ROIs for review (re-walk current labels)."""
    active = _active_labels(label_log, sid)
    if not active:
        return [], {}
    feats = _load_all_features(sid)
    quality_proba = _load_quality_proba(sid)
    bbox = pd.read_parquet(TIGHT_BBOX / f"{sid}_hcr_cell_tight_bbox.parquet")
    df = feats.merge(quality_proba, on="hcr_id").merge(bbox, on="hcr_id", how="inner")
    df = df[df["hcr_id"].astype(int).isin(active.keys())]
    if df.empty:
        return [], {}

    n = min(n_rois, len(df))
    idx = rng.choice(len(df), size=n, replace=False)
    sub = df.iloc[idx]
    rois = [_make_roi_from_row(sid, r) for _, r in sub.iterrows()]
    prior_labels = {roi.hcr_id: active[roi.hcr_id] for roi in rois}
    return rois, prior_labels


# ---------------------------------------------------------------------------
# main app
# ---------------------------------------------------------------------------
class StandaloneLabeller:
    """Single-window 3-D orthoview labelling app."""

    def __init__(
        self,
        sids: list[str],
        n_rois: int,
        score_band: tuple[float, float],
        reviewer: str,
        seed: int,
        candidates: dict[str, list[int]] | None = None,
    ) -> None:
        if not sids:
            raise ValueError("at least one sid is required")
        self.sids = list(sids)
        self.subject_idx = 0
        self.n_rois = n_rois
        self.score_band = score_band
        self.reviewer = reviewer
        self.seed = seed
        self.token = uuid.uuid4().hex[:8]
        self.candidates = candidates or {}

        # Per-subject ROI list and walk position; reset by _load_subject().
        self.rois: list[_ROI] = []
        self.idx = 0

        # Cross-subject history stack — undo always undoes the most recent
        # label regardless of which subject it was on.
        self.history: list[dict] = []
        self.crops: _RoiCrops | None = None
        self._suppress_subject_radio = False

        # Current slice indices (set per ROI from crops anchors; mutated by
        # mouse-wheel / keyboard nav).
        self._z_idx = 0
        self._y_idx = 0
        self._x_idx = 0

        # Channel / MIP state (preserved across subjects)
        self._channel = "405"
        self._mip = False

        # Top-N importance from the 4-class model — the right-hand
        # panel re-renders whenever this file's mtime changes.
        self._top_feats = _TopFeaturesCache(_QUALITY_4CLASS_MODEL, n=_TOP_FEATURE_N)

        # Sampling mode: "new" walks uncertain band (skipping already-labelled);
        # "review" re-walks ROIs that have a current (non-undone) label.
        self._mode = "new"
        self._prior_labels: dict[int, str] = {}

        # Load the initial subject's ROI pool; auto-advance past any
        # subjects with no work to do for the current mode, so the
        # user lands on the first one that has ROIs.
        self._load_subject()
        for _ in range(len(self.sids)):
            if self.rois:
                break
            self.subject_idx = (self.subject_idx + 1) % len(self.sids)
            self._load_subject()
        if not self.rois:
            raise RuntimeError(
                f"No ROIs to {self._mode}-label in any of the requested "
                f"subjects ({self.sids}) within score band {self.score_band}."
            )

        # ---- figure layout (axes positions in figure coords) ----
        # Left region (x < 0.74) holds the three image panels and all
        # controls.  Right region (x > 0.76) is reserved for the
        # features panel so its text can never overflow into buttons.
        title = (
            f"S11 labeller — sid={self.sids[0]}"
            if len(self.sids) == 1
            else f"S11 labeller — {len(self.sids)} subjects"
        )
        self.fig = plt.figure(f"{title}  reviewer={reviewer}", figsize=(15.5, 9.0))

        # Three image panels (top 53% of the figure, left region)
        self.ax_xy = self.fig.add_axes([0.030, 0.45, 0.225, 0.50])
        self.ax_xz = self.fig.add_axes([0.265, 0.45, 0.225, 0.50])
        self.ax_yz = self.fig.add_axes([0.500, 0.45, 0.225, 0.50])
        for ax, ttl in [
            (self.ax_xy, "xy (axial)"),
            (self.ax_xz, "xz (coronal)"),
            (self.ax_yz, "yz (sagittal)"),
        ]:
            ax.set_title(ttl, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

        self._im_xy = self.ax_xy.imshow(np.zeros((1, 1)), cmap="gray", vmin=0, vmax=1)
        self._im_xz = self.ax_xz.imshow(
            np.zeros((1, 1)), cmap="gray", vmin=0, vmax=1, aspect="auto"
        )
        self._im_yz = self.ax_yz.imshow(
            np.zeros((1, 1)), cmap="gray", vmin=0, vmax=1, aspect="auto"
        )
        self._contour_artists: list = []
        self._crosshair_artists: list = []

        # Header line (left region)
        self.ax_header = self.fig.add_axes([0.030, 0.395, 0.69, 0.04])
        self.ax_header.axis("off")
        self.text_header = self.ax_header.text(
            0, 0.5, "", va="center", ha="left", fontsize=11
        )

        # Subject "dropdown" (RadioButtons — matplotlib's closest equivalent).
        # Hidden when only one subject is in play, but always present so the
        # callback wiring is uniform.
        n_sub = max(2, len(self.sids))  # min height ratio still readable
        sub_h = max(0.08, min(0.16, 0.026 * n_sub))
        self.ax_subject = self.fig.add_axes(
            [0.030, 0.345 - sub_h, 0.105, sub_h]
        )
        self.ax_subject.set_title("subject (n/p)", fontsize=8.5, loc="left", pad=2)
        self.subject_radio = RadioButtons(
            self.ax_subject, tuple(self.sids), active=self.subject_idx
        )
        self.subject_radio.on_clicked(self._on_subject_radio)
        if len(self.sids) <= 1:
            self.ax_subject.set_visible(False)

        # Channel radio
        self.ax_radio = self.fig.add_axes([0.150, 0.235, 0.085, 0.10])
        self.ax_radio.set_title("channel (1/2/3)", fontsize=8.5, loc="left", pad=2)
        self.radio = RadioButtons(self.ax_radio, ("405", "488", "overlay"), active=0)
        self.radio.on_clicked(self._on_channel)

        # Mode radio (new vs review)
        self.ax_mode = self.fig.add_axes([0.245, 0.260, 0.085, 0.075])
        self.ax_mode.set_title("mode", fontsize=8.5, loc="left", pad=2)
        self.mode_radio = RadioButtons(self.ax_mode, ("new", "review"), active=0)
        self.mode_radio.on_clicked(self._on_mode)

        # MIP checkbox
        self.ax_check = self.fig.add_axes([0.340, 0.275, 0.075, 0.055])
        self.check = CheckButtons(self.ax_check, ("MIP (m)",), actives=[False])
        self.check.on_clicked(self._on_mip)

        # Reviewer TextBox — editable inside the GUI.  Press Enter to save.
        self.ax_reviewer = self.fig.add_axes([0.495, 0.285, 0.20, 0.040])
        self.tb_reviewer = TextBox(
            self.ax_reviewer, "reviewer ", initial=self.reviewer
        )
        self.tb_reviewer.on_submit(self._on_reviewer)

        # Action buttons — single row, no overlap with anything above
        btn_w, btn_h, btn_y = 0.082, 0.055, 0.10
        btn_xs = [0.030, 0.117, 0.204, 0.291, 0.378, 0.465, 0.552, 0.640]
        btn_labels = [
            "Good (g)", "Bad (b)", "Bad-OK (o)", "Merged (e)", "Unsure (u)",
            "Skip (s)", "Undo (z)", "Quit (q)",
        ]
        btn_colors = [
            "#a8e6a8", "#f5b5b5", "#f5d0a3", "#d5b5e5", "#ffe4a3",
            "#dcdcdc", "#dcdcdc", "#dcdcdc",
        ]
        callbacks = [
            lambda _e: self._record("good"),
            lambda _e: self._record("bad"),
            lambda _e: self._record("bad_ok"),
            lambda _e: self._record("merged"),
            lambda _e: self._record("unsure"),
            lambda _e: self._advance(),
            lambda _e: self._undo(),
            lambda _e: self._quit(),
        ]
        self.buttons = []
        for x, lbl, color, cb in zip(btn_xs, btn_labels, btn_colors, callbacks):
            ax = self.fig.add_axes([x, btn_y, btn_w, btn_h])
            b = Button(ax, lbl, color=color, hovercolor="#bcd")
            b.on_clicked(cb)
            self.buttons.append(b)

        # History strip (bottom of left region)
        self.ax_hist = self.fig.add_axes([0.030, 0.030, 0.69, 0.04])
        self.ax_hist.axis("off")
        self.text_hist = self.ax_hist.text(
            0, 0.5, "", va="center", ha="left", fontsize=9, color="#444"
        )

        # Features panel — right region, full vertical extent so text
        # never overlaps anything else regardless of how many lines we
        # render.  Lines clipped to axes box just in case.
        self.ax_feat = self.fig.add_axes([0.760, 0.030, 0.230, 0.93])
        self.ax_feat.axis("off")
        self.ax_feat.set_title(
            f"top-{_TOP_FEATURE_N} (4-class gain)",
            fontsize=10, loc="left",
        )
        self.text_feat = self.ax_feat.text(
            0.0, 0.985, "", va="top", ha="left",
            fontsize=9.5, family="monospace", clip_on=True,
            transform=self.ax_feat.transAxes,
        )

        # Keyboard + mouse wheel
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)

        # Render the first ROI
        self._render()

    # ---------------------------------------------------------------- events
    def _on_key(self, event) -> None:
        # Don't steal keys while the reviewer TextBox is being edited —
        # otherwise typing 'g' / 'b' / 'q' / etc. would label the current
        # ROI or quit the app instead of going into the name field.
        if getattr(self.tb_reviewer, "capturekeystrokes", False):
            return
        k = (event.key or "").lower()
        if k == "g":
            self._record("good")
        elif k == "b":
            self._record("bad")
        elif k == "o":
            self._record("bad_ok")
        elif k == "e":
            self._record("merged")
        elif k == "u":
            self._record("unsure")
        elif k == "s":
            self._advance()
        elif k == "z":
            self._undo()
        elif k == "q":
            self._quit()
        elif k in ("j", "down"):
            self._step_z(-1)
        elif k in ("k", "up"):
            self._step_z(+1)
        elif k == "m":
            # Programmatic toggle of CheckButton — call set_active to fire callback
            self.check.set_active(0)
        elif k in ("1", "2", "3"):
            new = ["405", "488", "overlay"][int(k) - 1]
            # set_active also fires on_clicked
            self.radio.set_active({"405": 0, "488": 1, "overlay": 2}[new])
        elif k == "n":
            self._switch_subject(+1)
        elif k == "p":
            self._switch_subject(-1)

    def _on_scroll(self, event) -> None:
        # event.step is +1 / -1 (or fractional on some backends);
        # event.button is 'up' / 'down' as a fallback.
        delta = 0
        if getattr(event, "step", None):
            delta = int(np.sign(event.step))
        elif getattr(event, "button", None) == "up":
            delta = +1
        elif getattr(event, "button", None) == "down":
            delta = -1
        if not delta:
            return
        if event.inaxes is self.ax_xz:
            self._step_y(delta)
        elif event.inaxes is self.ax_yz:
            self._step_x(delta)
        else:
            self._step_z(delta)

    def _on_channel(self, label) -> None:
        self._channel = label
        self._redraw_orthoview()

    def _on_mip(self, _label) -> None:
        # CheckButtons.get_status() -> [bool]
        self._mip = bool(self.check.get_status()[0])
        self._redraw_orthoview()

    def _on_subject_radio(self, label) -> None:
        if self._suppress_subject_radio:
            return
        try:
            target = self.sids.index(label)
        except ValueError:
            return
        self._select_subject(target)

    def _on_mode(self, label) -> None:
        if self._mode == label:
            return
        self._mode = label
        self._load_subject()
        self._render()

    def _on_reviewer(self, text) -> None:
        new = (text or "").strip() or "anonymous"
        if new == self.reviewer:
            return
        self.reviewer = new
        # Refresh just the header with the new name; no need to reload images.
        self._update_header_only()

    def _format_header(self) -> str:
        roi = self._current()
        if roi is None or self.crops is None:
            return ""
        Z, Y, X = self.crops.shape
        prior = self._prior_label_text(roi)
        pred_cls, pred_p = _predicted_class(roi.proba_4class)
        return (
            f"sid={roi.sid}   hcr_id={roi.hcr_id}   "
            f"bin={roi.score:.3f}  pred={pred_cls}({pred_p:.2f})   "
            f"shape (Z×Y×X)={Z}×{Y}×{X}   "
            f"pos (z,y,x)=({self._z_idx},{self._y_idx},{self._x_idx})   "
            f"[{self.idx + 1} / {len(self.rois)}]"
            f"{self._subject_tag()}   reviewer={self.reviewer}{prior}"
        )

    def _update_header_only(self) -> None:
        if self._current() is None or self.crops is None:
            return
        self.text_header.set_text(self._format_header())
        self.fig.canvas.draw_idle()

    def _prior_label_text(self, roi) -> str:
        if self._mode != "review":
            return ""
        prev = self._prior_labels.get(roi.hcr_id)
        if prev is None:
            return ""
        color_label = {
            "good": "GOOD", "bad": "BAD", "bad_ok": "BAD-OK",
            "merged": "MERGED", "unsure": "UNSURE",
        }.get(prev, prev.upper())
        return f"   prior={color_label}"

    # ---------------------------------------------------------------- subject
    def _load_subject(self) -> None:
        """Resample the ROI pool for the current (subject, mode) combo.

        new mode:    walk uncertain-band ROIs not yet in the label log.
        review mode: walk ROIs that currently have a non-undone label.
        """
        sid = self.sids[self.subject_idx]
        log = _load_label_log()
        rng = np.random.default_rng(self.seed + self.subject_idx)
        if self._mode == "review":
            rois, prior = _sample_labelled(sid, self.n_rois, rng, log)
            self.rois = rois
            self._prior_labels = prior
        else:
            skip = (
                set(log.loc[log["sid"] == sid, "hcr_id"].astype(int))
                if not log.empty
                else set()
            )
            if sid in self.candidates:
                self.rois = _sample_from_candidates(
                    sid, self.candidates[sid], self.n_rois, rng, skip,
                )
            else:
                self.rois = _sample_uncertain(
                    sid, self.n_rois, self.score_band, rng, skip
                )
            self._prior_labels = {}
        self.idx = 0
        self.crops = None

    def _select_subject(self, target_idx: int) -> None:
        if target_idx == self.subject_idx or not (0 <= target_idx < len(self.sids)):
            return
        self.subject_idx = target_idx
        self._load_subject()
        self._suppress_subject_radio = True
        try:
            self.subject_radio.set_active(target_idx)
        finally:
            self._suppress_subject_radio = False
        self._render()

    def _switch_subject(self, delta: int) -> None:
        if len(self.sids) <= 1:
            return
        self._select_subject((self.subject_idx + delta) % len(self.sids))

    # ---------------------------------------------------------------- nav
    def _step_z(self, delta: int) -> None:
        if self.crops is None:
            return
        Z = self.crops.shape[0]
        new = int(np.clip(self._z_idx + delta, 0, Z - 1))
        if new == self._z_idx:
            return
        self._z_idx = new
        self._redraw_orthoview()

    def _step_y(self, delta: int) -> None:
        if self.crops is None:
            return
        Y = self.crops.shape[1]
        new = int(np.clip(self._y_idx + delta, 0, Y - 1))
        if new == self._y_idx:
            return
        self._y_idx = new
        self._redraw_orthoview()

    def _step_x(self, delta: int) -> None:
        if self.crops is None:
            return
        X = self.crops.shape[2]
        new = int(np.clip(self._x_idx + delta, 0, X - 1))
        if new == self._x_idx:
            return
        self._x_idx = new
        self._redraw_orthoview()

    def _current(self) -> _ROI | None:
        if self.idx >= len(self.rois):
            return None
        return self.rois[self.idx]

    def _record(self, label: str) -> None:
        roi = self._current()
        if roi is None:
            return
        pred_cls, pred_p = _predicted_class(roi.proba_4class)
        # Self-contained record: embed this cell's model feature values (frozen at
        # the code_commit below) so training needs only the labels asset.
        feats: dict[str, float | None] = {}
        for k in FEATURE_COLUMNS:
            v = roi.features.get(k)
            try:
                fv = float(v)
                feats[k] = None if fv != fv else fv          # fv!=fv ⇒ NaN
            except (TypeError, ValueError):
                feats[k] = None
        rec = {
            "ts": _now_iso(),
            "sid": roi.sid,
            "hcr_id": roi.hcr_id,
            "label": label,
            "segmentation_asset": _segmentation_asset(roi.sid),  # {name, id} — disambiguates hcr_id
            "code_commit": _code_version(),                      # extractor + model version
            "features": feats,                                   # frozen feature values (self-contained)
            "model_binary_score": roi.score,
            "model_proba": dict(roi.proba_4class),
            "model_pred_class": pred_cls,
            "model_pred_p": pred_p,
            "reviewer": self.reviewer,
            "session_token": self.token,
        }
        _append_label(rec)
        self.history.append(rec)
        self._blink_label_color(label)
        self._advance()

    def _blink_label_color(self, label: str) -> None:
        """Briefly tint the three image-panel borders the chosen label's
        color so the reviewer sees their click was registered before the
        next ROI loads."""
        color = _LABEL_BLINK_COLORS.get(label)
        if color is None:
            return
        axes = (self.ax_xy, self.ax_xz, self.ax_yz)
        try:
            for ax in axes:
                for sp in ax.spines.values():
                    sp.set_edgecolor(color)
                    sp.set_linewidth(4.0)
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            plt.pause(_BLINK_SECONDS)
        finally:
            for ax in axes:
                for sp in ax.spines.values():
                    sp.set_edgecolor("black")
                    sp.set_linewidth(1.0)
            self.fig.canvas.draw_idle()

    def _advance(self) -> None:
        self.idx += 1
        self._render()

    # ---------------------------------------------------------------- features panel
    def _refresh_feature_panel(self, roi: _ROI | None) -> None:
        """Repopulate the right-hand panel with the live top-N gain
        features from the 4-class model.  Called per ROI and again
        whenever the cache reports the model file changed."""
        top, _ = self._top_feats.get()
        n_used = len(top)
        title = (
            f"top-{n_used} (4-class gain)"
            if n_used > 0
            else "top features (no model)"
        )
        self.ax_feat.set_title(title, fontsize=10, loc="left")
        if roi is None:
            self.text_feat.set_text("")
            return
        lines: list[str] = ["model:"]
        pred_cls, pred_p = _predicted_class(roi.proba_4class)
        lines.append(f"  binary score (good|bad_ok)  {roi.score:.3f}")
        lines.append(f"  predicted class             {pred_cls} ({pred_p:.3f})")
        for c in _PROBA_ORDER:
            if c in roi.proba_4class:
                lines.append(f"  p_{c:<24s}  {roi.proba_4class[c]:.3f}")
        lines.append("")
        if not top:
            err = self._top_feats.error or "no 4-class model file"
            lines.append(f"top-{_TOP_FEATURE_N} features: ({err})")
        else:
            lines.append(f"top-{n_used} features (gain rank → value):")
            name_w = max(len(name) for name, _ in top)
            name_w = min(name_w, 32)
            for i, (name, gain) in enumerate(top, 1):
                v = roi.features.get(name)
                txt = _format_feature_value(v) if v is not None else "—"
                short = name if len(name) <= name_w else name[: name_w - 1] + "…"
                lines.append(f"  {i:>2}. {short:<{name_w}s}  {txt}  (g={gain:.0f})")
        self.text_feat.set_text("\n".join(lines))

    def _undo(self) -> None:
        if not self.history or self.idx == 0:
            return
        last = self.history.pop()
        tomb = {
            "ts": _now_iso(),
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
        self.text_header.set_text(
            f"Session ended — {len(self.history)} labels written to "
            f"{LABEL_LOG}.  You can close this window."
        )
        self.fig.canvas.draw_idle()
        plt.close(self.fig)

    # ---------------------------------------------------------------- render
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
        for ax, m in [
            (self.ax_xy, xy_mask),
            (self.ax_xz, xz_mask),
            (self.ax_yz, yz_mask),
        ]:
            if m.any():
                cs = ax.contour(
                    m.astype(np.uint8), levels=[0.5],
                    colors="red", linewidths=1.0,
                )
                self._contour_artists.append(cs)
        if self._mip or self.crops is None:
            return
        z_idx = self._z_idx
        yc = self._y_idx
        xc = self._x_idx
        self._crosshair_artists.append(
            self.ax_xy.axvline(xc, color="cyan", lw=0.5, alpha=0.7))
        self._crosshair_artists.append(
            self.ax_xy.axhline(yc, color="cyan", lw=0.5, alpha=0.7))
        self._crosshair_artists.append(
            self.ax_xz.axvline(xc, color="cyan", lw=0.5, alpha=0.7))
        self._crosshair_artists.append(
            self.ax_xz.axhline(z_idx, color="cyan", lw=0.5, alpha=0.7))
        self._crosshair_artists.append(
            self.ax_yz.axvline(yc, color="cyan", lw=0.5, alpha=0.7))
        self._crosshair_artists.append(
            self.ax_yz.axhline(z_idx, color="cyan", lw=0.5, alpha=0.7))

    def _redraw_orthoview(self) -> None:
        if self.crops is None:
            return
        try:
            (xy_img, xy_mask), (xz_img, xz_mask), (yz_img, yz_mask) = _planes_for_channel(
                self.crops,
                channel=self._channel,
                mip=self._mip,
                z_idx=self._z_idx,
                y_idx=self._y_idx,
                x_idx=self._x_idx,
            )
            self._im_xy.set_data(xy_img)
            self._im_xz.set_data(xz_img)
            self._im_yz.set_data(yz_img)
            self._clear_overlay()
            self._draw_overlay(xy_mask, xz_mask, yz_mask)
            self.text_header.set_text(self._format_header())
            # If the model file was retrained while the GUI is open,
            # rebuild the right-hand panel with the new top-N list.
            _, changed = self._top_feats.get()
            if changed:
                self._refresh_feature_panel(self._current())
            self.fig.canvas.draw_idle()
        except Exception as exc:
            import traceback as _tb
            print(
                f"\n[redraw_orthoview] error on idx={self.idx} "
                f"sid={self.sids[self.subject_idx]} "
                f"channel={self._channel} mip={self._mip} "
                f"pos (z,y,x)=({self._z_idx},{self._y_idx},{self._x_idx}) "
                f"shape={getattr(self.crops, 'shape', '?')}",
                file=sys.stderr,
            )
            _tb.print_exc()
            self.text_header.set_text(
                f"redraw error: {type(exc).__name__}: {exc}.  "
                f"Press s to skip, z to undo, or change channel."
            )
            self.fig.canvas.draw_idle()

    def _resize_axes_for_crop(self) -> None:
        Z, Y, X = self.crops.shape
        self._im_xy.set_extent((-0.5, X - 0.5, Y - 0.5, -0.5))
        self.ax_xy.set_xlim(-0.5, X - 0.5)
        self.ax_xy.set_ylim(Y - 0.5, -0.5)
        self._im_xz.set_extent((-0.5, X - 0.5, Z - 0.5, -0.5))
        self.ax_xz.set_xlim(-0.5, X - 0.5)
        self.ax_xz.set_ylim(Z - 0.5, -0.5)
        self._im_yz.set_extent((-0.5, Y - 0.5, Z - 0.5, -0.5))
        self.ax_yz.set_xlim(-0.5, Y - 0.5)
        self.ax_yz.set_ylim(Z - 0.5, -0.5)

    def _init_indices_for_crop(self) -> None:
        if self.crops is None:
            return
        Z, Y, X = self.crops.shape
        self._z_idx = int(np.clip(self.crops.z_anchor, 0, max(0, Z - 1)))
        self._y_idx = int(np.clip(self.crops.y_anchor, 0, max(0, Y - 1)))
        self._x_idx = int(np.clip(self.crops.x_anchor, 0, max(0, X - 1)))

    def _update_history_strip(self) -> None:
        items = []
        for h in self.history[-6:]:
            color = {
                "good": "#1a7f37", "bad": "#cf222e",
                "bad_ok": "#bc4c00", "merged": "#6f42c1", "unsure": "#9a6700",
            }.get(h["label"], "#666")
            items.append(f"{h['label']}(#{h['hcr_id']})")
            _ = color  # color in matplotlib text is per-segment, omit for brevity
        self.text_hist.set_text("last: " + ", ".join(items) if items else "")

    def _subject_tag(self) -> str:
        if len(self.sids) <= 1:
            return ""
        return f"  subj {self.subject_idx + 1}/{len(self.sids)} (n,p)"

    def _render(self) -> None:
        roi = self._current()
        if roi is None:
            n_g = sum(1 for h in self.history if h["label"] == "good")
            n_b = sum(1 for h in self.history if h["label"] == "bad")
            n_o = sum(1 for h in self.history if h["label"] == "bad_ok")
            n_m = sum(1 for h in self.history if h["label"] == "merged")
            n_u = sum(1 for h in self.history if h["label"] == "unsure")
            cur_sid = self.sids[self.subject_idx]
            tail = (
                "  press n for next subject" if len(self.sids) > 1 else ""
            )
            self.text_header.set_text(
                f"sid={cur_sid} DONE — session totals: {len(self.history)} labels "
                f"({n_g} good / {n_b} bad / {n_o} bad-OK / {n_m} merged / {n_u} unsure)."
                f"{self._subject_tag()}{tail}"
            )
            self.text_feat.set_text("")
            self._clear_overlay()
            self._im_xy.set_data(np.zeros((1, 1)))
            self._im_xz.set_data(np.zeros((1, 1)))
            self._im_yz.set_data(np.zeros((1, 1)))
            self._update_history_strip()
            self.fig.canvas.draw_idle()
            return

        try:
            self.crops = _load_roi_crops(roi)
            Z, Y, X = self.crops.shape
            if Z == 0 or Y == 0 or X == 0:
                raise ValueError(f"degenerate crop shape (Z×Y×X)={Z}×{Y}×{X}")
            self._init_indices_for_crop()
            self._refresh_feature_panel(roi)
            self._resize_axes_for_crop()
            self._redraw_orthoview()  # also sets the header
            self._update_history_strip()
        except Exception as exc:
            import traceback as _tb
            _tb.print_exc()
            self.text_header.set_text(
                f"sid={roi.sid}  hcr_id={roi.hcr_id} #{self.idx + 1}/"
                f"{len(self.rois)} — render error: {type(exc).__name__}: {exc}.  "
                f"Press s to skip, z to undo."
            )
            self.text_feat.set_text("")
            self._clear_overlay()
            self._im_xy.set_data(np.zeros((1, 1)))
            self._im_xz.set_data(np.zeros((1, 1)))
            self._im_yz.set_data(np.zeros((1, 1)))
            self.fig.canvas.draw_idle()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
_BENCHMARK_SUBJECTS = ["755252", "767018", "767022", "782149", "788406", "790322"]


def _parse_sids(spec: str) -> list[str]:
    """Accept a single sid, a comma-separated list, or the literal `all`."""
    spec = spec.strip()
    if spec.lower() == "all":
        return list(_BENCHMARK_SUBJECTS)
    out = [s.strip() for s in spec.split(",") if s.strip()]
    if not out:
        raise argparse.ArgumentTypeError(f"invalid --sid value: {spec!r}")
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Standalone HCR ROI labelling GUI (no Jupyter).",
    )
    p.add_argument(
        "--sid", default=None, type=_parse_sids,
        help="Subject ID, e.g. 788406.  Comma-separated list (e.g. "
             "788406,790322) or `all` cycles through subjects via n / p keys.  "
             f"Default: all 6 benchmark subjects ({', '.join(_BENCHMARK_SUBJECTS)}).",
    )
    p.add_argument("--n-rois", type=int, default=80, help="ROIs to sample per subject")
    p.add_argument("--score-min", type=float, default=0.3,
                   help="Lower bound of binary-score uncertain band")
    p.add_argument("--score-max", type=float, default=0.7,
                   help="Upper bound of binary-score uncertain band")
    p.add_argument("--reviewer", default="anonymous",
                   help="Reviewer name to record in every label")
    p.add_argument("--seed", type=int, default=20260429,
                   help="RNG seed for reproducible sampling")
    p.add_argument("--candidates", default=None, type=Path,
                   help="CSV with `sid,hcr_id` columns (e.g. "
                        "outputs/label_candidates.csv). "
                        "When set, ROIs are walked in CSV order (top-priority "
                        "first) instead of random uncertain-band sampling. "
                        "Already-labelled rows are still skipped.")
    p.add_argument("--label-assets", default=None, type=Path,
                   help="Prior labels to READ (a file, or a directory of attached "
                        "*.jsonl label assets, merged newest-wins) — for priors/skip "
                        "and the review pool. Default: the single label log.")
    p.add_argument("--label-out", default=None, type=Path,
                   help="Where THIS session's labels are WRITTEN. A directory → a "
                        "per-session file roi_qc_actions_<UTCstamp>.jsonl inside it; "
                        "a file → that file. Default: $MFISH_LABEL_OUT_DIR (per-session) "
                        "else the default label log.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sids: list[str] = args.sid if args.sid else list(_BENCHMARK_SUBJECTS)

    # Per-session, timestamped label assets: READ prior labels from --label-assets
    # (a dir of *.jsonl, merged newest-wins), WRITE this session to its own file.
    import os as _os
    from . import gui as _gui
    if args.label_assets is not None:
        _gui.LABEL_READ_SRC = args.label_assets
    out_arg = args.label_out or (Path(_os.environ["MFISH_LABEL_OUT_DIR"])
                                 if _os.environ.get("MFISH_LABEL_OUT_DIR") else None)
    if out_arg is not None:
        out_arg = Path(out_arg)
        if out_arg.is_dir() or out_arg.suffix == "":
            stamp = _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            out_arg.mkdir(parents=True, exist_ok=True)
            _gui.LABEL_OUT = out_arg / f"roi_qc_actions_{stamp}.jsonl"
        else:
            _gui.LABEL_OUT = out_arg
    label_out = _gui.LABEL_OUT
    candidates: dict[str, list[int]] | None = None
    if args.candidates is not None:
        df = pd.read_csv(args.candidates)
        df["sid"] = df["sid"].astype(str)
        candidates = {
            sid: g["hcr_id"].astype(int).tolist()
            for sid, g in df.groupby("sid", sort=False)
        }
        print(f"  candidates loaded from {args.candidates}: "
              + ", ".join(f"{s}={len(v)}" for s, v in candidates.items()))
    print(f"S11 standalone labeller  backend={matplotlib.get_backend()}")
    print(f"  sids={sids}  n_rois={args.n_rois}  "
          f"band=({args.score_min}, {args.score_max})  reviewer={args.reviewer}"
          + ("  [CANDIDATES MODE]" if candidates else ""))
    app = StandaloneLabeller(
        sids=sids,
        n_rois=args.n_rois,
        score_band=(args.score_min, args.score_max),
        reviewer=args.reviewer,
        seed=args.seed,
        candidates=candidates,
    )
    print(f"  loaded {len(app.rois)} ROIs for sid={sids[app.subject_idx]}  writing → {label_out}")
    print("  shortcuts: g/b/o/e/u label (good/bad/bad_ok/merged/unsure)  s skip  z undo  q quit  "
          "j/k arrows or mouse-wheel scroll z  m MIP  1/2/3 channel  "
          "n/p next/prev subject")
    print("  GUI controls: subject + mode dropdowns on left, reviewer TextBox "
          "(press Enter to save)")
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
