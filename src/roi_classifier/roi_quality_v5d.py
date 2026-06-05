"""Production API for the v5d ROI-quality stage-2 classifier.

Single entry point for inference (`predict_subject`) and training (`train`).
Feature extraction merges the five cached parquets per subject (v2, v3_extra,
v4, v5, v6_vox) and computes within-subject percentile ranks for four size
columns.  The two LightGBM models (binary + 4-class) are loaded from
CACHE_DIR on first call to `predict`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from . import config as _cfg

# ──────────────────────────────────────────────────────────────────────────────
# constants — mirror 05h_train_stage2_v5d.py exactly
# ──────────────────────────────────────────────────────────────────────────────

# Feature parquets are read from ROI_QUALITY_DIR; models from MODELS_DIR.
# Both default to the same location but can be separated via env vars.
CACHE_DIR = _cfg.ROI_QUALITY_DIR

MODEL_BINARY = _cfg.MODELS_DIR / "roi_quality_stage2_binary_v5d.txt"
MODEL_FOUR_CLASS = _cfg.MODELS_DIR / "roi_quality_stage2_4class_v5d.txt"
META_JSON = _cfg.MODELS_DIR / "roi_quality_stage2_meta_v5d.json"

CLASS_NAMES = ["bad", "bad_ok", "good", "merged"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

BINARY_POS = {"good", "bad_ok"}
BINARY_NEG = {"bad", "merged"}

PCT_RANK_COLS = [
    "volume_vox_opened",
    "volume_vox_raw",
    "axis3d_lambda1_vox2",
    "surface_area_vox_opened",
]

# 92 columns from roi_quality_stage2_meta_v5d.json, in order.
FEATURE_COLUMNS: list[str] = [
    "volume_vox_raw",
    "aspect_zy",
    "aspect_zx",
    "aspect_yx",
    "solidity_raw",
    "bbox_occupancy_raw",
    "volume_vox_opened",
    "frac_kept_opening",
    "solidity_opened",
    "c405_raw_mean",
    "c405_raw_std",
    "c405_raw_p10",
    "c405_raw_p50",
    "c405_raw_p90",
    "c405_opened_mean",
    "c405_opened_std",
    "c405_opened_p10",
    "c405_opened_p50",
    "c405_opened_p90",
    "c405_core_p50_opened",
    "c405_shell_p50_opened",
    "c405_shell_minus_core_p50",
    "c405_shell_minus_core_p90",
    "c405_outside_p50",
    "c405_inside_minus_outside_p50",
    "c405_inside_minus_outside_p90",
    "n_touching_neighbors",
    "surface_touching_frac",
    "top_neighbor_overlap_frac",
    "mean_touching_score_stage1",
    "min_touching_score_stage1",
    "knn_d1",
    "axis3d_lambda_ratio_l1_l3",
    "axis3d_lambda_ratio_l1_l2",
    "axis3d_peak_prom_max_intens",
    "axis3d_min_over_max_inner_intens",
    "axis3d_section_min_over_med_area",
    "axis3d_section_min_over_max_area",
    "axis3d_peak_prom_max_area",
    "axis3d_raw_section_min_over_med_area",
    "axis3d_raw_peak_prom_max_intens",
    "proj_xy_main_peak_prom_max",
    "proj_xy_main_min_over_max_inner",
    "proj_xy_aspect_ratio",
    "proj_yz_main_peak_prom_max",
    "proj_yz_main_min_over_max_inner",
    "proj_yz_aspect_ratio",
    "proj_zx_main_peak_prom_max",
    "proj_zx_main_min_over_max_inner",
    "proj_zx_aspect_ratio",
    "sphericity_raw",
    "sphericity_opened",
    "n_protrusion_voxels",
    "protrusion_voxel_frac",
    "protrusion_rim_voxels",
    "protrusion_rim_other_frac",
    "protrusion_rim_bg_frac",
    "protrusion_top_neighbor_frac",
    "n_distinct_neighbor_ids_at_protrusion",
    "protrusion_into_neighbor_frac",
    "surface_area_vox_raw",
    "surface_area_vox_opened",
    "sa_to_vol_vox_inv_raw",
    "sa_to_vol_vox_inv_opened",
    "core4vox_voxel_frac_opened",
    "c405_core4vox_p50_opened",
    "c405_shell4vox_p50_opened",
    "c405_shell_minus_core4vox_p50",
    "c405_shell_minus_core4vox_p90",
    "c405_shell_over_core4vox_p50_ratio",
    "axis3d_extent_vox",
    "axis3d_lambda1_vox2",
    "axis3d_lambda2_vox2",
    "axis3d_lambda3_vox2",
    "axis3d_peak_sep_vox_intens",
    "axis3d_raw_extent_vox",
    "proj_xy_main_extent_vox",
    "proj_yz_main_extent_vox",
    "proj_zx_main_extent_vox",
    "proj_xy_orth_fwhm_vox",
    "proj_yz_orth_fwhm_vox",
    "proj_zx_orth_fwhm_vox",
    "bbox_z_extent_vox",
    "bbox_y_extent_vox",
    "bbox_x_extent_vox",
    "equivalent_diameter_vox_raw",
    "equivalent_diameter_vox_opened",
    "n_neighbors_30vox",
    "volume_vox_opened_pct_subj",
    "volume_vox_raw_pct_subj",
    "axis3d_lambda1_vox2_pct_subj",
    "surface_area_vox_opened_pct_subj",
]

assert len(FEATURE_COLUMNS) == 92, f"expected 92 feature columns, got {len(FEATURE_COLUMNS)}"

# LightGBM hyper-parameters (identical to 05h trainer)
LGB_BINARY = dict(
    objective="binary",
    learning_rate=0.05,
    num_leaves=31,
    min_data_in_leaf=20,
    feature_fraction=0.85,
    bagging_fraction=0.85,
    bagging_freq=5,
    lambda_l2=1.0,
    is_unbalance=True,
    metric=["binary_logloss", "auc"],
    verbosity=-1,
    seed=20260430,
)
LGB_MULTI = dict(
    objective="multiclass",
    num_class=len(CLASS_NAMES),
    learning_rate=0.05,
    num_leaves=31,
    min_data_in_leaf=15,
    feature_fraction=0.85,
    bagging_fraction=0.85,
    bagging_freq=5,
    lambda_l2=1.0,
    metric=["multi_logloss"],
    verbosity=-1,
    seed=20260430,
)
N_ESTIMATORS = 400
EARLY_STOP = 30

# Columns that are dropped from the model but may appear in raw parquets.
# Keeping these sets here so callers can reference them if needed.
_DROP_FROM_MODEL: frozenset[str] = frozenset({
    "hcr_id", "y", "label", "human_label", "sid",
    # µm-suffixed columns
    "axis3d_extent_um", "axis3d_lambda1_um2", "axis3d_lambda2_um2",
    "axis3d_lambda3_um2", "axis3d_peak_sep_um_intens", "axis3d_raw_extent_um",
    "bbox_x_extent_um", "bbox_y_extent_um", "bbox_z_extent_um",
    "c405_core4um_p50_opened", "c405_shell4um_p50_opened",
    "c405_shell_minus_core4um_p50", "c405_shell_minus_core4um_p90",
    "c405_shell_over_core4um_p50_ratio", "core4um_voxel_frac_opened",
    "equivalent_diameter_um_opened", "equivalent_diameter_um_raw",
    "n_neighbors_30um", "proj_xy_main_extent_um", "proj_xy_orth_fwhm_um",
    "proj_yz_main_extent_um", "proj_yz_orth_fwhm_um", "proj_zx_main_extent_um",
    "proj_zx_orth_fwhm_um", "sa_to_vol_um_inv_opened", "sa_to_vol_um_inv_raw",
    "surface_area_um2_opened", "surface_area_um2_raw",
    "volume_um3_opened", "volume_um3_raw", "volume_um3_raw_v4",
    # dead features (zero gain in both binary + 4-class)
    "boundary_touching", "n_components_after_opening",
    "tight_bbox_in_pickle_bbox", "volume_pickle_minus_zarr_l2_eq",
    "protrusion_touches_other",
    # low-gain v3 peak-counter columns
    "axis3d_n_peaks_inner_area", "axis3d_peak_prom_2nd_intens",
    "proj_yz_main_n_peaks_inner", "proj_zx_main_n_peaks_inner",
    "proj_xy_main_n_peaks_inner", "axis3d_raw_n_peaks_inner_intens",
    "proj_zx_orth_n_peaks_at_main", "proj_xy_orth_n_peaks_at_main",
    "proj_yz_orth_n_peaks_at_main", "axis3d_n_peaks_inner_intens",
})


# ──────────────────────────────────────────────────────────────────────────────
# feature extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_features(sid: str) -> pd.DataFrame:
    """Merge the five cached parquets for `sid` and return the v5d feature matrix.

    Reads from CACHE_DIR:
        {sid}_features_v2.parquet
        {sid}_features_v3_extra.parquet
        {sid}_features_v4.parquet
        {sid}_features_v5.parquet
        {sid}_features_v6_vox.parquet

    Returns a DataFrame with columns ["hcr_id"] + FEATURE_COLUMNS.
    Row count equals the number of ROIs for the subject.
    """
    paths = {
        "v2":    CACHE_DIR / f"{sid}_features_v2.parquet",
        "v3":    CACHE_DIR / f"{sid}_features_v3_extra.parquet",
        "v4":    CACHE_DIR / f"{sid}_features_v4.parquet",
        "v5":    CACHE_DIR / f"{sid}_features_v5.parquet",
        "v6":    CACHE_DIR / f"{sid}_features_v6_vox.parquet",
    }
    for name, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"[{sid}] {name} parquet missing: {p}")

    f   = pd.read_parquet(paths["v2"])
    g   = pd.read_parquet(paths["v3"])
    h   = pd.read_parquet(paths["v4"])
    k   = pd.read_parquet(paths["v5"])
    v   = pd.read_parquet(paths["v6"])

    n_ref = len(f)
    for name, df in [("v3", g), ("v4", h), ("v5", k), ("v6", v)]:
        if len(df) != n_ref:
            print(f"  [{sid}] WARNING: {name} has {len(df)} rows vs v2 {n_ref} rows — "
                  f"merge will be on hcr_id (left join)")

    out = f.merge(g, on="hcr_id", how="left", suffixes=("", "_v3"))
    out = out.merge(h, on="hcr_id", how="left", suffixes=("", "_v4"))
    out = out.merge(k, on="hcr_id", how="left", suffixes=("", "_v5d"))
    out = out.merge(v, on="hcr_id", how="left", suffixes=("", "_v6"))

    # within-subject percentile ranks (computed on ALL rows, not just labelled)
    for c in PCT_RANK_COLS:
        if c in out.columns:
            out[f"{c}_pct_subj"] = out[c].rank(pct=True, method="average")
        else:
            out[f"{c}_pct_subj"] = float("nan")

    # validate
    missing = [c for c in FEATURE_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"[{sid}] missing feature columns after merge: {missing}")

    out = out[["hcr_id"] + FEATURE_COLUMNS].copy()

    nan_counts = out[FEATURE_COLUMNS].isna().sum()
    cols_with_nan = nan_counts[nan_counts > 0]
    if len(cols_with_nan) > 0:
        print(f"  [{sid}] NaN counts in feature columns (expected for some intensity cols "
              f"on empty masks):")
        for col, cnt in cols_with_nan.items():
            print(f"    {col}: {cnt}")

    print(f"  [{sid}] extract_features: {len(out)} ROIs, {len(FEATURE_COLUMNS)} features")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# inference
# ──────────────────────────────────────────────────────────────────────────────

def predict(
    features_df: pd.DataFrame,
    model_dir: Path = _cfg.MODELS_DIR,
) -> tuple[pd.Series, pd.DataFrame]:
    """Run both production models on a pre-built feature matrix.

    Parameters
    ----------
    features_df : DataFrame containing at least FEATURE_COLUMNS (and hcr_id).
    model_dir   : directory containing the .txt model files.

    Returns
    -------
    binary_score : pd.Series indexed by hcr_id, float in [0, 1] (positive = good/bad_ok).
    four_class_proba : pd.DataFrame[hcr_id, bad, bad_ok, good, merged].
    """
    bin_path  = model_dir / "roi_quality_stage2_binary_v5d.txt"
    multi_path = model_dir / "roi_quality_stage2_4class_v5d.txt"
    for p in (bin_path, multi_path):
        if not p.exists():
            raise FileNotFoundError(f"model file missing: {p}")

    bst_bin   = lgb.Booster(model_file=str(bin_path))
    bst_multi = lgb.Booster(model_file=str(multi_path))

    X = features_df[FEATURE_COLUMNS].copy()
    for c in X.columns:
        if pd.api.types.is_bool_dtype(X[c]):
            X[c] = X[c].astype("float32")

    hcr_ids = features_df["hcr_id"].to_numpy("int64")

    scores = bst_bin.predict(X, num_iteration=bst_bin.best_iteration)
    binary_score = pd.Series(scores, index=hcr_ids, name="binary_score")
    binary_score.index.name = "hcr_id"

    proba = bst_multi.predict(X, num_iteration=bst_multi.best_iteration)
    four_class_proba = pd.DataFrame(proba, columns=CLASS_NAMES)
    four_class_proba.insert(0, "hcr_id", hcr_ids)

    return binary_score, four_class_proba


def predict_subject(sid: str) -> pd.DataFrame:
    """Convenience: extract features then predict; returns one row per ROI.

    Columns: hcr_id, binary_score, proba_bad, proba_bad_ok, proba_good,
             proba_merged, predicted_class.
    """
    feat = extract_features(sid)
    binary_score, four_class_proba = predict(feat)

    out = pd.DataFrame({"hcr_id": feat["hcr_id"].to_numpy("int64")})
    out["binary_score"] = binary_score.values
    for c in CLASS_NAMES:
        out[f"proba_{c}"] = four_class_proba[c].values
    out["predicted_class"] = four_class_proba[CLASS_NAMES].to_numpy().argmax(axis=1)
    out["predicted_class"] = out["predicted_class"].map(
        {i: c for i, c in enumerate(CLASS_NAMES)}
    )
    return out


# ──────────────────────────────────────────────────────────────────────────────
# training
# ──────────────────────────────────────────────────────────────────────────────

def _load_label_log(label_log_path: Path) -> pd.DataFrame:
    if not label_log_path.exists():
        raise FileNotFoundError(f"label log missing: {label_log_path}")
    return pd.read_json(label_log_path, lines=True)


def _active_labels(log: pd.DataFrame, sid: str) -> pd.DataFrame:
    sub = log[log["sid"].astype(str) == sid].copy()
    if sub.empty:
        return pd.DataFrame(columns=["hcr_id", "label"])
    tomb_ids: set[int] = set()
    if (sub["label"] == "_undone_").any():
        for _, r in sub[sub["label"] == "_undone_"].iterrows():
            ub = r.get("undoes") or {}
            try:
                tomb_ids.add(int(ub.get("hcr_id", -1)))
            except (TypeError, ValueError):
                pass
    sub = sub[sub["label"].isin(["good", "bad", "bad_ok", "merged", "unsure"])]
    sub = sub[~sub["hcr_id"].astype(int).isin(tomb_ids)]
    if "ts" in sub.columns:
        sub = sub.sort_values("ts")
    sub = sub.drop_duplicates(subset=["hcr_id"], keep="last")
    return sub[["hcr_id", "label"]].reset_index(drop=True)


def _build_matrix(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURE_COLUMNS].copy()
    for c in X.columns:
        if pd.api.types.is_bool_dtype(X[c]):
            X[c] = X[c].astype("float32")
    return X


def _early_stop_split(n: int, frac: float = 0.15, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    cut = max(1, int(round(n * (1 - frac))))
    return idx[:cut], idx[cut:]


def _train_binary(X_tr: pd.DataFrame, y_tr: np.ndarray) -> lgb.Booster:
    tr, va = _early_stop_split(len(X_tr), frac=0.15, seed=42)
    train_set = lgb.Dataset(X_tr.iloc[tr], y_tr[tr])
    valid_set = lgb.Dataset(X_tr.iloc[va], y_tr[va], reference=train_set)
    return lgb.train(
        LGB_BINARY, train_set,
        num_boost_round=N_ESTIMATORS,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(EARLY_STOP), lgb.log_evaluation(0)],
    )


def _train_multi(X_tr: pd.DataFrame, y_tr: np.ndarray) -> lgb.Booster:
    tr, va = _early_stop_split(len(X_tr), frac=0.15, seed=42)
    train_set = lgb.Dataset(X_tr.iloc[tr], y_tr[tr])
    valid_set = lgb.Dataset(X_tr.iloc[va], y_tr[va], reference=train_set)
    return lgb.train(
        LGB_MULTI, train_set,
        num_boost_round=N_ESTIMATORS,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(EARLY_STOP), lgb.log_evaluation(0)],
    )


def train(
    subjects: list[str],
    label_log_path: Path,
    out_dir: Path = _cfg.MODELS_DIR,
) -> dict:
    """LOSO cross-validation + production model training.

    Mirrors 05h_train_stage2_v5d.main() but uses this module's API.

    Writes to out_dir:
        roi_quality_stage2_binary_v5d.txt
        roi_quality_stage2_4class_v5d.txt
        roi_quality_stage2_meta_v5d.json
        {sid}_stage2_binary_score_v5d.parquet  (per-subject OOF)
        {sid}_stage2_4class_proba_v5d.parquet  (per-subject OOF)

    Returns the meta dict.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    log = _load_label_log(label_log_path)
    print(f"label log: {len(log)} rows")

    feats = {sid: extract_features(sid) for sid in subjects}
    labs  = {sid: _active_labels(log, sid) for sid in subjects}

    print("\nactive label counts per subject:")
    for sid in subjects:
        c = labs[sid]["label"].value_counts().to_dict()
        print(f"  {sid}: total={len(labs[sid]):3d}  " + ", ".join(
            f"{k}={c.get(k, 0)}" for k in ["good", "bad", "bad_ok", "merged", "unsure"]
        ))

    # ── BINARY LOSO ──────────────────────────────────────────────────────────
    print("\n" + "=" * 62 + "\nBINARY (v5d)\n" + "=" * 62)
    bin_metrics: list[dict] = []

    for held in subjects:
        tr_X, tr_y = [], []
        for sid in subjects:
            if sid == held:
                continue
            f = feats[sid]; l = labs[sid]
            l = l[l["label"].isin(BINARY_POS | BINARY_NEG)].copy()
            l["y"] = l["label"].isin(BINARY_POS).astype("int8")
            merged = f.merge(l[["hcr_id", "y"]], on="hcr_id", how="inner")
            tr_X.append(_build_matrix(merged))
            tr_y.append(merged["y"].to_numpy("int8"))
        X_tr = pd.concat(tr_X, axis=0).reset_index(drop=True)
        y_tr = np.concatenate(tr_y)

        f_held  = feats[held]
        X_held  = _build_matrix(f_held)
        l_held  = labs[held]
        l_held  = l_held[l_held["label"].isin(BINARY_POS | BINARY_NEG)].copy()
        l_held["y"] = l_held["label"].isin(BINARY_POS).astype("int8")

        booster    = _train_binary(X_tr, y_tr)
        y_pred_all = booster.predict(X_held, num_iteration=booster.best_iteration)

        scored  = pd.DataFrame({"hcr_id": f_held["hcr_id"].to_numpy(), "score": y_pred_all})
        eval_df = l_held.merge(scored, on="hcr_id", how="left")
        y_eval  = eval_df["y"].to_numpy("int8")
        p_eval  = eval_df["score"].to_numpy("float64")

        auc   = roc_auc_score(y_eval, p_eval) if len(np.unique(y_eval)) > 1 else float("nan")
        ap    = average_precision_score(y_eval, p_eval) if len(np.unique(y_eval)) > 1 else float("nan")
        brier = brier_score_loss(y_eval, p_eval)
        acc   = accuracy_score(y_eval, (p_eval >= 0.5).astype("int8"))

        bin_metrics.append({
            "held_subject": held,
            "n_train": int(len(X_tr)),
            "n_eval": int(len(y_eval)),
            "auc": auc, "ap": ap, "brier": brier, "acc@0.5": acc,
            "best_iter": booster.best_iteration,
            "n_pos_train": int((y_tr == 1).sum()),
            "n_neg_train": int((y_tr == 0).sum()),
            "n_pos_eval": int((y_eval == 1).sum()),
            "n_neg_eval": int((y_eval == 0).sum()),
        })
        print(
            f"[hold {held}] n_tr={len(X_tr):3d} (pos {(y_tr==1).sum()}/neg {(y_tr==0).sum()})  "
            f"n_ev={len(y_eval):3d}  AUC={auc:.4f}  AP={ap:.4f}  Brier={brier:.4f}  "
            f"acc@0.5={acc:.3f}  iter={booster.best_iteration}"
        )

        oof = pd.DataFrame({
            "hcr_id": f_held["hcr_id"].to_numpy("int64"),
            "score":  y_pred_all.astype("float32"),
        }).merge(
            l_held.rename(columns={"label": "human_label"})[["hcr_id", "human_label"]],
            on="hcr_id", how="left",
        )
        oof.to_parquet(out_dir / f"{held}_stage2_binary_score_v5d.parquet", index=False)

    bm = pd.DataFrame(bin_metrics).sort_values("held_subject")
    print(f"\nbinary v5d LOSO:  mean AUC={bm['auc'].mean():.4f}  "
          f"AP={bm['ap'].mean():.4f}  Brier={bm['brier'].mean():.4f}")

    # ── 4-CLASS LOSO ─────────────────────────────────────────────────────────
    print("\n" + "=" * 62 + "\n4-CLASS (v5d)\n" + "=" * 62)
    multi_metrics: list[dict] = []
    overall_cm = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int)

    for held in subjects:
        tr_X, tr_y = [], []
        for sid in subjects:
            if sid == held:
                continue
            f = feats[sid]; l = labs[sid]
            l = l[l["label"].isin(CLASS_NAMES)].copy()
            l["y"] = l["label"].map(CLASS_TO_IDX).astype("int8")
            merged = f.merge(l[["hcr_id", "y"]], on="hcr_id", how="inner")
            tr_X.append(_build_matrix(merged))
            tr_y.append(merged["y"].to_numpy("int8"))
        X_tr = pd.concat(tr_X, axis=0).reset_index(drop=True)
        y_tr = np.concatenate(tr_y)

        f_held  = feats[held]
        X_held  = _build_matrix(f_held)
        l_held  = labs[held]
        l_held  = l_held[l_held["label"].isin(CLASS_NAMES)].copy()
        l_held["y"] = l_held["label"].map(CLASS_TO_IDX).astype("int8")

        booster    = _train_multi(X_tr, y_tr)
        proba_full = booster.predict(X_held, num_iteration=booster.best_iteration)

        scored  = pd.DataFrame(proba_full, columns=[f"p_{c}" for c in CLASS_NAMES])
        scored["hcr_id"] = f_held["hcr_id"].to_numpy("int64")
        eval_df = l_held.merge(scored, on="hcr_id", how="left")
        y_eval  = eval_df["y"].to_numpy("int8")
        proba_eval = eval_df[[f"p_{c}" for c in CLASS_NAMES]].to_numpy("float64")
        y_pred  = proba_eval.argmax(axis=1)
        acc     = accuracy_score(y_eval, y_pred)
        f1m     = f1_score(y_eval, y_pred, average="macro", zero_division=0)
        f1p     = f1_score(y_eval, y_pred, average=None,
                           labels=list(range(len(CLASS_NAMES))), zero_division=0)
        cm      = confusion_matrix(y_eval, y_pred, labels=list(range(len(CLASS_NAMES))))
        overall_cm += cm

        row = {
            "held_subject": held,
            "n_train": int(len(X_tr)),
            "n_eval": int(len(y_eval)),
            "acc": acc, "f1_macro": f1m,
            "best_iter": booster.best_iteration,
        }
        for c, fv in zip(CLASS_NAMES, f1p):
            row[f"f1_{c}"] = float(fv)
        for c in CLASS_NAMES:
            row[f"n_train_{c}"] = int((y_tr == CLASS_TO_IDX[c]).sum())
            row[f"n_eval_{c}"]  = int((y_eval == CLASS_TO_IDX[c]).sum())
        multi_metrics.append(row)

        print(
            f"[hold {held}] n_tr={len(X_tr):3d}  n_ev={len(y_eval):3d}  "
            f"acc={acc:.3f}  f1_macro={f1m:.3f}  "
            + " ".join(f"f1_{c}={f:.2f}" for c, f in zip(CLASS_NAMES, f1p))
        )

        oof = scored[["hcr_id"] + [f"p_{c}" for c in CLASS_NAMES]].merge(
            l_held.rename(columns={"label": "human_label"})[["hcr_id", "human_label"]],
            on="hcr_id", how="left",
        )
        oof.to_parquet(out_dir / f"{held}_stage2_4class_proba_v5d.parquet", index=False)

    mm = pd.DataFrame(multi_metrics).sort_values("held_subject")
    print(f"\n4-class v5d LOSO:  mean acc={mm['acc'].mean():.4f}  "
          f"mean f1_macro={mm['f1_macro'].mean():.4f}")

    # ── PRODUCTION MODELS ────────────────────────────────────────────────────
    print("\n" + "=" * 62 + "\nproduction v5d models\n" + "=" * 62)

    Xb_parts, yb_parts = [], []
    for sid in subjects:
        f = feats[sid]; l = labs[sid]
        l = l[l["label"].isin(BINARY_POS | BINARY_NEG)].copy()
        l["y"] = l["label"].isin(BINARY_POS).astype("int8")
        m = f.merge(l[["hcr_id", "y"]], on="hcr_id", how="inner")
        Xb_parts.append(_build_matrix(m)); yb_parts.append(m["y"].to_numpy("int8"))
    Xb = pd.concat(Xb_parts, axis=0).reset_index(drop=True)
    yb = np.concatenate(yb_parts)
    n_iter_b = max(int(np.median([m["best_iter"] for m in bin_metrics])), 80)
    bin_prod = lgb.train(LGB_BINARY, lgb.Dataset(Xb, yb),
                         num_boost_round=n_iter_b,
                         callbacks=[lgb.log_evaluation(0)])
    bin_prod.save_model(str(out_dir / "roi_quality_stage2_binary_v5d.txt"))
    print(f"  binary: {len(Xb)} rows, {n_iter_b} iters")

    Xm_parts, ym_parts = [], []
    for sid in subjects:
        f = feats[sid]; l = labs[sid]
        l = l[l["label"].isin(CLASS_NAMES)].copy()
        l["y"] = l["label"].map(CLASS_TO_IDX).astype("int8")
        m = f.merge(l[["hcr_id", "y"]], on="hcr_id", how="inner")
        Xm_parts.append(_build_matrix(m)); ym_parts.append(m["y"].to_numpy("int8"))
    Xm = pd.concat(Xm_parts, axis=0).reset_index(drop=True)
    ym = np.concatenate(ym_parts)
    n_iter_m = max(int(np.median([m["best_iter"] for m in multi_metrics])), 80)
    multi_prod = lgb.train(LGB_MULTI, lgb.Dataset(Xm, ym),
                           num_boost_round=n_iter_m,
                           callbacks=[lgb.log_evaluation(0)])
    multi_prod.save_model(str(out_dir / "roi_quality_stage2_4class_v5d.txt"))
    print(f"  4-class: {len(Xm)} rows, {n_iter_m} iters")

    meta: dict = {
        "version": "v5d",
        "feature_columns": FEATURE_COLUMNS,
        "subjects": subjects,
        "class_names": CLASS_NAMES,
        "binary_pos": sorted(BINARY_POS),
        "binary_neg": sorted(BINARY_NEG),
        "pct_rank_columns": PCT_RANK_COLS,
        "binary": {
            "n_train_total": int(len(Xb)),
            "n_iter_prod": int(n_iter_b),
            "loso_mean_auc": float(bm["auc"].mean()),
            "loso_mean_ap": float(bm["ap"].mean()),
            "loso_mean_brier": float(bm["brier"].mean()),
            "loso_mean_acc05": float(bm["acc@0.5"].mean()),
            "params": LGB_BINARY,
        },
        "four_class": {
            "n_train_total": int(len(Xm)),
            "n_iter_prod": int(n_iter_m),
            "loso_mean_acc": float(mm["acc"].mean()),
            "loso_mean_f1_macro": float(mm["f1_macro"].mean()),
            "params": LGB_MULTI,
        },
    }
    (out_dir / "roi_quality_stage2_meta_v5d.json").write_text(json.dumps(meta, indent=2))
    print(f"  meta -> {out_dir / 'roi_quality_stage2_meta_v5d.json'}")
    return meta
