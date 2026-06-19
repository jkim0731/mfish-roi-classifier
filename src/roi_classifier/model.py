"""Production API for the ROI-quality classifier.

Public entry points:
    predict_subject(sid)  — extract features + run both models → per-ROI scores.
    predict(features_df)  — run both models on a pre-built feature matrix.
    train(subjects, ...)  — LOSO cross-validation + production model training.

Feature extraction is delegated to ``features.extract_features``.
Model artifacts are loaded from ``config.MODELS_DIR`` (defaults to the
``models/`` directory at the repository root via ``MFISH_MODELS_DIR``).
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
# constants
# ──────────────────────────────────────────────────────────────────────────────

# Feature parquets are read from ROI_QUALITY_DIR; models from MODELS_DIR.
# Both default to the same location but can be separated via env vars.
CACHE_DIR = _cfg.ROI_QUALITY_DIR

# Model files shipped inside models/ (version-agnostic names; lineage in meta).
MODEL_BINARY = _cfg.MODELS_DIR / "roi_quality_binary.txt"
MODEL_FOUR_CLASS = _cfg.MODELS_DIR / "roi_quality_4class.txt"
META_JSON = _cfg.MODELS_DIR / "roi_quality_meta.json"

CLASS_NAMES = ["bad", "bad_ok", "good", "merged"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

BINARY_POS = {"good", "bad_ok"}
BINARY_NEG = {"bad", "merged"}

# Feature columns — the model's feature-name contract. µm shape / surface / axis /
# protrusion features + a few raw voxel counts + 405 intensity + neighbour-quality
# aggregates. 405-only; no percentile-rank columns.
#
# Two sources, depending on the capsule:
#   * predict / Capsule 1  — read verbatim from the shipped model meta (a model
#     exists, so its schema is authoritative).
#   * train_embedded / Capsule 3 — there is no base model; the schema is DERIVED
#     from the labels themselves (each self-contained label embeds its features by
#     name) via `_derive_feature_columns`, then installed with `_set_feature_columns`.
# So the import-time load is tolerant: no meta → empty list (training will fill it).
def _load_feature_columns() -> list[str]:
    meta_path = META_JSON
    if not meta_path.exists():
        return []
    with open(meta_path) as _f:
        _meta = json.load(_f)
    return list(_meta.get("feature_columns", []))

FEATURE_COLUMNS: list[str] = _load_feature_columns()


def _set_feature_columns(cols) -> None:
    """Install the active feature schema (module-global) for this process. Used by
    `train_embedded` to adopt the schema derived from the labels (design A — the
    training capsule needs no base model)."""
    global FEATURE_COLUMNS
    FEATURE_COLUMNS = list(cols)

# Percentile-rank feature columns: none (kept for meta parity; pct_rank_columns: []).
PCT_RANK_COLS: list[str] = []

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
    bin_path   = model_dir / "roi_quality_binary.txt"
    multi_path = model_dir / "roi_quality_4class.txt"
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
    from .features import extract_features
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
    """Read one label-log file, or merge every ``*.jsonl`` under a directory.

    Labels are kept as **per-session, timestamped assets** (one jsonl per
    labelling session). Passing the directory of attached label assets merges
    them all into one event frame; newest-wins is resolved by ``_active_labels``
    using each event's ``ts``.
    """
    p = Path(label_log_path)
    files = sorted(p.glob("*.jsonl")) if p.is_dir() else ([p] if p.exists() else [])
    if not files:
        raise FileNotFoundError(f"no label log(s) found at: {label_log_path}")
    return pd.concat([pd.read_json(f, lines=True) for f in files], ignore_index=True)


_LABELS = ("good", "bad", "bad_ok", "merged", "unsure")


def _resolve_active(sub: pd.DataFrame) -> pd.DataFrame:
    """Last-event-per-(hcr_id) resolution over a single subject's event stream.

    For each cell, the *latest* event by ``ts`` wins: a real label sets it, an
    ``_undone_`` event (whose target is in ``undoes.hcr_id``) clears it. This
    handles label → undo → re-label correctly across merged sessions/assets.
    """
    if sub.empty:
        return pd.DataFrame(columns=["hcr_id", "label"])
    sub = sub.copy()

    def _target_hid(r):
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

    sub["_hid"] = sub.apply(_target_hid, axis=1)
    sub = sub[(sub["_hid"] >= 0) & (sub["label"].isin((*_LABELS, "_undone_")))]
    if sub.empty:
        return pd.DataFrame(columns=["hcr_id", "label"])
    if "ts" in sub.columns:
        sub = sub.sort_values("ts", kind="stable")
    last = sub.groupby("_hid", as_index=False).last()
    last = last[last["label"].isin(_LABELS)]          # drop cells whose last event was _undone_
    return pd.DataFrame({
        "hcr_id": last["_hid"].astype(int).to_numpy(),
        "label": last["label"].to_numpy(),
    }).reset_index(drop=True)


def _active_labels(log: pd.DataFrame, sid: str) -> pd.DataFrame:
    if log.empty:
        return pd.DataFrame(columns=["hcr_id", "label"])
    return _resolve_active(log[log["sid"].astype(str) == str(sid)])


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
    feats: dict | None = None,
) -> dict:
    """LOSO cross-validation + production model training.

    `feats`: optional {sid: DataFrame[hcr_id, *FEATURE_COLUMNS]} of pre-built
    feature matrices. When None (default) they are extracted via
    `features.extract_features`. `train_embedded` passes the label assets'
    embedded features here so no extraction / attached HCR data is needed.

    Writes to out_dir:
        roi_quality_binary.txt
        roi_quality_4class.txt
        roi_quality_meta.json
        {sid}_roi_quality_binary_oof.parquet  (per-subject OOF)
        {sid}_roi_quality_4class_oof.parquet  (per-subject OOF)

    Returns the meta dict.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    log = _load_label_log(label_log_path)
    print(f"label log: {len(log)} rows")

    if feats is None:
        from .features import extract_features
        feats = {sid: extract_features(sid) for sid in subjects}
    labs  = {sid: _active_labels(log, sid) for sid in subjects}

    print("\nactive label counts per subject:")
    for sid in subjects:
        c = labs[sid]["label"].value_counts().to_dict()
        print(f"  {sid}: total={len(labs[sid]):3d}  " + ", ".join(
            f"{k}={c.get(k, 0)}" for k in ["good", "bad", "bad_ok", "merged", "unsure"]
        ))

    # ── BINARY LOSO ──────────────────────────────────────────────────────────
    print("\n" + "=" * 62 + "\nBINARY\n" + "=" * 62)
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
        oof.to_parquet(out_dir / f"{held}_roi_quality_binary_oof.parquet", index=False)

    bm = pd.DataFrame(bin_metrics).sort_values("held_subject")
    print(f"\nbinary LOSO:  mean AUC={bm['auc'].mean():.4f}  "
          f"AP={bm['ap'].mean():.4f}  Brier={bm['brier'].mean():.4f}")

    # ── 4-CLASS LOSO ─────────────────────────────────────────────────────────
    print("\n" + "=" * 62 + "\n4-CLASS\n" + "=" * 62)
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
        oof.to_parquet(out_dir / f"{held}_roi_quality_4class_oof.parquet", index=False)

    mm = pd.DataFrame(multi_metrics).sort_values("held_subject")
    print(f"\n4-class LOSO:  mean acc={mm['acc'].mean():.4f}  "
          f"mean f1_macro={mm['f1_macro'].mean():.4f}")

    # ── PRODUCTION MODELS ────────────────────────────────────────────────────
    print("\n" + "=" * 62 + "\nproduction models\n" + "=" * 62)

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
    bin_prod.save_model(str(out_dir / "roi_quality_binary.txt"))
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
    multi_prod.save_model(str(out_dir / "roi_quality_4class.txt"))
    print(f"  4-class: {len(Xm)} rows, {n_iter_m} iters")

    meta: dict = {
        "version": "1",
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
    (out_dir / "roi_quality_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  meta -> {out_dir / 'roi_quality_meta.json'}")
    return meta


# ──────────────────────────────────────────────────────────────────────────────
# embedded-features training (light: labels asset only, no extraction)
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_active_records(log: pd.DataFrame, sid: str) -> pd.DataFrame:
    """Winning labeled records for `sid` (newest-wins) with each record's embedded
    `features` dict expanded to FEATURE_COLUMNS columns (+ hcr_id, label, code_commit)."""
    cols = ["hcr_id", "label", "code_commit", *FEATURE_COLUMNS]
    sub = log[log["sid"].astype(str) == str(sid)].copy()
    if sub.empty:
        return pd.DataFrame(columns=cols)

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
    sub = sub[(sub["_hid"] >= 0) & (sub["label"].isin((*_LABELS, "_undone_")))]
    if sub.empty:
        return pd.DataFrame(columns=cols)
    if "ts" in sub.columns:
        sub = sub.sort_values("ts", kind="stable")
    last = sub.groupby("_hid", as_index=False).last()
    last = last[last["label"].isin(_LABELS)]
    if last.empty:
        return pd.DataFrame(columns=cols)
    fser = (last["features"].apply(lambda x: x if isinstance(x, dict) else {})
            if "features" in last.columns else pd.Series([{}] * len(last)))
    fnorm = pd.json_normalize(fser).reindex(columns=FEATURE_COLUMNS)
    out = pd.DataFrame({
        "hcr_id": last["_hid"].astype(int).to_numpy(),
        "label": last["label"].to_numpy(),
        "code_commit": (last["code_commit"].to_numpy() if "code_commit" in last.columns
                        else [None] * len(last)),
    })
    return pd.concat([out.reset_index(drop=True), fnorm.reset_index(drop=True)], axis=1)


def _warn_label_provenance(log: pd.DataFrame, subjects: list[str]) -> None:
    """Warn on conflicting (changed) labels and re-labeled ROIs. Warnings only —
    training proceeds on the newest-wins result. Feature-set validity is enforced
    by the embedded-feature coverage check in `train_embedded` (by feature name),
    NOT by code_commit: the repo commit changes for reasons unrelated to extraction,
    so it would false-alarm. `code_commit` stays in each record as provenance only."""
    sub = log[log["sid"].astype(str).isin([str(s) for s in subjects])]
    lab = sub[sub["label"].isin(_LABELS)]
    relabeled = changed = 0
    for sid in subjects:
        g = lab[lab["sid"].astype(str) == str(sid)].groupby("hcr_id")["label"]
        if g.ngroups == 0:
            continue
        relabeled += int((g.count() > 1).sum())
        changed += int((g.nunique() > 1).sum())
    if relabeled:
        print(f"[warn] re-labeled ROIs: {relabeled} cell(s) have >1 label event (newest-wins applied).")
    if changed:
        print(f"[warn] label name mismatch: {changed} cell(s) received conflicting label VALUES "
              f"across events (the newest wins).")


def _derive_feature_columns(log: pd.DataFrame, subjects: list[str]) -> list[str]:
    """Feature schema taken from the labels themselves — design A, so the training
    capsule needs no base model. STRICT: every feature-bearing label record (across
    all subjects and all merged label assets) must embed the IDENTICAL set of feature
    names, else raise. A disagreement means the labels were made with different
    extractor versions; the fix is to re-extract + re-label, not to train on a mixed
    set. Order is canonical (sorted) for reproducibility — order is irrelevant to the
    model (features are selected by name), but a stable order makes the meta diffable.
    """
    sub = log[log["sid"].astype(str).isin([str(s) for s in subjects])]
    sub = sub[sub["label"].isin(_LABELS)]            # real labels (tombstones carry no features)
    keysets: dict[frozenset, int] = {}
    feats_ser = sub["features"] if "features" in sub.columns else pd.Series([], dtype=object)
    for f in feats_ser:
        if isinstance(f, dict) and f:
            ks = frozenset(f.keys())
            keysets[ks] = keysets.get(ks, 0) + 1
    if not keysets:
        raise ValueError(
            "no embedded features found in any label record — cannot derive the feature "
            "schema. Were the labels back-filled into the self-contained schema?"
        )
    if len(keysets) > 1:
        items = sorted(keysets.items(), key=lambda kv: -kv[1])
        base = set(items[0][0])
        lines = ["embedded feature sets DISAGREE across label records — refusing to train on a "
                 "mixed feature set (re-extract + re-label to unify). Distinct sets found:"]
        for ks, n in items:
            extra, missing = sorted(set(ks) - base), sorted(base - set(ks))
            lines.append(f"  - {len(ks)} features, {n} record(s)"
                         + (f"; extra={extra}" if extra else "")
                         + (f"; missing={missing}" if missing else ""))
        raise ValueError("\n".join(lines))
    cols = sorted(next(iter(keysets)))
    print(f"feature schema derived from labels: {len(cols)} features, identical across all "
          f"{sum(keysets.values())} feature-bearing label record(s).")
    return cols


def train_embedded(
    subjects: list[str],
    label_log_path: Path,
    out_dir: Path = _cfg.MODELS_DIR,
    feature_columns: list[str] | None = None,
) -> dict:
    """LIGHT trainer: build the matrix ONLY from each label's embedded `features`
    (no feature extraction, no attached HCR/features assets, NO base model).

    The feature schema is DERIVED from the labels (`_derive_feature_columns`, strict
    same-set check) and installed for this run, then written into the new model's
    meta. Pass `feature_columns` to override the derivation (e.g. a pinned schema).
    Warns on conflicting labels and re-labeled ROIs."""
    log = _load_label_log(label_log_path)
    print(f"label log: {len(log)} rows")
    _warn_label_provenance(log, subjects)

    # Self-contained schema: learn the feature set from the labels, then adopt it so
    # every downstream consumer of FEATURE_COLUMNS (matrix build, meta write) uses it.
    if feature_columns is None:
        feature_columns = _derive_feature_columns(log, subjects)
    _set_feature_columns(feature_columns)

    feats: dict = {}
    for sid in subjects:
        rec = _resolve_active_records(log, sid)
        usable = rec.dropna(subset=FEATURE_COLUMNS, how="all")
        n_missing = len(rec) - len(usable)
        if n_missing:
            print(f"[warn] {sid}: {n_missing} active label(s) carry no current-feature-set values "
                  f"→ skipped (embedded feature names don't match; light trainer does NOT extract).")
        feats[sid] = usable[["hcr_id", *FEATURE_COLUMNS]].reset_index(drop=True)
    return train(subjects=subjects, label_log_path=label_log_path, out_dir=out_dir, feats=feats)
