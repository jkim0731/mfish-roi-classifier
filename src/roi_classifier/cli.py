"""Command-line interface for mfish-roi-classifier.

Subcommands
-----------
build-bbox <sid>
    Build the per-cell tight-bbox cache (level-2 voxel frame) for subject
    <sid> into config.TIGHT_BBOX_DIR.  This is the cold-start prerequisite the
    shape/axis/surface feature extractors read.  --force rebuilds.

build-features <sid>
    Cold-start one subject end-to-end: build the tight-bbox cache (if needed)
    then all four feature-group parquets into config.ROI_QUALITY_DIR, leaving
    the subject ready for `predict`.

predict <sid>
    Extract features (from cached parquets) and run inference for subject
    <sid>, writing the contract parquet to config.ROI_QUALITY_DIR:
        {sid}_stage2_4class_proba_v5d_um.parquet
    Columns: hcr_id, p_bad, p_bad_ok, p_good, p_merged, human_label
    (human_label is always present; null for ROIs not in the label log)

train
    LOSO cross-validation + production model training on all labelled
    subjects found in the label log.  Writes model .txt files and a meta
    JSON to config.MODELS_DIR.

Usage examples
--------------
    # cold-start a fresh subject, then predict:
    roi-classifier build-features 790322
    roi-classifier predict 790322

    MFISH_DATA_ROOT=/data roi-classifier predict 790322
    roi-classifier train
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _cmd_predict(args: argparse.Namespace) -> int:
    from . import config as _cfg
    from .features import extract_features
    from .model import predict, _load_label_log, _active_labels

    sid = str(args.sid)
    print(f"[predict] subject={sid}")
    print(f"  reading feature parquets from: {_cfg.ROI_QUALITY_DIR}")
    print(f"  reading models from:           {_cfg.MODELS_DIR}")

    feat = extract_features(sid)
    print(f"  features: {len(feat)} ROIs, {feat.shape[1]-1} feature cols")

    _binary_score, four_class_proba = predict(feat)

    # Build the contract parquet.
    # Schema: hcr_id, p_bad, p_bad_ok, p_good, p_merged, human_label
    out = four_class_proba[["hcr_id", "bad", "bad_ok", "good", "merged"]].rename(
        columns={"bad": "p_bad", "bad_ok": "p_bad_ok",
                 "good": "p_good", "merged": "p_merged"}
    )

    # Always add human_label column (null for unlabelled rows).
    # Populate from the label log when it exists and has entries for this subject.
    import pandas as _pd
    out["human_label"] = _pd.NA
    label_log = _cfg.ROI_QUALITY_DIR / "roi_qc_actions.jsonl"
    if label_log.exists():
        try:
            log = _load_label_log(label_log)
            labs = _active_labels(log, sid)
            if not labs.empty:
                labs_renamed = labs.rename(columns={"label": "human_label"})
                out = out.drop(columns=["human_label"]).merge(
                    labs_renamed, on="hcr_id", how="left"
                )
                print(f"  human_label: {labs['label'].value_counts().to_dict()}")
        except Exception as e:
            print(f"  [warn] could not attach human labels: {e}", file=sys.stderr)

    # Write the contract parquet.
    _cfg.ROI_QUALITY_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _cfg.ROI_QUALITY_DIR / f"{sid}_stage2_4class_proba_v5d_um.parquet"
    out.to_parquet(out_path, index=False)
    print(f"  contract parquet written -> {out_path}")
    print(f"  shape: {out.shape}, cols: {list(out.columns)}")

    # Validation: check schema exactly.
    expected_cols = ["hcr_id", "p_bad", "p_bad_ok", "p_good", "p_merged", "human_label"]
    missing = set(expected_cols) - set(out.columns)
    if missing:
        print(f"  [ERROR] contract parquet missing columns: {missing}", file=sys.stderr)
        return 1
    print(f"  schema OK (argmax ok ids: {int((out[['p_good','p_bad_ok']].sum(axis=1) >= 0.5).sum())} / {len(out)})")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from . import config as _cfg
    from .model import train
    from .benchmark_data_loader import BENCHMARK_SUBJECTS

    label_log = Path(args.label_log) if args.label_log else (
        _cfg.ROI_QUALITY_DIR / "roi_qc_actions.jsonl"
    )
    if not label_log.exists():
        print(f"[train] label log not found: {label_log}", file=sys.stderr)
        print("  Pass --label-log <path> or set MFISH_ROI_QUALITY_DIR.", file=sys.stderr)
        return 1

    subjects = args.subjects if args.subjects else BENCHMARK_SUBJECTS
    print(f"[train] subjects={subjects}")
    print(f"  label log: {label_log}")
    print(f"  output dir: {_cfg.MODELS_DIR}")

    meta = train(subjects=subjects, label_log_path=label_log, out_dir=_cfg.MODELS_DIR)
    print(f"[train] done. binary LOSO AUC={meta['binary']['loso_mean_auc']:.4f}  "
          f"4-class f1_macro={meta['four_class']['loso_mean_f1_macro']:.4f}")
    return 0


def _cmd_build_bbox(args: argparse.Namespace) -> int:
    from .feat_tight_bbox import build_tight_bbox_sid, tight_bbox_cache_path

    sid = str(args.sid)
    print(f"[build-bbox] subject={sid}")
    df = build_tight_bbox_sid(sid, cache=True, force=args.force)
    print(f"[build-bbox] {sid}: {len(df)} cells -> {tight_bbox_cache_path(sid)}")
    return 0


# Module-level workers so they are picklable for a spawn ProcessPool.
# Each loads its own subject + zarr handles inside the worker process.
def _crops_worker(sid: str, force: bool) -> str:
    from .feat_per_cell_crops import build_per_cell_crops_sid
    build_per_cell_crops_sid(sid, cache=True, force=force)
    return "crops"


def _feature_worker(sid: str, group: str) -> str:
    from .benchmark_data_loader import load_subject
    s = load_subject(sid)
    mod = {
        "shape": "feat_shape", "axis": "feat_axis",
        "surface": "feat_surface", "protrusion": "feat_protrusion",
    }[group]
    import importlib
    importlib.import_module(f".{mod}", __package__).compute(s, cache=True)
    return group


def _cmd_build_crops(args: argparse.Namespace) -> int:
    from .feat_per_cell_crops import build_per_cell_crops_sid

    sid = str(args.sid)
    print(f"[build-crops] subject={sid}")
    path = build_per_cell_crops_sid(sid, cache=True, force=args.force)
    print(f"[build-crops] {sid}: -> {path}")
    return 0


def _cmd_build_features(args: argparse.Namespace) -> int:
    """Cold-start a subject: tight-bbox -> unified single-pass feature extraction.

    One extractor computes all feature families (shape, axis, surface,
    protrusion, intensity, adjacency) in a single z-strip pass — each cell's
    mask, opening, and 405 crop are computed once and shared across families
    (no redundant volume reads, no per-cell-crops cache). The z-strip loop
    parallelises across cores via `MFISH_FEAT_WORKERS` (default cpu-2).
    """
    from .benchmark_data_loader import load_subject
    from .feat_tight_bbox import build_tight_bbox
    from . import feat_shape

    sid = str(args.sid)
    print(f"[build-features] subject={sid}")
    s = load_subject(sid)
    build_tight_bbox(s, cache=True, force=args.force)   # locates cells; the only prerequisite
    feat_shape.compute(s, cache=True)                   # unified pass → {sid}_features_all.parquet
    print(f"[build-features] {sid}: done — run `roi-classifier predict {sid}` next.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="roi-classifier",
        description="mfish ROI-quality classifier — predict and train.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_predict = sub.add_parser("predict", help="Run inference for one subject.")
    p_predict.add_argument("sid", help="Subject ID (e.g. 790322)")

    p_bbox = sub.add_parser(
        "build-bbox",
        help="Build the per-cell tight-bbox cache for one subject (cold-start prerequisite).",
    )
    p_bbox.add_argument("sid", help="Subject ID (e.g. 790322)")
    p_bbox.add_argument("--force", action="store_true",
                        help="Rebuild even if the cache already exists.")

    p_crops = sub.add_parser(
        "build-crops",
        help="Build the per-cell mask-crop cache (needs the tight-bbox cache first).",
    )
    p_crops.add_argument("sid", help="Subject ID (e.g. 790322)")
    p_crops.add_argument("--force", action="store_true",
                         help="Rebuild even if the cache already exists.")

    p_feats = sub.add_parser(
        "build-features",
        help="Cold-start: build tight-bbox + run the unified feature extractor for one subject.",
    )
    p_feats.add_argument("sid", help="Subject ID (e.g. 790322)")
    p_feats.add_argument("--force", action="store_true",
                         help="Rebuild the tight-bbox cache even if it exists.")
    # Parallelism is controlled by MFISH_FEAT_WORKERS (z-strip cell-chunking inside
    # the unified pass), not a CLI flag.

    p_train = sub.add_parser("train", help="LOSO cross-validation + production model training.")
    p_train.add_argument(
        "--label-log", dest="label_log", default=None,
        help="Path to roi_qc_actions.jsonl (default: ROI_QUALITY_DIR/roi_qc_actions.jsonl)"
    )
    p_train.add_argument(
        "subjects", nargs="*",
        help="Subject IDs to include (default: all 6 benchmark subjects)"
    )

    args = parser.parse_args(argv)
    if args.command == "predict":
        return _cmd_predict(args)
    if args.command == "build-bbox":
        return _cmd_build_bbox(args)
    if args.command == "build-crops":
        return _cmd_build_crops(args)
    if args.command == "build-features":
        return _cmd_build_features(args)
    if args.command == "train":
        return _cmd_train(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
