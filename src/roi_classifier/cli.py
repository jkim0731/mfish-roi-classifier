"""Command-line interface for mfish-roi-classifier.

Subcommands
-----------
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
    from .roi_quality_v5d import extract_features, predict

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
            from .roi_quality_v5d import _load_label_log, _active_labels
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
    from .roi_quality_v5d import train
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="roi-classifier",
        description="mfish ROI-quality classifier — predict and train.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_predict = sub.add_parser("predict", help="Run inference for one subject.")
    p_predict.add_argument("sid", help="Subject ID (e.g. 790322)")

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
    if args.command == "train":
        return _cmd_train(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
