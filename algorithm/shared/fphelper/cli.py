"""Commands: extract, train, run, infer. All clinical derivatives stay local."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

os.environ.setdefault("MPLCONFIGDIR", str(Path("artifacts/matplotlib").resolve()))

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from .data import ITEMS, PHASE, load_dataset, sha256_file, totals
from .metrics import evaluate, paired_comparison
from .training import nested_validation, fit_final_model


def extract(args):
    import torch
    from .video import VideoExtractor
    manifest, audit = load_dataset(Path(args.data_root), args.output, args.seed)
    device = ("mps" if torch.backends.mps.is_available() else "cpu") if args.device == "auto" else args.device
    torch.set_num_threads(4)
    extractor = VideoExtractor(Path(args.face_model or Path(args.data_root) / "face_landmarker.task"), args.weights, device=device)
    (args.output / "extractor-config.json").write_text(json.dumps(extractor.config, indent=2))
    quality = []
    for index, row in enumerate(manifest):
        _, _, q = extractor.cached(row, args.output / "features")
        quality.append({"video_id": row["video_id"], **q})
        (args.output / "quality.json").write_text(json.dumps(quality, indent=2))
        print(f"FEATURES {index + 1}/{len(manifest)} {q['status']}", flush=True)
    print(f"EXTRACTION COMPLETE {sum(q['status'] == 'ok' for q in quality)}/{len(manifest)}", flush=True)


def cached_features(output):
    manifest = json.loads((output / "manifest.json").read_text())
    config = json.loads((output / "extractor-config.json").read_text())
    quality, eligible, geometry, appearance = [], [], [], []
    for index, row in enumerate(manifest):
        key = hashlib.sha256((row["video_sha256"] + json.dumps(config, sort_keys=True)).encode()).hexdigest()
        q = json.loads((output / "features" / f"{key}.json").read_text())
        quality.append({"video_id": row["video_id"], **q})
        if q["status"] != "ok":
            continue
        with np.load(output / "features" / f"{key}.npz", allow_pickle=False) as cache:
            geometry.append(cache["geometry"].copy())
            appearance.append(cache["appearance"].copy())
        eligible.append(index)
    if len(eligible) < 15:
        raise RuntimeError("Insufficient eligible videos for nested five-fold validation")
    return manifest, config, quality, np.array(eligible), np.asarray(geometry), np.asarray(appearance)


def training(args):
    from .report import write_report
    started = time.perf_counter()
    manifest, config, quality, eligible, geometry, appearance = cached_features(args.output)
    labels = {name: np.array([manifest[i]["labels"][name] for i in eligible]) for name in ("doctor01", "doctor02", "doctor03")}
    truth = labels["doctor01"]
    folds = np.array([manifest[i]["outer_fold"] for i in eligible])
    groups = np.array([manifest[i]["video_sha256"] for i in eligible])
    if set(folds) != set(range(5)):
        raise RuntimeError("The eligible data do not cover all five outer folds")
    def progress(log):
        (args.output / f"fold-{log['fold']}.json").write_text(json.dumps(log, indent=2))
        print(f"TRAIN fold={log['fold']} train={log['train_count']} test={log['test_count']} seconds={log['elapsed_seconds']:.1f} lambda={log['selected_regularization']}", flush=True)
    with threadpool_limits(limits=4):
        predictions, logs = nested_validation(geometry, appearance, truth, folds, groups, args.seed, progress)
        metrics = {name: evaluate(truth, value, args.seed) for name, value in {**predictions, "doctor02": labels["doctor02"], "doctor03": labels["doctor03"]}.items()}
        paired = {name: paired_comparison(truth, predictions["SB3"], labels[name], args.seed) for name in ("doctor02", "doctor03")}
        print("FITTING FINAL MODEL: full-data training is separate from reported out-of-fold results", flush=True)
        final_model, final_log = fit_final_model(geometry, appearance, truth, groups, args.seed)
    model_path = args.output / "sunnybrook-sb3.joblib"
    joblib.dump(final_model, model_path)
    reloaded = joblib.load(model_path)
    if not np.array_equal(reloaded.predict(geometry[:2], appearance[:2]), final_model.predict(geometry[:2], appearance[:2])):
        raise RuntimeError("Exported model does not reproduce its predictions")
    metadata = {"schema_version": 1, "model_version": "sunnybrook-first-validation-v1", "feature_config": config,
                "seed": args.seed, "trained_video_count": len(eligible), "items": list(ITEMS),
                "supported_labels": {key: head.supported.tolist() for key, head in zip(ITEMS, final_model.heads)},
                "model_sha256": sha256_file(model_path), "experimental": True, "final_training": final_log}
    (args.output / "model-metadata.json").write_text(json.dumps(metadata, indent=2))
    baseline_all = {name: evaluate(np.array([r["labels"]["doctor01"] for r in manifest]),
                                  np.array([r["labels"][name] for r in manifest]), args.seed)
                    for name in ("doctor02", "doctor03")}
    source = Path(__file__).parent
    versions = {}
    for name in ("numpy", "scipy", "scikit-learn", "torch", "torchvision", "mediapipe", "pillow"):
        from importlib.metadata import version
        versions[name] = version(name)
    code_hashes = {p.name: sha256_file(p) for p in sorted(source.glob("*.py"))}
    ffmpeg = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True).stdout.splitlines()[0]
    result = {"source_count": len(manifest), "eligible_count": len(eligible), "eligible_indices": eligible.tolist(),
              "metrics": metrics, "doctor_metrics_all_videos": baseline_all, "paired_comparison": paired,
              "seed": args.seed, "device": config["device"], "feature_config": config, "fold_logs": logs,
              "dependencies": versions, "python": platform.python_version(), "ffmpeg": ffmpeg,
              "code_sha256": code_hashes, "elapsed_training_seconds": time.perf_counter() - started,
              "primary_model": "SB3", "model_reload_verified": True,
              "failure_ids": [q["video_id"] for q in quality if q["status"] != "ok"]}
    (args.output / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    np.savez_compressed(args.output / "out-of-fold.npz", eligible=eligible, truth=truth, folds=folds, **predictions)
    with (args.output / "out-of-fold.csv").open("w", newline="") as stream:
        fields = ["video_id", "fold", "feature_status", "ground_truth_total", "doctor02_total", "doctor03_total"]
        fields += [f"{family}_raw_total" for family in predictions]
        fields += ["ground_truth_" + key for key in ITEMS] + ["SB3_" + key for key in ITEMS]
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        location = {int(index): i for i, index in enumerate(eligible)}
        for index, record in enumerate(manifest):
            row = {"video_id": record["video_id"], "fold": record["outer_fold"], "feature_status": quality[index]["status"],
                   "ground_truth_total": int(totals(np.array(record["labels"]["doctor01"]))),
                   "doctor02_total": int(totals(np.array(record["labels"]["doctor02"]))),
                   "doctor03_total": int(totals(np.array(record["labels"]["doctor03"])))}
            row.update({"ground_truth_" + key: value for key, value in zip(ITEMS, record["labels"]["doctor01"])})
            if index in location:
                i = location[index]
                row.update({f"{family}_raw_total": int(totals(value[i])) for family, value in predictions.items()})
                row.update({"SB3_" + key: int(value) for key, value in zip(ITEMS, predictions["SB3"][i])})
            writer.writerow(row)
    write_report(result, truth, predictions, args.report)
    for name, value in metrics.items():
        print(f"RESULT {name} n={value['count']} MAE={value['total_mae']:.4f} RMSE={value['total_rmse']:.4f} ICC={value['icc_absolute_single_mixed']}", flush=True)
    print(f"MODEL {model_path}; REPORT {args.report}", flush=True)


def inference(args):
    import torch
    from .video import VideoExtractor
    metadata = json.loads(args.model.with_name("model-metadata.json").read_text())
    reviewed = args.command == 'assess'
    if reviewed and args.output.exists():
        raise FileExistsError('Research assessment output already exists; choose a new output path')
    verified_phases, action_evidence = [], None
    if reviewed:
        from .score_contract import read_action_evidence, reviewed_result
        verified_phases, action_evidence = read_action_evidence(
            args.action_evidence, sha256_file(args.video), metadata['feature_config']['version'])
    if sha256_file(args.model) != metadata["model_sha256"]:
        raise RuntimeError("Model checksum does not match metadata")
    device = ("mps" if torch.backends.mps.is_available() else "cpu") if args.device == "auto" else args.device
    torch.set_num_threads(4)
    extractor = VideoExtractor(args.face_model, args.weights, fps=metadata["feature_config"]["fps"], device=device)
    for key, value in metadata["feature_config"].items():
        if key != "device" and extractor.config.get(key) != value:
            raise RuntimeError(f"Model/extractor mismatch: {key}")
    record = {"video_path": str(args.video), "video_sha256": sha256_file(args.video)}
    geometry, appearance, quality = extractor.cached(record, args.output.parent / "inference-cache")
    result = {"schema_version": 1, "model_version": metadata["model_version"], "quality": quality}
    if geometry is None:
        result.update(status=quality["status"], total=None, subscores=None)
    else:
        model = joblib.load(args.model)
        prediction = model.predict(geometry[None], appearance[None])[0]
        raw = int(totals(prediction))
        result.update(status="needs_review", reason="experimental_model_and_unverified_action_windows",
                      subscores={key: int(value) for key, value in zip(ITEMS, prediction)}, raw_total=raw,
                      total=raw if 0 <= raw <= 100 else None,
                      rest_penalty=int(5 * prediction[:3].sum()), voluntary_score=int(4 * prediction[3:8].sum()),
                      synkinesis_penalty=int(prediction[8:].sum()),
                      unsupported_predictions=[key for key, value in zip(ITEMS, prediction) if int(value) not in metadata["supported_labels"][key]])
    if reviewed:
        result = reviewed_result(result, verified_phases)
        result['action_evidence'] = action_evidence
        result['evaluation_scope'] = 'research_prediction; patient generalization not established'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"INFERENCE {result['status']} output={args.output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Local Sunnybrook first-validation pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("extract", "train", "run"):
        p = subparsers.add_parser(command)
        p.add_argument("--output", type=Path, default=Path("artifacts/first-validation"))
        p.add_argument("--seed", type=int, default=20261012)
        if command != "train":
            p.add_argument("--data-root", required=True)
            p.add_argument("--face-model")
            p.add_argument("--weights", type=Path, default=Path("artifacts/weights/resnet18-f37072fd.pth"))
            p.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
        if command != "extract":
            p.add_argument("--report", type=Path, default=Path("docs/experiments/first-validation.md"))
    for command in ('infer', 'assess'):
        p = subparsers.add_parser(command)
        p.add_argument("--video", type=Path, required=True)
        p.add_argument("--model", type=Path, default=Path("artifacts/first-validation/sunnybrook-sb3.joblib"))
        p.add_argument("--face-model", type=Path, required=True)
        p.add_argument("--weights", type=Path, default=Path("artifacts/weights/resnet18-f37072fd.pth"))
        p.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
        p.add_argument("--output", type=Path, default=Path("artifacts/first-validation/inference.json") if command == 'infer' else Path('artifacts/research-assessment/result.json'))
        if command == 'assess':
            p.add_argument('--action-evidence', type=Path, help='Optional reviewed actions tied to video SHA-256 and feature-window protocol')
    args = parser.parse_args()
    if args.command in ("extract", "run"):
        extract(args)
    if args.command in ("train", "run"):
        training(args)
    if args.command in ('infer', 'assess'):
        inference(args)


if __name__ == "__main__":
    main()
