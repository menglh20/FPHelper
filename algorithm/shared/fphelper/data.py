"""Read source CSVs without exporting names, accounts or cloud URLs."""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

ACTIONS = ("forehead_wrinkle", "eye_closure", "smile", "snarl", "lip_pucker")
ITEMS = ("rest_eye", "rest_cheek", "rest_mouth") + tuple("vol_" + a for a in ACTIONS) + tuple("syn_" + a for a in ACTIONS)
LOW = np.array([0, 0, 0] + [1] * 5 + [0] * 5)
HIGH = np.array([1, 2, 1] + [5] * 5 + [3] * 5)
PHASE = np.array([0] * 3 + list(range(1, 6)) * 2)
HEADER = ("id", "name", "result", "detail", "comment", "save_path", "time", "type", "fileId", "selfish", "disabledSide")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def decode_label(comment: str) -> np.ndarray:
    value = json.loads(comment)
    flat = []
    for group, keys in (("rest symmetry", ("eye", "cheek", "mouth")),
                        ("voluntary symmetry", ACTIONS), ("synkinesis", ACTIONS)):
        group_value = value[group]
        canonical = {k.replace(" ", "_"): v for k, v in group_value.items()}
        for key in keys:
            score = canonical[key]
            if isinstance(score, bool) or not isinstance(score, int):
                raise ValueError("Labels must be integers, not strings or booleans")
            flat.append(score)
    result = np.asarray(flat, dtype=np.int64)
    if np.any(result < LOW) or np.any(result > HIGH):
        raise ValueError("Sunnybrook subscore outside its legal range")
    return result


def totals(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    return 4 * labels[..., 3:8].sum(axis=-1) - 5 * labels[..., :3].sum(axis=-1) - labels[..., 8:].sum(axis=-1)


def read_doctor(path: Path, header: bool) -> dict:
    with path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.reader(source)
        if header and tuple(next(reader)) != HEADER:
            raise ValueError("Unexpected doctor01 CSV header")
        output = {}
        for values in reader:
            if len(values) != len(HEADER):
                raise ValueError("Unexpected CSV column count")
            row = dict(zip(HEADER, values))
            stem = datetime.strptime(row["time"], "%Y.%m.%d %H:%M:%S").strftime("%Y%m%d %H%M%S")
            if stem in output:
                raise ValueError("Duplicate assessment timestamp")
            output[stem] = {"labels": decode_label(row["comment"]),
                            "side": row["disabledSide"],
                            "object_fingerprint": hashlib.sha256(row["fileId"].encode()).hexdigest()}
    return output


def load_dataset(root: Path, output: Path, seed: int = 20261012) -> tuple[list[dict], dict]:
    doctors = {name: read_doctor(root / f"{name}.csv", name == "doctor01")
               for name in ("doctor01", "doctor02", "doctor03")}
    keys = set(doctors["doctor01"])
    if any(set(rows) != keys for rows in doctors.values()):
        raise ValueError("The three doctors do not describe the same assessments")
    videos = {p.stem: p for p in (root / "videos").glob("*.mp4")}
    if set(videos) != keys:
        raise ValueError("Video filenames and assessment timestamps do not match exactly")
    manifest = []
    for stem in sorted(keys):
        rows = [doctors[name][stem] for name in doctors]
        if len({r["object_fingerprint"] for r in rows}) != 1 or len({r["side"] for r in rows}) != 1:
            raise ValueError("Doctor records disagree on source video or patient side")
        manifest.append({"video_id": "v_" + hashlib.sha256(stem.encode()).hexdigest()[:12],
                         "video_path": str(videos[stem].resolve()),
                         "video_sha256": sha256_file(videos[stem]),
                         "side_metadata": rows[0]["side"], "patient_group_id": None,
                         "labels": {name: doctors[name][stem]["labels"].tolist() for name in doctors}})
    y = np.array([r["labels"]["doctor01"] for r in manifest])
    strata = np.digitize(totals(y), [20, 40, 60])
    groups = np.array([r["video_sha256"] for r in manifest])
    duplicate_files = len(groups) - len(np.unique(groups))
    splitter = (StratifiedGroupKFold(5, shuffle=True, random_state=seed) if duplicate_files
                else StratifiedKFold(5, shuffle=True, random_state=seed))
    folds = list(splitter.split(np.zeros(len(y)), strata, groups) if duplicate_files
                 else splitter.split(np.zeros(len(y)), strata))
    fold_ids = np.full(len(y), -1, dtype=int)
    for fold, (train, test) in enumerate(folds):
        if set(groups[train]) & set(groups[test]):
            raise ValueError("Identical video bytes appear in train and test")
        fold_ids[test] = fold
    for row, fold in zip(manifest, fold_ids):
        row["outer_fold"] = int(fold)
    audit = {"video_count": len(y), "doctor_rows": {name: len(rows) for name, rows in doctors.items()},
             "mapping": "assessment timestamp to local filename; same source-object fingerprint across three doctors",
             "duplicate_video_files": duplicate_files, "known_patient_groups": 0,
             "label_counts": {key: {str(score): int((y[:, j] == score).sum()) for score in range(LOW[j], HIGH[j] + 1)} for j, key in enumerate(ITEMS)},
             "source_csv_sha256": {name: sha256_file(root / f"{name}.csv") for name in doctors},
             "seed": seed, "fold_counts": np.bincount(fold_ids).tolist()}
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    (output / "data-audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    return manifest, audit
