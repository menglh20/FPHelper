"""Compare frozen legacy code, reproduced OOF methods and independent doctors."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import csv
import hashlib
import json
import os
import platform
from importlib.metadata import version
from pathlib import Path
import subprocess
import sys
import tempfile
import time

os.environ.setdefault('MPLCONFIGDIR', str(Path('artifacts/matplotlib').resolve()))

import numpy as np
from threadpoolctl import threadpool_limits
from fphelper.cli import cached_features
from fphelper.data import ITEMS, read_doctor, sha256_file, totals
from fphelper.training import nested_validation
from fphelper.versions import VERSION_FOLDERS
from metrics import evaluate, legal_mask, numeric_mask, paired

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
ALGORITHMS = ROOT / 'algorithm'
LEGACY = ALGORITHMS / 'legacy_v0'
SHARED = ALGORITHMS / 'shared/fphelper'
METHODS = ('LEGACY_E2E', 'LEGACY_WINDOW', 'SB0', 'SB1', 'SB2', 'SB3', 'doctor02', 'doctor03', 'MOCK100')
DESCRIPTIONS = {
    'LEGACY_E2E': '原版全视频极值取帧及手写评分，保留原始实现',
    'LEGACY_WINDOW': '研究窗口所选时间取帧及原版手写评分，仅作诊断',
    'SB0': '训练折的分项众数基线', 'SB1': '几何和 blendshape 摘要及有序回归',
    'SB2': '冻结 ResNet18 外观及有序回归', 'SB3': '几何与外观融合及有序回归',
    'doctor02': '年轻医生 02', 'doctor03': '年轻医生 03', 'MOCK100': '现有视频接口的固定满分占位输出'}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def runtime_environment():
    import cv2
    return {'python': platform.python_version(), 'machine': platform.machine(),
            'opencv': cv2.__version__,
            'packages': {name: version(name) for name in ('numpy', 'scipy', 'scikit-learn', 'mediapipe', 'torch', 'torchvision', 'pillow')}}


def check_environment(output, environment):
    path = output / 'environment.json'
    if path.is_file():
        if json.loads(path.read_text()) != environment:
            raise RuntimeError('Runtime changed; use a new output directory instead of mixing cached legacy results')
    else:
        if (output / 'legacy-cache').exists() and any((output / 'legacy-cache').iterdir()):
            raise RuntimeError('Unversioned legacy cache; use a new output directory')
        write_json(path, environment)


def run_legacy(job, cache, timeout, retry=False):
    key = hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()
    path = cache / f'{key}.json'
    if path.is_file():
        saved = json.loads(path.read_text())
        if not retry or saved['status'] == 'ok':
            return {**saved, 'cache_hit': True}
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='fp-benchmark-job-') as temporary:
        payload = Path(temporary) / 'job.json'
        # Parent owns the whole temporary directory, including decoded JPEGs.
        # It cleans up after success, a crashed worker, or a timeout.
        payload.write_text(json.dumps({**job, 'temp_root': temporary}))
        started = time.perf_counter()
        try:
            run = subprocess.run([sys.executable, str(HERE / 'legacy_worker.py'), str(payload)],
                                 capture_output=True, text=True, timeout=timeout)
            result = json.loads(run.stdout) if run.returncode == 0 else {
                'status': 'failed', 'error_type': 'WorkerProcessError', 'elapsed_seconds': time.perf_counter() - started}
        except subprocess.TimeoutExpired:
            result = {'status': 'failed', 'error_type': 'TimeoutExpired', 'elapsed_seconds': time.perf_counter() - started}
        result.update(cache_key=key, cache_hit=False, process_wall_seconds=time.perf_counter() - started)
        write_json(path, result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Existing first-validation artifact directory')
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, default=ROOT / 'docs/experiments/unified-evaluation.md')
    parser.add_argument('--date', default='2026-11-07')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--timeout', type=int, default=300)
    parser.add_argument('--retry-failures', action='store_true')
    parser.add_argument('--reuse-reproduced-training', action='store_true')
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    environment = runtime_environment()
    check_environment(args.output, environment)
    source = args.source.resolve()
    manifest, config, quality, eligible, geometry, appearance = cached_features(source)
    prior = json.loads((source / 'metrics.json').read_text())
    audit = json.loads((source / 'data-audit.json').read_text())
    if len(manifest) != len(eligible) or not np.array_equal(eligible, np.arange(len(manifest))):
        raise RuntimeError('This protocol expects the original complete OOF cohort; design a new protocol for other inputs')
    doctors = {name: read_doctor(args.data_root / f'{name}.csv', name == 'doctor01') for name in ('doctor01', 'doctor02', 'doctor03')}
    csv_hashes = {name: sha256_file(args.data_root / f'{name}.csv') for name in doctors}
    if csv_hashes != audit['source_csv_sha256']:
        raise RuntimeError('Doctor source CSVs changed; do not combine with historical OOF predictions')
    for row in manifest:
        stem = Path(row['video_path']).stem
        if any(not np.array_equal(doctors[name][stem]['labels'], row['labels'][name]) for name in doctors):
            raise RuntimeError('Manifest labels differ from current source CSVs')
    truth = np.asarray([r['labels']['doctor01'] for r in manifest])
    folds = np.asarray([r['outer_fold'] for r in manifest])
    groups = np.asarray([r['video_sha256'] for r in manifest])
    available = []
    for row in manifest:
        path = Path(row['video_path'])
        exists = path.is_file()
        if exists and sha256_file(path) != row['video_sha256']:
            raise RuntimeError('An accessible source video changed; refusing stale feature reuse')
        available.append(exists)
    available = np.asarray(available)
    if not available.any():
        raise RuntimeError('No original videos accessible')
    legacy_files = {name: sha256_file(LEGACY / name) for name in ('Detect.py', 'Recognize.py', 'face_landmarker.task')}
    training_code = {p.name: sha256_file(p) for p in SHARED.glob('*.py')}
    cache_hashes = {p.name: sha256_file(p) for p in sorted((source / 'features').iterdir()) if p.suffix in ('.json', '.npz')}
    protocol = {'version': 'unified-sunnybrook-evaluation-v1', 'date': args.date, 'seed': prior['seed'],
                'target_count': len(manifest), 'accessible_count': int(available.sum()),
                'accessible_indices': np.flatnonzero(available).tolist(),
                'missing_ids': [r['video_id'] for r, ok in zip(manifest, available) if not ok],
                'ground_truth': 'doctor01.comment:13 subscores, formula total; doctor02/03 never used for training',
                'unit': 'video assessment; historical identical-video SHA groups isolated; repeated patients may cross folds',
                'split': 'Frozen historical 5 outer / 3 inner folds and seed, without rerandomization',
                'numeric_comparison': '13 finite integer outputs; raw values retained even outside legal bounds',
                'valid_output': 'All 13 items legal AND raw formula total in [0,100]; failures never become normal scores',
                'primary_comparison': 'Same accessible input set; common numeric-output subset across all methods; pairwise intersections also reported',
                'bootstrap': '5000 paired video draws; descriptive only, excludes retraining uncertainty and patient dependence',
                'error_bands': '[5,10,15] points are descriptive engineering bands, not clinical equivalence thresholds',
                'legacy_window': 'Original IMAGE rule scorer, OpenCV seek to existing research rest midpoint and 5 peaks; diagnostic, not oracle manual boundaries',
                'legacy_timeout_seconds': args.timeout, 'workers': args.workers, 'environment': environment,
                'csv_sha256': csv_hashes, 'legacy_sha256': legacy_files,
                'source_artifact_sha256': {name: sha256_file(source / name) for name in ('manifest.json', 'extractor-config.json', 'out-of-fold.npz', 'metrics.json')},
                'feature_cache_sha256': cache_hashes, 'training_code_sha256': training_code,
                'algorithm_versions': dict(VERSION_FOLDERS),
                'algorithm_code_sha256': {str(p.relative_to(ALGORITHMS)): sha256_file(p)
                    for p in sorted(ALGORITHMS.rglob('*')) if p.suffix in ('.py', '.json') and '__pycache__' not in p.parts},
                'benchmark_code_sha256': {p.name: sha256_file(p) for p in HERE.glob('*.py')}}
    # Freeze the input cohort and policy before running any method.
    write_json(args.output / 'protocol.json', protocol)
    with np.load(source / 'out-of-fold.npz', allow_pickle=False) as saved:
        historical = {name: saved[name].copy() for name in ('SB0', 'SB1', 'SB2', 'SB3')}
        if not np.array_equal(saved['truth'], truth) or not np.array_equal(saved['folds'], folds):
            raise RuntimeError('Historical labels or folds are inconsistent')
    reproduction = args.output / 'reproduced-oof.npz'
    if args.reuse_reproduced_training and reproduction.is_file():
        with np.load(reproduction, allow_pickle=False) as saved:
            if not np.array_equal(saved['truth'], truth) or not np.array_equal(saved['folds'], folds):
                raise RuntimeError('Reproduced output belongs to another experiment')
            learned = {name: saved[name].copy() for name in historical}
    else:
        with threadpool_limits(limits=4):
            learned, logs = nested_validation(geometry, appearance, truth, folds, groups, prior['seed'],
                lambda log: print(f"REPRODUCE fold={log['fold']} seconds={log['elapsed_seconds']:.1f}", flush=True))
        write_json(args.output / 'training-folds.json', logs)
        np.savez_compressed(reproduction, truth=truth, folds=folds, **learned)
    identical = {name: bool(np.array_equal(learned[name], historical[name])) for name in historical}
    if not all(identical.values()):
        raise RuntimeError('Training reproduction differs from frozen results; do not silently mix experiments')
    predictions = {name: np.full(truth.shape, np.nan) for name in METHODS}
    predictions.update(learned)
    for name in ('doctor02', 'doctor03'):
        predictions[name] = np.asarray([r['labels'][name] for r in manifest])
    predictions['MOCK100'] = np.tile([0, 0, 0] + [5] * 5 + [0] * 5, (len(truth), 1))
    statuses = {name: [] for name in METHODS}
    for name in METHODS[2:]:
        statuses[name] = [{'video_id': row['video_id'], 'status': 'ok', 'origin': 'reproduced_oof' if name.startswith('SB') else 'independent_or_fixed_control'} for row in manifest]
    for name in METHODS[:2]:
        statuses[name] = [{'video_id': row['video_id'], 'status': 'source_missing' if not ok else 'not_run'} for row, ok in zip(manifest, available)]
    evidence = {q['video_id']: q['evidence'] for q in quality}
    jobs = []
    for index in np.flatnonzero(available):
        row = manifest[index]
        e = evidence[row['video_id']]
        timestamps = [e[0]['times'][1]] + [phase['peak'] for phase in e[1:]]
        for name in METHODS[:2]:
            jobs.append((int(index), name, {'method': name, 'algorithm': str(LEGACY),
                'video_path': row['video_path'], 'video_sha256': row['video_sha256'],
                'timestamps': timestamps if name == 'LEGACY_WINDOW' else [],
                'legacy_sha256': legacy_files, 'worker_sha256': sha256_file(HERE / 'legacy_worker.py')}))
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(run_legacy, job, args.output / 'legacy-cache', args.timeout, args.retry_failures): (index, name) for index, name, job in jobs}
        for done, future in enumerate(as_completed(futures), 1):
            index, name = futures[future]
            result = future.result()
            statuses[name][index] = {'video_id': manifest[index]['video_id'], **result}
            if result['status'] == 'ok':
                predictions[name][index] = result['prediction']
            write_json(args.output / 'status.json', statuses)
            print(f"LEGACY {done}/{len(jobs)} method={name} status={result['status']} seconds={result.get('elapsed_seconds', 0):.1f}", flush=True)
    selected = {name: value[available] for name, value in predictions.items()}
    reference = truth[available]
    numeric = {name: numeric_mask(value) for name, value in selected.items()}
    common = np.logical_and.reduce(list(numeric.values()))
    stats = {}
    for name, value in selected.items():
        finite = numeric[name]
        valid = legal_mask(value) & np.isfinite(totals(value)) & (totals(value) >= 0) & (totals(value) <= 100)
        records = [r for r, ok in zip(statuses[name], available) if ok]
        times = [r['elapsed_seconds'] for r in records if r.get('elapsed_seconds') is not None]
        stats[name] = {'description': DESCRIPTIONS[name], 'input_count': len(reference),
            'numeric_count': int(finite.sum()), 'numeric_coverage': float(finite.mean()),
            'valid_count': int(valid.sum()), 'valid_coverage': float(valid.mean()),
            'status_counts': dict(Counter(r['status'] for r in records)),
            'failure_type_counts': dict(Counter(r['error_type'] for r in records if 'error_type' in r)),
            'own_numeric_metrics': evaluate(reference[finite], value[finite], prior['seed']),
            'common_numeric_metrics': evaluate(reference[common], value[common], prior['seed']),
            'valid_only_metrics': evaluate(reference[valid], value[valid], prior['seed']),
            'runtime_seconds': {'median': float(np.median(times)), 'p95': float(np.percentile(times, 95)), 'count': len(times)} if times else None}
    pairwise = {f'{first}__minus__{second}': paired(reference, selected[first], selected[second], prior['seed'])
                for i, first in enumerate(METHODS) for second in METHODS[i + 1:]}
    results = {'protocol': protocol, 'historical_predictions_identical': identical,
               'common_numeric_count': int(common.sum()), 'methods': stats, 'paired_comparisons': pairwise,
               'historical_68_metrics': {name: evaluate(truth, predictions[name], prior['seed']) for name in METHODS[2:]},
               'common_numeric_ids': [row['video_id'] for row, flag in zip([r for r, ok in zip(manifest, available) if ok], common) if flag]}
    write_json(args.output / 'results.json', results)
    np.savez_compressed(args.output / 'predictions.npz', truth=truth, available=available, common_numeric=common, folds=folds, **predictions)
    with (args.output / 'predictions.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['video_id', 'outer_fold', 'source_accessible', 'method', 'numeric_output', 'valid_output', 'ground_truth_total', 'raw_total', *ITEMS])
        for i, row in enumerate(manifest):
            for name in METHODS:
                value = predictions[name][i:i + 1]
                finite = bool(numeric_mask(value)[0])
                total = int(totals(value)[0]) if finite else None
                valid = finite and bool(legal_mask(value)[0]) and 0 <= total <= 100
                writer.writerow([row['video_id'], row['outer_fold'], int(available[i]), name, int(finite), int(valid), int(totals(truth[i])), total, *[int(v) if finite else '' for v in value[0]]])
    from report import write_report
    write_report(results, args.report, args.output)
    print('COMPLETE: common numeric cohort', int(common.sum()), 'of', len(reference), flush=True)
    for name, entry in stats.items():
        metric = entry['own_numeric_metrics']
        print(name, 'numeric', entry['numeric_count'], 'valid', entry['valid_count'], 'MAE', round(metric['total_mae'], 4) if metric else None, flush=True)


if __name__ == '__main__':
    main()
