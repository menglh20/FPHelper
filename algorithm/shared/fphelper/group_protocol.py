"""Strict patient mapping and nested group folds for subsequent experiments.

Account names, timestamps and video hashes are not patient identities.
This module never invents patient IDs or replaces historical folds.
"""
import csv
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from .data import totals


def patient_groups(manifest, mapping_path):
    with open(mapping_path, newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['video_id', 'patient_group_id']:
            raise ValueError('Expected video_id,patient_group_id columns')
        rows = list(reader)
    ids = [r['video_id'] for r in rows]
    expected = [r['video_id'] for r in manifest]
    if len(set(ids)) != len(ids) or set(ids) != set(expected):
        raise ValueError('Mapping must cover every video exactly once, with no extra rows')
    mapping = {r['video_id']: r['patient_group_id'].strip() for r in rows}
    if any(not v for v in mapping.values()):
        raise ValueError('Patient identity mapping is incomplete; video-level fallback is prohibited')
    groups = np.array([mapping[k] for k in expected])
    by_hash = {}
    for record, group in zip(manifest, groups):
        previous = by_hash.setdefault(record['video_sha256'], group)
        if previous != group:
            raise ValueError('Identical video bytes are assigned to different patients')
    return groups


def group_splits(labels, groups, count, seed):
    groups = np.asarray(groups)
    if len(groups) != len(labels) or len(np.unique(groups)) < count:
        raise ValueError('Insufficient patient groups for requested fold count')
    strata = np.digitize(totals(labels), [20, 40, 60])
    splits = list(StratifiedGroupKFold(count, shuffle=True, random_state=seed).split(labels, strata, groups))
    for train, test in splits:
        if not len(train) or not len(test) or set(groups[train]) & set(groups[test]):
            raise ValueError('Invalid patient partition')
    return splits


def nested_group_protocol(labels, groups, seed=20261014):
    result = []
    for fold, (train, test) in enumerate(group_splits(labels, groups, 5, seed)):
        inner = group_splits(labels[train], groups[train], 3, seed + fold)
        result.append({'fold': fold, 'train_indices': train.tolist(), 'test_indices': test.tolist(),
                       'inner': [{'train_indices': train[a].tolist(), 'test_indices': train[b].tolist()}
                                 for a, b in inner]})
    return result
