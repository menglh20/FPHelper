"""One evaluation contract for rules, learned ordinal heads and doctors."""
from __future__ import annotations

import numpy as np
from fphelper.data import ITEMS, LOW, HIGH, totals
from fphelper.metrics import absolute_agreement_icc


def mean_interval(values, seed=20261012, draws=5000):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(values), size=(draws, len(values)))
    return np.percentile(values[samples].mean(axis=1), [2.5, 97.5]).tolist()


def numeric_mask(prediction):
    value = np.asarray(prediction, dtype=float)
    if value.ndim != 2 or value.shape[1] != 13:
        raise ValueError('Predictions must be an N by 13 matrix')
    return np.isfinite(value).all(axis=1)


def legal_mask(prediction):
    value = np.asarray(prediction, dtype=float)
    return numeric_mask(value) & ((value >= LOW) & (value <= HIGH) & (value == np.floor(value))).all(axis=1)


def qwk(reference, prediction, low, high):
    # Use the whole integer scale, including illegal predictions. Dropping zero
    # voluntary grades from a 1..5 confusion matrix would hide legacy errors.
    lower, upper = min(low, int(prediction.min())), max(high, int(prediction.max()))
    levels = np.arange(lower, upper + 1)
    confusion = np.array([[(reference == a)[prediction == b].sum() for b in levels] for a in levels])
    weights = (levels[:, None] - levels[None, :]) ** 2
    expected = np.outer(confusion.sum(axis=1), confusion.sum(axis=0)) / len(reference)
    denominator = float((weights * expected).sum())
    return (float(1 - (weights * confusion).sum() / denominator) if denominator > 0 else None), levels.tolist(), confusion.tolist()


def binary_metrics(reference, prediction):
    truth, pred = np.asarray(reference) > 0, np.asarray(prediction) > 0
    tp, fn = int((truth & pred).sum()), int((truth & ~pred).sum())
    fp, tn = int((~truth & pred).sum()), int((~truth & ~pred).sum())
    return {'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn, 'positive_support': tp + fn,
            'recall': tp / (tp + fn) if tp + fn else None,
            'precision': tp / (tp + fp) if tp + fp else None,
            'false_positive_rate': fp / (fp + tn) if fp + tn else None,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}


def evaluate(reference, prediction, seed=20261012):
    reference, prediction = np.asarray(reference), np.asarray(prediction)
    if reference.shape != prediction.shape or reference.shape[1:] != (13,):
        raise ValueError('Reference and predictions must have equal N by 13 shapes')
    if not len(reference):
        return None
    if not numeric_mask(prediction).all() or not np.equal(prediction, np.floor(prediction)).all():
        raise ValueError('Pass only complete finite integer predictions; account for failures separately')
    if not legal_mask(reference).all():
        raise ValueError('Invalid ground truth')
    truth_total, predicted_total = totals(reference), totals(prediction)
    error = predicted_total - truth_total
    absolute = np.abs(error)
    sd = float(error.std(ddof=1)) if len(error) > 1 else 0.0
    item_error = np.abs(prediction - reference)
    normalized = item_error / (HIGH - LOW)
    groups = (slice(0, 3), slice(3, 8), slice(8, 13))
    normalized_per_video = np.column_stack([normalized[:, group].mean(axis=1) for group in groups]).mean(axis=1)
    subscores = {}
    for j, name in enumerate(ITEMS):
        y, p = reference[:, j], prediction[:, j]
        kappa, levels, matrix = qwk(y, p, int(LOW[j]), int(HIGH[j]))
        classes = np.unique(y)
        recalls = [float((p[y == c] == c).mean()) for c in classes]
        f1s = [2 * int(((y == c) & (p == c)).sum()) / (int((y == c).sum()) + int((p == c).sum())) for c in classes]
        entry = {'mae': float(item_error[:, j].mean()), 'normalized_mae': float(normalized[:, j].mean()),
                 'exact_agreement': float((p == y).mean()), 'within_one_grade': float((np.abs(p - y) <= 1).mean()),
                 'quadratic_weighted_kappa': kappa, 'balanced_recall_observed_truth_classes': float(np.mean(recalls)),
                 'macro_f1_observed_truth_classes': float(np.mean(f1s)),
                 'confusion_levels': levels, 'confusion_matrix': matrix,
                 'reference_support': {str(int(c)): int((y == c).sum()) for c in classes},
                 'illegal_predictions': int(((p < LOW[j]) | (p > HIGH[j])).sum())}
        if j >= 8:
            entry['nonzero'] = binary_metrics(y, p)
        subscores[name] = entry
    nonzero = [subscores[name]['nonzero'] for name in ITEMS[8:]]
    def macro(key):
        values = [r[key] for r in nonzero if r[key] is not None]
        return float(np.mean(values)) if values else None
    strata = {}
    bins = np.digitize(truth_total, [20, 40, 60, 80], right=True)
    for index, name in enumerate(('0–20', '21–40', '41–60', '61–80', '81–100')):
        mask = bins == index
        strata[name] = {'count': int(mask.sum()), 'mae': float(absolute[mask].mean()) if mask.any() else None}
    return {'count': len(reference), 'total_mae': float(absolute.mean()),
            'mae_ci95_video_bootstrap_descriptive': mean_interval(absolute, seed),
            'total_rmse': float(np.sqrt(np.square(error).mean())), 'bias': float(error.mean()),
            'icc_absolute_single': absolute_agreement_icc(truth_total, predicted_total),
            'bland_altman_limits': [float(error.mean() - 1.96 * sd), float(error.mean() + 1.96 * sd)],
            'within_5': float((absolute <= 5).mean()), 'within_10': float((absolute <= 10).mean()),
            'within_15': float((absolute <= 15).mean()),
            'equal_dimension_normalized_mae': float(normalized_per_video.mean()),
            'dimension_normalized_mae': {name: float(normalized[:, group].mean()) for name, group in zip(('rest', 'voluntary', 'synkinesis'), groups)},
            'dimension_weighted_score_mae': {'rest': float(np.abs(5 * (prediction[:, :3] - reference[:, :3]).sum(axis=1)).mean()),
                                             'voluntary': float(np.abs(4 * (prediction[:, 3:8] - reference[:, 3:8]).sum(axis=1)).mean()),
                                             'synkinesis': float(np.abs((prediction[:, 8:] - reference[:, 8:]).sum(axis=1)).mean())},
            'synkinesis_macro_recall': macro('recall'), 'synkinesis_macro_fpr': macro('false_positive_rate'),
            'synkinesis_macro_f1': macro('f1'), 'truth_total_strata': strata,
            'invalid_item_rows': int((~legal_mask(prediction)).sum()),
            'invalid_raw_total_rows': int(((predicted_total < 0) | (predicted_total > 100)).sum()),
            'subscores': subscores}


def paired(reference, first, second, seed=20261012):
    mask = numeric_mask(first) & numeric_mask(second)
    if not mask.any():
        return {'count': 0, 'mae_difference': None, 'ci95_video_bootstrap_descriptive': None}
    y, a, b = totals(reference[mask]), totals(first[mask]), totals(second[mask])
    difference = np.abs(a - y) - np.abs(b - y)
    return {'count': int(mask.sum()), 'first_mae': float(np.abs(a - y).mean()),
            'second_mae': float(np.abs(b - y).mean()), 'mae_difference': float(difference.mean()),
            'ci95_video_bootstrap_descriptive': mean_interval(difference, seed),
            'first_smaller_error': int((difference < 0).sum()), 'equal_error': int((difference == 0).sum()),
            'second_smaller_error': int((difference > 0).sum())}
