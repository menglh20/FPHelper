"""Paired errors and absolute agreement, including failures and class coverage."""
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from .data import ITEMS, LOW, HIGH, totals
from .training import normalized_error


def absolute_agreement_icc(reference, prediction):
    value = np.column_stack([reference, prediction]).astype(float)
    n, k = value.shape
    if n < 2:
        return None
    row = value.mean(axis=1, keepdims=True)
    column = value.mean(axis=0, keepdims=True)
    grand = value.mean()
    ms_row = k * np.square(row - grand).sum() / (n - 1)
    ms_column = n * np.square(column - grand).sum() / (k - 1)
    ms_error = np.square(value - row - column + grand).sum() / ((n - 1) * (k - 1))
    denominator = ms_row + (k - 1) * ms_error + k * (ms_column - ms_error) / n
    return float((ms_row - ms_error) / denominator) if denominator > 1e-12 else None


def interval(values, seed=20261012, draws=5000):
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return np.percentile(means, [2.5, 97.5]).tolist()


def evaluate(reference, prediction, seed=20261012):
    truth, predicted = totals(reference), totals(prediction)
    error = predicted - truth
    sd = float(error.std(ddof=1)) if len(error) > 1 else 0.0
    subscores = {}
    for j, name in enumerate(ITEMS):
        allowed = np.arange(LOW[j], HIGH[j] + 1)
        qwk = None
        if len(np.unique(np.r_[reference[:, j], prediction[:, j]])) > 1:
            qwk = float(cohen_kappa_score(reference[:, j], prediction[:, j], labels=allowed, weights="quadratic"))
            if not np.isfinite(qwk):
                qwk = None
        entry = {"mae": float(np.abs(reference[:, j] - prediction[:, j]).mean()),
                 "exact_agreement": float((reference[:, j] == prediction[:, j]).mean()),
                 "quadratic_weighted_kappa": qwk,
                 "confusion_matrix": confusion_matrix(reference[:, j], prediction[:, j], labels=allowed).tolist(),
                 "reference_support": {str(c): int((reference[:, j] == c).sum()) for c in allowed}}
        if j >= 8:
            positive = reference[:, j] > 0
            entry["nonzero_recall"] = float((prediction[positive, j] > 0).mean()) if positive.any() else None
            entry["false_positive_rate"] = float((prediction[~positive, j] > 0).mean()) if (~positive).any() else None
        subscores[name] = entry
    return {"count": len(reference), "total_mae": float(np.abs(error).mean()), "mae_ci95_video_bootstrap": interval(np.abs(error), seed),
            "total_rmse": float(np.sqrt(np.square(error).mean())), "bias": float(error.mean()),
            "icc_absolute_single_mixed": absolute_agreement_icc(truth, predicted),
            "bland_altman_limits": [float(error.mean() - 1.96 * sd), float(error.mean() + 1.96 * sd)],
            "normalized_subscore_mae": normalized_error(reference, prediction),
            "invalid_raw_totals": int(((predicted < 0) | (predicted > 100)).sum()),
            "dimensions_mae": {"rest_penalty": float(np.abs(5 * (reference[:, :3] - prediction[:, :3]).sum(axis=1)).mean()),
                               "voluntary_score": float(np.abs(4 * (reference[:, 3:8] - prediction[:, 3:8]).sum(axis=1)).mean()),
                               "synkinesis_penalty": float(np.abs((reference[:, 8:] - prediction[:, 8:]).sum(axis=1)).mean())},
            "subscores": subscores}


def paired_comparison(reference, algorithm, doctor, seed=20261012):
    difference = np.abs(totals(algorithm) - totals(reference)) - np.abs(totals(doctor) - totals(reference))
    return {"count": len(difference), "mean_absolute_error_difference": float(difference.mean()),
            "ci95_video_bootstrap": interval(difference, seed),
            "algorithm_smaller_error": int((difference < 0).sum()),
            "equal_error": int((difference == 0).sum()), "doctor_smaller_error": int((difference > 0).sum())}
