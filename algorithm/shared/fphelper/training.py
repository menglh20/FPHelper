"""Nested validation: scalers, PCA and ordinal heads fit on training rows only."""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedGroupKFold

from .data import LOW, HIGH, PHASE, ITEMS, totals
from .estimation import fit_heads, predict_heads
from .versions import get_model

REGULARIZATION = (0.01, 0.1, 1.0)


def normalized_error(y, prediction):
    errors = np.mean(np.abs(y - prediction), axis=0) / (HIGH - LOW)
    return float(np.mean([errors[:3].mean(), errors[3:8].mean(), errors[8:].mean()]))


@dataclass
class FeatureProjector:
    dimensions: int = 16
    seed: int = 20261012

    def fit(self, features):
        self.scalers, self.pcas = [], []
        for phase in range(6):
            scaler = StandardScaler().fit(features[:, phase])
            standardized = scaler.transform(features[:, phase])
            components = min(self.dimensions, len(features) - 1, standardized.shape[1])
            pca = PCA(n_components=components, svd_solver="randomized", random_state=self.seed,
                      iterated_power=3).fit(standardized)
            self.scalers.append(scaler)
            self.pcas.append(pca)
        return self

    def transform(self, features):
        return [pca.transform(scaler.transform(features[:, phase])).astype(np.float64)
                for phase, (scaler, pca) in enumerate(zip(self.scalers, self.pcas))]


def inputs_for(family, geometry, appearance):
    """Compatibility entry point; implementations live in version folders."""
    return get_model(family).inputs_for(geometry, appearance)


def prepared_split(geometry, appearance, train, test, seed):
    gp = FeatureProjector(seed=seed).fit(geometry[train])
    ap = FeatureProjector(seed=seed).fit(appearance[train])
    return gp, ap, gp.transform(geometry[train]), ap.transform(appearance[train]), gp.transform(geometry[test]), ap.transform(appearance[test])


def inner_splits(labels, groups, seed):
    strata = np.digitize(totals(labels), [20, 40, 60])
    if len(np.unique(groups)) < len(groups):
        splitter = StratifiedGroupKFold(3, shuffle=True, random_state=seed)
        splits = list(splitter.split(np.zeros(len(labels)), strata, groups))
    else:
        counts = np.bincount(strata)
        nonzero = counts[counts > 0]
        splitter = (StratifiedKFold(3, shuffle=True, random_state=seed) if min(nonzero) >= 3
                    else KFold(3, shuffle=True, random_state=seed))
        splits = list(splitter.split(np.zeros(len(labels)), strata))
    for train, test in splits:
        if set(groups[train]) & set(groups[test]):
            raise ValueError("Duplicate video crossed an inner validation split")
    return splits


def tune_families(geometry, appearance, labels, groups, seed):
    candidates = {(family, reg): np.full(labels.shape, np.nan) for family in ("SB1", "SB2", "SB3") for reg in REGULARIZATION}
    for inner, (train, test) in enumerate(inner_splits(labels, groups, seed)):
        _, _, gtrain, atrain, gtest, atest = prepared_split(geometry, appearance, train, test, seed + inner)
        for family in ("SB1", "SB2", "SB3"):
            for regularization in REGULARIZATION:
                version = get_model(family)
                heads = version.fit(gtrain, atrain, labels[train], regularization)
                candidates[family, regularization][test] = version.predict(heads, gtest, atest)
    scores = []
    choices = {}
    for (family, regularization), prediction in candidates.items():
        if not np.isfinite(prediction).all():
            raise RuntimeError("Incomplete inner validation predictions")
        scores.append({"family": family, "regularization": regularization,
                       "normalized_subscore_mae": normalized_error(labels, prediction),
                       "total_mae": float(np.mean(np.abs(totals(labels) - totals(prediction))))})
    for family in ("SB1", "SB2", "SB3"):
        selected = min((s for s in scores if s["family"] == family),
                       key=lambda s: (round(s["normalized_subscore_mae"], 6), round(s["total_mae"], 6), -s["regularization"]))
        choices[family] = selected["regularization"]
    return choices, scores


@dataclass
class SunnybrookModel:
    geometry_projector: FeatureProjector
    appearance_projector: FeatureProjector
    heads: list
    family: str = "SB3"

    def predict(self, geometry, appearance):
        return get_model(self.family).predict(self.heads, self.geometry_projector.transform(geometry),
                                              self.appearance_projector.transform(appearance))


def nested_validation(geometry, appearance, labels, folds, groups, seed, progress=None):
    predictions = {family: np.full(labels.shape, np.nan) for family in ("SB0", "SB1", "SB2", "SB3")}
    logs = []
    for fold in sorted(np.unique(folds)):
        started = time.perf_counter()
        train, test = np.where(folds != fold)[0], np.where(folds == fold)[0]
        if set(groups[train]) & set(groups[test]):
            raise ValueError("Duplicate video crossed an outer validation split")
        choices, inner_scores = tune_families(geometry[train], appearance[train], labels[train], groups[train], seed + int(fold))
        gp, ap, gtrain, atrain, gtest, atest = prepared_split(geometry, appearance, train, test, seed + int(fold))
        baseline = get_model("SB0")
        predictions["SB0"][test] = baseline.predict(baseline.fit(labels[train]), len(test))
        training_log = {}
        for family in ("SB1", "SB2", "SB3"):
            version = get_model(family)
            heads = version.fit(gtrain, atrain, labels[train], choices[family])
            predictions[family][test] = version.predict(heads, gtest, atest)
            training_log[family] = {"nonconverged_heads": sum(not h.converged for h in heads),
                                    "max_iterations": max(h.iterations for h in heads)}
        support = {name: {str(score): int((labels[train, j] == score).sum()) for score in range(LOW[j], HIGH[j] + 1)} for j, name in enumerate(ITEMS)}
        unsupported_test = {name: int(sum(score not in np.unique(labels[train, j]) for score in labels[test, j])) for j, name in enumerate(ITEMS)}
        entry = {"fold": int(fold), "train_count": len(train), "test_count": len(test),
                 "selected_regularization": choices, "inner_scores": inner_scores,
                 "train_support": support, "test_labels_not_in_training": unsupported_test,
                 "optimizer": training_log, "elapsed_seconds": time.perf_counter() - started,
                 "test_indices": test.tolist()}
        logs.append(entry)
        if progress:
            progress(entry)
    if any(not np.isfinite(p).all() for p in predictions.values()):
        raise RuntimeError("Missing out-of-fold predictions")
    return predictions, logs


def fit_final_model(geometry, appearance, labels, groups, seed):
    choices, scores = tune_families(geometry, appearance, labels, groups, seed)
    gp = FeatureProjector(seed=seed).fit(geometry)
    ap = FeatureProjector(seed=seed).fit(appearance)
    heads = get_model("SB3").fit(gp.transform(geometry), ap.transform(appearance), labels, choices["SB3"])
    return SunnybrookModel(gp, ap, heads), {"selected_regularization": choices["SB3"], "inner_scores": scores,
                                          "nonconverged_heads": sum(not h.converged for h in heads)}
