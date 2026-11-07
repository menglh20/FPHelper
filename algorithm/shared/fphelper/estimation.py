"""Shared ordinal estimation; version folders select their own feature inputs."""
import numpy as np

from .data import LOW, HIGH, PHASE, ITEMS
from .ordinal import OrderedLogit


def fit_heads(inputs, labels, regularization):
    return [OrderedLogit(LOW[j], HIGH[j], regularization).fit(inputs[PHASE[j]], labels[:, j])
            for j in range(len(ITEMS))]


def predict_heads(heads, inputs):
    return np.column_stack([head.predict(inputs[PHASE[j]]) for j, head in enumerate(heads)])
