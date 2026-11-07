"""SB0 v1: fit one modal grade per item on training rows only."""
import numpy as np

from fphelper.data import LOW, HIGH, ITEMS

VERSION = 'sb0_v1_baseline'
FAMILY = 'SB0'


def fit(labels):
    labels = np.asarray(labels)
    if labels.ndim != 2 or labels.shape[1] != len(ITEMS) or not len(labels):
        raise ValueError('Expected nonempty N by 13 training labels')
    return np.asarray([int(LOW[j] + np.argmax(np.bincount(labels[:, j] - LOW[j],
                       minlength=HIGH[j] - LOW[j] + 1))) for j in range(len(ITEMS))])


def predict(grades, count):
    return np.tile(grades, (count, 1))
