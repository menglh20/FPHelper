"""SB3 v1: concatenate projected geometry and appearance, 13 ordinal heads."""
import numpy as np

from fphelper.estimation import fit_heads, predict_heads

VERSION = 'sb3_v1_fusion'
FAMILY = 'SB3'


def inputs_for(geometry, appearance):
    return [np.c_[g, a] for g, a in zip(geometry, appearance)]


def fit(geometry, appearance, labels, regularization):
    return fit_heads(inputs_for(geometry, appearance), labels, regularization)


def predict(heads, geometry, appearance):
    return predict_heads(heads, inputs_for(geometry, appearance))
