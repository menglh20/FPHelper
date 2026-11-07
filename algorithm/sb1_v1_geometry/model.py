"""SB1 v1: projected geometry and blendshape statistics, 13 ordinal heads."""
from fphelper.estimation import fit_heads, predict_heads

VERSION = 'sb1_v1_geometry'
FAMILY = 'SB1'


def inputs_for(geometry, appearance):
    return geometry


def fit(geometry, appearance, labels, regularization):
    return fit_heads(inputs_for(geometry, appearance), labels, regularization)


def predict(heads, geometry, appearance):
    return predict_heads(heads, inputs_for(geometry, appearance))
