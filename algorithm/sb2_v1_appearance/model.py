"""SB2 v1: projected frozen ResNet18 appearance features, 13 ordinal heads."""
from fphelper.estimation import fit_heads, predict_heads

VERSION = 'sb2_v1_appearance'
FAMILY = 'SB2'


def inputs_for(geometry, appearance):
    return appearance


def fit(geometry, appearance, labels, regularization):
    return fit_heads(inputs_for(geometry, appearance), labels, regularization)


def predict(heads, geometry, appearance):
    return predict_heads(heads, inputs_for(geometry, appearance))
