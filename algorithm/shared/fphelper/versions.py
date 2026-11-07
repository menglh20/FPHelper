"""Explicit version registry. Add new folders instead of editing frozen versions."""
from importlib import import_module

VERSION_FOLDERS = {
    'SB0': 'sb0_v1_baseline',
    'SB1': 'sb1_v1_geometry',
    'SB2': 'sb2_v1_appearance',
    'SB3': 'sb3_v1_fusion',
}


def get_model(family):
    try:
        folder = VERSION_FOLDERS[family]
    except KeyError:
        raise ValueError(f'Unknown model family: {family}') from None
    return import_module(f'fphelper.{folder}.model')
