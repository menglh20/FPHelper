"""Compatibility import path for source checkouts using PYTHONPATH=src."""
from pathlib import Path

__version__ = '0.1.0'
_algorithm = Path(__file__).resolve().parents[2] / 'algorithm'
__path__.insert(0, str(_algorithm / 'shared' / 'fphelper'))
__path__.append(str(_algorithm))
