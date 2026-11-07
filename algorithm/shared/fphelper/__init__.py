"""Local, offline Sunnybrook research pipeline and versioned algorithms."""
from pathlib import Path

__version__ = '0.1.0'
# Source packages live in sibling version folders. In a built distribution they
# are normal fphelper subpackages and need no path extension.
_algorithm = Path(__file__).resolve().parents[2]
if (_algorithm / 'sb1_v1_geometry').is_dir():
    __path__.append(str(_algorithm))
