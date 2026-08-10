"""
ctypes loader for the compiled zmat geometry kernels (see ``_zmat_c_kernels.c``).

The shared library is built by ``make compile``; it is deliberately not shipped
pre-built, since a binary compiled for one CPU/OS is not portable to another.
When it is absent, the pure-Python/numpy implementations in ``vectors.py`` and
``zmat.py`` are used instead, so ARC stays fully functional -- only slower.

Set ``ARC_NO_C_KERNELS=1`` to force the pure-Python fallback (useful for timing
comparisons, or to rule the kernels out while debugging).

Exposed attributes
------------------
lib : ctypes.CDLL | None
    The loaded library, or None if unavailable / disabled.
available : bool
    True iff the C kernels are loaded and ready.
load_error : Exception | None
    Why loading failed, or None if it succeeded (or was disabled via the env var).
    Inspect this when the kernels are unexpectedly unavailable.
"""

import ctypes
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

if sys.platform == 'darwin':
    _CANDIDATES = ('_zmat_c_kernels.dylib', '_zmat_c_kernels.so')
elif sys.platform == 'win32':
    _CANDIDATES = ('_zmat_c_kernels.dll',)
else:
    _CANDIDATES = ('_zmat_c_kernels.so',)

lib: ctypes.CDLL | None = None
available: bool = False
load_error: Exception | None = None


def _bind(so_path: str) -> ctypes.CDLL:
    """Load the shared library at ``so_path`` and declare each kernel's signature."""
    loaded = ctypes.CDLL(so_path)
    _d = ctypes.c_double

    # bond angle → degrees (full float64, matching calculate_angle's numpy path)
    loaded.zmat_a.restype = _d
    loaded.zmat_a.argtypes = [_d] * 9

    # distance (f32-precision to match np.asarray(..., float32) semantics)
    loaded.zmat_r_f32.restype = _d
    loaded.zmat_r_f32.argtypes = [_d] * 6

    # dihedral angle → degrees (0–360)
    loaded.zmat_d_f32.restype = _d
    loaded.zmat_d_f32.argtypes = [_d] * 12

    return loaded


if not os.environ.get('ARC_NO_C_KERNELS'):
    for _name in _CANDIDATES:
        _path = os.path.join(_HERE, _name)
        if not os.path.isfile(_path):
            continue
        try:
            lib = _bind(_path)
            available = True
            load_error = None
            break
        except Exception as e:
            # Record why, rather than failing silently: the numpy fallbacks are correct,
            # so this is never fatal, but an unexpected fallback is hard to diagnose
            # otherwise (a mis-compiled or ABI-mismatched .so looks just like a missing one).
            load_error = e
    else:
        if load_error is None:
            load_error = FileNotFoundError(
                f'None of {_CANDIDATES} were found in {_HERE}. Run `make compile` to build them.')
