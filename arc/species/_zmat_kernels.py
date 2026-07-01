"""
ctypes loader for _zmat_c_kernels.so.

Set ARC_NO_C_KERNELS=1 to force the pure-Python fallback (useful for timing
comparisons or when the .so is not available).

Exposed attributes
------------------
lib : ctypes.CDLL | None
    The loaded library, or None if unavailable / disabled.
available : bool
    True iff the C kernels are loaded and ready.
"""

import ctypes
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_SO = os.path.join(_HERE, '_zmat_c_kernels.so')

lib: ctypes.CDLL | None = None
available: bool = False

if not os.environ.get('ARC_NO_C_KERNELS'):
    try:
        _lib = ctypes.CDLL(_SO)

        _d = ctypes.c_double
        _dp = ctypes.POINTER(ctypes.c_double)

        # distance (f32-precision to match np.asarray(..., float32) semantics)
        _lib.zmat_r_f32.restype = _d
        _lib.zmat_r_f32.argtypes = [_d] * 6

        # bond angle → degrees
        _lib.zmat_a_f32.restype = _d
        _lib.zmat_a_f32.argtypes = [_d] * 9

        # dihedral angle → degrees (0–360)
        _lib.zmat_d_f32.restype = _d
        _lib.zmat_d_f32.argtypes = [_d] * 12

        # SN-NeRF atom placement (angles in degrees)
        _lib.zmat_nerf.restype = None
        _lib.zmat_nerf.argtypes = [_d] * 12 + [_dp, _dp, _dp]

        lib = _lib
        available = True
    except Exception:
        pass  # .so absent or mis-compiled; available stays False → pure-Python fallback used
