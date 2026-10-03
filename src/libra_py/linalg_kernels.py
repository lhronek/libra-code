"""
numpy-facing entry point to libra's LAPACK eigensolver. The array crosses into MATRIX / CMATRIX through the bulk
transfer (one memcpy each way), the work runs in LAPACK (dsyevd / zheevd) through liblinalg, and numpy arrays come
back. The name mirrors numpy's. A real input stays real (MATRIX), a complex input promotes to CMATRIX. For
NUMPY-RESIDENT operands numpy's own eigvalsh is faster (the transfer and the intermediate matrices cost 1.2-9x at
every size measured, 2026-09-06) - this entry point serves the flows whose data must run on libra's kernels. A build
without LAPACK (have_lapack() False) takes numpy's eigh under the same contract.
"""
import numpy as np
from liblibra_core import MATRIX, CMATRIX, have_lapack, lapack_syevd, lapack_heevd
from libra_py import data_conv


def _hermitian_check(A):
    """The relative Hermiticity test the kernels make (1e-8), on the numpy side of the fallback"""
    scale = max(float(np.abs(A).max()), 1e-300)
    dev = float(np.abs(A - A.conj().T).max()) / scale
    if dev > 1e-8:
        raise RuntimeError(F"eigh: the matrix is not Hermitian (relative deviation {dev:.2e})")


def _to_libra(a, cplx):
    a = np.asarray(a)
    if a.ndim != 2:
        raise ValueError(F"a 2D array is needed, got shape {a.shape}")
    return data_conv.nparray2CMATRIX(a) if cplx else data_conv.nparray2MATRIX(a)


def _to_numpy(m, cplx):
    return data_conv.MATRIX2nparray(m, np.complex128 if cplx else np.float64)


def eigh(A):
    """Eigenvalues (ascending, real) and eigenvectors (columns) of a Hermitian matrix - dsyevd /
    zheevd. A non-Hermitian input (beyond 1e-8 relative) raises, the kernels read one triangle."""
    A = np.asarray(A)
    cplx = np.iscomplexobj(A)
    if not have_lapack():
        # a build without LAPACK: numpy's eigh under the same contract
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(F"a square 2D array is needed, got shape {A.shape}")
        _hermitian_check(A)
        w, v = np.linalg.eigh(A)
        return w.real.copy(), v
    Am = _to_libra(A, cplx)
    n = A.shape[0]
    if cplx:
        E, C = CMATRIX(n, n), CMATRIX(n, n)
        lapack_heevd(Am, E, C)
    else:
        E, C = MATRIX(n, n), MATRIX(n, n)
        lapack_syevd(Am, E, C)
    return np.diag(_to_numpy(E, cplx)).real.copy(), _to_numpy(C, cplx)
