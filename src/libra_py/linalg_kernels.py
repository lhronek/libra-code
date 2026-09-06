"""
numpy-facing entry points to libra's BLAS/LAPACK kernels. The arrays cross into MATRIX / CMATRIX through
the bulk transfer (one memcpy each way), the work runs in OpenBLAS (dgemm / zgemm through the MATRIX
products) and LAPACK (dsyevd / zheevd, dgesdd / zgesdd, dgetri / zgetri, the pivot-product determinant)
through liblinalg, and numpy arrays come back. The names mirror numpy's. Real inputs stay real (MATRIX), any
complex input promotes the operation to CMATRIX. For NUMPY-RESIDENT operands numpy's own @, eigvalsh, svd and
inv are faster (the transfer and the intermediate matrices cost 1.2-9x at every size measured, 2026-09-06) -
these entry points serve the flows whose data must run on libra's kernels.
"""
import numpy as np
from liblibra_core import MATRIX, CMATRIX, lapack_syevd, lapack_heevd, lapack_gesdd, lapack_getri, lapack_det
from libra_py import data_conv


def _to_libra(a, cplx):
    a = np.asarray(a)
    if a.ndim != 2:
        raise ValueError(F"a 2D array is needed, got shape {a.shape}")
    return data_conv.nparray2CMATRIX(a) if cplx else data_conv.nparray2MATRIX(a)


def _to_numpy(m, cplx):
    return data_conv.MATRIX2nparray(m, np.complex128 if cplx else np.float64)


def product(*arrays):
    """A B C ... left to right (numpy's multi_dot association is NOT used - the order of the
    arguments is the order of the products), MATRIX products for real arrays, CMATRIX products when
    any array is complex. Returns a numpy array."""
    if len(arrays) < 2:
        raise ValueError("product needs at least two arrays")
    arrays = [np.asarray(a) for a in arrays]
    cplx = any(np.iscomplexobj(a) for a in arrays)
    acc = _to_libra(arrays[0], cplx)
    for a in arrays[1:]:
        acc = acc * _to_libra(a, cplx)
    return _to_numpy(acc, cplx)


def eigh(A):
    """Eigenvalues (ascending, real) and eigenvectors (columns) of a Hermitian matrix - dsyevd /
    zheevd. A non-Hermitian input (beyond 1e-8 relative) raises, the kernels read one triangle."""
    A = np.asarray(A)
    cplx = np.iscomplexobj(A)
    Am = _to_libra(A, cplx)
    n = A.shape[0]
    if cplx:
        E, C = CMATRIX(n, n), CMATRIX(n, n)
        lapack_heevd(Am, E, C)
    else:
        E, C = MATRIX(n, n), MATRIX(n, n)
        lapack_syevd(Am, E, C)
    return np.diag(_to_numpy(E, cplx)).real.copy(), _to_numpy(C, cplx)


def singular_values(A):
    """The singular values, descending - dgesdd / zgesdd."""
    A = np.asarray(A)
    cplx = np.iscomplexobj(A)
    Am = _to_libra(A, cplx)
    m, n = A.shape
    make = CMATRIX if cplx else MATRIX
    U, S, V = make(m, m), make(m, n), make(n, n)
    lapack_gesdd(Am, U, S, V)
    return np.diag(_to_numpy(S, cplx))[:min(m, n)].real.copy()


def inv(A):
    """The inverse of a general matrix - dgetrf + dgetri / zgetrf + zgetri (an exactly singular matrix
    raises)."""
    A = np.asarray(A)
    cplx = np.iscomplexobj(A)
    Am = _to_libra(A, cplx)
    out = (CMATRIX if cplx else MATRIX)(A.shape[0], A.shape[1])
    lapack_getri(Am, out)
    return _to_numpy(out, cplx)


def det(A):
    """The determinant of a general matrix - the LU pivot product with the permutation sign (exactly 0
    for an exactly singular matrix)."""
    A = np.asarray(A)
    return lapack_det(_to_libra(A, np.iscomplexobj(A)))
