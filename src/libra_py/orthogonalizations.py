# *********************************************************************************
# * Copyright (C) 2026 Alexey V. Akimov
# *
# * This file is distributed under the terms of the GNU General Public License
# * as published by the Free Software Foundation, either version 3 of
# * the License, or (at your option) any later version.
# * See the file LICENSE in the root directory of this distribution
# * or <http://www.gnu.org/licenses/>.
# *
# *********************************************************************************/

"""
.. module:: orthogonalizations
   :platform: Unix, Windows
   :synopsis: this module implements various general-purpose orthogonalization procedures

.. moduleauthor:: Alexey V. Akimov, ChatGPT

"""

import hashlib
import pickle

import numpy as np
from libra_py import linalg_kernels
from scipy.linalg import fractional_matrix_power


def carried_state_digest(state):
    """
    Content key for quantities carried between calls of a per-trajectory driver
    (e.g. the Lowdin factor of the previous step, stored together with a digest
    of the step data it was derived from). Compare the stored digest against the
    digest of this call's previous-step data before reusing the carried quantity:
    a mismatch means the params state went out of sync between calls.

    Args:
        state: any picklable object (arrays, nested lists/dicts of step data)

    Returns:
        bytes: a 16-byte content digest of the state
    """
    return hashlib.blake2b(pickle.dumps(state, protocol=4), digest_size=16).digest()


def _identity_deviation(S):
    """max|S - I| without materializing an identity matrix (an N=4000
    complex S costs ~0.5 GB in eye + temp on the per-step paths)."""
    d = np.array(S, copy=True)
    idx = np.arange(d.shape[0])
    d[idx, idx] -= 1.0
    return float(np.abs(d).max())


def validate_overlap_matrix(S, label="overlap", require_identity=None,
                            sym_tol=1e-8):
    """A valid overlap (Gram) matrix is finite, Hermitian and has a strictly
    positive diagonal; anything else is corrupted data, not physics, and
    no orthonormalization can repair it. With require_identity set, the
    matrix must additionally stay within that deviation from identity -
    the invariant of freshly produced MO self-overlaps (production-time
    check; loaders keep the softer warning so legacy archives remain
    repairable). Returns the deviation from identity."""
    S = np.asarray(S)
    if not np.all(np.isfinite(S)):
        raise ValueError(F"{label}: the overlap matrix contains non-finite "
                         "entries - corrupted data")
    scale = max(1.0, float(np.abs(S).max()))
    asym = float(np.abs(S - S.conj().T).max())
    if asym > sym_tol * scale:
        raise ValueError(F"{label}: the overlap matrix is not Hermitian "
                         F"(max|S - S^H| = {asym:.3e} at scale {scale:.3e}) - "
                         "a Gram matrix must be Hermitian; corrupted data")
    dmin = float(np.real(np.diag(S)).min())
    if dmin <= 0.0:
        raise ValueError(F"{label}: the overlap matrix has a non-positive "
                         F"diagonal entry ({dmin:.3e}) - vector norms must be "
                         "positive; corrupted data")
    if require_identity is None:
        return None
    dev = _identity_deviation(S)
    if dev > require_identity:
        raise ValueError(F"{label}: the overlap matrix deviates from identity "
                         F"by {dev:.3f} (limit {require_identity}) - freshly "
                         "produced MO self-overlaps must be orthonormal; the "
                         "producing chain (AO ordering, normalization, "
                         "conversion) is broken")
    return dev


def lowdin_inverse_sqrt(S, thresh=1e-10):
    """
    Compute the inverse square root of an overlap matrix using Löwdin symmetric orthogonalization.

    This function evaluates S^{-1/2} via eigenvalue decomposition:
        S = U diag(λ) U†  →  S^{-1/2} = U diag(λ^{-1/2}) U†

    Small eigenvalues below a specified threshold are treated as zero to ensure
    numerical stability and avoid divergences. The input matrix is explicitly
    symmetrized to enforce Hermiticity before diagonalization.

    Parameters
    ----------
    S : array_like (N, N)
        Real or complex overlap matrix. It is assumed to be Hermitian
        (or close to Hermitian within numerical precision).
    thresh : float, optional
        Eigenvalue cutoff below which eigenvalues are treated as zero.
        This prevents instabilities due to near-linear dependencies.
        Default is 1e-10.

    Returns
    -------
    S_inv_sqrt : ndarray (N, N)
        The inverse square root of the overlap matrix, S^{-1/2}, computed
        in the Löwdin (symmetric) orthogonalization scheme.

    Notes
    -----
    - The function enforces Hermiticity via (S + S†)/2 before diagonalization.
    - Input arrays are cast to float64 or complex128 to ensure compatibility
      with NumPy linear algebra routines.
    - Eigenvalues below `thresh` are set to zero, effectively projecting out
      linearly dependent components of the basis.
    - The resulting matrix can be used to orthonormalize a non-orthogonal basis:
          C_orth = S^{-1/2} C

    Raises
    ------
    LinAlgError
        If the eigenvalue decomposition fails.

    Examples
    --------
    >>> S = np.array([[1.0, 0.2], [0.2, 1.0]])
    >>> S_inv_sqrt = lowdin_inverse_sqrt(S)
    >>> np.allclose(S_inv_sqrt @ S @ S_inv_sqrt, np.eye(2))
    True
    """

    # Ensure numpy array
    S = np.array(S)

    # Force supported dtype
    if np.iscomplexobj(S):
        S = S.astype(np.complex128)
    else:
        S = S.astype(np.float64)

    # Enforce Hermiticity 
    S = 0.5 * (S + S.conj().T)

    eigvals, eigvecs = linalg_kernels.eigh(S)

    # a Gram matrix is positive semi-definite: eigenvalues below -noise are
    # corruption, not linear dependency - refuse to silently project them out
    neg_tol = max(thresh, 1e-8 * float(np.abs(eigvals).max()))
    if float(eigvals.min()) < -neg_tol:
        raise ValueError(
            F"lowdin_inverse_sqrt: the overlap matrix has a negative "
            F"eigenvalue ({float(eigvals.min()):.3e}) - not a valid Gram "
            "matrix; corrupted data")

    eigvals_inv_sqrt = np.zeros_like(eigvals)

    for i, val in enumerate(eigvals):
        if val > thresh:
            eigvals_inv_sqrt[i] = 1.0 / np.sqrt(val)
        else:
            eigvals_inv_sqrt[i] = 0.0

    S_inv_sqrt = eigvecs @ np.diag(eigvals_inv_sqrt) @ eigvecs.conj().T

    return S_inv_sqrt


def get_Lowdin_scipy(S):
    """
    This function computes the inverse square root of a matrix S^{-1/2}.

    Args:

        S (numpy array): Overlap matrix in form of a numpy array.

    Returns:

        S_inverse_square_root (numpy array): The inverse square root of S.
    """
    # a non-PSD input would make the inverse square root silently complex
    validate_overlap_matrix(S, "get_Lowdin_scipy")
    S_h = 0.5 * (np.asarray(S) + np.asarray(S).conj().T)
    w, V = linalg_kernels.eigh(S_h)
    if float(w.min()) < -1e-8:
        raise ValueError(F"get_Lowdin_scipy: the overlap matrix has a negative "
                         F"eigenvalue ({w.min():.3e}) - not a valid Gram matrix; "
                         "corrupted data")
    # one eigendecomposition serves the guard AND the inverse square root
    # (exact for Hermitian input; the general Schur route costs ~10x)
    return (V * (w ** -0.5)) @ V.conj().T


def apply_orthonormalization_scipy(S_1, S_2, St):
    """
    This function applies orthonormalization for the overlap matrices.

    Args:

        S_1 (numpy array): The overlap matrix of time n.

        S_2 (numpy array): The overlap matrix of time n+1.

        St (numpy array): The time-overlap matrix.

    Returns:

        St_orthonormalized (numpy array): The orthonormalized St.

        S_1_orthonormalized (numpy array): The orthonormalized S_1.

    """
    U1 = get_Lowdin_scipy(S_1)
    U2 = get_Lowdin_scipy(S_2)

    # U^H, not U^T: the Lowdin factor of a complex Hermitian S is complex
    # Hermitian, and its plain transpose is its conjugate
    St_orthonormalized = np.linalg.multi_dot([U1.conj().T, St, U2])
    S_1_orthonormalized = np.linalg.multi_dot([U1.conj().T, S_1, U1])

    return St_orthonormalized, S_1_orthonormalized


def lowdin_digest(S, tag=""):
    """Content key for the Lowdin factor caches (step3's per-process
    slot, the driver's per-trajectory dicts). S is a single array or a
    tuple/list of per-spin blocks (UKS) - the blocks are hashed in order."""
    h = hashlib.blake2b(digest_size=16)
    for blk in (S if isinstance(S, (tuple, list)) else (S,)):
        h.update(np.ascontiguousarray(blk).tobytes())
    h.update(tag.encode())
    return h.digest()
