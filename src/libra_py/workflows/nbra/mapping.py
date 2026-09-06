# *********************************************************************************
# * Copyright (C) 2024-2025 Mohammad Shakiba, Alexey V. Akimov
# * Copyright (C) 2018-2024 Brendan A. Smith, Wei Li, Alexey V. Akimov
# *
# * This file is distributed under the terms of the GNU General Public License
# * as published by the Free Software Foundation, either version 3 of
# * the License, or (at your option) any later version.
# * See the file LICENSE in the root directory of this distribution
# * or <http://www.gnu.org/licenses/>.
# *
# *********************************************************************************/
"""

.. module:: mapping
   :platform: Unix, Windows
   :synopsis: This module implements the methods to map single-electron properties (e.g. KS basis)
       to many-electron ones (e.g. Slater Determinat basis)

.. moduleauthor:: Brendan Smith, Wei Li, and Alexey V. Akimov

"""

import os
import sys
import math
import numpy as np
# Fisrt, we add the location of the library to test to the PYTHON path
if sys.platform == "cygwin":
    from cyglibra_core import *
elif sys.platform == "linux" or sys.platform == "linux2":
    from liblibra_core import *

import libra_py.data_conv as data_conv
import libra_py.linalg_kernels as linalg_kernels
import libra_py.citools.slatdet as slatdet


def sd2indx(inp, nbasis=0, do_sort=False, user_notation=0):
    """
    This function maps a list of integers defining the occupied spin-orbitals in a SD onto 
    the global indices of the orbitals that are being occupied.

    Args:
        inp ( list of ints ): indices of the occupied spin-orbitals.
            Indexing starts with 1, not 0!

            Positive number ```n``` correspond to an alpha electron occupying the
            orbital (whether it is an alpha-spatial or beta-spatial component) ```n```

            Negative number ```-n``` correspond to a beta electron occupying the
            orbital (whether it is an alpha-spatial or beta-spatial component) ```n```

        nbasis ( int ): the total number of orbitals orbitals (both alpha- and beta-
            spatial components ) in the selected active space

        do_sort ( Boolean ): the flag to tell whether the indices should be sorted in
            a particular order (according to canonic ordering defined in the `citools.slatdet`):
                - True - for new scheme (needed for the SAC) [default]
                - use with False for Pyxaid mapping!

        user_notation (int):
            - 0 : short-hand - in this case, the mapping goes into [0, N) range [default - because it has been used for a while]
            - 1 : extended - in this case, the mapping goes into [0, 2*N) range


    Returns:
        - list of ints: the indices of the orbitals occupied in this SD (different convention)
        - integer: parity - reordering of the returned SD compared to the original ordering of spin-orbitals in the input
        - list of ints: the SD indices in the internal notation

    Example:

        Assume out active space consists of 4 orbitals:  |1^a>, |1^b>, |2^a>, |2^b>.
        Here, the superscript indicates the spin-component of the given orbital.

        In the step2 they are grouped as |1^a>, |2^a>, |1^b>, |2^b>

        And indexed (overall) as:           1     2      3      4  (as used in the input)
                                            0     1      2      3  (as used in the output)
        Note: this indexing convention assumes starting from 1, not from 0!

        So the ```nbasis``` should be = 4

        Input SD = [1, -3] means that the orbital |1^a> is occupied by an alpha electron |a>
        and the orbital |1^b> is occupied by a beta electron |b>, so the SD is:

        SD = 1/sqrt(2) * | |1^a>|a>   |1^b>|b> |

        This function will return the global indices of the occupied orbitals in the
        set of active orbitals. That is the conversion will be:
        [1, -3] -> [0, 2]
        Note that the output indices start from 0

    """

    inp = np.asarray(inp)

    spat = np.where(
        inp > 0,
        inp - 1,
        np.abs(inp) - 1 + (int(nbasis/2) if user_notation == 1 else 0)
    )

    # Rearrange according to a pre-defined canonical ordering, compute
    # the parity of the correponding permutation (compared to the input)
    out = list(spat)
    parity = 1
    if do_sort:
        out = sorted(spat, key=slatdet.canonical_sort_key)
        parity = slatdet.permutation_parity(spat, out)

    # plain ints: the C++ bindings (MATRIX.get, Py2Cpp_int, ...) reject
    # numpy.int64 arguments
    return [int(x) for x in out], parity



def ovlp_arb(_SD1, _SD2, S, active_space=None, verbose=False):
    """Compute the overlap of two generic SDs: <SD1|SD2>
    This functions translates the explicit use of two Python `for` loops
    into a more efficient way using numpy
    Args:

        _SD1 ( lists of ints ): first SD, such that:
            SeeAlso: ```inp``` in the ```sd2indx(inp)``` function

        _SD2 ( lists of ints ): second SD, such that:
            SeeAlso: ```inp``` in the ```sd2indx(inp)``` function

        S ( numpy array(N,N) ): is the matrix in the space of 1-el orbital

        active_space ( list of ints ): indices of the orbitals (starting from 1) to
            include into consideration. If None - all the orbitals will be used [ default: None ]

        verbose ( Bool ): whether to print some extra info [ default: False] 

    Returns:
        complex: the overlap of the two determinants <SD1|SD2>

    """

    nbasis = S.shape[0] #num_of_rows

    SD1, SD2 = [], []

    if active_space == None:
        SD1, SD2 = _SD1, _SD2
    else:
        for a in _SD1:
            if abs(a) in active_space:
                SD1.append(a)
        for a in _SD2:
            if abs(a) in active_space:
                SD2.append(a)

    res = 0.0 + 0j
    if len(SD1) != len(SD2):
        raise ValueError("trying to compute an overlap of SDs with different "
                         "numbers of electrons")

        
    # Apply the determinant formula
    det_size = len(SD1)
    s = np.zeros( (det_size, det_size), dtype=np.float64); 
    
    # Let's build the matrix related to sd1 and sd2 from the KS orbitals
    # For this, we require to turn each element into matrix indices
    sd1, p1 = sd2indx(SD1, nbasis, do_sort=True, user_notation=1)
    sd2, p2 = sd2indx(SD2, nbasis, do_sort=True, user_notation=1)

    if verbose==True:
        print(sd1, sd2)
        print(p1, p2)

    """ 
    #ALEXEY on 12/08/2025: Instead of this trick, we use user_notation = 1 above

    # What about beta indices?! We should add `nbasis/2` to them. This is added on 7/22/2025
    # With this simple approach, there is no need for tensor product or the `if else` clause brought
    # previously since the elements corresponding to the alpha-beta orbitals are already
    # zero in the two-spinor format of the KS matrices :)

    beta_indices = np.where(np.array(SD1) < 0)
    sd1 = np.array(sd1)
    sd1[beta_indices] += int(nbasis/2)
    beta_indices = np.where(np.array(SD2) < 0)
    sd2 = np.array(sd2)
    sd2[beta_indices] += int(nbasis/2)

    """

    # Making the numpy array
    s = S[sd1,:][:,sd2]
    
    if verbose==True:
        print(s)

    # x the sort parities: det over sorted rows -> the listed-order overlap
    res = p1 * p2 * np.linalg.det(s)

    return res



def ovlp_mat_arb(SD1, SD2, _S, active_space=None, verbose=False):
    """Compute a matrix of overlaps in the SD basis

    Args:
        SD1 ( list of lists of N ints ): a list of N SD determinants, such that:
            SD1[iSD] is a list of integers defining which orbitals are
            occupied in SD with index ```iSD``` and how
            SeeAlso: ```inp``` in the ```sd2indx(inp)``` function

        SD2 ( list of lists of M ints ): a list of M SD determinants, such that:
            SD2[iSD] is a list of integers defining which orbitals are
            occupied in SD with index ```iSD``` and how
            SeeAlso: ```inp``` in the ```sd2indx(inp)``` function

        _S ( numpy array(N,N) ): is the matrix in the space of 1-el orbital. 
            Note - the mapped indices should not be larger than K-1

        active_space ( list of ints ): indices of the orbitals (starting from 1) to
            include into consideration. If None - all the orbitals will be used [ default: None ]

        verbose ( Bool ): whether to print some extra info [ default: False]

    Returns:
        numpy.array(N, M): overlap matrix composed of elements <SD1(i)|SD2(j)>

    """

    # For transition compatibility
    S = None
    if type(_S)==CMATRIX or type(_S)==MATRIX:
        S = data_conv.MATRIX2nparray(_S, _dtype=np.complex128)
    else:
        S = _S

    N, M = len(SD1), len(SD2)
    # the determinant of a complex overlap is complex - a float64 result
    # silently dropped the imaginary part (numpy ComplexWarning)
    res = np.zeros( (N, M), dtype=np.result_type(np.asarray(S).dtype, np.float64) )

    for n in range(0, N):
        for m in range(0, M):
            res[n, m] = ovlp_arb(SD1[n], SD2[m], S, active_space, verbose)

    return res



def map_gen_matrix(SD1, SD2, X):
    """
    Args:
        SD1 ( lists of ints ): first SD, such that:
            SeeAlso: ```inp``` in the ```sd2indx(inp,nbasis)``` function

        SD2 ( lists of ints ): second SD, such that:
            SeeAlso: ```inp``` in the ```sd2indx(inp,nbasis)``` function

        X ( CMATRIX(nmo, nmo)) : the amatrix in a single-particple picture

    Returns:
        complex: mapped SD-basis property for the two determinants SD1 and SD2

    """

    nbasis = X.num_of_rows
    # Converting the SDs provided by the user into the internal format to be read by Libra
    sd1, _ = sd2indx(SD1, nbasis, False, 0)
    sd2, _ = sd2indx(SD2, nbasis, False, 0)

    res = delta(Py2Cpp_int(sd1), Py2Cpp_int(sd2))

    x = 0.0 + 0.0j
    # The SDs are coupled
    if res[0] == 1:
        x = X.get(res[1], res[2])

    sd1 = sorted(SD1)
    sd2 = sorted(SD2)
    res = delta(Py2Cpp_int(sd1), Py2Cpp_int(sd2))

    spin_ovlp = 1.0
    # Found 1 different entry
    if res[0] == 0:
        spin_ovlp = 0.0
    elif res[0] == 1:
        if res[1] * res[2] > 0.0:
            spin_ovlp = 1.0
        else:
            spin_ovlp = 0.0

    return x * spin_ovlp


def energy_arb(SD, e):
    """Computes the energy of a Slater Determinat (SD) state

    Args:
        SD ( list of ints ): indices of the orbitals occupied in this SD
            SeeAlso: ```inp``` in the ```sd2indx(inp,nbasis)``` function

        e ( MATRIX(nbasis, nbasis) ): energies of all 1-electron levels (alpha- and beta- components)

    Returns:
        double: the sum of the orbital energies for the occupied spin-orbitals -
            This is the approximation of the SD energy

    """
    if isinstance(e, np.ndarray)==True:
        nbasis = e.shape[0]
        sd, _ = sd2indx(SD, nbasis)
        res = 0.0
        for i in sd:
            res = res + e[i, i]


    elif isinstance(e, CMATRIX)==True or isinstance(e, MATRIX)==True:
        nbasis = e.num_of_rows
        sd, _ = sd2indx(SD, nbasis)
        res = 0.0+0.0j
        for i in sd:
            res = res + e.get(i,i)

    return res


def energy_mat_arb(SD, e, dE):
    """Computes a matrix of the SD energies

    Args:
        SD (list of lists of ints ): a list of SD determinants, such that:
            SD[iSD] is a list of integers defining which orbitals are
            occupied in SD with index ```iSD``` and how
            SeeAlso: ```inp``` in the ```sd2indx(inp,nbasis)``` function

        e ( MATRIX(nbasis, nbasis) ): energies of all 1-electron levels (alpha- and beta- components)
        dE ( list of doubles ): energy corrections added to each SD

    Returns:
        CMATRIX(N,N): the matrix of energies in the SD basis. Here, N = len(SD) - the number of SDs.

    """
    E = None

    if isinstance(e, np.ndarray)==True:
        n = len(SD)
        E = np.zeros((n, n))
        E0 = energy_arb(SD[0], e) + dE[0]
        for i in range(0, n):
            E[i, i] = energy_arb(SD[i], e) + dE[i] - E0

    elif isinstance(e, CMATRIX)==True or isinstance(e, MATRIX)==True:
        n = len(SD)
        E = CMATRIX(n,n)
        E0 = energy_arb(SD[0], e) + dE[0]*(1.0+0.0j)
        for i in range(0, n):
            E.set(i,i, energy_arb(SD[i], e) + dE[i]*(1.0+0.0j) - E0 )

    return E


def orbs2spinorbs(s):
    """
    This function converts the matrices in the orbital basis (e.g. old PYXAID style)
    to the spin-orbital basis.
    Essentially, it makes a block matrix of a double dimension:
           ( s  0 )
    s -->  ( 0  s )

    This is meant to be used for backward compatibility with PYXIAD-generated data

    Args:
        s ( CMATRIX(N,N) ): whatever matrix in the orbital basis, N - is the number of
            doubly-occupied orbitals

    Returns:
        CMATRIX(2N, 2N): whatever matrix in the spin-orbital basis. It has a doubled dimensionality

    """

    sz = s.num_of_cols
    zero = CMATRIX(sz, sz)
    act_sp1 = list(range(0, sz))
    act_sp2 = list(range(sz, 2 * sz))

    S = CMATRIX(2 * sz, 2 * sz)

    push_submatrix(S, s, act_sp1, act_sp1)
    push_submatrix(S, zero, act_sp1, act_sp2)
    push_submatrix(S, zero, act_sp2, act_sp1)
    push_submatrix(S, s, act_sp2, act_sp2)

    return S


def ovlp_mat_arb_batched(SD1, SD2, S2, active_space=None, mem_limit_mb=256):
    """
    Batched equivalent of ovlp_mat_arb (listed-order determinant convention:
    rows enter the determinants as given): one LAPACK-batched det call per chunk
    instead of nSD1*nSD2 Python-level np.linalg.det calls. Bit-exact against
    the loop and 7-11x faster; chunked over bra SDs so the gathered submatrix
    stack stays under mem_limit_mb.

    Args:
        SD1, SD2: lists of signed spin-orbital SDs (Libra convention:
            +i alpha, -i beta, 1-based); any spin structure is allowed, but
            after active-space filtering all SDs must retain the same number
            of electrons (pairs with mismatched counts are an error in
            ovlp_arb as well)
        S2: two-spinor MO matrix (2*norb, 2*norb)
        active_space: 1-based orbital indices to keep (same filtering rule as
            ovlp_arb: SD entries with |a| outside the list are dropped)
        mem_limit_mb: cap for the temporary (chunk, nSD2, nelec, nelec) stack

    Returns:
        np.ndarray (nSD1, nSD2)
    """
    n_half = S2.shape[0] // 2

    def indices(SDs, label):
        rows = _sd_index_rows(SDs, n_half, active_space)
        lengths = {len(r) for r in rows}
        if len(lengths) != 1:
            raise ValueError(f"{label}: SDs retain different electron counts "
                             f"{sorted(lengths)} (after active-space filtering); "
                             f"overlaps between determinants of different size "
                             f"are not defined")
        rows = np.array(rows)
        # dets over sorted rows, x the sort parities = listed-order overlaps
        return np.sort(rows, axis=1), _sort_parities(rows)

    i1, p1 = indices(SD1, "SD1")
    i2, p2 = indices(SD2, "SD2")
    n1, n2, ne = len(i1), len(i2), i1.shape[1]
    if i2.shape[1] != ne:
        raise ValueError(f"SD1 and SD2 have different electron counts "
                         f"({ne} vs {i2.shape[1]})")
    res = np.empty((n1, n2), dtype=np.result_type(S2.dtype, np.float64))
    chunk = max(1, int(mem_limit_mb * 1e6 / (n2 * ne * ne * 8)))
    for a in range(0, n1, chunk):
        b = min(a + chunk, n1)
        sub = S2[i1[a:b, None, :, None], i2[None, :, None, :]]
        res[a:b] = np.linalg.det(sub)
    return res * np.outer(p1, p2)


def _sd_index_rows(SDs, n_half, active_space=None, sort=False):
    """
    Signed spin-orbital SDs -> per-SD 0-based two-spinor row-index arrays
    (+i alpha -> i-1, -i beta -> i-1+n_half), optionally filtered to an
    active_space (1-based orbital indices) and per-row sorted to the
    canonical order.
    """
    rows = []
    for sd in SDs:
        a = np.asarray(sd, dtype=int)
        if active_space is not None:
            a = a[np.isin(np.abs(a), active_space)]
        j = np.abs(a) - 1
        j[a < 0] += n_half
        rows.append(np.sort(j) if sort else j)
    return rows


def _sd_indices(SDs, n_half):
    """Rectangular (nSD, nelec) index array for uniform-length SD lists."""
    return np.array(_sd_index_rows(SDs, n_half))


def _perm_parity(row):
    """Sign of the permutation that sorts `row` (via cycle decomposition)."""
    perm = np.argsort(row, kind="stable")
    visited = np.zeros(len(row), dtype=bool)
    sign = 1
    for start in range(len(row)):
        if visited[start]:
            continue
        j, cyc = start, 0
        while not visited[j]:
            visited[j] = True
            j = perm[j]
            cyc += 1
        if cyc % 2 == 0:
            sign = -sign
    return sign


def _sort_parities(idx):
    """Per-row sign of the permutation that sorts the mapped indices -
    converts list-order determinants to the canonical (sorted) convention
    used by ovlp_arb."""
    return np.array([_perm_parity(row) for row in idx], dtype=float)


def _excitation_structure(idx):
    """
    Excitation structure of SD index rows relative to a common reference,
    treating every row as a SET of spin-orbitals (input order irrelevant).

    The reference is the sorted row of minimal index sum (replacing an
    occupied spin-orbital by a higher one strictly increases the sum).

    Returns (ref, deg, holes, parts, sigma), or None if a row carries a
    repeated spin-orbital:
        ref: sorted reference row (nelec,)
        deg[i]: excitation degree of row i relative to ref
        holes[i]: POSITIONS in ref of the removed orbitals, ascending
        parts[i]: the replacement spin-orbitals, ascending
        sigma[i]: parity converting the in-place arrangement (ref with
            ref[holes] := parts, paired in ascending order) to the canonical
            sorted order of row i
    """
    srt = np.sort(idx, axis=1)
    if srt.shape[1] > 1 and np.any(np.diff(srt, axis=1) == 0):
        return None                       # repeated spin-orbital
    ref = srt[np.argmin(srt.sum(axis=1))]
    ref_set = set(ref.tolist())
    pos = {o: m for m, o in enumerate(ref.tolist())}
    deg = np.empty(len(srt), dtype=int)
    holes, parts = [], []
    sigma = np.empty(len(srt))
    for k, row in enumerate(srt):
        row_set = set(row.tolist())
        prt = sorted(row_set - ref_set)
        hol = sorted(pos[o] for o in ref_set - row_set)
        deg[k] = len(prt)
        holes.append(hol)
        parts.append(prt)
        L = ref.copy()
        L[hol] = prt
        sigma[k] = _perm_parity(L)
    return ref, deg, holes, parts, sigma


def _degree_groups(deg, holes, parts, table_orbs):
    """Pack per-SD hole positions / particle table rows into per-degree
    arrays: {degree: (sd_indices, holes (n,d), particle_rows (n,d))}."""
    groups = {}
    for i, d in enumerate(deg):
        groups.setdefault(int(d), []).append(i)
    packed = {}
    for d, idxs in groups.items():
        idxs = np.asarray(idxs, dtype=int)
        P = np.array([holes[i] for i in idxs], dtype=int).reshape(len(idxs), d)
        X = np.array([parts[i] for i in idxs], dtype=int).reshape(len(idxs), d)
        packed[d] = (idxs, P, np.searchsorted(table_orbs, X))
    return packed


def ovlp_mat_arb_lemma(SD1, SD2, S2, active_space=None, mem_limit_mb=256):
    """
    Fast SD overlaps for common-reference excited bases via the generalized
    matrix determinant lemma - ARBITRARY excitation degrees. Listed-order
    determinant convention (rows enter the determinants as given).

    Every SD is treated as an excitation of the reference determinant of its
    list (the row of minimal spin-orbital index sum). A bra SD replacing k
    reference orbitals {p}->{a} against a ket SD replacing l of them {q}->{b}
    evaluates as the (k+l) x (k+l) determinant of precomputed tables

        det(M_ij) = det(A) * det( [[ R[a,p],  (T-W)[a,b] ],
                                   [ G[q,p],   C[q,b]    ]] )

    with A = S2[ref1, ref2], G = A^-1, R = U G, C = G V, T = R V,
    W = S2[a, b], where U/V hold the replacement rows/columns against the
    opposite reference. Singly-excited pairs (the TDDFPT/CIS case) reduce to
    the scalar identity R_p C_q - G[q,p] (T - w); doubles cost a 4x4
    determinant per pair - still O(1) in nelec. Pairs are processed per
    (k, l) degree group with one batched LAPACK det call per group, so the
    only O(nSD^2 * nelec) operation remains a single GEMM.

    Args:
        active_space: same filtering rule as ovlp_arb, applied before the
            structure detection
        mem_limit_mb: cap for the temporary per-group det stacks

    Returns None when the determinants have <= 2 electrons (decided before
    the structure detection - the (k+l) x (k+l) lemma determinants are at
    least as large as the direct ne x ne ones there), the SDs carry repeated
    spin-orbitals, the maximal bra degree + maximal ket degree reaches nelec
    (a direct determinant is as cheap), or the reference block is
    near-singular - callers fall back to ovlp_mat_arb_batched
    (ovlp_mat_arb_auto does this automatically).
    """
    n_half = S2.shape[0] // 2

    def indices(SDs):
        rows = _sd_index_rows(SDs, n_half, active_space)
        if len({len(r) for r in rows}) != 1:
            return None
        return np.array(rows)

    i1, i2 = indices(SD1), indices(SD2)
    if i1 is None or i2 is None or i1.shape[1] != i2.shape[1]:
        return None
    ne = i1.shape[1]
    if ne <= 2:
        return None
    s1 = _excitation_structure(i1)
    s2 = _excitation_structure(i2)
    if s1 is None or s2 is None:
        return None
    ref1, deg1, holes1, parts1, sig1 = s1
    ref2, deg2, holes2, parts2, sig2 = s2
    if deg1.max() + deg2.max() >= ne:
        return None

    A = S2[np.ix_(ref1, ref2)]
    # the reference determinant stays in the log domain (slogdet): the guard below reads log|det|
    # and a plain pivot product would under/overflow first for large occupied blocks
    sign, logdet = np.linalg.slogdet(A)
    if sign == 0 or logdet < -30:                 # singular/near-singular ref
        return None
    D0 = sign * np.exp(logdet)
    G = linalg_kernels.inv(A)

    pa = np.unique(np.concatenate([np.asarray(x, dtype=int) for x in parts1] +
                                  [np.empty(0, dtype=int)]))
    pb = np.unique(np.concatenate([np.asarray(x, dtype=int) for x in parts2] +
                                  [np.empty(0, dtype=int)]))
    U = S2[pa][:, ref2]                           # (na, ne): bra particle rows
    V = S2[ref1][:, pb]                           # (ne, nb): ket particle cols
    W = S2[np.ix_(pa, pb)]                        # (na, nb)
    R = linalg_kernels.product(U, G)              # (na, ne)
    C = linalg_kernels.product(G, V)              # (ne, nb)
    TW = linalg_kernels.product(R, V) - W         # (na, nb) - the one GEMM
    # the block determinants below stay on numpy's BATCHED det: one C loop over the stacked
    # submatrices calling LAPACK once each - a python loop over lapack_det would bring back the
    # per-call overhead this kernel removed

    g1 = _degree_groups(deg1, holes1, parts1, pa)
    g2 = _degree_groups(deg2, holes2, parts2, pb)

    res = np.empty((len(i1), len(i2)), dtype=np.result_type(S2.dtype, np.float64))
    for k, (I, P, Ax) in g1.items():
        for l, (J, Q, Bx) in g2.items():
            nk, ml = len(I), len(J)
            if k == 0 and l == 0:
                block = np.full((nk, ml), D0)
            elif l == 0:
                d = (R[Ax[:, 0], P[:, 0]] if k == 1 else
                     np.linalg.det(R[Ax[:, :, None], P[:, None, :]]))
                block = np.repeat(D0 * d[:, None], ml, axis=1)
            elif k == 0:
                d = (C[Q[:, 0], Bx[:, 0]] if l == 1 else
                     np.linalg.det(C[Q[:, :, None], Bx[:, None, :]]))
                block = np.repeat(D0 * d[None, :], nk, axis=0)
            elif k == 1 and l == 1:
                Rp = R[Ax[:, 0], P[:, 0]][:, None]
                Cq = C[Q[:, 0], Bx[:, 0]][None, :]
                Gqp = G[Q[None, :, 0], P[:, None, 0]]
                TWab = TW[Ax[:, None, 0], Bx[None, :, 0]]
                block = D0 * (Rp * Cq - Gqp * TWab)
            else:
                m = k + l
                Rblk = R[Ax[:, :, None], P[:, None, :]]            # (nk, k, k)
                Cblk = C[Q[:, :, None], Bx[:, None, :]]            # (ml, l, l)
                block = np.empty((nk, ml), dtype=res.dtype)
                chunk = max(1, int(mem_limit_mb * 1e6 / (ml * m * m * 8)))
                for s in range(0, nk, chunk):
                    e = min(s + chunk, nk)
                    B = np.empty((e - s, ml, m, m), dtype=res.dtype)
                    B[:, :, :k, :k] = Rblk[s:e, None]
                    B[:, :, :k, k:] = TW[Ax[s:e, None, :, None], Bx[None, :, None, :]]
                    B[:, :, k:, :k] = G[Q[None, :, :, None], P[s:e, None, None, :]]
                    B[:, :, k:, k:] = Cblk[None, :]
                    block[s:e] = np.linalg.det(B)
                block *= D0
            res[np.ix_(I, J)] = block
    # in-place arrangement -> sorted (sig) -> listed order (sort parities)
    res *= np.outer(sig1 * _sort_parities(i1), sig2 * _sort_parities(i2))
    return res


def ovlp_mat_arb_lemma_only(SD1, SD2, S2, active_space=None, mem_limit_mb=256):
    """ovlp_mat_arb_lemma, or a named error when the lemma declines - the
    user-facing 'lemma' kernel choice must not surface None into
    arithmetic."""
    res = ovlp_mat_arb_lemma(SD1, SD2, S2, active_space, mem_limit_mb)
    if res is None:
        raise RuntimeError(
            "sd_overlap_kernel='lemma': the determinant-lemma kernel declined "
            "this SD basis (<= 2 electrons, repeated spin-orbitals, excitation "
            "degree reaching nelec, or a near-singular reference block) - use "
            "'auto', which falls back to the batched kernel automatically")
    return res


def ovlp_mat_arb_auto(SD1, SD2, S2, active_space=None, mem_limit_mb=256):
    """Determinant-lemma kernel with automatic fallback to the batched one.
    Listed-order determinant convention, like every ovlp kernel here."""
    res = ovlp_mat_arb_lemma(SD1, SD2, S2, active_space, mem_limit_mb)
    if res is None:
        res = ovlp_mat_arb_batched(SD1, SD2, S2, active_space, mem_limit_mb)
    return res
