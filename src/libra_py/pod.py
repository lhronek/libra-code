"""
Projection-operator diabatization (POD) of a one-electron Hamiltonian in a fragment
partition of the AO basis: Kondov, Cizek, Benesch, Wang, Thoss, J. Phys. Chem. C 111,
11970 (2007), Futera, Blumberger, J. Phys. Chem. C 121, 19677 (2017).

The AO basis is Lowdin-orthogonalized, H' = S^-1/2 H S^-1/2, and reordered into the
fragment (block) AO sets. Every diagonal block H'_ff is diagonalized separately - its
eigenvectors are the block states, orbitals localized on fragment f - and the coupling
between the states of two fragments is the off-diagonal block transformed into the two
eigenbases, V_fg = U_f^T H'_fg U_g. The results carry the same keys as
libra_py.packages.cp2k.methods.read_pod_cp2k_files, which reads the same quantities from
the binary files of Futera's CP2K, so both feed pod_fragment_operators alike.

The heavy linear algebra runs through libra's MATRIX class and its LAPACK/OpenBLAS kernels:
sqrt_matrix (dsyevd + dgemm) for S^(+/-1/2), MATRIX products (dgemm) for the transforms,
lapack_syevd for the blocks (the Eigen solve_eigen when the build has no LAPACK backend).

The second-generation schemes (pod2_block_states) partition first, orthogonalize within the
fragments only and solve the fragment eigenproblems, so the fragment states stay local: POD2 reads
the coupling between them as they are (non-orthogonal across fragments), POD2L and POD2GS
orthogonalize the donor-acceptor pairs of states first - symmetrically (Lowdin) or keeping one
fragment's state (Gram-Schmidt).
"""
import numpy as np
from liblibra_core import MATRIX, intList, solve_eigen, sqrt_matrix, pop_submatrix, have_lapack, lapack_syevd, set_from_buffer
from libra_py import data_conv


def _ao_blocks(ao_atom, fragment_atom_lists):
    """The AO index lists (0-based, ascending) of the fragments (1-based atom lists, disjoint)."""
    ao_atom = np.asarray(ao_atom, dtype=int)
    natom = int(ao_atom.max()) + 1
    seen = set()
    blocks = []
    for f, atoms in enumerate(fragment_atom_lists):
        atoms = [int(a) for a in atoms]
        if not atoms or min(atoms) < 1 or max(atoms) > natom:
            raise ValueError(F"fragment {f} atom list {atoms} is empty or outside 1..{natom}")
        if seen & set(atoms):
            raise ValueError(F"fragment {f} shares atoms {sorted(seen & set(atoms))} with an earlier fragment")
        seen |= set(atoms)
        blocks.append([int(i) for i in np.flatnonzero(np.isin(ao_atom + 1, atoms))])
    return blocks


def _ilist(idx):
    out = intList()
    out.extend([int(i) for i in idx])
    return out


def _eigh_ascending(H, E, C):
    """Eigenpairs of a symmetric MATRIX, ascending: LAPACK dsyevd when the build linked it (one triangle of a
    block symmetric to rounding), else the Eigen solver with its symmetrization."""
    if have_lapack():
        lapack_syevd(H, E, C)
    else:
        solve_eigen(H, E, C, 1)


def pod_block_states(H_list, S, ao_atom, fragment_atom_lists):
    """
    POD of one or two spin channels of an AO Hamiltonian.

    Args:
        H_list (list of arrays (n_ao, n_ao)): the AO Kohn-Sham matrices, one per spin channel
        S (array (n_ao, n_ao)): the AO overlap in the same AO order
        ao_atom (array (n_ao,)): 0-based atom of every AO (load_wfn_mo_data supplies it)
        fragment_atom_lists (list of lists of int): 1-based atoms of the disjoint fragments,
            every atom of the system assigned (the blocks partition the AO space)

    Returns:
        dict: n_spins, n_blocks, n_ao_blk (AOs per block), tmr (S^+1/2), tmf (S^-1/2), S,
            energies {(block, spin): block-state energies [Ha], ascending}, couplings
            {(block_i, block_j, spin): U_i^T H'_ij U_j, (n_ao_i, n_ao_j)}, coeffs {(block,
            spin): block MOs in the AO basis, (n_ao, n_ao_block)}, blocks (the AO index lists)
    """
    S = np.asarray(S, dtype=float)
    n = S.shape[0]
    blocks = _ao_blocks(ao_atom, fragment_atom_lists)
    covered = sorted(i for b in blocks for i in b)
    if covered != list(range(n)):
        raise ValueError(F"the fragments cover {len(covered)} of {n} AOs - every atom must belong to a fragment")
    S_m = data_conv.nparray2MATRIX(S)
    Y = MATRIX(n, n)          # S^(+1/2)
    X = MATRIX(n, n)          # S^(-1/2)
    sqrt_matrix(S_m, Y, X)
    all_rows = _ilist(range(n))
    out = {"n_spins": len(H_list), "n_blocks": len(blocks), "n_ao_blk": [len(b) for b in blocks],
           "tmr": data_conv.MATRIX2nparray(Y, np.float64), "tmf": data_conv.MATRIX2nparray(X, np.float64),
           "S": S, "energies": {}, "couplings": {}, "coeffs": {}, "blocks": blocks}
    for sp, H in enumerate(H_list):
        Hp = X * data_conv.nparray2MATRIX(np.asarray(H, dtype=float)) * X      # DGEMM
        U, ids = [], []
        for f, blk in enumerate(blocks):
            nb = len(blk)
            ib = _ilist(blk)
            Hff = MATRIX(nb, nb)
            pop_submatrix(Hp, Hff, ib, ib)
            E = MATRIX(nb, nb)
            Uf = MATRIX(nb, nb)
            _eigh_ascending(Hff, E, Uf)
            U.append(Uf)
            ids.append(ib)
            out["energies"][(f, sp)] = np.diag(data_conv.MATRIX2nparray(E, np.float64)).copy()
            # block MOs back in the AO basis: c = S^-1/2 (U embedded in the block rows)
            Xf = MATRIX(n, nb)
            pop_submatrix(X, Xf, all_rows, ib)
            out["coeffs"][(f, sp)] = data_conv.MATRIX2nparray(Xf * Uf, np.float64)
        for i in range(len(blocks)):
            for j in range(i + 1, len(blocks)):
                Hij = MATRIX(len(blocks[i]), len(blocks[j]))
                pop_submatrix(Hp, Hij, ids[i], ids[j])
                out["couplings"][(i, j, sp)] = data_conv.MATRIX2nparray(U[i].T() * Hij * U[j], np.float64)
    return out


def _pairwise_couplings(H_raw, S_st, E_i, E_j, orthogonalization, keep_first):
    """
    The POD2L / POD2GS couplings of every donor-acceptor pair of fragment states from the raw matrix
    elements h = <i|H|j> and overlaps s = <i|j> of the (block-orthonormal, mutually non-orthogonal)
    fragment eigenstates with energies e_i, e_j: the two-state Lowdin orthogonalization gives
    (h - s (e_i + e_j) / 2) / (1 - s^2), the Gram-Schmidt orthogonalization that keeps the first
    state gives (h - s e_i) / sqrt(1 - s^2) (the second: e_j in place of e_i).
    """
    s = np.asarray(S_st, dtype=float)
    h = np.asarray(H_raw, dtype=float)
    if np.abs(s).max() >= 1.0 - 1e-12:
        raise ValueError(F"a pair of fragment states is parallel (|<i|j>| = {np.abs(s).max():.6f}) - the "
                         F"fragments share a state, no pairwise orthogonalization is defined")
    e_i = np.asarray(E_i, dtype=float)[:, None]
    e_j = np.asarray(E_j, dtype=float)[None, :]
    if orthogonalization == "lowdin":
        return (h - 0.5 * s * (e_i + e_j)) / (1.0 - s**2)
    e_kept = e_i if keep_first else e_j
    return (h - s * e_kept) / np.sqrt(1.0 - s**2)


def _orthonormalize_blocks(c_raw, S_m, cols, orthogonalization, kept_block):
    """
    The whole-set counterpart of the pairwise orthogonalization, for the fragment operators (a
    projector partition needs one orthonormal set): Lowdin - c (c^T S c)^-1/2 over all fragment
    states at once - Gram-Schmidt - the kept block untouched, every other block (in index order)
    projected orthogonal to the blocks before it and re-orthonormalized within itself by its own
    Lowdin step, so its states stay as close to the fragment eigenstates as the projection allows.
    """
    n = c_raw.shape[0]
    c_m = data_conv.nparray2MATRIX(c_raw)
    if orthogonalization == "lowdin":
        S_st = c_m.T() * S_m * c_m                                       # DGEMM
        Y = MATRIX(n, n)
        T = MATRIX(n, n)
        sqrt_matrix(S_st, Y, T)
        return data_conv.MATRIX2nparray(c_m * T, np.float64)
    order = [kept_block] + [g for g in range(len(cols)) if g != kept_block]
    c_orth = c_raw.copy()
    done = []
    for g in order:
        if not done:
            done.append(g)
            continue
        cg = data_conv.nparray2MATRIX(c_orth[:, cols[g]])
        for p_ in done:
            cp = data_conv.nparray2MATRIX(c_orth[:, cols[p_]])
            cg = cg - cp * (cp.T() * (S_m * cg))                         # DGEMM
        nb = len(cols[g])
        S_gg = cg.T() * S_m * cg
        Y = MATRIX(nb, nb)
        T = MATRIX(nb, nb)
        sqrt_matrix(S_gg, Y, T)
        c_orth[:, cols[g]] = data_conv.MATRIX2nparray(cg * T, np.float64)
        done.append(g)
    return c_orth


def pod2_block_states(H_list, S, ao_atom, fragment_atom_lists, orthogonalization="lowdin", kept_block=0):
    """
    Second-generation POD (POD2, POD2L, POD2GS - Futera and Blumberger) of one or two spin channels of an
    AO Hamiltonian: partition first, orthogonalize within each fragment only (S_ff^-1/2, no mixing of the
    fragments' AOs), solve the fragment eigenproblems - the states are local but not orthogonal across
    fragments - and block-diagonalize: POD2 reads the coupling between these non-orthogonal states as it
    is, POD2L and POD2GS first orthogonalize every donor-acceptor pair of states, symmetrically (Lowdin) or
    keeping one fragment's state (Gram-Schmidt). The site energies are the fragment eigenvalues. The
    fragment operators of POD2L and POD2GS come from the same orthogonalization applied to the whole set of
    fragment states (coeffs) - POD2 keeps the non-orthogonal states there, which form no projector
    partition. The raw local states stay available (coeffs_raw).

    Args:
        H_list, S, ao_atom, fragment_atom_lists: as in pod_block_states
        orthogonalization (str): "none" (POD2), "lowdin" (POD2L) or "gram_schmidt" (POD2GS)
        kept_block (int): POD2GS - the fragment whose states stay unchanged (a pair of two other
            fragments keeps the lower-index one)

    Returns:
        dict: n_spins, n_blocks, n_ao_blk, S, blocks, orthogonalization, kept_block, energies {(block,
            spin)}, couplings {(block_i, block_j, spin): the pairwise-orthogonalized couplings, (n_i, n_j)},
            couplings_raw {the same in the non-orthogonal fragment-state basis, the POD2 block}, state_overlaps
            {(block_i, block_j, spin): <i|j>}, coeffs {(block, spin): the whole-set orthonormalized states in the
            AO basis, (n_ao, n_block)}, coeffs_raw {the fragment eigenstates, local and non-orthogonal}
    """
    if orthogonalization not in ("none", "lowdin", "gram_schmidt"):
        raise ValueError(F"orthogonalization {orthogonalization!r} - use 'none' (POD2), 'lowdin' (POD2L) or "
                         F"'gram_schmidt' (POD2GS)")
    S = np.asarray(S, dtype=float)
    n = S.shape[0]
    blocks = _ao_blocks(ao_atom, fragment_atom_lists)
    covered = sorted(i for b in blocks for i in b)
    if covered != list(range(n)):
        raise ValueError(F"the fragments cover {len(covered)} of {n} AOs - every atom must belong to a fragment")
    nblk = len(blocks)
    if not 0 <= int(kept_block) < nblk:
        raise ValueError(F"kept_block {kept_block} outside the {nblk} fragments")
    kept_block = int(kept_block)
    S_m = data_conv.nparray2MATRIX(S)
    ids = [_ilist(b) for b in blocks]
    cols = []
    col = 0
    for b in blocks:
        cols.append(list(range(col, col + len(b))))
        col += len(b)
    # block-local Lowdin transformations X_f = S_ff^-1/2
    X = []
    for f, blk in enumerate(blocks):
        nb = len(blk)
        Sff = MATRIX(nb, nb)
        pop_submatrix(S_m, Sff, ids[f], ids[f])
        Yf = MATRIX(nb, nb)
        Xf = MATRIX(nb, nb)
        sqrt_matrix(Sff, Yf, Xf)
        X.append(Xf)
    out = {"n_spins": len(H_list), "n_blocks": nblk, "n_ao_blk": [len(b) for b in blocks], "S": S, "blocks": blocks,
           "orthogonalization": orthogonalization, "kept_block": kept_block, "energies": {}, "couplings": {},
           "couplings_raw": {}, "state_overlaps": {}, "coeffs": {}, "coeffs_raw": {}}
    for sp, H in enumerate(H_list):
        H_m = data_conv.nparray2MATRIX(np.asarray(H, dtype=float))
        C_blk = []
        for f, blk in enumerate(blocks):
            nb = len(blk)
            Hff = MATRIX(nb, nb)
            pop_submatrix(H_m, Hff, ids[f], ids[f])
            Ht = X[f] * Hff * X[f]                                        # DGEMM
            E = MATRIX(nb, nb)
            Ct = MATRIX(nb, nb)
            _eigh_ascending(Ht, E, Ct)
            C_blk.append(X[f] * Ct)                                       # fragment eigenstates in the block's AOs
            out["energies"][(f, sp)] = np.diag(data_conv.MATRIX2nparray(E, np.float64)).copy()
        for i in range(nblk):
            for j in range(i + 1, nblk):
                Hij = MATRIX(len(blocks[i]), len(blocks[j]))
                pop_submatrix(H_m, Hij, ids[i], ids[j])
                Sij = MATRIX(len(blocks[i]), len(blocks[j]))
                pop_submatrix(S_m, Sij, ids[i], ids[j])
                H_raw = data_conv.MATRIX2nparray(C_blk[i].T() * Hij * C_blk[j], np.float64)
                S_st = data_conv.MATRIX2nparray(C_blk[i].T() * Sij * C_blk[j], np.float64)
                out["couplings_raw"][(i, j, sp)] = H_raw
                out["state_overlaps"][(i, j, sp)] = S_st
                if orthogonalization == "none":
                    out["couplings"][(i, j, sp)] = H_raw
                else:
                    keep_first = (i == kept_block) or (j != kept_block)
                    out["couplings"][(i, j, sp)] = _pairwise_couplings(H_raw, S_st, out["energies"][(i, sp)],
                                                                       out["energies"][(j, sp)], orthogonalization, keep_first)
        c_raw = np.zeros((n, n))
        for f, blk in enumerate(blocks):
            c_raw[np.ix_(blk, cols[f])] = data_conv.MATRIX2nparray(C_blk[f], np.float64)
            out["coeffs_raw"][(f, sp)] = c_raw[:, cols[f]].copy()
        c_orth = c_raw if orthogonalization == "none" else _orthonormalize_blocks(c_raw, S_m, cols, orthogonalization, kept_block)
        for f in range(nblk):
            out["coeffs"][(f, sp)] = np.ascontiguousarray(c_orth[:, cols[f]])
    return out


def pod_fragment_operators(pod, C_win, spin=0):
    """
    The POD fragment projectors over an MO window - the analogue of the Mulliken fragment
    operators: P_f = X_f X_f^T with X_f = c_f^T S C_win^T, c_f the block MOs of fragment f.
    The blocks partition the Lowdin-orthogonalized AO space, so the projectors sum to
    C_win S C_win^T (the window's MO overlap).

    Args:
        pod (dict): pod_block_states or read_pod_cp2k_files output
        C_win (array (nwin, n_ao)): window MO coefficients in the same AO order
        spin (int): the spin channel of the block MOs

    Returns:
        list over blocks of arrays (nwin, nwin)
    """
    S_m = data_conv.nparray2MATRIX(np.asarray(pod["S"], dtype=float))
    Cw = data_conv.nparray2MATRIX(np.asarray(C_win, dtype=float))
    SC = S_m * Cw.T()                                                        # (n_ao, nwin), DGEMM
    out = []
    for ib in range(pod["n_blocks"]):
        cf = data_conv.nparray2MATRIX(np.asarray(pod["coeffs"][(ib, spin)], dtype=float))
        X = cf.T() * SC                                                      # (n_block, nwin)
        P = data_conv.MATRIX2nparray(X.T() * X, np.float64)                  # (nwin, nwin)
        out.append(np.ascontiguousarray(0.5 * (P + P.T)))
    return out


def pod_time_overlaps(pod_prev, pod_curr, S12, spin=0):
    """
    The time overlaps of the POD block states between two geometries, <s(prev)|s'(curr)> = c_prev^T S12 c_curr
    with S12 the AO overlap between the previous and the current geometry (the S(prev, curr) block of the
    OVERLAP_DELTAT run, in the coefficients' AO order), over all blocks in block order - the matrix that carries
    a block state from one frame to the next: its signs follow the arbitrary sign of every printed state, its
    antisymmetric part is the derivative coupling of the block states.

    Args:
        pod_prev, pod_curr (dict): read_pod_cp2k_files or pod_block_states outputs of the two frames
        S12 (array (n_ao, n_ao)): the AO time overlap
        spin (int): the spin channel of the block states

    Returns:
        array (n_states, n_states): rows the previous frame's block states, columns the current frame's, both in
        block order (block 0's states, then block 1's, ...)
    """
    if pod_prev["n_blocks"] != pod_curr["n_blocks"]:
        raise ValueError(F"{pod_prev['n_blocks']} POD blocks at the previous frame, {pod_curr['n_blocks']} at "
                         "the current")
    C_prev = np.concatenate([np.asarray(pod_prev["coeffs"][(ib, spin)], dtype=float)
                             for ib in range(pod_prev["n_blocks"])], axis=1)
    C_curr = np.concatenate([np.asarray(pod_curr["coeffs"][(ib, spin)], dtype=float)
                             for ib in range(pod_curr["n_blocks"])], axis=1)
    Cp = data_conv.nparray2MATRIX(np.ascontiguousarray(C_prev.T))                 # (n_states, n_ao)
    Cc = data_conv.nparray2MATRIX(np.ascontiguousarray(C_curr.T))
    S_m = data_conv.nparray2MATRIX(np.asarray(S12, dtype=float))
    return data_conv.MATRIX2nparray((Cp * S_m) * Cc.T(), np.float64)


def ks_matrix_from_mos(C, E, S, tol=1e-6):
    """
    The AO Kohn-Sham matrix from a COMPLETE MO set: H = S C^T diag(E) C S (the KS equations
    H C^T = S C^T E with C S C^T = 1). An incomplete set (fewer MOs than AOs) cannot
    reproduce H - raise instead of returning the projected matrix.

    Args:
        C (array (nmo, n_ao)): MO coefficients (rows), E (array (nmo,)): their energies,
        S (array (n_ao, n_ao)): the AO overlap
    """
    C = np.asarray(C, dtype=float)
    if C.shape[0] != C.shape[1]:
        raise ValueError(F"ks_matrix_from_mos needs the complete MO set: {C.shape[0]} MOs for "
                         F"{C.shape[1]} AOs (ADDED_MOS = n_ao - n_occ), or print the Kohn-Sham matrix")
    n = C.shape[0]
    C_m = data_conv.nparray2MATRIX(C)
    S_m = data_conv.nparray2MATRIX(np.asarray(S, dtype=float))
    SCt = S_m * C_m.T()                                                      # (n_ao, nmo), DGEMM
    ortho = data_conv.MATRIX2nparray(C_m * SCt, np.float64)
    dev = float(np.abs(ortho - np.eye(n)).max())
    if dev > tol:
        raise ValueError(F"the MOs are not orthonormal in S (|C S C^T - 1| = {dev:.1e})")
    E_m = MATRIX(n, n)
    set_from_buffer(E_m, np.ascontiguousarray(np.diag(np.asarray(E, dtype=float))))
    return data_conv.MATRIX2nparray(SCt * E_m * SCt.T(), np.float64)         # S C^T E C S
