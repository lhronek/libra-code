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
"""
import numpy as np
from liblibra_core import MATRIX, intList, solve_eigen, sqrt_matrix, pop_submatrix, have_lapack, lapack_syevd
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
    for i in idx:
        out.append(int(i))
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
            out["energies"][(f, sp)] = np.array([E.get(i, i) for i in range(nb)])
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
    SC = pod["S"] @ np.asarray(C_win).T
    out = []
    for ib in range(pod["n_blocks"]):
        X = pod["coeffs"][(ib, spin)].T @ SC
        P = X.T @ X
        out.append(np.ascontiguousarray(0.5 * (P + P.T)))
    return out


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
    dev = float(np.abs(C @ S @ C.T - np.eye(C.shape[0])).max())
    if dev > tol:
        raise ValueError(F"the MOs are not orthonormal in S (|C S C^T - 1| = {dev:.1e})")
    return S @ C.T @ np.diag(np.asarray(E, dtype=float)) @ C @ S
