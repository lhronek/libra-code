"""The spin-flip TDA states of a high-spin reference as the CP2K driver reads them: the eigenvectors of the TDDFPT
restart file (the AO-basis response of the alpha occupied orbitals - the NAMD_PRINT eigenvectors are projected on the
alpha virtuals by CP2K and are not usable for a spin flip), their expansion over the beta virtual MOs, the spin of
every root from S_+, and the determinant basis of the flipped states for the overlap kernels.

A spin-flip root is Psi = sum_{i a} X[a, i] |Phi_0 with the alpha electron of occupied orbital i moved into the beta
virtual orbital a> - a state of Ms = S_ref - 1 of a reference with S_ref = (n_alpha - n_beta) / 2 (Ms = 0 from a
triplet, Ms = 1/2 from a quartet). Its spin is not fixed: the low-spin partners (S = S_ref - 1: the singlets of a
triplet reference, the doublets of a quartet), the Ms components of the reference's own multiplicity and higher ones
and, with a single excitation space, spin-incomplete mixtures all come out, and <S^2> tells them apart: <S^2> =
||S_+ Psi||^2 + Ms (Ms + 1), S_+ moving a beta electron in orbital b into every alpha orbital p the determinant leaves
empty with the amplitude <alpha_p|beta_b> (the alpha MOs span the AO space, so the expansion is complete). The
determinants S_+ Psi lands on carry alpha orbitals of the alpha set alone and beta orbitals of the beta set alone, so
they are orthonormal and the norm is the sum of the squared coefficients over the distinct occupation patterns. The
excess ||S_+ Psi||^2 is zero for a low-spin partner and 2 S_ref for the reference's own component, whatever S_ref."""
import struct

import numpy as np


def read_tdwfn(path):
    """CP2K's TDDFPT restart (RESTART.tdwfn, qs_tddfpt2_restart.F): the records (nstates, nspins, nao), nmo_active per
    spin, the eigenvalues (Hartree) and, state by state and spin by spin, the eigenvector as cp_fm_write_unformatted
    writes it - one Fortran record per column, nao doubles each. Returns a dict with nstates, nspins, nao, nmo_active
    (list), evals (ndarray) and evects (list over states of lists over spins of (nao, nmo_active) arrays)."""
    with open(path, "rb") as fh:
        data = fh.read()
    pos = 0

    def record():
        nonlocal pos
        n = struct.unpack("<i", data[pos:pos + 4])[0]
        if pos + 8 + n > len(data) or struct.unpack("<i", data[pos + 4 + n:pos + 8 + n])[0] != n:
            raise ValueError(F"{path}: a Fortran record of {n} bytes at offset {pos} is not closed by its length")
        out = data[pos + 4:pos + 4 + n]
        pos += 8 + n
        return out
    nstates, nspins, nao = struct.unpack("<3i", record())
    nmo = list(struct.unpack(F"<{nspins}i", record()))
    evals = np.frombuffer(record(), dtype="<f8").copy()
    if evals.size != nstates:
        raise ValueError(F"{path}: {evals.size} eigenvalues for {nstates} states")
    evects = []
    for _ in range(nstates):
        per_spin = []
        for ispin in range(nspins):
            cols = [np.frombuffer(record(), dtype="<f8") for _ in range(nmo[ispin])]
            if any(c.size != nao for c in cols):
                raise ValueError(F"{path}: an eigenvector column is not {nao} long")
            per_spin.append(np.column_stack(cols).copy())
        evects.append(per_spin)
    if pos != len(data):
        raise ValueError(F"{path}: {len(data) - pos} bytes beyond the last eigenvector record")
    return {"nstates": nstates, "nspins": nspins, "nao": nao, "nmo_active": nmo, "evals": evals, "evects": evects}


def check_restart_orbitals(E_alpha_occ, nmo, nao, wd, gap_min=1e-5):
    """
    The TDDFPT restart stores the responses for the TDDFPT's own occupied orbitals. With every MO in the wfn (nmo = nao)
    tddfpt_init_ground_state_mos keeps the SCF orbitals (its do_eigen is false), so a degenerate pair keeps the wfn's
    rotation and the restart's columns are the wfn's. With fewer MOs it re-diagonalizes the Kohn-Sham matrix, and the
    rotation within a degenerate alpha occupied pair is lost - refused.
    """
    gaps = np.diff(np.sort(np.asarray(E_alpha_occ, dtype=float)))
    if nmo < nao and gaps.size and gaps.min() < gap_min:
        raise ValueError(F"{wd}: the wfn holds {nmo} of {nao} MOs, so the TDDFPT re-diagonalized its orbitals, and two alpha "
                         F"occupied orbitals lie {gaps.min():.1e} Ha apart - the responses' rotation within that pair is not "
                         F"the wfn's")


def cp2k_mo_phases(C):
    """CP2K's phase of every MO row of C (nmo, nao) in its own AO order, as tddfpt_init_ground_state_mos defines it
    for the TDDFPT restart: +1 when more expansion coefficients are positive than negative (|c| at or above the
    machine epsilon), -1 when more are negative, and for a tie +1 when the first positive coefficient precedes the
    first negative one. The restart writer scales every response column by the phase of its alpha occupied orbital,
    so the stored responses refer to orbitals of phase +1."""
    C = np.asarray(C, dtype=float)
    eps = np.finfo(float).eps
    phases = np.ones(C.shape[0])
    for i, row in enumerate(C):
        signs = np.where(row >= eps, 1, np.where(row <= -eps, -1, 0))
        total = int(signs.sum())
        if total > 0:
            continue
        if total < 0:
            phases[i] = -1.0
            continue
        pos, neg = np.flatnonzero(signs > 0), np.flatnonzero(signs < 0)
        first_pos = pos[0] if pos.size else C.shape[1]
        first_neg = neg[0] if neg.size else C.shape[1]
        phases[i] = 1.0 if first_pos <= first_neg else -1.0
    return phases


def spin_flip_amplitudes(evect, C_beta, S_ao, n_beta_occ, C_alpha_occ):
    """The amplitudes X[a, i] of one spin-flip root over the beta virtual MOs (a) and the alpha occupied orbitals (i)
    from its AO-basis response evect (nao, n_alpha_occ): the projection onto the beta MOs C_beta (nmo, nao) through the
    AO overlap S_ao, every matrix in one AO order. X[a, i] is the amplitude of a^+_{a beta} a_{i alpha} |0>, the
    determinant of spin_flip_sd_basis with the beta orbital in the slot of i, with one sign per column: the restart
    stores the response of every alpha occupied orbital for the orbital's phase +1 (cp2k_mo_phases), so the column
    takes the phase of the given alpha occupied orbital C_alpha_occ[i] (the orbitals the determinants are built from).
    The spin routines, derived over sorted determinants, take the slot parity themselves (s2_spin_flip). Also returned:
    the weight the projection leaves on the beta occupied orbitals (zero for a response that lives in the beta virtual
    space, as CP2K's does) and the weight the beta MOs carry in all (1 when they span the AO space - ADDED_MOS must
    cover every virtual)."""
    proj = np.asarray(C_beta, dtype=float) @ np.asarray(S_ao, dtype=float) @ np.asarray(evect, dtype=float)
    proj = proj * cp2k_mo_phases(C_alpha_occ)[np.newaxis, :]
    return proj[n_beta_occ:], float(np.sum(proj[:n_beta_occ] ** 2)), float(np.sum(proj ** 2))


def _s2_landing(X, n_alpha_occ, n_beta_occ, S_ab):
    """The landing patterns of S_+ on the state X (the listed amplitudes, s2_spin_flip): the coefficient of the
    reference pattern, the beta patterns of p = i (nvirt, n_beta_occ), the alpha patterns of b = a (n_alpha_occ,
    nmo_a, zero where p is occupied or p = i), and for the patterns unique to their term the amplitudes (n_alpha_occ,
    nvirt) with the weight sum over the empty p != i and the occupied b of S_ab[p, b]^2 per alpha orbital - two states
    share every pattern, so their bilinear form is the sum of the products (s2_spin_flip_matrix)"""
    X = np.asarray(X, dtype=float)
    S_ab = np.asarray(S_ab, dtype=float)
    nvirt, nocc = X.shape
    if nocc != n_alpha_occ:
        raise ValueError(F"X has {nocc} occupied columns, the reference {n_alpha_occ} alpha occupied orbitals")
    # the listed amplitude enters the sorted lists with the slot parity (the docstring of s2_spin_flip)
    X = X * ((-1.0) ** (n_alpha_occ - 1 - np.arange(n_alpha_occ)))[np.newaxis, :]
    nmo_a = S_ab.shape[0]
    alpha_occ = np.arange(n_alpha_occ)
    beta_occ = np.arange(n_beta_occ)
    virt = n_beta_occ + np.arange(nvirt)                    # the beta orbital index of every virtual a
    # the parity of the alpha insertion: p into the sorted list alpha_occ without i - the count of its members below p
    p_all = np.arange(nmo_a)
    below = (p_all[None, :] > alpha_occ[:, None]).sum(axis=0)            # members of alpha_occ below p
    # members of alpha_occ \\ {i} below p = below[p] - (i < p)
    count_p = below[None, :] - (alpha_occ[:, None] < p_all[None, :])     # (i, p)
    par_p = np.where(count_p % 2 == 0, 1.0, -1.0)                        # (i, p)
    empty = np.ones((n_alpha_occ, nmo_a), dtype=bool)                    # p is empty in the determinant of term i
    empty[:, :n_alpha_occ] = False
    empty[alpha_occ, alpha_occ] = True                                   # p = i is empty again
    n_alpha_left = n_alpha_occ - 1
    # the beta list of term a: beta_occ + {virt[a]} sorted - the virtual sits above every occupied, so its position
    # is n_beta_occ and the occupied b keeps its own position
    par_b_occ = np.where((n_alpha_left + beta_occ) % 2 == 0, 1.0, -1.0)  # (b,) for b in beta_occ
    par_b_virt = 1.0 if (n_alpha_left + n_beta_occ) % 2 == 0 else -1.0   # b = the virtual a itself
    S_occ = S_ab[:, :n_beta_occ]                                         # (p, b) for b in beta_occ
    S_virt = S_ab[:, virt]                                               # (p, a)
    # (1) p = i and b = a: the reference pattern - one coefficient over all terms
    c_ref = sum(X[a, i] * par_p[i, i] * par_b_virt * S_virt[i, a] for i in range(n_alpha_occ) for a in range(nvirt))
    # (2) p = i, b in beta_occ: the alpha reference with the beta pattern (a, b) - summed over i
    c2 = np.einsum("ai,i,ib->ab", X, par_p[alpha_occ, alpha_occ], S_occ[alpha_occ, :]) * par_b_occ[None, :]
    # (3) b = a, p != i and empty: the beta reference with the alpha pattern (i, p) - summed over a
    c3 = np.einsum("ai,pa->ip", X, S_virt) * par_p * par_b_virt
    c3 = np.where(empty, c3, 0.0)
    c3[alpha_occ, alpha_occ] = 0.0
    # (4) p != i, b != a: every term its own pattern, with the weight of its landings
    sp = np.where(empty, 1.0, 0.0)
    sp[alpha_occ, alpha_occ] = 0.0                                       # p != i
    s4 = (sp @ (S_occ ** 2)).sum(axis=1)                                 # (i,): sum over empty p != i and b of S_ab[p, b]^2
    return float(c_ref), c2, c3, X.T, s4


def s2_spin_flip(X, n_alpha_occ, n_beta_occ, S_ab):
    """||S_+ Psi||^2 of the spin-flip state X[a, i] (beta virtual a, alpha occupied i - the orbitals 0-based, the alpha
    occupied the first n_alpha_occ of the alpha set, the beta occupied the first n_beta_occ of the beta set, the beta
    virtual a the beta orbital n_beta_occ + a) with S_ab = C_alpha S C_beta^T over ALL MOs of both spins - the excess
    <S^2> - Ms (Ms + 1), which is <S^2> itself for the Ms = 0 states of a triplet reference (s2_total adds the rest). The
    determinant of the term (i, a) lists the alpha orbitals first, then the beta ones, each sorted, so the amplitude
    X[a, i] of the listed determinant (the beta orbital in the slot of i, spin_flip_amplitudes) enters with the parity
    (-1)^(n_alpha_occ - 1 - i) of moving that orbital past the alpha orbitals above the slot - its place among the beta
    orbitals costs every term the same sign, which the squares cancel. S_+ takes the beta
    electron of orbital b into the empty alpha orbital p with the coefficient S_ab[p, b] and the parity of removing b
    from the sorted beta list and inserting p into the sorted alpha list. Four kinds of landing pattern: p = i and b
    = a restore the reference pattern (one pattern for all terms), p = i alone leaves the alpha reference with the
    beta pattern of (a, b), b = a alone the beta reference with the alpha pattern of (i, p), and p != i, b != a is
    unique to its term - the coefficients of a shared pattern add before squaring (_s2_landing)."""
    c_ref, c2, c3, Xs, s4 = _s2_landing(X, n_alpha_occ, n_beta_occ, S_ab)
    return float(c_ref ** 2 + np.sum(c2 ** 2) + np.sum(c3 ** 2) + np.sum(Xs ** 2 * s4[:, None]))


def s2_spin_flip_matrix(X_list, n_alpha_occ, n_beta_occ, S_ab):
    """The excess-<S^2> matrix over the states X_list: <S_+ Psi_k | S_+ Psi_l> by the landing patterns of _s2_landing,
    every pattern's coefficients of the two states multiplied (the diagonal is s2_spin_flip) - s2_total on the
    diagonal gives <S^2>, the off-diagonal elements the spin mixing of the roots (purify_spin_mixtures)."""
    parts = [_s2_landing(X, n_alpha_occ, n_beta_occ, S_ab) for X in X_list]
    n = len(parts)
    M = np.zeros((n, n))
    for k in range(n):
        c_ref_k, c2_k, c3_k, Xs_k, s4 = parts[k]
        for l in range(k, n):
            c_ref_l, c2_l, c3_l, Xs_l, _ = parts[l]
            M[k, l] = M[l, k] = (c_ref_k * c_ref_l + np.sum(c2_k * c2_l) + np.sum(c3_k * c3_l)
                                 + np.sum(Xs_k * Xs_l * s4[:, None]))
    return M


def _s2_patterns(X, n_alpha_occ, n_beta_occ, S_ab, threshold=1e-12):
    """The landing patterns of S_+ on the state X, every pattern (the sorted alpha set, the sorted beta set) with its
    coefficient - the explicit form the closed one is checked against, the listed amplitude entering with the slot
    parity of the sorted lists (s2_spin_flip)."""
    X = np.asarray(X, dtype=float)
    S_ab = np.asarray(S_ab, dtype=float)
    nvirt, nocc = X.shape
    X = X * ((-1.0) ** (n_alpha_occ - 1 - np.arange(n_alpha_occ)))[np.newaxis, :]
    coeff = {}
    for i in range(n_alpha_occ):
        A = [o for o in range(n_alpha_occ) if o != i]
        for a in range(nvirt):
            x = X[a, i]
            if abs(x) < threshold:
                continue
            B = list(range(n_beta_occ)) + [n_beta_occ + a]
            for b_pos, b in enumerate(B):
                sign_b = -1.0 if (len(A) + b_pos) % 2 else 1.0
                B2 = tuple(B[:b_pos] + B[b_pos + 1:])
                for p in range(S_ab.shape[0]):
                    if p in A or abs(S_ab[p, b]) < threshold:
                        continue
                    sign_p = -1.0 if sum(1 for o in A if o < p) % 2 else 1.0
                    key = (tuple(sorted(A + [p])), B2)
                    coeff[key] = coeff.get(key, 0.0) + x * sign_b * sign_p * S_ab[p, b]
    return coeff


def s2_spin_flip_brute(X, n_alpha_occ, n_beta_occ, S_ab, threshold=1e-12):
    """The same ||S_+ Psi||^2 by the explicit landing patterns of every term in a dictionary - the reference the closed
    form is checked against."""
    return float(sum(v * v for v in _s2_patterns(X, n_alpha_occ, n_beta_occ, S_ab, threshold).values()))


def s2_spin_flip_brute_matrix(X_list, n_alpha_occ, n_beta_occ, S_ab, threshold=1e-12):
    """The excess-<S^2> matrix over the states by the explicit landing patterns - the reference of s2_spin_flip_matrix."""
    dicts = [_s2_patterns(X, n_alpha_occ, n_beta_occ, S_ab, threshold) for X in X_list]
    n = len(dicts)
    M = np.zeros((n, n))
    for k in range(n):
        for l in range(k, n):
            M[k, l] = M[l, k] = sum(v * dicts[l].get(key, 0.0) for key, v in dicts[k].items())
    return M


def purify_spin_mixtures(X_list, omegas, s2_matrix, window, *, coupling_min, coupling_ramp, s2_tolerance, window_fade):
    """The roots whose spin mixing the Davidson left in (a degenerate singlet and triplet pair delivered as two mixtures,
    a singlet crossing a spin-incomplete root, S0 and the triplet mixing at the twist, the V character spread over three
    roots on the hot ground state) rotated to the eigenvectors of their excess-<S^2> block, group by group: two roots are
    coupled when they lie within window of each other and |s2_matrix[k, l]| exceeds coupling_min (the driver's
    spin_purify_coupling_min) - whatever their classes, a
    pair mixing below the class thresholds (a "singlet" at 0.4 with a "triplet" at 1.6) included - a group is the
    strongest coupled pair extended by every root coupled to all of its members (a clique of the coupling graph), the
    groups are taken by decreasing coupling of their first pair, a root in one group at most, and every group is
    rotated, however small the mixing: a rotation threshold on the weight makes the energy jump by the weight times the
    gap when the mixing crosses it, while the rotation itself is continuous in the geometry - the force floor on the
    weights is the driver's (spin_purify_min_weight). A coupling enters its group's block damped, scaled by the cubic
    step from 0 at coupling_min to 1 at coupling_min + coupling_ramp (spin_purify_coupling_ramp), so that a root
    crossing the threshold joins or leaves with no step in any state's energy or amplitudes (the hard switch stepped them
    by the root's weight, (coupling_min / the excess gap)^2 - a ramp of 0 keeps the hard switch). A coupling fades out
    towards the window's edge as well, scaled
    by the cubic step over the last window_fade of the window (the driver's spin_purify_window_fade - 0.5 eV at a 2 eV
    window: from 1.5 to 2 eV), so that a pair crossing the edge forms or dissolves with no step either (0 keeps the hard
    edge). Two eigenvectors of one spin
    cannot be told apart by S^2 - a
    group holding a second singlet-like root beside V would hand V an arbitrary share of it - so within every run of
    eigenvalues closer than s2_tolerance (spin_purify_s2_tolerance) the energy is diagonalized and a group's singlet
    subspace is spanned by its energy eigenstates (a tolerance of 0 keeps the plain eigenvectors). A clique, not a connected
    component: a chain of weak couplings
    would spread the lowest eigenvector over roots far above and move its energy by tenths of an eV from step to step.
    Not pairs alone: three mutually coupled roots sharing one singlet leave a pair's eigenvector contaminated and flip
    the pairing when the strongest coupling changes hands (the V state at an excess 0.34, its time overlap 0.66 at the
    flip). A state is a dict with X (the rotated amplitudes), omega (the expectation value sum_k U_kj^2 omegas[k]), s2
    (the excess eigenvalue), roots (the group's 0-based roots), weights (U_kj^2 over them) and group (the group's index,
    None for a root left alone). Returns the states in the order of their omega and the list of purified groups (roots,
    the excess eigenvalues, the weights) for the record."""
    omegas = np.asarray(omegas, dtype=float)
    n = len(X_list)

    def coupled(k, l):
        return abs(s2_matrix[k, l]) > coupling_min and abs(omegas[k] - omegas[l]) < window

    def onset(c):
        if coupling_ramp <= 0.0:
            return 1.0
        t = min(max((abs(c) - coupling_min) / coupling_ramp, 0.0), 1.0)
        return t * t * (3.0 - 2.0 * t)

    def fade(k, l):
        if window_fade <= 0.0:
            return 1.0
        t = min(max((window - abs(omegas[k] - omegas[l])) / window_fade, 0.0), 1.0)
        return t * t * (3.0 - 2.0 * t)
    edges = sorted(((abs(s2_matrix[k, l]), k, l) for k in range(n) for l in range(k + 1, n) if coupled(k, l)), reverse=True)
    used = set()
    groups = []
    for _, k, l in edges:
        if k in used or l in used:
            continue
        group = [k, l]
        # the unused roots coupled to every member join, the strongest first, each checked against the grown group
        joiners = sorted(((min(abs(s2_matrix[m, j]) for j in group), m) for m in range(n)
                          if m not in used and m not in group and all(coupled(m, j) for j in group)), reverse=True)
        for _, m in joiners:
            if all(coupled(m, j) for j in group):
                group.append(m)
        used.update(group)
        groups.append(sorted(group))
    states, purified = [], []

    def single(k):
        return {"X": np.asarray(X_list[k], dtype=float), "omega": float(omegas[k]), "s2": float(s2_matrix[k, k]),
                "roots": [k], "weights": [1.0], "group": None}
    for k in range(n):
        if k not in used:
            states.append(single(k))
    for group in groups:
        block = np.array([[s2_matrix[a, b] * (1.0 if a == b else onset(s2_matrix[a, b]) * fade(a, b)) for b in group] for a in group],
                         dtype=float)
        block = 0.5 * (block + block.T)
        vals, U = np.linalg.eigh(block)
        # the energy diagonalized within every run of near-degenerate S^2 eigenvalues
        H = np.diag(omegas[group])
        j = 0
        while j < len(group):
            k = j + 1
            while k < len(group) and vals[k] - vals[k - 1] < s2_tolerance:
                k += 1
            if k - j > 1:
                _, V = np.linalg.eigh(U[:, j:k].T @ H @ U[:, j:k])
                U[:, j:k] = U[:, j:k] @ V
            j = k
        s2_of = np.einsum("ij,ik,kj->j", U, block, U)
        gi = len(purified)
        for j in range(len(group)):
            w = U[:, j] ** 2
            X_new = sum(U[i, j] * np.asarray(X_list[r], dtype=float) for i, r in enumerate(group))
            states.append({"X": X_new, "omega": float(np.dot(w, omegas[group])), "s2": float(s2_of[j]),
                           "roots": list(group), "weights": [float(v) for v in w], "group": gi})
        purified.append({"roots": list(group), "s2": [float(v) for v in s2_of],
                         "weights": [[float(v) for v in U[:, j] ** 2] for j in range(len(group))]})
    states.sort(key=lambda s: s["omega"])
    return states, purified


SPIN_NAMES = {0: "singlet", 1: "doublet", 2: "triplet", 3: "quartet", 4: "quintet", 5: "sextet", 6: "septet", 7: "octet"}


def spin_name(S):
    """The multiplicity name of the total spin S (0, 1/2, 1, ...): singlet, doublet, triplet, ..."""
    twice = int(round(2.0 * float(S)))
    if abs(2.0 * float(S) - twice) > 1e-6 or twice < 0:
        raise ValueError(F"S = {S} is no total spin")
    return SPIN_NAMES.get(twice, F"{twice + 1}-et")


def s_reference(n_alpha_occ, n_beta_occ):
    """The total spin of the high-spin reference, (n_alpha - n_beta) / 2 - at least 1 for a spin flip."""
    if n_alpha_occ - n_beta_occ < 2:
        raise ValueError(F"{n_alpha_occ} alpha and {n_beta_occ} beta electrons - a spin-flip reference has at least two "
                         F"more alpha than beta electrons")
    return 0.5 * (n_alpha_occ - n_beta_occ)


def s2_total(excess, n_alpha_occ, n_beta_occ):
    """<S^2> of a spin-flip state from its excess ||S_+ Psi||^2 (s2_spin_flip): the states have Ms = S_ref - 1 and
    <S^2> = ||S_+ Psi||^2 + Ms (Ms + 1) - the excess itself for the Ms = 0 states of a triplet reference."""
    ms = s_reference(n_alpha_occ, n_beta_occ) - 1.0
    return float(excess) + ms * (ms + 1.0)


def classify_spin(excess, low_spin_max=0.5, high_spin_min=1.5, s_ref=1.0):
    """The class of a spin-flip root by its excess ||S_+ Psi||^2 (s2_spin_flip): below low_spin_max the low-spin partner of the
    reference (S = s_ref - 1: 'singlet' from a triplet reference, 'doublet' from a quartet), from high_spin_min on the
    reference's own multiplicity (S = s_ref, the excess of its exact Ms component is 2 s_ref: 'triplet', 'quartet'),
    'mixed' between - the spin-incomplete states of a single-excitation spin-flip space, which belong to no
    manifold. The thresholds are on the excess, so the same values serve every reference."""
    if excess < low_spin_max:
        return spin_name(s_ref - 1.0)
    if excess >= high_spin_min:
        return spin_name(s_ref)
    return "mixed"


def spin_flip_sd_basis(n_alpha_occ, n_beta_occ, nmo):
    """The determinants of the spin-flip space in the signed spin-orbital convention of the overlap kernels (+n the
    alpha orbital n, -n the beta orbital n, 1-based, one orbital window of nmo orbitals for both spins): the reference
    [+1 .. +n_alpha_occ, -1 .. -n_beta_occ] with +i replaced in place by -a for every alpha occupied i and beta virtual
    a - the listed order of a^+_{a beta} a_{i alpha} |Phi_0>, so the kernels' listed-order overlaps pair with the
    amplitudes X[a, i] as they are. Returns the list of determinants and the row of every (a, i) in it."""
    ref = list(range(1, n_alpha_occ + 1)) + [-j for j in range(1, n_beta_occ + 1)]
    sd_states, index = [], {}
    for i in range(n_alpha_occ):
        for a in range(nmo - n_beta_occ):
            sd = list(ref)
            sd[i] = -(n_beta_occ + a + 1)
            index[(a, i)] = len(sd_states)
            sd_states.append(sd)
    return sd_states, index


def spin_flip_sd2ci(X_list, index, nsd):
    """The amplitude matrix T (nsd, nstates) of the spin-flip roots over the determinant basis - no reference column,
    the reference is no state of the spin-flip manifold."""
    T = np.zeros((nsd, len(X_list)))
    for k, X in enumerate(X_list):
        for (a, i), row in index.items():
            T[row, k] = X[a, i]
    return T


def spin_flip_densities(X_list, n_alpha_occ, n_beta_occ, nmo):
    """The one-particle transition density matrices of the spin-flip states over the MO bases of the reference, per
    spin: gamma_a[k, l] = delta_kl 1_occ - X_k^T X_l on the alpha occupied block (zero on the alpha virtuals), gamma_b[k, l]
    = delta_kl 1_occ + X_l X_k^T on the beta virtual block (the beta occupied stay full), so that the matrix element of a
    one-body operator O of spin sigma between the states is tr(gamma_sigma[k, l] O). The Slater-Condon elements of the
    in-place determinants behind them: two flips with the same beta orbital differ in one alpha orbital and the swap that
    puts the shared beta orbital at one position costs a sign (the hole part enters with the minus), two flips with the
    same alpha orbital differ in the beta orbital at that position alone (the electron part enters with the plus) -
    one_body_matrix_brute is the reference.

    Args:
        X_list (list of arrays (nmo - n_beta_occ, n_alpha_occ)): the amplitudes of the states over (beta virtual a, alpha
            occupied i), the states orthonormal
        n_alpha_occ, n_beta_occ, nmo (int): the occupied counts of the reference and the MO count of each spin

    Returns:
        gamma_a, gamma_b (arrays (nstates, nstates, nmo, nmo))
    """
    n = len(X_list)
    X = [np.asarray(x, dtype=float) for x in X_list]
    for k, x in enumerate(X):
        if x.shape != (nmo - n_beta_occ, n_alpha_occ):
            raise ValueError(F"state {k}: X has the shape {x.shape}, the reference ({nmo - n_beta_occ}, {n_alpha_occ})")
    gamma_a = np.zeros((n, n, nmo, nmo))
    gamma_b = np.zeros((n, n, nmo, nmo))
    occ_a = np.arange(n_alpha_occ)
    occ_b = np.arange(n_beta_occ)
    virt_b = np.arange(n_beta_occ, nmo)
    for k in range(n):
        gamma_a[k, k, occ_a, occ_a] = 1.0
        gamma_b[k, k, occ_b, occ_b] = 1.0
        for l in range(n):
            gamma_a[k, l][np.ix_(occ_a, occ_a)] -= X[k].T @ X[l]
            gamma_b[k, l][np.ix_(virt_b, virt_b)] += X[l] @ X[k].T
    return gamma_a, gamma_b


def _listed_order_parity(det_from, det_to):
    """The sign of the permutation that reorders the spin orbitals of det_from into the order of det_to (the same set)."""
    pos = {m: j for j, m in enumerate(det_to)}
    perm = [pos[m] for m in det_from]
    sign = 1
    seen = [False] * len(perm)
    for start in range(len(perm)):
        if seen[start]:
            continue
        length = 0
        j = start
        while not seen[j]:
            seen[j] = True
            j = perm[j]
            length += 1
        if length % 2 == 0:
            sign = -sign
    return sign


def one_body_matrix_brute(sd_states, O_alpha, O_beta):
    """The matrix of a one-body spin operator over determinants in the signed spin-orbital notation of the kernels (+n the
    alpha orbital n, -n the beta orbital n, 1-based, listed order) by the Slater-Condon rules with the parity of the
    listed orders: the same set gives the sum over the occupied spin orbitals, one differing spin orbital the element
    between the two with the parity that aligns the lists, more differing orbitals zero. O_alpha / O_beta (nmo, nmo)
    over the spatial MOs, 0-based. The reference spin_flip_densities is checked against."""
    O_alpha = np.asarray(O_alpha, dtype=float)
    O_beta = np.asarray(O_beta, dtype=float)

    def elem(m1, m2):
        if (m1 > 0) != (m2 > 0):
            return 0.0
        O = O_alpha if m1 > 0 else O_beta
        return float(O[abs(m1) - 1, abs(m2) - 1])
    n = len(sd_states)
    sets = [frozenset(d) for d in sd_states]
    diag = [sum(elem(m, m) for m in d) for d in sd_states]
    M = np.zeros((n, n))
    for p in range(n):
        for q in range(n):
            only_p = sets[p] - sets[q]
            if len(only_p) == 0:
                M[p, q] = _listed_order_parity(sd_states[p], sd_states[q]) * diag[p]
            elif len(only_p) == 1:
                m1 = next(iter(only_p))
                m2 = next(iter(sets[q] - sets[p]))
                aligned = [m2 if m == m1 else m for m in sd_states[p]]
                M[p, q] = _listed_order_parity(aligned, sd_states[q]) * elem(m1, m2)
    return M


def spin_flip_carrier_operators(gamma_a, gamma_b, P_a, P_b, occ_threshold=0.5):
    """The hole and electron fragment operators of the excited spin-flip states relative to the lowest one (state 0),
    over the states - driver_fragments.carrier_operators_from_densities on the spin-flip transition densities
    (spin_flip_densities) and the fragment operators of both spins over the whole MO set. Returns O_hole, O_elec (lists
    over fragments of (nstates, nstates)) and the natural occupations of state 0 per spin."""
    from libra_py.packages.driver_fragments import carrier_operators_from_densities
    return carrier_operators_from_densities(gamma_a, gamma_b, P_a, P_b, occ_threshold)


def spin_flip_carrier_operators_high_spin(gamma_b, P_b, n_beta_occ, span_tol=0.25):
    """The electron and hole fragment operators of a spin-flip manifold relative to the HIGH-SPIN REFERENCE, over all the
    states (state 0 among them): the flipped beta electron is the carrier - its transition densities are the beta
    virtual blocks of gamma_b (gamma_b[k, l] - delta_kl 1_occ), traced with the beta fragment operators P_b (lists over
    fragments of (nmo, nmo) in the beta MO basis) - and the hole is its complement within the span of the manifold's
    flipped electrons (the top n eigenvectors of sum_k E_kk, n the number of states - a cation's hole states fill
    the pi manifold's beta orbitals one at a time, the hole is the one left empty): O_hole = delta_kl tr(P_F Pi) -
    O_elec. The alpha channel carries no carrier here (every state of the manifold removed the same alpha electron).
    The span must hold exactly n orbitals within span_tol (the eigenvalues above 1 - span_tol, the rest below span_tol)
    or the manifold is not one of single flips into a common orbital set and the construction refuses.

    Returns:
        O_hole, O_elec (lists over fragments of arrays (nstates, nstates)), span_values (the eigenvalues of
        sum_k E_kk, descending)
    """
    n = gamma_b.shape[0]
    nmo = gamma_b.shape[2]
    E = np.empty((n, n, nmo, nmo))
    occ_b = np.arange(n_beta_occ)
    for k in range(n):
        for l in range(n):
            E[k, l] = gamma_b[k, l]
            if k == l:
                E[k, l][occ_b, occ_b] -= 1.0
    total = sum(E[k, k] for k in range(n))
    vals, vecs = np.linalg.eigh(0.5 * (total + total.T))
    order = np.argsort(-vals)
    vals, vecs = vals[order], vecs[:, order]
    if vals[:n].min() < 1.0 - span_tol or (n < nmo and vals[n:].max() > span_tol):
        raise ValueError(F"the flipped electrons of the {n} states span no clean set of {n} beta orbitals - the eigenvalues "
                         F"of their summed densities {np.round(vals[:n + 2], 3).tolist()} (1 - {span_tol} .. 1 and below "
                         F"{span_tol} expected): the manifold is not one of single flips into a common orbital set")
    Pi = vecs[:, :n] @ vecs[:, :n].T
    O_hole, O_elec = [], []
    for P in P_b:
        P = 0.5 * (np.asarray(P, dtype=float) + np.asarray(P, dtype=float).T)
        Oe = np.array([[float(np.sum(P * E[k, l].T)) for l in range(n)] for k in range(n)])
        Oe = 0.5 * (Oe + Oe.T)
        Oh = np.eye(n) * float(np.sum(P * Pi.T)) - Oe
        O_elec.append(Oe)
        O_hole.append(Oh)
    return O_hole, O_elec, vals
