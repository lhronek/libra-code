"""The spin-flip TDA states of a high-spin reference as the CP2K driver reads them: the eigenvectors of the TDDFPT
restart file (the AO-basis response of the alpha occupied orbitals - the NAMD_PRINT eigenvectors are projected on the
alpha virtuals by CP2K and are not usable for a spin flip), their expansion over the beta virtual MOs, the spin of
every root from S_+, and the determinant basis of the flipped states for the overlap kernels.

A spin-flip root is Psi = sum_{i a} X[a, i] |Phi_0 with the alpha electron of occupied orbital i moved into the beta
virtual orbital a> - an Ms = 0 state of a reference with Ms = 1 (n_alpha = n_beta + 2). Its spin is not fixed: the
singlets, the Ms = 0 components of the triplets and quintets and, with a single excitation space, spin-incomplete
mixtures all come out, and <S^2> tells them apart: <S^2> = ||S_+ Psi||^2 for Ms = 0, S_+ moving a beta electron in
orbital b into every alpha orbital p the determinant leaves empty with the amplitude <alpha_p|beta_b> (the alpha MOs
span the AO space, so the expansion is complete). The determinants S_+ Psi lands on carry alpha orbitals of the alpha
set alone and beta orbitals of the beta set alone, so they are orthonormal and the norm is the sum of the squared
coefficients over the distinct occupation patterns."""
import struct

import numpy as np


def read_tdwfn(path):
    """CP2K's TDDFPT restart (RESTART.tdwfn, qs_tddfpt2_restart.F): the records (nstates, nspins, nao), nmo_active per
    spin, the eigenvalues (Hartree) and, state by state and spin by spin, the eigenvector as cp_fm_write_unformatted
    writes it - one Fortran record per column, nao doubles each. Returns a dict with nstates, nspins, nao, nmo_active
    (list), evals (ndarray) and evects (list over states of lists over spins of (nao, nmo_active) arrays)."""
    data = open(path, "rb").read()
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


def spin_flip_amplitudes(evect, C_beta, S_ao, n_beta_occ):
    """The amplitudes X[a, i] of one spin-flip root over the beta virtual MOs (a) and the alpha occupied orbitals (i)
    from its AO-basis response evect (nao, n_alpha_occ): the projection onto the beta MOs C_beta (nmo, nao) through the
    AO overlap S_ao, every matrix in one AO order. Also returned: the weight the projection leaves on the beta occupied
    orbitals (zero for a response that lives in the beta virtual space, as CP2K's does) and the weight the beta MOs
    carry in all (1 when they span the AO space - ADDED_MOS must cover every virtual)."""
    proj = np.asarray(C_beta, dtype=float) @ np.asarray(S_ao, dtype=float) @ np.asarray(evect, dtype=float)
    return proj[n_beta_occ:], float(np.sum(proj[:n_beta_occ] ** 2)), float(np.sum(proj ** 2))


def s2_spin_flip(X, n_alpha_occ, n_beta_occ, S_ab):
    """<S^2> of the Ms = 0 spin-flip state X[a, i] (beta virtual a, alpha occupied i - the orbitals 0-based, the alpha
    occupied the first n_alpha_occ of the alpha set, the beta occupied the first n_beta_occ of the beta set, the beta
    virtual a the beta orbital n_beta_occ + a) with S_ab = C_alpha S C_beta^T over ALL MOs of both spins. The
    determinant of the term (i, a) lists the alpha orbitals first, then the beta ones, each sorted. S_+ takes the beta
    electron of orbital b into the empty alpha orbital p with the coefficient S_ab[p, b] and the parity of removing b
    from the sorted beta list and inserting p into the sorted alpha list. Four kinds of landing pattern: p = i and b
    = a restore the reference pattern (one pattern for all terms), p = i alone leaves the alpha reference with the
    beta pattern of (a, b), b = a alone the beta reference with the alpha pattern of (i, p), and p != i, b != a is
    unique to its term - the coefficients of a shared pattern add before squaring."""
    X = np.asarray(X, dtype=float)
    S_ab = np.asarray(S_ab, dtype=float)
    nvirt, nocc = X.shape
    if nocc != n_alpha_occ:
        raise ValueError(F"X has {nocc} occupied columns, the reference {n_alpha_occ} alpha occupied orbitals")
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
    total = 0.0
    # (1) p = i and b = a: the reference pattern - one coefficient over all terms
    c_ref = sum(X[a, i] * par_p[i, i] * par_b_virt * S_virt[i, a] for i in range(n_alpha_occ) for a in range(nvirt))
    total += c_ref ** 2
    # (2) p = i, b in beta_occ: the alpha reference with the beta pattern (a, b) - summed over i
    c2 = np.einsum("ai,i,ib->ab", X, par_p[alpha_occ, alpha_occ], S_occ[alpha_occ, :]) * par_b_occ[None, :]
    total += float(np.sum(c2 ** 2))
    # (3) b = a, p != i and empty: the beta reference with the alpha pattern (i, p) - summed over a
    c3 = np.einsum("ai,pa->ip", X, S_virt) * par_p * par_b_virt
    c3 = np.where(empty, c3, 0.0)
    c3[alpha_occ, alpha_occ] = 0.0
    total += float(np.sum(c3 ** 2))
    # (4) p != i, b != a: every term its own pattern
    w4 = (X ** 2).T                                                      # (i, a)
    sp = np.where(empty, 1.0, 0.0)
    sp[alpha_occ, alpha_occ] = 0.0                                       # p != i
    sum_p = sp @ (S_occ ** 2)                                            # (i, b): sum over empty p != i of S_ab[p, b]^2
    total += float(np.sum(w4 * sum_p.sum(axis=1)[:, None]))
    return float(total)


def s2_spin_flip_brute(X, n_alpha_occ, n_beta_occ, S_ab, threshold=1e-12):
    """The same <S^2> by the explicit landing patterns of every term in a dictionary - the reference the closed form
    is checked against."""
    X = np.asarray(X, dtype=float)
    S_ab = np.asarray(S_ab, dtype=float)
    nvirt, nocc = X.shape
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
    return float(sum(v * v for v in coeff.values()))


def classify_spin(s2, singlet_max=0.5, triplet_min=1.5):
    """'singlet' below singlet_max, 'triplet' from triplet_min on (the Ms = 0 component of a triplet, 2), 'mixed'
    between - the spin-incomplete states of a single-excitation spin-flip space, which belong to no manifold."""
    if s2 < singlet_max:
        return "singlet"
    if s2 >= triplet_min:
        return "triplet"
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
    over the states: the natural orbitals of state 0 split every spin's MO space into its occupied part (occupation above
    occ_threshold) and its virtual part, the transition densities less state 0's own (delta_kl gamma[0, 0]) are projected
    on them - the occupied part with the sign reversed is the hole (a removed electron counts positive), the virtual part
    the electron - and traced with the fragment operators P_a / P_b (lists over fragments of (nmo, nmo) in the respective
    spin's MO basis: the POD projectors or the Mulliken operators of the step). The row and column of state 0 are zero,
    the operators of a fragment are symmetric, and the hole operators of all fragments sum to the number of electrons an
    excited state removes from the occupied space of state 0 (one for a single excitation relative to it).

    Returns:
        O_hole, O_elec (lists over fragments of arrays (nstates, nstates)), occupations (dict spin -> the natural
        occupations of state 0, ascending)
    """
    n = gamma_a.shape[0]
    if len(P_a) != len(P_b):
        raise ValueError(F"{len(P_a)} alpha and {len(P_b)} beta fragment operators")
    nfrag = len(P_a)
    O_hole = [np.zeros((n, n)) for _ in range(nfrag)]
    O_elec = [np.zeros((n, n)) for _ in range(nfrag)]
    occupations = {}
    for spin, gamma, P_list in (("alpha", gamma_a, P_a), ("beta", gamma_b, P_b)):
        rho0 = 0.5 * (gamma[0, 0] + gamma[0, 0].T)
        vals, vecs = np.linalg.eigh(rho0)
        occupations[spin] = vals
        occ = vecs[:, vals > occ_threshold]
        virt = vecs[:, vals <= occ_threshold]
        P_occ, P_virt = occ @ occ.T, virt @ virt.T
        for f, P in enumerate(P_list):
            P = 0.5 * (np.asarray(P, dtype=float) + np.asarray(P, dtype=float).T)
            A_hole = P_occ @ P @ P_occ          # tr(P P_occ D P_occ) = tr(A_hole D)
            A_elec = P_virt @ P @ P_virt
            for k in range(1, n):
                for l in range(1, n):
                    D = gamma[k, l] - (rho0 if k == l else 0.0)
                    O_hole[f][k, l] -= float(np.sum(A_hole * D.T))
                    O_elec[f][k, l] += float(np.sum(A_elec * D.T))
    O_hole = [0.5 * (O + O.T) for O in O_hole]
    O_elec = [0.5 * (O + O.T) for O in O_elec]
    return O_hole, O_elec, occupations
