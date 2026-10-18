"""
The carriers of electronic states relative to a reference state, as operators over the states, for the fragment
diabatic basis of the on-the-fly drivers: from the one-particle transition densities of the states per spin (the MO
basis of the step) and the fragment operators of the step (Mulliken, libra's POD or CP2K's POD prints, in the same MO
basis). State 0 is the reference: its natural orbitals split each spin's MO space into an occupied and a virtual part,
the transition densities less state 0's own are projected on them - the occupied part with the sign reversed is the
hole, the virtual part the electron. The spin-flip mode (packages.cp2k.spin_flip) and the PySCF driver (packages.pyscf)
read them.
"""
import numpy as np


def carrier_operators_from_densities(gamma_a, gamma_b, P_a, P_b, occ_threshold=0.5):
    """The hole and electron fragment operators of the excited states relative to state 0, over the states: the natural
    orbitals of state 0 per spin split every MO space into its occupied part (occupation above occ_threshold) and its
    virtual part, the transition densities less state 0's own (delta_kl gamma[0, 0]) are projected on them - the
    occupied part with the sign reversed is the hole (a removed electron counts positive), the virtual part the electron
    - and traced with the fragment operators P_a / P_b (lists over fragments of (nmo, nmo) in the respective spin's MO
    basis). The row and column of state 0 are zero, the operators of a fragment are symmetric, and the hole operators of
    all fragments sum to the number of electrons an excited state removes from the occupied space of state 0 (one for a
    single excitation relative to it, about two for a double).

    Args:
        gamma_a, gamma_b (arrays (nstates, nstates, nmo, nmo)): the one-particle transition densities per spin, with
            tr(gamma[k, l] O) the matrix element of a one-body operator O between the states
        P_a, P_b (lists over fragments of arrays (nmo, nmo)): the fragment operators per spin
        occ_threshold (float): the natural occupation that separates state 0's occupied from its virtual space

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


def single_excitation_rows(O_hole, double_threshold=1.5):
    """The excited states (rows 1..) that remove fewer than double_threshold electrons from state 0's occupied space -
    the single excitations - and the others (the doubles and higher), from the diagonal of the summed hole operators."""
    holes = sum(np.asarray(O, dtype=float) for O in O_hole).diagonal()
    singles = [k for k in range(1, holes.size) if holes[k] < double_threshold]
    doubles = [k for k in range(1, holes.size) if holes[k] >= double_threshold]
    return singles, doubles, holes


def single_excitation_manifold(O_hole, n_single, tol=0.25):
    """The single-excitation manifold of the excited states (rows 1..): the n_single eigenvectors of the summed hole
    operator N_h over the excited states whose eigenvalues lie nearest one - the combinations of the states that remove
    one electron from state 0's occupied space, whichever adiabatic states the correlation mixes them into (a single
    mixed with a double counts 1.3 - 1.4 holes by row and the pair un-mixes here). The manifold is refused unless every
    chosen eigenvalue lies within tol of one and every other eigenvalue farther than tol from one - the gap that makes
    it a definite subspace.

    Args:
        O_hole (list over fragments of arrays (nstates, nstates)): the hole operators of carrier_operators_from_densities
        n_single (int): the number of single excitations - the occupied times the virtual orbitals of the active space
        tol (float): the allowed distance of a manifold state's hole count from one (occupancy_tolerance)

    Returns:
        W (array (nstates, n_single)): the orthonormal manifold over the states, row 0 zero, hole_in (array (n_single,)):
        the hole counts of its states, hole_out (array): the hole counts of the complement's excited states
    """
    n = np.asarray(O_hole[0]).shape[0]
    exc = np.arange(1, n)
    if not 0 < n_single <= exc.size:
        raise ValueError(F"single_excitation_manifold: {n_single} single excitations among {exc.size} excited states")
    Nh = sum(np.asarray(O, dtype=float)[np.ix_(exc, exc)] for O in O_hole)
    Nh = 0.5 * (Nh + Nh.T)
    vals, vecs = np.linalg.eigh(Nh)
    order = np.argsort(np.abs(vals - 1.0))
    chosen, rest = order[:n_single], order[n_single:]
    dev_in = float(np.abs(vals[chosen] - 1.0).max())
    dev_out = float(np.abs(vals[rest] - 1.0).min()) if rest.size else np.inf
    if dev_in > tol or dev_out <= tol:
        raise ValueError(F"the single-excitation manifold of {n_single} states is not definite within {tol}: the hole counts "
                         F"nearest one are {np.round(np.sort(vals[chosen]), 3).tolist()}, the next "
                         F"{np.round(vals[rest][np.argsort(np.abs(vals[rest] - 1.0))][:2], 3).tolist()} - a manifold state "
                         F"farther than {tol} from one hole or a complement state within it (occupancy_tolerance up to 0.5, "
                         F"or a state average that holds the whole manifold)")
    W = np.zeros((n, n_single))
    W[1:, :] = vecs[:, chosen]
    return W, vals[chosen], vals[rest]


def double_excitation_manifold(O_hole, W_single, tol=0.25):
    """The double-excitation manifold of the excited states: the eigenvectors of the summed hole operator N_h within the
    orthogonal complement of the single-excitation manifold W_single (over the excited states, rows 1..) whose
    eigenvalues lie within tol of two - the combinations that remove two electrons from state 0's occupied space,
    whichever adiabatic states the correlation mixes them into. The other complement states (triples and higher, a
    state between the integers) are not part of it and stay energy eigenstates of the complement.

    Args:
        O_hole (list over fragments of arrays (nstates, nstates)): the hole operators of carrier_operators_from_densities
        W_single (array (nstates, n_single)): the single-excitation manifold (single_excitation_manifold)
        tol (float): the allowed distance of a manifold state's hole count from two (occupancy_tolerance)

    Returns:
        W (array (nstates, n_double)): the orthonormal manifold over the states, row 0 zero, hole_in (array
        (n_double,)): the hole counts of its states, hole_out (array): the hole counts of the remaining complement states
    """
    n = np.asarray(O_hole[0]).shape[0]
    W_single = np.asarray(W_single, dtype=float)
    Nh = sum(np.asarray(O, dtype=float) for O in O_hole)
    Nh = 0.5 * (Nh + Nh.T)
    # the complement of the singles among the excited states: the eigenvectors of the projector with eigenvalue one
    Q = np.eye(n) - W_single @ W_single.T
    Q[0, :] = 0.0
    Q[:, 0] = 0.0
    qv, qvec = np.linalg.eigh(0.5 * (Q + Q.T))
    C = qvec[:, qv > 0.5]
    if C.shape[1] == 0:
        return np.zeros((n, 0)), np.zeros(0), np.zeros(0)
    vals, vecs = np.linalg.eigh(C.T @ Nh @ C)
    chosen = np.abs(vals - 2.0) <= tol
    W = C @ vecs[:, chosen]
    return W, vals[chosen], vals[~chosen]
