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
