#***********************************************************
# * Copyright (C) 2026 Alexey V. Akimov
# * This file is distributed under the terms of the
# * GNU General Public License as published by the
# * Free Software Foundation; either version 3 of the
# * License, or (at your option) any later version.
# * http://www.gnu.org/copyleft/gpl.txt
#***********************************************************/
"""
Fixed single-particle determinant (SD) bases and their CI transforms,
shared by the on-the-fly drivers (CP2K, DFTB+): the window enumeration
(make_sd_basis), the SD -> CI transformation matrix with per-state
dropped weights (sd2ci_matrix), the chained state-sign alignment
(align_ci_state_signs), the couplings of a step taken after the states are
followed through it (followed_nac) and the capture contract of a diabatic SD
basis (check_sd_capture).
"""
import numpy as np


def make_sd_basis(homo_w, num_occ_states, num_unocc_states, norb_w,
                  homo_beta_w=None, num_occ_beta=None, num_unocc_beta=None):
    """
    Fixed single-particle SD basis for the orbital window.

    Excitations occ -> vir with occ in [homo_w-num_occ_states+1, homo_w] and
    vir in [homo_w+1, homo_w+num_unocc_states], all indices 1-based WITHIN the window.

    Restricted by default - passing homo_beta_w switches to the UKS basis:
    electron-only excitations per spin channel (alpha windowed around homo_w,
    beta around homo_beta_w, both over the same norb_w orbital window), with
    the beta excitation counts defaulting to the alpha ones. The determinants
    list alpha spin-orbitals as +i and beta as -i (the step2_many_body signed
    convention, same as reindex_cp2k_sd_states with both channels).

    Returns:
        sd_unique (list): [[occ, vir], 'alp'/'bet'] pairs (no ground state)
        sd_states (list): Libra signed-spin-orbital SDs, ground state FIRST

    Raises ValueError when a channel's window does not bracket its HOMO or the
    SD window exceeds the orbitals available in it - invalid indices would
    otherwise reindex into duplicate/garbage determinants (singular SD
    overlaps, NaN through everything downstream).
    """
    def _check(homo_c, no_c, nu_c, ch):
        occ_kw = "num_occ_states" if ch == "alp" else "num_occ_beta"
        vir_kw = "num_unocc_states" if ch == "alp" else "num_unocc_beta"
        lab = "" if ch == "alp" else "beta "
        if not 1 <= homo_c <= norb_w - 1:
            raise ValueError(F"the {lab}orbital window does not bracket the HOMO "
                             F"(window-relative HOMO={homo_c}, window size {norb_w}) - "
                             F"choose lowest_orbital <= HOMO < highest_orbital")
        if not 1 <= no_c <= homo_c:
            raise ValueError(F"{occ_kw}={no_c} but the {lab}orbital window holds only "
                             F"{homo_c} occupied orbitals - reduce {occ_kw} or lower "
                             F"lowest_orbital")
        if not 1 <= nu_c <= norb_w - homo_c:
            raise ValueError(F"{vir_kw}={nu_c} but the {lab}orbital window holds only "
                             F"{norb_w - homo_c} virtuals above the HOMO - reduce "
                             F"{vir_kw} or raise highest_orbital")

    _check(homo_w, num_occ_states, num_unocc_states, "alp")
    if homo_beta_w is None:
        sd_unique = []
        for o in range(homo_w - num_occ_states + 1, homo_w + 1):
            for v in range(homo_w + 1, homo_w + num_unocc_states + 1):
                sd_unique.append([[o, v], 'alp'])
        from libra_py.workflows.nbra import step2_many_body
        sd_states = step2_many_body.reindex_cp2k_sd_states(
            homo_w, list(range(1, norb_w + 1)), sd_unique, sd_format=2)
        return sd_unique, sd_states

    no_b = num_occ_states if num_occ_beta is None else num_occ_beta
    nu_b = num_unocc_states if num_unocc_beta is None else num_unocc_beta
    _check(homo_beta_w, no_b, nu_b, "bet")
    gs = []
    for i in range(1, max(homo_w, homo_beta_w) + 1):
        if i <= homo_w:
            gs.append(i)
        if i <= homo_beta_w:
            gs.append(-i)
    sd_unique, sd_states = [], [list(gs)]
    for o in range(homo_w - num_occ_states + 1, homo_w + 1):
        for v in range(homo_w + 1, homo_w + num_unocc_states + 1):
            sd_unique.append([[o, v], 'alp'])
            st = list(gs)
            st[st.index(o)] = v
            sd_states.append(st)
    for o in range(homo_beta_w - no_b + 1, homo_beta_w + 1):
        for v in range(homo_beta_w + 1, homo_beta_w + nu_b + 1):
            sd_unique.append([[o, v], 'bet'])
            st = list(gs)
            st[st.index(-o)] = -v
            sd_states.append(st)
    return sd_unique, sd_states


_severe_drop_warned = set()


def sd2ci_matrix(ci_data, sd_unique, lowest_orbital):
    """
    SD -> CI transformation matrix T (nSD+1 x nstates+1) on the FIXED SD basis.

    Column 0 = ground state (pure reference determinant). Column i>0 holds the
    normalized amplitudes of excited state i distributed over the SD basis.
    Configs in ci_data are absolute orbital indices (per spin channel, matched
    through ci_data["spins"] against the 'alp'/'bet' tags of sd_unique) -
    sd_unique is window-relative.

    Components outside the SD WINDOW (hole below or particle above it) cannot be
    represented there: they are dropped, NAMED in the report, and the column is
    renormalized over the kept part. A component that is not an excitation of the
    reference at all (hole above the reference HOMO, particle at or below it)
    raises - it means the vectors came from a different occupation (the same
    truncate-then-renormalize convention the log reader and
    normalize_ci_coefficients apply upstream - bit-identical when nothing is
    dropped). The lost weight is printed per state when it exceeds 1e-6 -
    widen num_occ_states/num_unocc_states and/or the MO window to capture more.
    A state lying entirely outside the basis raises.

    Returns:
        T (np.ndarray nSD+1 x nstates+1), drops (np.ndarray nstates+1):
        drops[i] = the CI weight of unit-normalized state i NOT represented
        in the basis - window-truncated components and any sub-normalization
        of the input amplitudes (e.g. a read-time ci_threshold truncation,
        the DFTB+ reader convention) both land here (0.0 for the GS column)
    """
    nsd = len(sd_unique) + 1
    nstates = len(ci_data["configs"]) + 1
    T = np.zeros((nsd, nstates))
    drops = np.zeros(nstates)
    T[0, 0] = 1.0
    # the reference occupation of the basis, per spin channel: the
    # basis holds holes in [occ_lo, homo_w] and particles in [homo_w+1, vir_hi]
    # (window-relative), so a config's hole above homo_w or particle at or
    # below it is NOT an excitation of this reference at all - a different
    # occupation produced the vectors (a smeared LUMO counted as occupied, a
    # changed charge or multiplicity). That is a mismatch to stop on, not a
    # truncation to renormalize away
    ref = {}
    for [o, v], sp in sd_unique:
        lo_o, hi_o, lo_v, hi_v = ref.get(sp, (o, o, v, v))
        ref[sp] = (min(lo_o, o), max(hi_o, o), min(lo_v, v), max(hi_v, v))
    # the row of every basis configuration (the first row of a repeated entry) - one dict lookup per
    # configuration instead of a scan of sd_unique (states x configurations x SDs comparisons per call)
    rows = {}
    for j, ([o, v], sp) in enumerate(sd_unique):
        rows.setdefault(((o, v), sp), j)
    for i, (confs, amps) in enumerate(zip(ci_data["configs"], ci_data["amps"])):
        spins = ci_data["spins"][i] if "spins" in ci_data else ["alp"] * len(confs)
        truncated = []
        for c, a, sp in zip(confs, amps, spins):
            cw = [c[0] - lowest_orbital + 1, c[1] - lowest_orbital + 1]
            j = rows.get(((cw[0], cw[1]), sp))
            if j is None:
                if sp not in ref:
                    raise ValueError(
                        F"CI state {i+1} carries a '{sp}' configuration {c[0]}->{c[1]} "
                        F"but the SD basis has no '{sp}' channel - the vectors come "
                        F"from a different spin treatment than the basis") from None
                occ_lo, homo_w, vir_lo, vir_hi = ref[sp]
                if cw[0] > homo_w or cw[1] < vir_lo:
                    homo_abs = homo_w + lowest_orbital - 1
                    raise ValueError(
                        F"CI state {i+1} excites {c[0]}->{c[1]}"
                        F"{' (' + sp + ')' if sp != 'alp' else ''}, which is not an "
                        F"excitation of the reference the SD basis was built for "
                        F"(occupied through orbital {homo_abs}): the vectors were "
                        F"produced with a different occupation - a fractionally "
                        F"filled orbital counted as occupied (&SMEAR), or a changed "
                        F"charge or multiplicity. A fixed SD basis cannot represent "
                        F"a moving reference") from None
                truncated.append((c[0], c[1], sp, float(a) ** 2))
                continue
            T[j + 1, i + 1] = a
        nrm2 = float(T[:, i + 1] @ T[:, i + 1])
        if nrm2 == 0.0:
            raise ValueError(f"CI state {i+1} lies entirely outside the SD basis - "
                             f"increase num_occ_states/num_unocc_states (and the MO window)")
        dropped = max(0.0, 1.0 - nrm2)
        T[:, i + 1] /= np.sqrt(nrm2)
        # name the dropped amplitudes - a total alone hides which orbitals the
        # window is missing
        truncated.sort(key=lambda x: -x[3])
        named = ", ".join(F"{o}->{v}{'' if sp == 'alp' else ' ' + sp} ({w:.1%})"
                          for o, v, sp, w in truncated[:4])
        if dropped > 0.5:
            # the state is MOSTLY outside the basis - a banner once per state and process, since everything built
            # from this column (overlaps, couplings) uses only the remnant - the one-line record below follows it
            # at every occurrence
            if (i + 1) not in _severe_drop_warned:
                _severe_drop_warned.add(i + 1)
                bar = "!" * 78
                print(bar)
                print(F"!!! WARNING sd2ci_matrix: CI state {i+1} lies MOSTLY OUTSIDE the SD basis")
                print(F"!!! dropped CI weight = {dropped:.1%} - only {1.0 - dropped:.1%} of the state is")
                print(F"!!! representable, and its overlaps and couplings are built from that")
                print(F"!!! remnant. Configurations outside the window: {named}")
                print(F"!!! Widen num_occ_states/num_unocc_states (and the MO window).")
                print(F"!!! The banner is shown once for this state, the one-line record at every occurrence.")
                print(bar)
        if dropped > 1e-6:
            print(F"sd2ci_matrix: state {i+1}: dropped CI weight {dropped:.2e} "
                  F"outside the SD basis (column renormalized): {named} - widen "
                  F"num_occ_states/num_unocc_states or the MO window to capture it")
        drops[i + 1] = dropped
    return T, drops


def align_ci_state_signs(st_ci, ci_curr, T_curr=None):
    """
    Chain a smooth gauge for the CI states: eigenvector global signs are
    arbitrary per step (fresh Davidson), and raw output flips them on ~70%%
    of MD steps. The previous step's data is already aligned, so a negative
    diagonal of the state time-overlap marks a flip of the CURRENT state -
    flip its amplitudes IN PLACE (the driver caches ci_curr, carrying the
    alignment forward), the st_ci column, and (when given) the T column.

    |st_ci[j, j]| < 0.5 is genuine-crossing territory and is deliberately
    left untouched - state ORDER is the engine's state-tracking job, only
    the gauge is smoothed here. Column 0 (the ground state) never flips.
    Everything gauge-invariant (|st|, populations, energies) is unchanged.

    See also: the engine-side compute_phase_corrections/compute_projector
    (dyn_projectors.cpp) correct the AMPLITUDES via proj_adi during the
    dynamics - step3's apply_phase_correction family corrects NBRA St file
    series. This helper aligns the emitted driver data itself, so every
    consumer - the engine under any setting, archives, restarts - sees
    one smooth gauge.

    Args:
        st_ci (np.ndarray): state time-overlap <prev_i|curr_j>, modified in place
        ci_curr (dict): read_ci_data dict of the current step, amps flipped in place
        T_curr (np.ndarray or None): sd2ci matrix of the current step, columns
                                     flipped in place

    Returns:
        list: the flipped state indices (1-based, matching st_ci columns)
    """
    flips = []
    for j in range(1, st_ci.shape[1]):
        if st_ci[j, j] < -0.5:
            amps = ci_curr["amps"][j - 1]
            for k in range(len(amps)):
                amps[k] = -amps[k]
            st_ci[:, j] = -st_ci[:, j]
            if T_curr is not None:
                T_curr[:, j] = -T_curr[:, j]
            flips.append(j)
    return flips



def followed_nac(st, dt, min_overlap=0.5):
    """
    The time-derivative couplings of a driver step taken AFTER the states are followed through it:
    the new state pi(i) that continues the old state i (the assignment of the largest |st|) with its
    sign s_i gives the overlap of the followed states S~ = st[:, pi] s, their coupling is
    d~ = (S~ - S~^T) / (2 dt), and it goes back under the raw labels of the new states -
    d[pi(i), pi(i')] = s_i s_i' d~[i, i'] - the labels and signs the engine receives the step under.

    The drivers hand over no coupling: the engine derives the followed coupling from the time overlap with
    its own projector (nac_update_method 2), so its permutation and the one that relabels the amplitudes are
    one object - this function serves a model that must hand couplings over itself.

    A NAC-carrying integrator under a relabel projector needs this coupling. At an unresolved
    crossing (a pair that swaps within the step) the engine swaps the labels and moves the
    amplitudes along, while the raw (st - st^T) / (2 dt) of the swapped pair is 1 / dt or 0 by the
    arbitrary sign of a fresh diagonalization - the amplitudes are turned twice or not at all.
    Without a swap and with positive diagonals this is the raw formula to the last bit. A step in
    which a state finds no partner above min_overlap (it left the propagated block) keeps the raw
    formula - nothing can be followed there.

    Args:
        st (np.ndarray): real state time-overlap <prev_i|curr_j>, square
        dt (float): the nuclear time step, a.u.
        min_overlap (float): the smallest |st| of a followed pair

    Returns:
        np.ndarray: the antisymmetric coupling matrix in the raw order of the current states
    """
    from scipy.optimize import linear_sum_assignment
    st = np.asarray(st, dtype=np.float64)
    n = st.shape[0]
    rows, cols = linear_sum_assignment(-np.abs(st))
    pi = np.empty(n, dtype=int)
    pi[rows] = cols
    matched = st[np.arange(n), pi]
    if np.abs(matched).min() < min_overlap:
        return (st - st.T) / (2.0 * dt)
    sg = np.where(matched < 0.0, -1.0, 1.0)
    followed = st[:, pi] * sg[None, :]
    d_followed = (followed - followed.T) / (2.0 * dt)
    d = np.zeros((n, n))
    d[np.ix_(pi, pi)] = sg[:, None] * d_followed * sg[None, :]
    return d


def check_sd_capture(ci_drops, nstates, max_drop, knob="dia_basis_max_drop"):
    """
    Enforce the SD-window capture contract: a renormalized T column that
    lost most of its CI weight to the window truncation is noise, and
    every propagated state feeds the consumer - the diabatic (SD)
    transform, or the SD-projected CI overlaps. Call at every step - the
    CI coefficients evolve along a trajectory, so a window adequate at
    t = 0 can degrade.

    Args:
        ci_drops (np.ndarray): per-state dropped CI weights from sd2ci_matrix
        nstates (int): number of propagated states (leading T columns)
        max_drop (float): largest tolerated dropped weight
        knob (str): name of the driver keyword that carries max_drop,
                    quoted in the error message
    """
    bad = [(j, d) for j, d in enumerate(ci_drops[:nstates]) if d > max_drop]
    if bad:
        detail = ", ".join(F"state {j}: {d:.1%}" for j, d in bad)
        raise ValueError(
            F"the SD basis captures too little of propagated states "
            F"({detail}, limit {knob} = {max_drop:.1%}) - the "
            F"renormalized T columns would be meaningless. Where the SD "
            F"window cuts the configurations, widen num_occ_states/"
            F"num_unocc_states (and the MO window) - where the reader does "
            F"(CP2K's MIN_AMPLITUDE, ci_threshold), lower those thresholds - "
            F"or reduce nstates, or raise {knob} deliberately")


def compose_drops(a, b):
    """the weight lost to two successive cuts of a renormalized vector: 1 - (1 - a)(1 - b) (arrays or scalars)"""
    return 1.0 - (1.0 - a) * (1.0 - b)


def check_read_capture(read_drops, nstates, max_drop, knob="sd_capture_max_drop"):
    """check_sd_capture over the per-excited-state read-stage drops, the ground state (no drop) prepended"""
    check_sd_capture(np.concatenate(([0.0], read_drops)), nstates, max_drop, knob=knob)


def refuse_tolerance_key(params):
    """the renamed reader key: 'tolerance' became ci_threshold (the same |C|^2 cut) - a dict still carrying it is
    refused rather than silently read with the default"""
    if "tolerance" in params:
        raise ValueError('tolerance was unified into "ci_threshold" (the same '
                         "|C|^2 cut) - rename the key, the value is unchanged")
