# *********************************************************************************
# * Copyright (C) 2026 Lukas Hronek
# *
# * This file is distributed under the terms of the GNU General Public License
# * as published by the Free Software Foundation, either version 3 of
# * the License, or (at your option) any later version.
# * See the file LICENSE in the root directory of this distribution
# * or <http://www.gnu.org/licenses/>.
# *
# *********************************************************************************/
"""
.. module:: sqc
   :platform: Unix, Windows
   :synopsis: Cotton-Miller symmetrical quasi-classical (SQC) windows on the
       Meyer-Miller mapping: window sampling of initial amplitudes and window
       binning of saved amplitudes into renormalized populations. The engine
       does both natively (init_type 5/6 and compute_average_sqc_pop / the
       sqc_pop_adi|dia savers); this module is the numpy reference and the
       post-processor for amplitudes saved by earlier runs.

.. moduleauthor:: Lukas Hronek

References:
    Cotton, S. J.; Miller, W. H. J. Phys. Chem. A 2013, 117, 7190 (square windows)
    Cotton, S. J.; Miller, W. H. J. Chem. Phys. 2016, 145, 144108 (triangle, two states)
    Cotton, S. J.; Miller, W. H. J. Chem. Phys. 2019, 150, 104101 (triangle, many states)
"""

import numpy as np

SQUARE, TRIANGLE = 0, 1
GAMMA_SQUARE = 0.366          # (sqrt(3) - 1)/2, the standard square-window ZPE
GAMMA_TRIANGLE = 1.0 / 3.0


def default_gamma(window):
    """The standard Meyer-Miller gamma of a window type (0 square, 1 triangle)."""
    if window == SQUARE:
        return GAMMA_SQUARE
    if window == TRIANGLE:
        return GAMMA_TRIANGLE
    raise ValueError("window must be 0 (square) or 1 (triangle)")


def sample_shifted_actions(nstates, istate, window, gamma, rng):
    """
    One draw of the shifted Meyer-Miller actions e_k = n_k + gamma = |c_k|^2 from
    the window of state `istate`.

    square:   e_istate uniform in [1, 1 + 2 gamma], every other e_k uniform in [0, 2 gamma]
    triangle: uniform over { 1 <= e_istate <= 2, 0 <= e_k <= 2 - e_istate for k != istate }
              (rejection from the box [1, 2] x [0, 1]^(N-1))
    """
    if window == SQUARE:
        e = 2.0 * gamma * rng.uniform(size=nstates)
        e[istate] += 1.0
        return e
    if window == TRIANGLE:
        while True:
            e = rng.uniform(size=nstates)
            e[istate] += 1.0
            others = np.delete(e, istate)
            if np.all(others <= 2.0 - e[istate]):
                return e
    raise ValueError("window must be 0 (square) or 1 (triangle)")


def sample_amplitudes(nstates, istate, window, gamma=None, ntraj=1, rng=None):
    """
    SQC-window-sampled complex amplitudes c_k = sqrt(e_k) exp(i theta_k), theta_k
    uniform on [0, 2 pi) - the numpy twin of init_amplitudes with init_type 5/6.

    Returns:
        numpy.ndarray(nstates, ntraj) of complex amplitudes; sum_k |c_k|^2 lies in
        [1, 1 + 2 N gamma] (square, mean 1 + N gamma) or [1, N] (triangle) - NOT normalized.
    """
    rng = np.random.default_rng() if rng is None else rng
    if gamma is None:
        gamma = default_gamma(window)
    C = np.zeros((nstates, ntraj), dtype=complex)
    for traj in range(ntraj):
        e = sample_shifted_actions(nstates, istate, window, gamma, rng)
        theta = 2.0 * np.pi * rng.uniform(size=nstates)
        C[:, traj] = np.sqrt(e) * np.exp(1j * theta)
    return C


def window_indicator(e, window, gamma):
    """
    Which state's window (if any) the shifted actions `e` (shape (nstates,)) fall
    into at binning time.

    square:   state k iff e_k in [1, 1 + 2 gamma] and every other e_j in [0, 2 gamma]
    triangle: state k iff e_k >= 1 and every other e_j <= 1  (the Cotton-Miller bin)

    Returns:
        the state index, or -1 when the point lies in no window
    """
    e = np.asarray(e, dtype=float)
    n = len(e)
    for k in range(n):
        others = np.delete(e, k)
        if window == SQUARE:
            inside = (1.0 <= e[k] <= 1.0 + 2.0 * gamma) and np.all((others >= 0.0) & (others <= 2.0 * gamma))
        elif window == TRIANGLE:
            inside = (e[k] >= 1.0) and np.all(others <= 1.0)
        else:
            raise ValueError("window must be 0 (square) or 1 (triangle)")
        if inside:
            return k
    return -1


def sqc_populations(amplitudes, window, gamma=None):
    """
    Bin trajectory amplitudes into SQC populations, renormalized over the states.

    Args:
        amplitudes ( numpy.ndarray ): complex, shape (ntraj, nstates) for one time
            or (nsteps, ntraj, nstates) for a whole run (the layout of the saved
            "Cadi"/"Cdia" datasets)
        window ( int ): 0 square, 1 triangle
        gamma ( float ): the Meyer-Miller gamma; None -> the window's standard value

    Returns:
        numpy.ndarray of populations, shape (nstates,) or (nsteps, nstates):
        P_k = <W_k> / sum_m <W_m>; all-zero when no trajectory falls in any window
    """
    if gamma is None:
        gamma = default_gamma(window)
    A = np.asarray(amplitudes)
    single = (A.ndim == 2)
    if single:
        A = A[None, ...]
    nsteps, ntraj, nstates = A.shape
    P = np.zeros((nsteps, nstates))
    for t in range(nsteps):
        for traj in range(ntraj):
            k = window_indicator(np.abs(A[t, traj]) ** 2, window, gamma)
            if k >= 0:
                P[t, k] += 1.0
        tot = P[t].sum()
        if tot > 0.0:
            P[t] /= tot
    return P[0] if single else P
