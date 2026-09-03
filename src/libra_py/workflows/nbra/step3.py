# *********************************************************************************
# * Copyright (C) 2017-2019 Brendan A. Smith, Wei Li, Alexey V. Akimov
# *
# * This file is distributed under the terms of the GNU General Public License
# * as published by the Free Software Foundation, either version 2 of
# * the License, or (at your option) any later version.
# * See the file LICENSE in the root directory of this distribution
# * or <http://www.gnu.org/licenses/>.
# *
# *********************************************************************************/
#
#
#
"""
.. module:: step3
   :platform: Unix, Windows
   :synopsis: This module is designed to convert the results of QE calculations
       (KS orbital energies and time-overlaps in the KS basis) to the generic Hvib
       matrices, which account for:

           - state reordering;
           - phase corrections;
           - multi-electron wavefunction (Slater determinants) and spin-adaptation
           - scissor operator corrections to energy levels

.. moduleauthor:: Brendan A. Smith, Wei Li, and Alexey V. Akimov

"""


import sys
import cmath
import functools
import math
import os
import time
import glob
import json
import numpy as np
from libra_py.workflows.nbra.mp_context import MP_CTX as _MP_CTX
import scipy.sparse as sp
if sys.platform == "cygwin":
    from cyglibra_core import *
elif sys.platform == "linux" or sys.platform == "linux2":
    from liblibra_core import *

from . import mapping, step2_many_body, step3_many_body
from libra_py.citools import slatdet
import util.libutil as comn
import libra_py.packages.cp2k.methods as CP2K_methods
import libra_py.units as units
import libra_py.data_read as data_read
import libra_py.data_conv as data_conv
import libra_py.hungarian as hungarian
from libra_py import orthogonalizations
from libra_py.orthogonalizations import (validate_overlap_matrix,
                                         get_Lowdin_scipy,
                                         apply_orthonormalization_scipy,
                                         lowdin_digest)


def get_step2_data(_params):
    """
    A light function to obtain the step2 data: S, St, hvib

    Args:
        params ( dictionary ): Control paramerter of this type of simulation. Can include the follwing keys:

        * **params["read_S_data"]** ( int ): whether to read S data (1) or not (0) [ default: 1 ]

        * **params["S_data_re_prefix"]** ( string ): prefix of files containing real part of orbital overlaps [ default: "S_dia_ks_" ]

        * **params["S_data_re_suffix"]** ( string ): suffix of files containing real part of orbital overlaps [ default: "_re" ]

        * **params["S_data_im_prefix"]** ( string ): prefix of files containing imaginary part of orbital overlaps [ default: "S_dia_ks_" ]

        * **params["S_data_im_suffix"]** ( string ): suffix of files containing imaginary part of orbital overlaps [ default: "_im" ]

        * **params["read_St_data"]** ( int ): whether to read St data (1) or not (0) [ default: 1 ]

        * **params["St_data_re_prefix"]** ( string ): prefix of files containing real part of orbital time-overlaps [ default: "St_dia_ks_" ]

        * **params["St_data_re_suffix"]** ( string ): suffix of files containing real part of orbital time-overlaps [ default: "_re" ]

        * **params["St_data_im_prefix"]** ( string ): prefix of files containing imaginary part of orbital time-overlaps [ default: "St_dia_ks_" ]

        * **params["St_data_im_suffix"]** ( string ): suffix of files containing imaginary part of orbital time-overlaps [ default: "_im" ]

        * **params["read_hvib_data"]** ( int ): whether to read hvib data (1) or not (0) [ default: 1 ]

        * **params["hvib_data_re_prefix"]** ( string ): prefix of files containing real part of vibronic Hamiltonian [ default: "hvib_dia_ks_" ]

        * **params["hvib_data_re_suffix"]** ( string ): suffix of files containing real part of vibronic Hamiltonian [ default: "_re" ]

        * **params["hvib_data_im_prefix"]** ( string ): prefix of files containing imaginary part of vibronic Hamiltonian [ default: "hvib_dia_ks_" ]

        * **params["hvib_data_im_suffix"]** ( string ): suffix of files containing imaginary part of vibronic Hamiltonian [ default: "_im" ]

        Note: the _im files/values follow the canonical convention Im(Hvib) = -d = -(St - St^T)/(2 dt); directories written by the current step 3 carry hvib_convention.json, markerless directories are legacy and hold +d (the NAC itself)

        * **params["data_set_paths"]** ( list of strings ): see in :func:`libra_py.data_read.get_data_sets`

        * **params["data_dim"]** ( int ): see in :func:`libra_py.data_read.get_data`

        * **params["active_space"]** ( list of ints ): see in :func:`libra_py.data_read.get_data`

        * **params["isnap"]** ( int ): see in :func:`libra_py.data_read.get_data`

        * **params["fsnap"]** ( int ): see in :func:`libra_py.data_read.get_data`

    Returns:
        tuple: S, St, hvib

        * **S** ( list of lists of CMATRIX objects) : S[iset][itime] an Norb x Norb CMATRIX of overlaps for data set `iset` at time `itime`

        * **St** ( list of lists of CMATRIX objects) : St[iset][itime] an Norb x Norb CMATRIX of time-overlaps for data set `iset` at time `itime`

        * **hvib** ( list of lists of CMATRIX objects) : hvib[iset][itime] an Norb x Norb CMATRIX of vibronic Hamiltonians for data set `iset` at time `itime`

    """

    params = dict(_params)

    critical_params = []
    default_params = {"read_S_data": 1, "read_S_re": 1, "read_S_im": 1,
                      "S_data_re_prefix": "S_dia_ks_", "S_data_re_suffix": "_re",
                      "S_data_im_prefix": "S_dia_ks_", "S_data_im_suffix": "_im",
                      "read_St_data": 1, "read_St_re": 1, "read_St_im": 1,
                      "St_data_re_prefix": "St_dia_ks_", "St_data_re_suffix": "_re",
                      "St_data_im_prefix": "St_dia_ks_", "St_data_im_suffix": "_im",
                      "read_hvib_data": 1, "read_hvib_re": 1, "read_hvib_im": 1,
                      "hvib_data_re_prefix": "hvib_dia_ks_", "hvib_data_re_suffix": "_re",
                      "hvib_data_im_prefix": "hvib_dia_ks_", "hvib_data_im_suffix": "_im"
                      }
    comn.check_input(params, default_params, critical_params)

    S, St, Hvib = [], [], []
    prms = dict(_params)

    # Fetching the overlap matricies
    if (params["read_S_data"] == 1):
        prms["get_real"] = params["read_S_re"]
        prms["get_imag"] = params["read_S_im"]
        prms["data_re_prefix"] = params["S_data_re_prefix"]
        prms["data_re_suffix"] = params["S_data_re_suffix"]
        prms["data_im_prefix"] = params["S_data_im_prefix"]
        prms["data_im_suffix"] = params["S_data_im_suffix"]
        S = data_read.get_data_sets(prms)

    # Fetching the time-derivative overlap matricies
    if (params["read_St_data"] == 1):
        prms["get_real"] = params["read_St_re"]
        prms["get_imag"] = params["read_St_im"]
        prms["data_re_prefix"] = params["St_data_re_prefix"]
        prms["data_re_suffix"] = params["St_data_re_suffix"]
        prms["data_im_prefix"] = params["St_data_im_prefix"]
        prms["data_im_suffix"] = params["St_data_im_suffix"]
        St = data_read.get_data_sets(prms)

    # Fetching the vibronic Hamiltonian matricies
    if (params["read_hvib_data"] == 1):
        prms["get_real"] = params["read_hvib_re"]
        prms["get_imag"] = params["read_hvib_im"]
        prms["data_re_prefix"] = params["hvib_data_re_prefix"]
        prms["data_re_suffix"] = params["hvib_data_re_suffix"]
        prms["data_im_prefix"] = params["hvib_data_im_prefix"]
        prms["data_im_suffix"] = params["hvib_data_im_suffix"]
        Hvib = data_read.get_data_sets(prms)

    return S, St, Hvib


def print_SD_basis(SD_basis):
    """
    Just a light function to print the SD basis

    Args:
        SD_basis ( list of lists of ints ): Slater determinant basis in terms of Kohn-Sham orbital indicies

            Possible ground state configurations

            Ex. 1. SD_basis[0] = [ 5, -15 ]
            Ex. 2. SD_basis[0] = [ 2, 3, 4, 5, -12, -13, -14, -15 ]

            Possible ground state configurations

            Ex. 1. SD_basis[N > 0] = [ 6, -15 ]
            Ex. 2. SD_basis[N > 0] = [ 2, 3, 4, 6, -12, -13, -14, -15 ]

            Where we have 10 alpha and beta spin orbitals, and the cbm index for the alpha spin-channel is 5
            and the cbm index for the beta spin-channel is 15.
    """

    for i in range(len(SD_basis)):
        if i == 0:
            print(" GS: ", SD_basis[i])
        else:
            print(" ES " + str(i) + ": ", SD_basis[i])
    print("WARNING: The Slater determinant bases above may not be sorted based on their energies\n")


def sort_SD_energies(Hvib):
    """
    This function goes into the Hvib (SD basis) files and sorts the energies
    For each Hvib file, we are going to obtain a list of lists of orbital index and energy pair
    These orbital index and energy pairs for each step will be sorted based on energies

    Args:
        Hvib ( list of lists of CMATRIX objects ): vibronic Hamiltonian in the Slater determinant basis

    """

    ntraj = len(Hvib)
    nsnaps = len(Hvib[0])
    nSD = Hvib[0][0].num_of_cols

    orbital_index_energy_pairs = []

    for traj in range(ntraj):

        orbital_index_energy_pairs.append([])

        for snap in range(nsnaps):

            index_energy_pairs = []

            for sd_index in range(nSD):
                index_energy_pairs.append([sd_index, Hvib[traj][snap].get(sd_index, sd_index).real])

            sorted_index_energy_pairs = merge_sort(index_energy_pairs)
            orbital_index_energy_pairs[traj].append(sorted_index_energy_pairs)

    return orbital_index_energy_pairs


def output_sorted_Hvibs(Hvib, orbital_index_energy_pairs, _params={}):
    """
    This function outputs the vibronic Hamiltonians in the SD basis according to their sorted order

    Args:
        Hvib ( list of lists of CMATRIX objects ): vibronic Hamiltonian in the Slater determinant basis
        orbital_index_energy_pairs ( list of lists of lists of lists ): orbital index and energy pair lists for each step and nuclear trajectory
                               Ex. orbital_index_energy_pairs[i][j][k][0] = kth slater determinant index at the jth step on the ith nuclear trajectory
                               Ex. orbital_index_energy_pairs[i][j][k][1] = kth slater determinant energy at the jth step on the ith nuclear trajectory

    """

    params = dict(_params)
    critical_params = []
    default_params = {"save_files": 1,
                      "output_dir_prefix": "res_sorted_traj",
                      "hvib_data_re_prefix": "Hvib_sorted_", "hvib_data_re_suffix": "_re",
                      "hvib_data_im_prefix": "Hvib_sorted_", "hvib_data_im_suffix": "_im"
                      }
    comn.check_input(params, default_params, critical_params)

    save_files = params["save_files"]
    output_dir_prefix = params["output_dir_prefix"]
    prefix_re = params["hvib_data_re_prefix"]
    suffix_re = params["hvib_data_re_suffix"]
    prefix_im = params["hvib_data_im_prefix"]
    suffix_im = params["hvib_data_im_suffix"]

    ntraj = len(orbital_index_energy_pairs)
    nsnaps = len(orbital_index_energy_pairs[0])
    nSD = len(orbital_index_energy_pairs[0][0])

    Hvibs_sorted = []

    for traj in range(ntraj):

        if (save_files):
            rd_sorted = F"{output_dir_prefix}{traj}"
            os.system(F"rm -r {rd_sorted}")
            os.system(F"mkdir {rd_sorted}")

        Hvibs_sorted.append([])

        for snap in range(nsnaps):
            Hvib_sorted = CMATRIX(nSD, nSD)
            for i in range(nSD):
                for j in range(nSD):
                    a = orbital_index_energy_pairs[traj][snap][i][0]
                    b = orbital_index_energy_pairs[traj][snap][j][0]
                    Hvib_sorted.set(i, j, Hvib[traj][snap].get(a, b))

            Hvibs_sorted[traj].append(Hvib_sorted)

            if (save_files):
                Hvibs_sorted[traj][snap].real().show_matrix(F"{rd_sorted}/{prefix_re}{snap}{suffix_re}")
                Hvibs_sorted[traj][snap].imag().show_matrix(F"{rd_sorted}/{prefix_im}{snap}{suffix_im}")

    return Hvibs_sorted


def build_SD_basis(data_dim, cbm_alpha_index, alpha_include, cbm_beta_index, beta_include, excitation_type):
    """
    Builds a Slater Determinant basis based on the indexing notation scheme used in Libra

    Args:
        data_dim (int): how many rows or columns in the vibronic Hamiltonian matrix. This will be an even number becuase the
                        number of alpha orbtials should equal the number of beta orbitals
        cbm_(alpha/beta)_index (int): index of VBM (or HOMO) in the matrix of the vibronic Hamiltonian
                                      (row or column index). Note, this index is from 1
        (alpha/beta)_include (int): how many orbitals to include from the cbm_(alpha/beta)_index
        excitation_type (int):  0: Make SDs with beta electrons excited
                                1: Make SDs with alpha electrons excited
                                2: Make two sets of SDs, one for beta and one for alpha electrons excited
    """

    num_same_spin_orbitals = int(data_dim / 2)

    alpha_electrons = []
    beta_electrons = []

    # Validate the requested basis against the available orbitals: out-of-range
    # indices would silently build duplicate/garbage determinants (a negative
    # beta index even flips sign into the ALPHA block), and a silently clamped
    # basis changes the downstream state count - raise with guidance instead
    if data_dim % 2 > 0:
        raise ValueError("data_dim must be even (equal numbers of alpha and beta spin-orbitals)")

    if not 1 <= cbm_alpha_index <= num_same_spin_orbitals:
        raise ValueError(F"cbm_alpha_index={cbm_alpha_index} must lie in the alpha block "
                         F"[1, {num_same_spin_orbitals}]")
    if not num_same_spin_orbitals + 1 <= cbm_beta_index <= 2 * num_same_spin_orbitals:
        raise ValueError(F"cbm_beta_index={cbm_beta_index} must lie in the beta block "
                         F"[{num_same_spin_orbitals + 1}, {2 * num_same_spin_orbitals}]")

    if not 0 <= alpha_include <= cbm_alpha_index - 1:
        raise ValueError(F"alpha_include={alpha_include} but only {cbm_alpha_index - 1} "
                         F"occupied alpha orbitals lie below the alpha CBM")
    if cbm_alpha_index + 1 + alpha_include > num_same_spin_orbitals:
        raise ValueError(F"alpha_include={alpha_include} requests {alpha_include + 1} alpha "
                         F"virtuals but only {num_same_spin_orbitals - cbm_alpha_index} exist "
                         F"above the alpha CBM")
    if not 0 <= beta_include <= cbm_beta_index - num_same_spin_orbitals - 1:
        raise ValueError(F"beta_include={beta_include} but only "
                         F"{cbm_beta_index - num_same_spin_orbitals - 1} occupied beta "
                         F"orbitals lie below the beta CBM")
    if cbm_beta_index + 1 + beta_include > 2 * num_same_spin_orbitals:
        raise ValueError(F"beta_include={beta_include} requests {beta_include + 1} beta "
                         F"virtuals but only {2 * num_same_spin_orbitals - cbm_beta_index} "
                         F"exist above the beta CBM")

    # Make ground state SD
    for i in range(cbm_alpha_index - alpha_include, cbm_alpha_index + 1):
        alpha_electrons.append(i)
    for i in range(cbm_beta_index - beta_include, cbm_beta_index + 1):
        beta_electrons.append(-i)

    gs_SD = alpha_electrons[:] + beta_electrons[:]

    # Make excited state SDs
    es_SD = []
    SD_basis = []

    # Excite only alpha electrons
    if excitation_type == 0:
        for i in alpha_electrons:
            for j in range(cbm_alpha_index + 1, cbm_alpha_index + 2 + alpha_include):

                # es_sd = [ j if electron == i else electron for electron in gs_SD[:] ]  # compact version
                es_sd = []
                for electron in gs_SD[:]:
                    if electron == i:
                        es_sd.append(j)
                    else:
                        es_sd.append(electron)
                es_SD.append(es_sd)

    # Excite only beta electrons
    elif excitation_type == 1:
        for i in beta_electrons:
            for j in range(cbm_beta_index + 1, cbm_beta_index + 2 + beta_include):

                # es_sd = [ -j if electron == i else electron for electron in gs_SD[:] ]  # compact version
                es_sd = []
                for electron in gs_SD[:]:
                    if electron == i:
                        es_sd.append(-j)
                    else:
                        es_sd.append(electron)
                es_SD.append(es_sd)

    # Excite both alpha and beta electrons
    else:
        for i in alpha_electrons:
            for j in range(cbm_alpha_index + 1, cbm_alpha_index + 2 + alpha_include):

                # es_sd = [ j if electron == i else electron for electron in gs_SD[:] ]  # compact version
                es_sd = []
                for electron in gs_SD[:]:
                    if electron == i:
                        es_sd.append(j)
                    else:
                        es_sd.append(electron)
                es_SD.append(es_sd)

        for i in beta_electrons:
            for j in range(cbm_beta_index + 1, cbm_beta_index + 2 + beta_include):

                # es_sd = [ -j if electron == i else electron for electron in gs_SD[:] ]  # compact version
                es_sd = []
                for electron in gs_SD[:]:
                    if electron == i:
                        es_sd.append(-j)
                    else:
                        es_sd.append(electron)
                es_SD.append(es_sd)

    SD_basis.append(gs_SD)
    for i in range(len(es_SD)):
        SD_basis.append(es_SD[i])

    return SD_basis


def get_Lowdin(S):
    """
    Find the S_i_half for the S matrix - alpha and beta components

    Args:
        S ( CMATRIX(2N, 2N) ): is a matrix of MO overlaps. It has a block structure as:

            .. math::
                S =
                \\begin{vmatrix}
                S_{aa}  & S_{ab} \\\
                S_{ba}  & S_{bb}
                \\end{vmatrix}

            Here, S_xy are the overlaps of the MOs for spin channels x and y (alpha, beta) - only
            spatial components of the orbitals are taken into account here.
            Here, N - is the total number of orbitals (double occupancies)

    Returns:
        tuple: (S_aa_i_half, S_bb_i_half), where:

            * S_aa_i_half ( CMATRIX(N,N) ): S_aa^{-1/2} - inverse square root matrix for the alpha-alpha block
            * S_bb_i_half ( CMATRIX(N,N) ): S_bb^{-1/2} - inverse square root matrix for the beta-beta block

    """

    nstates = int(S.num_of_cols / 2)  # division by 2 because it is a super-matrix

    alp = list(range(0, nstates))
    bet = list(range(nstates, 2 * nstates))

    S_aa = CMATRIX(nstates, nstates)
    S_bb = CMATRIX(nstates, nstates)

    pop_submatrix(S, S_aa, alp, alp)
    pop_submatrix(S, S_bb, bet, bet)

    _validate_cmatrix_overlap(S_aa, "get_Lowdin S_aa")
    is_inv = FullPivLU_rank_invertible(S_aa)
    if is_inv[1] != 1:
        raise ValueError("S_aa is not invertible")

    _validate_cmatrix_overlap(S_bb, "get_Lowdin S_bb")
    is_inv = FullPivLU_rank_invertible(S_bb)
    if is_inv[1] != 1:
        raise ValueError("S_bb is not invertible")

    S_aa_half = CMATRIX(nstates, nstates)
    S_aa_i_half = CMATRIX(nstates, nstates)
    sqrt_matrix(S_aa, S_aa_half, S_aa_i_half)

    S_bb_half = CMATRIX(nstates, nstates)
    S_bb_i_half = CMATRIX(nstates, nstates)
    sqrt_matrix(S_bb, S_bb_half, S_bb_i_half)

    return S_aa_i_half, S_bb_i_half


def apply_normalization(S, St):
    """

    Transforms the input transition density matrix computed with potentially
    non-orthogonalized orbitals such that it would correspond to the properly
    orthonormalized ones

    Args:
        S ( CMATRIX(2N, 2N) ): is a matrix of MO overlaps S_ij = <i|j>. It has a block structure as:

            .. math::
                S =
                \\begin{vmatrix}
                S_{aa}  & S_{ab} \\\
                S_{ba}  & S_{bb}
                \\end{vmatrix}

            Here, S_xy are the overlaps of the MOs for spin channels x and y (alpha, beta) - only
            spatial components of the orbitals are taken into account here.
            Here, N - is the total number of orbitals (double occupancies)

        St ( CMATRIX(2N, 2N) ): the transition density matrix St_ij = <i|d/dt|j>. It has a block structure as:

            .. math::
                St =
                \\begin{vmatrix}
                St_{aa}  & St_{ab} \\\
                St_{ba}  & St_{bb}
                \\end{vmatrix}

            Here, St_xy are the transition density matrix for spin channels x and y (alpha, beta) - only
            spatial components of the orbitals are taken into account here.
            Here, N - is the total number of orbitals (double occupancies)

    Returns:
        None: but the input matrix ```St``` is changed


    """

    nsteps = len(S)

    nstates = int(St[0].num_of_cols / 2)  # division by 2 because it is a super-matrix

    alp = list(range(0, nstates))
    bet = list(range(nstates, 2 * nstates))

    St_aa = CMATRIX(nstates, nstates)
    St_ab = CMATRIX(nstates, nstates)
    St_ba = CMATRIX(nstates, nstates)
    St_bb = CMATRIX(nstates, nstates)

    for i in range(0, nsteps - 1):

        U1_a, U1_b = get_Lowdin(S[i])    # time n
        U2_a, U2_b = get_Lowdin(S[i + 1])  # time n+1

        pop_submatrix(St[i], St_aa, alp, alp)
        pop_submatrix(St[i], St_ab, alp, bet)
        pop_submatrix(St[i], St_ba, bet, alp)
        pop_submatrix(St[i], St_bb, bet, bet)

        St_aa = U1_a.H() * St_aa * U2_a
        St_ab = U1_a.H() * St_ab * U2_b
        St_ba = U1_b.H() * St_ba * U2_a
        St_bb = U1_b.H() * St_bb * U2_b

        push_submatrix(St[i], St_aa, alp, alp)
        push_submatrix(St[i], St_ab, alp, bet)
        push_submatrix(St[i], St_ba, bet, alp)
        push_submatrix(St[i], St_bb, bet, bet)


def get_Lowdin_general(S):
    """
    Find the S_i_half for the S matrix
    Args:
        S ( CMATRIX(N, N) ): is a matrix of MO overlaps.
    Returns:
        tuple: S_i_half, where:
            * S_i_half ( CMATRIX(N,N) ): S^{-1/2} - inverse square root matrix

    """

    nstates = int(S.num_of_cols)  # division by 2 because it is a super-matrix
    _validate_cmatrix_overlap(S, "get_Lowdin_general")
    is_inv = FullPivLU_rank_invertible(S)
    if is_inv[1] != 1:
        S.show_matrix()
        raise ValueError(F"S is not invertible, true rank = { is_inv[0] }")
    S_half = CMATRIX(nstates, nstates)
    S_i_half = CMATRIX(nstates, nstates)
    sqrt_matrix(S, S_half, S_i_half)
    return S_i_half


def apply_orthonormalization_general(S, St):
    """

    Transforms the input transition density matrix computed with potentially
    non-orthogonalized orbitals such that it would correspond to the properly
    orthonormalized ones

    Args:
        S  ( CMATRIX(N, N) ): is a matrix of MO overlaps S_ij = <i|j>
        St ( CMATRIX(N, N) ): the transition density matrix St_ij = <i|d/dt|j>

    Returns:
        None: but the input matricies ```S``` and ```St``` are changed
    """

    nsteps = len(S)
    nstates = int(St[0].num_of_cols)  # division by 2 because it is a super-matrix

    # For St
    for i in range(0, nsteps - 1):
        U1 = get_Lowdin_general(S[i])   # time n
        U2 = get_Lowdin_general(S[i + 1])  # time n+1
        # print("St matrix before")
        # St[i].show_matrix()
        St_normalized = U1.H() * St[i] * U2
        # print("St matrix after")
        # St_normalized.show_matrix()
        push_submatrix(St[i], St_normalized, list(range(0, nstates)), list(range(0, nstates)))

    # For S
    for i in range(0, nsteps):
        U1 = get_Lowdin_general(S[i])  # time n
        # print("S matrix before")
        # S[i].show_matrix()
        S_normalized = U1.H() * S[i] * U1
        # print("S matrix after")
        # S_normalized.show_matrix()
        push_submatrix(S[i], S_normalized, list(range(0, nstates)), list(range(0, nstates)))


_LOWDIN_CACHE = {"step": None, "digest": None, "U": None}


def _orthonormalization_mode(value):
    """Resolves the tri-state `apply_orthonormalization` value (same
    vocabulary as cp2k_compute_adi): True -> "lowdin", False/None -> "none",
    and the strings "lowdin", "lowdin_cached" and "none" pass through.
    """
    if value is True:
        return "lowdin"
    if value is False or value is None:
        return "none"
    if value in ("lowdin", "lowdin_cached", "none"):
        return value
    raise ValueError(
        F"apply_orthonormalization must be True/'lowdin', 'lowdin_cached' or False/'none', got {value!r}")


def _lowdin_cache_take(step, digest):
    """The previous step's ket factor, when it provably is this step's bra
    factor (the expected step index AND the content digest both match);
    None otherwise. Hits occur on every step after the first in a serial
    loop and within each contiguous Pool.starmap chunk; a miss just means
    the factor is recomputed."""
    if _LOWDIN_CACHE["step"] == step and _LOWDIN_CACHE["digest"] == digest:
        return _LOWDIN_CACHE["U"]
    return None


def _lowdin_cache_put(step, digest, U):
    _LOWDIN_CACHE.update(step=step, digest=digest, U=U)


def make_active_space(num_occ,
                      num_unocc,
                      data_dim,
                      ks_homo_index,
                      isUKS=0,
                      num_occ_alpha=0, num_unocc_alpha=0,
                      num_occ_beta=0, num_unocc_beta=0):
    """
    This function makes an active space based on the number of occupied and
    unoccupied orbitals and the initial KS HOMO index. **Note that the ks_homo_index
    starts from 1.**

    Args:

        num_occ (integer): Number of occupied orbitals from HOMO

        num_unocc (integer): Number of unoccupied orbitals from LUMO

        data_dim (integer): The data dimension of the 'raw' overlap matrices

        ks_homo_index (list/int): The KS HOMO indexes (which starts from 1) of the 'raw' matrices.
        If it's a list, it should be in the format [homo_alpha, homo_beta]

        isUKS : request unrestricted spin calculation. If specified, instead of new_ks_homo_index
        returns (new_ks_homo_alpha_index, new_ks_homo_beta_index). Default is 0.

        num_occ_alpha/beta: Number of occupied orbitals from HOMO (alpha/beta channel). (isUKS=True)

        num_unocc_alpha/beta: Number of unoccupied orbitals from LUMO (alpha/beta channel). (isUKS=True)

    Returns:

        new_active_space (list): The new active space

        new_ks_homo_index (integer): The new KS HOMO index (starts from 1)

        or

        (new_ks_homo_alpha_index, new_ks_homo_beta_index): The new KS HOMOs for alpha and beta channels

    """
    num_states = int(data_dim / 2)
    # Check the input
    if (isUKS and num_occ_alpha == 0) and (isUKS and num_occ_beta == 0):
        raise ValueError('please specify num_occ_alpha and num_occ_beta')
    if isinstance(ks_homo_index, list):  # isUKS = True
        homo_alpha, homo_beta = ks_homo_index
    elif isinstance(ks_homo_index, int):  # isUKS = False
        homo_alpha, homo_beta = ks_homo_index, ks_homo_index
    if not isUKS:
        # occupied-side and HOMO-bracketing violations would produce NEGATIVE
        # window indices, which numpy fancy indexing silently wraps to the end
        # of the matrix (beta-block data read as alpha occupied) - raise instead
        if not 1 <= ks_homo_index <= num_states - 1:
            raise ValueError(F"the stored orbital range does not bracket the HOMO "
                             F"(per-spin HOMO index {ks_homo_index} of {num_states} "
                             F"orbitals) - adjust lowest_orbital/highest_orbital in step2")
        if not 1 <= num_occ <= ks_homo_index:
            raise ValueError(F"num_occ={num_occ} but only {ks_homo_index} occupied "
                             F"orbitals exist at/below the HOMO - reduce num_occ_states "
                             F"or lower lowest_orbital in step2")
        if not 1 <= num_unocc <= num_states - ks_homo_index:
            raise ValueError(F"num_unocc={num_unocc} but only "
                             F"{num_states - ks_homo_index} virtuals exist above the "
                             F"HOMO - reduce num_unocc_states or raise highest_orbital "
                             F"in step2")
        occ_indicies_alp = range(ks_homo_index - num_occ, ks_homo_index)
        unocc_indices_alp = range(ks_homo_index, ks_homo_index + num_unocc)
        occ_indicies_bet = range(ks_homo_index - num_occ + num_states, ks_homo_index + num_states)
        unocc_indices_bet = range(ks_homo_index + num_states, ks_homo_index + num_unocc + num_states)
        new_active_space = list(occ_indicies_alp) + list(unocc_indices_alp)
        new_active_space += list(occ_indicies_bet) + list(unocc_indices_bet)
        new_ks_homo_index = num_occ
        return new_active_space, new_ks_homo_index
    elif isUKS:
        # per-channel bounds
        if not 1 <= homo_alpha <= num_states - 1 or not 1 <= homo_beta <= num_states - 1:
            raise ValueError(F"the stored orbital range does not bracket the HOMOs "
                             F"(per-spin HOMO indices alpha={homo_alpha}, "
                             F"beta={homo_beta} of {num_states} orbitals) - adjust "
                             F"lowest_orbital/highest_orbital in step2")
        if not 1 <= num_occ_alpha <= homo_alpha or not 1 <= num_occ_beta <= homo_beta:
            raise ValueError(F"num_occ_alpha={num_occ_alpha}, num_occ_beta={num_occ_beta} "
                             F"but only {homo_alpha}/{homo_beta} occupied orbitals exist "
                             F"at/below the HOMOs")
        if (not 1 <= num_unocc_alpha <= num_states - homo_alpha
                or not 1 <= num_unocc_beta <= num_states - homo_beta):
            raise ValueError(F"num_unocc_alpha={num_unocc_alpha}, "
                             F"num_unocc_beta={num_unocc_beta} but only "
                             F"{num_states - homo_alpha}/{num_states - homo_beta} "
                             F"virtuals exist above the HOMOs")
        if not num_occ_alpha + num_unocc_alpha == num_occ_beta + num_unocc_beta:
            raise ValueError("the number of orbitals in both spin channels must be the same")
        # Alpha channel
        occ_indicies_alp = range(homo_alpha - num_occ_alpha, homo_alpha)
        unocc_indices_alp = range(homo_alpha, homo_alpha + num_unocc_alpha)
        new_active_space = list(occ_indicies_alp) + list(unocc_indices_alp)
        # Beta channel
        occ_indicies_bet = range(homo_beta - num_occ_beta + num_states, homo_beta + num_states)
        unocc_indices_bet = range(homo_beta + num_states, homo_beta + num_unocc_beta + num_states)
        new_active_space += list(occ_indicies_bet) + list(unocc_indices_bet)
        new_ks_homo_alpha_index = num_occ_alpha
        new_ks_homo_beta_index = num_occ_beta
        return new_active_space, new_ks_homo_alpha_index, new_ks_homo_beta_index


def make_cost_mat(orb_mat_inp, en_mat_inp, alpha):
    """

    Makes the cost matrix from a given TDM and information on states' energies

    Args:
        orb_mat_inp  ( CMATRIX(nstates,nstates) or MATRIX(nstates,nstate) ): the transition density matrix
            TDM in a given basis. Here, ```nstates``` - the number of states (e.g. the number of doubly-occupied
            orbitals )

        en_mat_inp ( MATRIX(nstates, nstates) ): Matrix of energies in a given basis [units: a.u.]

        alpha ( float ): Parameter controlling the range of the orbitals that can participate in
            the reordering. Setting is to 0 makes all orbitals be considered for reordering
            Setting it to a large number makes the effective number of orbitals participating
            in the reordering smaller - this can be used to turn off the reordering. [units: a.u.^-1]

    Returns:
        MATRIX(nstates, nstates): the matrix of the cost values for different pairs of states

    """

    nstates = orb_mat_inp.num_of_cols
    cost_mat = MATRIX(nstates, nstates)

    for a in range(0, nstates):
        for b in range(0, nstates):

            s = orb_mat_inp.get(a, b)
            s2 = (s * s.conjugate()).real
            dE = (en_mat_inp.get(a, a) - en_mat_inp.get(b, b)).real
            val = s2 * math.exp(-(alpha * dE)**2)
            cost_mat.set(a, b, val)

    return cost_mat


def _spin_block_permutation(perm_aa, perm_bb, nstates):
    """The 2n-index permutation of a spin-block matrix from its alpha and beta block permutations."""
    p = intList()
    for k in range(nstates):
        p.append(perm_aa[k])
    for k in range(nstates):
        p.append(nstates + perm_bb[k])
    return p


def apply_state_reordering(St, E, params):
    """

    Performs the state's identity reordering in a given basis for all time steps.
    This is reflects in the corresponding changess of the TDM.

    Args:
        St ( list of CMATRIX(nstates, nstates) ): TDM for each timestep
        E ( list of CMATRIX(nstates, nstates) ): energies of all states at every step
        params ( dictionary ): parameters controlling the reordering
            * **params["do_state_reordering"]** ( int ): option to select the state reordering algorithm
                Available options:
                    - 1: older version developed by Kosuke Sato, may not the working all the times
                    - 2: Munkres-Kuhn (Hungarian) method [default]

            * **params["state_reordering_alpha"]** ( double ): a parameter that controls how
                many states will be included in the reordering

    Returns:
        list of lists of int: the label permutation of every step - E[i] (and the rows of
        St[i]) were permuted with entry i, the trailing E entries (beyond len(St)) with the
        final one. Apply it to any other per-step quantity of the same basis. St and E are
        changed in place.
    """

    critical_params = []
    default_params = {"do_state_reordering": 2, "state_reordering_alpha": 0.0}
    comn.check_input(params, default_params, critical_params)

    nsteps = len(St)
    nstates = int(St[0].num_of_cols / 2)  # division by 2 because it is a super-matrix

    alp = list(range(0, nstates))
    bet = list(range(nstates, 2 * nstates))

    # Initialize the cumulative permutation as the identity permutation
    perm_cum_aa = intList()  # cumulative permutation for alpha spatial orbitals
    perm_cum_bb = intList()  # cumulative permutation for beta  spatial orbtials
    for a in range(0, nstates):
        perm_cum_aa.append(a)
        perm_cum_bb.append(a)

    # Current permutation
    perm_t_aa = intList()
    perm_t_bb = intList()
    for a in range(0, nstates):
        perm_t_aa.append(a)
        perm_t_bb.append(a)

    # full-matrix cumulative permutation (algo 1) and the per-step label permutations
    perm_cum = intList()
    for a in range(0, 2 * nstates):
        perm_cum.append(a)
    perms = []

    # Temporary matrices for the Hungarian method
    aa = CMATRIX(nstates, nstates)
    ab = CMATRIX(nstates, nstates)
    ba = CMATRIX(nstates, nstates)
    bb = CMATRIX(nstates, nstates)

    en_mat_aa = CMATRIX(nstates, nstates)
    en_mat_bb = CMATRIX(nstates, nstates)

    for i in range(0, nsteps):

        if params["do_state_reordering"] == 1:
            """
            A simple approach based on permuations - but this is not robust
            may have loops
            """
            perm_t = get_reordering(St[i])

            # apply the cumulative permutation
            update_permutation(perm_t, perm_cum)

            # apply the permutation
            # Because St = <psi(t)|psi(t+dt)> - we permute only columns
            St[i].permute_cols(perm_cum)

            E[i].permute_cols(perm_cum)
            E[i].permute_rows(perm_cum)
            perms.append([perm_cum[k] for k in range(2 * nstates)])

        elif params["do_state_reordering"] == 2:
            """
            The Hungarian approach
            """
            pop_submatrix(St[i], aa, alp, alp)
            pop_submatrix(St[i], ab, alp, bet)
            pop_submatrix(St[i], ba, bet, alp)
            pop_submatrix(St[i], bb, bet, bet)

            # the energies carry the row labels - unpermuted, a state relabeled at a
            # crossing would keep the other state's energy
            perm_full = _spin_block_permutation(perm_t_aa, perm_t_bb, nstates)
            E[i].permute_rows(perm_full)
            E[i].permute_cols(perm_full)
            perms.append([perm_full[k] for k in range(2 * nstates)])

            # Extract the alpha and beta orbtial energies
            pop_submatrix(E[i], en_mat_aa, alp, alp)
            pop_submatrix(E[i], en_mat_bb, bet, bet)

            # Permute rows
            aa.permute_rows(perm_t_aa)
            ab.permute_rows(perm_t_aa)
            ba.permute_rows(perm_t_bb)
            bb.permute_rows(perm_t_bb)

            # compute the cost matrices for diagonal blocks
            cost_mat_aa = make_cost_mat(aa, en_mat_aa, params["state_reordering_alpha"])
            cost_mat_bb = make_cost_mat(bb, en_mat_bb, params["state_reordering_alpha"])

            # Solve the optimal assignment problem for diagonal blocks
            res_aa = hungarian.maximize(cost_mat_aa)
            res_bb = hungarian.maximize(cost_mat_bb)

            # Convert the list of lists into the permutation object
            for ra in res_aa:
                perm_t_aa[ra[0]] = ra[1]  # for < alpha | alpha > this becomes a new value: perm_t = P_{n+1}
            for rb in res_bb:
                perm_t_bb[rb[0]] = rb[1]  # for < beta | beta > this becomes a new value: perm_t = P_{n+1}

            # Permute the blocks by col
            aa.permute_cols(perm_t_aa)
            ab.permute_cols(perm_t_bb)
            ba.permute_cols(perm_t_aa)
            bb.permute_cols(perm_t_bb)

            # Reconstruct St matrix
            push_submatrix(St[i], aa, alp, alp)
            push_submatrix(St[i], ab, alp, bet)
            push_submatrix(St[i], ba, bet, alp)
            push_submatrix(St[i], bb, bet, bet)

    # the trailing step (E has one entry more than St) takes the final labels
    if params["do_state_reordering"] == 1:
        final = perm_cum
    else:
        final = _spin_block_permutation(perm_t_aa, perm_t_bb, nstates)
    for i in range(nsteps, len(E)):
        E[i].permute_rows(final)
        E[i].permute_cols(final)
        perms.append([final[k] for k in range(2 * nstates)])
    return perms


def do_phase_corr(cum_phase1, St, cum_phase2, phase_i):
    """

    This function changes the St matrix according to
    the previous cumulative phases and the current
    phase correction as:

    St = <bra|ket>

    St -> St = F_n * St * (F_{n+1})^+ = F_n * St * (F_{n})^+ * (f_{n+1})^+

    Args:
        cum_phase1 ( CMATRIX(nstates, 1) ): cumulative phase corrections up to step n (F_n) for bra-vectors
        St         ( CMATRIX(nstates, nstates) ): input/output TDM to be processed:
            could be alpha-alpha, beta-beta, alpha-beta, or beta-alpha sub-blocks
        cum_phase2 ( CMATRIX(nstates, 1) ): cumulative phase corrections up to step n (F_n) for ket-vectors
        phase_i    ( CMATRIX(nstates, 1) ): the current step phase corrections (f_{n+1}) for a given pair of vectors


    Returns:
        None: but changes the input matrix St

    """

    nstates = St.num_of_rows

    ### Correct the TDM matrix ###
    for a in range(0, nstates):
        for b in range(0, nstates):
            fab = cum_phase1.get(a) * cum_phase2.get(b).conjugate() * phase_i.get(b).conjugate()
            St.scale(a, b, fab)


def _check_orbital_window(params, data_dim, where):
    """The step-2 npz supermatrices are 2*(highest_orbital - lowest_orbital
    + 1) on a side (alpha+beta blocks, both spins share the window), so the
    claimed window is checked against the stored data; a res-dir
    orbital_window.json written by the producers additionally pins the
    absolute window, which the dimension alone cannot (an equal-width
    shift moves the HOMO row silently).
    """
    l = int(params['lowest_orbital'])
    h = int(params['highest_orbital'])
    meta_file = os.path.join(params['path_to_npz_files'], 'orbital_window.json')
    if os.path.exists(meta_file):
        with open(meta_file) as f:
            meta = json.load(f)
        if int(meta['lowest_orbital']) != l or int(meta['highest_orbital']) != h:
            raise ValueError(
                F"{where}: lowest_orbital/highest_orbital [{l}, {h}] disagree "
                F"with the step-2 window [{meta['lowest_orbital']}, "
                F"{meta['highest_orbital']}] recorded in {meta_file} - the HOMO "
                F"would be located at the wrong matrix row and the active space "
                F"silently shifted; pass the stored window and select the "
                F"physics window via the num_occ/num_unocc counts")
    nw = 2 * (h - l + 1)
    if int(data_dim) != nw:
        raise ValueError(
            F"{where}: the stored npz matrices are {data_dim} x {data_dim} but "
            F"lowest_orbital/highest_orbital [{l}, {h}] imply {nw} "
            F"spin-orbitals - these params must be the step-2 stored window "
            F"(the physics window is chosen by the num_occ/num_unocc counts)")


def limit_active_space(params):
    """
    This function limits the active space of Kohn-Sham matrices
    generated in step 2 based on the number of
    occupied and unoccupied orbitals specified by user.
    Args:
        params (dict):
            num_occ_orbitals (integer): The number of occupied orbitals
            num_unocc_orbitals (integer): The number of unoccupied orbitals
            path_to_npz_files (string): The full path to res directory from step 2
            path_to_save_npz_files (string): The full path to save the Kohn-Sham files
            logfile_directory (string): The full path to all log files
            lowest_orbital (integer): The lowest_orbital defined in step 2
            highest_orbital (integer): The highest_orbital defined in step 2
    Returns:
        l2 (integer): The new value of lowest_orbital
        h2 (integer): the new value of highest_orbital
    """
    nocc = params['num_occ_orbitals']
    nunocc = params['num_unocc_orbitals']
    l1 = params['lowest_orbital']
    h1 = params['highest_orbital']
    sample_logfile = glob.glob(f'{params["logfile_directory"]}/*.log')[0]
    sample_npz_file = glob.glob(f'{params["path_to_npz_files"]}/*.npz')[0]
    _check_orbital_window(params, int(sp.load_npz(sample_npz_file).shape[0]),
                          'limit_active_space')
    os.system(f'mkdir {params["path_to_save_npz_files"]}')
    # currently working only for restricted KS
    homo_index = CP2K_methods.read_homo_index(sample_logfile, isUKS=False)
    if not l1 <= homo_index < h1:
        raise ValueError(F"the stored orbital window [{l1}, {h1}] does not bracket the "
                         F"HOMO ({homo_index}) - lowest_orbital/highest_orbital must be "
                         F"the step2 window")
    if not 1 <= nocc <= homo_index - l1 + 1:
        raise ValueError(F"num_occ_orbitals={nocc} but the stored window holds only "
                         F"{homo_index - l1 + 1} occupied orbitals - a negative slice "
                         F"start would silently wrap to the end of the matrices")
    if not 1 <= nunocc <= h1 - homo_index:
        raise ValueError(F"num_unocc_orbitals={nunocc} but the stored window holds only "
                         F"{h1 - homo_index} virtuals above the HOMO")
    l2 = homo_index - nocc + 1
    h2 = homo_index + nunocc
    l2_index = l2 - l1
    h2_index = h2 - l1 + 1
    norbitals = nocc + nunocc
    zero_mat = np.zeros((norbitals, norbitals))
    for prop in ['S', 'St', 'E']:
        files = glob.glob(f'{params["path_to_npz_files"]}/{prop}_*.npz')
        print(f'Restricting the active space for {prop}..')
        for file in files:
            mat = sp.load_npz(file).todense().real
            new_mat = mat[l2_index:h2_index, :][:, l2_index:h2_index]
            new_mat_two_spinor = data_conv.form_block_matrix(new_mat, zero_mat, zero_mat, new_mat)
            new_mat_sparse = sp.csc_matrix(new_mat_two_spinor)
            step = int(file.split('/')[-1].replace(prop, '').replace('_',
                       '').replace('ks', '').replace('.', '').replace('npz', ''))
            sp.save_npz(f'{params["path_to_save_npz_files"]}/{prop}_ks_{step}.npz', new_mat_sparse)
    with open(f'{params["path_to_save_npz_files"]}/orbital_window.json', 'w') as f:
        json.dump({'lowest_orbital': l2, 'highest_orbital': h2}, f)
    print('Done with limiting the active space')
    print(f'Your new path_to_npz_files is now: {params["path_to_save_npz_files"]}')
    print(f'Use the new lowest_orbital: {l2} and highest_orbital: {h2}')

    return l2, h2


def apply_phase_correction(St):
    """Performs the phase correction according to:
    Akimov, A. V. J. Phys. Chem. Lett, 2018, 9, 6096

    Args:
        St ( list of CMATRIX(N,N) ): St_ij[n] = <i(n)|j(n+1)> transition density matrix for
            the timestep n, where N is the number of spin-orbitals in the active space.
            Spin-orbitals, not just orbitals! So it is composed as:

            .. math::
                St =
                \\begin{vmatrix}
                St_{aa}  & St_{ab} \\\
                St_{ba}  & St_{bb}
                \\end{vmatrix}

    Returns:
        None: but changes the input St matrices

    """

    nsteps = len(St)
    nstates = int(St[0].num_of_cols / 2)  # division by 2 because it is a super-matrix

    alp = list(range(0, nstates))
    bet = list(range(nstates, 2 * nstates))

    ### Initiate the cumulative phase correction factors ###
    cum_phase_aa = CMATRIX(nstates, 1)  # F(n-1)  cumulative phase
    cum_phase_bb = CMATRIX(nstates, 1)  # F(n-1)  cumulative phase
    for a in range(0, nstates):
        cum_phase_aa.set(a, 0, 1.0 + 0.0j)
        cum_phase_bb.set(a, 0, 1.0 + 0.0j)

    St_aa = CMATRIX(nstates, nstates)
    St_ab = CMATRIX(nstates, nstates)
    St_ba = CMATRIX(nstates, nstates)
    St_bb = CMATRIX(nstates, nstates)

    for i in range(0, nsteps):

        pop_submatrix(St[i], St_aa, alp, alp)
        pop_submatrix(St[i], St_bb, bet, bet)
        pop_submatrix(St[i], St_ab, alp, bet)
        pop_submatrix(St[i], St_ba, bet, alp)

        ### Compute the instantaneous phase correction factors for diag. blocks ###
        phase_i_aa = compute_phase_corrections(St_aa)   # f(i)
        phase_i_bb = compute_phase_corrections(St_bb)   # f(i)

        ### Do the  phase correstions for the diag. blocks ###
        do_phase_corr(cum_phase_aa, St_aa, cum_phase_aa, phase_i_aa)
        do_phase_corr(cum_phase_bb, St_bb, cum_phase_bb, phase_i_bb)

        ### Do the  phase correstions for the off-diag. blocks ###
        do_phase_corr(cum_phase_aa, St_ab, cum_phase_bb, phase_i_bb)
        do_phase_corr(cum_phase_bb, St_ba, cum_phase_aa, phase_i_aa)

        ### Push the corrected diag. blocks to orig. St matrix ###
        push_submatrix(St[i], St_aa, alp, alp)
        push_submatrix(St[i], St_ab, alp, bet)
        push_submatrix(St[i], St_ba, bet, alp)
        push_submatrix(St[i], St_bb, bet, bet)

        ### Update the cumulative phase correction factors for diag. blocks ###
        for a in range(0, nstates):
            cum_phase_aa.scale(a, 0, phase_i_aa.get(a))
            cum_phase_bb.scale(a, 0, phase_i_bb.get(a))


def apply_phase_correction_scipy(St, step, cum_phase_aa, cum_phase_bb, two_spinor_format=False):
    """
    This function is exactly the same as apply_phase_correction but for scipy sparse
    time-overlap matrices. The only difference is that we return St and cumulative
    phase correction factors to be used for the next step.

    Args:
        St (scipy.sparse): The time-overlap matrix but in sparse format (csc_matrix, csr_matrix, etc)
        step (integer): The time-step of the MD.
        cum_phase_aa (numpy array): The numpy array for cumulative phase factors for alpha-spin orbitals.
        cum_phase_bb (numpy array): The numpy array for cumulative phase factors for beta-spin orbitals.
        two_spinor_format (bool): The flag for two-spinor format. The False and True values are equivalent to
                                  apply_phase_correction_general and apply_phase_correction respectively.
    Returns:
        St_phase_corrected (scipy.sparse): The phase-correctedtime-overlap matrix
                                           but in sparse format (csc_matrix, csr_matrix, etc)
        cum_phase_aa (numpy array): The numpy array for cumulative phase factors for alpha-spin orbitals for this step.
        cum_phase_bb (numpy array): The numpy array for cumulative phase factors for beta-spin orbitals for this step.
    """
    # nsteps = len(St)
    if two_spinor_format:
        nstates = int(St.shape[0] / 2)  # division by 2 because it is a super-matrix
    else:
        nstates = int(St.shape[0])

    alp = list(range(0, nstates))
    if two_spinor_format:
        bet = list(range(nstates, 2 * nstates))

    St_aa = St[alp, :][:, alp]
    if two_spinor_format:
        St_ab = St[alp, :][:, bet]
        St_ba = St[bet, :][:, alp]
        St_bb = St[bet, :][:, bet]

    ### Compute the instantaneous phase correction factors for diag. blocks ###
    phase_i_aa = compute_phase_corrections_scipy(St_aa)   # f(i)
    if two_spinor_format:
        phase_i_bb = compute_phase_corrections_scipy(St_bb)   # f(i)

    ### Do the  phase correstions for the diag. blocks ###
    St_aa = do_phase_corr_scipy(cum_phase_aa, St_aa, cum_phase_aa, phase_i_aa)
    if two_spinor_format:
        St_bb = do_phase_corr_scipy(cum_phase_bb, St_bb, cum_phase_bb, phase_i_bb)

        ### Do the  phase correstions for the off-diag. blocks ###
        St_ab = do_phase_corr_scipy(cum_phase_aa, St_ab, cum_phase_bb, phase_i_bb)
        St_ba = do_phase_corr_scipy(cum_phase_bb, St_ba, cum_phase_aa, phase_i_aa)

    ### Push the corrected diag. blocks to orig. St matrix ###
    St_phase_corrected = sp.csc_matrix(St_aa.todense())
    if two_spinor_format:
        St_phase_corrected = sp.csc_matrix(
            data_conv.form_block_matrix(
                St_aa.todense(),
                St_ab.todense(),
                St_ba.todense(),
                St_bb.todense()))

    ### Update the cumulative phase correction factors for diag. blocks ###
    cum_phase_aa = np.multiply(cum_phase_aa, phase_i_aa)
    if two_spinor_format:
        cum_phase_bb = np.multiply(cum_phase_bb, phase_i_bb)

    return St_phase_corrected, cum_phase_aa, cum_phase_bb


def do_phase_corr_scipy(cum_phase1, St, cum_phase2, phase_i):
    """
    This function is exactly the same as do_phase_corr but it can be used for
    scipy.sparse matrices.
    Args:
        cum_phase1 (numpy array): cumulative phase corrections up to step n (F_n) for bra-vectors.
        St (scipy.sparse): The time-overlap matrix in csc_matrix, csr_matrix, etc format.
        cum_phase2 (numpy array): cumulative phase corrections up to step n (F_n) for ket-vectors.
        phase_i (numpy array): the current step phase corrections (f_{n+1}) for a given pair of vectors
    Returns:
        St (scipy.sparse): The phase corrected time-overlap.
    """
    f_n_nplus = np.multiply(cum_phase2, phase_i)
    fab_mat = np.kron(cum_phase1, f_n_nplus.T)
    St = St.multiply(fab_mat)

    return St


def compute_phase_corrections_scipy(St):
    """
    This function is the Python version of the C++ compute_phase_corrections function.
    Args:
        St (scipy.sparse): The time-overlap matrix in csc_matrix, csr_matrix, etc format.
    Returns:
        phase_corr (numpy array): The phase correction factors.
    """
    nstates = St.shape[0]
    phase_corr = np.ones((nstates, 1))
    tol = 1.0e-3

    for i in range(nstates):
        f = St[i, i]
        af = np.linalg.norm(f)
        if af > tol:
            phase_corr[i, 0] = f / af

    return phase_corr


def sac_matrices(coeff, basis, S_ks):
    """
    This function makes the Phi-to-Chi (P2C) transformation matrix.
    Normalization factros for the Chi states are computed based on the
    overlaps of Phi states.
    < Chi_i | Chi_j > = 1
    = N_i * N_j * < Phi_i - Phi_i' | Phi_j - Phi_j' > = 1

    coeff [List of lists] - P2C as initialized by the user
    basis [Phi basis] - as initialized by the user
    S_ks  [CMATRIX] - Time overlap matrix of elementary KS orbtials, from step2
    """

    n_chi = len(coeff)
    n_phi = len(coeff[0])

    P2C = CMATRIX(n_phi, n_chi)
    for j in range(0, n_chi):
        for i in range(0, n_phi):
            P2C.set(i, j, coeff[j][i] * (1.0 + 0.0j))

    # Compute the overlaps of the SDs:
    #
    # ovlp_mat_arb returns an ndarray - lift it for the CMATRIX algebra
    Ssd = data_conv.nparray2CMATRIX(np.asarray(mapping.ovlp_mat_arb(basis, basis, S_ks)).astype(complex))

    # Normalize the Chi wavefunctions #
    norm = (P2C.H() * Ssd * P2C).real()
    for i in range(0, n_chi):
        if norm.get(i, i) > 0.0:
            P2C.scale(-1, i, 1.0 / math.sqrt(norm.get(i, i)))
        else:
            raise ValueError("CHI normalization: some combination gives zero norm")

    return P2C


def scale_H_vib(hvib, en_gap, dNAC, sc_nac_method):
    """
    This function scales the energies and NACs in the vibrionic Hamiltonian
    in the Chi basis.

    Args:
        hvib ( list of CMATRIX objects ):
            CMATRIXlist of vibronic hamiltonians in the Chi basis
        en_gap ( float ): The desired energy gap (E_1 - E_0), for the Chi basis
        dNAC ( list of lists of (list, float) ):
            The scaling terms by which specific nacs will
            be scaled datatype = list of lists of (list, float)

            [  [ [i,j], val ], ...  ]

            n and n+1 are the col (and thereby row) indicies of
            the nacs to be scaled by the value val
        sc_nac_method ( int ): The method used to scale NACs in the Chi basis,
            chosen by the user.
            If sc_nac_method = 1, then the NACs are scaled by the ivnerse of the
            magnitude of the change in energy, according to Lin et al.

            Reference: Lin, Y. & Akimov, A. V. J. Phys. Chem. A (2016)
    """

    traj_len = len(hvib)

    # Do the scaling
    nrows = hvib[0].num_of_rows
    ncols = hvib[0].num_of_cols
    for i in range(0, traj_len):

        prev_gap = hvib[i].get(1, 1) - hvib[i].get(0, 0)
        shift = en_gap - prev_gap

        # Scale the energy gap, by adding the scaling factors to all excited states
        # This is to keep the ground state enegy equal to 0.0 by definition
        for j in range(1, ncols):
            hvib[i].add(j, j, shift)

        if sc_nac_method == 0:

            # Scales nacs manually as set by user
            for it in dNAC:
                a, b = it[0][0], it[0][1]
                val = it[1]
                hvib[i].scale(a, b, val)
                hvib[i].scale(b, a, val)

        elif sc_nac_method == 1:
            # Scales nacs by the inverse of the change in energy gap
            for j in range(1, ncols):
                scl_nac = prev_gap / (shift + prev_gap)
                hvib[i].scale(0, j, scl_nac)
                hvib[i].scale(j, 0, scl_nac)

    return hvib


def compute_Hvib(basis, St_ks, E_ks, dE, dt):
    """Compute the vibronic Hamiltonian matrix

    Args:
        basis ( list of lists of integers ): defines the basis of Slater Determinants,
            such that: basis[iSD][iks] is the indicator of the spin-orbital occupied by
            the electron iks in the Slater Determinant iSD

            Example:

                The following example defines a ground state SD (the lowest KS of the active space) and two
                single excitations, which are different from each other by two spin flips of the electrons
                The convention is to start indexing from 1 (corresponds to index 0 in the KS matrices)
                Positive - for alpha electrons, negative - for beta electrons
                Need to be consistent: [ -1, 2 ] and [ 2, -1 ] are treated differently, this is needed for spin-adaptation

                >> basis = [ [ 1,-1 ], [ 1,-2 ], [ 2,-1 ] ]

                The next example is for a system of 4 electrons and hole excitations
                >> basis = [ [ 1,-1, 2, -2 ], [ 3, -1, 2, -2 ], [ 1, -3, 2, -2 ] ]


        St_ks ( CMATRIX(2*norbs, 2*norbs) ): transition density matrix in the KS spin-orbitals basis, where
            norb - the number of double-occupied orbitals.

        E_ks ( CMATRIX(2*norbs, 2*norbs) ): the orbital energies in the KS spin-orbitals basis, where
            norb - the number of double-occupied orbitals.

        dE ( list of doubles ): define corrections of the SD state energies in comparison to
            the energy give by the sum energies of the occupied spin-orbitals.
            The convention is: dE[iSD] is the correction to energy of the SD with index iSD.
            This is a constant correction - same for all energies in the set [units: Ha]

            Example:
                For instance, for the SD examples above, the corrections could be:
                >> dE = [0.0, 0.01, 0.05]

        dt ( double ): the timestep for MD integrations [units: a.u.]

    Returns:
        CMATRIX(nstates, nstates) The Vibronic Hamiltonian

    """

    # ovlp_mat_arb returns a real ndarray - form the antisymmetric part in
    # numpy and lift it into the CMATRIX vibronic Hamiltonian
    St = np.asarray(mapping.ovlp_mat_arb(basis, basis, St_ks))
    H_el = mapping.energy_mat_arb(basis, E_ks, dE)
    H_vib = H_el - (0.5j / dt) * data_conv.nparray2CMATRIX((St - St.T).astype(complex))

    return H_vib


def run(S_dia_ks, St_dia_ks, E_dia_ks, params):
    """
    The procedure to converts the results of QE calculations (KS orbital energies and
    time-overlaps = transition density matrices in the KS basis) to the generic Hvib matrices,
    which (optionally) account for:

    - enforces orthogonalization of the input KS states
    - state reordering
    - phase corrections
    - multi-electron wavefunction (Slater determinants) and spin-adaptation

    Args:
        S_dia_ks ( list of lists of CMATRIX objects ): overlaps of the KS orbitals along trajectories
            for each data set. Such that S_dia_ks[idata][istep].get(i,j) is <i(istep)|j(istep)> for the
            trajectory (=dataset) ```idata```.

        St_dia_ks ( list of lists of CMATRIX objects ): time-overlaps (=transition density matrices)
            in the basis of the KS orbitals along trajectories for each data set.
            Such that St_dia_ks[idata][istep].get(i,j) is <i(istep)|j(istep+1)> for the trajectory (=dataset) ```idata```.

        E_dia_ks ( list of lists of CMATRIX objects ): energies the KS orbitals at the mid-points along trajectories
            for each data set. Such that E_dia_ks[idata][istep].get(i,i) is 0.5*(E_i(istep) + E_i(istep+1)) for the
            trajectory (=dataset) ```idata```

        params ( dictionary ): Control paramerter of this type of simulation. Can include the follwing keys:

            * **params["SD_basis"]** ( list of lists of ints ): define the Slater Determinants basis
                The convention is:  params["SD_basis"][iSD][iks] is the indicator of the spin-orbital occupied by
                the electron iks in the Slater Determinant iSD [required!]

                Example:

                    The following example defines a ground state SD (the lowest KS of the active space) and two
                    single excitations, which are different from each other by two spin flips of the electrons
                    The convention is to start indexing from 1 (corresponds to index 0 in the KS matrices)
                    Positive - for alpha electrons, negative - for beta electrons
                    Need to be consistent: [ -1, 2 ] and [ 2, -1 ] are treated differently, this is needed for spin-adaptation

                    >> params["SD_basis"] = [ [ 1,-1 ], [ 1,-2 ], [ 2,-1 ] ]

                    The next example is for a system of 4 electrons and hole excitations
                    >> params["SD_basis"] = [ [ 1,-1, 2, -2 ], [ 3, -1, 2, -2 ], [ 1, -3, 2, -2 ] ]


            * **params["SD_energy_corr"]** ( list of doubles ): define corrections of the SD state energies in comparison to
                the energy give by the sum energies of the occupied spin-orbitals.
                The convention is: params["SD_energy_corr"][iSD] is the correction to energy of the SD with index iSD.
                This is a constant correction - same for all energies in the set [units: Ha] [required!]

                Example:
                    For instance, for the SD examples above, the corrections could be:
                    >> params["SD"] = [0.0, 0.01, 0.05]

            * **params["CI_basis"]** ( list of lists of complex number ): configuration interaction coefficients
                that define a superpositions to SDs that are considered the states of interest, e.g. spin-adapted configurations
                The convention is: params["CI_basis"][iCI][iSD] is a coefficient of ```iSD```-th SD in the expansion of the CI
                with index ```iCI```. These coefficients don't have to account for the overal CI's normalization - the
                normalization will be done on the go. [required!]

                Example:

                    For the SD example above we can construct the following combinations:
                    >> params["CI_basis"] = [ [1.0, 0.0, 0.0 ],
                                              [0.0, 1.0,-1.0 ],
                                              [0.0, 1.0, 1.0 ]
                                            ]

            * **params["output_set_paths"]** ( list of strings ): the directory pathes where the resulting files
                are to be written (if so!). If you don't plan on writing the files, just provide a list of empty strings
                or whatever else - they will not be used in that case. The number of the strings should be equal to
                the number of the input data sets, e.g. to len(St_dia_ks)  [required!]


            * **params["dt"]** ( double ): nuclear dynamics integration time step [units: a.u. of time, default: 41.0]

            * **params["do_orthogonalization"]** ( int ): the option to do Lowdin orthogonalization of the orbitals - using
                the "raw" overlaps (at the same time). This option is needed because the wavefunction output by QE are
                not exactly orthonormal (because of the use of pseudopotentials). So before we use them (implicitly)
                in the rest of the calculations here, we may need to account for this non-ideality effect.
                Options:

                - 0: don't do the orthogonalization - this is the same as in Pyxaid [default]
                - 1: do the orthogonalization

            * **params["do_state_reordering"]** ( int ): the option to control the state reordering
                Options:

                - 0: no state reordering - same as in Pyxaid
                - 1: older method (is not robust, may or may not work)
                - 2: Hungarian algorithm [default]

            * **params["state_reordering_alpha"]** ( double ): the parameter that controls the width of
                the energy interval within wich the state reordering is in effect. Zero value means all
                available orbitals, larger positive value decreases the width of the window. This parameter
                is not in effect unless the Hungarian algorithm is selected [default: 0.0]

            * **params["do_phase_correction"]** ( int ): option to do the phase correction

                - 0 - don't do
                - 1 - do it [default]

            * **params["do_output"]** ( int ): whether to print out the Hvib matrices ( = 1) to the files or not ( = 0).

            * **params["Hvib_re_prefix"]** ( string ): common prefix of the output files with real part of the vibronic
                Hamiltonian at all times [default: "Hvib_"]

            * **params["Hvib_re_suffix"]** ( string ): common suffix of the output files with real part of the vibronic
                Hamiltonian at all times [default: "_re"]

            * **params["Hvib_im_prefix"]** ( string ): common prefix of the output files with imaginary part of the vibronic
                Hamiltonian at all times [default: "Hvib_"]

            * **params["Hvib_im_suffix"]** ( string ): common suffix of the output files with imaginary part of the vibronic
                Hamiltonian at all times [default: "_im"]

            Note: the _im files/values follow the canonical convention Im(Hvib) = -d = -(St - St^T)/(2 dt); directories written by the current step 3 carry hvib_convention.json, markerless directories are legacy and hold +d (the NAC itself)

    Returns:
        list of lists of CMATRIX(N,N): Hvib, such that:
            Hvib[idata][istep] is a CMATRIX(N,N) containing the vibronic Hamiltonian for the
            trajectory (dataset) ```idata``` and for the timestep ```istep```. Here, N is the number
            of states included in the active space.

    """

    # ====== Defaults and local parameters ===============

    critical_params = ["SD_basis", "SD_energy_corr", "CI_basis", "output_set_paths"]
    default_params = {"dt": 1.0 * units.fs2au,
                      "do_orthogonalization": 0,
                      "do_state_reordering": 2, "state_reordering_alpha": 0.0,
                      "do_phase_correction": 1,
                      "do_output": 0,
                      "Hvib_re_prefix": "Hvib_", "Hvib_im_prefix": "Hvib_",
                      "Hvib_re_suffix": "_re", "Hvib_im_suffix": "_im",

                      }
    comn.check_input(params, default_params, critical_params)

    dt = params["dt"]
    do_orthogonalization = params["do_orthogonalization"]
    do_phase_correction = params["do_phase_correction"]
    do_state_reordering = params["do_state_reordering"]
    ndata = len(St_dia_ks)
    nsteps = len(St_dia_ks[0])
    nstates = len(params["CI_basis"])  # the number of CI states to consider

    do_output = params["do_output"]

    # ====== Generic sanity checks ===============
    if do_output:
        if (ndata) != len(params["output_set_paths"]):
            raise ValueError(F"the number of output set paths "
                             F"({len(params['output_set_paths'])}) must equal the "
                             F"number of data sets ({ndata})")

    # ====== Calculations  ===============
    H_vib = []

    for idata in range(0, ndata):

        # 1. Do the KS orbitals orthogonalization
        if do_orthogonalization > 0:
            apply_normalization(S_dia_ks[idata], St_dia_ks[idata])

        # 2. Apply state reordering to KS
        if do_state_reordering > 0:
            apply_state_reordering(St_dia_ks[idata], E_dia_ks[idata], params)

        # 3. Apply phase correction to KS
        if do_phase_correction > 0:
            apply_phase_correction(St_dia_ks[idata])

        Hvib = []
        for i in range(0, nsteps):

            # 4. Construct the Hvib in the basis of Slater determinants (SDs)
            hvib_sd = compute_Hvib(
                params["SD_basis"],
                St_dia_ks[idata][i],
                E_dia_ks[idata][i],
                params["SD_energy_corr"],
                dt)

            # 5. Convert the Hvib to the basis of symmery-adapted configurations (SAC)
            SD2CI = sac_matrices(params["CI_basis"], params["SD_basis"], S_dia_ks[idata][i])
            hvib_ci = SD2CI.H() * hvib_sd * SD2CI
            Hvib.append(hvib_ci)

        if do_output:
            # Output the resulting Hamiltonians
            for i in range(0, nsteps):
                re_filename = params["output_set_paths"][idata] + \
                    params["Hvib_re_prefix"] + str(i) + params["Hvib_re_suffix"]
                im_filename = params["output_set_paths"][idata] + \
                    params["Hvib_im_prefix"] + str(i) + params["Hvib_im_suffix"]
                Hvib[i].real().show_matrix(re_filename)
                Hvib[i].imag().show_matrix(im_filename)

        H_vib.append(Hvib)

    return H_vib


def map_Hvib(H, basis, dE):
    """Computes a vibronic Hamiltonian matrix in the basis of SD configurations

    This function seems to be deprecated

    Args:
        SD (list of lists of ints ): a list of SD determinants, such that:
            SD[iSD] is a list of integers defining which orbitals are
            occupied in SD with index ```iSD``` and how
            SeeAlso: ```inp``` in the ```sd2indx(inp,nbasis)``` function

        H ( MATRIX(nbasis, nbasis) ): vibronic Hamiltonian in 1-electron (e.g. KS-) basis (alpha- and beta- components)
        dE ( list of doubles ): energy corrections added to each SD

    Returns:
        CMATRIX(N,N): the matrix of energies in the SD basis. Here, N = len(SD) - the number of SDs.

    """

    # First, compute the matrix with energies on the diagonals
    H_vib = mapping.energy_mat_arb(basis, H, dE)

    # Now handle the couplings and nonadiabatic couplings (off-diagonal elements)
    nstates = len(basis)
    for i in range(0, nstates):
        for j in range(0, nstates):

            res = delta(Py2Cpp_int(basis[i]), Py2Cpp_int(basis[j]))

            if res[0] != 0:
                if res[1] * res[2] > 0:
                    # This means the two configurations are coupled, so we take the
                    # corresponding 1-particles matrix elements and insert them here
                    a, _ = mapping.sd2indx([res[1], res[2]], nbasis, False)
                    H_vib.add(i, j, H.get(a[0], a[1]))

    return H_vib


def pyxaid2libra(Hvib_pyxaid, params):
    """
    """

    # ====== Defaults and local parameters ===============

    critical_params = ["SD_basis", "SD_energy_corr", "CI_basis", "output_set_paths"]
    default_params = {"do_output": 0,
                      "Hvib_re_prefix": "Hvib_", "Hvib_im_prefix": "Hvib_",
                      "Hvib_re_suffix": "_re", "Hvib_im_suffix": "_im",
                      }
    comn.check_input(params, default_params, critical_params)

    ndata = len(Hvib_pyxaid)
    nsteps = len(Hvib_pyxaid[0])
    norbs = Hvib_pyxaid[0][0].num_of_cols
    nstates = len(params["CI_basis"])  # the number of CI states to consider
    do_output = params["do_output"]

    S = CMATRIX(2 * norbs, 2 * norbs)
    H = CMATRIX(2 * norbs, 2 * norbs)
    s = CMATRIX(norbs, norbs)
    s.identity()
    alp = list(range(0, norbs))
    bet = list(range(norbs, 2 * norbs))

    push_submatrix(S, s, alp, alp)
    push_submatrix(S, s, alp, bet)
    push_submatrix(S, s, bet, alp)
    push_submatrix(S, s, bet, bet)

    # ====== Generic sanity checks ===============
    if do_output:
        if (ndata) != len(params["output_set_paths"]):
            raise ValueError(F"the number of output set paths "
                             F"({len(params['output_set_paths'])}) must equal the "
                             F"number of data sets ({ndata})")

    # ====== Calculations  ===============
    H_vib = []

    for idata in range(0, ndata):

        Hvib = []
        for i in range(0, nsteps):

            # 1. Construct the Hvib in the basis of Slater determinants (SDs)
            push_submatrix(H, Hvib_pyxaid[idata][i], alp, alp)
            push_submatrix(H, Hvib_pyxaid[idata][i], alp, bet)
            push_submatrix(H, Hvib_pyxaid[idata][i], bet, alp)
            push_submatrix(H, Hvib_pyxaid[idata][i], bet, bet)
            hvib_sd = map_Hvib(H, params["SD_basis"], params["SD_energy_corr"])

            # 2. Convert the Hvib to the basis of symmery-adapted configurations (SAC)
            SD2CI = sac_matrices(params["CI_basis"], params["SD_basis"], S)
            hvib_ci = SD2CI.H() * hvib_sd * SD2CI
            Hvib.append(hvib_ci)

        if do_output:
            # Output the resulting Hamiltonians
            for i in range(0, nsteps):
                re_filename = params["output_set_paths"][idata] + \
                    params["Hvib_re_prefix"] + str(i) + params["Hvib_re_suffix"]
                im_filename = params["output_set_paths"][idata] + \
                    params["Hvib_im_prefix"] + str(i) + params["Hvib_im_suffix"]
                Hvib[i].real().show_matrix(re_filename)
                Hvib[i].imag().show_matrix(im_filename)

        H_vib.append(Hvib)

    return H_vib


def apply_state_reordering_general(St, E, params):
    """
    Performs the state's identity reordering in a given basis for all time steps.
    This is reflects in the corresponding changess of the TDM.

    This function is for dat NOT in spin-block format

    Args:
        St ( list of CMATRIX(nstates, nstates) ): TDM for each timestep
        E ( list of CMATRIX(nstates, nstates) ): energies of all states at every step
        params ( dictionary ): parameters controlling the reordering
            * **params["do_state_reordering"]** ( int ): option to select the state reordering algorithm
                Available options:
                    - 1: older version developed by Kosuke Sato, may not the working all the times
                    - 2: Munkres-Kuhn (Hungarian) method [default]
            * **params["state_reordering_alpha"]** ( double ): a parameter that controls how
                many states will be included in the reordering
    Returns:
        list of lists of int: the label permutation of every step - E[i] (and the rows of
        St[i]) were permuted with entry i, the trailing E entries (beyond len(St)) with the
        final one. Apply it to any other per-step quantity of the same basis. St and E are
        changed in place.
    """

    critical_params = []
    default_params = {"do_state_reordering": 2, "state_reordering_alpha": 0.0}
    comn.check_input(params, default_params, critical_params)

    nsteps = len(St)
    nstates = St[0].num_of_cols

    # Initialize the cumulative permutation as the identity permutation
    perm_cum = intList()  # cumulative permutation for alpha spatial orbitals
    for i in range(0, nstates):
        perm_cum.append(i)

    # Current permutation
    perm_t = intList()
    for i in range(0, nstates):
        perm_t.append(i)
    perms = []

    for i in range(0, nsteps):

        if params["do_state_reordering"] == 1:
            """
            A simple approach based on permuations - but this is not robust
            may have loops
            """
            perm_t = get_reordering(St[i])

            # apply the cumulative permutation
            update_permutation(perm_t, perm_cum)

            # apply the permutation
            # Because St = <psi(t)|psi(t+dt)> - we permute only columns
            St[i].permute_cols(perm_cum)

            E[i].permute_cols(perm_cum)
            E[i].permute_rows(perm_cum)
            perms.append([perm_cum[k] for k in range(nstates)])

        elif params["do_state_reordering"] == 2:
            """
            The Hungarian approach
            """

            # Permute rows
            St[i].permute_rows(perm_t)
            # the energies carry the row labels - unpermuted, a state relabeled at a
            # crossing would keep the other state's energy
            E[i].permute_rows(perm_t)
            E[i].permute_cols(perm_t)
            perms.append([perm_t[k] for k in range(nstates)])
            # compute the cost matrices
            cost_mat = make_cost_mat(St[i], E[i], params["state_reordering_alpha"])

            # Solve the optimal assignment problem for diagonal blocks
            res = hungarian.maximize(cost_mat)

            # Convert the list of lists into the permutation object
            for row in res:
                perm_t[row[0]] = row[1]  # for < i | i > this becomes a new value: perm_t = P_{n+1}

            # Permute the blocks by col
            St[i].permute_cols(perm_t)

    # the trailing step (E has one entry more than St) takes the final labels
    final = perm_cum if params["do_state_reordering"] == 1 else perm_t
    for i in range(nsteps, len(E)):
        E[i].permute_rows(final)
        E[i].permute_cols(final)
        perms.append([final[k] for k in range(nstates)])
    return perms


def apply_phase_correction_general(St):
    """Performs the phase correction according to:
    Akimov, A. V. J. Phys. Chem. Lett, 2018, 9, 6096

    This function is for dat NOT in spin-block format

    Args:
        St ( list of CMATRIX(N,N) ): St_ij[n] = <i(n)|j(n+1)> transition density matrix for
            the timestep n, where N is the number of states in the active space.
            Spin-orbitals, not just orbitals! So it is composed as:

    Returns:
        None: but changes the input St matrices
    """

    nsteps = len(St)
    nstates = St[0].num_of_cols

    ### Initiate the cumulative phase correction factors ###
    cum_phase = CMATRIX(nstates, 1)  # F(n-1)  cumulative phase
    for i in range(0, nstates):
        cum_phase.set(i, 0, 1.0 + 0.0j)

    print("number of steps , nsteps= ", nsteps)

    for i in range(0, nsteps):

        ### Compute the instantaneous phase correction factors for diag. blocks ###
        phase_i = compute_phase_corrections(St[i])  # f(i)

        phase_i.show_matrix()

        ### Do the  phase corrections ###
        do_phase_corr(cum_phase, St[i], cum_phase, phase_i)

        ### Update the cumulative phase correction factors for diag. blocks ###
        for j in range(0, nstates):
            cum_phase.scale(j, 0, phase_i.get(j))


def sort_unique_SD_basis(E_ks, sd_states_unique, sd_states_reindexed, _params):
    """
        This function computes the energies of the SP transitions (according to the sum of 1 electron terms) - no J or K
        It then may sort the order of the sd_states either based on their energy at each timestep

        Args:
            E_ks (list of CMATRIX): KS orbital energies at each timestep. Spin block style
                                                                          Ex)     [ alp*alp  alp*bet ]
                                                                                  [ bet*alp  bet*bet ]
            sd_states_unique (list of lists): all SP transitions and which spin it was
                                              Ex) [ [ ['28 29'], ['alp'] ]. [ ['28 30'], ['alp'] ] ]
            sd_states_reindexed (list of lists): sd_states_unique but in internal  Libra notation
                                                  Ex) [ [1,-1,3,-2], [3,-1,2,-2] ]
            sorting_type ( (string) ): "energy"   - sort by energy
                                       "identity" - sort by identity

            isnap (int): step from which to start counting
            fsnap (int): step at which to stop counting

        Returns:
            E_sd (list of CMATRIX): SD energies at each timestep
            sd_states_unique_sorted (list of lists): All SP transitions and which spin it is, but now sorted either by identity (no sorting) or energy
            sd_states_reindexed_sorted (list of lists): The sd_states_unique_sorted, but in Libra's notation
            reindex_nsteps (list of lists): The energy ordering of the SD for each step in terms of the index of the SD from the initial step
    """

    params = dict(_params)

    critical_params = []
    # Default parameters
    default_params = {"isnap": 0, "fsnap": 1, "sorting_type": "energy"}
    # Check input
    comn.check_input(params, default_params, critical_params)

    istep = params["isnap"]
    fstep = params["fsnap"]
    sorting_type = params["sorting_type"]

    E_sd = []
    sd_states_reindexed_sorted = []
    sd_states_unique_sorted = []
    SD_energy_corr = [0.0] * len(sd_states_reindexed)
    nstates_sd = len(sd_states_reindexed)
    reindex_nsteps = []

    for step in range(fstep - istep):

        # Append a CMATRIX of dimension nstates_sd x nstates_sd
        E_sd.append(CMATRIX(nstates_sd, nstates_sd))

        # At this step, compute the energy of the SD
        E_this_sd = mapping.energy_mat_arb(sd_states_reindexed, E_ks[step], SD_energy_corr)

        # Make a list for the final ordering of the sd_states_unique.
        # This will not contain the ground state, which we will manually add later.
        sd_states_unique_sorted.append([])
        sd_states_reindexed_sorted.append([])

        # Make an array of zeros, these will be overwritten with the energy of each SD
        e = np.zeros(nstates_sd)
        for state in range(nstates_sd):
            e[state] = E_this_sd.get(state, state).real
            # Obtain the indexing fo the SDs by their energies
        reindex = np.argsort(e)

        if sorting_type == "identity":

            reindex_nsteps.append(list(range(nstates_sd)))

            for state in range(nstates_sd):

                # This is making the energy matrix. And the "sorted" sds (written in Libra's input format) are just appened
                # to sd_states_reindexed_sorted[step]. For identity ordering - no energy sorting is done!
                E_sd[step].set(state, state, E_this_sd.get(state, state))
                sd_states_reindexed_sorted[step].append(sd_states_reindexed[state])

                # print( sd_states_reindexed_sorted[step][ state ], ( E_sd[step].get( state, state ) - E_sd[step].get( 0, 0 ) ).real * units.au2ev )

                # This is reindexing the list of SD bases at this time step according to their energies
                # We are adding the ground state SD later, so skip it for now. In this list sd_states_unique,
                # the ground state is not there - this list is the single-particle transitions (and spin) given by the
                # ES software. So, for example, if nstates_sd = 4, we take only the first 3, because the ground state is not
                # in sd_states_unique
                # Ex) sd_states_unique = [  [ ['28,29'], ['alp'] ] , [ ['27,29'], ['alp'] ] , [ ['26,29'], ['alp'] ] ]
                # 28 = homo
                if state < nstates_sd - 1:
                    sd_states_unique_sorted[step].append(sd_states_unique[state])

        elif sorting_type == "energy":

            reindex_nsteps.append(reindex)

            # For each SD basis, make the energy matrix and reindex the list of basis according to their energies
            for i in range(len(reindex)):
                # This is making the energy matrix
                E_sd[step].set(i, i, E_this_sd.get(int(reindex[i]), int(reindex[i])))
                # This is reindexing the list of SD bases at this time step according to their energies
                sd_states_reindexed_sorted[step].append(sd_states_reindexed[int(reindex[i])])
                # print( sd_states_reindexed_sorted[step][i], ( E_sd[step].get( i, i ) - E_sd[step].get( 0, 0 ) ).real * units.au2ev )

            for i in range(1, len(reindex)):
                sd_states_unique_sorted[step].append(sd_states_unique[int(reindex[i]) - 1])

    return E_sd, sd_states_unique_sorted, sd_states_reindexed_sorted, reindex_nsteps


def sort_unique_SD_basis_scipy(step, sd_states_unique, sd_states_reindexed, _params):
    """
    This function is another version of the sort_unique_SD_basis that can work with
    scipy sparse matrices.
    Args:
        step (integer): The MD step number.
        sd_states_unique (list): The list of the unique Slater determinants.
        sd_states_reindexed (list): The list of reindexed Slater determinants.
        _params (dictionary): A dictionary that contains the following parameters.
            Required parameter keys:

            * **params["active_space"]** (list): The active space built from make_active_space
                                                 function in step3.py.
            * **params["path_to_npz_files"]** (string): The full path to raw npz files.
            .. note::
                In addition, requires parameters described in
                :func:`libra_py.workflows.nbra.step3.sort_unique_SD_basis`

    """
    params = dict(_params)

    critical_params = []
    # Default parameters
    default_params = {"isnap": 0, "fsnap": 1, "sorting_type": "energy"}
    # Check input
    comn.check_input(params, default_params, critical_params)

    istep = params["isnap"]
    fstep = params["fsnap"]
    sorting_type = params["sorting_type"]

    SD_energy_corr = [0.0] * len(sd_states_reindexed)
    nstates_sd = len(sd_states_reindexed)
    reindex_nsteps = []

    # At this step, compute the energy of the SD
    active_space = params['active_space']
    E_ks = np.array(sp.load_npz(params['path_to_npz_files'] +
                                F'/E_ks_{step}.npz').todense().real)[active_space, :][:, active_space]
    # E_ks_MATRIX = data_conv.nparray2MATRIX( E_ks )
    # E_this_sd = mapping.energy_mat_arb( sd_states_reindexed, E_ks_MATRIX, SD_energy_corr )
    E_this_sd = mapping.energy_mat_arb(sd_states_reindexed, E_ks, SD_energy_corr)
    # print('Flag E_this_sd before:', E_this_sd)

    # Make a list for the final ordering of the sd_states_unique.
    # This will not contain the ground state, which we will manually add later.
    sd_states_unique_sorted = []
    sd_states_reindexed_sorted = []

    # Make an array of zeros, these will be overwritten with the energy of each SD
    e = np.zeros(nstates_sd)
    for state in range(nstates_sd):
        # e[state] =  E_this_sd.get(state,state).real
        e[state] = E_this_sd[state, state].real
        # Obtain the indexing fo the SDs by their energies
    reindex = np.argsort(e)
    # Turning the CMATRIX into numpy
    E_this_sd = E_this_sd.real  # ()
    # Only the diagonals are needed
    # E_this_sd = np.diag( data_conv.MATRIX2nparray(E_this_sd) )
    E_this_sd = np.diag(E_this_sd)

    if sorting_type == "identity":

        reindex_nsteps.append(list(range(nstates_sd)))

        for state in range(nstates_sd):

            sd_states_reindexed_sorted.append(sd_states_reindexed[state])

            # This is reindexing the list of SD bases at this time step according to their energies
            # We are adding the ground state SD later, so skip it for now. In this list sd_states_unique,
            # the ground state is not there - this list is the single-particle transitions (and spin) given by the
            # ES software. So, for example, if nstates_sd = 4, we take only the first 3, because the ground state is not
            # in sd_states_unique
            # Ex) sd_states_unique = [  [ ['28,29'], ['alp'] ] , [ ['27,29'], ['alp'] ] , [ ['26,29'], ['alp'] ] ]
            # 28 = homo
            if state < nstates_sd - 1:
                sd_states_unique_sorted.append(sd_states_unique[state])

    elif sorting_type == "energy":

        reindex_nsteps.append(reindex)

        # For each SD basis, make the energy matrix and reindex the list of basis according to their energies
        for i in range(len(reindex)):
            # This is reindexing the list of SD bases at this time step according to their energies
            sd_states_reindexed_sorted.append(sd_states_reindexed[int(reindex[i])])
        E_this_sd = E_this_sd[reindex]

        for i in range(1, len(reindex)):
            sd_states_unique_sorted.append(sd_states_unique[int(reindex[i]) - 1])
    # print('Flag E_this_sd:',E_this_sd.shape)
    # print('Flag E_this_sd itself:',E_this_sd)
    # print('Flag np.diag(E_this_sd):',np.diag(E_this_sd))
    E_this_sd_sparse = sp.csc_matrix(np.diag(E_this_sd))

    return E_this_sd_sparse, sd_states_unique_sorted, sd_states_reindexed_sorted, reindex_nsteps


# shared with the TSH dynamics loop (dyn_params["blas_num_threads"]);
# the underscore alias serves the call sites below
from libra_py.blas_threads import blas_thread_limit as _blas_thread_limit


def _worker_blas_threads(params):
    """BLAS threads for one worker.

    An explicit blas_num_threads wins (hybrid mode: nprocs Pool workers x
    blas_num_threads BLAS threads each - worthwhile when the per-step matrices
    are large enough for threaded BLAS). Otherwise: 1 thread per Pool worker,
    or the whole nprocs budget when running serially.
    """
    n = params.get('blas_num_threads', None)
    if n is not None:
        return int(n)
    return 1 if params.get('use_multiprocessing', False) else int(params.get('nprocs', 1))


def _parent_blas_threads(params):
    """Whole-run cap for the serial (parent) phases.

    The parent is idle while a Pool runs, so its serial numpy/Eigen
    stages may use the full nprocs x per-worker budget. An explicit
    blas_num_threads = 0 disables all capping, as everywhere."""
    n = params.get('blas_num_threads', None)
    if n is not None and int(n) == 0:
        return 0
    per_worker = _worker_blas_threads(params)
    if params.get('use_multiprocessing', False):
        return int(params.get('nprocs', 1)) * per_worker
    return per_worker


_IDENTITY_WARNED = set()


def _check_input_overlap(S, tag, step, threshold=0.05, allow_dirty=False):
    """The stored step-2 MO self-overlaps must be near-identity: an
    identity-far S is the Gram matrix of the WRONG vectors (a step-2
    defect - AO mis-mapping, ordering, truncation), and the Lowdin
    repair of the downstream results is only empirical (~1-4% NAC
    agreement measured for the known corruption class, unguaranteed for
    other causes). Raise by default; allow_dirty (the
    allow_nonorthonormal_overlaps parameter) turns it into a
    once-per-process warning for conscious legacy reprocessing.
    Structural invalidity (non-finite, non-Hermitian, non-positive
    diagonal) raises unconditionally - nothing can repair it."""
    validate_overlap_matrix(S, F"{tag} at step {step}")
    dev = orthogonalizations._identity_deviation(S)
    if dev > threshold:
        if not allow_dirty:
            raise ValueError(
                F"{tag} at step {step} deviates from identity by {dev:.3f} "
                F"(threshold {threshold}) - the step-2 MO overlaps are not "
                F"orthonormal, so the stored data describe the wrong "
                F"vectors; results after the Lowdin repair are only "
                F"empirically ~1-4% accurate for the known corruption "
                F"class. Regenerate step 2, or set "
                F"allow_nonorthonormal_overlaps=True to reprocess this "
                F"legacy data consciously")
        if tag not in _IDENTITY_WARNED:
            _IDENTITY_WARNED.add(tag)
            print(F"WARNING: {tag} deviates from identity by {dev:.3f} at "
                  F"step {step} (threshold {threshold}) - reprocessing "
                  F"non-orthonormal legacy data (allow_nonorthonormal_"
                  F"overlaps=True); the Lowdin repair is empirical (~1-4% "
                  F"for the known corruption class) - regenerating step 2 "
                  F"is recommended")
    return dev


def _validate_cmatrix_overlap(S, label):
    """CMATRIX front-end of validate_overlap_matrix for the legacy Lowdin
    implementations, plus the positive-semidefiniteness raise their
    sqrt_matrix stage would otherwise hide (an indefinite S passes the
    rank check and produces garbage silently)."""
    S_np = (data_conv.MATRIX2nparray(S.real()) +
            1j * data_conv.MATRIX2nparray(S.imag()))
    # 8-significant-digit text archives carry ~1e-8-scale round-off
    # asymmetry - the structural Hermiticity test must sit above it
    validate_overlap_matrix(S_np, label, sym_tol=1e-6)
    w_min = float(np.linalg.eigvalsh(0.5 * (S_np + S_np.conj().T)).min())
    if w_min < -1e-8 * max(1.0, float(np.abs(S_np).max())):
        raise ValueError(
            F"{label}: the overlap matrix has a negative eigenvalue "
            F"({w_min:.3e}) - not a valid Gram matrix; corrupted data")


def _check_contraction(St, step, basis, tol=1e-6):
    """An overlap between two orthonormalized frames is U^H V with
    orthonormal columns, so every singular value is <= 1. A violation
    after the Lowdin stage means the two sides were orthonormalized
    against inconsistent metrics (e.g. a wrong-time overlap) and the
    couplings are inflated."""
    smax = float(np.linalg.svd(St, compute_uv=False).max())
    if smax > 1.0 + tol:
        raise ValueError(
            F"orthonormalized {basis} time-overlap at step {step} has "
            F"sigma_max = {smax:.6f} > 1 - impossible for orthonormal "
            F"frames; the orthonormalization metrics are inconsistent")
    return smax


def _with_parent_blas_cap(fn):
    """Cap the whole step-3 run to the parallelization budget: the
    serial parent stages (CI contractions, Hvib assembly, reordering)
    run heavy numpy/Eigen too, and an uncapped BLAS/OpenMP pool there
    oversubscribes past nprocs regardless of the worker caps."""
    @functools.wraps(fn)
    def wrapped(params):
        with _blas_thread_limit(_parent_blas_threads(params)):
            return fn(params)
    return wrapped


@_with_parent_blas_cap
def run_step3_ks_nacs_libint(params):
    """
    This function runs the step3 for computing NACs for Kohn-Sham states.
    It performs phase-correction and orthonormalization for the overlaps.
    For now, we do not include the state reordering algorithm but will add it in
    the future.

    Args:

        params (dictionary):

            * **params['nprocs']** (integer): number of processors to be used.
            * **params['use_multiprocessing']**: the flag to use multiprocessing or not
            * **params['path_to_npz_files']** (string):  the path to the KS orbital overlaps which are stored
                in .npz format in step2.
            * **params['time_step']** (float): the time-step in femtosecond
            * **params['path_to_save_ks_Hvibs']** (string): the path for storing the KS Hvibs
            * **params['path_to_save_sd_Hvibs']** (string): the path for saving the SD Hvibs
            * **params['active_space']** (list): a list that contains the indices of specific states
            * **params['ks_homo_index']** (integer): the index of the KS HOMO + 1 (it starts from 1)
            * **params['data_dim']** (integer): the dimension of the energy/overlap matrices stored as .npz
            * **params['start_time']** (integer): the initial step for reading the files
            * **params['finish_time']** (integer): the final step for reading the files
            * **params['sorting_type']** (string): it performs sorting the elements of the overlap matrices
                either by their 'energy' or 'identity'.
            * **params['apply_phase_correction']** (bool): a True or False flag for performing the phase-correction
            * **params['apply_orthonormalization']**: orthonormalization of the KS overlaps
                (same vocabulary as cp2k_compute_adi):
                    - "lowdin" or True: Lowdin-orthonormalize every step (two fractional
                      matrix powers per step)
                    - "lowdin_cached" [default]: bit-identical result, cheaper - the ket factor
                      S_2^{-1/2} is carried to the next step in a per-process cache (one
                      fractional matrix power per step serially; also hits inside
                      contiguous Pool chunks). This stage historically always
                      orthonormalized, whatever the flag said - that output is preserved
                      exactly.
                    - "none" or False: skip - raw St/S pass through
            * **params['do_state_reordering']** (integer): the value for performing the state reordering
                the vlaues it takes are:
                    - 0: no state reordering - same as in Pyxaid
                    - 1: older method (is not robust, may or may not work)
                    - 2: Hungarian algorithm [default]
            * **params['nac_algo']** ( int ): selection of a method to compute NACs
                    - 0 : Hamess-Shiffer-Tully (HST), finite difference [ default ]
                    - 1 : Meek-Levine NPI
    Returns:

        None
    """
    # We can define the full active space and data dimension automatically
    # by reading just a sample file.
    critical_params = []  # ['active_space','data_dim']
    default_params = {'nprocs': 2, 'path_to_npz_files': os.getcwd() + '/res',
                      'path_to_save_ks_Hvibs': os.getcwd() + '/res-ks',
                      'time_step': 1.0, 'start_time': 0, 'finish_time': 1,
                      'apply_phase_correction': False, 'apply_orthonormalization': 'lowdin_cached',
                      'do_state_reordering': 0, 'state_reordering_alpha': 0, 'es_software': 'cp2k',
                      'nac_algo': 0, 'blas_num_threads': None
                      }
    comn.check_input(params, default_params, critical_params)
    params['apply_orthonormalization'] = _orthonormalization_mode(params['apply_orthonormalization'])
    isUKS = params.get('isUKS', 0)
    if params['es_software'].lower() == 'cp2k':
        sample_logfile = glob.glob(params['logfile_directory'] + '/*.log')[0]
        if isUKS:
            params['homo_alpha_index'], params['homo_beta_index'] = CP2K_methods.read_homo_index(
                sample_logfile, isUKS=True)
            params['npz_file_ks_homo_alpha_index'] = params['homo_alpha_index'] - params['lowest_orbital'] + 1
            params['npz_file_ks_homo_beta_index'] = params['homo_beta_index'] - params['lowest_orbital'] + 1
        else:
            params['homo_index'] = CP2K_methods.read_homo_index(sample_logfile)
    if not isUKS:
        params['npz_file_ks_homo_index'] = params['homo_index'] - params['lowest_orbital'] + 1
    sample_npz_file = glob.glob(params['path_to_npz_files'] + '/*.npz')[0]
    params['data_dim'] = int(sp.load_npz(sample_npz_file).shape[0])
    _check_orbital_window(params, params['data_dim'], 'run_step3_ks_nacs_libint')
    # Number of occupied states
    num_occ_states = params['num_occ_states']
    # Number of unoccupied states
    num_unocc_states = params['num_unocc_states']
    # The dimension of the super-matrix
    data_dim = params['data_dim']
    # Create active space based on the number of occupied and unoccupied states.
    # UKS: each channel's window is built around ITS OWN HOMO (they differ for
    # any radical) - a single-homo active space would misalign the beta block
    if isUKS:
        ks_active_space, ks_homo_index, _ks_homo_index_beta = make_active_space(
            num_occ_states, num_unocc_states, data_dim,
            [params['npz_file_ks_homo_alpha_index'], params['npz_file_ks_homo_beta_index']],
            isUKS=1,
            num_occ_alpha=params.get('num_occ_alpha', num_occ_states),
            num_occ_beta=params.get('num_occ_beta', num_occ_states),
            num_unocc_alpha=params.get('num_unocc_alpha', num_unocc_states),
            num_unocc_beta=params.get('num_unocc_beta', num_unocc_states))
    else:
        # The KS HOMO index in the sparse npz files - this value starts from 1
        npz_file_ks_homo_index = params['npz_file_ks_homo_index']
        ks_active_space, ks_homo_index = make_active_space(
            num_occ_states, num_unocc_states, data_dim, npz_file_ks_homo_index)
    # Append the created active space into params
    params['active_space'] = ks_active_space
    # The new KS HOMO index in the new active space
    params['ks_homo_index'] = ks_homo_index
    # The number of the states in the super-matrix (the two-spinor format)
    nstates = int(len(ks_active_space) / 2)
    # The start time-step
    start_time = params['start_time']
    # The final time-step
    finish_time = params['finish_time']
    # The number of processors
    nprocs = params['nprocs']
    # The dt value is in fs and will be turned into atomic unit
    dt = params['time_step'] * units.fs2au
    # Algorithm to compute NACs:
    nac_algo = params["nac_algo"]
    if nac_algo not in (0, 1):
        raise ValueError(F"nac_algo = {nac_algo} is not a listed option - "
                         F"0 (HST) and 1 (Meek-Levine NPI) are supported")
    # The path to raw npz files that are produced in step2
    res_dir_1 = params['path_to_npz_files']
    # The path to save the KS Hvibs that will be calculated in this function
    res_dir_2 = params['path_to_save_ks_Hvibs']
    # Create the path
    try:
        os.system(F'mkdir {res_dir_2}')
    except BaseException:
        print(F'The directory {res_dir_2} already exists.')
    # the _im files store Im(Hvib) = -d; files WITHOUT this marker predate
    # the convention restoration and hold +d (the NAC itself)
    marker_file = os.path.join(res_dir_2, 'hvib_convention.json')
    if not os.path.exists(marker_file) and glob.glob(os.path.join(res_dir_2, '*_im.npz')):
        raise ValueError(F"{res_dir_2} already holds *_im.npz files without a "
                         F"convention marker (legacy +d data) - writing canonical "
                         F"-d files into it would mix conventions in one directory; "
                         F"use a fresh output path or move the legacy files away")
    with open(marker_file, 'w') as f:
        json.dump({'im_file_contains': 'Im(Hvib) = -d'}, f)
    t2 = time.time()
    # If we're using multiprocessing, create a pool of processors
    if params['use_multiprocessing']:
        var_pool = []
        for step in range(start_time, finish_time):
            var_pool.append((step, params))

        with _MP_CTX.Pool(nprocs) as pool:
            # Reading and orthonormalizing the KS overlaps
            pool.starmap(orthonormalize_ks_overlaps, var_pool)
            pool.close()
            pool.join()
    else:
        for step in range(start_time, finish_time):
            orthonormalize_ks_overlaps(step, params)
    print('Done with orthonormalization step. Elapsed time:', time.time() - t2)

    zero = MATRIX(nstates, nstates)

    # Whether to perform state reordering or not
    if params['do_state_reordering'] == 2 or params['do_state_reordering'] == 1:
        # This step may be very time-consuming for large matrices but since
        # we have the matrices in sparse format we need to first turn them into
        # CMATRIX and then use them for state reordering.
        t2 = time.time()
        St_ks_cmatrices = []
        E_ks_cmatrices = []
        for step in range(start_time, finish_time):
            # Extract the real part of the dense matrix. The imaginary part is zero.
            St_ks = np.array(sp.load_npz(F'{res_dir_2}/St_ks_orthonormalized_{step}.npz').todense().real)
            St_ks_cmatrix = data_conv.nparray2CMATRIX(St_ks)
            St_ks_cmatrices.append(St_ks_cmatrix)
            # E_ks = np.array( sp.load_npz(F'{res_dir_2}/Hvib_ks_{step-start_time}_re.npz').todense().real )
            E_ks = np.array(sp.load_npz(F'{res_dir_2}/E_ks_{step}.npz').todense().real)
            E_ks_cmatrix = data_conv.nparray2CMATRIX(E_ks)
            E_ks_cmatrices.append(E_ks_cmatrix)
        # Now apply the state-reordering
        apply_state_reordering(St_ks_cmatrices, E_ks_cmatrices, params)
        # After that the state-reordering is done, we can turn them back into sparse and save them.
        # For now, this is done so that the code is consistent with the rest of the code.
        for step in range(start_time, finish_time):
            St_ks_sparse = data_conv.MATRIX2scipynpz(St_ks_cmatrices[step - start_time].real())
            sp.save_npz(F'{res_dir_2}/St_ks_orthonormalized_{step}.npz', St_ks_sparse)
            E_ks_sparse = data_conv.MATRIX2scipynpz(E_ks_cmatrices[step - start_time].real())
            # sp.save_npz(F'{res_dir_2}/Hvib_ks_{step-start_time}_re.npz', E_ks_sparse)
            sp.save_npz(F'{res_dir_2}/E_ks_{step}.npz', E_ks_sparse)
        print('Done with state-reordering of the KS orbitals. Elapsed time:', time.time() - t2)

    # Applying phase correction
    if params['apply_phase_correction']:
        t2 = time.time()
        print('Performing phase correction...')
        for step in range(start_time, finish_time):
            print(F'Applying phase-correction to step {step}')
            # Initialize the cum_phase_aa and cum_phase_bb
            if step == start_time:
                cum_phase_aa = np.ones((nstates, 1))
                cum_phase_bb = np.ones((nstates, 1))
            # Load the orthnormalized St matrices. This might looks like a redundant step but
            # for very large matrices it would be hard to keep them in memory and saving is more efficient.
            # These files will be removed after phase-correction is done.
            St_step = sp.load_npz(F'{res_dir_2}/St_ks_orthonormalized_{step}.npz')
            St_step_phase_corrected, cum_phase_aa, cum_phase_bb = \
                apply_phase_correction_scipy(St_step, step, cum_phase_aa, cum_phase_bb, two_spinor_format=True)
            if nac_algo == 0:
                # HST: Im(Hvib) = -d = (St^T - St)/(2 dt), with St_ij = <i(t)|j(t+dt)>
                Hvib_ks = 0.5 / dt * (St_step_phase_corrected.todense().T - St_step_phase_corrected.todense())
            elif nac_algo == 1:
                # Meek-Levine approach
                # scipy sparse to MATRIX so that we can use it in nac_npi
                St_step_phase_corrected_MATRIX = data_conv.nparray2MATRIX(
                    np.array((St_step_phase_corrected.todense().real)))
                # Computing the NAC values using Meek-Levine approach
                Hvib_ks_MATRIX = -1.0 * nac_npi(St_step_phase_corrected_MATRIX, dt)
                # Turn the MATRIX to numpy array so that we can save it using scipy.sparse library
                Hvib_ks = data_conv.MATRIX2nparray(Hvib_ks_MATRIX)

            # sp.save_npz(F'{res_dir_2}/St_ks_{step-start_time}_re.npz', St_step_phase_corrected )
            # sp.save_npz(F'{res_dir_2}/Hvib_ks_{step-start_time}_im.npz', sp.csc_matrix( Hvib_ks ))
            sp.save_npz(F'{res_dir_2}/St_ks_{step}.npz', St_step_phase_corrected)
            sp.save_npz(F'{res_dir_2}/Hvib_ks_{step}_im.npz', sp.csc_matrix(Hvib_ks))
            os.system(F'rm {res_dir_2}/St_ks_orthonormalized_{step}.npz')
    else:
        for step in range(start_time, finish_time):
            St_step = sp.load_npz(F'{res_dir_2}/St_ks_orthonormalized_{step}.npz')
            if nac_algo == 0:
                # HST: Im(Hvib) = -d = (St^T - St)/(2 dt), with St_ij = <i(t)|j(t+dt)>
                Hvib_ks = 0.5 / dt * (St_step.todense().T - St_step.todense())
            elif nac_algo == 1:
                # Meek-Levine approach
                # scipy sparse to MATRIX
                St_step_MATRIX = data_conv.nparray2MATRIX(np.array(St_step.todense().real))
                #
                Hvib_ks_MATRIX = -1.0 * nac_npi(St_step_MATRIX, dt)
                # MATRIX to numpy array for saving data using scipy sparse library
                Hvib_ks = data_conv.MATRIX2nparray(Hvib_ks_MATRIX)
                # St_step_real = St_step.real()
                # nac = nac_npi(St_step_real, dt)
                # Hvib_ks = -1.0 * CMATRIX(nac, zero)
            # sp.save_npz(F'{res_dir_2}/Hvib_ks_{step-start_time}_im.npz', sp.csc_matrix( Hvib_ks ))
            sp.save_npz(F'{res_dir_2}/Hvib_ks_{step}_im.npz', sp.csc_matrix(Hvib_ks))
            # os.system(F'mv {res_dir_2}/St_ks_orthonormalized_{step}.npz {res_dir_2}/St_ks_{step-start_time}_re.npz')
            os.system(F'mv {res_dir_2}/St_ks_orthonormalized_{step}.npz {res_dir_2}/St_ks_{step}.npz')

    print('Done with phase correction. Elapsed time:', time.time() - t2)


def orthonormalize_ks_overlaps(step, params):
    """
    This fuction is an auiliary function used in run_step3_ks_nacs_libint function
    to perform orthonormalization of the overlaps.
    """
    with _blas_thread_limit(_worker_blas_threads(params)):
        return _orthonormalize_ks_overlaps(step, params)


def _orthonormalize_ks_overlaps(step, params):
    active_space = params['active_space']
    start_time = params['start_time']
    t2 = time.time()
    print('Computing orthonormalization of St matrices in step', step)
    # The time-overlaps. Note that we need to turn this ndarray to np.array so that we can
    # apply data_conv.nparray2MATRIX function.
    St_step = sp.load_npz(F'{params["path_to_npz_files"]}/St_ks_{step}.npz')[active_space, :][:, active_space]
    # The overlap matrix of the time 'step'
    S_1 = sp.load_npz(F'{params["path_to_npz_files"]}/S_ks_{step}.npz')[active_space, :][:, active_space]
    # The overlap of the time 'step+1'
    S_2 = sp.load_npz(F'{params["path_to_npz_files"]}/S_ks_{step+1}.npz')[active_space, :][:, active_space]
    # The energies of the time 'step'
    E_1 = sp.load_npz(F'{params["path_to_npz_files"]}/E_ks_{step}.npz')[active_space, :][:, active_space]
    # The energies of the time 'step+1'
    E_2 = sp.load_npz(F'{params["path_to_npz_files"]}/E_ks_{step+1}.npz')[active_space, :][:, active_space]
    # The mid-point energies
    E_step = 0.5 * (E_1 + E_2)
    # Applying orthonormalization: "lowdin"/True (the historical behavior of this
    # stage), "lowdin_cached" (default; identical result - the ket factor is carried
    # to the next step, one fractional matrix power per step instead of two), or
    # "none"/False (raw pass-through)
    mode = _orthonormalization_mode(params.get('apply_orthonormalization', 'lowdin_cached'))
    _allow_dirty = params.get('allow_nonorthonormal_overlaps', False)
    _check_input_overlap(np.asarray(S_1.todense()), 'S_ks(t)', step,
                         allow_dirty=_allow_dirty)
    _check_input_overlap(np.asarray(S_2.todense()), 'S_ks(t+1)', step,
                         allow_dirty=_allow_dirty)
    if mode == "none":
        St_step, S_step = St_step.todense(), S_1.todense()
    elif mode == "lowdin_cached":
        S_1d, S_2d = S_1.todense(), S_2.todense()
        U1 = _lowdin_cache_take(step, lowdin_digest(S_1d))
        if U1 is None:
            U1 = get_Lowdin_scipy(S_1d)
        U2 = get_Lowdin_scipy(S_2d)
        _lowdin_cache_put(step + 1, lowdin_digest(S_2d), U2)
        St_step = np.linalg.multi_dot([U1.conj().T, St_step.todense(), U2])
        S_step = np.linalg.multi_dot([U1.conj().T, S_1d, U1])
    else:
        St_step, S_step = apply_orthonormalization_scipy(S_1.todense(), S_2.todense(), St_step.todense())

    if mode != "none":
        _check_contraction(np.asarray(St_step).real, step, 'KS')

    St_step_sparse = sp.csc_matrix(St_step)
    S_step_sparse = sp.csc_matrix(S_step)
    validate_overlap_matrix(np.asarray(S_step_sparse.todense()),
                            F"S_ks_{step} (step-3 output)")

    sp.save_npz(F'{params["path_to_save_ks_Hvibs"]}/St_ks_orthonormalized_{step}.npz', St_step_sparse)
    sp.save_npz(F'{params["path_to_save_ks_Hvibs"]}/S_ks_{step}.npz', S_step_sparse)
    # sp.save_npz(F'{params["path_to_save_ks_Hvibs"]}/Hvib_ks_{step-start_time}_re.npz', E_step)
    sp.save_npz(F'{params["path_to_save_ks_Hvibs"]}/E_ks_{step}.npz', E_step)
    print('Done with step', step, '. Elapsed time:', time.time() - t2)


def koopmans_sd_basis(reference, charge, nact):
    """
    The frozen-orbital (Koopmans) SD basis of an ionized closed-shell reference: charge +1
    removes one beta electron from every occupied orbital of the reference (the hole basis,
    N-1 electrons), charge -1 adds one alpha electron to every active virtual (the electron
    basis, N+1). The orbitals and their energies stay those of the neutral SCF - the hole or
    electron never relaxes the density, so the site energies keep their geometric detuning
    (a self-consistent ion screens it).

    Args:
        reference (list of int): the reference SD in the reindexed signed notation
        charge (int): +1 or -1
        nact (int): active spatial orbitals per spin (the virtuals are the alpha indices
            1..nact absent from the reference)

    Returns:
        sd_unique_basis (list): one [[orbital, orbital], spin] descriptor per determinant -
            the removed ('bet') or added ('alp') spin-orbital
        sd_states_reindexed (list of lists of int): the determinants, the reference's list
            order kept (in-place removal or an appended orbital)
    """
    ref = [int(o) for o in reference]
    if charge == 1:
        holes = sorted([o for o in ref if o < 0], key=abs)
        return ([[[-h, -h], 'bet'] for h in holes],
                [[o for o in ref if o != h] for h in holes])
    if charge == -1:
        virt = [a for a in range(1, nact + 1) if a not in ref]
        return ([[[a, a], 'alp'] for a in virt],
                [ref + [a] for a in virt])
    raise ValueError(F"koopmans_charge {charge} - use 0, +1 (hole basis) or -1 (electron basis)")


def _sd_block_index(orb, nact):
    """Row of a signed reindexed spin-orbital in the 2-block active-space matrices
    (alpha j -> j-1, beta -j -> nact+j-1, the ovlp_arb convention)."""
    return orb - 1 if orb > 0 else nact + (-orb) - 1


def _permute_rows(W, perm):
    """Row permutation of a numpy matrix in the CMATRIX.permute_rows convention."""
    Wc = data_conv.nparray2CMATRIX(np.asarray(W, dtype=float))
    p = intList()
    for k in perm:
        p.append(int(k))
    Wc.permute_rows(p)
    return data_conv.MATRIX2nparray(Wc.real(), np.float64)


def _spin_partner(orb, active_space, data_dim):
    """The reindexed spin-orbital of the opposite spin on the same spatial orbital, or None
    (the two channels' active spaces may hold different spatial orbitals). active_space:
    the 2-block window indices of the step, data_dim: the window's 2-block dimension."""
    n = data_dim // 2
    alpha = [a for a in active_space if a < n]
    beta = [b - n for b in active_space if b >= n]
    if orb > 0:
        spatial = alpha[orb - 1]
        return -(beta.index(spatial) + 1) if spatial in beta else None
    spatial = beta[-orb - 1]
    return alpha.index(spatial) + 1 if spatial in alpha else None


def sd_diabatization_groups(sd_lists, kind, active_space=None, data_dim=None):
    """
    The localization manifolds of an SD basis: single excitations of the reference
    determinant that share the hole ("particle": the excited electron varies), the
    particle ("hole": the hole varies), or the spin channel ("joint": the complete
    hole x particle product of a channel, holes and electrons localized together).
    An open-shell reference joins the manifold whose defining orbital it occupies singly
    (a hole manifold whose particle's spin partner is occupied - the cation's hole - or a
    particle manifold whose hole's partner is empty - the anion's electron): its own
    configuration is one of the localized states.

    Args:
        sd_lists (list of lists of int): the SDs of one step in the signed reindexed
            spin-orbital notation of ovlp_arb, in their row order
        kind (str): "particle", "hole" or "joint"
        active_space, data_dim: the 2-block window indices and dimension of the step - the
            spin-partner test of an open-shell reference needs them (None: the reference
            is taken as closed-shell and never joins a manifold)

    Returns:
        gs (int): row of the reference determinant (the lowest orbital sum)
        groups (list of dict): manifolds with >= 2 members - "key" (the shared spin-orbital,
            or the channel sign for "joint"), "rows" (member rows), "tau" (parity signs).
            particle/hole: "orbs" (the varying spin-orbital of every member, ascending) and
            "sigma" - the localized state with the varying orbital psi_A = sum_j w_jA orb_j
            reads sum_j tau_j sigma_j w_jA |SD_row_j> (tau_j: parity between the member's list
            and the fixed-orbitals-first order, sigma_j: the cofactor sign (-1)^j of a hole
            manifold, 1 for a particle manifold). joint: "pairs" (the (hole, particle) of every
            member), "holes" and "parts" (the ascending hole and particle spin-orbitals) -
            the state a+_Y a_X |ref> with localized psi_X = sum_i wh_iX hole_i and
            phi_Y = sum_a wp_aY part_a reads sum_j tau_j wh_(i_j)X wp_(a_j)Y |SD_row_j>, tau_j
            the parity between the member's list and the in-place replacement of its hole in
            the reference list (a+_a a_i |ref> is the in-place determinant)
        singles (list of int): rows left untransformed (the reference, one-member manifolds,
            non-single excitations)

    Raises ValueError for an unknown kind, for "joint" when a channel's single
    excitations do not form the complete hole x particle product or when the reference
    is open-shell (the joint states presume a closed-shell reference).
    """
    if kind not in ("particle", "hole", "joint"):
        raise ValueError(F"diabatization {kind!r} - use 'none', 'particle', 'hole' or 'joint'")
    sets = [frozenset(int(o) for o in sd) for sd in sd_lists]
    gs = min(range(len(sets)), key=lambda k: sum(abs(o) for o in sets[k]))
    ref = sets[gs]

    def singly_occupied(orb):
        # orb's spatial orbital holds exactly one electron in the reference
        if active_space is None:
            return False
        partner = _spin_partner(orb, active_space, data_dim)
        return partner is not None and ((orb in ref) != (partner in ref))

    manifolds = {}
    singles = [gs]
    for k, occ in enumerate(sets):
        if k == gs:
            continue
        hole, part = ref - occ, occ - ref
        if len(hole) != 1 or len(part) != 1:
            singles.append(k)
            continue
        h, pt = next(iter(hole)), next(iter(part))
        if kind == "joint":
            if singly_occupied(h) or singly_occupied(pt):
                raise ValueError("joint localization presumes a closed-shell reference - the "
                                 F"reference holds {h if singly_occupied(h) else pt} singly "
                                 "(use 'hole' for a cation, 'particle' for an anion)")
            manifolds.setdefault(1 if pt > 0 else -1, []).append(((h, pt), k))
            continue
        key, var = (h, pt) if kind == "particle" else (pt, h)
        manifolds.setdefault(key, []).append((var, k))
    groups = []
    ref_list = [int(o) for o in sd_lists[gs]]
    for key in sorted(manifolds):
        members = sorted(manifolds[key])
        if kind != "joint" and singly_occupied(key):
            # the reference is the member whose varying orbital is the manifold's own
            # defining orbital (the hole sits in the particle p, or the electron in the hole h)
            members = sorted(members + [(key, gs)])
            if gs in singles:
                singles.remove(gs)
        if len(members) < 2:
            singles.append(members[0][1])
            continue
        if kind == "joint":
            pairs = [ia for ia, k in members]
            rows = [k for ia, k in members]
            holes = sorted({h for h, pt in pairs})
            parts = sorted({pt for h, pt in pairs})
            if len(set(pairs)) != len(pairs) or len(pairs) != len(holes) * len(parts):
                raise ValueError(F"joint localization needs the complete hole x particle product of "
                                 F"the {'alpha' if key > 0 else 'beta'} channel: {len(holes)} holes x "
                                 F"{len(parts)} particles but {len(pairs)} single excitations")
            tau = []
            for j, (h, pt) in enumerate(pairs):
                in_place = list(ref_list)
                in_place[in_place.index(h)] = pt
                tau.append(slatdet.permutation_parity(in_place, [int(o) for o in sd_lists[rows[j]]]))
            groups.append({"key": key, "rows": rows, "pairs": pairs, "holes": holes, "parts": parts,
                           "tau": tau, "kind": kind})
            continue
        orbs = [v for v, k in members]
        rows = [k for v, k in members]
        if kind == "particle":
            # the fixed orbitals: the reference without the shared hole (an open-shell
            # reference is a member itself, its varying orbital is that hole)
            fixed = sorted(ref - {key} - set(orbs))
            lists = [fixed + [v] for v in orbs]
            sigma = [1] * len(orbs)
        else:
            # member j holds every hole orbital but its own: the localized-hole state
            # expands with the cofactors of the orthogonal rotation, (-1)^j w_jA
            fixed = sorted((ref - set(orbs)) | ({key} if key not in orbs else set()))
            lists = [fixed + [o for o in orbs if o != v] for v in orbs]
            sigma = [(-1) ** j for j in range(len(orbs))]
        tau = [slatdet.permutation_parity(lists[j], [int(o) for o in sd_lists[rows[j]]])
               for j in range(len(orbs))]
        groups.append({"key": key, "rows": rows, "orbs": orbs, "tau": tau, "sigma": sigma, "kind": kind})
    return gs, groups, sorted(singles)


def localize_fragment_block(P_frag, rows):
    """
    Orbitals localized on fragments within the span of the given orbitals, by
    sequential projection: fragment 0 takes the top round(tr P_0) eigenvectors of its
    operator in the span, every next fragment the top ones of the remaining span, the
    last fragment the rest.

    Args:
        P_frag (list of arrays): the fragment operators in a common orthonormal orbital
            basis (only the [rows, rows] blocks are read)
        rows (list of int): the orbitals spanning the localization space

    Returns:
        w (array (m, m)): columns = localized orbitals in the span, fragment-major, within
            a fragment by decreasing own-fragment population
        frag (list of int): the fragment of every column
        pops (array (m, nfrag)): population of every column on every fragment

    Raises ValueError when a fragment holds a non-integer number of orbitals of the span
    (|tr - round(tr)| > 0.25): the span mixes with orbitals the fragments do not localize.
    """
    m = len(rows)
    blocks = [0.5 * (B + B.T) for B in
              (np.asarray(P, dtype=float)[np.ix_(rows, rows)] for P in P_frag)]
    Q = np.eye(m)
    cols, frag = [], []
    for f, B in enumerate(blocks):
        left = Q.shape[1]
        if left == 0:
            break
        M = Q.T @ B @ Q
        if f == len(blocks) - 1:
            n_f = left
        else:
            tr = float(np.trace(M))
            n_f = int(round(tr))
            if abs(tr - n_f) > 0.25 or n_f > left:
                raise ValueError(F"fragment {f} holds {tr:.3f} orbitals of the span {rows} - "
                                 F"not an integer, the fragments do not localize this manifold")
        if n_f == 0:
            continue
        vals, vecs = np.linalg.eigh(M)
        vecs = vecs[:, np.argsort(-vals)]
        cols.append(Q @ vecs[:, :n_f])
        frag += [f] * n_f
        Q = Q @ vecs[:, n_f:]
    w = np.concatenate(cols, axis=1)
    pops = np.stack([np.einsum("jc,jk,kc->c", w, B, w) for B in blocks], axis=1)
    return w, frag, pops


def sd_diabatization_step(sd_lists, P_frag, kind, dia_index=None, active_space=None, data_dim=None):
    """
    The SD -> diabatic coefficients W of one step, |Phi_A> = sum_k |SD_k> W_kA: the
    reference and the untransformed SDs map onto themselves, every localization manifold
    onto its fragment-localized states (sd_diabatization_groups, localize_fragment_block).

    Args:
        sd_lists (list of lists of int): the SDs of the step in their row order
        P_frag (list of arrays): fragment operators in the active-space 2-block layout
            (orthonormal orbitals)
        kind (str): "particle", "hole" or "joint"
        dia_index (list, optional): the diabatic-state index of an earlier step - the
            columns keep its order, None builds it (reference and singles first, then
            the manifolds by key, fragment-major - hole-major then electron for "joint")
        active_space, data_dim: passed to sd_diabatization_groups (open-shell references)

    Returns:
        W (array (nsd, nsd)), dia_index (list), pops (array (nsd, nfrag)), pops_hole
        (array (nsd, nfrag)): fragment populations of every diabatic state's localized
        electron (varying) orbital and of its localized hole ("joint" only), nan elsewhere

    Raises ValueError when the SD content or the per-fragment orbital counts differ from
    the index (the localization must be stable along the trajectory) and when W is not
    orthogonal to 1e-8.
    """
    nsd = len(sd_lists)
    nact = P_frag[0].shape[0] // 2
    nfrag = len(P_frag)
    sets = [frozenset(int(o) for o in sd) for sd in sd_lists]
    gs, groups, singles = sd_diabatization_groups(sd_lists, kind, active_space, data_dim)
    loc = {}
    for g in groups:
        if g["kind"] == "joint":
            loc_h = localize_fragment_block(P_frag, [_sd_block_index(o, nact) for o in g["holes"]])
            loc_p = localize_fragment_block(P_frag, [_sd_block_index(o, nact) for o in g["parts"]])
            loc[g["key"]] = (g, loc_h, loc_p)
        else:
            w, frag, pops = localize_fragment_block(P_frag, [_sd_block_index(o, nact) for o in g["orbs"]])
            loc[g["key"]] = (g, w, frag, pops)

    def _cols(frag, f, r, what, key):
        cols_f = [c for c in range(len(frag)) if frag[c] == f]
        if r >= len(cols_f):
            raise ValueError(F"manifold {key}: fragment {f} holds {len(cols_f)} localized {what} "
                             F"orbitals at this step, at least {r + 1} expected - the "
                             F"localization is not stable along the trajectory")
        return cols_f[r]

    if dia_index is None:
        dia_index = [("sd", sets[k]) for k in singles]
        for key in sorted(loc):
            if loc[key][0]["kind"] == "joint":
                frag_h, frag_p = loc[key][1][1], loc[key][2][1]
                for f_h in range(nfrag):
                    for r_h in range(frag_h.count(f_h)):
                        for f_p in range(nfrag):
                            for r_p in range(frag_p.count(f_p)):
                                dia_index.append(("joint", key, f_h, r_h, f_p, r_p))
            else:
                frag = loc[key][2]
                for f in range(nfrag):
                    for r in range(frag.count(f)):
                        dia_index.append(("loc", key, f, r))
    if len(dia_index) != nsd:
        raise ValueError(F"{len(dia_index)} diabatic states indexed for {nsd} SDs")
    W = np.zeros((nsd, nsd))
    pops_out = np.full((nsd, nfrag), np.nan)
    pops_hole = np.full((nsd, nfrag), np.nan)
    for col, entry in enumerate(dia_index):
        if entry[0] == "sd":
            match = [k for k in singles if sets[k] == entry[1]]
            if len(match) != 1:
                raise ValueError(F"the SD {sorted(entry[1])} of the diabatic index is not an "
                                 F"untransformed SD of this step - the basis changed")
            W[match[0], col] = 1.0
            continue
        key = entry[1]
        if key not in loc:
            raise ValueError(F"manifold {key} of the diabatic index is absent at this step")
        if entry[0] == "joint":
            _, _, f_h, r_h, f_p, r_p = entry
            g, (wh, frag_h, ph), (wp, frag_p, pp) = loc[key]
            X = _cols(frag_h, f_h, r_h, "hole", key)
            Y = _cols(frag_p, f_p, r_p, "electron", key)
            for j, k in enumerate(g["rows"]):
                h, pt = g["pairs"][j]
                W[k, col] = g["tau"][j] * wh[g["holes"].index(h), X] * wp[g["parts"].index(pt), Y]
            pops_out[col] = pp[Y]
            pops_hole[col] = ph[X]
            continue
        _, _, f, r = entry
        g, w, frag, pops = loc[key]
        c = _cols(frag, f, r, "", key)
        for j, k in enumerate(g["rows"]):
            W[k, col] = g["tau"][j] * g["sigma"][j] * w[j, c]
        pops_out[col] = pops[c]
    dev = float(np.abs(W.T @ W - np.eye(nsd)).max())
    if dev > 1e-8:
        raise ValueError(F"the diabatic transformation is not orthogonal (|W^T W - 1| = {dev:.2e})")
    return W, dia_index, pops_out, pops_hole


def _sd_diabatization_series(params, sd_sorted, active_space, start_time, finish_time, kind):
    """W, the diabatic index and the populations of every step start_time..finish_time-1,
    from the step-2 fragment operators sliced to the active space (Lowdin-orthonormalized
    when the SD overlaps are)."""
    res_dir_1 = params['path_to_npz_files']
    prefixes = {"mulliken": "P_frag", "pod": "P_pod", "pod_cp2k": "P_pod_cp2k"}
    operator = str(params['fragment_operator']).lower()
    if operator not in prefixes:
        raise ValueError(F"fragment_operator {operator!r} - use 'mulliken', 'pod' or 'pod_cp2k'")
    prefix = prefixes[operator]
    nfrag = len(glob.glob(F'{res_dir_1}/{prefix}*_ks_{start_time}.npz'))
    if nfrag < 2:
        raise ValueError(F"diabatization needs the step-2 fragment operators "
                         F"{prefix}{{f}}_ks_{{step}}.npz of at least two fragments in {res_dir_1} "
                         F"({dict(mulliken='fragment_atom_lists', pod='compute_pod', pod_cp2k='read_pod_cp2k_files')[operator]} of "
                         F"step2), found {nfrag}")
    mode = params['apply_orthonormalization']
    W_list, pops_list, hole_list, dia_index = [], [], [], None
    for i, step in enumerate(range(start_time, finish_time)):
        P_frag = [np.array(sp.load_npz(F'{res_dir_1}/{prefix}{f}_ks_{step}.npz').todense()
                           [active_space, :][:, active_space]).real for f in range(nfrag)]
        if mode != "none":
            s_ks = np.array(sp.load_npz(F'{res_dir_1}/S_ks_{step}.npz').todense()
                            [active_space, :][:, active_space]).real
            X = orthogonalizations.get_Lowdin_scipy(s_ks)
            P_frag = [X.T @ P @ X for P in P_frag]
        W, dia_index, pops, pops_hole = sd_diabatization_step(sd_sorted[i], P_frag, kind, dia_index,
                                                              active_space, params['data_dim'])
        W_list.append(W)
        pops_list.append(pops)
        hole_list.append(pops_hole)
    return W_list, dia_index, pops_list, hole_list


def _write_diabatization_records(res_dir, kind, dia_index, pops_list, hole_list, start_time, operator="mulliken"):
    """U_sd_labels.json (the meaning of every diabatic state) and U_sd_pops.txt (per step,
    the fragment populations of every diabatic state's localized electron orbital, then of
    its localized hole - nan where the state has none)."""
    states = []
    for col, entry in enumerate(dia_index):
        if entry[0] == "sd":
            states.append({"index": col, "type": "sd", "sd": sorted(entry[1])})
        elif entry[0] == "joint":
            states.append({"index": col, "type": "joint", "channel": int(entry[1]),
                           "hole_fragment": int(entry[2]), "hole_rank": int(entry[3]),
                           "electron_fragment": int(entry[4]), "electron_rank": int(entry[5])})
        else:
            states.append({"index": col, "type": "localized", "manifold": int(entry[1]),
                           "fragment": int(entry[2]), "rank": int(entry[3])})
    with open(F'{res_dir}/U_sd_labels.json', 'w') as f:
        json.dump({"kind": kind, "fragment_operator": operator,
                   "note": "particle: the excited electron of the manifold (shared hole = "
                           "manifold) sits on 'fragment', hole: the hole (shared particle = "
                           "manifold) sits on 'fragment', joint: the hole sits on "
                           "'hole_fragment' and the electron on 'electron_fragment' (channel "
                           "1 = alpha, -1 = beta excitations), ranks order the orbitals of one "
                           "fragment by decreasing population",
                   "states": states}, f, indent=1)
    nfrag = pops_list[0].shape[1]
    nst = pops_list[0].shape[0]
    rows = [np.concatenate(([start_time + i], pops_list[i].reshape(-1), hole_list[i].reshape(-1)))
            for i in range(len(pops_list))]
    header = ("step " + " ".join(F"p_e(state{c},frag{f})" for c in range(nst) for f in range(nfrag))
              + " " + " ".join(F"p_h(state{c},frag{f})" for c in range(nst) for f in range(nfrag)))
    np.savetxt(F'{res_dir}/U_sd_pops.txt', np.array(rows), header=header, fmt="%.6g")


def _save_u_sd(res_dir, W, cum_phase, step):
    """U_sd = (W F)^T: rows = diabatic states, columns = the saved (phase-corrected) SDs."""
    Wf = np.asarray(W, dtype=float)
    if cum_phase is not None:
        Wf = Wf * np.asarray(cum_phase, dtype=float).reshape(-1, 1)
    sp.save_npz(F'{res_dir}/U_sd_{step}.npz', sp.csc_matrix(Wf.T))


@_with_parent_blas_cap
def run_step3_sd_nacs_libint(params):
    """
    This function runs the step3 for computing NACs between SDs.
    It performs phase-correction and orthonormalization for the overlaps.
    For now, we do not include state reordering algorithm but it will be added later.

    Args:

        params (dictionary):

            * **params['nprocs']** (integer): number of processors to be used.
            * **params['use_multiprocessing']** (bool): the flag to use multiprocessing or not.
            * **params['path_to_npz_files']** (string):  the path to the KS orbital overlaps which are stored
                                                         in .npz file format in step2.
            * **params['time_step']** (float): the time-step in femtosecond
            * **params['path_to_save_ks_Hvibs']** (string): the path for storing the KS Hvibs
            * **params['path_to_save_sd_Hvibs']** (string): the path for saving the SD Hvibs
            * **params['active_space']** (list): a list that contains the indices of specific states
            * **params['ks_homo_index']** (integer): the index of the KS HOMO + 1 (it starts from 1)
            * **params['data_dim']** (integer): the dimension of the energy/overlap matrices stored as .npz
            * **params['start_time']** (integer): the initial step for reading the files
            * **params['finish_time']** (integer): the final step for reading the files
            * **params['sorting_type']** (string): it performs sorting the elements of the overlap matrices
                                                   either by their 'energy' or 'identity'.
            * **params['apply_phase_correction']** (bool): a True or False flag for performing the phase-correction
            * **params['apply_orthonormalization']**: orthonormalization of the SD-level
                overlaps (same vocabulary as cp2k_compute_adi): "lowdin"/True,
                "lowdin_cached" [default] (identical to "lowdin"; the ket factor is carried to
                the next step in a per-process cache - one fractional matrix power per
                step serially, hits inside contiguous Pool chunks too), "none"/False.
                The default CHANGED from the historical False (raw overlaps) to
                "lowdin_cached" - the Lowdin route is the validated convention;
                set "none" explicitly to reproduce pre-change data
            * **params['koopmans_charge']** (int): 0 [default] keeps the SCF reference and its single
                excitations. +1 replaces the SD basis by the frozen-orbital (Koopmans) hole basis of
                the cation: one beta electron removed from each of the num_occ_states active occupied
                orbitals of the closed-shell reference (num_occ_states determinants, energies from the
                neutral orbital energies, no self-consistent screening of the site detuning). -1 is
                the electron basis of the anion (one alpha electron added to each active virtual).
                Needs isUKS 0. Pairs with diabatization "hole" / "particle".
            * **params['fragment_operator']** (string): the fragment operators the diabatization
                localizes with - "mulliken" [default] reads P_frag{f}_ks_{step}.npz (step2
                fragment_atom_lists), "pod" reads P_pod{f}_ks_{step}.npz (step2 compute_pod, libra's own
                projection-operator diabatization), "pod_cp2k" reads P_pod_cp2k{f}_ks_{step}.npz (step2
                read_pod_cp2k_files, the blocks of CP2K's POD).
            * **params['diabatization']** (string): "none" [default] keeps the SD basis. "particle" or
                "hole" localizes, in every single-excitation manifold sharing the hole (resp. the
                particle), the varying spin-orbital on the fragments of the step-2 fragment operators
                (P_frag{f}_ks_{step}.npz from step2's fragment_atom_lists). "joint" localizes the holes
                among the occupied orbitals AND the electrons among the virtuals of every spin
                channel's complete hole x particle manifold - the diabatic states are the excitations
                from a localized hole into a localized electron (Frenkel and charge-transfer states of
                a fragment pair). All three write the
                SD -> diabatic transformation U_sd_{step}.npz (rows: diabatic states, columns: the
                saved SD states, in the gauge of Hvib_sd/St_sd - sorted, reordered, phase-corrected)
                with U_sd_labels.json (what every diabatic state is) and U_sd_pops.txt (fragment
                populations of the localized orbitals per step). A runner hands U_sd_{step} to
                obj.basis_transform.
            * **params['npz_file_ks_homo_index']** (integer): the KS HOMO index of the raw npz files.
            * **params['is_many_body']** (bool): a True or False flag for computing the overlaps and energies
                                                 for many-body states
            * **params['use_multiprocessing']** (bool): a True or False flag for using multiprocessing
            * **params['isUKS']** (integer): if this value is set to `1`, the unrestricted spin calculations are considered.
            * **params['logfile_directory']** (string): the full path to logfile directory
            * **params['es_software']** (string): software name - values that it can take are 'cp2k', 'gaussian', and 'dftb+'
            * **params['ci_threshold']** (float): a value to select SDs that the square value of their CI is more than ci_threshold
            * **params['number_of_states']** (integer): the number of TD-DFT excited states to be considered
            * **params['homo_index']** (integer): the HOMO index specified by user (starts from 1)
            * **params['num_occ_states']** (integer): the number of occupied states to be considered for single-particle basis. (isUKS=False)
            * **params['num_unocc_states']** (integer): the number of unoccupied states to be considered for single-particle basis. (isUKS=False)
            * **params['num_occ_alpha']** (integer): the number of occupied states to be considered for single-particle basis. (isUKS=True)
            * **params['num_unocc_beta']** (integer): the number of unoccupied states to be considered for single-particle basis. (isUKS=True)
            * **params['verbosity']** (integer): a value for printing the TD-DFT data
            * **params['do_state_reordering']** (integer): the value for performing the state-reordering
                the vlaues it takes are:
                    - 0: no state reordering - same as in Pyxaid
                    - 1: older method (is not robust, may or may not work)
                    - 2: Hungarian algorithm [default]
            * **params['state_reordering_alpha']** (float): a value for state-reordering
            * **params['nac_algo']** ( int ): selection of a method to compute NACs
                    - 0 : Hamess-Shiffer-Tully (HST), finite difference [ default ]
                    - 1 : Meek-Levine NPI
            * **params['blas_num_threads']** (integer): BLAS threads per worker. Default (None):
                    1 per worker with use_multiprocessing, else nprocs. Set explicitly for
                    hybrid parallelism: nprocs workers x blas_num_threads threads each.
            * **params['sd_overlap_kernel']** (string): kernel for the SD overlap matrices
                    - "auto" : batched LAPACK determinants with the determinant-lemma shortcut
                               for low-degree excitations (singles, doubles, ...); machine
                               precision vs "per_pair" [ default ]
                    - "batched" : batched LAPACK determinants only
                    - "lemma" : determinant-lemma kernel only
                    - "per_pair" : the original per-pair mapping.ovlp_mat_arb loop

    Returns:

        None
    """

    critical_params = ['lowest_orbital', 'highest_orbital']
    default_params = {'nprocs': 2, 'path_to_npz_files': os.getcwd() + '/res',
                      'path_to_save_sd_Hvibs': os.getcwd() + '/res-sd',
                      'path_to_save_ks_Hvibs': os.getcwd() + '/res-ks',
                      'time_step': 1.0, 'start_time': 0, 'finish_time': 1,
                      'sorting_type': 'energy', 'apply_phase_correction': False, 'spin_adapted': False,
                      'apply_orthonormalization': 'lowdin_cached', 'do_state_reordering': 0,
                      'state_reordering_alpha': 0, 'is_many_body': False, 'num_occ_states': 1,
                      'num_occ_alpha': 1, 'num_unocc_alpha': 1,
                      'num_occ_beta': 1, 'num_unocc_beta': 1, 'active_space_num_occ_orbitals': 0,
                      'num_unocc_states': 1, 'verbosity': 0, 'isUKS': 0, 'es_software': 'cp2k',
                      'use_multiprocessing': False, 'logfile_directory': os.getcwd() + '/all_logfiles', 'nac_algo': 0,
                      'sd_overlap_kernel': 'auto', 'blas_num_threads': None,
                      'diabatization': 'none', 'koopmans_charge': 0, 'fragment_operator': 'mulliken'
                      }

    if params['is_many_body'] and params['isUKS']:
        raise Exception('Unrestricted-spin calculations are not supported for many-body basis yet!')  # TODO

    # Check input, set output dirs
    comn.check_input(params, default_params, critical_params)
    params['apply_orthonormalization'] = _orthonormalization_mode(params['apply_orthonormalization'])
    res_dir_1 = params['path_to_npz_files']
    res_dir_2 = params['path_to_save_sd_Hvibs']

    # CP2K parsing
    if params['es_software'].lower() == 'cp2k':
        sample_logfile = glob.glob(params['logfile_directory'] + '/*.log')[0]
        sample_npz_file = glob.glob(params['path_to_npz_files'] + '/*.npz')[0]
        params['data_dim'] = int(sp.load_npz(sample_npz_file).shape[0])
        _check_orbital_window(params, params['data_dim'], 'run_step3_sd_nacs_libint')
        if params['isUKS']:
            params['homo_alpha_index'], params['homo_beta_index'] = CP2K_methods.read_homo_index(
                sample_logfile, isUKS=True)
            params['npz_file_ks_homo_alpha_index'] = params['homo_alpha_index'] - params['lowest_orbital'] + 1
            params['npz_file_ks_homo_beta_index'] = params['homo_beta_index'] - params['lowest_orbital'] + 1
        else:  # nonUKS
            params['homo_index'] = CP2K_methods.read_homo_index(sample_logfile)
            params['npz_file_ks_homo_index'] = params['homo_index'] - params['lowest_orbital'] + 1

    # Setting function vars
    data_dim = params['data_dim']
    start_time = params['start_time']
    finish_time = params['finish_time']
    dt = params['time_step'] * units.fs2au
    nac_algo = params['nac_algo']
    if nac_algo not in (0, 1):
        raise ValueError(F"nac_algo = {nac_algo} is not a listed option - "
                         F"0 (HST) and 1 (Meek-Levine NPI) are supported")
    nprocs = params['nprocs']

    try:
        os.system(F'mkdir {res_dir_2}')
    except BaseException:
        print(F'The directory {res_dir_2} already exists.')
    # the _im files store Im(Hvib) = -d; files WITHOUT this marker predate
    # the convention restoration and hold +d (the NAC itself)
    marker_file = os.path.join(res_dir_2, 'hvib_convention.json')
    if not os.path.exists(marker_file) and glob.glob(os.path.join(res_dir_2, '*_im.npz')):
        raise ValueError(F"{res_dir_2} already holds *_im.npz files without a "
                         F"convention marker (legacy +d data) - writing canonical "
                         F"-d files into it would mix conventions in one directory; "
                         F"use a fresh output path or move the legacy files away")
    with open(marker_file, 'w') as f:
        json.dump({'im_file_contains': 'Im(Hvib) = -d'}, f)

    if params['is_many_body']:
        params['isnap'] = params['start_time']
        params['fsnap'] = params['finish_time']

        # The KS HOMO index specified by the user
        ks_homo_index = params['homo_index']

        # Generate the excitation data
        res = step3_many_body.get_step2_mb_sp_properties(params)

        # The unique SDs in the TD-DFT results in all logfiles
        sd_unique_basis = res[0]

        # The ci_basis_states
        ci_basis_states = res[1]

        # The CI coefficients
        ci_coefficients = res[2]

        # The TD-DFT excitation energies
        ci_energies = res[3]

        # And their spin-components (alpha and beta)
        spin_components = res[4]

        # Now we need to generate the active space and KS HOMO index in that
        # active space for use in reading the npz files
        # The target KS states in TD-DFT
        sd_fstates = []

        # The initial KS states in TD-DFT
        sd_tstates = []
        for i in range(len(sd_unique_basis)):
            sd_fstates.append(sd_unique_basis[i][0][0])
            sd_tstates.append(sd_unique_basis[i][0][1])

        # The min_band present in the excitation
        min_band = min(sd_fstates)

        # The max_band present in the excitation
        max_band = max(sd_tstates)

        # number of occupied and unoccupied KS states
        num_occ = ks_homo_index - min_band + 1
        num_unocc = max_band - ks_homo_index  # + 1

        # The new KS active space and HOMO index
        ks_active_space, ks_homo_index_1 = make_active_space(num_occ,
                                                             num_unocc,
                                                             params['data_dim'],
                                                             params['npz_file_ks_homo_index'])
        params['active_space'] = ks_active_space
        # print('Flag ks_active_space:', ks_active_space)
        # print('Flag num_occ:',num_occ)
        # print('Flag num_unocc',num_unocc)
        # print('Flag params[data_dim]',params['data_dim'])
        # print('Flag npz file ks homo index',params['npz_file_ks_homo_index'])
        # The KS orbital indices
        ks_orbital_indicies = range(min_band, max_band + 1)

        # This parameter will be used in the reindexing of the SDs
        params['orbital_indices'] = ks_orbital_indicies
        # ks_homo_index = params['homo_index']
    else:
        # Create single-particle bases based on the
        # the number of occupied and unoccupied KS states
        sd_unique_basis = []

        # In this case the min_band will be 1 and the
        # max_band is the data_dim/2
        min_band = 1
        max_band = int(data_dim / 2)
        ks_orbital_indicies = range(min_band, max_band + 1)

        # Create the new active space and generate the KS HOMO index in that active space
        if params['isUKS']:
            alpha, beta = params['npz_file_ks_homo_alpha_index'], params['npz_file_ks_homo_beta_index']
            ks_active_space, ks_homo_index_alpha, ks_homo_index_beta = make_active_space(params['num_occ_states'],
                                                                                         params['num_unocc_states'],
                                                                                         params['data_dim'],
                                                                                         [alpha, beta],
                                                                                         isUKS=1,
                                                                                         num_occ_alpha=params['num_occ_alpha'],
                                                                                         num_occ_beta=params['num_occ_beta'],
                                                                                         num_unocc_alpha=params['num_unocc_alpha'],
                                                                                         num_unocc_beta=params['num_unocc_beta'])
            ks_homo_index = ks_homo_index_alpha
        else:
            ks_active_space, ks_homo_index = make_active_space(params['num_occ_states'],
                                                               params['num_unocc_states'],
                                                               params['data_dim'],
                                                               params['npz_file_ks_homo_index'])
        params['active_space'] = ks_active_space

        # Now, start buidling the unique SDs for both spin channels
        if not params['isUKS']:
            min_band = ks_homo_index + 1
            max_band = ks_homo_index + params['num_unocc_states']
            for occ_state in range(ks_homo_index - params['num_occ_states'] + 1, ks_homo_index + 1):
                for unocc_state in range(ks_homo_index + 1, max_band + 1):
                    # Keep only alpha channel, since beta is the same
                    sd_tmp = [[occ_state, unocc_state], 'alp']
                    if sd_tmp not in sd_unique_basis:
                        sd_unique_basis.append(sd_tmp)

        elif params['isUKS']:
            # Alpha channel
            min_band_alpha = ks_homo_index_alpha + 1
            max_band_alpha = ks_homo_index_alpha + params['num_unocc_alpha']
            for occ_state in range(ks_homo_index_alpha - params['num_occ_alpha'] + 1, ks_homo_index_alpha + 1):
                for unocc_state in range(min_band_alpha, max_band_alpha + 1):
                    sd_tmp = [[occ_state, unocc_state], 'alp']
                    if sd_tmp not in sd_unique_basis:
                        sd_unique_basis.append(sd_tmp)
            # Beta channel
            min_band_beta = ks_homo_index_beta + 1
            max_band_beta = ks_homo_index_beta + params['num_unocc_beta']
            for occ_state in range(ks_homo_index_beta - params['num_occ_beta'] + 1, ks_homo_index_beta + 1):
                for unocc_state in range(min_band_beta, max_band_beta + 1):
                    sd_tmp = [[occ_state, unocc_state], 'bet']
                    if sd_tmp not in sd_unique_basis:
                        sd_unique_basis.append(sd_tmp)

        # I will keep this part for now as previously used and only commented the hole-only part
        # Add hole-only SDs
        # min_band = ks_homo_index-params['num_occ_states']+1
        # max_band = ks_homo_index+1
        # for state in range(min_band, max_band):
        #    sd_tmp = [[state, ks_homo_index+1], 'alp']
        #    if sd_tmp not in sd_unique_basis:
        #        sd_unique_basis.append(sd_tmp)
        #    if params['isUKS']==1:
        #        sd_tmp = [[state, ks_homo_index+1], 'bet']
        #    if sd_tmp not in sd_unique_basis:
        #        sd_unique_basis.append(sd_tmp)

    # The params is updated with sd_unique_basis for reindexing
    params['sd_unique_basis'] = sd_unique_basis

    # Reindex the SD basis
    if params['isUKS']:
        sd_states_reindexed = step2_many_body.reindex_cp2k_sd_states(
            ks_homo_index_alpha,
            ks_orbital_indicies,
            sd_unique_basis,
            sd_format=1,
            ks_beta_homo_index=ks_homo_index_beta,
            active_space_num_occ_orbitals=params['active_space_num_occ_orbitals'])
    elif not params['isUKS']:
        sd_states_reindexed = step2_many_body.reindex_cp2k_sd_states(
            ks_homo_index,
            ks_orbital_indicies,
            sd_unique_basis,
            sd_format=2,
            active_space_num_occ_orbitals=params['active_space_num_occ_orbitals'])

    # Some printings
    koopmans_charge = int(params['koopmans_charge'])
    if koopmans_charge != 0:
        if params['isUKS']:
            raise ValueError("koopmans_charge needs a closed-shell reference (isUKS 0) - an "
                             "unrestricted reference is already the ion, use its own SD basis")
        # the reference is the first reindexed determinant of the closed-shell basis
        sd_unique_basis, sd_states_reindexed = koopmans_sd_basis(sd_states_reindexed[0], koopmans_charge,
                                                                 len(ks_active_space) // 2)
        params['sd_unique_basis'] = sd_unique_basis
        print(F'koopmans_charge {koopmans_charge}: {len(sd_states_reindexed)} frozen-orbital '
              F'{"hole" if koopmans_charge > 0 else "electron"} determinants of the neutral reference')
    print('sd_unique_basis is:', sd_unique_basis)
    print('sd_states_reindexed is:', sd_states_reindexed)
    print('ks_homo_index', ks_homo_index)
    print('ks_orbital_indicies', ks_orbital_indicies)
    params['isnap'] = start_time
    params['fsnap'] = finish_time

    if ks_homo_index - min(ks_active_space) < params['active_space_num_occ_orbitals']:
        ks_active_space_ = []
        for i in range(ks_homo_index - params['active_space_num_occ_orbitals'], ks_homo_index):
            ks_active_space_.append(i)
            ks_active_space_.append(i + int(data_dim / 2))
        for i in ks_active_space:
            if i not in ks_active_space_:
                ks_active_space_.append(i)
        ks_active_space_ = np.sort(ks_active_space_).tolist()
        ks_active_space = ks_active_space_
        params['active_space'] = ks_active_space_

    print('Flag ks_active_space:', ks_active_space)

    # The flag for phase-correction algorithm
    apply_phase_correction = params['apply_phase_correction']

    print('Sorting and computing the SDs energies...')
    t2 = time.time()
    E_sds = []
    sd_states_reindexed_sorted = []
    sd_states_unique_sorted = []

    for step in range(start_time, finish_time):
        E_sd_step, sd_states_unique_sorted_step, sd_states_reindexed_sorted_step, reindex_nsteps = sort_unique_SD_basis_scipy(
            step, sd_unique_basis, sd_states_reindexed, params)
        sd_states_reindexed_sorted.append(sd_states_reindexed_sorted_step)
        sd_states_unique_sorted.append(sd_states_unique_sorted_step)
        E_sds.append(np.array(E_sd_step.todense().real))

        if step > start_time:
            E_midpoint = 0.5 * (E_sd_step + E_sd_step_plus)
            # print('Flag E_midpoint.shape:', E_midpoint.shape)
            sp.save_npz(F'{params["path_to_save_sd_Hvibs"]}/Hvib_sd_{step-1}_re.npz', E_midpoint)
        E_sd_step_plus = E_sd_step
    print('Done with sorting and computing the SDs energies. Elapsed time:', time.time() - t2)

    # Update the params with sd_states_reindexed_sorted
    params.update({'sd_states_reindexed_sorted': sd_states_reindexed_sorted})
    diabatization = str(params['diabatization']).lower()
    W_dia = None
    if diabatization != 'none':
        W_dia, dia_index, dia_pops, dia_hole = _sd_diabatization_series(params, sd_states_reindexed_sorted,
                                                                        ks_active_space, start_time,
                                                                        finish_time, diabatization)
        _write_diabatization_records(res_dir_2, diabatization, dia_index, dia_pops, dia_hole, start_time,
                                     str(params['fragment_operator']).lower())
    if params['is_many_body']:
        # Add midpoint energies for CI states in here
        ci_midpoint_energies = step3_many_body.compute_ci_energies_midpoint(ci_energies, params)
        # Now, generate the SD2CI matrix
        if params['sorting_type'] == 'identity':
            # The new, faster way
            SD2CI = step3_many_body.make_T_matrices_fast(
                ci_coefficients, ci_basis_states, spin_components, sd_unique_basis, params)
        else:
            # The old way
            SD2CI = step3_many_body.make_T_matrices(
                ci_coefficients,
                ci_basis_states,
                spin_components,
                sd_states_unique_sorted,
                params)

        # These parameters are in CMATRIX format. We need
        # to convert them into numpy arrays
        for i in range(len(SD2CI)):
            SD2CI[i] = data_conv.MATRIX2nparray(SD2CI[i].real())

        # We create the var_pool whether we use the multiprocessing or not
        var_pool = make_var_pool(params, sd_states_reindexed_sorted, start_time, finish_time)
    else:
        var_pool = make_var_pool(params, sd_states_reindexed_sorted, start_time, finish_time)

    t2 = time.time()
    if params['use_multiprocessing']:
        with _MP_CTX.Pool(nprocs) as pool:
            St_sds = pool.starmap(compute_sd_overlaps_in_parallel, var_pool)
            pool.close()
            pool.join()
    else:
        St_sds = []
        for variable in var_pool:
            St_sd = compute_sd_overlaps_in_parallel(variable[0], variable[1])
            St_sds.append(St_sd)

    print('Done with computing the SD overlaps. Elapsed time:', time.time() - t2)
    if W_dia is not None:
        # the localization eigenvectors carry arbitrary signs - the diabatic states
        # follow the trajectory through the raw SD time overlaps of their own basis
        for i in range(len(St_sds)):
            O = W_dia[i].T @ np.asarray(St_sds[i].todense()).real @ W_dia[i + 1]
            W_dia[i + 1] = W_dia[i + 1] * np.where(np.diag(O) < 0.0, -1.0, 1.0)
    if params['do_state_reordering'] == 2 or params['do_state_reordering'] == 1:
        t2 = time.time()
        # For state reordering we need a list of CMATRIX for both St and energy
        St_sds_cmatrix = []
        E_sds_cmatrix = []
        for i in range(len(St_sds)):
            St_sd_cmatrix = data_conv.nparray2CMATRIX(np.array(St_sds[i].todense().real))
            St_sds_cmatrix.append(St_sd_cmatrix)
            E_sd_cmatrix = data_conv.nparray2CMATRIX(E_sds[i])
            E_sds_cmatrix.append(E_sd_cmatrix)
        # the trailing step has energies but no interval - it takes the final labels
        for i in range(len(St_sds), len(E_sds)):
            E_sds_cmatrix.append(data_conv.nparray2CMATRIX(E_sds[i]))
        print('Applying state-reordering to SDs overlaps...\n')
        perms = apply_state_reordering_general(St_sds_cmatrix, E_sds_cmatrix, params)
        # the relabeled overlaps AND energies replace the sorted ones - the saved
        # Hvib_sd_re midpoints carry the labels of St_sd
        for i in range(len(St_sds)):
            St_sds[i] = data_conv.MATRIX2scipynpz(St_sds_cmatrix[i].real())
        for i in range(len(E_sds)):
            E_sds[i] = data_conv.MATRIX2nparray(E_sds_cmatrix[i].real(), np.float64)
            if i > 0:
                sp.save_npz(F'{params["path_to_save_sd_Hvibs"]}/Hvib_sd_{start_time+i-1}_re.npz',
                            sp.csc_matrix(0.5 * (E_sds[i] + E_sds[i - 1])))
        if W_dia is not None:
            W_dia = [_permute_rows(W_dia[i], perms[i]) for i in range(len(W_dia))]
        print('Done with state-reordering of SDs. Elapsed time:', time.time() - t2)

    if params['is_many_body']:
        St_cis = []

        # every step start_time..finish_time-1 has CI energies: the midpoints cover the
        # same finish_time-start_time-1 intervals as the SD data
        for step in range(start_time, finish_time):
            # Adding 0.0 to the total energy
            E_ci_step = np.concatenate((np.array([0.0]), np.array(ci_energies[step - start_time])))
            if step > start_time:
                E_midpoint = np.diag(0.5 * (E_ci_step + E_ci_step_plus) * units.ev2Ha)
                sp.save_npz(F'{params["path_to_save_sd_Hvibs"]}/Hvib_ci_{step-1}_re.npz', sp.csc_matrix(E_midpoint))
            E_ci_step_plus = E_ci_step

        if params['do_state_reordering'] == 2 or params['do_state_reordering'] == 1:
            # Since we have performed state-reordering we need to
            # convert to scipy npz format now
            t2 = time.time()
            for i in range(len(St_sds_cmatrix)):
                sd2ci_prev = SD2CI[i]
                sd2ci_curr = SD2CI[i + 1]
                # Compute the St_ci
                St_ci = np.linalg.multi_dot([sd2ci_prev.T, St_sds[i].todense().real, sd2ci_curr])
                St_cis.append(sp.csc_matrix(St_ci))

            # Now we need to apply state-reordering to St_cis
            # Again the same procedure is needed to convert between data types
            # Set up the counter
            c = 0
            E_cis_cmatrix = []
            St_cis_cmatrix = []

            for step in range(start_time, finish_time):
                # Adding 0.0 to the total energy
                E_ci_step = np.concatenate((np.array([0.0]), np.array(ci_energies[step - start_time])))
                E_cis_cmatrix.append(data_conv.nparray2CMATRIX(np.diag(E_ci_step)))
                # one interval fewer than steps: the last step carries energies only
                if c < len(St_cis):
                    St_cis_cmatrix.append(data_conv.nparray2CMATRIX(np.array(St_cis[c].todense().real)))
                # if step>start_time:
                #    E_midpoint = np.diag(0.5*(E_ci_step+E_ci_step_plus)*units.ev2Ha)
                #    sp.save_npz(F'{params["path_to_save_sd_Hvibs"]}/Hvib_ci_{step-1}_re.npz',sp.csc_matrix(E_midpoint))
                # E_ci_step_plus = E_ci_step
                c += 1

            # if params['do_state_reordering']==2 or params['do_state_reordering']==1:
            print('Applying state-reordering to St_ci...\n')
            apply_state_reordering_general(St_cis_cmatrix, E_cis_cmatrix, params)
            # the relabeled CI energies replace the midpoints saved above
            for i in range(1, len(E_cis_cmatrix)):
                E_ci_prev = data_conv.MATRIX2nparray(E_cis_cmatrix[i - 1].real(), np.float64)
                E_ci_curr = data_conv.MATRIX2nparray(E_cis_cmatrix[i].real(), np.float64)
                sp.save_npz(F'{params["path_to_save_sd_Hvibs"]}/Hvib_ci_{start_time+i-1}_re.npz',
                            sp.csc_matrix(0.5 * (E_ci_prev + E_ci_curr) * units.ev2Ha))
            print('Done with state-reordering to St_cis. Elapsed time:', time.time() - t2)

            # Now convert back to scipy npz
            for i in range(len(St_cis_cmatrix)):
                St_cis[i] = data_conv.MATRIX2scipynpz(St_cis_cmatrix[i].real())

        else:
            for i in range(len(St_sds)):
                sd2ci_prev = SD2CI[i]
                sd2ci_curr = SD2CI[i + 1]
                # Compute the St_ci
                St_ci = np.linalg.multi_dot([sd2ci_prev.T, St_sds[i].todense().real, sd2ci_curr])
                St_cis.append(sp.csc_matrix(St_ci))
                # indexed by the loop variable: `step` is stale here (frozen at
                # finish_time - 2 by the energy loop above), so every iteration
                # overwrote the same file
                sp.save_npz(F'{params["path_to_save_sd_Hvibs"]}/St_ci_{i+start_time}_re.npz', sp.csc_matrix(St_ci))

    # Now, we move to applying phase-corrections
    if params['apply_phase_correction']:

        t3 = time.time()
        for step in range(len(St_sds)):
            # Initialize the nstates and cum_phase_aa and cum_phase_bb
            print('Applying phase-correction to St_sd matrix of step', step)
            if step == 0:
                nstates = int(St_sds[0].shape[0])  # /2)
                cum_phase_aa = np.ones((nstates, 1))
                cum_phase_bb = np.ones((nstates, 1))

            if W_dia is not None:
                # U_sd(n) refers to the phase-corrected SDs F(n)|SD(n)>: the cumulative
                # phases BEFORE this step's update
                _save_u_sd(res_dir_2, W_dia[step], cum_phase_aa, step + start_time)
            St_step_phase_corrected, cum_phase_aa, cum_phase_bb = apply_phase_correction_scipy(
                St_sds[step].real, step, cum_phase_aa, cum_phase_bb, two_spinor_format=False)
            if nac_algo == 0:
                # HST: Im(Hvib) = -d = (St^T - St)/(2 dt), with St_ij = <i(t)|j(t+dt)>
                Hvib_sd = 0.5 / dt * (St_step_phase_corrected.todense().T - St_step_phase_corrected.todense())
            elif nac_algo == 1:
                # Using Meek-Levine approach
                # turn the overlap matrix from scipy sparse format into MATRIX
                St_step_phase_corrected_MATRIX = data_conv.nparray2MATRIX(
                    np.array(St_step_phase_corrected.todense().real))
                # Computing the NAC values using Meek-Levine approach
                Hvib_sd_MATRIX = -1.0 * nac_npi(St_step_phase_corrected_MATRIX, dt)
                # Turn the MATRIX to numpy array so that we can save it using scipy.sparse library
                Hvib_sd = data_conv.MATRIX2nparray(Hvib_sd_MATRIX)

            sp.save_npz(F'{params["path_to_save_sd_Hvibs"]}/Hvib_sd_{step+start_time}_im.npz', sp.csc_matrix(Hvib_sd))
            sp.save_npz(F'{params["path_to_save_sd_Hvibs"]}/St_sd_{step+start_time}_re.npz', St_step_phase_corrected)
        print('Done with applying phase-correction to St_sd matrices. Elapsed time:', time.time() - t3)
        if W_dia is not None:
            _save_u_sd(res_dir_2, W_dia[len(St_sds)], cum_phase_aa, len(St_sds) + start_time)

        if params['is_many_body']:
            # Set up the counter
            c = 0
            t3 = time.time()
            for step in range(finish_time - start_time - 1):
                print('Applying phase-correction to St_ci matrix of step', step)
                if step == 0:
                    nstates = int(St_cis[0].shape[0])
                    cum_phase_aa = np.ones((nstates, 1))
                    cum_phase_bb = np.ones((nstates, 1))

                St_ci_step_phase_corrected, cum_phase_aa, cum_phase_bb = apply_phase_correction_scipy(
                    St_cis[step].real, step, cum_phase_aa, cum_phase_bb, two_spinor_format=False)
                if nac_algo == 0:
                    # HST: Im(Hvib) = -d = (St^T - St)/(2 dt), with St_ij = <i(t)|j(t+dt)>
                    Hvib_ci = 0.5 / dt * (St_ci_step_phase_corrected.todense().T -
                                           St_ci_step_phase_corrected.todense())
                elif nac_algo == 1:
                    # Using Meek-Levine approach
                    # turn the overlap matrix from scipy sparse format into MATRIX
                    St_ci_step_phase_corrected_MATRIX = data_conv.nparray2MATRIX(
                        np.array(St_ci_step_phase_corrected.todense().real))
                    # Computing the NAC values using Meek-Levine approach
                    Hvib_ci_MATRIX = -1.0 * nac_npi(St_ci_step_phase_corrected_MATRIX, dt)
                    # Turn the MATRIX to numpy array so that we can save it using scipy.sparse library
                    Hvib_ci = data_conv.MATRIX2nparray(Hvib_ci_MATRIX)

                sp.save_npz(
                    F'{params["path_to_save_sd_Hvibs"]}/Hvib_ci_{step+start_time}_im.npz',
                    sp.csc_matrix(Hvib_ci))
                sp.save_npz(
                    F'{params["path_to_save_sd_Hvibs"]}/St_ci_{step+start_time}_re.npz',
                    sp.csc_matrix(St_ci_step_phase_corrected))
                c += 1
            print('Done with applying phase-correction to St_ci matrices. Elapsed time:', time.time() - t3)

    else:
        for step in range(len(St_sds)):
            if nac_algo == 0:
                # HST: Im(Hvib) = -d = (St^T - St)/(2 dt), with St_ij = <i(t)|j(t+dt)>
                Hvib_sd = 0.5 / dt * (St_sds[step].todense().real.T - St_sds[step].todense().real)
            elif nac_algo == 1:
                # scipy sparse to MATRIX
                St_sd_step_MATRIX = data_conv.nparray2MATRIX(np.array(St_sds[step].todense().real))
                # Computing the NAC values using Meek-Levine approach
                Hvib_sd_MATRIX = -1.0 * nac_npi(St_sd_step_MATRIX, dt)
                # Turn the MATRIX to numpy array so that we can save it using scipy.sparse library
                Hvib_sd = data_conv.MATRIX2nparray(Hvib_sd_MATRIX)
            sp.save_npz(F'{params["path_to_save_sd_Hvibs"]}/Hvib_sd_{step+start_time}_im.npz', sp.csc_matrix(Hvib_sd))
            sp.save_npz(F'{params["path_to_save_sd_Hvibs"]}/St_sd_{step+start_time}_re.npz', St_sds[step])
        if W_dia is not None:
            for i in range(len(W_dia)):
                _save_u_sd(res_dir_2, W_dia[i], None, i + start_time)

        if params['is_many_body']:
            for step in range(finish_time - start_time - 1):
                St_ci_step = St_cis[step].todense().real
                if nac_algo == 0:
                    # HST: Im(Hvib) = -d = (St^T - St)/(2 dt), with St_ij = <i(t)|j(t+dt)>
                    Hvib_ci = 0.5 / dt * (St_ci_step.T - St_ci_step)
                elif nac_algo == 1:
                    # scipy sparse to MATRIX
                    St_ci_step_MATRIX = data_conv.nparray2MATRIX(np.array(St_cis[step].todense().real))
                    # Computing the NAC values using Meek-Levine approach
                    Hvib_ci_MATRIX = -1.0 * nac_npi(St_ci_step_MATRIX, dt)
                    # Turn the MATRIX to numpy array so that we can save it using scipy.sparse library
                    Hvib_ci = data_conv.MATRIX2nparray(Hvib_ci_MATRIX)
                sp.save_npz(
                    F'{params["path_to_save_sd_Hvibs"]}/Hvib_ci_{step+start_time}_im.npz',
                    sp.csc_matrix(Hvib_ci))
                sp.save_npz(
                    F'{params["path_to_save_sd_Hvibs"]}/St_ci_{step+start_time}_re.npz',
                    sp.csc_matrix(St_ci_step))


def make_var_pool(params, sd_states_reindexed_sorted, start_time, finish_time):
    """Per-step argument tuples for compute_sd_overlaps_in_parallel.

    Each task ships only the two SD lists it consumes - putting the full
    per-step list into every task makes the Pool pickling cost O(nsteps^2).
    """
    var_pool = []
    for step in range(finish_time - start_time - 1):
        p_task = dict(params)
        p_task['sd_states_reindexed_sorted'] = {step: sd_states_reindexed_sorted[step],
                                                step + 1: sd_states_reindexed_sorted[step + 1]}
        var_pool.append((step, p_task))
    return var_pool


def compute_sd_overlaps_in_parallel(step, params):
    """
    This function is used as an auxilary function that computes the SDs overlaps.
    It is used in the run_step3_sd_nacs_libint function.
    """
    with _blas_thread_limit(_worker_blas_threads(params)):
        return _compute_sd_overlaps(step, params)


def _compute_sd_overlaps(step, params):
    start_time = params['start_time']
    res_dir_1 = params['path_to_npz_files']
    active_space = params.get('active_space', None)
    active_space2 = ( np.array(active_space) + 1 ).tolist() # conversion to a different convention
    sd_states_reindexed_sorted = params['sd_states_reindexed_sorted']

    # SD overlap kernel: "auto" = batched LAPACK determinants + determinant-lemma shortcut
    # for low-degree excitations (machine precision vs the legacy per-pair loop, orders of
    # magnitude faster for large SD bases); "batched"/"lemma" force one; "per_pair" = ovlp_mat_arb
    kernel = params.get('sd_overlap_kernel', 'auto')
    ovlp = {'per_pair': mapping.ovlp_mat_arb,
            'batched': mapping.ovlp_mat_arb_batched,
            'lemma': mapping.ovlp_mat_arb_lemma_only,
            'auto': mapping.ovlp_mat_arb_auto}[kernel]

    #print("active_space = ", active_space)
    #print("active_space2 = ", active_space2)
    #print("sd_states_reindexed_sorted = ", sd_states_reindexed_sorted)

    t2 = time.time()
    print('Computing the SD overlaps for step', step)

    # The same procedure as above
    st_ks = np.array(sp.load_npz(F'{res_dir_1}/St_ks_{step+start_time}.npz').todense()
                     [active_space, :][:, active_space]).real
    s_ks_1 = np.array(sp.load_npz(F'{res_dir_1}/S_ks_{step+start_time}.npz').todense()
                      [active_space, :][:, active_space]).real
    s_ks_2 = np.array(sp.load_npz(F'{res_dir_1}/S_ks_{step+start_time+1}.npz').todense()
                      [active_space, :][:, active_space]).real
    _allow_dirty = params.get('allow_nonorthonormal_overlaps', False)
    _check_input_overlap(s_ks_1, 'S_ks(t)', step + start_time,
                         allow_dirty=_allow_dirty)
    _check_input_overlap(s_ks_2, 'S_ks(t+1)', step + start_time,
                         allow_dirty=_allow_dirty)
    # This is not needed anymore as we use the numpy approach
    # that works with numpy array of the matrices below
    x = sp.load_npz(F'{res_dir_1}/St_ks_{step+start_time}.npz').todense() 
    print("original dimensions of the S and ST matrices", x.shape )
    print("the dimensions of the reduced S and ST matrices", st_ks.shape)

    # Computing the overlaps for SDs
    t2 = time.time()
    
    mode = _orthonormalization_mode(params.get('apply_orthonormalization', 'lowdin_cached'))
    if mode == "lowdin_cached":
        # bra factor: reuse the previous step's ket factor when it provably
        # matches (expected step index + digest of the source matrix/SD list)
        U1 = _lowdin_cache_take(step, lowdin_digest(s_ks_1, repr(sd_states_reindexed_sorted[step])))
        if U1 is None:
            s_sd_1 = ovlp(
                sd_states_reindexed_sorted[step],
                sd_states_reindexed_sorted[step],
                s_ks_1,
                None)
            U1 = get_Lowdin_scipy(s_sd_1.real)
        s_sd_2 = ovlp(
            sd_states_reindexed_sorted[step + 1],
            sd_states_reindexed_sorted[step + 1],
            s_ks_2,
            None)
        U2 = get_Lowdin_scipy(s_sd_2.real)
        _lowdin_cache_put(step + 1, lowdin_digest(s_ks_2, repr(sd_states_reindexed_sorted[step + 1])), U2)
    elif mode == "lowdin":
        s_sd_1 = ovlp(
            sd_states_reindexed_sorted[step],
            sd_states_reindexed_sorted[step],
            s_ks_1,
            None)
        s_sd_2 = ovlp(
            sd_states_reindexed_sorted[step + 1],
            sd_states_reindexed_sorted[step + 1],
            s_ks_2,
            None)

    st_sd = ovlp(sd_states_reindexed_sorted[step],
                 sd_states_reindexed_sorted[step + 1],
                 st_ks,
                 None)

    if mode == "lowdin_cached":
        print('Applying orthonormalization for SDs for step', step)
        st_sd = np.linalg.multi_dot([U1.conj().T, st_sd.real, U2])
    elif mode == "lowdin":
        print('Applying orthonormalization for SDs for step', step)
        st_sd, s_sd = apply_orthonormalization_scipy(s_sd_1.real, s_sd_2.real, st_sd.real)
    # before the spin adaptation - the sqrt(2) GS scaling breaks the
    # bound by design
    if mode != "none":
        _check_contraction(np.asarray(st_sd).real, step + start_time, 'SD')
    if params['spin_adapted']:
        st_sd[:, 0] *= np.sqrt(2)
        st_sd[0, :] *= np.sqrt(2)
        st_sd[0, 0] *= 0.5
    print(F'Done with computing the SD overlap of step {step}. Elapsed time {time.time()-t2}')

    return sp.csc_matrix(st_sd)
