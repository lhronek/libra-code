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
import re
import time
import glob
import json
import numpy as np
from libra_py.workflows.nbra.mp_context import MP_CTX as _MP_CTX
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
if sys.platform == "cygwin":
    from cyglibra_core import *
elif sys.platform == "linux" or sys.platform == "linux2":
    from liblibra_core import *

from . import mapping, step2_many_body, step3_many_body
from libra_py.citools import slatdet
import util.libutil as comn
from libra_py.params_check import refuse_unknown_keys, restrict
import libra_py.packages.cp2k.methods as CP2K_methods
import libra_py.units as units
import libra_py.data_read as data_read
import libra_py.data_conv as data_conv
import libra_py.hungarian as hungarian

# the continuity floor of a followed state: more than half of its weight on its own continuation (cos 45 degrees) -
# the default of state_reordering_min_overlap and diabatic_state_reordering_min_overlap, the edge and record floor
# of _classify_continuations
CONTINUITY_FLOOR = math.sqrt(0.5)
from libra_py import orthogonalizations
from libra_py.orthogonalizations import (validate_overlap_matrix,
                                         get_Lowdin_scipy,
                                         apply_orthonormalization_scipy,
                                         lowdin_digest)


# every key the step-3 workflow reads - the union over its entries (run, the three run_step3_* routes, the SD-basis sorting, get_step2_data,
# pyxaid2libra, the reorderings, output_sorted_Hvibs, step3_many_body.run and its helpers) and the readers they forward the dict to - one
# dict serves the whole step, so every entry refuses a key outside this set (tier0_workflow_keys [step3] recomputes the union from the source)
STEP3_KEYS = frozenset((
    "CI_basis", "Hvib_im_prefix", "Hvib_im_suffix", "Hvib_re_prefix", "Hvib_re_suffix", "SD_basis", "SD_energy_corr",
    "S_data_im_prefix", "S_data_im_suffix", "S_data_re_prefix", "S_data_re_suffix", "St_data_im_prefix", "St_data_im_suffix",
    "St_data_re_prefix", "St_data_re_suffix", "active_space", "active_space_num_occ_orbitals", "active_space_selection",
    "active_space_selection_algo", "active_space_selection_alpha", "allow_nonorthonormal_overlaps", "apply_orthonormalization",
    "apply_phase_correction", "blas_num_threads", "ci_read_drops", "ci_threshold", "curr_step", "data_dim", "data_im_prefix",
    "data_im_suffix", "data_re_prefix", "data_re_suffix", "data_set_paths", "diabatic_state_reordering_alpha",
    "diabatic_state_reordering_min_overlap", "diabatization_scheme", "do_diabatic_state_reordering", "do_orthogonalization",
    "do_output", "do_phase_correction", "do_state_reordering", "dt", "es_software", "finish_time", "fsnap", "get_imag",
    "get_real", "highest_orbital", "homo_alpha_index", "homo_beta_index", "homo_index", "hvib_data_im_prefix",
    "hvib_data_im_suffix", "hvib_data_re_prefix", "hvib_data_re_suffix", "isUKS", "is_many_body", "isnap", "ks_homo_index",
    "ks_orbital_indicies", "localized_carrier", "logfile_directory", "logfile_name", "lowest_orbital", "nac_algo", "nprocs",
    "npz_file_ks_homo_alpha_index", "npz_file_ks_homo_beta_index", "npz_file_ks_homo_index", "num_occ_alpha", "num_occ_beta",
    "num_occ_states", "num_occ_states_per_block", "num_unocc_alpha", "num_unocc_beta", "num_unocc_states", "number_of_states",
    "occupancy_tolerance", "orbital_indices", "orbital_normalization", "orbital_perms", "orbital_phase_correction",
    "orbital_space", "outdir", "output_dir_prefix", "output_set_paths", "path_to_npz_files", "path_to_save_ks_Hvibs",
    "path_to_save_sd_Hvibs", "pod_occupied", "read_S_data", "read_S_im", "read_S_re", "read_St_data", "read_St_im",
    "read_St_re", "read_hvib_data", "read_hvib_im", "read_hvib_re", "save_files", "sd_capture_max_drop", "sd_charge",
    "sd_overlap_kernel", "sd_states_reindexed_sorted", "sd_unique_basis", "sorting_type", "spin_adapted", "start_time",
    "state_normalization", "state_phase_correction", "state_reordering_alpha", "state_reordering_min_overlap", "time_step",
    "use_multiprocessing", "verbosity"
))


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

        Note: the _im files/values follow the canonical convention Im(Hvib) = -d = -(St - St^T)/(2 dt) - directories
        written by the current step 3 carry hvib_convention.json, markerless directories are legacy and hold +d (the NAC
        itself)

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
    refuse_unknown_keys(params, STEP3_KEYS, "step3.get_step2_data", "params", "no function of the step-3 workflow reads")
    comn.check_input(params, default_params, critical_params)

    S, St, Hvib = [], [], []
    prms = restrict(_params, data_read.DATA_READ_KEYS)

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
    refuse_unknown_keys(params, STEP3_KEYS, "step3.output_sorted_Hvibs", "params", "no function of the step-3 workflow reads")
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
    factor (the expected step index AND the content digest both match) -
    None otherwise. Hits occur on every step after the first in a serial
    loop and within each contiguous Pool.starmap chunk - a miss just means
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


def _assignment_weights(O, e_rows, e_cols, alpha):
    """The Munkres-Kuhn cost of the state followings, |O_ab|^2 exp(-(alpha (e_a - e_b))^2): O the time overlap
    (real or complex), e_rows the energies of the rows (the followed states), e_cols those of the columns (the
    candidate positions, the rows' again when None), alpha the width in 1/Hartree - the one formula of
    make_cost_mat and _assign_states. No penalty under alpha 0 or without energies."""
    O = np.asarray(O)
    W = (O * O.conj()).real if np.iscomplexobj(O) else O * O
    if alpha and e_rows is not None:
        e = np.nan_to_num(np.asarray(e_rows, dtype=float))
        ec = e if e_cols is None else np.nan_to_num(np.asarray(e_cols, dtype=float))
        W = W * np.exp(-(alpha * (e[:, None] - ec[None, :])) ** 2)
    return W


def make_cost_mat(orb_mat_inp, en_mat_inp, alpha, en_mat_cols=None):
    """

    Makes the cost matrix from a given TDM and information on states' energies

    Args:
        orb_mat_inp  ( CMATRIX(nstates,nstates) or MATRIX(nstates,nstate) ): the transition density matrix
            TDM in a given basis. Here, ```nstates``` - the number of states (e.g. the number of doubly-occupied
            orbitals )

        en_mat_inp ( MATRIX(nstates, nstates) ): Matrix of energies in a given basis [units: a.u.] - the energies
            of the rows (the states followed so far, in their labels)

        alpha ( float ): Parameter controlling the range of the orbitals that can participate in
            the reordering. Setting is to 0 makes all orbitals be considered for reordering
            Setting it to a large number makes the effective number of orbitals participating
            in the reordering smaller - this can be used to turn off the reordering. [units: a.u.^-1]

        en_mat_cols ( MATRIX(nstates, nstates) ): the energies of the columns (the candidate positions of the next
            frame, taken at the earlier frame in their stored order) - the row matrix itself when None, which is
            right only while no state has been relabeled

    Returns:
        MATRIX(nstates, nstates): the matrix of the cost values for different pairs of states

    """

    e_rows = np.diag(data_conv.MATRIX2nparray(en_mat_inp)).real
    e_cols = None if en_mat_cols is None else np.diag(data_conv.MATRIX2nparray(en_mat_cols)).real
    W = _assignment_weights(data_conv.MATRIX2nparray(orb_mat_inp), e_rows, e_cols, alpha)
    return data_conv.nparray2MATRIX(np.ascontiguousarray(W, dtype=np.float64))


def _spin_block_permutation(perm_aa, perm_bb, nstates):
    """The 2n-index permutation of a spin-block matrix from its alpha and beta block permutations."""
    p = intList()
    for k in range(nstates):
        p.append(perm_aa[k])
    for k in range(nstates):
        p.append(nstates + perm_bb[k])
    return p


def _maximize_assignment(cost_mat, reordering):
    """The [row, column] pairs of the assignment that maximizes the total of a square cost matrix (a libra MATRIX,
    make_cost_mat) - do_state_reordering 2 libra_py.hungarian.maximize, 22 scipy's linear_sum_assignment, the same
    assignment."""
    if reordering == 2:
        return hungarian.maximize(cost_mat)
    if reordering == 22:
        rows, cols = linear_sum_assignment(data_conv.MATRIX2nparray(cost_mat, np.float64), maximize=True)
        return [[int(r), int(c)] for r, c in zip(rows, cols)]
    raise ValueError(F"do_state_reordering = {reordering!r} is not a Munkres-Kuhn assignment - 2 (libra_py.hungarian) "
                     F"and 22 (scipy's linear_sum_assignment) are")


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
                    - 22: the Munkres-Kuhn assignment by scipy's linear_sum_assignment - the same result as 2, in a
                      fraction of its time

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
    refuse_unknown_keys(params, STEP3_KEYS, "step3.apply_state_reordering", "params", "no function of the step-3 workflow reads")
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
    en_pos_aa = CMATRIX(nstates, nstates)
    en_pos_bb = CMATRIX(nstates, nstates)

    for i in range(0, nsteps):

        if params["do_state_reordering"] == 1:
            """
            A simple approach based on permuations - but this is not robust
            may have loops
            """
            # the frame's own labels first - its energies and the rows of its time overlap, recorded as this step's
            # permutation, as the Hungarian branch does
            St[i].permute_rows(perm_cum)
            E[i].permute_rows(perm_cum)
            E[i].permute_cols(perm_cum)
            perms.append([perm_cum[k] for k in range(2 * nstates)])

            # the search on the relabeled rows: perm_t[r] is the column of the next frame that carries the state of
            # label r, so it is the next frame's label permutation itself
            perm_t = get_reordering(St[i])
            for k in range(2 * nstates):
                perm_cum[k] = perm_t[k]

            # Because St = <psi(t)|psi(t+dt)> - the columns take the labels of the next frame
            St[i].permute_cols(perm_cum)

        elif params["do_state_reordering"] in (2, 22):
            """
            The Hungarian approach
            """
            pop_submatrix(St[i], aa, alp, alp)
            pop_submatrix(St[i], ab, alp, bet)
            pop_submatrix(St[i], ba, bet, alp)
            pop_submatrix(St[i], bb, bet, bet)

            # the energies carry the row labels - unpermuted, a state relabeled at a
            # crossing would keep the other state's energy - the stored order is kept for the columns, the
            # candidate positions of the next frame at this frame's energies
            E_pos = CMATRIX(E[i])
            perm_full = _spin_block_permutation(perm_t_aa, perm_t_bb, nstates)
            E[i].permute_rows(perm_full)
            E[i].permute_cols(perm_full)
            perms.append([perm_full[k] for k in range(2 * nstates)])

            # Extract the alpha and beta orbtial energies
            pop_submatrix(E[i], en_mat_aa, alp, alp)
            pop_submatrix(E[i], en_mat_bb, bet, bet)
            pop_submatrix(E_pos, en_pos_aa, alp, alp)
            pop_submatrix(E_pos, en_pos_bb, bet, bet)

            # Permute rows
            aa.permute_rows(perm_t_aa)
            ab.permute_rows(perm_t_aa)
            ba.permute_rows(perm_t_bb)
            bb.permute_rows(perm_t_bb)

            # compute the cost matrices for diagonal blocks
            cost_mat_aa = make_cost_mat(aa, en_mat_aa, params["state_reordering_alpha"], en_pos_aa)
            cost_mat_bb = make_cost_mat(bb, en_mat_bb, params["state_reordering_alpha"], en_pos_bb)

            # Solve the optimal assignment problem for diagonal blocks
            res_aa = _maximize_assignment(cost_mat_aa, params["do_state_reordering"])
            res_bb = _maximize_assignment(cost_mat_bb, params["do_state_reordering"])

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
        else:
            raise ValueError(F"do_state_reordering = {params['do_state_reordering']!r} has no branch here - 1 (Sato), "
                             F"2 (Munkres-Kuhn, libra_py.hungarian) and 22 (the same assignment by scipy) are")

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


def _refuse_former_names(params):
    """An option set by its former name would leave the present one at its default without a word - refused with the
    present name, in every step-3 writer."""
    for former, present in (('koopmans_charge', 'sd_charge'), ('diabatization', 'localized_carrier'),
                            ('fragment_operator', 'diabatization_scheme')):
        if former in params:
            raise ValueError(F"params['{former}'] is params['{present}'] now - rename it in the runner"
                             + (" (its values 'particle' and 'joint' are 'electron' and 'both')"
                                if former == 'diabatization' else ""))


def _check_reordering_options(params, route):
    """The state-reordering options the step-3 writers share, refused outside their domains: do_state_reordering and
    do_diabatic_state_reordering in 0 (none), 1 (Sato), 2 (Munkres-Kuhn by libra_py.hungarian), 22 (the same
    assignment by scipy's linear_sum_assignment) - the diabatic one also in 5 and 52 (the reordering of 2 and 22
    with the orthogonal alignment of a group's states, _polar_factor) - the two alphas non-negative widths in
    1/Hartree, the two continuity thresholds between 0 (off) and 1 - the keys a route carries. One contract for
    run_step3_ks_nacs_libint, run_step3_sd_nacs_libint and run_step3_pod."""
    for key in ('do_state_reordering', 'do_diabatic_state_reordering'):
        listed = (0, 1, 2, 22, 5, 52) if key == 'do_diabatic_state_reordering' else (0, 1, 2, 22)
        if key in params and params[key] not in listed:
            raise ValueError(F"{route}: {key} = {params[key]!r} is not a listed option - 0 (none), 1 (Sato), 2 "
                             F"(Munkres-Kuhn, libra_py.hungarian) and 22 (the same assignment by scipy) are supported"
                             + (", and 5 / 52 (the reordering of 2 / 22 with the orthogonal alignment of a group's "
                                "states to the previous frame)" if key == 'do_diabatic_state_reordering' else ""))
    for key in ('state_reordering_alpha', 'diabatic_state_reordering_alpha'):
        if key in params and not (np.isreal(params[key]) and params[key] >= 0.0):
            raise ValueError(F"{route}: {key} = {params[key]!r} - a non-negative width in 1/Hartree is needed")
    for key in ('state_reordering_min_overlap', 'diabatic_state_reordering_min_overlap'):
        if key in params and not (np.isreal(params[key]) and 0.0 <= params[key] <= 1.0):
            raise ValueError(F"{route}: {key} = {params[key]!r} - a threshold between 0 (off) and 1 is needed")


def _check_orbital_window(params, data_dim, where):
    """The step-2 npz supermatrices are 2*(highest_orbital - lowest_orbital
    + 1) on a side (alpha+beta blocks, both spins share the window), so the
    claimed window is checked against the stored data - a res-dir
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
    sample_npz_file = sorted(glob.glob(f'{params["path_to_npz_files"]}/S_ks_*.npz'))[0]
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
            P2C.set(i, j, complex(coeff[j][i]))

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

            Note: the _im files/values follow the canonical convention Im(Hvib) = -d = -(St - St^T)/(2 dt) - directories
            written by the current step 3 carry hvib_convention.json, markerless directories are legacy and hold +d (the
            NAC itself)

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
    refuse_unknown_keys(params, STEP3_KEYS, "step3.run", "params", "no function of the step-3 workflow reads")
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
    refuse_unknown_keys(params, STEP3_KEYS, "step3.pyxaid2libra", "params", "no function of the step-3 workflow reads")
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
                    - 22: the Munkres-Kuhn assignment by scipy's linear_sum_assignment - the same result as 2, in a
                      fraction of its time
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
    refuse_unknown_keys(params, STEP3_KEYS, "step3.apply_state_reordering_general", "params", "no function of the step-3 workflow reads")
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
            # the frame's own labels first - its energies and the rows of its time overlap, recorded as this step's
            # permutation, as the Hungarian branch does
            St[i].permute_rows(perm_cum)
            E[i].permute_rows(perm_cum)
            E[i].permute_cols(perm_cum)
            perms.append([perm_cum[k] for k in range(nstates)])

            # the search on the relabeled rows: perm_t[r] is the column of the next frame that carries the state of
            # label r, so it is the next frame's label permutation itself
            perm_t = get_reordering(St[i])
            for k in range(nstates):
                perm_cum[k] = perm_t[k]

            # Because St = <psi(t)|psi(t+dt)> - the columns take the labels of the next frame
            St[i].permute_cols(perm_cum)

        elif params["do_state_reordering"] in (2, 22):
            """
            The Hungarian approach
            """

            # Permute rows
            St[i].permute_rows(perm_t)
            # the energies carry the row labels - unpermuted, a state relabeled at a
            # crossing would keep the other state's energy - the stored order is kept for the columns, the
            # candidate positions of the next frame at this frame's energies
            E_pos = CMATRIX(E[i])
            E[i].permute_rows(perm_t)
            E[i].permute_cols(perm_t)
            perms.append([perm_t[k] for k in range(nstates)])
            # compute the cost matrices
            cost_mat = make_cost_mat(St[i], E[i], params["state_reordering_alpha"], E_pos)

            # Solve the optimal assignment problem for diagonal blocks
            res = _maximize_assignment(cost_mat, params["do_state_reordering"])

            # Convert the list of lists into the permutation object
            for row in res:
                perm_t[row[0]] = row[1]  # for < i | i > this becomes a new value: perm_t = P_{n+1}

            # Permute the blocks by col
            St[i].permute_cols(perm_t)
        else:
            raise ValueError(F"do_state_reordering = {params['do_state_reordering']!r} has no branch here - 1 (Sato), "
                             F"2 (Munkres-Kuhn, libra_py.hungarian) and 22 (the same assignment by scipy) are")

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
    refuse_unknown_keys(params, STEP3_KEYS, "step3.sort_unique_SD_basis", "params", "no function of the step-3 workflow reads")
    comn.check_input(params, default_params, critical_params)

    istep = params["isnap"]
    fstep = params["fsnap"]
    sorting_type = params["sorting_type"]

    E_sd = []
    sd_states_reindexed_sorted = []
    sd_states_unique_sorted = []
    SD_energy_corr = [0.0] * len(sd_states_reindexed)
    nstates_sd = len(sd_states_reindexed)
    # the descriptors name the excitations alone: a basis led by the reference determinant (the SCF
    # ground state) carries one determinant more than descriptors, the frozen-orbital bases of
    # sd_charge +-1 (koopmans_sd_basis) one descriptor per determinant
    offset = nstates_sd - len(sd_states_unique)
    if offset not in (0, 1):
        raise ValueError(F"{nstates_sd} determinants against {len(sd_states_unique)} descriptors - the basis carries "
                         "either the reference determinant first (one more) or one descriptor per determinant")
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

                # the reference determinant (offset 1) has no descriptor - the descriptors follow the excitations
                if state >= offset:
                    sd_states_unique_sorted[step].append(sd_states_unique[state - offset])

        elif sorting_type == "energy":

            reindex_nsteps.append(reindex)

            # For each SD basis, make the energy matrix and reindex the list of basis according to their energies
            for i in range(len(reindex)):
                # This is making the energy matrix
                E_sd[step].set(i, i, E_this_sd.get(int(reindex[i]), int(reindex[i])))
                # This is reindexing the list of SD bases at this time step according to their energies
                sd_states_reindexed_sorted[step].append(sd_states_reindexed[int(reindex[i])])
                # print( sd_states_reindexed_sorted[step][i], ( E_sd[step].get( i, i ) - E_sd[step].get( 0, 0 ) ).real * units.au2ev )

            for i in range(len(reindex)):
                if int(reindex[i]) >= offset:
                    sd_states_unique_sorted[step].append(sd_states_unique[int(reindex[i]) - offset])

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
    refuse_unknown_keys(params, STEP3_KEYS, "step3.sort_unique_SD_basis_scipy", "params", "no function of the step-3 workflow reads")
    comn.check_input(params, default_params, critical_params)

    istep = params["isnap"]
    fstep = params["fsnap"]
    sorting_type = params["sorting_type"]

    SD_energy_corr = [0.0] * len(sd_states_reindexed)
    nstates_sd = len(sd_states_reindexed)
    # the descriptors name the excitations alone: a basis led by the reference determinant (the SCF
    # ground state) carries one determinant more than descriptors, the frozen-orbital bases of
    # sd_charge +-1 (koopmans_sd_basis) one descriptor per determinant
    offset = nstates_sd - len(sd_states_unique)
    if offset not in (0, 1):
        raise ValueError(F"{nstates_sd} determinants against {len(sd_states_unique)} descriptors - the basis carries "
                         "either the reference determinant first (one more) or one descriptor per determinant")
    reindex_nsteps = []

    # At this step, compute the energy of the SD
    active_space = params['active_space']
    E_ks = _window_matrix(params, 'E_ks', step, active_space)
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

            # the reference determinant (offset 1) has no descriptor - the descriptors follow the excitations
            if state >= offset:
                sd_states_unique_sorted.append(sd_states_unique[state - offset])

    elif sorting_type == "energy":

        reindex_nsteps.append(reindex)

        # For each SD basis, make the energy matrix and reindex the list of basis according to their energies
        for i in range(len(reindex)):
            # This is reindexing the list of SD bases at this time step according to their energies
            sd_states_reindexed_sorted.append(sd_states_reindexed[int(reindex[i])])
        E_this_sd = E_this_sd[reindex]

        for i in range(len(reindex)):
            if int(reindex[i]) >= offset:
                sd_states_unique_sorted.append(sd_states_unique[int(reindex[i]) - offset])
    # print('Flag E_this_sd:',E_this_sd.shape)
    # print('Flag E_this_sd itself:',E_this_sd)
    # print('Flag np.diag(E_this_sd):',np.diag(E_this_sd))
    E_this_sd_sparse = sp.csc_matrix(np.diag(E_this_sd))

    return E_this_sd_sparse, sd_states_unique_sorted, sd_states_reindexed_sorted, reindex_nsteps


# shared with the TSH dynamics loop (dyn_params["blas_num_threads"]) -
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
    other causes). Raise by default - allow_dirty (the
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
                      fractional matrix power per step serially - also hits inside
                      contiguous Pool chunks). This stage historically always
                      orthonormalized, whatever the flag said - that output is preserved
                      exactly.
                    - "none" or False: skip - raw St/S pass through
            * **params['do_state_reordering']** (integer): the value for performing the state reordering
                the vlaues it takes are:
                    - 0: no state reordering - same as in Pyxaid
                    - 1: older method (is not robust, may or may not work)
                    - 2: Hungarian algorithm [default]
                    - 22: the Hungarian assignment by scipy's linear_sum_assignment - the same result as 2, faster
            * **params['nac_algo']** ( int ): selection of a method to compute NACs
                    - 0 : Hamess-Shiffer-Tully (HST), finite difference [ default ]
                    - 1 : Meek-Levine NPI
            Every key the route does not read is refused: the former option names with their present names, and any
            key of the SD or POD route (active_space_selection, do_diabatic_state_reordering, is_many_body and kin)
            with the list of the route's keys - a runner hands every route its own dict.
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
    # the keys of the KS route: its defaults, the window and HOMO keys it reads, the keys it writes into the dict for
    # its helpers (a runner may hand the dict in again) - every other key is refused, as on the POD route: the SD
    # and POD options (active_space_selection, do_diabatic_state_reordering, the many-body keys) have no reading here
    known = set(default_params) | {'lowest_orbital', 'highest_orbital', 'num_occ_states', 'num_unocc_states',
                                   'num_occ_alpha', 'num_unocc_alpha', 'num_occ_beta', 'num_unocc_beta', 'isUKS',
                                   'logfile_directory', 'use_multiprocessing', 'active_space', 'data_dim',
                                   'homo_index', 'ks_homo_index', 'homo_alpha_index', 'homo_beta_index',
                                   'npz_file_ks_homo_index', 'npz_file_ks_homo_alpha_index',
                                   'npz_file_ks_homo_beta_index'}
    _refuse_former_names(params)
    unknown = sorted(set(params) - known)
    if unknown:
        raise ValueError(F"run_step3_ks_nacs_libint: params {unknown} are not options of the KS route - its keys are "
                         F"{sorted(known)}")
    refuse_unknown_keys(params, STEP3_KEYS, "step3.run_step3_ks_nacs_libint", "params", "no function of the step-3 workflow reads")
    comn.check_input(params, default_params, critical_params)
    params['apply_orthonormalization'] = _orthonormalization_mode(params['apply_orthonormalization'])
    _check_reordering_options(params, 'run_step3_ks_nacs_libint')
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
    sample_npz_file = sorted(glob.glob(params['path_to_npz_files'] + '/S_ks_*.npz'))[0]
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
    # the _im files store Im(Hvib) = -d - files WITHOUT this marker predate
    # the convention restoration and hold +d (the NAC itself)
    marker_file = os.path.join(res_dir_2, 'hvib_convention.json')
    if not os.path.exists(marker_file) and glob.glob(os.path.join(res_dir_2, '*_im.npz')):
        raise ValueError(F"{res_dir_2} already holds *_im.npz files without a "
                         F"convention marker (legacy +d data) - writing canonical "
                         F"-d files into it would mix conventions in one directory; "
                         F"use a fresh output path or move the legacy files away")
    # the record a step-4 runner reads: isUKS decides with sd_charge whether the spin-adaptation
    # factor sqrt 2 belongs on the NACs of the reference SD (closed-shell neutral reference only)
    with open(marker_file, 'w') as f:
        json.dump({'im_file_contains': 'Im(Hvib) = -d', 'isUKS': int(params.get('isUKS', 0))}, f)
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
    if params['do_state_reordering'] in (1, 2, 22):
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
    # stage), "lowdin_cached" (default - identical result - the ket factor is carried
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
    raise ValueError(F"sd_charge {charge} - use 0, +1 (hole basis) or -1 (electron basis)")


def _sd_block_index(orb, nact):
    """Row of a signed reindexed spin-orbital in the 2-block active-space matrices
    (alpha j -> j-1, beta -j -> nact+j-1, the ovlp_arb convention)."""
    return orb - 1 if orb > 0 else nact + (-orb) - 1


def _permute_rows(W, perm):
    """Row permutation of a numpy matrix in the CMATRIX.permute_rows convention: row k of the result is
    row perm[k] of W, the same relabeling apply_state_reordering_general applies to the overlaps. The dtype
    is kept (the T matrices are complex-typed with zero imaginary parts)."""
    return np.asarray(W)[np.asarray(perm, dtype=int), :]


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
    determinant that share the hole ("electron": the excited electron varies), the
    particle ("hole": the hole varies), or the spin channel ("both": the complete
    hole x particle product of a channel, holes and electrons localized together).
    An open-shell reference joins the manifold whose defining orbital it occupies singly
    (a hole manifold whose particle's spin partner is occupied - the cation's hole - or a
    particle manifold whose hole's partner is empty - the anion's electron): its own
    configuration is one of the localized states. That reference holds ONE singly-occupied
    orbital (a doublet) - with several it would join every one of their manifolds, and is
    refused.

    Args:
        sd_lists (list of lists of int): the SDs of one step in the signed reindexed
            spin-orbital notation of ovlp_arb, in their row order
        kind (str): "electron", "hole" or "both"
        active_space, data_dim: the 2-block window indices and dimension of the step - the
            spin-partner test of an open-shell reference needs them (None: the reference
            is taken as closed-shell and never joins a manifold)

    Returns:
        gs (int): row of the reference determinant (the lowest orbital sum)
        groups (list of dict): manifolds with >= 2 members - "key" (the shared spin-orbital,
            or the channel sign for "both"), "rows" (member rows), "tau" (parity signs).
            electron/hole: "orbs" (the varying spin-orbital of every member, ascending) and
            "sigma" - the localized state with the varying orbital psi_A = sum_j w_jA orb_j
            reads sum_j tau_j sigma_j w_jA |SD_row_j> (tau_j: parity between the member's list
            and the fixed-orbitals-first order, sigma_j: the cofactor sign (-1)^j of a hole
            manifold, 1 for a particle manifold). both: "pairs" (the (hole, particle) of every
            member), "holes" and "parts" (the ascending hole and particle spin-orbitals) -
            the state a+_Y a_X |ref> with localized psi_X = sum_i wh_iX hole_i and
            phi_Y = sum_a wp_aY part_a reads sum_j tau_j wh_(i_j)X wp_(a_j)Y |SD_row_j>, tau_j
            the parity between the member's list and the in-place replacement of its hole in
            the reference list (a+_a a_i |ref> is the in-place determinant)
        singles (list of int): rows left untransformed (the reference, one-member manifolds,
            non-single excitations)

    Raises ValueError for an unknown kind, for a reference with more than one singly-occupied
    orbital, for "both" when a channel's single excitations do not form the complete
    hole x particle product or when the reference is open-shell (the joint states presume a
    closed-shell reference).
    """
    if kind not in ("electron", "hole", "both"):
        raise ValueError(F"localized_carrier {kind!r} - use 'none', 'hole', 'electron' or 'both' (the former 'particle' "
                         F"is 'electron', the former 'joint' is 'both')")
    sets = [frozenset(int(o) for o in sd) for sd in sd_lists]
    gs = min(range(len(sets)), key=lambda k: sum(abs(o) for o in sets[k]))
    ref = sets[gs]

    def singly_occupied(orb):
        # orb's spatial orbital holds exactly one electron in the reference
        if active_space is None:
            return False
        partner = _spin_partner(orb, active_space, data_dim)
        return partner is not None and ((orb in ref) != (partner in ref))

    if kind != "both" and active_space is not None:
        open_shells = sorted({abs(int(o)) for o in ref if singly_occupied(o)})
        if len(open_shells) > 1:
            raise ValueError(F"the reference determinant holds {len(open_shells)} singly-occupied orbitals "
                             F"{open_shells} - the localized manifolds of the '{kind}' carrier take an open-shell "
                             "reference with one singly-occupied orbital (a doublet), which joins the manifold of "
                             "that orbital - localize this reference with localized_carrier 'none'")
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
        if kind == "both":
            if singly_occupied(h) or singly_occupied(pt):
                raise ValueError("the localization of both carriers presumes a closed-shell reference - the "
                                 F"reference holds {h if singly_occupied(h) else pt} singly "
                                 "(use 'hole' for a cation, 'electron' for an anion)")
            manifolds.setdefault(1 if pt > 0 else -1, []).append(((h, pt), k))
            continue
        key, var = (h, pt) if kind == "electron" else (pt, h)
        manifolds.setdefault(key, []).append((var, k))
    groups = []
    ref_list = [int(o) for o in sd_lists[gs]]
    for key in sorted(manifolds):
        members = sorted(manifolds[key])
        if kind != "both" and singly_occupied(key):
            # the reference is the member whose varying orbital is the manifold's own
            # defining orbital (the hole sits in the particle p, or the electron in the hole h)
            members = sorted(members + [(key, gs)])
            if gs in singles:
                singles.remove(gs)
        if len(members) < 2:
            singles.append(members[0][1])
            continue
        if kind == "both":
            pairs = [ia for ia, k in members]
            rows = [k for ia, k in members]
            holes = sorted({h for h, pt in pairs})
            parts = sorted({pt for h, pt in pairs})
            if len(set(pairs)) != len(pairs) or len(pairs) != len(holes) * len(parts):
                raise ValueError(F"the localization of both carriers needs the complete hole x particle product of "
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
        if kind == "electron":
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


def localize_fragment_block(P_frag, rows, energy, tol=0.25):
    """
    Orbitals localized on fragments within the span of the given orbitals, by
    sequential projection: fragment 0 takes the top round(tr P_0) eigenvectors of its
    operator in the span, every next fragment the top ones of the remaining span, the
    last fragment the rest. Within a fragment the population settles nothing (its orbitals
    all sit on the fragment, the eigenvalues differ by the tails), so the fragment's
    orbitals are rotated to diagonalize the energy operator there and ordered by increasing
    energy - the fragment's own states, energy-resolved, as the block diagonalization of POD
    gives them.

    Args:
        P_frag (list of arrays): the fragment operators in a common orthonormal orbital
            basis (only the [rows, rows] blocks are read)
        rows (list of int): the orbitals spanning the localization space
        energy (array): the operator in the same basis whose expectation value is the energy
            of the diabatic state up to a constant (only the [rows, rows] block is read) - the
            orbital energies for an electron manifold, their negative for a hole manifold
        tol (float): the integer-occupancy tolerance - the trace of a fragment's operator in the span
            may differ from its integer by at most tol (params['occupancy_tolerance'] of the step-3
            routes, 0.25 by default), the last fragment included
    Returns:
        w (array (m, m)): columns = localized orbitals in the span, fragment-major, within
            a fragment by increasing energy
        frag (list of int): the fragment of every column
        pops (array (m, nfrag)): population of every column on every fragment
        e (array (m,)): the energy of every column, diag(w^T energy w)

    Raises ValueError when a fragment holds a non-integer number of orbitals of the span
    (|tr - round(tr)| > tol, the last fragment against the rest of the span): the span mixes with
    orbitals the fragments do not localize.
    """
    if not 0.0 < tol < 0.5:
        raise ValueError(F"occupancy_tolerance {tol} - the integer test needs 0 < tol < 0.5")
    m = len(rows)
    blocks = [0.5 * (B + B.T) for B in
              (np.asarray(P, dtype=float)[np.ix_(rows, rows)] for P in P_frag)]
    H = np.asarray(energy, dtype=float)[np.ix_(rows, rows)]
    H = 0.5 * (H + H.T)
    Q = np.eye(m)
    cols, frag = [], []
    for f, B in enumerate(blocks):
        left = Q.shape[1]
        if left == 0:
            break
        M = Q.T @ B @ Q
        tr = float(np.trace(M))
        # the last fragment takes the rest of the span - its trace has to say so as well
        n_f = left if f == len(blocks) - 1 else int(round(tr))
        if abs(tr - n_f) > tol or n_f > left:
            raise ValueError(F"fragment {f} holds {tr:.3f} orbitals of the span {rows} - "
                             F"not an integer within {tol}, the fragments do not localize this manifold")
        if n_f == 0:
            continue
        vals, vecs = np.linalg.eigh(M)
        vecs = vecs[:, np.argsort(-vals)]
        w_f = Q @ vecs[:, :n_f]
        if n_f > 1:
            # eigh returns the fragment's energy eigenvectors by increasing eigenvalue
            _, v_f = np.linalg.eigh(w_f.T @ H @ w_f)
            w_f = w_f @ v_f
        cols.append(w_f)
        frag += [f] * n_f
        Q = Q @ vecs[:, n_f:]
    w = np.concatenate(cols, axis=1)
    pops = np.stack([np.einsum("jc,jk,kc->c", w, B, w) for B in blocks], axis=1)
    e = np.einsum("jc,jk,kc->c", w, H, w)
    return w, frag, pops, e


def sd_diabatization_step(sd_lists, P_frag, E_orb, kind, dia_index=None, active_space=None, data_dim=None,
                          tol=0.25):
    """
    The SD -> diabatic coefficients W of one step, |Phi_A> = sum_k |SD_k> W_kA: the
    reference and the untransformed SDs map onto themselves, every localization manifold
    onto its fragment-localized states (sd_diabatization_groups, localize_fragment_block) -
    within a fragment the localized orbitals diagonalize the orbital energies and the states
    are ranked by increasing energy (the hole in the highest orbital, the electron in the
    lowest orbital first).

    Args:
        sd_lists (list of lists of int): the SDs of the step in their row order
        P_frag (list of arrays): fragment operators in the active-space 2-block layout
            (orthonormal orbitals)
        E_orb (array): the orbital-energy operator in the same layout
        kind (str): "electron", "hole" or "both"
        dia_index (list, optional): the diabatic-state index of an earlier step - the
            columns keep its order, None builds it (reference and singles first, then
            the manifolds by key, fragment-major - hole-major then electron for "both" -
            within a fragment by increasing state energy)
        active_space, data_dim: passed to sd_diabatization_groups (open-shell references)
        tol (float): the integer-occupancy tolerance of localize_fragment_block

    Returns:
        W (array (nsd, nsd)), dia_index (list), pops (array (nsd, nfrag)), pops_hole
        (array (nsd, nfrag)): fragment populations of every diabatic state's localized
        electron (varying) orbital and of its localized hole ("both" only), nan elsewhere,
        energies (array (nsd, 2)): the orbital energies of the same two orbitals, nan elsewhere

    Raises ValueError when the SD content or the per-fragment orbital counts differ from
    the index (the localization must be stable along the trajectory) and when W is not
    orthogonal to 1e-8.
    """
    nsd = len(sd_lists)
    nact = P_frag[0].shape[0] // 2
    nfrag = len(P_frag)
    sets = [frozenset(int(o) for o in sd) for sd in sd_lists]
    gs, groups, singles = sd_diabatization_groups(sd_lists, kind, active_space, data_dim)
    # the energy of a state up to a constant: minus the orbital energy of its hole, plus that of its electron
    E_orb = np.asarray(E_orb, dtype=float)
    loc = {}
    for g in groups:
        if g["kind"] == "both":
            loc_h = localize_fragment_block(P_frag, [_sd_block_index(o, nact) for o in g["holes"]], -E_orb, tol)
            loc_p = localize_fragment_block(P_frag, [_sd_block_index(o, nact) for o in g["parts"]], E_orb, tol)
            loc[g["key"]] = (g, loc_h, loc_p)
        else:
            sign = -1.0 if g["kind"] == "hole" else 1.0
            w, frag, pops, e = localize_fragment_block(P_frag, [_sd_block_index(o, nact) for o in g["orbs"]],
                                                       sign * E_orb, tol)
            loc[g["key"]] = (g, w, frag, pops, sign * e)

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
            if loc[key][0]["kind"] == "both":
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
    energies = np.full((nsd, 2), np.nan)
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
            g, (wh, frag_h, ph, eh), (wp, frag_p, pp, ep) = loc[key]
            X = _cols(frag_h, f_h, r_h, "hole", key)
            Y = _cols(frag_p, f_p, r_p, "electron", key)
            for j, k in enumerate(g["rows"]):
                h, pt = g["pairs"][j]
                W[k, col] = g["tau"][j] * wh[g["holes"].index(h), X] * wp[g["parts"].index(pt), Y]
            pops_out[col] = pp[Y]
            pops_hole[col] = ph[X]
            energies[col] = (ep[Y], -eh[X])
            continue
        _, _, f, r = entry
        g, w, frag, pops, e = loc[key]
        c = _cols(frag, f, r, "", key)
        for j, k in enumerate(g["rows"]):
            W[k, col] = g["tau"][j] * g["sigma"][j] * w[j, c]
        pops_out[col] = pops[c]
        energies[col, 0] = e[c]
    dev = float(np.abs(W.T @ W - np.eye(nsd)).max())
    if not np.isfinite(dev) or dev > 1e-8:
        raise ValueError(F"the diabatic transformation is not orthogonal (|W^T W - 1| = {dev:.2e})")
    return W, dia_index, pops_out, pops_hole, energies


def fragment_operator_count(res_dir, prefix, step):
    """The number of step-2 fragment operators {prefix}{f}_ks_{step}.npz in res_dir - the fragment
    index follows the prefix directly and the indices must be 0..n-1 (the prefixes of the POD family
    share their stem, so a wildcard after the prefix would count the P_pod2l and P_pod2gs files as
    pod fragments)."""
    pattern = re.compile(F"{re.escape(prefix)}(\\d+)_ks_{int(step)}\\.npz")
    found = sorted(int(hit.group(1)) for hit in (pattern.fullmatch(os.path.basename(path))
                   for path in glob.glob(F"{res_dir}/{prefix}*_ks_{int(step)}.npz")) if hit)
    if found != list(range(len(found))):
        raise ValueError(F"the fragment operators {prefix}{{f}}_ks_{step}.npz in {res_dir} carry the fragment "
                         F"indices {found} - 0..n-1 is needed")
    return len(found)


def _sd_diabatization_series(params, sd_sorted, active_space, start_time, finish_time, kind):
    """W, the diabatic index, the populations, the orbital energies and the fragment operators of every
    step start_time..finish_time-1, from the step-2 fragment operators and orbital energies sliced
    to the active space, in the orthonormal-orbital layout (_fragment_operators_of_step) - the
    operators returned for the CI series, which localizes the same steps."""
    prefix, nfrag = _fragment_operator_prefix(params, params['path_to_npz_files'], start_time)
    W_list, pops_list, hole_list, energy_list, operators, dia_index = [], [], [], [], [], None
    for i, step in enumerate(range(start_time, finish_time)):
        P_frag, E_orb = _fragment_operators_of_step(params, prefix, nfrag, step, active_space)
        operators.append(P_frag)
        W, dia_index, pops, pops_hole, energies = sd_diabatization_step(sd_sorted[i], P_frag, E_orb, kind, dia_index,
                                                                        active_space, params['data_dim'],
                                                                        params['occupancy_tolerance'])
        W_list.append(W)
        pops_list.append(pops)
        hole_list.append(pops_hole)
        energy_list.append(energies)
    return W_list, dia_index, pops_list, hole_list, energy_list, operators


def _fragment_operator_prefix(params, res_dir_1, start_time):
    """The step-2 file prefix of the requested fragment operators and their count - at least two fragments."""
    # no prefix may equal another prefix followed by digits - the fragment index follows the prefix directly
    prefixes = {"mulliken": "P_frag", "pod": "P_pod", "pod2l": "P_pod2l", "pod2gs": "P_pod2gs", "pod_cp2k": "P_pod_cp2k"}
    operator = str(params['diabatization_scheme']).lower()
    if operator == "pod2":
        raise ValueError("diabatization_scheme 'pod2' has no projectors - its fragment states are not orthogonal across "
                         "fragments (step2 writes its POD2_hab files only), use 'pod2l' or 'pod2gs'")
    if operator not in prefixes:
        raise ValueError(F"diabatization_scheme {operator!r} - use 'mulliken', 'pod', 'pod2l', 'pod2gs' or 'pod_cp2k'")
    prefix = prefixes[operator]
    nfrag = fragment_operator_count(res_dir_1, prefix, start_time)
    if nfrag < 2:
        raise ValueError(F"diabatization needs the step-2 fragment operators "
                         F"{prefix}{{f}}_ks_{{step}}.npz of at least two fragments in {res_dir_1} "
                         F"({dict(mulliken='fragment_atom_lists', pod='compute_pod', pod2l='compute_pod with pod_schemes', pod2gs='compute_pod with pod_schemes', pod_cp2k='pod_cp2k_operators')[operator]} in "
                         F"step2), found {nfrag}")
    return prefix, nfrag


def _fragment_operators_of_step(params, prefix, nfrag, step, active_space, with_energies=True):
    """The step-2 fragment operators and the orbital-energy operator of one step sliced to the active space (through
    the orbital relabeling of active_space_selection 'identity') in the orthonormal-orbital layout
    sd_diabatization_step and sd_fragment_operators read: expressed in the Lowdin orbitals X = S_ks^-1/2 whenever
    the step's orbital overlap S_ks is not the identity (the SD-level apply_orthonormalization of the time overlaps
    is another transform and no condition here) - the energies None when the caller does not read them."""
    P_frag = [_window_matrix(params, F'{prefix}{f}_ks', step, active_space) for f in range(nfrag)]
    E_orb = _window_matrix(params, 'E_ks', step, active_space) if with_energies else None
    s_ks = _window_matrix(params, 'S_ks', step, active_space)
    if np.any(s_ks != np.eye(s_ks.shape[0])):
        X = orthogonalizations.get_Lowdin_scipy(s_ks)
        P_frag = [X.T @ P @ X for P in P_frag]
        if with_energies:
            E_orb = X.T @ E_orb @ X
    return P_frag, E_orb


def _single_excitations(sd_lists):
    """
    The single excitations of an SD basis: the reference row (the lowest orbital sum) and, for every
    row that differs from it in exactly one occupied and one virtual spin-orbital, (hole, particle, tau) -
    tau the parity between the row's list and the in-place determinant a+_particle a_hole |ref> (the hole's
    slot of the reference list taken by the particle), so that tau_k |SD_k> is that in-place determinant.
    Rows that are the reference or not single excitations are absent from the dict.
    """
    sets = [frozenset(int(o) for o in sd) for sd in sd_lists]
    gs = min(range(len(sets)), key=lambda k: sum(abs(o) for o in sets[k]))
    ref, ref_list = sets[gs], [int(o) for o in sd_lists[gs]]
    singles = {}
    for k, occ in enumerate(sets):
        if k == gs:
            continue
        hole, part = ref - occ, occ - ref
        if len(hole) != 1 or len(part) != 1:
            continue
        h, pt = next(iter(hole)), next(iter(part))
        in_place = list(ref_list)
        in_place[in_place.index(h)] = pt
        singles[k] = (h, pt, slatdet.permutation_parity(in_place, [int(o) for o in sd_lists[k]]))
    return gs, singles


def sd_fragment_operators(sd_lists, P_frag):
    """
    The fragment population operators of the hole and of the excited electron in an SD basis of single
    excitations, as matrices over the SD rows: between a+_a a_i |ref> and a+_b a_j |ref> the hole operator is
    delta_ab <i|P_F|j> and the electron operator delta_ij <a|P_F|b> (one-body operators, Slater-Condon), the
    in-place parities tau of the rows attached. Rows that are the reference or not single excitations carry zeros.

    Args:
        sd_lists (list of lists of int): the SDs of one step in the signed reindexed spin-orbital notation
        P_frag (list of arrays): the fragment operators in the active-space 2-block layout (orthonormal orbitals)

    Returns:
        O_hole, O_elec (lists of arrays (nsd, nsd)): one pair per fragment
    """
    nsd = len(sd_lists)
    nact = P_frag[0].shape[0] // 2
    gs, singles = _single_excitations(sd_lists)
    rows = sorted(singles)
    holes = [_sd_block_index(singles[k][0], nact) for k in rows]
    parts = [_sd_block_index(singles[k][1], nact) for k in rows]
    tau = np.array([singles[k][2] for k in rows], dtype=float)
    same_part = np.equal.outer(parts, parts)
    same_hole = np.equal.outer(holes, holes)
    O_hole, O_elec = [], []
    for P in P_frag:
        B = 0.5 * (np.asarray(P, dtype=float) + np.asarray(P, dtype=float).T)
        block_h = B[np.ix_(holes, holes)] * same_part * np.outer(tau, tau)
        block_e = B[np.ix_(parts, parts)] * same_hole * np.outer(tau, tau)
        Oh, Oe = np.zeros((nsd, nsd)), np.zeros((nsd, nsd))
        Oh[np.ix_(rows, rows)] = block_h
        Oe[np.ix_(rows, rows)] = block_e
        O_hole.append(Oh)
        O_elec.append(Oe)
    return O_hole, O_elec


def ci_diabatization_step(T, O_hole, O_elec, E_ci, kind, dia_index=None, tol=0.25):
    """
    The CI -> diabatic coefficients V of one step, |Lambda_A> = sum_I |Psi_I> V_IA: the ground state (column 0 of
    T) maps onto itself, the excited CI states onto fragment-localized states of their own span - the hole and
    electron fragment operators carried into the CI space through T (O_ci = T_exc^T O_sd T_exc) and localized by
    the sequential projection of localize_fragment_block: "hole" localizes the hole, "electron" the electron,
    "both" the hole first and the electron within every hole-fragment block. Within a fragment (a fragment pair
    for "both") the states diagonalize the CI energies and are ranked by increasing energy.

    Args:
        T (array (nsd, nci + 1)): the SD -> CI coefficients of the step, column 0 the ground state
        O_hole, O_elec (lists of arrays (nsd, nsd)): sd_fragment_operators of the step
        E_ci (array (nci,)): the energies of the excited CI states of the step, in the order of T's columns
        kind (str): "electron", "hole" or "both"
        dia_index (list, optional): the diabatic-state index of an earlier step - the columns keep its order,
            None builds it (the ground state first, then fragment-major - hole-major then electron for "both")
        tol (float): the integer-occupancy tolerance of localize_fragment_block

    Returns:
        V (array (nci + 1, nci + 1)), dia_index (list), pops (array (nci + 1, nfrag)), pops_hole (array
        (nci + 1, nfrag)): fragment populations of every diabatic state's electron and hole (nan where the kind
        does not localize it, and for the ground state), energies (array (nci + 1, 2)): the energy of every
        diabatic state in column 0 (the units of E_ci), nan for the ground state and in column 1

    Raises ValueError when the CI space does not hold whole localization manifolds (a fragment then carries a
    non-integer number of states - include the complete set of excitations of every hole x particle manifold in
    number_of_states), when the fragment counts change along the trajectory, and when V is not orthogonal.
    """
    if kind not in ("electron", "hole", "both"):
        raise ValueError(F"localized_carrier {kind!r} - use 'none', 'hole', 'electron' or 'both' (the former 'particle' "
                         F"is 'electron', the former 'joint' is 'both')")
    T = np.asarray(T)
    if np.iscomplexobj(T):
        # the T matrices of the runner are complex-typed with zero imaginary parts
        T = T.real
    T = np.asarray(T, dtype=float)
    nci = T.shape[1] - 1
    nfrag = len(O_hole)
    Te = T[:, 1:]
    E_ci = np.asarray(E_ci, dtype=float).reshape(-1)
    if E_ci.shape[0] != nci:
        raise ValueError(F"{E_ci.shape[0]} CI energies for {nci} excited CI states")
    H = np.diag(E_ci)
    hint = (" - the CI space must hold whole hole x particle manifolds, include the complete set of their "
            "excitations in number_of_states")
    try:
        if kind in ("hole", "both"):
            Oh = [Te.T @ O @ Te for O in O_hole]
            wh, frag_h, _, _ = localize_fragment_block(Oh, list(range(nci)), H, tol)
        if kind in ("electron", "both"):
            Oe = [Te.T @ O @ Te for O in O_elec]
    except ValueError as err:
        raise ValueError(F"many-body diabatization: {err}{hint}") from err
    cols, labels = [], []
    if kind == "both":
        for f_h in range(nfrag):
            block = [c for c in range(nci) if frag_h[c] == f_h]
            if not block:
                continue
            wb = wh[:, block]
            try:
                wp, frag_p, _, _ = localize_fragment_block([wb.T @ O @ wb for O in Oe], list(range(len(block))),
                                                            wb.T @ H @ wb, tol)
            except ValueError as err:
                raise ValueError(F"many-body diabatization, electrons of the hole-on-fragment-{f_h} states: {err}{hint}") from err
            ranks_h = {}
            for c, f_p in enumerate(frag_p):
                ranks_h[f_p] = ranks_h.get(f_p, -1) + 1
                cols.append(wb @ wp[:, c])
                labels.append(("joint", 1, f_h, 0, f_p, ranks_h[f_p]))
    elif kind == "hole":
        for c, f in enumerate(frag_h):
            cols.append(wh[:, c])
            labels.append(("loc", 0, f, sum(1 for x in frag_h[:c] if x == f)))
    else:
        try:
            wp, frag_p, _, _ = localize_fragment_block(Oe, list(range(nci)), H, tol)
        except ValueError as err:
            raise ValueError(F"many-body diabatization: {err}{hint}") from err
        for c, f in enumerate(frag_p):
            cols.append(wp[:, c])
            labels.append(("loc", 0, f, sum(1 for x in frag_p[:c] if x == f)))
    Vexc = np.stack(cols, axis=1) if cols else np.zeros((nci, 0))
    order = list(range(len(labels)))
    if dia_index is None:
        dia_index = [("gs",)] + labels
    else:
        if len(dia_index) != nci + 1 or dia_index[0] != ("gs",) or sorted(dia_index[1:]) != sorted(labels):
            raise ValueError("the fragment counts of the many-body diabatization changed along the trajectory - "
                             F"{dia_index[1:]} indexed, {labels} at this step")
        order = [labels.index(entry) for entry in dia_index[1:]]
    V = np.zeros((nci + 1, nci + 1))
    V[0, 0] = 1.0
    V[1:, 1:] = Vexc[:, order]
    pops = np.full((nci + 1, nfrag), np.nan)
    pops_hole = np.full((nci + 1, nfrag), np.nan)
    energies = np.full((nci + 1, 2), np.nan)
    for a in range(nci):
        v = V[1:, a + 1]
        if kind in ("electron", "both"):
            pops[a + 1] = [v @ O @ v for O in Oe]
        if kind in ("hole", "both"):
            pops_hole[a + 1] = [v @ O @ v for O in Oh]
        energies[a + 1, 0] = v @ H @ v
    dev = float(np.abs(V.T @ V - np.eye(nci + 1)).max())
    if not np.isfinite(dev) or dev > 1e-8:
        raise ValueError(F"the many-body diabatic transformation is not orthogonal (|V^T V - 1| = {dev:.2e})")
    return V, dia_index, pops, pops_hole, energies


def _ci_diabatization_series(params, sd_sorted, SD2CI, ci_energies, active_space, start_time, finish_time, kind,
                             fragment_operators=None):
    """V, the diabatic index, the populations and the energies of every step start_time..finish_time-1 in the CI
    basis: the step-2 fragment operators as SD-space hole / electron operators (sd_fragment_operators), carried
    through the T matrices and localized (ci_diabatization_step) - ci_energies holds the excitation energies of
    the CI states of every step. fragment_operators: the per-step operators the SD series built (its last
    return) - loaded and orthonormalized here when None."""
    if fragment_operators is None:
        prefix, nfrag = _fragment_operator_prefix(params, params['path_to_npz_files'], start_time)
    V_list, pops_list, hole_list, energy_list, dia_index = [], [], [], [], None
    for i, step in enumerate(range(start_time, finish_time)):
        if fragment_operators is not None:
            P_frag = fragment_operators[i]
        else:
            P_frag, _ = _fragment_operators_of_step(params, prefix, nfrag, step, active_space, with_energies=False)
        O_hole, O_elec = sd_fragment_operators(sd_sorted[i], P_frag)
        V, dia_index, pops, pops_hole, energies = ci_diabatization_step(SD2CI[i], O_hole, O_elec, ci_energies[i],
                                                                        kind, dia_index,
                                                                        tol=params['occupancy_tolerance'])
        V_list.append(V)
        pops_list.append(pops)
        hole_list.append(pops_hole)
        energy_list.append(energies)
    return V_list, dia_index, pops_list, hole_list, energy_list


def _munkres_assignment(weights, reordering):
    """The column assigned to every row of a square weight matrix so that the assigned weights sum to the maximum,
    the Munkres-Kuhn step of the followings - reordering 2 libra_py.hungarian.maximize (the step of the legacy
    SD-state reordering, a pure-Python loop that takes seconds on a window of a hundred orbitals), 22 scipy's
    linear_sum_assignment (the same assignment). The followings assign the states of a group, a block or a window
    to the same positions of the next frame, so the matrix is square."""
    W = np.asarray(weights, dtype=float)
    nrow, ncol = W.shape
    if ncol != nrow:
        raise ValueError(F"{nrow} states against {ncol} candidates - the assignment takes a square weight matrix, the "
                         "states of a group against the same positions of the next frame")
    col_of = np.full(nrow, -1, dtype=int)
    if reordering == 2:
        for row, col in hungarian.maximize(data_conv.nparray2MATRIX(W)):
            col_of[row] = col
    elif reordering == 22:
        rows, cols = linear_sum_assignment(W, maximize=True)
        col_of[rows] = cols
    else:
        raise ValueError(F"reordering = {reordering!r} is not a Munkres-Kuhn assignment - 2 (libra_py.hungarian) and "
                         F"22 (scipy's linear_sum_assignment) are")
    if np.any(col_of < 0) or len(set(col_of.tolist())) != nrow:
        raise RuntimeError("the Munkres assignment left a state without a distinct candidate")
    return col_of


def _tracking_groups(dia_index):
    """The columns of a diabatic index that differ in their ranks alone - one fragment's localized states of a
    manifold, one fragment pair's joint states - as lists of columns, the reference and the untransformed SDs as
    singletons."""
    groups = {}
    for col, entry in enumerate(dia_index):
        if entry[0] == "loc":
            key = ("loc", entry[1], entry[2])
        elif entry[0] == "joint":
            key = ("joint", entry[1], entry[2], entry[4])
        else:
            key = ("single", col)
        groups.setdefault(key, []).append(col)
    return list(groups.values())


def _assign_states(O, reordering, alpha=0.0, energies=None, col_energies=None):
    """The column of every row of a square time-overlap block that carries its state on, by the reordering of the SD
    states: 1 the Sato search of libra's get_reordering, 2 the Munkres-Kuhn assignment on |O|^2 exp(-(alpha dE)^2)
    by libra_py.hungarian, 22 the same assignment by scipy's linear_sum_assignment - dE the energy difference
    between the row's state and the column's position, both at the earlier frame (the cost of make_cost_mat):
    energies are the rows' - the states followed so far, in their labels - and col_energies the columns' - the
    candidate positions in their stored order, the rows' vector again when None, which is right only while no state
    has been relabeled. alpha 0 or no energies put no penalty."""
    O = np.asarray(O, dtype=float)
    if reordering == 1:
        return np.array([int(c) for c in get_reordering(data_conv.nparray2CMATRIX(O))])
    if reordering not in (2, 22):
        raise ValueError(F"reordering = {reordering!r} is not a listed option - 1 (Sato), 2 (Munkres-Kuhn, "
                         F"libra_py.hungarian) and 22 (the same assignment by scipy) are")
    return _munkres_assignment(_assignment_weights(O, energies, col_energies, alpha), reordering)


def _assignment_of(reordering):
    """The assignment step of a diabatic reordering value - 5 and 52 (the alignment) assign as 2 and 22."""
    return {5: 2, 52: 22}.get(reordering, reordering)


def _polar_factor(O):
    """The orthogonal alignment of a group's states to the previous frame: R, the polar factor of the group's
    time-overlap block O = <d(k)|d(k+1)> (O = U S V^T, R = V U^T), is the orthogonal transform of the later frame's
    states that makes O R symmetric positive definite - the nearest the group can come to continuing itself. It
    holds the signs and the rotation within degenerate states, whose orientation is the eigensolver's otherwise."""
    U, _, Vt = np.linalg.svd(np.asarray(O, dtype=float))
    return Vt.T @ U.T


def _rotation_angle(R):
    """The rotation of an orthogonal transform beyond its nearest signed permutation, in degrees (the largest
    principal angle) - of the polar factor of a raw time-overlap block the turn of the eigensolver's orientation
    between two frames, of a cumulative alignment the drift of the aligned states from the energy-ordered ones."""
    P = np.zeros_like(R)
    for i, c in enumerate(_munkres_assignment(np.abs(R), 2)):
        P[i, c] = 1.0 if R[i, c] >= 0.0 else -1.0
    return float(np.degrees(np.abs(np.angle(np.linalg.eigvals(P.T @ R)))).max())


def _state_energies(energy_records):
    """The energy of every diabatic state per frame from the records of the diabatization (the orbital energy of a
    localized state - the same sign for all states of a group - the electron minus the hole orbital energy of a
    joint state, nan for the reference and the untransformed SDs), in the units of the records."""
    out = []
    for e in energy_records:
        e = np.asarray(e, dtype=float)
        out.append(e[:, 0] - np.where(np.isnan(e[:, 1]), 0.0, e[:, 1]))
    return out


def _partner(row, taken):
    """The state that took most of a departing or mixing state: the largest |overlap| of its row outside the column
    that carries it on."""
    r = np.abs(np.asarray(row, dtype=float)).copy()
    r[taken] = -1.0
    return int(np.argmax(r))


def _classify_continuations(d, inside, threshold):
    """The states of a frame whose continuation |<c|c>| = d falls below max(threshold, CONTINUITY_FLOOR), classified
    by what the space carries of them (inside = sqrt(sum_j |<c|j>|^2) over the space's states): EDGES - the space
    carries less than that floor of the state, it left the space through its edge (recorded by every writer, the
    remedy is the space), MIXED - the space carries the state but no single state does, below the threshold (an
    internal mixing within one frame - refused after a reordering while the threshold is above 0), UNRESOLVED -
    carried, below the floor (let through and recorded). Three index arrays in the states' order - the one rule of
    the SD, CI, diabatic,
    orbital and POD continuations, whose edge records share the shape [frame, state, kept, carried by the space,
    the state that took most of it (_partner)]."""
    d = np.asarray(d, dtype=float)
    inside = np.asarray(inside, dtype=float)
    floor = max(float(threshold), CONTINUITY_FLOOR)
    below = np.where(d < floor)[0]
    edges = np.array([c for c in below if inside[c] < floor], dtype=int)
    carried = [c for c in below if inside[c] >= floor]
    mixed = np.array([c for c in carried if d[c] < threshold], dtype=int)
    unresolved = np.array([c for c in carried if d[c] < CONTINUITY_FLOOR], dtype=int)
    return edges, mixed, unresolved


def _follow_diabatic_states(W_list, St_list, start_time, basis="sd", dia_index=None, reordering=0, alpha=0.0,
                            energies=None, min_overlap=CONTINUITY_FLOOR):
    """The localization eigenvectors carry arbitrary signs: every diabatic state follows the trajectory through the
    time overlaps of its own basis, O = W(i)^T St(i) W(i+1), and the sign of every column of W(i+1) is set by the
    sign of diag(O) (in place). reordering 0: a column whose |O_cc| falls below min_overlap does not continue itself -
    two states of one fragment, or the two fragments' states at a symmetric geometry, swap between the frames - and
    the run is refused: from that frame on the labels would mean other states (min_overlap 0 switches the check
    off). reordering 1 or 2: within every group of states that differ in their ranks alone (one fragment's states of
    a manifold, one fragment pair's joint states) the columns of W(i+1) are assigned to the states of W(i) as the SD
    states are reordered (1 Sato, 2 Munkres-Kuhn on |O|^2 with the energy penalty exp(-(alpha dE)^2), energies the
    per-frame state energies of the columns) and permuted into the order of W(i) (in place), so that a state keeps
    the identity it had at start_time through a crossing and the frame's energy order no longer names it - the
    penalty compares the followed state's energy with the candidate position's, both at the earlier frame. A state
    whose continuation carries less than min_overlap even so is one of two things: an EDGE DEPARTURE when the whole
    space carries less than min_overlap of it, sqrt(sum_j |O_cj|^2) < min_overlap - the state left the active space
    through its edge, the reordering has nothing to assign and the remedy is the space, so the frame is recorded
    (edges) and the run goes on - or an INTERNAL MIXING when the space carries it but no single state does (the
    states mixed by more than the threshold's angle within one frame, 1/sqrt(2) = 45 degrees), which is refused: the
    frame spacing does not resolve that crossing. A threshold below the floor lets internal mixings through and records
    those below it. reordering 5 or 52: the assignment and the signs of 2 or 22 follow the energy-ordered states as
    before (the swap frames keep their meaning), and the aligned states are those followed states combined by a
    cumulative orthogonal transform A(i) of every group - A(i+1) the polar factor of the aligned states' overlap with
    the followed states of frame i+1 (_polar_factor), so that the aligned time-overlap block is symmetric positive
    definite: the signs and the rotation within degenerate states, whose orientation the eigensolver sets anew at
    every frame, are taken out, the within-group derivative coupling of a diabatic propagation vanishes, and the
    continuity is the diagonal of the aligned block. W(i) becomes W(i) A(i) in place at the end and the records of
    the states are rotated as diagonal quantities (_rotate_records).

    Returns:
        perms (list of arrays): perms[i][c] is the column, in the energy order of frame i, that carries the state
            c of start_time (the identity for start_time itself) - the rows of the per-frame records follow it
        swaps (list of int): the frames at which the assignment differs from the previous frame's
        unresolved (list of int): the frames let through with an internal continuation below the floor
        min_continuity (float): the smallest |O_cc| of the run
        edges (list of lists): the edge departures as [frame, state, |O_cc|, the weight kept in the space]
        align (dict): "rotations" - the cumulative orthogonal transform A of every frame (the identity without
            the alignment), "angles" - the turn of the eigensolver's orientation within a group between the frames
            of every interval in degrees (the rotation of the raw block's polar factor beyond its signed
            permutation, empty without the alignment), "drifts" - the rotation of A of every frame from the
            energy-ordered states, "groups" - the tracking groups
    """
    n = W_list[0].shape[1]
    groups = _tracking_groups(dia_index) if (reordering and dia_index is not None) else [[c] for c in range(n)]
    perms = [np.arange(n)]
    swaps, unresolved, edges = [], [], []
    rots, angles, drifts = [np.eye(n)], [], [0.0]
    min_continuity = 1.0
    for i in range(len(St_list)):
        O = W_list[i].T @ np.asarray(St_list[i].todense()).real @ W_list[i + 1]
        perm = np.arange(n)
        if reordering:
            # the frame's energies stand in its energy order (positions) - the rows are the followed states, whose
            # positions at this frame are perms[i]
            e = None if energies is None else np.asarray(energies[i], dtype=float)
            for g in groups:
                if len(g) > 1:
                    perm[g] = np.array(g)[_assign_states(O[np.ix_(g, g)], _assignment_of(reordering), alpha,
                                                         None if e is None else e[perms[i]][g],
                                                         None if e is None else e[g])]
            if np.any(perm != np.arange(n)):
                W_list[i + 1] = W_list[i + 1][:, perm]
                O = O[:, perm]
        # the sign of every followed state - an exactly zero overlap keeps the sign (np.sign would drop the state
        # from every later frame)
        sgn = np.where(np.diag(O) < 0.0, -1.0, 1.0)
        W_list[i + 1] = W_list[i + 1] * sgn
        O = O * sgn[None, :]
        rot = np.eye(n)
        if reordering in (5, 52):
            # the aligned states of frame i against the followed states of frame i + 1, the polar factor of every
            # group the transform of frame i + 1 - the angles of the raw blocks and of the transform recorded
            O_al = rots[i].T @ O
            angle, drift = 0.0, 0.0
            for g in groups:
                if len(g) > 1:
                    R = _polar_factor(O_al[np.ix_(g, g)])
                    rot[np.ix_(g, g)] = R
                    angle = max(angle, _rotation_angle(_polar_factor(O[np.ix_(g, g)])))
                    drift = max(drift, _rotation_angle(R))
            O = O_al @ rot
            angles.append(angle)
            drifts.append(drift)
        d = np.abs(np.diag(O))
        min_continuity = min(min_continuity, float(d.min()))
        inside = np.sqrt(np.sum(O ** 2, axis=1))
        edge_idx, mixed_idx, unresolved_idx = _classify_continuations(d, inside, min_overlap)
        for c in edge_idx:
            edges.append([start_time + i + 1, int(c), round(float(d[c]), 4), round(float(inside[c]), 4),
                          _partner(O[c], c)])
        for c in mixed_idx[:1]:
            row = np.abs(O[c]).copy()
            row[c] = -1.0
            partner = int(np.argmax(row))
            if reordering:
                raise ValueError(F"the diabatic {basis.upper()} state {c} of frame {start_time + i} carries "
                                 F"|<{c}|{c}>| = {d[c]:.3f} into frame {start_time + i + 1} even after the reordering"
                                 F"{' and the alignment' if reordering in (5, 52) else ''} "
                                 F"within its group (the largest other overlap |<{c}|{partner}>| = {row[partner]:.3f}, "
                                 F"the space carries {inside[c]:.3f} of the state, the threshold "
                                 F"{min_overlap:.4g}) - the "
                                 F"states mix by more than the threshold's angle within one frame: the frame spacing "
                                 F"does not resolve the crossing (a lower diabatic_state_reordering_min_overlap lets "
                                 F"the frame through and records it)")
            raise ValueError(F"the diabatic {basis.upper()} state {c} of frame {start_time + i} continues into "
                             F"state {partner} of frame {start_time + i + 1} (|<{c}|{c}>| = {d[c]:.3f} below the "
                             F"threshold {min_overlap:.4g}, |<{c}|{partner}>| = {row[partner]:.3f}) - two localized "
                             F"states swap between the frames and the labels would mean other states from there "
                             F"on: narrow the manifold or the active space, follow the states with "
                             F"do_diabatic_state_reordering 2 (within a tracking group - one fragment's states of a "
                             F"manifold), or let the window follow the orbital with active_space_selection "
                             F"'identity' (across groups, or a group of one state)")
        if len(unresolved_idx):
            unresolved.append(start_time + i + 1)
        perms.append(perm)
        rots.append(rot)
        if np.any(perm != perms[i]):
            swaps.append(start_time + i + 1)
    if reordering in (5, 52):
        for k in range(len(W_list)):
            W_list[k] = W_list[k] @ rots[k]
    return perms, swaps, unresolved, min_continuity, edges, {"rotations": rots, "angles": angles, "drifts": drifts,
                                                              "groups": groups}


def _permute_records(records, perms):
    """The per-frame record rows (one row per diabatic state in the frame's energy order) in the order of the
    followed states."""
    return [np.asarray(r)[p] for r, p in zip(records, perms)]


def _rotate_records(records, align):
    """The per-frame record rows of states aligned by the orthogonal transforms of align["rotations"] (block-diagonal
    over align["groups"]): a diagonal quantity q of a group's states becomes diag(R^T diag(q) R) = (R * R)^T q,
    column by column - the records keep no within-group off-diagonal elements, so a rotated energy is the diagonal
    of the rotated energy matrix alone."""
    out = []
    for r, R in zip(records, align["rotations"]):
        r = np.array(r, dtype=float)
        for g in align["groups"]:
            if len(g) > 1:
                r[g] = (R[np.ix_(g, g)] ** 2).T @ r[g]
        out.append(r)
    return out


def _alignment_record(reordering, angles, frames, drifts):
    """The "alignment" entry of a labels file or a report under the alignment (None under the other reorderings):
    the turn of the eigensolver's orientation within a group between the frames of every interval (frames[i] ->
    frames[i + 1]) - its largest and mean value and the later frames of the intervals above 30 degrees - and the
    largest drift of the aligned states from the energy-ordered ones over the frames."""
    if reordering not in (5, 52):
        return None
    angles = [float(a) for a in angles]
    return {"max_angle_deg": round(max(angles), 3) if angles else 0.0,
            "mean_angle_deg": round(float(np.mean(angles)), 3) if angles else 0.0,
            "frames_above_30_deg": [int(frames[i + 1]) for i, a in enumerate(angles) if a > 30.0],
            "max_drift_deg": round(max(float(x) for x in drifts), 3) if drifts else 0.0}


def _tracking_record(reordering, alpha, min_overlap, perms, swaps, unresolved, min_continuity, edges, start_time,
                     align=None):
    """The entries of the labels file that say how the states were followed."""
    off = int(sum(np.any(p != np.arange(len(p))) for p in perms))
    angles = align["angles"] if align is not None else []
    drifts = align["drifts"] if align is not None else []
    return {"do_diabatic_state_reordering": int(reordering), "diabatic_state_reordering_alpha": float(alpha),
            "diabatic_state_reordering_min_overlap": float(min_overlap),
            "swap_frames": [int(k) for k in swaps], "frames_off_energy_order": off,
            "unresolved_frames": [int(k) for k in unresolved], "min_continuity": float(min_continuity),
            "edge_departures": edges,
            "alignment": _alignment_record(reordering, angles, list(range(start_time, start_time + len(perms))), drifts),
            "reordering_note": F"the ranks are the energy order at the starting frame {start_time}" + (
                " and the states keep that identity through the crossings of swap_frames (a followed state may "
                "lie above its lower-ranked partner from a swap frame on)" if reordering else
                " and every state continues itself between the frames") + (
                " - the states of a group are aligned to the previous frame by the orthogonal transform of their "
                "largest overlap (the signs and the rotation within degenerate states), alignment holds the turn of "
                "the eigensolver's orientation between the frames and the drift of the aligned states from the "
                "energy-ordered ones" if reordering in (5, 52) else "") + " - an internal mixing below the threshold "
                "is refused, unresolved_frames lists the frames a lower threshold let through with less than it, "
                "edge_departures the states that left the active space ([frame, state, kept, carried by the space, "
                "the state that took most of it])"}


def _continuity_record(St_list, start_time, reordering, alpha, min_overlap, perms=None, basis="SD"):
    """How the states of a basis continue between the frames as labeled. A state whose |St_cc| falls below
    min_overlap is an EDGE DEPARTURE when the whole space carries less than min_overlap of it (sqrt(sum_j |St_cj|^2)
    - it left the active space through its edge, the remedy is the space) and is recorded, or an INTERNAL MIXING
    when the space carries it but no single state does - refused after a reordering (1 or 2: the states mixed by
    more than the threshold's angle within one frame, the frame spacing does not resolve the crossing, min_overlap
    0 switches the check off), never refused in the energy order (reordering 0: a crossing of adiabatic states is the
    convention there and is left to the propagation). Returns the record: the frames let through with an internal
    continuation below the floor (unresolved), the edge departures ([frame, state, kept, carried by the space, the
    state that took most of it]), the
    smallest |St_cc|, and the frames at which the reordering changed the labels (perms: the per-frame label
    permutations of apply_state_reordering_general)."""
    conts, edges, unresolved = [], [], []
    for i, St in enumerate(St_list):
        A = np.abs(np.asarray(St.todense()).real)
        c = np.diag(A)
        inside = np.sqrt(np.sum(A ** 2, axis=1))
        conts.append(c)
        edge_idx, mixed_idx, unresolved_idx = _classify_continuations(c, inside, min_overlap)
        for j in edge_idx:
            edges.append([start_time + i + 1, int(j), round(float(c[j]), 4), round(float(inside[j]), 4),
                          _partner(A[j], j)])
        for j in mixed_idx[:1]:
            if reordering and min_overlap > 0.0:
                row = A[j].copy()
                row[j] = -1.0
                partner = int(np.argmax(row))
                raise ValueError(F"the {basis} state {j} of frame {start_time + i} continues into state {partner} "
                                 F"of frame {start_time + i + 1} after the reordering (|<{j}|{j}>| = {c[j]:.3f} below "
                                 F"the threshold {min_overlap:.4g}, |<{j}|{partner}>| = {row[partner]:.3f}, the space "
                                 F"carries {inside[j]:.3f} of the state) - an internal mixing the frame spacing does "
                                 F"not resolve: a lower state_reordering_min_overlap lets the frame through and "
                                 F"records it, 0 switches the check off")
        if len(unresolved_idx):
            unresolved.append(start_time + i + 1)
    reordered = []
    if perms is not None:
        for i in range(1, min(len(perms), len(St_list) + 1)):
            if list(perms[i]) != list(perms[i - 1]):
                reordered.append(start_time + i)
    return {"do_state_reordering": int(reordering), "state_reordering_alpha": float(alpha),
            "state_reordering_min_overlap": float(min_overlap),
            "unresolved_frames": unresolved, "edge_departures": edges,
            "min_continuity": float(min(c.min() for c in conts)) if conts else 1.0, "reordered_frames": reordered,
            "note": "unresolved_frames: the frames in which a labeled state continued with less than the floor "
                    "(1/sqrt(2)) inside the "
                    "space and was let through - in the energy order the trivial crossings left to the propagation, "
                    "with a reordering the frames a lowered threshold accepted - edge_departures: [frame, state, kept, "
                    "carried by the space, the state that took most of it] of the states that left the active space "
                    "through its edge - "
                    "reordered_frames: the frames at which the reordering changed the labels"}


def _window_matrix(params, name, frame, active_space, frame_cols=None):
    """The step-2 window matrix {name}_{frame}.npz of path_to_npz_files, dense and real, cut to the active space -
    through the label permutations of active_space_selection 'identity' (params['orbital_perms'], {frame: perm}) when
    they are set: the rows take the frame's followed orbitals, the columns those of frame_cols (the ket frame of a
    time overlap) or of the frame itself."""
    perms = params.get('orbital_perms')
    rows = cols = list(active_space)
    if perms is not None:
        rows = [perms[frame][i] for i in active_space]
        cols = [perms[frame if frame_cols is None else frame_cols][i] for i in active_space]
    M = sp.load_npz(F"{params['path_to_npz_files']}/{name}_{frame}.npz").todense()
    return np.array(M[rows, :][:, cols]).real


def _follow_orbitals(params, start_time, finish_time):
    """The Kohn-Sham orbitals of the stored window followed by identity through St_ks from start_time to
    finish_time - 1, every spin block by itself (active_space_selection 'identity'): within an interval the orbitals
    of the later frame are assigned to those of the earlier one as the SD states are reordered -
    active_space_selection_algo 1 the Sato search of get_reordering, 2 the Munkres-Kuhn assignment on |St|^2
    exp(-(alpha dE)^2) with the orbital energies of the earlier frame (the cost of make_cost_mat,
    active_space_selection_alpha the width) by libra_py.hungarian, 22 the same assignment by scipy. Returns
    {frame: perm}, the label permutation of every frame - position r
    of the frame's window matrices holds the orbital that stands at position perm[r] of the stored window, the
    identity at start_time - and the record of the window: the frames at which the labels changed, the count of
    frames off the energy order, the smallest continuity, the orbitals that left the window (edge_departures:
    [frame, label, kept, carried by the window, the window position that took most of it] - the window carries less
    than the floor of the orbital) and the
    unresolved crossings ([frame, label]: an orbital continuing with less than the floor while the window carries it).
    Nothing is refused here: the SD states built from the followed orbitals are checked as labeled
    (state_reordering_min_overlap), _selection_record reads the record against the active space."""
    res_dir, n = params['path_to_npz_files'], params['data_dim'] // 2
    reordering, alpha = params['active_space_selection_algo'], params['active_space_selection_alpha']
    blocks = [np.arange(n), np.arange(n, 2 * n)]
    # a closed shell stores the alpha block in both spin blocks - one assignment serves both
    assigned = blocks if int(params.get('isUKS', 0)) else blocks[:1]
    perm = np.arange(2 * n)
    perms = {start_time: perm.tolist()}
    reordered, unresolved, edges = [], [], []
    min_continuity = 1.0
    for k in range(start_time, finish_time - 1):
        St = np.abs(np.asarray(sp.load_npz(F'{res_dir}/St_ks_{k}.npz').todense()).real)
        # the orbital energies enter the penalty alone
        e_pos = np.diag(np.asarray(sp.load_npz(F'{res_dir}/E_ks_{k}.npz').todense()).real) if alpha else None
        e = None if e_pos is None else e_pos[perm]
        new = np.empty_like(perm)
        for b in assigned:
            new[b] = b[_assign_states(St[np.ix_(perm[b], b)], reordering, alpha, None if e is None else e[b],
                                      None if e_pos is None else e_pos[b])]
        if len(assigned) == 1:
            new[blocks[1]] = new[blocks[0]] + n
        d = St[perm, new]
        inside = np.sqrt(np.sum(St[perm] ** 2, axis=1))
        min_continuity = min(min_continuity, float(d.min()))
        edge_idx, _, unresolved_idx = _classify_continuations(d, inside, CONTINUITY_FLOOR)
        for r in edge_idx:
            edges.append([k + 1, int(r), round(float(d[r]), 4), round(float(inside[r]), 4),
                          _partner(St[perm[r]], new[r])])
        for r in unresolved_idx:
            unresolved.append([k + 1, int(r)])
        if np.any(new != perm):
            reordered.append(k + 1)
        perm = new
        perms[k + 1] = perm.tolist()
    identity = list(range(2 * n))
    window = {"reordered_frames": reordered, "frames_off_energy_order": sum(p != identity for p in perms.values()),
              "min_continuity": float(min_continuity), "edge_departures": edges, "unresolved": unresolved}
    return perms, window


def _selection_record(params, perms, window, active_space):
    """The "active_space" block of sd_route_report.json under active_space_selection 'identity': the record of the
    window (_follow_orbitals) read against the active space - the frames at which an orbital of the active space
    exchanged its position with one outside it (active_space_edge_frames), the window positions of the active
    orbitals in every frame off their energy positions (tracked_orbitals), the unresolved crossings of the window
    and those of the active orbitals."""
    active = np.asarray(active_space)
    active_set = set(active.tolist())
    frames = sorted(perms)
    tracked, edge_frames = {}, []
    for prev, k in zip(frames, frames[1:]):
        p, q = np.asarray(perms[prev]), np.asarray(perms[k])
        if sorted(q[active].tolist()) != sorted(p[active].tolist()):
            edge_frames.append(k)
        if np.any(q[active] != active):
            tracked[k] = q[active].tolist()
    unresolved = sorted({k for k, _ in window["unresolved"]})
    unresolved_active = sorted({k for k, r in window["unresolved"] if r in active_set})
    return {"active_space_selection": "identity",
            "active_space_selection_algo": int(params['active_space_selection_algo']),
            "active_space_selection_alpha": float(params['active_space_selection_alpha']),
            "reordered_frames": window["reordered_frames"],
            "frames_off_energy_order": window["frames_off_energy_order"],
            "active_space_edge_frames": edge_frames, "tracked_orbitals": tracked,
            "min_continuity": window["min_continuity"], "unresolved_frames": unresolved,
            "active_space_unresolved_frames": unresolved_active, "edge_departures": window["edge_departures"],
            "note": "the window orbitals followed by identity, the SD basis of every frame built from the orbitals "
                    "that stood in the active space at the starting frame - reordered_frames: the frames at which "
                    "the window labels changed - active_space_edge_frames: the frames at which an orbital of the "
                    "active space exchanged its position with one outside it - tracked_orbitals: the window "
                    "positions (0-based in the stored window, alpha then beta) of the active-space orbitals in the "
                    "frames they stand off their energy positions - unresolved_frames: an orbital continuing with "
                    "less than the floor (1/sqrt(2)) while the window carries it "
                    "(active_space_unresolved_frames: one of the active "
                    "space) - edge_departures: [frame, orbital, kept, carried by the window, the position that took "
                    "most of it] of the orbitals that "
                    "left the window through its edge - nothing is refused here, the SD states built from the "
                    "followed orbitals are checked as labeled"}


def _relabel_configurations(ci_basis_states, spin_components, perms, start_time, lowest_orbital):
    """The TD-DFT configurations of every frame with their orbitals named by the labels of active_space_selection
    'identity' instead of the positions the electronic structure code prints: an orbital index o of frame k
    (1-based, the stored window starting at lowest_orbital) stands at window position o - lowest_orbital, which the
    frame's permutation assigns to the orbital of one label - that label, 1-based again, names the orbital in the
    configuration. Both spins share the spatial orbitals (a closed-shell reference). Returns the relabeled
    configurations, [frame][state][k] = [occupied, virtual], and the union basis of the labeled configurations with
    their spins in the order of first appearance (as get_step2_mb_sp_properties builds it over the positions)."""
    n = len(perms[start_time]) // 2
    relabeled, unique, seen = [], [], set()
    for step, states in enumerate(ci_basis_states):
        label_of_position = np.argsort(np.asarray(perms[start_time + step][:n]))
        frame = []
        for i, configurations in enumerate(states):
            confs = []
            for conf in configurations:
                positions = [int(o) - lowest_orbital for o in conf]
                if min(positions) < 0 or max(positions) >= n:
                    raise ValueError(F"the TD-DFT configuration {list(conf)} of frame {start_time + step} names an "
                                     F"orbital outside the stored window [{lowest_orbital}, {lowest_orbital + n - 1}] "
                                     F"- widen lowest_orbital / highest_orbital in step 2")
                confs.append([int(label_of_position[p]) + lowest_orbital for p in positions])
            frame.append(confs)
        step3_many_body.collect_unique_sd_states(frame, spin_components[step], unique, seen)
        relabeled.append(frame)
    return relabeled, unique


def _write_diabatization_records(res_dir, kind, dia_index, pops_list, hole_list, energy_list, start_time,
                                 operator="mulliken", basis="sd", extra=None):
    """U_{basis}_labels.json (the meaning of every diabatic state, with the mean energies over the
    steps: the orbital energies of the localized orbitals of an SD-level state, the excitation
    energy of a many-body one, in eV) and U_{basis}_pops.txt (per step, the fragment populations
    of every diabatic state's localized electron orbital, then of its localized hole - nan where
    the state has none) - basis "sd" for the SD-level states, "ci" for the many-body ones (whose
    untransformed state is the ground state). A localized state without an SD manifold (the POD
    route) carries manifold null. extra: further top-level entries of the labels file."""
    E = np.stack([np.asarray(e, dtype=float) for e in energy_list])
    known = ~np.isnan(E)
    e_mean = np.where(known, E, 0.0).sum(axis=0) / np.maximum(known.sum(axis=0), 1)
    e_mean = np.where(known.any(axis=0), e_mean, np.nan)
    if basis == "sd":
        e_mean = e_mean * units.au2ev
    states = []
    for col, entry in enumerate(dia_index):
        if entry[0] == "gs":
            states.append({"index": col, "type": "gs"})
        elif entry[0] == "sd":
            states.append({"index": col, "type": "sd", "sd": sorted(entry[1])})
        elif entry[0] == "joint":
            states.append({"index": col, "type": "joint", "channel": int(entry[1]),
                           "hole_fragment": int(entry[2]), "hole_rank": int(entry[3]),
                           "electron_fragment": int(entry[4]), "electron_rank": int(entry[5])})
            if basis == "sd":
                states[-1].update({"hole_orbital_energy_eV": round(float(e_mean[col, 1]), 4),
                                   "electron_orbital_energy_eV": round(float(e_mean[col, 0]), 4)})
            else:
                states[-1]["energy_eV"] = round(float(e_mean[col, 0]), 4)
        else:
            states.append({"index": col, "type": "localized",
                           "manifold": None if entry[1] is None else int(entry[1]),
                           "fragment": int(entry[2]), "rank": int(entry[3]),
                           "orbital_energy_eV" if basis == "sd" else "energy_eV": round(float(e_mean[col, 0]), 4)})
    with open(F'{res_dir}/U_{basis}_labels.json', 'w') as f:
        json.dump({"kind": kind, "diabatization_scheme": operator, "basis": basis,
                   "note": "electron: the excited electron of the manifold (shared hole = "
                           "manifold) sits on 'fragment', hole: the hole (shared particle = "
                           "manifold) sits on 'fragment', both: the hole sits on "
                           "'hole_fragment' and the electron on 'electron_fragment' (channel "
                           "1 = alpha, -1 = beta excitations), ranks order the states of one "
                           "fragment by increasing energy - the hole in the highest orbital, the "
                           "electron in the lowest orbital first - the energies are means over "
                           "the steps in eV (the orbital energies of the localized orbitals of an "
                           "SD-level state, the excitation energy of a many-body state)",
                   "states": states, **(extra or {})}, f, indent=1)
    nfrag = pops_list[0].shape[1]
    nst = pops_list[0].shape[0]
    rows = [np.concatenate(([start_time + i], pops_list[i].reshape(-1), hole_list[i].reshape(-1)))
            for i in range(len(pops_list))]
    header = ("step " + " ".join(F"p_e(state{c},frag{f})" for c in range(nst) for f in range(nfrag))
              + " " + " ".join(F"p_h(state{c},frag{f})" for c in range(nst) for f in range(nfrag)))
    np.savetxt(F'{res_dir}/U_{basis}_pops.txt', np.array(rows), header=header, fmt="%.6g")


def _save_u(res_dir, W, cum_phase, step, basis="sd"):
    """U_{basis} = (W F)^T: rows = diabatic states, columns = the saved (phase-corrected) SDs or CI states."""
    Wf = np.asarray(W, dtype=float)
    if cum_phase is not None:
        Wf = Wf * np.asarray(cum_phase, dtype=float).reshape(-1, 1)
    sp.save_npz(F'{res_dir}/U_{basis}_{step}.npz', sp.csc_matrix(Wf.T))


def _save_u_sd(res_dir, W, cum_phase, step):
    """U_sd = (W F)^T: rows = diabatic states, columns = the saved (phase-corrected) SDs."""
    _save_u(res_dir, W, cum_phase, step, "sd")


def _pod_route_frame(hab, spin=0):
    """h_POD of one frame in the block-state basis, block order: the block energies on the diagonal, the printed
    couplings V_b{i}_b{j} (i < j) and their transposes between the blocks. Returns (h, nmo), nmo the states of every
    block."""
    nblk = len([key for key in hab.files if key.startswith("E_b") and key.endswith(F"_s{spin}")])
    nmo = [len(hab[F"E_b{ib}_s{spin}"]) for ib in range(nblk)]
    off = np.concatenate(([0], np.cumsum(nmo)))
    h = np.zeros((off[-1], off[-1]))
    for ib in range(nblk):
        e = np.asarray(hab[F"E_b{ib}_s{spin}"], dtype=float)
        # the ranks of the route are the energy order of every block - the print must be ascending
        if np.any(np.diff(e) < 0.0):
            raise ValueError(F"the POD block {ib} prints its block energies out of ascending order (E_b{ib}_s{spin}: "
                             F"{e.tolist()}) - the ranks of the route are the energy order of every block")
        h[off[ib]:off[ib + 1], off[ib]:off[ib + 1]] = np.diag(e)
        for jb in range(ib + 1, nblk):
            v = np.asarray(hab[F"V_b{ib}_b{jb}_s{spin}"], dtype=float)
            h[off[ib]:off[ib + 1], off[jb]:off[jb + 1]] = v
            h[off[jb]:off[jb + 1], off[ib]:off[ib + 1]] = v.T
    return h, nmo


def _pod_route_hole_basis(nmo, occupied, per_block):
    """The hole basis over the occupied block states: of every block the highest per_block[f] of its occupied[f]
    occupied states (the block states are printed by increasing energy), block-major, the block HOMO first.
    Returns the state indices in block order, the block and the rank of every basis state."""
    off = np.concatenate(([0], np.cumsum(nmo)))
    idx, frag, rank = [], [], []
    for f in range(len(nmo)):
        for r in range(per_block[f]):
            idx.append(int(off[f] + occupied[f] - 1 - r))
            frag.append(f)
            rank.append(r)
    return idx, frag, rank


def _cofactor_matrix(A):
    """C_ij = (-1)^(i+j) det(A without row i and column j) - for A the time overlaps of the orbitals of an occupied
    set, the exact time overlaps of the one-hole determinants in the phase convention (-1)^i of run_step3_pod. From
    the singular value decomposition A = U diag(s) V^T: C = det(U) det(V) U diag(prod of the other s) V^T - the
    form holds for a singular A as well, where det(A) A^-T does not."""
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    if n == 1:
        return np.ones((1, 1))
    U, s, Vt = np.linalg.svd(A)
    others = np.array([np.prod(np.delete(s, i)) for i in range(n)])
    return np.linalg.det(U) * np.linalg.det(Vt) * (U * others[None, :]) @ Vt


def run_step3_pod(params):
    """
    The POD-Hamiltonian route of step 3: the hole dynamics of a cation in the basis of the occupied POD block
    states of the fragments themselves, with the Hamiltonian CP2K's POD prints - the block energies and the
    block-block couplings - and the nonadiabatic couplings from the time overlaps of the block states between
    the frames (step2 read_pod_cp2k_files: POD_cp2k_hab_{step}.npz and St_pod_cp2k_{step}.npz). No orbital of
    the dimer is localized: the diabatic states are the block states, complete within every fragment block.

    The diabatic basis: of every block f the highest num_occ_states_per_block[f] of its pod_occupied[f] occupied
    block states, block-major, the block HOMO first (rank 0). A basis state is the one-hole determinant of the
    cation with the hole in that block state, all other occupied block states of the basis filled (the block
    states outside the basis are frozen, as the orbitals outside the active space of the SD route). With the phase
    (-1)^i on the determinant of the i-th basis state (i in the basis order) the one-body Hamiltonian h between the
    determinants is H_hole = tr(h_act) 1 - h_act, h_act the POD Hamiltonian over the basis states, and their time
    overlap is det(St_act) times the cofactor matrix of St_act, St_act the time overlaps of the basis states -
    equal to St_act itself to first order in the frame spacing. The signs of the printed block states are followed
    along the trajectory by the sign of <s(k)|s(k+1)> - a basis state whose |<s(k)|s(k+1)>| falls below the floor
    1/sqrt(2) does not continue itself (two block states of one fragment swap between the frames) and the run is
    refused, as in the SD route.

    Written into path_to_save_sd_Hvibs - the file set of run_step3_sd_nacs_libint, so that the step-4 runner and the
    plots read it unchanged, the 'sd' states here being the ADIABATIC hole states, the eigenstates of H_hole of every
    frame by increasing energy (identity-followed under do_state_reordering 2):
        Hvib_sd_{k}_re.npz  the adiabatic energies at the midpoint of the frames k and k+1 (as the SD route),
        Hvib_sd_{k}_im.npz  Im(Hvib) = -d of the interval [k, k+1], (St^T - St) / (2 dt) of St_sd_{k},
        St_sd_{k}_re.npz    the time overlaps of the adiabatic states, V(k)^T St_hole(k) V(k+1),
        U_sd_{k}.npz        rows the diabatic (block) states, columns the adiabatic ones - V(k), U H_sd U^T = H_hole,
        U_sd_labels.json    what every diabatic state is (fragment = block, rank, mean block energy in eV,
                            manifold null), with "route": "pod_hamiltonian" and the basis counts,
        U_sd_pops.txt       the block populations of the states (1 on their own block),
        hvib_convention.json, pod_route_report.json (the continuity minimum, the first-order deviation of the
                            determinant overlaps from the block-state overlaps, the reordered frames, the block gaps).

    Args:
        params (dictionary):

            * **params['path_to_npz_files']** (string): the step-2 directory with POD_cp2k_hab_{step}.npz of every
                frame start_time..finish_time-1 and St_pod_cp2k_{step}.npz of start_time..finish_time-2
            * **params['logfile_directory']** (string): the step-2 CP2K logs - the HOMO index of the neutral system
                must equal the sum of pod_occupied
            * **params['path_to_save_sd_Hvibs']** (string): the output directory [default: cwd/res-pod]
            * **params['start_time']**, **params['finish_time']** (int): the frames read, finish_time one past the
                last - Hvib and St are written for start_time..finish_time-2, U for start_time..finish_time-1
            * **params['time_step']** (float): the frame spacing in fs [default: 1.0]
            * **params['pod_occupied']** (list of int): the occupied block states of every block - the valence
                electron pairs of every fragment - in the block order of the POD section
            * **params['num_occ_states_per_block']** (int or list of int): the basis states of every block, from the
                block HOMO down [default: None = all occupied block states]
            * **params['sd_charge']** (int): 1, the hole basis of the cation - the only value implemented
            * **params['isUKS']** (int): 0 - the only value implemented
            * **params['localized_carrier']** (string): 'hole' - the only value implemented
            * **params['diabatization_scheme']** (string): 'pod_cp2k' - the only value implemented
            * **params['apply_phase_correction']** (bool): follow the signs of the adiabatic states through
                <a(k)|a(k+1)> [default: True]. The sign of every eigenvector column is arbitrary frame by frame, so
                False leaves those signs in St_sd and Hvib_sd_im - an element with one flipped state turns the
                antisymmetric coupling formula into the symmetric part - and prints a warning. The energies do not
                depend on the signs.
            * **params['do_state_reordering']** (int): 0 [default] the adiabatic states by increasing energy at every
                frame, 1 the Sato search (get_reordering), 2 the Munkres-Kuhn assignment of the adiabatic states of
                frame k+1 to those of frame k by the largest squared time overlap with the energy penalty of
                state_reordering_alpha (identity following through the avoided crossings, libra_py.hungarian), 22
                the same assignment by scipy's linear_sum_assignment
            * **params['state_reordering_alpha']** (float): the energy penalty of the assignment,
                exp(-(alpha dE)^2) [default: 0]
            * **params['do_diabatic_state_reordering']** (int): the reordering of the block states between the
                frames, with the vocabulary of do_state_reordering. 0 [default]: none - every block state of the
                basis must continue itself (|<s(k)|s(k+1)>| >= 1/sqrt(2)), a swap of two block states is refused, the
                ranks are the energy order of every frame. 1: Sato (get_reordering), 2: Munkres-Kuhn
                (libra_py.hungarian), 22: the same assignment by scipy's linear_sum_assignment - ALL block states of
                a block are reordered against the block's states of the next frame and every basis state follows
                the block state assigned to it, so a state keeps the block state that carries it through a crossing
                of the block's levels, at the window edge too - the labels record swap_frames and
                frames_off_energy_order, the report the followed block states of the frames off the starting order,
                and the ranks name the energy order at start_time only. 5: the reordering of 2 and then the
                orthogonal alignment of every block's basis states to the previous frame (52: with the assignment of
                22) - the polar factor of the block's time-overlap block combines the followed block states so that
                the block is symmetric positive definite: the signs and the rotation within degenerate block states,
                whose orientation the eigensolver sets anew at every frame, are taken out, the couplings and the
                time overlaps of the basis are transformed along (the block energies become the diagonal of the
                rotated block Hamiltonian, the adiabatic files do not change, U_sd does), and the labels and the
                report carry the rotation angles (alignment). A continuation
                below diabatic_state_reordering_min_overlap is refused when it is an internal mixing and recorded
                when the state left the basis of its block (edge_departures - without a reordering only, the
                reordering follows a state anywhere in its block).
            * **params['diabatic_state_reordering_alpha']** (float): the energy-penalty width of the Munkres-Kuhn
                cost of the block states, exp(-(alpha dE)^2) on the block energies of the earlier frame, in
                1/Hartree [default: 0 - no penalty].
            * **params['diabatic_state_reordering_min_overlap']** (float): the continuity threshold of the block
                states [default: 1/sqrt(2) = 0.707 - more than half of a state's weight on its own continuation, a
                mixing of 45 degrees within one frame, 0 switches the check off]: below it an internal mixing refuses
                the run, with or without a reordering, and a departure from the basis of the block (the basis carries
                less than the threshold of the state - without a reordering) is recorded in edge_departures - a
                threshold below 1/sqrt(2) lets internal mixings through (0.5 admits 60 degrees) and the labels and
                the report list them (unresolved_frames, min_continuity).
            * **params['state_reordering_min_overlap']** (float): the same threshold for the adiabatic states as
                labeled by do_state_reordering 1 or 2 - refused below it after the reordering, 0 switches the check
                off - recorded under "adiabatic" in the report (the frames let through below the floor, min_continuity,
                reordered_frames - the frames at which the labels changed - and frames_off_energy_order) [default:
                1/sqrt(2) = 0.707, more than half of a state's weight on its own continuation]. In the energy order
                (0) nothing is refused.
            * **params['active_space_selection']** (string): 'energy' alone - the POD basis holds no Kohn-Sham
                orbitals to follow, its block states are followed by do_diabatic_state_reordering.
            * **params['nac_algo']** (int): 0 alone - the route writes the HST couplings (St^T - St) / 2 dt.
            Every key the route does not read is refused: the former option names with their present names, the
            keys of the SD route's Kohn-Sham basis (num_occ_states and its kin) with a pointer to
            num_occ_states_per_block, any other unknown key with the list of the route's keys.
            * **params['apply_orthonormalization']**: the vocabulary of the SD route is accepted and has no effect -
                the block states are orthonormal within every frame by construction (S^-1/2 v of the Lowdin frame),
                their Lowdin factors are the identity

    Returns:
        dict: the report written to pod_route_report.json
    """
    critical_params = ['path_to_npz_files', 'logfile_directory', 'start_time', 'finish_time', 'pod_occupied']
    default_params = {'path_to_save_sd_Hvibs': os.getcwd() + '/res-pod', 'time_step': 1.0,
                      'num_occ_states_per_block': None, 'sd_charge': 1, 'isUKS': 0, 'localized_carrier': 'hole',
                      'diabatization_scheme': 'pod_cp2k', 'apply_phase_correction': True,
                      'apply_orthonormalization': 'lowdin_cached', 'do_state_reordering': 0,
                      'state_reordering_alpha': 0.0, 'do_diabatic_state_reordering': 0,
                      'diabatic_state_reordering_alpha': 0.0,
                      'diabatic_state_reordering_min_overlap': CONTINUITY_FLOOR,
                      'state_reordering_min_overlap': CONTINUITY_FLOOR, 'active_space_selection': 'energy',
                      'nac_algo': 0}
    _refuse_former_names(params)
    # the keys of the SD route's Kohn-Sham basis have no reading here - num_occ_states in place of
    # num_occ_states_per_block ran the whole occupied space without a word
    for key in ('num_occ_states', 'num_unocc_states', 'lowest_orbital', 'highest_orbital', 'num_occ_alpha',
                'num_unocc_alpha', 'num_occ_beta', 'num_unocc_beta', 'active_space_num_occ_orbitals', 'is_many_body',
                'number_of_states', 'ci_threshold', 'sorting_type', 'spin_adapted', 'sd_overlap_kernel'):
        if key in params:
            raise ValueError(F"run_step3_pod: params['{key}'] is a key of the SD route - the POD route has no "
                             F"Kohn-Sham basis, num_occ_states_per_block selects the hole basis of every block from "
                             F"its pod_occupied states")
    unknown = sorted(set(params) - set(default_params) - set(critical_params))
    if unknown:
        raise ValueError(F"run_step3_pod: params {unknown} are not options of the POD route - its keys are "
                         F"{sorted(set(default_params) | set(critical_params))}")
    refuse_unknown_keys(params, STEP3_KEYS, "step3.run_step3_pod", "params", "no function of the step-3 workflow reads")
    comn.check_input(params, default_params, critical_params)
    # the values the route implements - every other value is refused
    for key, allowed, hint in (('sd_charge', (1,), "the frozen-orbital hole basis of the cation"),
                               ('isUKS', (0,), "closed-shell block states"),
                               ('localized_carrier', ('hole',), "the hole of the cation"),
                               ('diabatization_scheme', ('pod_cp2k',), "the POD prints of CP2K"),
                               ('active_space_selection', ('energy',), "no Kohn-Sham orbitals in the POD basis - the "
                                                                       "block states are followed by "
                                                                       "do_diabatic_state_reordering"),
                               ('nac_algo', (0,), "the POD route writes the HST couplings (St^T - St) / 2 dt, the NPI "
                                                  "of nac_algo 1 is not implemented on it")):
        if params[key] not in allowed:
            raise ValueError(F"run_step3_pod: params['{key}'] = {params[key]!r} is not implemented on the POD "
                             F"route - {list(allowed)} ({hint})")
    _orthonormalization_mode(params['apply_orthonormalization'])
    _check_reordering_options(params, 'run_step3_pod')
    dia_reordering, dia_alpha = params['do_diabatic_state_reordering'], params['diabatic_state_reordering_alpha']
    dia_min_overlap, sd_min_overlap = (params['diabatic_state_reordering_min_overlap'],
                                       params['state_reordering_min_overlap'])
    res_dir_1, res_dir_2 = params['path_to_npz_files'], params['path_to_save_sd_Hvibs']
    start_time, finish_time = int(params['start_time']), int(params['finish_time'])
    if finish_time - start_time < 2:
        raise ValueError("run_step3_pod needs at least two frames (finish_time - start_time >= 2)")
    dt = params['time_step'] * units.fs2au
    os.makedirs(res_dir_2, exist_ok=True)
    marker_file = os.path.join(res_dir_2, 'hvib_convention.json')
    if not os.path.exists(marker_file) and glob.glob(os.path.join(res_dir_2, '*_im.npz')):
        raise ValueError(F"{res_dir_2} already holds *_im.npz files without a convention marker (legacy +d data) - "
                         F"writing canonical -d files into it would mix conventions in one directory - use a fresh "
                         F"output path or move the legacy files away")

    frames = list(range(start_time, finish_time))
    # the block-state energies of every frame - the full h of a frame is sliced to the tracked states later
    h_diag, nmo_list = [], []
    for k in frames:
        h, nmo = _pod_route_frame(np.load(F'{res_dir_1}/POD_cp2k_hab_{k}.npz'))
        h_diag.append(np.diag(h).copy())
        nmo_list.append(nmo)
    if any(nmo != nmo_list[0] for nmo in nmo_list):
        raise ValueError("the numbers of POD block states change along the trajectory")
    nmo = nmo_list[0]
    nblk = len(nmo)
    occupied = [int(o) for o in params['pod_occupied']]
    if len(occupied) != nblk:
        raise ValueError(F"pod_occupied lists {len(occupied)} blocks, the POD prints carry {nblk}")
    per_block = params['num_occ_states_per_block']
    if per_block is None:
        per_block = list(occupied)
    elif np.isscalar(per_block):
        per_block = [int(per_block)] * nblk
    else:
        per_block = [int(o) for o in per_block]
    if len(per_block) != nblk:
        raise ValueError(F"num_occ_states_per_block lists {len(per_block)} blocks, the POD prints carry {nblk}")
    for f in range(nblk):
        if not 0 <= per_block[f] <= occupied[f] <= nmo[f]:
            raise ValueError(F"block {f}: 0 <= num_occ_states_per_block {per_block[f]} <= pod_occupied {occupied[f]} "
                             F"<= {nmo[f]} block states is needed")
    if sum(per_block) < 2:
        raise ValueError("the hole basis needs at least two block states")
    if not params['apply_phase_correction']:
        print("WARNING: apply_phase_correction False - the eigenvector signs of every frame are arbitrary, St_sd and "
              "Hvib_sd_im carry them and the couplings are not those of followed states")
    homo = CP2K_methods.read_homo_index(sorted(glob.glob(params['logfile_directory'] + '/*.log'))[0])
    if sum(occupied) != homo:
        raise ValueError(F"pod_occupied {occupied} sums to {sum(occupied)} occupied block states, the neutral system "
                         F"of the step-2 log has {homo} occupied orbitals")
    off = np.concatenate(([0], np.cumsum(nmo)))
    block_gap = {}
    for f in range(nblk):
        if occupied[f] < nmo[f]:
            a, b = off[f] + occupied[f] - 1, off[f] + occupied[f]
            gaps = [hd[b] - hd[a] for hd in h_diag]
            if min(gaps) <= 0.0:
                raise ValueError(F"block {f}: the block state {occupied[f]} (first virtual) lies below the state "
                                 F"{occupied[f] - 1} (last occupied) at frame {frames[int(np.argmin(gaps))]} - "
                                 F"pod_occupied does not sit at a gap of the block")
            block_gap[str(f)] = float(min(gaps) * units.au2ev)

    idx, frag, rank = _pod_route_hole_basis(nmo, occupied, per_block)
    n = len(idx)
    # the block state that carries every diabatic state at every frame (the energy-ordered basis of start_time at
    # the start), the printed sign of every block state followed along the trajectory - A[k] the cumulative
    # orthogonal transform of the followed basis, the signs alone or, aligned, the rotation within every block -
    # one frame's time overlaps in memory at a time, the slice over the tracked states kept
    track = [list(idx)]
    A = [np.eye(n)]
    angles, drifts = [], [0.0]
    swaps, unresolved, edges = [], [], []
    St_act = []
    min_continuity = 1.0
    for i, k in enumerate(frames[:-1]):
        St_k = np.asarray(sp.load_npz(F'{res_dir_1}/St_pod_cp2k_{k}.npz').todense()).real
        ntot = St_k.shape[0] // 2
        if ntot != off[-1]:
            raise ValueError(F"St_pod_cp2k_{k}.npz carries {ntot} block states per spin, the POD prints {off[-1]}")
        St = St_k[:ntot, :ntot]
        cur = track[i]
        nxt = list(cur)
        if dia_reordering:
            # all states of a block are reordered against the block's states of the next frame, as the SD states
            # are (Sato or Munkres-Kuhn, the penalty on the block energies of the earlier frame) - every basis state
            # follows the block state assigned to the one that carried it
            for f in range(nblk):
                cols = list(range(off[f], off[f + 1]))
                assigned = _assign_states(St[np.ix_(cols, cols)], _assignment_of(dia_reordering), dia_alpha,
                                          h_diag[i][cols])
                for c in range(n):
                    if frag[c] == f:
                        nxt[c] = cols[assigned[cur[c] - off[f]]]
        # the overlap of the aligned basis of frame i with the followed block states of frame i + 1, then the
        # transform of the next frame's basis: the signs of the diagonal (an exactly zero overlap keeps the sign -
        # np.sign would drop the state from every later frame), aligned the polar factor within every block
        O = A[i].T @ St[np.ix_(cur, nxt)]
        A_next = np.diag(np.where(np.diag(O) < 0.0, -1.0, 1.0))
        if dia_reordering in (5, 52):
            angle, drift = 0.0, 0.0
            for f in range(nblk):
                cols_f = [c for c in range(n) if frag[c] == f]
                if len(cols_f) > 1:
                    R = _polar_factor(O[np.ix_(cols_f, cols_f)])
                    A_next[np.ix_(cols_f, cols_f)] = R
                    raw = St[np.ix_([cur[c] for c in cols_f], [nxt[c] for c in cols_f])]
                    angle = max(angle, _rotation_angle(_polar_factor(raw)))
                    drift = max(drift, _rotation_angle(R))
            angles.append(angle)
            drifts.append(drift)
        O = O @ A_next
        rows = A[i].T @ St[cur, :]
        d = np.abs(np.diag(O))
        min_continuity = min(min_continuity, float(d.min()))
        # the space of a basis state: the basis of its block without a reordering (a partner outside it is an edge
        # departure, recorded), the whole block with one (a low continuation the block carries is an internal mixing)
        space = [list(range(off[frag[c]], off[frag[c] + 1])) if dia_reordering
                 else [cur[b] for b in range(n) if frag[b] == frag[c]] for c in range(n)]
        inside = np.array([float(np.sqrt(np.sum(rows[c, space[c]] ** 2))) for c in range(n)])
        edge_idx, mixed_idx, unresolved_idx = _classify_continuations(d, inside, dia_min_overlap)
        for c in edge_idx:
            edges.append([int(frames[i + 1]), int(c), round(float(d[c]), 4), round(float(inside[c]), 4),
                          _partner(rows[c], nxt[c])])
        for c in mixed_idx[:1]:
            row = np.abs(rows[c]).copy()
            row[nxt[c]] = -1.0
            partner = int(np.argmax(row))
            if dia_reordering:
                raise ValueError(F"the POD block state {cur[c]} (block {frag[c]}, rank {rank[c]} of frame "
                                 F"{start_time}) carries |<s|s'>| = {d[c]:.3f} into frame {frames[i + 1]} even after "
                                 F"the reordering{' and the alignment' if dia_reordering in (5, 52) else ''} of its "
                                 F"block (the largest other overlap {row[partner]:.3f} into the "
                                 F"block state {partner}, the threshold "
                                 F"{dia_min_overlap:.4g}) - the states mix by more "
                                 F"than the threshold's angle within one frame: the frame spacing does not resolve the "
                                 F"crossing (a lower diabatic_state_reordering_min_overlap lets the frame through and "
                                 F"records it)")
            # the largest other overlap may sit on a basis state of the block (a swap inside the basis) or on a block
            # state outside it while the basis still carries the state - named as what it is
            if partner in cur:
                b = cur.index(partner)
                where = F"the basis state {b} (block {frag[b]}, rank {rank[b]})"
            else:
                where = F"the block state {partner} outside the basis"
            raise ValueError(F"the POD block state {cur[c]} (block {frag[c]}, rank {rank[c]}) of frame {frames[i]} "
                             F"continues into {where} of frame {frames[i + 1]} (|<s|s>| = {d[c]:.3f} below the "
                             F"threshold {dia_min_overlap:.4g}, |<s|s'>| = {row[partner]:.3f}, the basis carries "
                             F"{inside[c]:.3f} of the state) - two block states swap between the frames and the labels "
                             F"would mean other states from there on: narrow num_occ_states_per_block, or follow the "
                             F"states with do_diabatic_state_reordering 2")
        # every state let through below the floor that the basis still carries is an unresolved continuation
        if len(unresolved_idx):
            unresolved.append(frames[i + 1])
        if nxt != cur:
            swaps.append(frames[i + 1])
        track.append(nxt)
        A.append(A_next)
        St_act.append(A[i].T @ St[np.ix_(cur, nxt)] @ A[i + 1])
    h_act = []
    for k, frame in enumerate(frames):
        h, _ = _pod_route_frame(np.load(F'{res_dir_1}/POD_cp2k_hab_{frame}.npz'))
        h_act.append(A[k].T @ h[np.ix_(track[k], track[k])] @ A[k])
    alignment = _alignment_record(dia_reordering, angles, frames, drifts)
    # the one-hole determinants in the phase convention (-1)^i
    H_hole = [np.trace(h) * np.eye(n) - h for h in h_act]
    St_hole = [np.linalg.det(St) * _cofactor_matrix(St) for St in St_act]

    E_adi, V = [], []
    for k in range(len(frames)):
        e, v = np.linalg.eigh(H_hole[k])
        E_adi.append(e)
        V.append(v)
    # the eigenvalues in the frame's own order - the candidate positions' energies of the assignment, while E_adi
    # follows the labels
    E_pos = [e.copy() for e in E_adi]
    reordered, adi_unresolved, adi_edges = [], [], []
    adi_off_order = 0
    adi_min_continuity = 1.0
    perm_prev = np.arange(n)
    for k in range(1, len(frames)):
        O = V[k - 1].T @ St_hole[k - 1] @ V[k]
        if params['do_state_reordering'] in (1, 2, 22):
            # the assignment maps the followed labels to the frame's energy order, so it is the frame's label
            # permutation itself - a frame off the energy order counts, a change of the labels is a reordering
            perm = _assign_states(O, params['do_state_reordering'], params['state_reordering_alpha'], E_adi[k - 1],
                                  E_pos[k - 1])
            if np.any(perm != np.arange(n)):
                adi_off_order += 1
            if np.any(perm != perm_prev):
                reordered.append(frames[k])
            perm_prev = perm
            V[k], E_adi[k], O = V[k][:, perm], E_adi[k][perm], O[:, perm]
        d_adi = np.abs(np.diag(O))
        adi_min_continuity = min(adi_min_continuity, float(d_adi.min()))
        # an adiabatic state the basis no longer carries inherited an edge departure of the basis (recorded), one the
        # basis carries but no single state does mixed internally (refused after a reordering)
        inside_adi = np.sqrt(np.sum(O ** 2, axis=1))
        edge_idx, mixed_idx, unresolved_idx = _classify_continuations(d_adi, inside_adi, sd_min_overlap)
        for j in edge_idx:
            adi_edges.append([int(frames[k]), int(j), round(float(d_adi[j]), 4), round(float(inside_adi[j]), 4),
                              _partner(O[j], j)])
        for j in mixed_idx[:1]:
            if params['do_state_reordering'] in (1, 2, 22) and sd_min_overlap > 0.0:
                raise ValueError(F"the adiabatic hole state {j} of frame {frames[k - 1]} carries |<{j}|{j}>| = "
                                 F"{d_adi[j]:.3f} into frame {frames[k]} after the reordering, below the threshold "
                                 F"{sd_min_overlap:.4g} (the basis carries {inside_adi[j]:.3f} of the state) - an "
                                 F"internal "
                                 F"mixing the frame spacing does not resolve: a lower state_reordering_min_overlap "
                                 F"lets the frame through and records it, 0 switches the check off")
        if len(unresolved_idx):
            adi_unresolved.append(frames[k])
        if params['apply_phase_correction']:
            V[k] = V[k] * np.where(np.diag(O) < 0.0, -1.0, 1.0)

    for k in range(len(frames) - 1):
        St_adi = V[k].T @ St_hole[k] @ V[k + 1]
        sp.save_npz(F'{res_dir_2}/St_sd_{frames[k]}_re.npz', sp.csc_matrix(St_adi))
        sp.save_npz(F'{res_dir_2}/Hvib_sd_{frames[k]}_re.npz',
                    sp.csc_matrix(np.diag(0.5 * (E_adi[k] + E_adi[k + 1]))))
        sp.save_npz(F'{res_dir_2}/Hvib_sd_{frames[k]}_im.npz', sp.csc_matrix((St_adi.T - St_adi) / (2.0 * dt)))
    for k in range(len(frames)):
        sp.save_npz(F'{res_dir_2}/U_sd_{frames[k]}.npz', sp.csc_matrix(V[k]))
    with open(marker_file, 'w') as f:
        json.dump({'im_file_contains': 'Im(Hvib) = -d', 'isUKS': 0, 'sd_charge': 1, 'route': 'pod_hamiltonian'}, f)
    dia_index = [("loc", None, frag[c], rank[c]) for c in range(n)]
    pops_list = [np.eye(nblk)[frag] for _ in frames]
    hole_list = [np.full((n, nblk), np.nan) for _ in frames]
    energy_list = [np.column_stack((np.diag(h), np.full(n, np.nan))) for h in h_act]
    # the followed block states of the frames off the energy-ordered basis of start_time
    tracked = {str(frames[k]): [int(c) for c in track[k]] for k in range(len(frames)) if track[k] != track[0]}
    off_order = len(tracked)
    _write_diabatization_records(res_dir_2, 'hole', dia_index, pops_list, hole_list, energy_list, start_time,
                                 operator='pod_cp2k', basis='sd',
                                 extra={"route": "pod_hamiltonian", "pod_occupied": occupied,
                                        "num_occ_states_per_block": per_block,
                                        "route_note": "the diabatic states are the occupied POD block states "
                                                      "themselves (the one-hole determinants of the cation), the "
                                                      "'sd' states their adiabatic combinations - the eigenstates of "
                                                      "the POD hole Hamiltonian",
                                        "do_diabatic_state_reordering": int(dia_reordering),
                                        "diabatic_state_reordering_alpha": float(dia_alpha),
                                        "diabatic_state_reordering_min_overlap": float(dia_min_overlap),
                                        "swap_frames": [int(k) for k in swaps], "frames_off_energy_order": off_order,
                                        "unresolved_frames": [int(k) for k in unresolved],
                                        "min_continuity": float(min_continuity), "edge_departures": edges,
                                        "alignment": alignment,
                                        "reordering_note": F"the ranks are the energy order of the block states at the "
                                                           F"starting frame {start_time}" + (
                                            " and every state keeps the block state that carries it through the "
                                            "crossings of swap_frames (pod_route_report.json lists the followed block "
                                            "states of the frames off that order)" if dia_reordering else
                                            " and every state continues itself between the frames") + (
                                            " - the basis states of a block are aligned to the previous frame by the "
                                            "orthogonal transform of their largest overlap (the signs and the rotation "
                                            "within degenerate block states), alignment holds the turn of the "
                                            "eigensolver's orientation between the frames and the drift of the aligned "
                                            "states from the energy-ordered ones" if dia_reordering in (5, 52) else "") + " - an "
                                            "internal mixing below the threshold is refused, unresolved_frames lists "
                                            "the frames a lower threshold let through with less than the floor, "
                                            "edge_departures the states that left the basis of their block ([frame, "
                                            "state, kept, carried by the basis, the block state taken])"})
    report = {"frames": [start_time, finish_time], "n_states": n, "blocks": nblk, "block_states": nmo,
              "pod_occupied": occupied, "num_occ_states_per_block": per_block,
              "min_continuity": float(min(np.abs(np.diag(St)).min() for St in St_act)),
              "max_hole_minus_block_overlap": float(max(np.abs(St_hole[k] - St_act[k]).max()
                                                        for k in range(len(St_act)))),
              "max_nonunitarity": float(max(np.abs(St.T @ St - np.eye(n)).max() for St in St_hole)),
              "reordered_frames": reordered, "block_gap_eV": block_gap,
              "do_diabatic_state_reordering": int(dia_reordering), "diabatic_state_reordering_alpha": float(dia_alpha),
              "diabatic_state_reordering_min_overlap": float(dia_min_overlap),
              "swap_frames": [int(k) for k in swaps], "frames_off_energy_order": off_order,
              "unresolved_frames": [int(k) for k in unresolved], "edge_departures": edges, "alignment": alignment,
              "adiabatic": {"do_state_reordering": int(params['do_state_reordering']),
                            "state_reordering_alpha": float(params['state_reordering_alpha']),
                            "state_reordering_min_overlap": float(sd_min_overlap),
                            "unresolved_frames": [int(k) for k in adi_unresolved], "edge_departures": adi_edges,
                            "min_continuity": float(adi_min_continuity), "reordered_frames": reordered,
                            "frames_off_energy_order": adi_off_order},
              "tracked_block_states": tracked}
    with open(F'{res_dir_2}/pod_route_report.json', 'w') as f:
        json.dump(report, f, indent=1)
    print(F"run_step3_pod: {n} hole states of {nblk} blocks ({per_block} of {occupied} occupied) over "
          F"{len(frames)} frames - continuity min {report['min_continuity']:.4f}, |St_hole - St_block| max "
          F"{report['max_hole_minus_block_overlap']:.2e}, |St^T St - 1| max {report['max_nonunitarity']:.2e}, "
          F"{len(reordered)} reordered frames of the adiabatic states ({len(adi_unresolved)} let through below the "
          F"floor), {len(swaps)} swap frames of the block states ({len(unresolved)} let through below the floor, "
          F"{len(edges)} edge "
          F"departures), block gaps {block_gap} eV")
    return report


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
                "lowdin_cached" [default] (identical to "lowdin" - the ket factor is carried to
                the next step in a per-process cache - one fractional matrix power per
                step serially, hits inside contiguous Pool chunks too), "none"/False.
                The default CHANGED from the historical False (raw overlaps) to
                "lowdin_cached" - the Lowdin route is the validated convention -
                set "none" explicitly to reproduce pre-change data
            * **params['sd_charge']** (int): the charge of the SD basis. 0 [default] keeps the SCF reference and its single
                excitations. +1 replaces the SD basis by the frozen-orbital (Koopmans) hole basis of
                the cation: one beta electron removed from each of the num_occ_states active occupied
                orbitals of the closed-shell reference (num_occ_states determinants, energies from the
                neutral orbital energies, no self-consistent screening of the site detuning). -1 is
                the electron basis of the anion (one alpha electron added to each active virtual).
                Needs isUKS 0. Pairs with localized_carrier "hole" / "electron".
            * **params['diabatization_scheme']** (string): the fragment operators the diabatization
                localizes with - "mulliken" [default] reads P_frag{f}_ks_{step}.npz (step2
                fragment_atom_lists), "pod" reads P_pod{f}_ks_{step}.npz (step2 compute_pod, libra's own
                projection-operator diabatization), "pod2l" and "pod2gs" read P_pod2l{f} / P_pod2gs{f}_ks_{step}.npz
                (step2 pod_schemes - the second-generation schemes, fragment-first orthogonalization with the
                fragment states orthonormalized by Lowdin or by Gram-Schmidt - the plain "pod2" of step2 has no
                projectors, its states are not orthogonal across fragments), "pod_cp2k" reads
                P_pod_cp2k{f}_ks_{step}.npz (step2 read_pod_cp2k_files, the blocks of CP2K's POD).
            * **params['localized_carrier']** (string): "none" [default] keeps the SD basis. "electron" or
                "hole" localizes, in every single-excitation manifold sharing the hole (resp. the
                particle), the varying spin-orbital on the fragments of the step-2 fragment operators
                (P_frag{f}_ks_{step}.npz from step2's fragment_atom_lists). "both" localizes the holes
                among the occupied orbitals AND the electrons among the virtuals of every spin
                channel's complete hole x particle manifold - the diabatic states are the excitations
                from a localized hole into a localized electron (Frenkel and charge-transfer states of
                a fragment pair). All three write the
                SD -> diabatic transformation U_sd_{step}.npz (rows: diabatic states, columns: the
                saved SD states, in the gauge of Hvib_sd/St_sd - sorted, reordered, phase-corrected)
                with U_sd_labels.json (what every diabatic state is: fragment and rank - within a
                fragment the localized orbitals diagonalize the orbital energies and the rank counts
                the states by increasing energy, the hole in the highest orbital or the electron in
                the lowest first - with the mean orbital energies of the steps) and U_sd_pops.txt
                (fragment populations of the localized orbitals per step). A runner hands U_sd_{step} to
                obj.basis_transform. With is_many_body the same fragment operators are carried into
                the CI space through the T matrices (hole and electron operators of the SD basis,
                sd_fragment_operators, ci_diabatization_step) and U_ci_{step}.npz (rows: the ground
                state and the diabatic many-body states, columns: the saved CI states in the gauge of
                Hvib_ci/St_ci) is written with U_ci_labels.json and U_ci_pops.txt - the CI space must
                hold whole hole x particle manifolds (number_of_states closes every manifold).
            * **params['occupancy_tolerance']** (float): 0.25 [default] - a fragment's trace of a localization
                manifold's projector may differ from its integer by at most this much (every fragment, the
                last against the rest of the span), else the manifold is refused as not localized
            * **params['do_diabatic_state_reordering']** (int): the reordering of the diabatic states of one
                fragment between the frames through O = W(k)^T St(k) W(k+1), with the vocabulary of
                do_state_reordering. 0 [default]: none - every state must continue itself (|O_cc| >= 1/sqrt(2)), a
                frame pair in which two states of a fragment swap is refused, their ranks are the energy order of every
                frame. 1: Sato (get_reordering), 2: Munkres-Kuhn (libra_py.hungarian), 22: the same assignment by
                scipy's linear_sum_assignment - within every fragment (and manifold, and fragment pair of the joint
                states) the states of frame k+1 are assigned to those of frame k, so a state keeps the identity it
                had at start_time through a crossing of its fragment's levels - U_sd_labels.json records swap_frames
                (the frames at which the assignment changed) and frames_off_energy_order, and the ranks name the
                energy order at start_time only. 5: the assignment of 2 and then the orthogonal alignment of every
                fragment's states to the previous frame (52: with the assignment of 22) - the polar factor of the
                fragment's block of O combines the states of frame k+1 so that the block is symmetric positive
                definite: the signs and the rotation within degenerate states, whose orientation the eigensolver
                sets anew at every frame, are taken out, U_sd rotates within the fragment while the SD-level files
                do not change, the within-fragment derivative coupling of a diabatic propagation vanishes, the
                records' energies and populations are those of the aligned states as diagonal quantities, and
                U_sd_labels.json carries the rotation angles (alignment). A state whose
                continuation carries less than diabatic_state_reordering_min_overlap even after the reordering is
                refused as well when the space still carries it (an internal mixing the frame spacing does not
                resolve) and recorded when it left the active space through its edge (edge_departures). Applies to
                the CI-level diabatic states (U_ci) as well.
            * **params['diabatic_state_reordering_alpha']** (float): the energy-penalty width of the Munkres-Kuhn
                cost of the diabatic states, exp(-(alpha dE)^2) with dE the difference between the followed state's
                energy and the candidate position's at the earlier frame, in 1/Hartree [default: 0 - no penalty] - as
                state_reordering_alpha for the SD states.
            * **params['diabatic_state_reordering_min_overlap']** (float): the continuity threshold of the diabatic
                states, |O_cc| below which a state does not continue itself between two frames [default: 1/sqrt(2) =
                0.707 - more than half of a state's weight on its own continuation, a mixing of 45 degrees within one
                frame, 0 switches the check off]. Below it, a state the space still
                carries (sqrt(sum_j |O_cj|^2) >= the threshold) mixed internally and the run is refused, with or
                without a reordering - a state the space no longer carries left the active space through its edge
                and is recorded (edge_departures in U_sd_labels.json: [frame, state, kept, carried by the space, the
                state that took most of it]),
                the run goes on. A threshold below 1/sqrt(2) lets internal mixings through (0.5 admits 60 degrees):
                unresolved_frames lists them beside min_continuity.
            * **params['state_reordering_min_overlap']** (float): the same threshold for the SD (and CI) states as
                labeled by do_state_reordering 1 or 2: an internal mixing below it after the reordering refuses the
                run (the frame spacing does not resolve the crossing, 0 switches the check off), an edge departure
                is recorded [default: 1/sqrt(2) = 0.707, more than half of a state's weight on its own continuation].
                In the energy order (do_state_reordering 0) nothing is refused - a crossing of adiabatic states is the
                convention there and is left to the propagation.
                sd_route_report.json records unresolved_frames (internal continuations below the floor let through),
                edge_departures, min_continuity and reordered_frames.
            * **params['active_space_selection']** (string): how the orbitals of the active space are picked at every
                frame. 'energy' [default]: the energy positions of the frame (an orbital that crosses the edge of the
                active space leaves the basis, the SD state it carried is recorded as an edge departure). 'identity':
                the orbitals that stood in the active space at start_time, followed through the whole stored window
                by their time overlaps - the Kohn-Sham orbitals of the window are followed by identity between the
                frames, per interval and spin block, and the label permutation of every frame relabels E_ks, S_ks,
                St_ks and the fragment operators before the active space is cut out of them, so the SD basis of every
                frame is built from the same orbitals wherever their energies took them, as the POD basis of
                run_step3_pod is followed within its block. Nothing is refused for the orbitals themselves - the SD
                states built from them are checked as labeled (state_reordering_min_overlap) - sd_route_report.json
                records under "active_space" the reordered frames of the window, the frames at which the active space
                crossed its edge (active_space_edge_frames), the window positions of the active orbitals in every
                frame off their energy positions (tracked_orbitals, 0-based in the stored window, alpha then beta),
                min_continuity, unresolved_frames (an orbital continuing with less than the floor inside the window,
                active_space_unresolved_frames those of the active space) and edge_departures (an orbital that left
                the window). With is_many_body the TD-DFT configurations of every frame are renamed from the orbital
                positions the log files print to the labels before the union basis and the T matrices are built, so
                the CI states are expanded in the followed determinants.
            * **params['active_space_selection_algo']** (int): the assignment that follows the orbitals under
                'identity', with the values of do_state_reordering: 2 [default] Munkres-Kuhn (libra_py.hungarian on
                |St_ks|^2 exp(-(alpha dE)^2), the cost of the KS route's reordering), 22 the same assignment by
                scipy's linear_sum_assignment, 1 Sato (get_reordering).
            * **params['active_space_selection_alpha']** (float): the energy-penalty width of that Munkres-Kuhn cost,
                in 1/Hartree [default: 0 - no penalty] - as state_reordering_alpha.
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
                    - 22: the Hungarian assignment by scipy's linear_sum_assignment - the same result as 2, faster
            * **params['state_reordering_alpha']** (float): a value for state-reordering
            * **params['nac_algo']** ( int ): selection of a method to compute NACs
                    - 0 : Hamess-Shiffer-Tully (HST), finite difference [ default ]
                    - 1 : Meek-Levine NPI
            * **params['blas_num_threads']** (integer): BLAS threads per worker. Default (None):
                    1 per worker with use_multiprocessing, else nprocs. Set explicitly for
                    hybrid parallelism: nprocs workers x blas_num_threads threads each.
            * **params['sd_overlap_kernel']** (string): kernel for the SD overlap matrices
                    - "auto" : batched LAPACK determinants with the determinant-lemma shortcut
                               for low-degree excitations (singles, doubles, ...) - machine
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
                      'localized_carrier': 'none', 'sd_charge': 0, 'diabatization_scheme': 'mulliken',
                      'occupancy_tolerance': 0.25,
                      'do_diabatic_state_reordering': 0, 'diabatic_state_reordering_alpha': 0.0,
                      'diabatic_state_reordering_min_overlap': CONTINUITY_FLOOR,
                      'state_reordering_min_overlap': CONTINUITY_FLOOR,
                      'active_space_selection': 'energy', 'active_space_selection_algo': 2,
                      'active_space_selection_alpha': 0.0
                      }

    _refuse_former_names(params)
    # written below from active_space_selection into this dict, which a runner may hand in again - never an input
    params.pop('orbital_perms', None)

    # Check input, set output dirs
    refuse_unknown_keys(params, STEP3_KEYS, "step3.run_step3_sd_nacs_libint", "params", "no function of the step-3 workflow reads")
    comn.check_input(params, default_params, critical_params)
    if params['is_many_body'] and params['isUKS']:
        raise Exception('Unrestricted-spin calculations are not supported for many-body basis yet!')  # TODO
    params['apply_orthonormalization'] = _orthonormalization_mode(params['apply_orthonormalization'])
    _check_reordering_options(params, 'run_step3_sd_nacs_libint')
    dia_reordering, dia_alpha = params['do_diabatic_state_reordering'], params['diabatic_state_reordering_alpha']
    dia_min_overlap, sd_min_overlap = (params['diabatic_state_reordering_min_overlap'],
                                       params['state_reordering_min_overlap'])
    selection = str(params['active_space_selection']).lower()
    if selection not in ('energy', 'identity'):
        raise ValueError(F"active_space_selection = {params['active_space_selection']!r} is not a listed option - "
                         F"'energy' (the energy positions of every frame) and 'identity' (the orbitals of start_time "
                         F"followed through the window) are supported")
    algo, orb_alpha = params['active_space_selection_algo'], params['active_space_selection_alpha']
    if algo not in (1, 2, 22):
        raise ValueError(F"active_space_selection_algo = {algo!r} is not a listed option - 1 (Sato), 2 (Munkres-Kuhn, "
                         F"libra_py.hungarian) and 22 (the same assignment by scipy) are supported")
    if not (np.isreal(orb_alpha) and orb_alpha >= 0.0):
        raise ValueError(F"active_space_selection_alpha = {orb_alpha!r} - a non-negative width in 1/Hartree is needed")
    res_dir_1 = params['path_to_npz_files']
    res_dir_2 = params['path_to_save_sd_Hvibs']

    # CP2K parsing
    if params['es_software'].lower() == 'cp2k':
        sample_logfile = glob.glob(params['logfile_directory'] + '/*.log')[0]
        sample_npz_file = sorted(glob.glob(params['path_to_npz_files'] + '/S_ks_*.npz'))[0]
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
    # the _im files store Im(Hvib) = -d - files WITHOUT this marker predate
    # the convention restoration and hold +d (the NAC itself)
    marker_file = os.path.join(res_dir_2, 'hvib_convention.json')
    if not os.path.exists(marker_file) and glob.glob(os.path.join(res_dir_2, '*_im.npz')):
        raise ValueError(F"{res_dir_2} already holds *_im.npz files without a "
                         F"convention marker (legacy +d data) - writing canonical "
                         F"-d files into it would mix conventions in one directory; "
                         F"use a fresh output path or move the legacy files away")
    # the record a step-4 runner reads: a closed-shell neutral reference (isUKS 0, sd_charge 0) makes
    # the SD basis stand for the singlet SACs of S0, whose NACs to the reference carry sqrt 2 against the
    # SD-level ones - a Koopmans hole / electron basis or a UKS basis holds open-shell states only, no factor
    with open(marker_file, 'w') as f:
        json.dump({'im_file_contains': 'Im(Hvib) = -d', 'isUKS': int(params.get('isUKS', 0)),
                   'sd_charge': int(params.get('sd_charge', 0))}, f)

    # the active space by identity: the window orbitals followed between the frames before any basis is built
    if selection == 'identity':
        orbital_perms, orbital_window = _follow_orbitals(params, start_time, finish_time)
    else:
        orbital_perms, orbital_window = None, None

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

        # the configurations name the orbital positions of their frame - under the identity selection the labels of
        # the followed orbitals name them, and the union basis is the union over the labels
        if selection == 'identity':
            ci_basis_states, sd_unique_basis = _relabel_configurations(ci_basis_states, spin_components,
                                                                       orbital_perms, start_time,
                                                                       int(params['lowest_orbital']))

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
    sd_charge = int(params['sd_charge'])
    if sd_charge != 0:
        if params['isUKS']:
            raise ValueError("sd_charge needs a closed-shell reference (isUKS 0) - an "
                             "unrestricted reference is already the ion, use its own SD basis")
        if params['is_many_body']:
            raise ValueError("sd_charge needs the single-determinant route (is_many_body False) - the TD-DFT "
                             "configurations of the neutral reference do not combine with the ion's "
                             "frozen-orbital basis")
        # the reference is the first reindexed determinant of the closed-shell basis
        sd_unique_basis, sd_states_reindexed = koopmans_sd_basis(sd_states_reindexed[0], sd_charge,
                                                                 len(ks_active_space) // 2)
        params['sd_unique_basis'] = sd_unique_basis
        print(F'sd_charge {sd_charge}: {len(sd_states_reindexed)} frozen-orbital '
              F'{"hole" if sd_charge > 0 else "electron"} determinants of the neutral reference')
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

    # the label permutations relabel every window matrix the SD route cuts the active space out of (E_ks, S_ks, St_ks
    # and the fragment operators)
    if selection == 'identity':
        params['orbital_perms'] = orbital_perms
        selection_record = _selection_record(params, orbital_perms, orbital_window, ks_active_space)
        print(F"active space by identity: {len(selection_record['reordered_frames'])} reordered frames of the window, "
              F"the active space off its energy positions in {len(selection_record['tracked_orbitals'])} frames and "
              F"across its edge at {len(selection_record['active_space_edge_frames'])}, continuity min "
              F"{selection_record['min_continuity']:.4f}, {len(selection_record['unresolved_frames'])} unresolved "
              F"frames ({len(selection_record['active_space_unresolved_frames'])} of the active space), "
              F"{len(selection_record['edge_departures'])} departures from the window")
    else:
        selection_record = {"active_space_selection": "energy",
                            "note": "the active space is the energy positions of every frame"}

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
    localized_carrier = str(params['localized_carrier']).lower()
    # a hole basis holds no electron to localize and an electron basis no hole: the localization would find no
    # manifold and hand the SD states back as they are
    own_carrier = {1: 'hole', -1: 'electron'}.get(sd_charge)
    if own_carrier is not None and localized_carrier not in ('none', own_carrier):
        raise ValueError(F"localized_carrier {localized_carrier!r} does not fit sd_charge {sd_charge:+d} - its SD basis holds "
                         F"the {own_carrier} only, use localized_carrier '{own_carrier}' or 'none'")
    W_dia = None
    if localized_carrier != 'none':
        W_dia, dia_index, dia_pops, dia_hole, dia_energies, dia_operators = _sd_diabatization_series(
            params, sd_states_reindexed_sorted, ks_active_space, start_time, finish_time, localized_carrier)
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

        # the many-body diabatization: the same fragment operators carried into the CI space through the T
        # matrices and localized there - U_ci next to U_sd
        V_dia = None
        if W_dia is not None:
            V_dia, ci_dia_index, ci_pops, ci_hole, ci_dia_energies = _ci_diabatization_series(
                params, sd_states_reindexed_sorted, SD2CI, ci_energies, ks_active_space, start_time, finish_time,
                localized_carrier, fragment_operators=dia_operators)

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
        # the records follow the states: written once the columns of W stand in the order of start_time
        perms_dia, swaps_dia, unresolved_dia, cont_dia, edges_dia, align_dia = _follow_diabatic_states(
            W_dia, St_sds, start_time, "sd", dia_index, dia_reordering, dia_alpha, _state_energies(dia_energies),
            dia_min_overlap)
        followed = lambda records: _rotate_records(_permute_records(records, perms_dia), align_dia)
        _write_diabatization_records(res_dir_2, localized_carrier, dia_index, followed(dia_pops), followed(dia_hole),
                                     followed(dia_energies), start_time, str(params['diabatization_scheme']).lower(),
                                     extra=_tracking_record(dia_reordering, dia_alpha, dia_min_overlap, perms_dia,
                                                            swaps_dia, unresolved_dia, cont_dia, edges_dia, start_time,
                                                            align_dia))
        print(F"diabatic SD states: {len(swaps_dia)} swap frames, {len(unresolved_dia)} frames let through with a "
              F"continuation below the floor, {len(edges_dia)} edge departures, continuity min {cont_dia:.4f}")
    if params['do_state_reordering'] in (1, 2, 22):
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
    # the continuity of the SD states as labeled: refused below the threshold after a reordering, recorded otherwise
    route_report = {"sd": _continuity_record(St_sds, start_time, params['do_state_reordering'],
                                             params['state_reordering_alpha'], sd_min_overlap,
                                             perms if params['do_state_reordering'] in (1, 2, 22) else None)}
    print(F"SD states: {len(route_report['sd']['reordered_frames'])} reordered frames, "
          F"{len(route_report['sd']['unresolved_frames'])} frames let through with a continuation below the floor, "
          F"{len(route_report['sd']['edge_departures'])} edge departures (a state left the active space), continuity "
          F"min {route_report['sd']['min_continuity']:.4f}")

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

        if params['do_state_reordering'] in (1, 2, 22):
            # Since we have performed state-reordering we need to
            # convert to scipy npz format now
            t2 = time.time()
            # the relabeled SD overlaps carry the labels of perms - the rows of the T matrices
            # (the sorted SDs of every step) follow them, St_ci is then invariant to the relabeling
            for i in range(len(SD2CI)):
                SD2CI[i] = _permute_rows(SD2CI[i], perms[i])
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
            perms_ci = apply_state_reordering_general(St_cis_cmatrix, E_cis_cmatrix, params)
            if V_dia is not None:
                # the relabeled CI states are the rows of V
                V_dia = [_permute_rows(V_dia[i], perms_ci[i]) for i in range(len(V_dia))]
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

        if V_dia is not None:
            # the CI records hold eV, the penalty width alpha is per Hartree as for the SD states
            perms_ci_dia, swaps_ci_dia, unresolved_ci_dia, cont_ci_dia, edges_ci_dia, align_ci = _follow_diabatic_states(
                V_dia, St_cis, start_time, "ci", ci_dia_index, dia_reordering, dia_alpha,
                [e / units.au2ev for e in _state_energies(ci_dia_energies)], dia_min_overlap)
            followed_ci = lambda records: _rotate_records(_permute_records(records, perms_ci_dia), align_ci)
            _write_diabatization_records(res_dir_2, localized_carrier, ci_dia_index, followed_ci(ci_pops),
                                         followed_ci(ci_hole), followed_ci(ci_dia_energies), start_time,
                                         str(params['diabatization_scheme']).lower(), basis="ci",
                                         extra=_tracking_record(dia_reordering, dia_alpha, dia_min_overlap,
                                                                perms_ci_dia, swaps_ci_dia, unresolved_ci_dia,
                                                                cont_ci_dia, edges_ci_dia, start_time, align_ci))
        route_report["ci"] = _continuity_record(St_cis, start_time, params['do_state_reordering'],
                                                params['state_reordering_alpha'], sd_min_overlap,
                                                perms_ci if params['do_state_reordering'] in (1, 2, 22) else None, "CI")

    route_report["active_space"] = selection_record
    with open(F'{res_dir_2}/sd_route_report.json', 'w') as f:
        json.dump(route_report, f, indent=1)

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

                if V_dia is not None:
                    # U_ci(n) refers to the phase-corrected CI states F(n)|Psi(n)>: the cumulative
                    # phases BEFORE this step's update
                    _save_u(res_dir_2, V_dia[step], cum_phase_aa, step + start_time, "ci")
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
            if V_dia is not None:
                _save_u(res_dir_2, V_dia[len(St_cis)], cum_phase_aa, len(St_cis) + start_time, "ci")
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
            if V_dia is not None:
                for i in range(len(V_dia)):
                    _save_u(res_dir_2, V_dia[i], None, i + start_time, "ci")


def make_var_pool(params, sd_states_reindexed_sorted, start_time, finish_time):
    """Per-step argument tuples for compute_sd_overlaps_in_parallel.

    Each task ships only the two SD lists (and the two orbital permutations of active_space_selection 'identity') it
    consumes - putting the full per-step list into every task makes the Pool pickling cost O(nsteps^2).
    """
    var_pool = []
    for step in range(finish_time - start_time - 1):
        p_task = dict(params)
        p_task['sd_states_reindexed_sorted'] = {step: sd_states_reindexed_sorted[step],
                                                step + 1: sd_states_reindexed_sorted[step + 1]}
        if 'orbital_perms' in params:
            frames = (start_time + step, start_time + step + 1)
            p_task['orbital_perms'] = {k: params['orbital_perms'][k] for k in frames}
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
    # magnitude faster for large SD bases) - "batched"/"lemma" force one - "per_pair" = ovlp_mat_arb
    ovlp = mapping.sd_overlap_kernel(params.get('sd_overlap_kernel', 'auto'))

    #print("active_space = ", active_space)
    #print("active_space2 = ", active_space2)
    #print("sd_states_reindexed_sorted = ", sd_states_reindexed_sorted)

    t2 = time.time()
    print('Computing the SD overlaps for step', step)

    # The same procedure as above
    st_ks = _window_matrix(params, 'St_ks', step + start_time, active_space, step + start_time + 1)
    s_ks_1 = _window_matrix(params, 'S_ks', step + start_time, active_space)
    s_ks_2 = _window_matrix(params, 'S_ks', step + start_time + 1, active_space)
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
