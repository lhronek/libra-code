# *********************************************************************************
# * Copyright (C) 2026 Mohammad Shakiba, Brendan Smith, Alexey V. Akimov
# * This file is distributed under the terms of the GNU General Public License
# * as published by the Free Software Foundation, either version 3 of
# * the License, or (at your option) any later version.
# * See the file LICENSE in the root directory of this distribution
# * or <http://www.gnu.org/licenses/>.
# *
# *********************************************************************************/
"""
.. module:: CP2K_methods
   :platform: Unix, Windows
   :synopsis: This module implements functions for processing the CP2K inputs and outputs.
.. moduleauthors::
       Mohammad Shakiba, Brendan Smith, Alexey V. Akimov

"""


import os, sys, math, re, copy, time, glob, shlex, subprocess, atexit
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
import scipy.sparse as sp
import scipy.linalg
from libra_py.workflows.nbra import mapping, step2_many_body
import scipy.io as io
import libra_py.citools.ci as ci
from libra_py.citools.sd_basis import (make_sd_basis, sd2ci_matrix,
                                       align_ci_state_signs, check_sd_capture)

if sys.platform == "cygwin":
    from cyglibra_core import *
elif sys.platform == "linux" or sys.platform == "linux2":
    from liblibra_core import *
import util.libutil as comn
from libra_py import data_conv

from libra_py import data_outs, data_read, data_stat, data_conv
from libra_py import units, molden_methods, orthogonalizations
import libra_py.orthogonalizations as ortho


def ndigits(integer_number: int):
    """
    This function calculates the number of digits for an integer number.

    Args:

        integer_number ( integer ): An integer number.

    Returns:

        digit_num ( integer ): The number of digits for 'num'.
    """

    # Setting up the digit number counter
    digit_num = 0

    while (integer_number > 0):

        digit_num = digit_num + 1
        integer_number = integer_number // 10

    return digit_num


def state_num_cp2k(state_num: int):
    """
    This function turns an integer number to a string. This string
    has the format in five digits and if the number of digits is less than
    five it will append zeros before the number. This is necessary to read
    the .cube files produced by CP2K.

    Args:

        state_num (integer): Integer number which is in fact the energy level.

    Returns:

        state_num_str (string): The number in five digits (or more) in string format.

    """
    # Setting up an initial string
    state_num_str = ''

    # Now count the number of digits for the state sumber
    # and then add 0 before the number in the string

    if ndigits(state_num) == 1:

        state_num_str = '0000%d' % state_num

    elif ndigits(state_num) == 2:

        state_num_str = '000%d' % state_num

    elif ndigits(state_num) == 3:

        state_num_str = '00%d' % state_num

    elif ndigits(state_num) == 4:

        state_num_str = '0%d' % state_num

    else:
        state_num_str = '%d' % state_num

    return state_num_str


def cube_file_names_cp2k(params):
    """
    This function will produce the cube file names produced by CP2K both for energy calculations
    and molecular dynamics.

    Args:

        params( dictionary ): contains dynamical paramters

            project_name ( string ): The project name used in CP2K input file.

            min_band ( integer ): The minimum state number to be considered.

            max_band ( integer ): The maximum state number to be considered.

            curr_step ( integer ): The time current step in CP2K calculations.

            isUKS ( int ): 1 if spin-polarized calculations used

    Returns:

        names ( list ): The list of cube file names.

    """
    # Critical parameters
    critical_params = ["project_name", "min_band", "max_band", "curr_step"]
    # Default parameters
    default_params = {"isUKS": 0}
    # Check input
    comn.check_input(params, default_params, critical_params)

    project_name = params["project_name"]
    min_band = params["min_band"]
    max_band = params["max_band"]
    curr_step = params["curr_step"]
    isUKS = params["isUKS"]
    es_software = params["es_software"]

    # isUKS needs to be epressed as a string, even though it is used as a Boolean in other parts of the workflows
    spin = []
    if isUKS == 1:
        spin = [1, 2]
    else:
        spin = [1]

    # Define the names of the cube files for time t.
    cube_file_names = []

    for i in range(min_band, max_band + 1):

        # Using the function state_num_cp2k to obtain the names of the .cube files for state 'i'.
        state_num = state_num_cp2k(i)

        # For the current time step "curr_step"
        if isUKS == 1:

            if es_software == "cp2k" or es_software == "gaussian":

                cube_file_name_of_band_i = str(
                    'cubefiles/%s-%d-WFN_%s_%d-1_0.cube' %
                    (project_name, curr_step, state_num, spin[0]))
                cube_file_name_of_band_j = str(
                    'cubefiles/%s-%d-WFN_%s_%d-1_0.cube' %
                    (project_name, curr_step, state_num, spin[1]))
                cube_file_names.append(cube_file_name_of_band_i)
                cube_file_names.append(cube_file_name_of_band_j)

            elif software == "dftb+":
                raise ValueError("reading cubefiles from spin-polarized dftb+ "
                                 "calculations is not currently available")

        else:

            # Using the function state_num_cp2k to obtain the names of the .cube files for state 'i'.
            state_num = state_num_cp2k(i)
            cube_file_name_of_band_i = str(
                'cubefiles/%s-%d-WFN_%s_%d-1_0.cube' %
                (project_name, curr_step, state_num, spin[0]))
            cube_file_names.append(cube_file_name_of_band_i)

    return cube_file_names



def read_cp2k_tddfpt_log_file(params):
    """
    Parse a CP2K TD-DFPT log file and extract excitation energies and CI expansion data.

    This function reads the output of a CP2K Time-Dependent Density Functional
    Perturbation Theory (TD-DFPT) calculation and reconstructs the excited-state
    wavefunctions in terms of single-particle (Kohn–Sham) excitations.

    In CP2K TD-DFPT, each excited state is represented as a linear combination of
    single excitations (Slater determinants) of the form:

        |Ψ_I⟩ = Σ_{i→a} C_{i→a}^{(I)} |Φ_i^a⟩

    where:
        - i denotes an occupied orbital
        - a denotes a virtual (unoccupied) orbital
        - C_{i→a}^{(I)} are CI-like expansion coefficients
        - |Φ_i^a⟩ is a singly-excited Slater determinant

    The function extracts:
        1. Excitation energies for each TD-DFPT state
        2. CI basis configurations (orbital transitions i → a)
        3. Corresponding expansion coefficients
        4. Spin labels (for UKS calculations)

    Only configurations with coefficient magnitude satisfying:

        |C_{i→a}|^2 > ci_threshold

    are retained.

    ------------------------------------------------------------------------
    Args:
        params (dict): Dictionary of input parameters.

            logfile_name (str, optional):
                Path to the CP2K log file.
                Default: "output.log"

            number_of_states (int, optional):
                Maximum number of excited states to extract.
                If larger than the number available in the file,
                it will be truncated automatically.

            ci_threshold (float, optional):
                Threshold for filtering CI coefficients.
                Only configurations with |C|^2 > ci_threshold are kept.
                Default: 1e-6

            isUKS (bool or int, optional):
                Flag indicating whether the calculation is spin-polarized.
                - True / 1  → UKS (spin-polarized)
                - False / 0 → RKS (spin-restricted)

                Affects how the CI lines are parsed and whether spin labels
                are explicitly read from the log file.

            orbital_space (iterable of int, optional):
                Subset of orbital indices to define an "active space".
                If None, all orbitals [1, ..., nmo] are used.

    ------------------------------------------------------------------------
    Returns:

        info (dict):
            Dictionary containing system and excitation metadata:

                nocc (int):
                    Number of occupied orbitals.

                nelec (int):
                    Total number of electrons.

                nao (int):
                    Number of atomic orbitals (approximate; set equal to nmo).

                nmo (int):
                    Total number of molecular orbitals.

                nci (int):
                    Number of excited states actually returned.

                min_occ, max_occ (int):
                    Minimum and maximum occupied orbital indices involved
                    in the CI expansions.

                min_vir, max_vir (int):
                    Minimum and maximum virtual orbital indices involved
                    in the CI expansions.

                nact (int):
                    Size of the active orbital space.

                actual_orbital_space (list of int):
                    List of orbital indices used in the active space.

        data (list):
            A list containing:

                excitation_energies (list of float):
                    Excitation energies (in CP2K units, typically eV).

                ci_basis (list of list of [int, int]):
                    CI configurations for each state.
                    Each configuration is represented as:
                        [i, a]
                    where i = occupied orbital index,
                          a = virtual orbital index.

                ci_coefficients (list of list of float):
                    Corresponding CI coefficients for each configuration.

                spin_components (list of list of string):
                    Describe spins for the single-particle excitations (no spin-flips)

    ------------------------------------------------------------------------
    Notes:

        • The parser relies on specific formatting of CP2K output, particularly:
              - "TDDFPT|" lines for excitation energies
              - "Excitation analysis" section for CI expansions

        • The function assumes single excitations only (as provided by TD-DFPT).

        • For UKS calculations, spin labels (alpha/beta) are extracted explicitly.
          For RKS calculations, all excitations are labeled as "alpha".

        • The mapping:
              occupied orbitals → [1, ..., nocc]
              virtual orbitals → [nocc+1, ..., nmo]
          is assumed but not strictly enforced.

        • nao is set equal to nmo, which may not strictly hold in CP2K.

        • The tolerance parameter is important for reducing the size of the CI
          expansion and focusing on dominant configurations.

    ------------------------------------------------------------------------
    Potential Improvements:

        - Add unit conversion for excitation energies
        - Support parsing oscillator strengths
        - Improve robustness against CP2K format changes
        - Return spin_components explicitly in the output

    """

    # ===================== Parameters =====================
    logfile_name = params.get("logfile_name", "output.log")
    nstates_req = params.get("number_of_states", 1)
    if "tolerance" in params:
        raise ValueError('tolerance was unified into "ci_threshold" (the same '
                         "|C|^2 cut) - rename the key, the value is unchanged")
    tol = params.get("ci_threshold", 1e-6)
    isUKS = bool(params.get("isUKS", False))
    orbital_space = params.get("orbital_space", None)

    # ===================== Read file ======================
    # the driver parses one step's log several times (CI data, info,
    # homo) - the stat-keyed cache reads it once
    lines = data_read.read_lines_cached(logfile_name)

    nlines = len(lines)

    # ===================== General info ===================
    nelec = nocc = nmo = nexcitations = 0

    for line in lines:
        if "Number of electrons:" in line:
            nelec = int(float(line.split()[3]))
        elif "Number of occupied orbitals:" in line:
            nocc = int(float(line.split()[4]))
        elif "Number of molecular orbitals:" in line:
            nmo = int(float(line.split()[4]))
        elif "TDDFPT| Number of states calculated" in line:
            nexcitations = int(float(line.split()[5]))

    nao = nmo  # NOTE: may differ in CP2K
    nci = min(nstates_req, nexcitations)
    # the reference occupation is the electron count - the printed "occupied"
    # count is CP2K's smeared HOMO under &SMEAR (cross-checked, warned once)
    nocc_printed = nocc
    ref = reference_occupation(logfile_name, isUKS=isUKS)
    nref_of = {"alp": ref[0], "bet": ref[1]} if isUKS else {"alp": ref, "bet": ref}
    nocc = ref[0] if isUKS else ref

    # Orbital space
    if orbital_space is None:
        actual_orbital_space = list(range(1, nmo + 1))
    else:
        actual_orbital_space = list(orbital_space)

    nact = len(actual_orbital_space)

    # ===================== Locate sections =================
    exc_anal_line = None
    for i, line in enumerate(lines):
        l = line.lower()
        if "excitation analysis" in l:
            exc_anal_line = i
            break

    if exc_anal_line is None:
        raise RuntimeError("Could not find 'Excitation analysis' section in log file.")

    # ===================== Excitation energies =============
    excitation_energies = []

    for i, line in enumerate(lines):
        if "TDDFPT|" in line and "Molecular" not in lines[i - 1]:
            parts = line.split()
            try:
                excitation_energies.append(float(parts[2]) * units.ev2Ha )
            except (IndexError, ValueError):
                continue

    # ===================== State block indices ============
    state_num_lines = []

    for i in range(exc_anal_line + 5, nlines):
        tokens = lines[i].split()

        if len(tokens) == 0 or "----" in lines[i]:
            state_num_lines.append(i)
            break

        if isUKS:
            if len(tokens) in (1, 3):
                state_num_lines.append(i)
        else:
            if len(tokens) == 1:
                state_num_lines.append(i)
            elif len(tokens) > 1 and tokens[0].isdigit() and not tokens[1].isdigit():
                state_num_lines.append(i)

    # ===================== CI data ========================
    ci_basis = []
    ci_coefficients = []
    spin_components = []
    read_drops = np.zeros(nci)

    for i in range(nci):
        tmp_basis = []
        tmp_coeffs = []
        tmp_spin = []
        kept2 = 0.0
        foreign = []

        for j in range(state_num_lines[i] + 1, state_num_lines[i + 1]):
            tokens = lines[j].split()

            if not tokens:
                continue

            try:
                if isUKS:
                    coeff = float(tokens[4])
                    if coeff**2 > tol:
                        spin = tokens[1].replace("(", "").replace(")", "")
                        occ = int(tokens[0])
                        vir = int(tokens[2])

                else:
                    coeff = float(tokens[2])
                    if coeff**2 > tol:
                        spin = "alp"
                        occ = int(tokens[0])
                        vir = int(tokens[1])

                if coeff**2 > tol:
                    nref = nref_of.get(spin, nref_of["alp"])
                    if vir <= nref:
                        raise ValueError(
                            F"{logfile_name}: state {i + 1} excites {occ}->{vir} into an "
                            F"orbital the electron-count reference ({nref}) occupies - the "
                            F"vectors do not belong to this reference")
                    if occ > nref:
                        # an excitation FROM an orbital above the reference: CP2K's
                        # smeared partition counted it as occupied. No integer-
                        # occupation basis holds it - counted as dropped weight
                        foreign.append((occ, vir, spin, coeff ** 2))
                        continue
                    tmp_basis.append([occ, vir])
                    tmp_coeffs.append(coeff)
                    tmp_spin.append(spin)
                    kept2 += coeff ** 2

            except (IndexError, ValueError) as err:
                if isinstance(err, ValueError) and "electron-count reference" in str(err):
                    raise
                continue

        # TDA/sTDA vectors are unit-norm, so 1 - kept2 is the exact CI weight
        # this state loses to the log's 5e-2 print floor plus the tolerance
        # cut - downstream normalization hides it from every later norm
        read_drops[i] = max(0.0, 1.0 - kept2)
        if foreign:
            foreign.sort(key=lambda x: -x[3])
            print(F"read_cp2k_tddfpt_log_file: state {i + 1}: {sum(w for *_, w in foreign):.1%} of "
                  F"the vector excites FROM orbital(s) above the electron-count reference "
                  F"({nref_of['alp']}{'/' + str(nref_of['bet']) if isUKS else ''}) - counted as "
                  F"dropped weight: " + ", ".join(F"{o}->{v}{'' if sp == 'alp' else ' ' + sp} ({w:.1%})"
                                                   for o, v, sp, w in foreign[:4]))
        if read_drops[i] > 1e-6:
            print(F"read_cp2k_tddfpt_log_file: state {i + 1}: dropped CI weight "
                  F"{read_drops[i]:.2e} to the print floor + ci_threshold cut"
                  F"{' + excitations from above the reference' if foreign else ''} "
                  F"(vs the TDA unit norm)")

        ci_basis.append(tmp_basis)
        ci_coefficients.append(tmp_coeffs)
        spin_components.append(tmp_spin)

    # ===================== Orbital bounds =================
    min_occ, max_occ = nocc, 1
    min_vir, max_vir = nmo, nocc + 1

    for state in ci_basis:
        for occ, vir in state:
            min_occ = min(min_occ, occ)
            max_occ = max(max_occ, occ)
            min_vir = min(min_vir, vir)
            max_vir = max(max_vir, vir)

    # ===================== Output =========================
    info = {
        "nocc": nocc,
        "nocc_printed": nocc_printed,
        "nelec": nelec,
        "nao": nao,
        "nmo": nmo,
        "nci": nci,
        "min_occ": min_occ,
        "max_occ": max_occ,
        "min_vir": min_vir,
        "max_vir": max_vir,
        "nact": nact,
        "actual_orbital_space": actual_orbital_space,
    }

    data = [excitation_energies[:nci], ci_basis, ci_coefficients, spin_components]

    # capture metric for the callers - the returned coefficients are raw, but
    # every consumer normalizes them, so the loss is only knowable here
    params["ci_read_drops"] = read_drops

    return info, data



def read_cp2k_tddfpt_log_file_old(params):
    """
    A new version is added on 3/22/2026, this one is to be DEPRECATED
    However, let's keep it for a while to make sure we are backward-compatible, if we need to
    
    This function reads log files generated from TD-DFPT calculations using CP2K and returns the TD-DFPT
    excitation energies, Slater determinent states in terms of the Kohn-Sham orbital indicies,
    and the Slater determinent coefficients that comprise the multi-configurational electronic states

    Args:
        params ( dictionary ): parameters dictionary

            logfile_name ( string ): name of the log file for this particular timestep

            number_of_states ( int ): how many ci states to consider

            tolerance ( float ): the tolerance for accepting SDs are members of the CI wavefunctions

            isUKS ( Boolean ): flag for whether or not a spin-polarized Kohn-Sham basis is being used. TRUE means
                                 that a spin-polarized Kohn-Sham basis is being used.
    Returns:
        excitation_energies ( list ): The excitation energies in the log file.

        ci_basis ( list ): The CI-basis configuration.

        ci_coefficients ( list ): The coefficients of the CI-states.

        spin_components (list): The spin component of the excited states.

    """

    # Critical parameters
    critical_params = ["logfile_name", "number_of_states"]
    # Default parameters
    default_params = {"ci_threshold": 0.05, "isUKS": 0, "lowest_orbital": 1, "highest_orbital": int(1e10)}
    # Check input
    comn.check_input(params, default_params, critical_params)

    logfile_name = params["logfile_name"]
    number_of_states = int(params["number_of_states"])
    if "tolerance" in params:
        raise ValueError('tolerance was unified into "ci_threshold" (the same '
                         "|C|^2 cut) - rename the key, the value is unchanged")
    tolerance = float(params["ci_threshold"])
    isUKS = int(params["isUKS"])

    f = open(logfile_name, 'r')
    lines = f.readlines()
    f.close()

    for i in range(0, len(lines)):
        tmp_line = lines[i].lower().split()
        if 'excitation' in tmp_line:
            if 'analysis' in tmp_line:

                # When found the line in which contains 'Excitation analysis' in
                # the log file, append it in the variable exc_anal_line
                exc_anal_line = i

        if 'states' in tmp_line:
            if 'multiplicity' in tmp_line:

                # Here we search for the line that contains 'R-TDDFPT states of multiplicity 1'
                # or 'U-TDDFPT states of multiplicity 1' in the log file
                r_tddfpt_line = i

    excitation_energies = []
    # Old version
    # Start from 5 lines after finding the line contaning 'R-TDDFPT states of multiplicity 1'
    # This is because they contain the energies from that line.
    # for i in range( r_tddfpt_line+5, len( lines ) ):
    # tmp_line = lines[i].split()
    # if len( tmp_line ) == 0:
    # break
    # excitation_energies.append( float( tmp_line[2] ) )

    for i in range(len(lines)):
        if 'TDDFPT|' in lines[i]:
            if 'Molecular' not in lines[i - 1]:
                try:
                    tmp = lines[i].split()
                    exc_ener = float(tmp[2])
                    excitation_energies.append(exc_ener)
                except BaseException:
                    pass

    # Start from 5 lines after finding the line contaning 'Excitation analysis'
    # From that point we have the state numbers with their configurations.
    # So, we append the lines which contain only 'State number' and stop
    # whenever we reach to a blank line.
    state_num_lines = []
    for i in range(exc_anal_line + 5, len(lines)):
        tmp_line = lines[i].split()
        if isUKS == 1:

            if len(tmp_line) == 0 or '----' in lines[i]:
                state_num_lines.append(i)
                break
            if len(tmp_line) == 1 or len(tmp_line) == 3:
                state_num_lines.append(i)

        else:

            if len(tmp_line) == 0 or '----' in lines[i]:
                state_num_lines.append(i)
                break
            if len(lines[i].split()) == 1:
                state_num_lines.append(i)
            elif len(tmp_line) > 1:
                if tmp_line[0].isdigit() and not (tmp_line[1].isdigit()):
                    state_num_lines.append(i)

    # Setting up the CI-basis list and their coefficients list
    ci_basis = []
    ci_coefficients = []
    spin_components = []
    for i in range(number_of_states):

        # states and their coefficients for each excited state
        tmp_state = []
        tmp_state_coefficients = []
        tmp_spin = []
        for j in range(state_num_lines[i] + 1, state_num_lines[i + 1]):

            # Splitting lines[j] into tmp_splitted_line
            tmp_splitted_line = lines[j].split()

            # If we are using the spin-polarized Kohn-Sham basis, then the size of the split line is different
            # from the size of the split line when using the spin-unpolarized Kohn-Sham basis
            if isUKS == 1:

                ci_coefficient = float(tmp_splitted_line[4])
                if ci_coefficient**2 > tolerance:
                    if int(tmp_splitted_line[0]) >= params["lowest_orbital"] and int(
                            tmp_splitted_line[1]) <= params["highest_orbital"]:
                        # We need to remove the paranthesis from the 2nd element of the temporary splitted line
                        tmp_spin.append(tmp_splitted_line[1].replace('(', '').replace(')', ''))
                        tmp_state.append([int(tmp_splitted_line[0]), int(tmp_splitted_line[2])])
                        tmp_state_coefficients.append(ci_coefficient)

            # Here, we have the spin-unpolarize Kohn-Sham basis
            # For this case, spin-components will just return all alpha
            else:
                ci_coefficient = float(tmp_splitted_line[2])
                if ci_coefficient**2 > tolerance:
                    if int(tmp_splitted_line[0]) >= params["lowest_orbital"] and int(
                            tmp_splitted_line[1]) <= params["highest_orbital"]:
                        tmp_spin.append("alp")
                        tmp_state.append([int(tmp_splitted_line[0]), int(tmp_splitted_line[1])])
                        tmp_state_coefficients.append(ci_coefficient)

        # Append the CI-basis and and their coefficients for
        # this state into the ci_basis and ci_coefficients lists
        ci_basis.append(tmp_state)
        ci_coefficients.append(tmp_state_coefficients)
        spin_components.append(tmp_spin)

    return excitation_energies[0:number_of_states], ci_basis, ci_coefficients, spin_components


def cp2k_distribute(istep, fstep, nsteps_this_job, cp2k_positions, cp2k_input, curr_job_number):
    """
    Distributes cp2k jobs for trivial parallelization
    *** Make sure that your cp2k input file has absolute paths to the following input parameters:
        BASIS
        POTENTIAL
        < any other file dependent parameter (dftd3, etc) >
    Args:

        istep (int): The initial time step for molecular dynamics trajectory which
                     the user wants to start the calculations with. This paramter
                     is set in the input file for submitting the jobs.

        fstep (int): The final time step for molecular dynamics trajectory which
                     the user wants to start the calculations with. This paramter
                     is set in the input file for submitting the jobs.

        nsteps_this_job (int): The number of step for the job to be distributed.

        cp2k_positions (str): The full path to the molecular dynamics trajectory xyz file.

        curr_job_number (int): The current job number.

    Returns:

        None

    """

    # Now we need to distribute the jobs into job batches
    # First, make a working directory where the calculations will take place
    os.chdir("wd")

    # total number of steps
    nsteps = fstep - istep + 1

    # number of jobs obtained from nsteps/nsteps_this_job
    njobs = int(nsteps / nsteps_this_job)

    # The current step is set to istep
    curr_step = istep

    # Making the job directories in the wd folder like job0, job1, ...
    os.system("mkdir job" + str(curr_job_number) + "")

    # Chnage directory to job folder
    os.chdir("job" + str(curr_job_number) + "")

    # Copy the trajectory xyz file
    os.system("cp ../../" + cp2k_positions + " .")

    # Copy the CP2K sample input
    os.system("cp ../../" + cp2k_input + " .")

    # Go back to the main directory
    os.chdir("../../")


def read_trajectory_xyz_file(file_name: str, step: int):
    """
    This function reads the trajectory of a molecular dynamics .xyz file and
    extract the 'step' th step then writes it to coord-step.xyz file. This function
    is used in single point calculations for calculations of the NACs where one has
    previously obtained the trajectory via any molecular dynamics packages.

    Args:

        file_name (string): The trajectory .xyz file name.
            step (integer): The desired time to extract its .xyz
                            coordinates which starts from zero.

    Returns:

        (list, MATRIX(ndof, 1)): labels of all atoms, and their coordinates, ndof = 3 * natoms

    """

    f = open(file_name, 'r')
    lines = f.readlines()
    f.close()

    # The number of atoms for each time step in the .xyz file of the trajectory.
    number_of_atoms = int(lines[0].split()[0])

    q = MATRIX(3 * number_of_atoms, 1)

    # Write the coordinates of the 't' th step in file coord-t.xyz
    f = open('coord-%d' % step + '.xyz', 'w')

    # This is used to skip the first two lines for each time step.
    n = number_of_atoms + 2

    # Write the coordinates of the 'step'th time step into the file
    for i in range(n * step, n * (step + 1)):
        f.write(lines[i])
    f.close()

    labels = []
    for i in range(number_of_atoms):
        tmp = lines[n * step + 2 + i].split()
        labels.append(tmp[0])
        x = float(tmp[1]) * units.Angst  # convert Angstrom to Bohr
        y = float(tmp[2]) * units.Angst  # convert Angstrom to Bohr
        z = float(tmp[3]) * units.Angst  # convert Angstrom to Bohr
        q.set(3 * i + 0, 0, x)
        q.set(3 * i + 1, 0, y)
        q.set(3 * i + 2, 0, z)

    return labels, q


def read_velocity_xyz_file(file_name: str, step: int):
    """
    The velocity counterpart of read_trajectory_xyz_file: extracts the
    'step' th frame of a CP2K velocity trajectory (*-vel-1.xyz). CP2K
    writes velocities in ATOMIC UNITS, so no unit conversion is applied
    (the position reader converts Angstrom to Bohr), and no side file is
    written.

    Args:

        file_name (string): The velocity trajectory .xyz file name.
            step (integer): The desired frame, starting from zero.

    Returns:

        (list, MATRIX(ndof, 1)): labels of all atoms, and their
            velocities in atomic units, ndof = 3 * natoms

    """

    f = open(file_name, 'r')
    lines = f.readlines()
    f.close()

    number_of_atoms = int(lines[0].split()[0])
    v = MATRIX(3 * number_of_atoms, 1)
    n = number_of_atoms + 2

    labels = []
    for i in range(number_of_atoms):
        tmp = lines[n * step + 2 + i].split()
        labels.append(tmp[0])
        v.set(3 * i + 0, 0, float(tmp[1]))
        v.set(3 * i + 1, 0, float(tmp[2]))
        v.set(3 * i + 2, 0, float(tmp[3]))

    return labels, v


def CP2K_input_static(cp2k_sample_energy_input: str, project_name: str,
                      trajectory_file: str, time_step: int):
    """
    This function provides the inputs required for single point calculations used for
    calculation of NACs and overlap matrix. This function requires a sample CP2K input
    used for ENERGY calculations and the project name which will used to identify the
    produced cube files.
    Args:
        cp2k_sample_energy_input (string): The CP2K sample input file for energy calculations.

        project_name (string): The poject name of the CP2K sample input (in &GLOBAL-> PROJECT)
                               This is required for identifying the cube files.

        trajectory_file (string): The .xyz trajectory file obtained from previous
                                  molecular dynamics calculations

        time_step (integer): The time step of the single point calculation. It starts from 0.

    Returns:

        None

    """

    # Reading the .xyz trajectory file and extract the coordinates at time 'time_step'.
    read_trajectory_xyz_file(trajectory_file, time_step)

    # The new project name for each time_step.
    f_new = project_name + '-%d' % time_step + '.inp'

    f = open(cp2k_sample_energy_input, 'r')
    lines = f.readlines()
    f.close()

    # wfn_restart_file name
    c = 0

    f = open(f_new, 'w')
    for i in range(0, len(lines)):
        # Writing the COORD_FILE_NAME in the input
        if 'COORD_FILE_NAME'.lower() in lines[i].lower().split():
            f.write('      COORD_FILE_NAME')
            f.write(' coord-%d' % time_step + '.xyz')
            f.write('\n')
        # Writing the project name
        elif 'PROJECT'.lower() in lines[i].lower().split():
            f.write('  PROJECT')
            f.write(' %s' % project_name + '-%d' % time_step)
            f.write('\n')
        elif 'PROJECT_NAME'.lower() in lines[i].lower().split():
            f.write('  PROJECT')
            f.write(' %s' % project_name + '-%d' % time_step)
            f.write('\n')
        # Writing the WFN_RESTART_FILE_NAME of the previous time step
        elif 'WFN_RESTART_FILE_NAME'.lower() in lines[i].lower().split() and ".wfn" in lines[i].lower() and "!" not in lines[i].lower():
            f.write('    WFN_RESTART_FILE_NAME')
            f.write(' %s' % project_name + '-%d' % (time_step - 1) + '-RESTART.wfn')
            f.write('\n')
        # Writing the WFN_RESTART_FILE_NAME of the previous time step for the tddft calculation
        elif 'WFN_RESTART_FILE_NAME'.lower() in lines[i].lower().split() and ".tdwfn" in lines[i].lower() and "!" not in lines[i].lower():
            f.write('    WFN_RESTART_FILE_NAME')
            f.write(' %s' % project_name + '-%d' % (time_step - 1) + '-RESTART.tdwfn')
            f.write('\n')
        # Setting the scf guess to restart, CP2K will automatically switch to atomic if the wfn file does not exist
        elif 'SCF_GUESS'.lower() in lines[i].lower().split():
            f.write('    SCF_GUESS RESTART')
            f.write('\n')
        else:
            f.write(lines[i])


def read_energies_from_cp2k_log_file(params):
    """
    This function reads the energies from CP2K log file for a specific spin.

    Args:

        params (dictionary):

            logfile_name (str): The output file produced by CP2K

            time (int): The time step of the molecular dynamics. For single point calculations
                        it should be set to 0.

            min_band (int): The minimum state number.

            max_band (int): The maximum state number.

            spin (int): This number can only accept two values: 1 for alpha and 2 for beta spin.

    Returns:

        E (1D numpy array): The vector consisting of the KS energies from min_band to max_band.

        total_energy (float): The total energy obtained from the log file.

    """

    # Critical parameters
    critical_params = ["logfile_name", "min_band", "max_band"]
    # Default parameters
    default_params = {"time": 0, "spin": 1}
    # Check input
    comn.check_input(params, default_params, critical_params)

    cp2k_log_file_name = params["logfile_name"]
    # Time step, for molecular dynamics it will read the energies
    # of time step 'time', but for static calculations the time is set to 0
    time = params["time"]

    # Koh-Sham orbital indicies
    # ks_orbital_indicies = params["ks_orbital_indicies"]

    # The minimum state number
    min_band = params["min_band"]  # ks_orbital_indicies[0]
    # The maximum state number
    max_band = params["max_band"]  # ks_orbital_indicies[-1]

    spin = params["spin"]

    # First openning the file and reading its lines
    f = open(cp2k_log_file_name, 'r')
    lines = f.readlines()
    f.close()

    # The lines containing 'Eigenvalues of the occupied subspace'
    lines_with_occupied = []
    # The lines containing 'Eigenvalues of the unoccupied subspace'
    lines_with_unoccupied = []

    # The lines containing the energies of the occupied states
    occ_energies_last_line = []
    # The lines containing the energies of the unoccupied states
    unocc_energies_last_line = []
    # Set the total energy to zero
    total_energy = 0.0

    for i in range(0, len(lines)):
        # Find the occupied states lines
        if 'Eigenvalues of the occupied subspace'.lower() in lines[i].lower():

            if str(spin) in lines[i].lower().split():
                lines_with_occupied.append(i)
                for j in range(i, len(lines)):
                    # Find the last line number containing the energies for occupied states
                    if len(lines[j].split()) == 0:
                        occ_energies_last_line.append(j)
                        break
        # Find the unoccupied states lines
        if 'Eigenvalues of the unoccupied subspace'.lower() in lines[i].lower():
            if str(spin) in lines[i].lower().split():
                lines_with_unoccupied.append(i)
                for j in range(i, len(lines)):
                    # Find the last line number containing the energies for unoccupied states
                    if len(lines[j].split()) == 0:
                        unocc_energies_last_line.append(j)
                        break

        if "Total energy:" in lines[i]:
            total_energy = float(lines[i].split()[2])

    # The Kohn-Sham energies
    ks_energies = []
    # Start after two lines of the 'Eigenvalues of the occupied subspace'
    for i in range(lines_with_occupied[time] + 2, occ_energies_last_line[time] - 1):
        for j in range(0, len(lines[i].split())):
            ks_energies.append(float(lines[i].split()[j]))

    # Start after two lines of the 'Eigenvalues of the unoccupied subspace'
    for i in range(lines_with_unoccupied[time] + 2, unocc_energies_last_line[time]):
        if not 'Reached'.lower() in lines[i].lower().split():
            for j in range(0, len(lines[i].split())):
                ks_energies.append(float(lines[i].split()[j]))

    # Now appending all the energies into a numpy array
    ks_energies = np.array(ks_energies)

    # Returning the energeis from min_band to max_band
    return ks_energies[min_band - 1:max_band], total_energy


def read_energies_from_cp2k_md_log_file(params):
    """
    This function reads the energies from CP2K molecular dynamics log file for a specific spin. The difference between this function
    and the function `read_energies_from_cp2k_log_file` is that it is more efficient and faster for molecular dynamics energy levels.

    Args:

        params (dictionary):

            logfile_name (str): The output file produced by CP2K

            init_time (int): The initial time step of the molecular dynamics.

            final_time (int): The final time step of the molecular dynamics.

            min_band (int): The minimum state number.

            max_band (int): The maximum state number.

            spin (int): This number can only accept two values: 1 for alpha and 2 for beta spin.

    Returns:

        E (1D numpy array): The vector consisting of the KS energies from min_band to max_band.

        total_energy (float): The total energy obtained from the log file.

    """

    # Critical parameters
    critical_params = ["logfile_name", "min_band", "max_band"]
    # Default parameters
    default_params = {"spin": 1, "init_time": 0, "final_time": 1}
    # Check input
    comn.check_input(params, default_params, critical_params)

    cp2k_log_file_name = params["logfile_name"]
    # Time step, for molecular dynamics it will read the energies
    # of time step 'time', but for static calculations the time is set to 0
    init_time = params["init_time"]
    final_time = params["final_time"]

    # Koh-Sham orbital indicies
    # ks_orbital_indicies = params["ks_orbital_indicies"]

    # The minimum state number
    min_band = params["min_band"]  # ks_orbital_indicies[0]
    # The maximum state number
    max_band = params["max_band"]  # ks_orbital_indicies[-1]

    spin = params["spin"]

    # First openning the file and reading its lines
    f = open(cp2k_log_file_name, 'r')
    lines = f.readlines()
    f.close()

    # The lines containing 'Eigenvalues of the occupied subspace'
    lines_with_occupied = []
    # The lines containing 'Eigenvalues of the unoccupied subspace'
    lines_with_unoccupied = []

    # The lines containing the energies of the occupied states
    occ_energies_last_line = []
    # The lines containing the energies of the unoccupied states
    unocc_energies_last_line = []
    # Set the total energy to zero
    total_energy = 0.0

    for i in range(0, len(lines)):
        # Find the occupied states lines
        if 'Eigenvalues of the occupied subspace'.lower() in lines[i].lower():

            if str(spin) in lines[i].lower().split():
                lines_with_occupied.append(i)
                for j in range(i, len(lines)):
                    # Find the last line number containing the energies for occupied states
                    if len(lines[j].split()) == 0:
                        occ_energies_last_line.append(j)
                        break
        # Find the unoccupied states lines
        if 'Eigenvalues of the unoccupied subspace'.lower() in lines[i].lower():
            if str(spin) in lines[i].lower().split():
                lines_with_unoccupied.append(i)
                for j in range(i, len(lines)):
                    # Find the last line number containing the energies for unoccupied states
                    if len(lines[j].split()) == 0:
                        unocc_energies_last_line.append(j)
                        break

        if "Total energy:" in lines[i]:
            total_energy = float(lines[i].split()[2])

    # All KS energies
    KS_energies = []
    for time in range(init_time, final_time):
        # The Kohn-Sham energies
        ks_energies = []
        # Start after two lines of the 'Eigenvalues of the occupied subspace'
        for i in range(lines_with_occupied[time] + 2, occ_energies_last_line[time] - 1):
            for j in range(0, len(lines[i].split())):
                ks_energies.append(float(lines[i].split()[j]))

        # Start after two lines of the 'Eigenvalues of the unoccupied subspace'
        for i in range(lines_with_unoccupied[time] + 2, unocc_energies_last_line[time]):
            if not 'Reached'.lower() in lines[i].lower().split():
                for j in range(0, len(lines[i].split())):
                    ks_energies.append(float(lines[i].split()[j]))

        # Now appending all the energies into a numpy array
        ks_energies = np.array(ks_energies)
        KS_energies.append(ks_energies[min_band - 1:max_band])

    # Returning the energeis from min_band to max_band
    return KS_energies, total_energy


_warned_smeared_partition = set()


def read_cp2k_electron_count(filename):
    """Electron-count bookkeeping of a CP2K log.

    Returns dict(nelec_printed, nocc_printed, nmo): the "Number of electrons"
    values (one for RKS - the total, two for UKS - alpha then beta), the "Number
    of occupied orbitals" values in the same order, and the number of MOs. Under
    &SMEAR the printed occupied count is CP2K's smeared HOMO - every orbital
    populated beyond EPS_FERMI_DIRAC - while the electron count is exact, so the
    two disagree exactly when a fractionally populated orbital sits above the
    aufbau reference.
    """
    nelec, nocc, nmo = [], [], None
    with open(filename, errors="ignore") as f:
        for line in f:
            if "Number of electrons:" in line:
                nelec.append(int(float(line.split()[3])))
            elif "Number of occupied orbitals:" in line:
                nocc.append(int(float(line.split()[4])))
            elif "Number of molecular orbitals:" in line and nmo is None:
                nmo = int(float(line.split()[4]))
    if not nelec:
        raise ValueError(F"no 'Number of electrons' line in {filename} - not a CP2K log "
                         F"or its SCF section is missing")
    return {"nelec_printed": nelec, "nocc_printed": nocc, "nmo": nmo}


def reference_occupation(filename, isUKS=False):
    """The reference occupation of a CP2K run FROM ITS ELECTRON COUNT.

    RKS: nelec/2 (an odd count raises - no closed-shell reference exists). UKS:
    [n_alpha, n_beta], the per-spin electron counts CP2K prints. The printed
    "Number of occupied orbitals" is cross-checked: a disagreement means a
    fractionally populated orbital above the reference is being counted as
    occupied (&SMEAR with a population above EPS_FERMI_DIRAC) - it is reported
    once per process and the electron-count value is returned, because a fixed
    SD basis and MO window are only ever defined for the aufbau reference.
    """
    ec = read_cp2k_electron_count(filename)
    nelec, nocc_p = ec["nelec_printed"], ec["nocc_printed"]
    if isUKS:
        if len(nelec) < 2:
            raise ValueError(F"{filename}: a UKS reference needs the per-spin electron "
                             F"counts but the log prints {len(nelec)} 'Number of electrons' "
                             F"line(s) - is the run really unrestricted?")
        ref = [nelec[0], nelec[1]]
    else:
        if nelec[0] % 2:
            raise ValueError(F"{filename}: {nelec[0]} electrons cannot form a closed-shell "
                             F"reference - the run is not RKS or the charge is wrong")
        ref = nelec[0] // 2
    for ch, (r, printed) in enumerate(zip(ref if isUKS else [ref], nocc_p)):
        if printed != r and ("partition", ch) not in _warned_smeared_partition:
            _warned_smeared_partition.add(("partition", ch))
            print(F"WARNING in reference_occupation: {filename} counts {printed} occupied "
                  F"orbitals{' (spin ' + 'ab'[ch] + ')' if isUKS else ''} but its electron "
                  F"count gives {r} - an orbital above the reference carries a population "
                  F"beyond EPS_FERMI_DIRAC (&SMEAR). The electron-count reference {r} is "
                  F"used and the printed partition is re-mapped onto it (once per process)")
    return ref


def read_cp2k_mo_occupations(filename):
    """MO occupation numbers from the last 'MO| EIGENVALUES AND OCCUPATION
    NUMBERS' table(s) of a CP2K log (&DFT &PRINT &MO OCCUPATION_NUMBERS T).

    Returns a list of np.ndarray - one for RKS, [alpha, beta] for UKS - or None
    when the log carries no such table. Under &SMEAR these are the fractional
    populations (2 = full for RKS, 1 for UKS); the wfn restart and NAMD_PRINT
    never show them.
    """
    tables, cur, alpha_seen = [], None, False
    with open(filename, errors="ignore") as f:
        for line in f:
            if "EIGENVALUES AND OCCUPATION NUMBERS" in line:
                cur = []
                tables.append(cur)
                alpha_seen = alpha_seen or ("ALPHA" in line)
                continue
            if cur is not None:
                m = re.match(r"\s*MO\|\s+(\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$", line)
                if m:
                    cur.append(float(m.group(4)))
                elif cur and not line.strip().startswith("MO|"):
                    cur = None
    tables = [np.asarray(x) for x in tables if x]
    if not tables:
        return None
    return tables[-2:] if alpha_seen and len(tables) >= 2 else tables[-1:]


def repartition_newtonx(nx, nocc_ref, theta=0.01, occupations=None, nelec=None):
    """Bring NAMD_PRINT data onto the electron-count reference occupation.

    CP2K prints its occupied/virtual partition from the smeared HOMO. When that
    is wider than the reference (orbitals above it populated beyond
    EPS_FERMI_DIRAC), the extra orbitals are moved to the virtual block (energy
    order kept) and the excitation vectors are re-mapped: excitations FROM the
    moved orbitals have no place in an integer-occupation basis - their weight
    is returned per state as dropped weight (to be combined into the capture
    metric) - and excitations INTO them are absent from the smeared vectors
    (zero rows). A partition narrower than the reference raises: an orbital the
    reference occupies is missing, the charge or multiplicity changed.
    With occupation numbers given, populations are checked against theta (as a
    fraction of the full occupation): the total must equal the electron count,
    and any orbital above the reference populated beyond theta, or below it
    depleted beyond theta, is a fractional reference - it raises, because no
    integer-occupation SD basis represents it.
    Args:
        nx (dict): read_cp2k_newtonx_output result (a modified copy is returned)
        nocc_ref (int or [int, int]): the electron-count reference
        theta (float): population tolerance, fraction of the full occupation
        occupations (list of np.ndarray or None): read_cp2k_mo_occupations
        nelec (list of int or None): per-channel electron counts for the sum check
    Returns:
        (nx_new, drops, moved): drops (np.ndarray, per state) the weight lost to
        excitations from moved orbitals, moved a list of (channel, [orbitals])
    """
    nx = dict(nx)
    is_uks = nx.get("nspin", 1) == 2 and nx.get("X_b") is not None
    refs = list(nocc_ref) if is_uks else [int(nocc_ref)]
    channels = [("", refs[0])] + ([("_b", refs[1])] if is_uks else [])
    maxocc = 1.0 if is_uks else 2.0
    drops = np.zeros(len(nx["X"]))
    moved = []
    for ic, (s, nref) in enumerate(channels):
        Xs = nx["X" + s]
        Co, Cv = nx.get("C_occ" + s), nx.get("C_virt" + s)
        nocc_p = Co.shape[1] if Co is not None else np.asarray(Xs[0]).shape[1]
        tag = " (spin " + "ab"[ic] + ")" if is_uks else ""
        if nocc_p < nref:
            raise ValueError(
                F"NAMD_PRINT lists {nocc_p} occupied orbitals{tag} but the electron count "
                F"requires {nref}: an orbital the reference occupies is missing - the charge "
                F"or multiplicity changed, nothing downstream is valid")
        if occupations is not None:
            occ = np.asarray(occupations[ic] if ic < len(occupations) else occupations[0])
            if nelec is not None:
                ne = nelec[ic] if ic < len(nelec) else nelec[0]
                # an electron count is an integer - half an electron separates any two
                if abs(float(occ.sum()) - float(ne)) > 0.5:
                    raise ValueError(
                        F"the MO occupation numbers sum to {occ.sum():.4f} electrons{tag} but "
                        F"the log counts {ne} - the printed occupations do not describe this "
                        F"electron count")
            above = [(i + 1, float(occ[i])) for i in range(nref, len(occ)) if occ[i] > theta * maxocc]
            below = [(i + 1, float(occ[i])) for i in range(nref) if occ[i] < (1.0 - theta) * maxocc]
            if above or below:
                raise ValueError(
                    F"fractional reference{tag}: orbital(s) {[i for i, _ in above]} above the "
                    F"electron-count reference {nref} carry populations "
                    F"{[round(v, 3) for _, v in above]} and/or occupied orbital(s) "
                    F"{[i for i, _ in below]} are depleted to {[round(v, 3) for _, v in below]} "
                    F"(full = {maxocc:g}, occupation_threshold = {theta}). No integer-"
                    F"occupation SD basis represents this - the frontier orbitals share the "
                    F"electron pair (&SMEAR at a near-degeneracy). Raise occupation_threshold "
                    F"only if that population is genuinely negligible for the dynamics")
        if nocc_p == nref:
            continue
        k = nocc_p - nref
        moved.append((("alp", "bet")[ic] if is_uks else "alp", list(range(nref + 1, nocc_p + 1))))
        if Co is not None:
            nx["C_occ" + s] = Co[:, :nref]
            nx["C_virt" + s] = np.hstack([Co[:, nref:], Cv]) if Cv is not None else Co[:, nref:]
        for key in ("E_occ", "phases_occ"):
            v = nx.get(key + s)
            if v is not None:
                v = np.asarray(v)
                vk = key.replace("occ", "virt") + s
                nx[vk] = np.concatenate([v[nref:], np.asarray(nx[vk])]) if nx.get(vk) is not None else v[nref:]
                nx[key + s] = v[:nref]
        new = []
        for ist, M in enumerate(Xs):
            M = np.asarray(M)
            tot2 = float(np.sum(M ** 2))
            lost2 = float(np.sum(M[:, nref:] ** 2))
            drops[ist] = 1.0 - (1.0 - drops[ist]) * (1.0 - (lost2 / tot2 if tot2 > 0 else 0.0))
            R = np.zeros((M.shape[0] + k, nref))
            R[k:, :] = M[:, :nref]
            new.append(R)
        nx["X" + s] = new
    return nx, drops, moved


def read_homo_index(filename: str, isUKS: bool = False):
    """
    This function extract the HOMO index from CP2K log file. This index starts from 1.
    Args:

        filename (string): The full path to the log file.

        isUKS (bool): Return 2 indexes for alpha and beta.

    Returns:

        homos (list): The HOMO indexes (the number of occupied orbitals) in the form of [alpha_homo, beta_homo]
        or
        homo (int) The HOMO index (in the case of spin-restricted calculations)
    """
    # the value is the ELECTRON-COUNT reference - the printed 'occupied' count is
    # only cross-checked (under &SMEAR it is the smeared HOMO, see
    # reference_occupation)
    return reference_occupation(filename, isUKS=isUKS)


def read_molog_file(filename: str):
    """
    This function reads the coefficiets of the molecular orbitals printed out
    during the MD or a single-point calculation for a structure using CP2K.
    The format of the MO coefficients in CP2K is in column format so the coeffiecients
    are written in a column. You can check this for an MOLog file printed out by CP2K.
    The number of columns in MOLog file is based on the DFT%PRINT%MO%NDIGITS. The larger the
    number of digits (NDIGITS), the lower number of columns. This function is written in a way
    that will automatically extract all the coefficients for all the eigenvectors and returns them
    and their energies. It will also designed for reading the coefficients for MOs in different
    K-points if used by user in the CP2K input (DFT%KPOINT).

    Args:

        filename (string): The name of the MOLog ile.

    Returns:

        mo_coeffs (list): The list containing the MO coefficients for each K-point.

        mo_energies (liest): The list containing the energies of MOs for each K-point.

    """
    # first is there any k-point or not??
    file = open(filename, 'r')
    lines = file.readlines()
    file.close()

    is_kpoint = False
    # The K-point will show itself at the beginning
    # so since the files are large we only search the first
    # couple of lines, say first 10
    for i in range(len(lines)):
        if 'K-POINT' in lines[i]:
            is_kpoint = True
            break

    # set a timer
    timer = time.time()
    if is_kpoint:
        # If K-point
        print('Found K-point, will proceed reading the coefficients',
              'for each K-POINT...')
        # Lines with K-POINT
        kpoint_lines = []
        # Lines with Fermi energy
        fermi_lines = []
        for i in range(len(lines)):
            if 'K-POINT'.lower() in lines[i].lower():
                kpoint_lines.append(i)
            if 'Fermi'.lower() in lines[i].lower():
                fermi_lines.append(i)
        # All the coefficients
        mo_coeffs = []
        # All the energies
        mo_energies = []
        # Loop over all the K-points
        for i in range(len(kpoint_lines)):
            # start lines
            startl = kpoint_lines[i]
            if i == len(kpoint_lines) - 1:
                # end line
                endl = len(lines)
            else:
                endl = kpoint_lines[i + 1]
            # All the eigenvectors for each K-point
            eigenvectors = []
            # Lines with length less than or equal to 4
            leq_4_lines = []
            # Energies for a K-point
            energies = []
            # Find the leq_4_lines indicies
            for j in range(startl, endl):
                # lines with less than or equal to 4
                tmp = lines[j].split()
                if 0 < len(tmp) <= 4:
                    if 'Fermi' not in tmp and 'gap' not in tmp:
                        leq_4_lines.append(j)
            # The lines with MO number
            mo_num_lines = leq_4_lines[0::3]
            # The lines wit energies
            energy_lines = leq_4_lines[1::3]
            # Find the energies for a K-point
            for j in energy_lines:
                tmp = lines[j].split()
                for k1 in range(4):
                    try:
                        energy = float(tmp[k1])
                        energies.append(energy)
                    except BaseException:
                        pass
            # Append the energies for this K-point
            mo_energies.append(np.array(energies))
            # Occupation lines, Needed to define the start and end
            # lines to find the coefficients
            occ_lines = leq_4_lines[2::3]
            # Now reading the coefficients
            for j in range(len(mo_num_lines)):
                start_l = occ_lines[j] + 1
                if j == len(mo_num_lines) - 1:
                    end_l = fermi_lines[i]
                else:
                    end_l = mo_num_lines[j + 1]
                # This part with try and except make the code
                # to consider for different number of columns
                # So the user does not need to specify the number
                # of columns or the number of atomic orbitals etc...
                for k1 in range(4, 8):
                    # Each eigenvector for this K-point
                    eigenvector = []
                    for k2 in range(start_l, end_l):
                        tmp = lines[k2].split()
                        try:
                            eig_val = float(tmp[k1])
                            eigenvector.append(eig_val)
                        except BaseException:
                            pass
                    if len(eigenvector) != 0:
                        # Now this variable contains all the eigenvectors
                        # for this K-point
                        eigenvectors.append(np.array(eigenvector))

            print('Done reading coefficients for K-point %d' % (i + 1))
            # The mo_coeffs contains the eigenvectors. After this
            # we start for another K-point.
            mo_coeffs.append(eigenvectors)

    else:
        # The same procedure as above for K-point with this
        # difference that there is no K-point
        fermi_lines = []
        for i in range(len(lines)):
            if 'Fermi'.lower() in lines[i].lower():
                fermi_lines.append(i)

        mo_coeffs = []
        mo_energies = []
        # start lines
        startl = 3
        endl = fermi_lines[0]
        eigenvectors = []
        leq_4_lines = []
        energies = []
        for j in range(startl, endl):
            # lines with less than or equal to 4
            tmp = lines[j].split()
            if 0 < len(tmp) <= 4:
                if 'Fermi' not in tmp and 'gap' not in tmp:
                    leq_4_lines.append(j)

        mo_num_lines = leq_4_lines[0::3]
        energy_lines = leq_4_lines[1::3]
        for j in energy_lines:
            tmp = lines[j].split()
            for k1 in range(4):
                try:
                    energy = float(tmp[k1])
                    energies.append(energy)
                except BaseException:
                    pass
        mo_energies.append(np.array(energies))
        occ_lines = leq_4_lines[2::3]
        for j in range(len(mo_num_lines)):
            start_l = occ_lines[j] + 1
            if j == len(mo_num_lines) - 1:
                end_l = fermi_lines[0]
            else:
                end_l = mo_num_lines[j + 1]
            for k1 in range(4, 8):
                eigenvector = []
                for k2 in range(start_l, end_l):
                    tmp = lines[k2].split()
                    try:
                        eig_val = float(tmp[k1])
                        eigenvector.append(eig_val)
                    except BaseException:
                        pass
                if len(eigenvector) != 0:
                    eigenvectors.append(np.array(eigenvector))

        print('Done reading coefficients for the MOLog file')
        mo_coeffs.append(eigenvectors)
    print('Elapsed time:', time.time() - timer)

    return mo_energies, mo_coeffs


def read_ao_matrices(filename):
    """
    This function reads the files printed by CP2K for atomic orbital matrices data.
    If the AO_MATRICES keyword is activated in CP2K input, then all the specified
    matrices in that section will be appened in one file. This function reads all the
    matrices and data.
    Args:
        filename (string): The name of the file that contains AO matrices data.
    Returs:
        data (list): A list that contains the AO matrices with the same order as in the file.
    """

    file = open(filename, 'r')
    lines = file.readlines()
    file.close()

    mat_lines = []
    data_lines = []
    for i in range(len(lines)):
        tmp = lines[i].split()
        if 'MATRIX' in lines[i]:
            mat_lines.append(i)
            # print(lines[i])
        if 0 < len(tmp) <= 4 and 'MATRIX' not in lines[i]:
            data_lines.append(i)
    data_lines = np.array(data_lines)
    properties = []
    data = []
    for i in range(len(mat_lines)):
        print(lines[mat_lines[i]])
        properties.append(lines[mat_lines[i]].split()[0])
        data_0 = []
        start_line = mat_lines[i]
        if i == len(mat_lines) - 1:
            end_line = len(lines) - 1
        else:
            end_line = mat_lines[i + 1]
        # print(start_line, end_line)
        # and data_lines.all()<end_line)
        ind = np.where(np.logical_and(start_line < data_lines, data_lines < end_line))
        # print(ind[0])
        # print(data_lines[ind])
        for j in range(len(ind[0])):
            start_line_1 = int(data_lines[int(ind[0][j])])
            if j == len(ind[0]) - 1:
                end_line_1 = int(end_line)
            else:
                end_line_1 = int(data_lines[int(ind[0][j + 1])])
            # print(lines[start_line_1])
            # print(lines[end_line_1])
            for k1 in range(4):
                tmp = []
                try:
                    for k2 in range(start_line_1 + 1, end_line_1):
                        # print(lines[k2])
                        if len(lines[k2].split()) > 0:
                            # print(lines[k2])
                            tmp.append(float(lines[k2].split()[4 + k1]))

                    data_0.append(tmp)
                except BaseException:
                    pass
        data.append(np.array(data_0))

    return properties, data


def extract_coordinates(trajectory_xyz_file_name: str, time_step: int):
    """
    This function reads the trajectory xyz file and extract the coordinates of a
    time step.

    Args:

        trajcetory_xyz_file_name (string): The trajectory xyz file name.

        time_step (integer): The time step.

    Returns:

        coordinates (list): The list of the x, y, and z coordinates.
    """
    # Reading the file and its lines
    file = open(trajectory_xyz_file_name, 'r')
    lines = file.readlines()
    file.close()
    # number of atoms obtianed from the first line of
    # the xyz file
    natoms = int(lines[0].split()[0])
    # the coordinates list
    coordinates = []
    # start lines
    start_l = time_step * (natoms + 2)
    # end line
    end_l = (time_step + 1) * (natoms + 2)
    for i in range(start_l + 2, end_l):
        tmp = lines[i].split()
        name = tmp[0]
        # turn Angstrom into Bohr unit
        x = float(tmp[1]) * units.Angst
        y = float(tmp[2]) * units.Angst
        z = float(tmp[3]) * units.Angst
        coordinates.append([name, x, y, z])

    return coordinates


def find_basis_set(basis_set_files_path: list, unique_atoms: list, basis_set_names: list):
    """
    This function searches the atoms basis sets for a set of unique atoms in
    a set of basis sets files and reads them in a form so that we can create
    a shell for atomic orbital overlap computation using libint.

    Args:

        basis_set_files_path (list): A list containing the full path to the
                                     basis sets files (like BASIS_MOLOPT).
        unique_atoms (list): The list of unique atoms.

        basis_set_names (list): A list containing the name of the basis sets used
                                for each of the unique atoms. The length of this
                                list should be the same as unique_atoms list.

    Returns:

        basis_set_data (list): This lists contains the information of all the basis
                               sets. They include, the angular momentum values,
                               the exponents, and the contraction coefficients.
    """
    # The variables for appending the l_values,
    # exponents, and contraction coefficients
    l_vals = []
    exponents = []
    coeffs = []
    basis_set_data = []
    # loop over all the basis set files
    for i in range(len(basis_set_files_path)):
        file = open(basis_set_files_path[i], 'r')
        lines = file.readlines()
        file.close()
        # find the basis set for each atom in unique_atoms
        for j in range(len(unique_atoms)):
            # initializing the empty lists for each atom
            exp_atom = []
            coeff_atom = []
            l_vals_atom = []
            data_atom = []
            for k in range(len(lines)):
                tmp = lines[k].lower().split()
                # we use a try and except so that if it couldn't find the name it doesn't crash
                try:
                    if (unique_atoms[j].lower()) == tmp[0] and basis_set_names[j].lower() == tmp[1]:
                        print('Found basis set %s for atom %s in %s' % (basis_set_names[j],
                                                                        unique_atoms[j], basis_set_files_path[i]))
                        print('Reading the data for atom %s in the basis file' % unique_atoms[j])
                        # start reading the basis set
                        # number of basis set
                        num_basis = int(lines[k + 1])
                        # Lines with information about the basis set
                        # These lines are like this
                        # 3 0 1 5 2 2
                        # their order is like this:
                        # principal quantum number, minimum angular momentum value
                        # maximum angular momentum value, number of exponents
                        # the number of contractions for each of the angular momentum values
                        info_lines = []
                        info_lines.append(k + 2)
                        if num_basis > 1:
                            c = 0
                            while len(info_lines) < num_basis:
                                tmp_1 = lines[k + 2 + c].split()
                                n_exponent = int(tmp_1[3])
                                c += n_exponent + 1
                                info_lines.append(k + 2 + c)
                        for p in info_lines:
                            exp = []
                            coeffs = []
                            l_min = int(lines[p].split()[1])
                            l_max = int(lines[p].split()[2])
                            n_l = l_max - l_min + 1
                            l_vals = []
                            c = 1
                            l_start = l_min
                            if l_min == l_max:
                                l_end = l_min + 1
                            else:
                                l_end = l_max + 1
                            for l_val in range(l_start, l_end):
                                l_val_rep = int(lines[p].split()[3 + c])
                                c += 1
                                for pp in range(l_val_rep):
                                    l_vals.append(l_val)
                            n_exponent = int(lines[p].split()[3])
                            start_l = p + 1
                            end_l = p + n_exponent + 1
                            for p1 in range(start_l, end_l):
                                exp.append(float(lines[p1].split()[0]))
                            c = 0
                            for p2 in range(len(l_vals)):
                                c += 1
                                coeffs = []
                                for p1 in range(start_l, end_l):
                                    coeffs.append(float(lines[p1].split()[c]))
                                data_atom.append([l_vals[p2], exp, coeffs])

                except BaseException:
                    pass
            if len(data_atom) != 0:
                # The basis_set data appends all the data for this atom
                basis_set_data.append([unique_atoms[j], data_atom])

    return basis_set_data


def make_shell(coordinates: list, basis_set_data: list, is_spherical: bool):
    """
    This function makes a libint shell from the coordinates, basis_set_data from
    the find_basis_set function. The sphecrical or cartesian coordinates flags define
    the basis it needs to work in.
    """
    # setting up a counter for initializing the shell using
    # liblibra_core.initialize_shell function
    c = 0
    # for each of the atoms
    for i in range(len(coordinates)):
        atom_name = coordinates[i][0]
        # making the coordinates into a VECTOR type to be able to use in C++
        coords_init = [coordinates[i][1], coordinates[i][2], coordinates[i][3]]
        a = VECTOR(coords_init[0], coords_init[1], coords_init[2])
        # for each of the atoms types basis set data
        for j in range(len(basis_set_data)):
            # check if the atom type is for that
            # specific atom in the coordinates
            if atom_name == basis_set_data[j][0]:
                for k in range(len(basis_set_data[j][1])):
                    # l_value
                    l_val = basis_set_data[j][1][k][0]
                    # exponents
                    exp = Py2Cpp_double(list(basis_set_data[j][1][k][1]))
                    # contraction coefficients
                    coeff = Py2Cpp_double(list(basis_set_data[j][1][k][2]))
                    if c == 0:
                        # if it is the first atom and
                        # the first basis set initialize the libint shell
                        shell = initialize_shell(l_val, is_spherical, exp, coeff, a)
                    else:
                        # for all other atoms
                        # add to the created shell
                        add_to_shell(shell, l_val, is_spherical, exp, coeff, a)
                    # add 1 to the counter
                    c += 1

    return shell


def resort_molog_eigenvectors(l_vals, purity=None):
    """
    This function returns the resotring indices for resoting the MOLog
    eigenvectors according to this order:

    MOLog order (example for Cd atom):
    2s, 3s | 3py, 3pz, 3px | 4py, 4pz, 4px | 4d-2, 4d-1, 4d0, 4d+1, 4d+2 | 5d-2, 5d-1, 5d0, 5d+1, 5d+2 | ...
    However, the atomic orbital overla computed from the libint version of Psi4 is not ordered
    as above. The ordering is like this:
    2s, 3s | 3pz, 3px, 3py | 4pz, 4px, 4py | 4d0, 4d+1, 4d-1, 4d+2, 4d-2 | 5d0, 5d+1, 5d-1, 5d+2, 5d-2 | ...
    Therefore, we need to resort the eigenvectors to be able to use the code properly.

    Args:

        l_vals (list): A list containing the angular momentum values for atoms
                       in the order of the MOLog files.

    Returns:

        new_indices (numpy array): The new indices that needs to be used for reordering.

    """
    # new indices
    new_indices = []
    # setting up the counter
    c = 0
    # loop over all the angular momentum values
    for i in range(len(l_vals)):
        l_val = l_vals[i]
        # find the reorder indices needed for this l_val
        if purity is not None and l_val >= 1 and not purity.get(l_val, True):
            # cartesian shells arrive already in the lexicographic canonical
            # order (molden_methods resorts them in one stage) - identity
            reordered_indices = np.arange((l_val + 1) * (l_val + 2) // 2)
        else:
            reordered_indices = index_reorder(l_val)
        for j in range(len(reordered_indices)):
            # now append it by plus the counter since
            # we aim to do it for all the eigenvectors
            # and l values
            new_indices.append(c + reordered_indices[j])
        # increase the counter with t
        c += len(reordered_indices)
    # Return the new indices
    return new_indices


def index_reorder(l_val):
    """
    This function returns the new ordering of the angular momentum value based on the
    order used in Psi4. Detailed explanation was given for the resort_molog_eigenvectors function.

    Args:

        l_val (integer): The angular momentum value.

    Returns:

        new_order (numpy array): The new order of the indices for the l_val.

    """

    # for s orbital
    if l_val == 0:
        new_order = [1]
    # for p orbital
    elif l_val == 1:
        new_order = [2, 3, 1]
    # for d orbital
    elif l_val == 2:
        new_order = [3, 4, 2, 5, 1]
    # for f orbital
    elif l_val == 3:
        new_order = [4, 5, 3, 6, 2, 7, 1]
    # for g orbital
    elif l_val == 4:
        new_order = [5, 6, 4, 7, 3, 8, 2, 9, 1]

    # The indeices
    return np.array(new_order) - 1


def generate_translational_vectors(origin, N, periodicity_type):
    """
    This function generates the translational vectors for periodic systems.
    For monolayers the generated vectors does not add the orthogonal axis but
    for bulk all directions are added.

    Args:
        origin (list): The translational vectors are obtained with respect to this origin.
        N (list): An array that contains the number of cells to be considered in
                     each of the X, Y, and Z directions.
        periodicity_type (string): The periodicity type. It can only get these values:
                                   'XY', 'XZ', and 'YZ' for monolayers and 'XYZ' for bulk systems.

    Returns:
        translational_vectors (numpy array): The translational vectors for that system.

    """
    # Basis vectors
    i_vec = np.array([1, 0, 0])
    j_vec = np.array([0, 1, 0])
    k_vec = np.array([0, 0, 1])
    # origin
    origin = np.array(origin)
    translational_vectors = []
    x_range = [0]
    y_range = [0]
    z_range = [0]
    Nx = N[0]
    Ny = N[1]
    Nz = N[2]
    if 'x' in periodicity_type.lower():
        x_range = range(-Nx, Nx + 1)
    if 'y' in periodicity_type.lower():
        y_range = range(-Ny, Ny + 1)
    if 'z' in periodicity_type.lower():
        z_range = range(-Nz, Nz + 1)

    for n_i in x_range:
        for n_j in y_range:
            for n_k in z_range:
                vec = origin + n_i * i_vec + n_j * j_vec + n_k * k_vec
                if not np.array_equal(vec, origin):
                    translational_vectors.append(vec)

    return np.array(translational_vectors)


def molog_lvals(filename: str):
    """
    This function returns all the angular momentum values in the order
    the eigenvectors are written. Unlike the molden files we need to extract
    this from the molog files.

    Args:

        filename (string): The MOLog file name.

    Returns:

        l_vals_all (list): The list for all angular momentum values.
    """
    # Opening the file and reading all the lines
    file = open(filename, 'r')
    lines = file.readlines()
    file.close()
    # all l_values including the
    # ones which are repeated as well.
    l_vals = []
    # principal quantum number
    # this will be used to distinguish
    # between the l_values we want to read.
    p_nums = []
    # Lines with length less than or equal
    # to one which are the molecular orbital
    # number, their energies, and their occupation numbers
    leq_4_lines = []
    for i in range(3, len(lines)):
        tmp = lines[i].split()
        if len(tmp) < 4 and len(tmp) != 0 and 'Fermi' not in lines[i] and 'MO' not in lines[i]:
            leq_4_lines.append(i)
    # We only need for one part in the MOLog file
    start_l = leq_4_lines[2]
    end_l = leq_4_lines[3]
    for i in range(start_l, end_l):
        tmp = lines[i].split()
        try:
            if 's' in tmp[3]:
                # n is the principal quantum number
                n = int(tmp[3][0])
                p_nums.append(n)
                l_vals.append(0)
            if 'p' in tmp[3]:
                n = int(tmp[3][0])
                p_nums.append(n)
                l_vals.append(1)
            if 'd' in tmp[3]:
                n = int(tmp[3][0])
                p_nums.append(n)
                l_vals.append(2)
            if 'f' in tmp[3]:
                n = int(tmp[3][0])
                p_nums.append(n)
                l_vals.append(3)
            if 'g' in tmp[3]:
                n = int(tmp[3][0])
                p_nums.append(n)
                l_vals.append(4)
        except BaseException:
            pass
    # initializing the l_vals_all
    l_vals_all = []
    # Now we need to get rid of the repeated l_values
    # Below will do it all!
    for i in range(len(l_vals)):
        if i == 1:
            l_vals_all.append(l_vals[0])
        if i != 1 and i != (len(l_vals) - 1):
            if p_nums[i] != p_nums[i - 1]:
                l_vals_all.append(l_vals[i])
            if p_nums[i] == p_nums[i - 1] and l_vals[i] != l_vals[i - 1]:
                l_vals_all.append(l_vals[i])

    return l_vals_all


def cp2k_xtb_ot_inp(ot_inp_temp: str, traj_xyz_filename: str, step: int):
    """
    This function gets the xTB orbital transformation (OT) input template and the trajectrory xyz file name
    and makes a new input based on the time step. The aim of this input is to obtain a converged
    wfn RESTART file and use it for diagonalization so that we can print out the MOLog or molden files.

    Args:

        ot_inp_temp (string): The name of the OT input template.

        traj_xyz_filename (string): The name of the trajectory xyz file name.

        step (integer): The time step.

    Returns:

        None
    """

    file = open(ot_inp_temp)
    lines = file.readlines()
    file.close()

    read_trajectory_xyz_file(traj_xyz_filename, step)

    file = open('xtb_ot_step_%d.inp' % step, 'w')

    for i in range(len(lines)):
        if 'PROJECT_NAME'.lower() in lines[i].lower():
            file.write('PROJECT_NAME OT_%d\n' % step)
        elif 'COORD_FILE_NAME'.lower() in lines[i].lower():
            file.write('COORD_FILE_NAME coord-%d.xyz\n' % step)
        elif 'WFN_RESTART_FILE_NAME'.lower() in lines[i].lower():
            file.write('WFN_RESTART_FILE_NAME OT_%d-RESTART.wfn\n' % (step - 1))
        elif 'SCF_GUESS'.lower() in lines[i].lower():
            file.write('SCF_GUESS RESTART\n')
        else:
            file.write(lines[i])

    file.close()


def cp2k_xtb_diag_inp(diag_inp_temp: str, step: int, wfn_restart_file=None, traj_xyz_filename=None):
    """
    This function gets the xTB diagonalization input template
    and makes a new input based on the time step. By default this input will
    use the wfn RESTART file name obtained from the OT calclulations.

    Args:

        diag_inp_temp (string): The name of the diagonalization input template.

        step (integer): The time step.

        wfn_restart_file (string): The wfn file the diagonalization restarts
                                   from. None keeps the OT-chain default,
                                   'OT_{step}-RESTART.wfn' (the same-step OT
                                   run's output); OT-free workflows pass e.g.
                                   'Diag_{step-1}-RESTART.wfn' to chain the
                                   diagonalization runs directly (SCF_GUESS
                                   RESTART falls back to the atomic guess
                                   while the file does not exist yet - that
                                   is how the first step boots).

        traj_xyz_filename (string): When given, extract coord-{step}.xyz from
                                    this trajectory here. The two-stage flow
                                    leaves this None because cp2k_xtb_ot_inp
                                    already wrote the coord file; without the
                                    OT stage the diag generator must extract
                                    it itself.

    Returns:

        None
    """

    file = open(diag_inp_temp)
    lines = file.readlines()
    file.close()

    if wfn_restart_file is None:
        wfn_restart_file = 'OT_%d-RESTART.wfn' % step
    if traj_xyz_filename is not None:
        read_trajectory_xyz_file(traj_xyz_filename, step)

    file = open('xtb_diag_step_%d.inp' % step, 'w')

    for i in range(len(lines)):
        if 'PROJECT_NAME'.lower() in lines[i].lower():
            file.write('PROJECT_NAME Diag_%d\n' % step)
        elif 'COORD_FILE_NAME'.lower() in lines[i].lower():
            file.write('COORD_FILE_NAME coord-%d.xyz\n' % step)
        elif 'WFN_RESTART_FILE_NAME'.lower() in lines[i].lower():
            file.write('WFN_RESTART_FILE_NAME %s\n' % wfn_restart_file)
        elif 'SCF_GUESS'.lower() in lines[i].lower():
            file.write('SCF_GUESS RESTART\n')
        elif 'FILENAME'.lower() in lines[i].lower():
            file.write('FILENAME libra\n')
        else:
            file.write(lines[i])

    file.close()


def run_cp2k_xtb(params):
    """
    This function is for running the CP2K for xTB inputs.

    Args:

        params (dictionary): A dictionary containing the following parameters

          * **cp2k_ot_input_template** (string): The CP2K OT input template file name.
            None (or absent) skips the OT stage entirely: the diagonalization run
            then restarts from the previous step's 'Diag_{step-1}-RESTART.wfn'
            (falling back to the atomic guess while that file does not exist yet,
            i.e. at the first step) and extracts the coordinates itself. The
            two-stage OT+diag flow is a density warm start the minimal-basis xTB
            SCF does not need - dropping it halves the CP2K launches per step.
            The diag template must then carry full standalone SCF settings
            (EPS_SCF, MAX_SCF, mixing if needed) rather than assume a converged
            restart.

          * **ot_first_call_only** (bool): with an OT template given, run the
            OT stage only where the diagonalization chain starts cold (the
            previous step's 'Diag_{step-1}-RESTART.wfn' does not exist - the
            first frame of a job slice); warm frames skip straight to the
            chained diagonalization [default: False]

          * **cp2k_diag_input_template** (string): The CP2K diagonalization input template file name.

          * **trajectory_xyz_filename** (string): The trajectory xyz file name.

          * **step** (integer): The time step.

          * **cp2k_exe** (string): The full path to CP2K executable.

          * **nprocs** (integer): Number of processors (MPI ranks).
          * **omp_num_threads** (integer): OMP threads per rank [default: 1].
            nprocs=1 skips the MPI launcher; for hybrid runs (nprocs > 1 with
            omp_num_threads > 1) add binding flags to mpi_executable
            ("--map-by slot:PE=N" or "--bind-to none") - Open MPI's default
            bind-to-core pins each rank's OMP threads onto a single core.

    """

    print('**************** Running CP2K ****************')
    ot_input_template = params.get('cp2k_ot_input_template', None)
    # OT only where the SCF starts cold: with a warm diagonalization chain
    # (the previous step's restart exists) the OT warm start adds nothing
    if (ot_input_template is not None
            and params.get('ot_first_call_only', False)
            and os.path.exists('Diag_%d-RESTART.wfn' % (params['step'] - 1))):
        ot_input_template = None
    diag_input_template = params['cp2k_diag_input_template']
    trajectory_xyz_filename = params['trajectory_xyz_filename']
    step = params['step']
    cp2k_exe = params['cp2k_exe']
    mpi_executable = params['mpi_executable']
    nprocs = params['nprocs']
    # threads per MPI rank - without an explicit value many psmp builds default
    # to ALL cores per rank, oversubscribing the node for nprocs > 1
    omp = int(params.get('omp_num_threads', 1))

    if ot_input_template is not None:
        # Run OT; the diag run restarts from the same-step OT wfn (the
        # generator's default) and cp2k_xtb_ot_inp extracts coord-{step}.xyz
        print('Step', step, 'Computing the OT method wfn file...')
        cp2k_xtb_ot_inp(ot_input_template, trajectory_xyz_filename, step)
        t1 = time.time()
        execute_cp2k(cp2k_exe, F'xtb_ot_step_{step}.inp', F'OUT-ot_{step}.log',
                     omp_num_threads=omp, mpi_executable=mpi_executable,
                     nprocs=nprocs)
        print('Done with OT wfn. Elapsed time:', time.time() - t1)
        diag_wfn_restart, diag_traj = None, None
    else:
        # OT-free: chain the diagonalization runs directly; SCF_GUESS RESTART
        # falls back to the atomic guess while the file does not exist yet
        # (the first step) - the same way the OT chain itself boots
        print('Step', step, 'No OT template - chaining diagonalization restarts')
        diag_wfn_restart = 'Diag_%d-RESTART.wfn' % (step - 1)
        diag_traj = trajectory_xyz_filename

    # Run diagonalization
    t1 = time.time()
    print('Computing the wfn file using diagonalization method...')
    cp2k_xtb_diag_inp(diag_input_template, step, diag_wfn_restart, diag_traj)
    execute_cp2k(cp2k_exe, F'xtb_diag_step_{step}.inp', F'OUT-diag_{step}.log',
                 omp_num_threads=omp, mpi_executable=mpi_executable,
                 nprocs=nprocs)
    print('Done with diagonalization. Elapsed time:', time.time() - t1)


def distribute_cp2k_libint_jobs(
        submit_template: str,
        run_python_file: str,
        istep: int,
        fstep: int,
        njobs: int,
        run_slurm: bool,
        submission_exe='sbatch'):
    """
    This function distributes the jobs to perform CP2K calculations and computing and saving the MO overlaps.

    Args:

        submit_template (string): The name of the submit template to submit a job.

        run_python_file (string): The name of the python file for running the run_cp2k_xtb_step2 function which
                                  contains the parameters.

        istep (integer): The initial step of the MD trajectory.

        fstep (integer): The final step of the MD trajectory.

        njobs (integer): The number of jobs.

        run_slurm (bool): The flag for running the computations either as bash or submitting through sbatch.

        submission_exe (string): The submission executable. For slurm envs, it is 'sbatch' and for pbs envs, it is 'qsub'.

    Return:
        None: but performs the action
    """

    file = open(run_python_file, 'r')
    lines = file.readlines()
    file.close()

    # nsteps_job = int((fstep-istep)/njobs)
    nsteps_job = np.linspace(istep, fstep, njobs + 1, endpoint=True, dtype=int).tolist()
    for njob in range(njobs):
        # istep_job = njob*nsteps_job+istep
        # fstep_job = (njob+1)*nsteps_job+istep+1
        istep_job = nsteps_job[njob]
        fstep_job = nsteps_job[njob + 1] + 1
        # if njob==(njobs-1):
        #    fstep_job = fstep

        print('Submitting job', njob + 1)
        print('Job', njob, 'istep', istep_job, 'fstep', fstep_job, 'nsteps', fstep_job - istep_job)

        if os.path.exists('job%d' % (njob + 1)):
            os.system('rm -rf job%d' % (njob + 1))
        os.system('mkdir job%d' % (njob + 1))
        os.chdir('job%d' % (njob + 1))
        os.system('cp ../%s %s' % (submit_template, submit_template))
        file = open('run.py', 'w')

        for i in range(len(lines)):
            if 'istep' in lines[i]:
                file.write("params['istep'] = %d\n" % istep_job)
            elif 'fstep' in lines[i]:
                file.write("params['fstep'] = %d\n" % fstep_job)
            else:
                file.write(lines[i])
        file.close()
        if run_slurm:
            os.system('%s %s' % (submission_exe, submit_template))
        else:
            # Just in case you want to use a bash file and not submitting
            os.system('sh %s' % submit_template)
        print('Submitted job', njob)
        os.chdir('../')
        # run_cp2k_xtb_step2(params)


def cp2k_find_excitation_energies(filename):
    """
    This function finds and extracts excitation energies and oscillator strengths
    from the .log files of the TD-DFPT calculations in CP2k

    Args:
        file ( string ): name of the file to read

    Returns:
        (list, list):

           * the first list contains the energies of excited states
           * the second list contains the oscillator strengths

    """

    f = open(filename, "r")
    A = f.readlines()
    f.close()

    E, F = [], []
    for a in A:
        tmp = a.split()

        if len(tmp) == 7:
            if tmp[0] == "TDDFPT|":
                istate = int(float(tmp[1]))
                e = float(tmp[2])
                f = float(tmp[6])

                E.append(e)
                F.append(f)

    return E, F
    
    
def parse_spectrum_data_from_log(file_path):
    """Parse energy levels and intensities from a CP2K log file.

    Args:
        file_path (str): Path to the log file.

    Returns:
        tuple: Arrays of energy levels and intensities.
    """
    energy_levels, intensities = [], []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith(' TDDFPT|') and len(line.split()) >= 7:
                parts = line.split()
                try:
                    energy_levels.append(float(parts[2]))
                    intensities.append(float(parts[-1]))
                except ValueError:
                    continue
    return np.array(energy_levels), np.array(intensities)



def gaussian_function(a, mu, sigma, num_points, x_min, x_max):
    """
    This function computes the value of a Gaussian function for a set of points.
    Args:
        a (float): The prefactor
        mu (float): The average value, \\mu
        sigma (float): The standard deviation, \\sigma
        num_points (integer): The number of points in the grid
        x_min (float): The minimum value of x in the grid
        x_max (float): The maximum value of x in the grid
    Returns:
        x (nparray): The grid
        gaussian_fun (nparray): The computed values of Gaussian function in the grid points, x.
    """
    pre_fact = (a / sigma) / (np.sqrt(2 * np.pi))
    x = np.linspace(x_min, x_max, num_points)
    x_input = np.array((-1 / 2) / (np.square(sigma)) * np.square(x - mu))
    gaussian_fun = pre_fact * np.exp(x_input)

    return x, gaussian_fun


def gaussian_function_vector(a_vec, mu_vec, sigma, num_points, x_min, x_max):
    """
    This function computes the Gaussian function for a set of grid sets using `gaussian_function` and sums them together.
    Args:
        a_vec (list): A list containing the values of prefactors
        mu_vec (list): A list containing the values of averages
        sigma (float): The standard deviation
        num_points (integer): The number of points in the grid
        x_min (float): The minimum value of x in the grid
        x_max (float): The maximum value of x in the grid
    Returns:
        x (nparray): The grid
        sum_vec (nparray): The summed values of the Gaussian functions for grids.
    """
    for i in range(len(a_vec)):
        if i == 0:
            sum_vec = np.zeros(num_points)
        x, conv_vec = gaussian_function(a_vec[i], mu_vec[i], sigma, num_points, x_min, x_max)
        sum_vec += conv_vec
    return x, sum_vec


def aux_pdos(
        c1,
        atoms,
        pdos_files,
        margin,
        homo_occ,
        orbitals_cols,
        sigma,
        npoints,
        ave_pdos_convolved_all,
        orbitals,
        labels):
    """
    c1 - index of the atom kinds
    atoms - the
    pdos_files (list of strings): names of the files containing pdos data
    shift (double): the amount of "padding" around the minimal and maximal values of the energy scale [units: eV]
    homo_occ (double): occupancy of the HOMO: 2 - for non-polarized, 1 - for polarized
    orbital_cols (list of lists of integers): integers that define which columns correspond to which types of orbitals
    sigma (double): broadening of the pDOS
    npoints (int): how many points to use to compte the convolved pDOS

    """

    pdos_ave = np.zeros(np.loadtxt(pdos_files[0]).shape)

    cnt = len(pdos_files)
    for c2, pdos_file in enumerate(pdos_files):
        pdos_mat = np.loadtxt(pdos_file)
        if c2 == 0:
            pdos_ave = np.zeros(pdos_mat.shape)
        pdos_ave += pdos_mat
    pdos_ave /= cnt

    pdos_ave[:, 1] *= units.au2ev
    e_min = np.min(pdos_ave[:, 1]) - margin
    e_max = np.max(pdos_ave[:, 1]) + margin
    homo_level = np.max(np.where(pdos_ave[:, 2] == homo_occ))
    homo_energy = pdos_ave[:, 1][homo_level]
    # labels = []

    for c3, orbital_cols in enumerate(orbitals_cols):
        try:
            sum_pdos_ave = np.sum(pdos_ave[:, orbital_cols], axis=1)
            ave_energy_grid, ave_pdos_convolved = gaussian_function_vector(
                sum_pdos_ave, pdos_ave[:, 1], sigma, npoints, e_min, e_max)
            ave_pdos_convolved_all.append(ave_pdos_convolved)
            pdos_label = F"{atoms[1][c1]}, {orbitals[c3]}"
            labels.append(pdos_label)
        except BaseException:
            pass

    return ave_energy_grid, homo_energy, ave_pdos_convolved_all, labels


def pdos(params):
    """
    This function computes the weighted pdos for a set of CP2K pdos files.
    Args:
        params (dict):
            * path_to_all_pdos (str): The full path to the pdos files produced by CP2K
            * atoms (list): This list contains the atom types number and names in a separate list.
                            CP2K produces the pdos files in *k{i}*.pdos files where `i` is the atom type number.
                            e.g.: [ [1,2] , ['Ti', 'O'] ]
            * orbitals (list): A list containing the atomic angular momentum.
            * orbital_cols (list): The columns for specific angular momentum as in `orbitals`
                            e.g. For orbitals of ['s', 'p',        'd',         'f']
                                 [                [3], range(4,7), range(7,12), range(12,19)  ]
            * npoints (integer): The number of grid points for convolution with Gaussian functions.
            * sigma (float): The standard deviation value
            * shift (float): The amount of shifting the grid from the minimum and maximum energy values.
            * is_spin_polarized (int): 0 - non-spin-polarized, 1 - spin-polarized
    Returns:
        ave_energy_grid (nparray): The average energy grid for all pdos files
        homo_energy (float): The value of HOMO energy in eV
        ave_pdos_convolved_all (li st): The list containing all the values
        labels (lis t): A list containing the labels of the orbital resolved pdos
                       `ave_pdos_convolved_all` is sorted based on the labels in this list
        ave_pdos_convolved_total (nparray): The total density of states
    """
    # Critical parameters
    critical_params = ["path_to_all_pdos", "atoms"]
    # Default parameters
    default_params = {"orbitals_cols": [[3], range(4, 7), range(7, 12), range(12, 19)],
                      "orbitals": ['s', 'p', 'd', 'f'],
                      "sigma": 0.05, "shift": 2.0, "npoints": 2000, "is_spin_polarized": 0
                      }
    # Check input
    comn.check_input(params, default_params, critical_params)

    path_to_all_pdos = params['path_to_all_pdos']
    atoms = params['atoms']
    orbitals_cols = params['orbitals_cols']
    orbitals = params['orbitals']
    npoints = params['npoints']
    sigma = params['sigma']
    shift = params['shift']
    is_spin_polarized = params["is_spin_polarized"]

    if is_spin_polarized == 0:  # non-polarized case
        homo_occ = 2.0
        labels = []
        ave_pdos_convolved_all = []

        for c1, i in enumerate(atoms[0]):
            # Finding all the pdos files
            pdos_files = glob.glob(path_to_all_pdos + F'/*k{i}*.pdos')

            ave_energy_grid, homo_energy, ave_pdos_convolved_all, labels = aux_pdos(
                c1, atoms, pdos_files, shift, homo_occ, orbitals_cols, sigma, npoints, ave_pdos_convolved_all, orbitals, labels)

            """
            # Finding all the pdos files
            pdos_files = glob.glob(path_to_all_pdos+F'/*k{i}*.pdos')
            pdos_ave = np.zeros(np.loadtxt(pdos_files[0]).shape)

            cnt = len(pdos_files)
            for c2, pdos_file in enumerate(pdos_files):
                pdos_mat = np.loadtxt(pdos_file)
                if c2==0:
                    pdos_ave = np.zeros(pdos_mat.shape)
                pdos_ave += pdos_mat
            pdos_ave /= cnt
            pdos_ave[:,1] *= units.au2ev
            e_min = np.min(pdos_ave[:,1])-shift
            e_max = np.max(pdos_ave[:,1])+shift
            homo_level = np.max(np.where(pdos_ave[:,2]==2.0))
            homo_energy = pdos_ave[:,1][homo_level]
            for c3, orbital_cols in enumerate(orbitals_cols):
                try:
                    sum_pdos_ave = np.sum(pdos_ave[:,orbital_cols],axis=1)
                    ave_energy_grid, ave_pdos_convolved = gaussian_function_vector(sum_pdos_ave, pdos_ave[:,1], sigma,
                                                                                   npoints, e_min, e_max)
                    ave_pdos_convolved_all.append(ave_pdos_convolved)
                    pdos_label = atoms[1][c1]+F', {orbitals[c3]}'
                    labels.append(pdos_label)
                except:
                    pass
            """

        ave_pdos_convolved_all = np.array(ave_pdos_convolved_all)
        ave_pdos_convolved_total = np.sum(ave_pdos_convolved_all, axis=0)

        return ave_energy_grid, homo_energy, ave_pdos_convolved_all, labels, ave_pdos_convolved_total

    else:  # spin-polarized case
        homo_occ = 1.0

        labels_alp, labels_bet = [], []
        ave_pdos_convolved_all_alp, ave_pdos_convolved_all_bet = [], []

        for c1, i in enumerate(atoms[0]):
            # Finding all the pdos files
            pdos_files_alp = glob.glob(path_to_all_pdos + F'/*ALPHA*k{i}*.pdos')
            pdos_files_bet = glob.glob(path_to_all_pdos + F'/*BETA*k{i}*.pdos')

            ave_energy_grid_alp, homo_energy_alp, ave_pdos_convolved_all_alp, labels_alp = aux_pdos(
                c1, atoms, pdos_files_alp, shift, homo_occ, orbitals_cols, sigma, npoints, ave_pdos_convolved_all_alp, orbitals, labels_alp)

            ave_energy_grid_bet, homo_energy_bet, ave_pdos_convolved_all_bet, labels_bet = aux_pdos(
                c1, atoms, pdos_files_bet, shift, homo_occ, orbitals_cols, sigma, npoints, ave_pdos_convolved_all_bet, orbitals, labels_bet)

        ave_pdos_convolved_all_alp = np.array(ave_pdos_convolved_all_alp)
        ave_pdos_convolved_total_alp = np.sum(ave_pdos_convolved_all_alp, axis=0)

        ave_pdos_convolved_all_bet = np.array(ave_pdos_convolved_all_bet)
        ave_pdos_convolved_total_bet = np.sum(ave_pdos_convolved_all_bet, axis=0)

        return ave_energy_grid_alp, homo_energy_alp, ave_pdos_convolved_all_alp, labels_alp, ave_pdos_convolved_total_alp, \
            ave_energy_grid_bet, homo_energy_bet, ave_pdos_convolved_all_bet, labels_bet, ave_pdos_convolved_total_bet


def exc_analysis(params):
    """
    This function computes the average configuration interaction coefficients for excited states (originally written by Brendan Smith).
    Args:
        params (dict):
            * path_to_logfiles (string): The full path to all logfiles
            * number_of_states (integer): The number of excited states appeared in the excitation analysis of CP2K outputs
            * tolerance (float): the tolerance value for CI coefficients
            * nsds (integer): The number of SDs to be considered for plotting
            * isUKS (integer): A flag for considering restricted or unrestricted spin calculations:
                               - 0 : No UKS
                               - 1 : With UKS
    Returns:
        coeffs_avg (list): The average CI coefficients
        coeffs_error (list): The error bars for average CI coefficients
    """
    # Critical parameters
    critical_params = ["path_to_logfiles", "number_of_states"]
    # Default parameters
    default_params = {"isUKS": 0, "ci_threshold": 0.01, "nsds": 2}
    # Check input
    comn.check_input(params, default_params, critical_params)

    path_to_logfiles = params['path_to_logfiles']
    nsds = params['nsds']
    nstates = params['number_of_states']

    logfiles = glob.glob(F"{path_to_logfiles}/*.log")
    params.update({'logfile_name': ''})

    ci_coeffs = []
    for logfile in logfiles:
        params.update({"logfile_name": logfile})
        info, data = read_cp2k_tddfpt_log_file(params)
        excitation_energies, ci_basis_raw, ci_coefficients_raw_unnorm, spin_components = data[0], data[1], data[2], data[3]

        ci_coefficients_raw_norm = step2_many_body.normalize_ci_coefficients(ci_coefficients_raw_unnorm)

        for j in range(len(ci_coefficients_raw_norm)):
            for k in range(len(ci_coefficients_raw_norm[j])):
                ci_coefficients_raw_norm[j][k] = ci_coefficients_raw_norm[j][k]**2
        ci_coeffs.append(ci_coefficients_raw_norm)

    nsteps = len(ci_coeffs)

    coeffs = []
    coeffs_avg = []
    coeffs_error = []

    for state in range(nstates):
        coeffs.append([])
        coeffs_avg.append([])
        coeffs_error.append([])

        for sd in range(nsds):

            coeffs[state].append([])
            coeffs_avg[state].append([])
            coeffs_error[state].append([])

            for step in range(nsteps):
                if len(ci_coeffs[step][state]) < nsds and sd > len(ci_coeffs[step][state]) - 1:
                    coeffs[state][sd].append(0.0)
                else:
                    coeffs[state][sd].append(ci_coeffs[step][state][sd])

            mb_coeff_avg, mb_coeff_std = data_stat.scalar_stat(coeffs[state][sd])
            coeffs_avg[state][sd].append(mb_coeff_avg)
            coeffs_error[state][sd].append(1.96 * mb_coeff_std / np.sqrt(nsteps))

    return coeffs_avg, coeffs_error


def extract_energies_sparse(params):
    """
    This function extract the energies vs time data from scipy.sparse files
    Args:
        params (dict):
            * path_to_energy_files (string): The full path to sparse energy files
            * dt (float): The time step
            * prefix (string): The prefix of the files
            * suffix (string): The suffix of the files
            * istep (integer): The initial step
            * fstep (integer): The final step
    Returns:
        md_time (nparray): Contains the md time and can be used for plotting
        energies (nparray): The energies of each state
    """
    critical_params = ["path_to_energy_files"]
    # Default parameters
    default_params = {"dt": 1.0, "prefix": "Hvib_sd_", "suffix": "_re", "istep": 0, "fstep": 5}
    # Check input
    comn.check_input(params, default_params, critical_params)

    path_to_energy_files = params['path_to_energy_files']
    dt = params['dt']
    prefix = params['prefix']
    suffix = params['suffix']
    istep = params['istep']
    fstep = params['fstep']

    energies = []
    for step in range(istep, fstep):
        file = F'{path_to_energy_files}/{prefix}{step}{suffix}.npz'
        energies.append(np.diag(sp.load_npz(file).todense().real))
    energies = np.array(energies)
    md_time = np.arange(0, energies.shape[0] * dt, dt)

    return md_time, energies


def read_wfn_file(filename):
    """
    This function reads the data stored in wfn files produced by CP2K.
    These data are stored in this format (from the cp2k/src/qs_mo_io.F file):

    From the subroutine 'write_mo_set_low':
    ### ==================================================
    WRITE (ires) natom, nspin, nao, nset_max, nshell_max
    WRITE (ires) nset_info
    WRITE (ires) nshell_info
    WRITE (ires) nso_info

    DO ispin = 1, nspin
         nmo = mo_array(ispin)%nmo
         IF ((ires > 0) .AND. (nmo > 0)) THEN
            WRITE (ires) nmo, &
               mo_array(ispin)%homo, &
               mo_array(ispin)%lfomo, &
               mo_array(ispin)%nelectron
            WRITE (ires) mo_array(ispin)%eigenvalues(1:nmo), &
               mo_array(ispin)%occupation_numbers(1:nmo)
         END IF
         IF (PRESENT(rt_mos)) THEN
            DO imat = 2*ispin - 1, 2*ispin
               CALL cp_fm_write_unformatted(rt_mos(imat), ires)
            END DO
         ELSE
            CALL cp_fm_write_unformatted(mo_array(ispin)%mo_coeff, ires)
         END IF
      END DO
    ### ==================================================
    SUMMARY:
    natom, nspin, nao, nset_max, nshell_max, nset_info, nshell_info, nso_info
    for ispin in range(nspins):
        nmo, homo, lfomo, nelectron, (eigenvalues(1:nmo), occupation_numbers(1:nmo)), mo_coeff

    Args:
        filename (string): The name of the wfn file

    Returns:
        basis_data (list): A list that contains the basis set data
        spin_data (list): A list containing the alpha and beta spin data
        eigen_vals_and_occ_nums (list): A list that contains the energies and occupation numbers.
    """
    wfn_file = io.FortranFile(filename, 'r')
    basis_data = []
    natom, nspin, nao, nset_max, nshell_max = wfn_file.read_ints()
    nset_info = wfn_file.read_ints()
    nshell_info = wfn_file.read_ints()
    nso_info = wfn_file.read_ints()
    basis_data.append(natom)
    basis_data.append(nspin)
    basis_data.append(nao)
    basis_data.append(nset_max)
    basis_data.append(nshell_max)
    basis_data.append(nset_info)
    basis_data.append(nshell_info)
    basis_data.append(nso_info)
    spin_data = []
    eigen_vals_and_occ_nums = []
    mo_coeffs = []
    for spin in range(nspin):
        spin_tmp = []
        nmo, homo, lfomo, nelectron = wfn_file.read_ints()
        spin_tmp.append(nmo)
        spin_tmp.append(homo)
        spin_tmp.append(lfomo)
        spin_tmp.append(nelectron)
        spin_data.append(spin_tmp)
        eigen_val_and_occ_num = wfn_file.read_reals(dtype=np.float64)
        eigen_vals_and_occ_nums.append(eigen_val_and_occ_num)
        mo_coeff = []
        for mo in range(nmo):
            coeffs = wfn_file.read_reals()
            mo_coeff.append(coeffs)
        mo_coeffs.append(mo_coeff)
    wfn_file.close()
    return basis_data, spin_data, eigen_vals_and_occ_nums, mo_coeffs


def write_wfn_file(filename, basis_data, spin_data, eigen_vals_and_occ_nums, mo_coeffs):
    """
    This function writes a set of data including basis sets, spin data, eigenvalues and occupation numbers, and
    MO coefficients into a binary file (wfn) readable by CP2K.
    Args:
        filename (string): The name of the output wfn file.
        basis_data (list): A list that contains the basis set data
        spin_data (list): A list containing the alpha and beta spin data
        eigen_vals_and_occ_nums (list): A list that contains the energies and occupation numbers.
        mo_coeffs (numpy array): The numpy array that contains the MO coefficients.
    Returns:
        None
    """
    wfn_file = io.FortranFile(filename, 'w')
    natom = basis_data[0]
    nspin = basis_data[1]
    nao = basis_data[2]
    nset_max = basis_data[3]
    nshell_max = basis_data[4]
    nset_info = basis_data[5]
    nshell_info = basis_data[6]
    nso_info = basis_data[7]
    wfn_file.write_record(natom, nspin, nao, nset_max, nshell_max)
    wfn_file.write_record(nset_info)
    wfn_file.write_record(nshell_info)
    wfn_file.write_record(nso_info)
    for spin in range(nspin):
        wfn_file.write_record(spin_data[spin][0], spin_data[spin][1], spin_data[spin][2], spin_data[spin][3])
        nmo = spin_data[spin][0]
        wfn_file.write_record(eigen_vals_and_occ_nums[spin])  # , dtype=np.float32)
        # zero_eig_occ = np.zeros(eigen_vals_and_occ_nums[spin].shape)
        # wfn_file.write_record(zero_eig_occ)#, dtype=np.float32)
        for mo in range(nmo):
            wfn_file.write_record(mo_coeffs[spin][mo])
            # zero_coeffs = np.random.random(mo_coeffs[spin][mo].shape)
            # wfn_file.write_record(zero_coeffs)
    wfn_file.close()


def compute_energies_coeffs(ks_mat, overlap):
    """
    This function solves the general eigenvalue problem described above using a Cholesky decomposition
    of the overlap matrix. The eigenvalues are sorted.
    More information: https://doi.org/10.1016/j.cpc.2004.12.014
    Args:
        ks_mat (numpy array): The Kohn-Sham matrix
        overlap (numpy array): The atomic orbital overlap matrix
    Returns:
        eigenvalues (numpy array): The energies (eigenvalues)
        eigenvectors (numpy array): The MO coefficients
    """
    # Cholesky decomposition of the overlap matrix
    U = np.linalg.cholesky(overlap).T
    # One ca also use the following as well but it is computationally more demanding
    # U = scipy.linalg.fractional_matrix_power(S, 0.5)
    U_inv = np.linalg.inv(U)
    UT_inv = np.linalg.inv(U.T)
    K_prime = np.linalg.multi_dot([UT_inv, ks_mat, U_inv])
    # the symmetric solver: eig leaves (near-)degenerate subspaces
    # non-orthogonal, breaking V S V^T = I downstream
    eigenvalues, eigenvectors = np.linalg.eigh(K_prime)
    # Transform back the coefficients
    eigenvectors = np.dot(U_inv, eigenvectors)
    sorted_indices = np.argsort(eigenvalues)
    eigenvectors = eigenvectors[:, sorted_indices].T

    return eigenvalues[sorted_indices], eigenvectors


def compute_energies_coeffs_scipy(ks_mat, overlap):
    """
    This function solves the general eigenvalue problem described above using a Cholesky decomposition
    of the overlap matrix. The eigenvalues are sorted.
    More information: https://doi.org/10.1016/j.cpc.2004.12.014
    Args:
        ks_mat (numpy array): The Kohn-Sham matrix
        overlap (numpy array): The atomic orbital overlap matrix
    Returns:
        eigenvalues (numpy array): The energies (eigenvalues)
        eigenvectors (numpy array): The MO coefficients
    """
    # Cholesky decomposition of the overlap matrix; scipy returns the
    # UPPER factor (S = U^T U) - numpy returns the lower, hence no .T here
    U = scipy.linalg.cholesky(overlap)
    # One ca also use the following as well but it is computationally more demanding
    # U = scipy.linalg.fractional_matrix_power(S, 0.5)
    U_inv = scipy.linalg.inv(U)
    UT_inv = scipy.linalg.inv(U.T)
    # K_prime = scipy.linalg.multi_dot( [UT_inv, ks_mat, U_inv] )
    K_prime = UT_inv @ ks_mat @ U_inv
    # the symmetric solver, as in compute_energies_coeffs
    eigenvalues, eigenvectors = scipy.linalg.eigh(K_prime)
    # Transform back the coefficients
    eigenvectors = U_inv @ eigenvectors
    sorted_indices = np.argsort(eigenvalues)
    eigenvectors = eigenvectors[:, sorted_indices].T

    return eigenvalues[sorted_indices], eigenvectors


def compute_density_matrix(eigenvectors, homo_index):
    """
    This function computes the density matrix: P=2\times c_{occ}\times c_{occ}^T
    Args:
        eigenvectors (numpy array): The set of eigenvectors
        homo_index (integer): The HOMO index to select only the occupied orbitals
    Returns:
        density_mat (numpy array): The density matrix.
    """
    occupied_eigenvectors = eigenvectors[0:homo_index]
    density_mat = 2 * np.dot(occupied_eigenvectors.T, occupied_eigenvectors)
    return density_mat


def compute_convergence(ks_mat, overlap, density_mat):
    """
    This function computes the convergence error based on the commutator relation: e=KPS-SPK
    More information: https://doi.org/10.1016/j.cpc.2004.12.014
    At this point, this function gives not exact results as in CP2K and the results seems to be different.
    TODO: Need to check this in more details by checking the CP2K source code.
    """
    return np.linalg.multi_dot([ks_mat, density_mat, overlap]) - np.linalg.multi_dot([overlap, density_mat, ks_mat])


def scf_timing(log_filename):
    """
    This function returns the timing data related to CP2K log file.
    Args:
        log_filename (string): The name of the logfile
    Returns:
        ncycle (int): Number of SCF cycles in the log file
        convergence (numpy array): The convergence value at each SCF step
        timing (float): The total time of the SCF cycle
    """
    f = open(log_filename, 'r')
    lines = f.readlines()
    f.close()
    for i in range(len(lines)):
        if '*** SCF run ' in lines[i] or 'Leaving inner SCF loop' in lines[i]:
            end = i
        elif 'Step' in lines[i] and 'Convergence' in lines[i]:
            start = i + 2
    timing = 0
    ncycle = 0
    convergence = []
    for i in range(start, end):
        tmp = lines[i].split()
        if len(tmp) > 0:
            ncycle += 1
            timing += float(tmp[3])
            convergence.append(float(tmp[4]))
    return ncycle, np.array(convergence), timing


def resort_ao_matrices(l_vals):
    """
    This function returns the resotring indices for resoting the atomic orbital
    matrices from Psi4 to CP2K order according to this order:

    AO matrices order (example for Cd atom):
    2s, 3s | 3py, 3pz, 3px | 4py, 4pz, 4px | 4d-2, 4d-1, 4d0, 4d+1, 4d+2 | 5d-2, 5d-1, 5d0, 5d+1, 5d+2 | ...
    However, the atomic orbital overla computed from the libint version of Psi4 is not ordered
    as above. The ordering is like this:
    2s, 3s | 3pz, 3px, 3py | 4pz, 4px, 4py | 4d0, 4d+1, 4d-1, 4d+2, 4d-2 | 5d0, 5d+1, 5d-1, 5d+2, 5d-2 | ...
    Therefore, we need to resort the eigenvectors to be able to use the code properly.

    Args:

        l_vals (list): A list containing the angular momentum values for atoms
                       in the order of the MOLog files.

    Returns:

        new_indices (numpy array): The new indices that needs to be used for reordering.

    """
    # new indices
    new_indices = []
    # setting up the counter
    c = 0
    # loop over all the angular momentum values
    for i in range(len(l_vals)):
        l_val = l_vals[i]
        # find the reorder indices needed for this l_val
        reordered_indices = index_reorder_v2(l_val)
        for j in range(len(reordered_indices)):
            # now append it by plus the counter since
            # we aim to do it for all the eigenvectors
            # and l values
            new_indices.append(c + reordered_indices[j])
        # increase the counter with t
        c += len(reordered_indices)
    # Return the new indices
    return new_indices


def index_reorder_v2(l_val):
    """
    This function returns the new ordering of the angular momentum value based on the
    order used in Psi4. Detailed explanation was given for the resort_ao_matrices function.

    Args:

        l_val (integer): The angular momentum value.

    Returns:

        new_order (numpy array): The new order of the indices for the l_val.

    """

    # for s orbital
    if l_val == 0:
        new_order = [1]
    # for p orbital
    elif l_val == 1:
        new_order = [3, 1, 2]
    # for d orbital
    elif l_val == 2:
        new_order = [5, 3, 1, 2, 4]
    # for f orbital
    elif l_val == 3:
        new_order = [7, 5, 3, 1, 2, 4, 6]
    # for g orbital
    elif l_val == 4:
        new_order = [9, 7, 5, 3, 1, 2, 4, 6, 8]

    # The indeices
    return np.array(new_order) - 1


def atom_components_cp2k(filename):
    """
    This function finds the components of the angular momentum of each atom
    in the system and returns a list that can be used to find specific blocks
    in the atomic orbital matrices such as Hamiltonians or overlaps.
    Args:
        filename (string): The name of the AO matrices file
    Returns:
        indices (list): The list of indices related to the angular momentum
                        components of each atom in the system.
    """
    f = open(filename, 'r')
    lines = f.readlines()
    f.close()
    atoms = []
    for i in range(len(lines)):
        tmp = lines[i].split()
        if len(tmp) > 4:
            try:
                atom = int(tmp[1])
                if len(atoms) == 0:
                    atoms.append(atom)
                else:
                    if atom >= atoms[-1]:
                        atoms.append(atom)
                    else:
                        break
            except BaseException:
                pass
    np.unique(atoms).shape
    indices = []
    for i in np.unique(atoms):
        index = list(np.where(atoms == i)[0])
        indices.append(index)
    return indices


def compute_mo_overlap(params):
    r"""
    This function computes the molecular orbital overlap calculations between
    two geometries: $\langle\Phi_{i,R}\|\Phi_{j,R'}\rangle$. It takes a dictionary as parameters.
    Args:
        params (dictionary):
            molden_file_1 (string): The path to the first geometry molden file
            molden_file_2 (string): The path to the second geometry molden file
            is_spherical (bool): The spherical coordinate boolean flga
            nprocs (integer): The number of pocessors
    Returns:
        mo_overlap_matrix (numpy array): The overlap between molecular orbitals
    """
    # Create Libint integral shells for each molden file
    shell_1, l_vals_1 = molden_methods.molden_file_to_libint_shell(params['molden_file_1'], params['is_spherical'])
    shell_2, l_vals_2 = molden_methods.molden_file_to_libint_shell(params['molden_file_2'], params['is_spherical'])
    new_indices = resort_molog_eigenvectors(
        l_vals_1, molden_methods.molden_purity(params['molden_file_1'], l_vals_1))
    eig_vect_1, energies_1 = molden_methods.eigenvectors_molden(params['molden_file_1'], nbasis(shell_1), l_vals_1)
    eig_vect_2, energies_2 = molden_methods.eigenvectors_molden(params['molden_file_2'], nbasis(shell_2), l_vals_2)
    
    # The new eigenvectors
    resortted_eig_vect_1 = []
    resortted_eig_vect_2 = []
    for j in range(len(eig_vect_1)):
        # the new and sorted eigenvector
        eigenvector1 = eig_vect_1[j]
        eigenvector1 = eigenvector1[new_indices]
        eigenvector2 = eig_vect_2[j]
        eigenvector2 = eigenvector2[new_indices]
        # append it to the eigenvectors list
        resortted_eig_vect_1.append(eigenvector1)
        resortted_eig_vect_2.append(eigenvector2)
    resortted_eig_vect_1 = np.array(resortted_eig_vect_1)
    resortted_eig_vect_2 = np.array(resortted_eig_vect_2)
    ao_matrix = compute_overlaps(shell_1, shell_2, params['nprocs'])
    ao_matrix = data_conv.MATRIX2nparray(ao_matrix)
    mo_overlap_matrix = np.linalg.multi_dot([resortted_eig_vect_1, ao_matrix, 
                                             resortted_eig_vect_2.T])
    return mo_overlap_matrix




def read_cp2k_newtonx_output(filename):
    """
    Parse the Newton-X interface file written by &TDDFPT &PRINT &NAMD_PRINT
    (with PRINT_VIRTUALS T): <project>-CP2K_NEWTONX-1_0.inp.

    Contains, at ~1e-12 precision and WITHOUT amplitude thresholding:
      - per state: the FULL MO-basis excitation matrix X (nvirt x nocc,
        L2-normalized - the metrically correct CI expansion; the log's
        excitation-analysis amplitudes are printed in a different metric)
      - per state: PHASES ES EIGENVECTORS (nocc values)
      - OCCUPIED/VIRTUAL MO COEFFICIENTS (nao x nmo, CP2K MOLog AO order)
        and MO EIGENVALUES

    UKS layout (the writer, qs_tddfpt2_restart.F, is spin-generic): the ES
    EIGENVECTORS blocks alternate alpha/beta WITHIN each state, the MO
    coefficient blocks come once per spin, and the scalar blocks (eigenvalues,
    phases) are concatenated alpha-then-beta. The spin channels are detected
    from the MO blocks (their per-spin occupied counts differ from a naive
    split), so UKS parsing requires PRINT_VIRTUALS T.

    Returns:
        dict: X (list of np.ndarray (nvirt, nocc) per state),
              C_occ, C_virt (np.ndarray or None), E_occ, E_virt (np.ndarray or None),
              nspin (1 or 2); for UKS additionally the beta channel as
              X_b, C_occ_b, C_virt_b, E_occ_b, E_virt_b, phases_occ_b, phases_virt_b
    """
    lines = open(filename).readlines()

    def read_triplet_matrix(start):
        rows, i = [], start
        while i < len(lines):
            t = lines[i].split()
            if len(t) == 3 and re.match(r"-?\d+$", t[0]) and re.match(r"-?\d", t[2]):
                rows.append((int(t[0]), int(t[1]), float(t[2])))
            elif rows:
                break
            i += 1
        n_r = max(r for r, _, _ in rows)
        n_c = max(c for _, c, _ in rows)
        M = np.zeros((n_r, n_c))
        for r, c, v in rows:
            M[r - 1, c - 1] = v
        return M, i

    X_all, occ_blocks, virt_blocks = [], [], []
    extra = {}
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if s == "ES EIGENVECTORS":
            M, i = read_triplet_matrix(i + 1)
            X_all.append(M)
            continue
        if s == "OCCUPIED MO COEFFICIENTS":
            M, i = read_triplet_matrix(i + 1)
            occ_blocks.append(M)
            continue
        if s == "VIRTUAL MO COEFFICIENTS":
            M, i = read_triplet_matrix(i + 1)
            virt_blocks.append(M)
            continue
        if s in ("PHASES OCCUPIED ORBITALS", "PHASES VIRTUAL ORBITALS",
                 "OCCUPIED MO EIGENVALUES", "VIRTUAL MO EIGENVALUES"):
            vals, j = [], i + 1
            while j < len(lines):
                t = lines[j].split()
                if len(t) == 1 and re.match(r"-?\d", t[0]):
                    vals.append(float(t[0])); j += 1
                elif vals:
                    break
                else:
                    j += 1
            key = {"PHASES OCCUPIED ORBITALS": "phases_occ",
                   "PHASES VIRTUAL ORBITALS": "phases_virt",
                   "OCCUPIED MO EIGENVALUES": "E_occ",
                   "VIRTUAL MO EIGENVALUES": "E_virt"}[s]
            extra[key] = np.array(vals)
            i = j
            continue
        i += 1
    if not X_all:
        raise ValueError(f"no ES EIGENVECTORS blocks in {filename}")
    if len(occ_blocks) > 2 or len(occ_blocks) != len(virt_blocks):
        raise ValueError(f"{filename}: {len(occ_blocks)} occupied / "
                         f"{len(virt_blocks)} virtual MO blocks - expected "
                         f"matching counts of 1 (restricted) or 2 (UKS)")

    if len(occ_blocks) == 2:
        # UKS: coefficient blocks per spin, ES blocks alternating alpha/beta
        # per state, scalar blocks concatenated alpha-then-beta
        C_occ, C_occ_b = occ_blocks
        C_virt, C_virt_b = virt_blocks
        na, nb = C_occ.shape[1], C_occ_b.shape[1]
        nva = C_virt.shape[1]
        if len(X_all) % 2:
            raise ValueError(f"{filename}: odd number of ES EIGENVECTORS blocks "
                             f"({len(X_all)}) with two spin channels")
        X, X_b = X_all[0::2], X_all[1::2]
        if any(M.shape[1] != na for M in X) or any(M.shape[1] != nb for M in X_b):
            raise ValueError(f"{filename}: ES EIGENVECTORS column counts do not "
                             f"alternate as the per-spin occupied counts "
                             f"({na} alpha / {nb} beta)")
        split = lambda a, n: (None, None) if a is None else (a[:n], a[n:])
        E_occ, E_occ_b = split(extra.get("E_occ"), na)
        E_virt, E_virt_b = split(extra.get("E_virt"), nva)
        p_occ, p_occ_b = split(extra.get("phases_occ"), na)
        p_virt, p_virt_b = split(extra.get("phases_virt"), nva)
    else:
        X = X_all
        C_occ = occ_blocks[0] if occ_blocks else None
        C_virt = virt_blocks[0] if virt_blocks else None
        X_b = C_occ_b = C_virt_b = E_occ_b = E_virt_b = p_occ_b = p_virt_b = None
        E_occ, E_virt = extra.get("E_occ"), extra.get("E_virt")
        p_occ, p_virt = extra.get("phases_occ"), extra.get("phases_virt")
    # the writer UNCONDITIONALLY column-scales X by the occupied-MO phases
    # (qs_tddfpt2_restart.F); undo it so the amplitudes refer to the raw MOs
    # (the same convention as the molden coefficients). Requires PRINT_PHASES T.
    if p_occ is not None:
        for M in X:
            M *= p_occ[np.newaxis, :]                   # phases are +-1
    if X_b is not None and p_occ_b is not None:
        for M in X_b:
            M *= p_occ_b[np.newaxis, :]
    return {"X": X, "C_occ": C_occ, "C_virt": C_virt,
            "E_occ": E_occ, "E_virt": E_virt,
            "phases_occ": p_occ, "phases_virt": p_virt,
            "nspin": 2 if X_b is not None else 1,
            "X_b": X_b, "C_occ_b": C_occ_b, "C_virt_b": C_virt_b,
            "E_occ_b": E_occ_b, "E_virt_b": E_virt_b,
            "phases_occ_b": p_occ_b, "phases_virt_b": p_virt_b}


def read_cp2k_tdfrc(filename):
    """
    Parse a CP2K TDDFPT multi-state force file (*.tdfrc) produced by
    &PROPERTIES &TDDFPT &PRINT &FORCES (mechanism B).

    File layout per state (verified on CP2K 2026.1-dev):
        # STATE NR.   n            E eV       Oscillator strength:  f
              natom
        <blank>
        natom rows of three floats (TOTAL forces = GS + response, Ha/Bohr)

    States screened out by the oscillator-strength THRESHOLD appear with the
    header only (no force block) - they are returned in `energies`/`fosc`
    but absent from `forces`.

    Returns:
        forces (dict): state index (1-based) -> np.ndarray (natom, 3)
        energies (dict): state index -> excitation energy, Hartree
        fosc (dict): state index -> oscillator strength
    """
    ev2Ha = units.ev2Ha
    forces, energies, fosc = {}, {}, {}
    with open(filename) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        m = re.match(r"\s*#\s*STATE NR\.\s+(\d+)\s+(-?[\d.]+)\s*eV\s+Oscillator strength:\s*(-?[\d.]+)",
                     lines[i])
        if not m:
            i += 1
            continue
        state = int(m.group(1))
        energies[state] = float(m.group(2)) * ev2Ha
        fosc[state] = float(m.group(3))
        i += 1
        if i < len(lines) and re.match(r"\s*(\d+)\s*$", lines[i]):
            nat = int(lines[i].split()[0])
            i += 1
            rows = []
            while i < len(lines) and len(rows) < nat:
                parts = lines[i].split()
                if len(parts) == 3:
                    rows.append([float(x) for x in parts])
                i += 1
            arr = np.array(rows)
            if arr.shape != (nat, 3) or not np.all(np.isfinite(arr)):
                raise ValueError(f"Malformed force block for state {state} in {filename}")
            forces[state] = arr
    if not energies:
        raise ValueError(f"No STATE blocks found in {filename}")
    return forces, energies, fosc


def _ao_overlap_blocks_fast(it):
    """Strict single-pass fast path for the regular MOLog block layout:
    header lines carry ONLY column indices, every block prints the same
    number of data rows ("irow iatom Element shell v ..."), and the
    main-width data lines stream from the file iterator STRAIGHT into one
    C-engine np.loadtxt call - no line list, no joined string, so the
    transient is the loadtxt array (~n^2(1+1/k) doubles) plus the result.
    MOLog data lines are fixed-width, so after block 1 establishes the
    length, a single len() comparison classifies the ~99.9% hot path; the
    full string predicates run only on headers, blanks and the rare
    remainder-width block (side-buffered and parsed the same way). A
    region-end line accidentally matching the length reaches loadtxt and
    raises there. Raises on any structural surprise - the caller falls
    back to the flexible parser."""
    import itertools as _itertools
    headers = []                     # column-index lists, in print order
    rem = []                         # data lines of non-main-width blocks
    state = {"kmain": None, "cur": None, "Lmain": None, "L1": None,
             "uniform": True, "prev_max": 0}

    def classified():
        started = False
        for l in it:
            L = state["Lmain"]
            if L is not None and len(l) == L:              # fixed-width data
                if state["cur"] == state["kmain"]:
                    yield l
                else:
                    rem.append(l)
                continue
            ls = l.strip()
            if not ls:
                continue
            if ls.strip(" 0123456789") == "":              # header: int tokens only
                cols = [int(x) for x in ls.split()]
                # MOLog headers are one contiguous column run continuing the
                # previous one - a stray integer line after the region (a
                # count print) would otherwise inflate the matrix silently
                start = 1 if state["kmain"] is None else state["prev_max"] + 1
                if cols != list(range(start, start + len(cols))):
                    raise ValueError("non-contiguous column header")
                state["prev_max"] = cols[-1]
                if state["kmain"] is None:
                    state["kmain"] = len(cols)             # block 1 sets the main width
                elif state["Lmain"] is None and state["uniform"]:
                    state["Lmain"] = state["L1"]           # lock after block 1
                state["cur"] = len(cols)
                headers.append(cols)
                started = True
                continue
            if ls[0].isdigit():
                if not started:
                    raise ValueError("data before the first header")
                if state["Lmain"] is None:                 # inside block 1
                    if state["L1"] is None:
                        state["L1"] = len(l)
                    elif len(l) != state["L1"]:
                        state["uniform"] = False           # stay on full predicates
                if state["cur"] == state["kmain"]:
                    yield l
                else:
                    rem.append(l)
                continue
            if started:
                break                                      # left the matrix region

    g = classified()
    first = next(g, None)            # the first header fixes kmain for usecols
    if first is None:
        raise ValueError("no matrix blocks")
    kmain = state["kmain"]
    arr = np.loadtxt(_itertools.chain([first], g),
                     usecols=(0,) + tuple(range(4, 4 + kmain)),
                     dtype=float, ndmin=2, comments=None)
    if len({len(h) for h in headers}) > 2:                 # main width + remainder
        raise ValueError("irregular header widths")
    n = max(max(h) for h in headers)
    S = np.zeros((n, n))
    main_b = [b for b, h in enumerate(headers) if len(h) == kmain]
    nrow, ragged = divmod(arr.shape[0], len(main_b))
    if ragged:
        raise ValueError("ragged blocks")
    if nrow == 0:
        raise ValueError("header without data rows")
    for j, b in enumerate(main_b):
        blk = arr[j * nrow:(j + 1) * nrow]
        S[np.ix_(blk[:, 0].astype(int) - 1, np.asarray(headers[b]) - 1)] = \
            blk[:, 1:1 + kmain]
    rem_b = [b for b, h in enumerate(headers) if len(h) != kmain]
    if rem_b:
        k2 = len(headers[rem_b[0]])
        arr2 = np.loadtxt(iter(rem), usecols=(0,) + tuple(range(4, 4 + k2)),
                          dtype=float, ndmin=2)
        nrow2, ragged2 = divmod(arr2.shape[0], len(rem_b))
        if ragged2:
            raise ValueError("ragged blocks")
        if nrow2 == 0:
            raise ValueError("header without data rows")
        for j, b in enumerate(rem_b):
            blk = arr2[j * nrow2:(j + 1) * nrow2]
            S[np.ix_(blk[:, 0].astype(int) - 1, np.asarray(headers[b]) - 1)] = \
                blk[:, 1:1 + k2]
    return S


def _ao_overlap_blocks_flex(it, logfile):
    """Flexible line-by-line fallback: tolerates irregular block shapes at
    python parsing speed. Streams one block at a time into the result
    (allocated from the first block and grown on demand), instead of
    buffering every block as python floats - the old all-blocks buffer
    held ~60x the final matrix."""
    S = None
    cols, rows_i, vals = None, [], []

    def flush():
        nonlocal S
        m = max(max(rows_i), max(cols))
        if S is None:
            S = np.zeros((m, m))
        elif m > S.shape[0]:                               # irregular log - grow
            S2 = np.zeros((m, m))
            S2[:S.shape[0], :S.shape[1]] = S
            S = S2
        S[np.ix_(np.asarray(rows_i) - 1, np.asarray(cols) - 1)] = \
            np.asarray(vals, dtype=float)

    for l in it:
        t = l.split()
        if not t:
            continue
        # data rows ("irow iatom Element shell v ...") are ~99.9% of the
        # region - test them first with two cheap predicates; the header
        # check with its per-token scan runs only on the rare misses
        if len(t) >= 5 and t[0].isdigit() and t[2][0].isalpha():
            try:
                v = [float(x) for x in t[4:]]
            except ValueError:
                if S is not None or rows_i:                # left the matrix region
                    break
                continue
            rows_i.append(int(t[0]))
            vals.append(v)
            continue
        if all(x.isdigit() for x in t):                   # column-header line
            if cols and rows_i:
                flush()
                rows_i, vals = [], []
            cols = [int(x) for x in t]
            continue
        if S is not None or rows_i:                        # left the matrix region
            break
    if cols and rows_i:
        flush()
    if S is None:
        raise RuntimeError(F"empty AO overlap print in {logfile}")
    return S


def _skip_to_overlap_banner(f):
    """Advance a log-file iterator past the AO-overlap banner line.
    GPW: "OVERLAP MATRIX"; xTB: "XTB OVERLAP MATRIX"."""
    for l in f:
        if l.strip().endswith("OVERLAP MATRIX"):
            return True
    return False


def _count_banners(logfile, suffix):
    """How many banner lines end with `suffix` - the AO_MATRICES prints of the
    Hamiltonian repeat every SCF iteration, the last one is the converged matrix."""
    n = 0
    with open(logfile) as f:
        for l in f:
            if l.strip().endswith(suffix):
                n += 1
    return n


def _skip_to_nth_banner(f, suffix, n):
    """Advance a log-file iterator past the n-th (1-based) banner ending with `suffix`."""
    seen = 0
    for l in f:
        if l.strip().endswith(suffix):
            seen += 1
            if seen == n:
                return True
    return False


def read_cp2k_ao_ks_print(logfile, n_spins=1):
    """
    The AO Kohn-Sham (Hamiltonian) matrices printed by &DFT &PRINT &AO_MATRICES
    (KOHN_SHAM_MATRIX T, NDIGITS 12) in the same MOLog block layout as the overlap print.
    CP2K prints them at every SCF iteration (xTB banner "HAMILTONIAN MATRIX", GPW
    "KOHN-SHAM MATRIX", one print per spin channel) - the last n_spins prints are the
    matrices of the converged density, in the CP2K-internal AO order.

    Returns:
        list of np.ndarray (n, n), one per spin channel
    """
    suffix = None
    for cand in ("HAMILTONIAN MATRIX", "KOHN-SHAM MATRIX"):
        if _count_banners(logfile, cand):
            suffix = cand
            break
    if suffix is None:
        raise RuntimeError(F"no AO Kohn-Sham matrix print found in {logfile} - the run needs "
                           F"&DFT &PRINT &AO_MATRICES KOHN_SHAM_MATRIX T (NDIGITS 12)")
    total = _count_banners(logfile, suffix)
    if total < n_spins:
        raise ValueError(F"{logfile}: {total} Kohn-Sham matrix prints for {n_spins} spin channels")
    out = []
    for k in range(total - n_spins + 1, total + 1):
        with open(logfile) as f:
            _skip_to_nth_banner(f, suffix, k)
            try:
                H = _ao_overlap_blocks_fast(f)
            except (AssertionError, ValueError, IndexError, KeyError):
                H = None
        if H is None:
            with open(logfile) as f:
                _skip_to_nth_banner(f, suffix, k)
                H = _ao_overlap_blocks_flex(f, logfile)
        if np.abs(H - H.T).max() > 1e-8:
            raise ValueError(F"parsed Kohn-Sham matrix not symmetric ({logfile})")
        out.append(0.5 * (H + H.T))
    return out


def read_cp2k_ao_overlap_print(logfile):
    """
    Parse the AO overlap matrix printed by &DFT &PRINT &AO_MATRICES (OVERLAP T,
    NDIGITS 12) - the OVERLAP_DELTAT / Newton-X route.

    Intended use (CP2K-native alternative to the molden+libint route, exactly
    the ODIN doubled-geometry trick): run a DOUBLED geometry (atoms at t-dt
    then atoms at t) with `&DFT &EXCITED_STATES OVERLAP_DELTAT T` - CP2K skips
    the SCF and all properties, builds and prints S in ~1 s. The doubled
    template additionally needs CONNECTIVITY OFF (&TOPOLOGY) and no
    &VDW_POTENTIAL (both die on near-coincident atoms). The matrix is printed
    in CP2K-internal (MOLog) AO order - apply resort_molog_eigenvectors(l_vals)
    to both indices to convert to libint order:

        S = read_cp2k_ao_overlap_print("out.log")
        idx = CP2K_methods.resort_molog_eigenvectors(l_vals)      # one monomer
        St_ao = S[:nao, nao:][np.ix_(idx, idx)]                   # <chi(t-dt)|chi(t)>

    Validated against libint cross-geometry overlaps to 3.5e-7 (molden print
    precision - the CP2K matrix is the exact one in the code's own basis).

    Returns:
        np.ndarray (n, n): the full printed overlap matrix
    """
    with open(logfile) as f:
        if not _skip_to_overlap_banner(f):
            raise RuntimeError(
                F"no AO overlap print found in {logfile} - the OVERLAP_DELTAT run "
                F"produced neither the S_CSR_WRITE file nor the dense log print. "
                F"On stock CP2K (no Libra S_CSR_WRITE patch) set "
                F"overlap_csr_write=False so the generated input carries the dense "
                F"&AO_MATRICES OVERLAP print, or switch to mo_source='molden' + "
                F"ao_overlap_method='libint'; if the print is still missing, the run "
                F"itself failed - check the log tail")
        try:
            S = _ao_overlap_blocks_fast(f)
        except (AssertionError, ValueError, IndexError, KeyError):
            S = None
    if S is None:                                          # re-read for the fallback
        with open(logfile) as f:
            _skip_to_overlap_banner(f)
            S = _ao_overlap_blocks_flex(f, logfile)
    if np.abs(S - S.T).max() > 1e-10:
        raise ValueError(f"parsed overlap not symmetric ({logfile})")
    return S


def read_cp2k_s_csr(filename, n=None, chunk_rows=None):
    """
    Parse the sparse AO overlap written by &DFT &PRINT &S_CSR_WRITE
    (requires the qs_energy.F patch in tools/cp2k-overlap-deltat-csr.patch for
    OVERLAP_DELTAT runs - upstream only fires this print post-SCF).

    Format: 1-based "row col value" triplets, both triangles, 15-digit
    scientific - more precise AND O(nnz) instead of the O(n^2) dense log print.

    numpy's C reader engages on the filename, on file objects and across
    max_rows chunks alike, so chunking costs ~20% time while capping the
    transient at ~chunk size instead of the ~40 bytes per triplet of the
    whole-file array (never benchmark this under tracemalloc - it taxes the
    chunked reader ~6x and once misattributed the cost). Default: whole
    file. chunk_rows (~200000 is a good size) enables chunking; n (the
    matrix dimension, e.g. 2 x nao on the doubled ODT geometry) preallocates
    the result - out-of-range triplets then raise - and is otherwise grown
    on demand.

    Returns:
        np.ndarray (n, n)
    """
    if chunk_rows is None:
        rows = np.loadtxt(filename)
        if rows.size == 0:
            raise ValueError(f"empty CSR overlap file ({filename})")
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        m = int(max(rows[:, 0].max(), rows[:, 1].max()))
        if n is None:
            n = m
        elif m > n:
            raise ValueError(
                f"triplet index {m} exceeds the declared dimension {n} "
                f"({filename})")
        S = np.zeros((n, n))
        S[rows[:, 0].astype(int) - 1, rows[:, 1].astype(int) - 1] = rows[:, 2]
    else:
        S = None if n is None else np.zeros((n, n))
        hi = 0
        seen = False
        with open(filename) as f:
            while True:
                a = np.loadtxt(f, dtype=float, ndmin=2, max_rows=int(chunk_rows))
                if a.size == 0:
                    break
                seen = True
                m = int(a[:, :2].max())
                hi = max(hi, m)
                if S is None:
                    S = np.zeros((m, m))
                elif m > S.shape[0]:
                    if n is not None:
                        raise ValueError(
                            f"triplet index {m} exceeds the declared "
                            f"dimension {n} ({filename})")
                    size = max(m, 2 * S.shape[0])            # grow by doubling
                    S2 = np.zeros((size, size))
                    S2[:S.shape[0], :S.shape[1]] = S
                    S = S2
                S[a[:, 0].astype(int) - 1, a[:, 1].astype(int) - 1] = a[:, 2]
                if a.shape[0] < int(chunk_rows):
                    break
        if not seen:
            raise ValueError(f"empty CSR overlap file ({filename})")
        if n is None and hi < S.shape[0]:                    # trim growth headroom
            S = np.ascontiguousarray(S[:hi, :hi])
    if np.abs(S - S.T).max() > 1e-10:
        raise ValueError(f"CSR overlap not symmetric ({filename})")
    return S


def write_xyz(filename, atom_labels, coords_bohr, comment=""):
    """
    Write an xyz file (Angstrom) from a flat coordinate array in Bohr.

    Args:
        filename (str): output path
        atom_labels (list of str): N atomic symbols
        coords_bohr (array-like, 3N): flattened [x1,y1,z1, x2,...] in Bohr
        comment (str): second line of the xyz file
    """
    c = np.asarray(coords_bohr, dtype=float).reshape(-1, 3) / units.Angst
    nat = len(atom_labels)
    if c.shape[0] != nat:
        raise ValueError(f"coords length {3*c.shape[0]} does not match 3*natoms={3*nat}")
    with open(filename, "w") as f:
        f.write(f"{nat}\n{comment}\n")
        for lab, (x, y, z) in zip(atom_labels, c):
            f.write(f"{lab} {x:18.10f} {y:18.10f} {z:18.10f}\n")


def make_cp2k_input(params, working_directory="."):
    """
    Generate a CP2K input file from a template, in a thread-safe manner.

    The template must be a complete, working CP2K input for an
    excited-state force calculation (RUN_TYPE ENERGY_FORCE, &PROPERTIES &TDDFPT,
    &DFT &EXCITED_STATES with a STATE line, &FORCE_EVAL &PRINT &FORCES ON,
    &MO_MOLDEN print if orbitals are needed). This function rewrites:

      - PROJECT / PROJECT_NAME             -> params["project"]
      - COORD_FILE_NAME                    -> params["coord_file"]
      - WFN_RESTART_FILE_NAME  *.wfn       -> {project}-RESTART.wfn
      - WFN_RESTART_FILE_NAME  *.tdwfn     -> {project}-RESTART.tdwfn
      - SCF_GUESS                          -> RESTART (CP2K falls back to ATOMIC
                                              if the restart file does not exist)
      - STATE   (inside &EXCITED_STATES)   -> params["act_state"]  (1-based excited index)
      - RUN_TYPE                           -> ENERGY when params["force_source"] == "none"
      - NSTATES (inside &TDDFPT)           -> params["number_of_states"] (if given)

    If params["act_state"] == 0 (ground state), the whole &EXCITED_STATES ...
    &END block is omitted, so the forces/energy are ground-state ones, while
    the TDDFPT property section still runs (excitation energies + tdwfn
    continuity are preserved).

    Args:
        params (dict):
            input_template (str): path to the template file  [required]
            act_state (int): active state; 0 = ground state   [default: 0]
            project (str): CP2K project name                  [default: "libra"]
            coord_file (str): xyz file name (relative to wd)  [default: "coord.xyz"]
            number_of_states (int or None): rewrite TDDFPT NSTATES  [default: None = keep template]
            input_name (str): output file name                [default: "input.inp"]
            wfn_restart_file (str or None): explicit SCF restart source; injected
                as a DFT-level WFN_RESTART_FILE_NAME if the template has none
                (used by the OT+diag mode)                    [default: None]
        working_directory (str): where the input file is written

    Returns:
        str: path of the generated input file
    """
    template = params["input_template"]
    act_state = int(params.get("act_state", 0))
    project = params.get("project", "libra")
    coord_file = params.get("coord_file", "coord.xyz")
    number_of_states = params.get("number_of_states", None)
    input_name = params.get("input_name", "input.inp")
    # explicit SCF restart source (OT+diag mode); None -> CP2K's default
    # {project}-RESTART.wfn, which our fixed project name chains automatically
    wfn_restart_file = params.get("wfn_restart_file", None)

    with open(template) as f:
        lines = f.readlines()

    # an &EXCITED_STATES block carrying OVERLAP_DELTAT is an overlap-run
    # directive, not an excited-forces request - never drop it for act_state=0
    keep_excited = any("OVERLAP_DELTAT" in ln.upper() for ln in lines)

    # inject a DFT-level WFN_RESTART_FILE_NAME only if the template has none
    # (templates typically rely on the default restart name)
    inject_restart = wfn_restart_file is not None and not any(
        ln.split() and ln.split()[0].upper() == "WFN_RESTART_FILE_NAME"
        and ".tdwfn" not in ln.lower() for ln in lines)

    out = []
    stack = []  # CP2K section stack, upper-case names
    for line in lines:
        tokens = line.split()
        keyword = tokens[0].upper() if tokens else ""

        # --- maintain the section stack ---
        entering, ending = None, keyword.startswith("&END")
        if keyword.startswith("&") and not ending:
            entering = keyword[1:]

        # membership must be evaluated BEFORE the pop so that the
        # "&END EXCITED_STATES" line itself still counts as inside the block
        in_excited = "EXCITED_STATES" in stack or entering == "EXCITED_STATES"
        in_tddfpt = "TDDFPT" in stack or entering == "TDDFPT"

        if ending:
            if stack:
                stack.pop()
        elif entering is not None:
            stack.append(entering)

        # --- drop the whole &EXCITED_STATES block for ground-state runs ---
        if in_excited and act_state == 0 and not keep_excited:
            continue

        # --- keyword rewrites ---
        if keyword in ("PROJECT", "PROJECT_NAME"):
            out.append(f"  PROJECT {project}\n")
        elif keyword == "COORD_FILE_NAME":
            out.append(f"      COORD_FILE_NAME {coord_file}\n")
        elif keyword == "WFN_RESTART_FILE_NAME" and ".tdwfn" in line.lower():
            out.append(f"      WFN_RESTART_FILE_NAME {project}-RESTART.tdwfn\n")
        elif keyword == "WFN_RESTART_FILE_NAME":
            out.append(f"      WFN_RESTART_FILE_NAME "
                       f"{wfn_restart_file or f'{project}-RESTART.wfn'}\n")
        elif keyword == "SCF_GUESS":
            out.append("      SCF_GUESS RESTART\n")
        elif keyword == "RUN_TYPE" and params.get("force_source", "active_state") == "none":
            out.append("  RUN_TYPE ENERGY\n")
        elif keyword == "STATE" and in_excited:
            out.append(f"      STATE {act_state}\n")
        elif keyword == "NSTATES" and in_tddfpt and number_of_states is not None:
            out.append(f"      NSTATES {int(number_of_states)}\n")
        elif keyword == "LIST" and in_tddfpt and number_of_states is not None:
            states = " ".join(str(k + 1) for k in range(int(number_of_states)))
            out.append(f"          LIST {states}\n")
        else:
            out.append(line)
            if inject_restart and entering == "DFT":
                out.append(f"    WFN_RESTART_FILE_NAME {wfn_restart_file}\n")
                inject_restart = False

    os.makedirs(working_directory, exist_ok=True)
    input_path = os.path.join(working_directory, input_name)
    with open(input_path, "w") as f:
        f.writelines(out)
    return input_path


def execute_cp2k(exe, input_name, logname, wd=".", omp_num_threads=1,
                 mpi_executable=None, nprocs=1, cp2k_data_dir=None):
    """Launch one CP2K run: argv list (no shell), OMP/data-dir env for
    the child only, single-rank mpirun skip, log dedup (CP2K's -o
    APPENDS), and a fail-loud tail on nonzero exit."""
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(omp_num_threads)
    if cp2k_data_dir:
        env["CP2K_DATA_DIR"] = cp2k_data_dir
    logfile = os.path.join(wd, logname)
    if os.path.exists(logfile):
        os.remove(logfile)
    cmd = [exe, "-i", input_name, "-o", logname]
    # a single rank gains nothing from mpirun, and OpenMPI's default
    # bind-to-core would pin all OMP threads of the rank onto ONE core.
    # shlex: the launcher may carry flags ("mpirun --bind-to none")
    if mpi_executable is not None and int(nprocs) > 1:
        cmd = shlex.split(str(mpi_executable)) + ["-n", str(nprocs)] + cmd
    res = subprocess.run(cmd, cwd=wd, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        tail = ""
        if os.path.exists(logfile):
            with open(logfile) as f:
                tail = "".join(f.readlines()[-30:])
        raise RuntimeError(
            F"CP2K failed (exit {res.returncode}) in {wd} ({input_name})\n"
            F"stderr: {res.stderr.decode()[-2000:]}\nlog tail:\n{tail}")
    return logfile


def run_cp2k(coords_bohr, params):
    """
    Execute one CP2K calculation in a trajectory-specific working directory.

    Thread-safe (no os.chdir; subprocess runs with cwd=working_directory), so
    multiple trajectories can run concurrently in separate directories.
    Restart chaining is automatic: with a fixed project name, CP2K overwrites
    {project}-RESTART.wfn / .tdwfn at every step and the generated input
    points SCF_GUESS RESTART at those same files.

    Args:
        coords_bohr (array-like, 3N): current geometry, Bohr
        params (dict):
            atom_labels (list of str): atomic symbols                       [required]
            input_template (str): path to the CP2K template            [required]
            act_state (int): active state (0 = GS)                     [default 0]
            cp2k_exe (str): CP2K executable                            [default "cp2k.psmp"]
            mpi_executable (str or None): e.g. "mpirun"; None = serial [default None]
            nprocs (int): MPI ranks (used with mpi_executable)         [default 1]
                nprocs=1 skips the launcher entirely; for hybrid runs
                (nprocs > 1 with omp_num_threads > 1) add binding flags to
                mpi_executable (e.g. "mpirun --map-by slot:PE=4" or
                "--bind-to none") - Open MPI's default bind-to-core pins
                each rank's OMP threads onto a single core
            omp_num_threads (int): OMP threads for the run             [default 1]
            cp2k_data_dir (str or None): sets CP2K_DATA_DIR            [default None]
            working_directory (str): run directory                     [default "wd"]
            ot_template (str or None): OT pre-convergence template. When given,
                each call first runs a cheap OT ENERGY calculation (project
                {project}_ot, restart chained across steps) and the main run
                restarts from its wavefunction - the two-run scheme of the
                NBRA step2 xTB workflow, for large/hard-SCF systems [default None]
            retry_templates (list of str): templates to fall back to, in order,
                when the run fails - each retries the whole step (OT run
                included). Meant for the geometries where the SCF of the main
                template stalls: give a template with a different mixing or
                MAX_SCF, since re-running the same input only repeats the
                failure                                                [default []]
            allow_unconverged_scf (int): 1 accepts a step whose log reports an
                unconverged SCF (only reachable with CP2K's
                IGNORE_CONVERGENCE_FAILURE, which downgrades the abort to a
                warning); 0 raises on it                                [default 0]
            project (str), coord_file (str), number_of_states (int): see make_cp2k_input

    Returns:
        str: path to the output log file

    Raises:
        RuntimeError: if CP2K exits non-zero (tail of the log is included) with
            the main template and every retry_templates entry, or if the run
            that succeeded reports an unconverged SCF
    """
    atom_labels = params["atom_labels"]
    exe = params.get("cp2k_exe", "cp2k.psmp")
    mpi = params.get("mpi_executable", None)
    nprocs = int(params.get("nprocs", 1))
    wd = params.get("working_directory", "wd")
    coord_file = params.get("coord_file", "coord.xyz")

    os.makedirs(wd, exist_ok=True)
    write_xyz(os.path.join(wd, coord_file), atom_labels, coords_bohr)

    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(params.get("omp_num_threads", 1))
    if params.get("cp2k_data_dir", None):
        env["CP2K_DATA_DIR"] = params["cp2k_data_dir"]

    def _execute(input_name, logname):
        return execute_cp2k(exe, input_name, logname, wd=wd,
                            omp_num_threads=params.get("omp_num_threads", 1),
                            mpi_executable=mpi, nprocs=nprocs,
                            cp2k_data_dir=params.get("cp2k_data_dir", None))

    # ---- optional OT pre-convergence run (large systems / hard SCF) ----
    # Two-run scheme a la the NBRA step2 xTB workflow: a cheap OT ENERGY run
    # converges the density robustly ({project}_ot-RESTART.wfn, chained across
    # steps), then the main diagonalization run restarts from it.
    project = params.get("project", "libra")

    def _attempt(template):
        """one whole step: the OT pre-convergence run (where configured) and
        then the main run from `template` - a retry repeats both, so it does
        not inherit the density a failed attempt left in the restart files"""
        main_params = dict(params)
        main_params["input_template"] = template
        if params.get("ot_template", None):
            ot_params = dict(params)
            ot_params.update({"input_template": params["ot_template"],
                              "project": f"{project}_ot",
                              "input_name": "ot_input.inp",
                              "act_state": 0})
            make_cp2k_input(ot_params, working_directory=wd)
            _execute("ot_input.inp", "ot.log")
            main_params["wfn_restart_file"] = f"{project}_ot-RESTART.wfn"
        make_cp2k_input(main_params, working_directory=wd)
        return _execute(main_params.get("input_name", "input.inp"), "out.log")

    # a stalled SCF is a property of the geometry, not of the starting density:
    # distorted excited-state geometries carry near-degenerate solutions the
    # mixing ping-pongs between, and every guess falls into the same cycle - so
    # a retry pays only with a template that changes the SCF itself
    templates = [params["input_template"]] + list(params.get("retry_templates", []) or [])
    for itry, template in enumerate(templates):
        try:
            logfile = _attempt(template)
        except RuntimeError as err:
            if itry + 1 == len(templates):
                if len(templates) > 1:
                    raise RuntimeError(
                        f"CP2K failed in {wd} with the main template and all "
                        f"{len(templates) - 1} retry_templates {templates[1:]}\n"
                        f"last error: {err}")
                raise
            print(f"WARNING in run_cp2k: CP2K failed in {wd} with {template} - "
                  f"retrying this step with {templates[itry + 1]}")
            continue
        break

    # CP2K aborts on an unconverged SCF, but IGNORE_CONVERGENCE_FAILURE turns
    # that abort into a warning - the run then exits 0 carrying a density, an
    # energy and forces that never converged
    if int(params.get("allow_unconverged_scf", 0)) == 0:
        with open(logfile) as fp:
            if any("SCF run NOT converged" in line for line in fp):
                raise RuntimeError(
                    f"CP2K reported an unconverged SCF in {wd} ({logfile}) - its "
                    f"energy and forces are not usable. Fix the SCF (density "
                    f"mixing, MAX_SCF) or add a retry_templates entry that does; "
                    f"allow_unconverged_scf = 1 accepts the step regardless")
    return logfile


def read_cp2k_forces(logfile):
    """
    Parse atomic forces from a CP2K log ("FORCES| Atomic forces [hartree/bohr]").

    Takes the LAST such block in the file. With &DFT &EXCITED_STATES active,
    these are the excited-state (GS + linear-response) forces of the selected
    state; otherwise ground-state forces.

    Returns:
        np.ndarray (natoms, 3): forces in Hartree/Bohr
    """
    forces, current = [], None
    with open(logfile) as f:
        for line in f:
            if line.lstrip().startswith("FORCES| Atomic forces"):
                current = []
            elif current is not None:
                m = re.match(r"\s*FORCES\|\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)", line)
                if m:
                    current.append([float(m.group(k)) for k in (2, 3, 4)])
                elif current:  # rows collected and a non-row line -> block ended
                    forces = current
                    current = None
                elif not line.lstrip().startswith("FORCES|"):
                    current = None  # left the FORCES| region without any rows
                # else: the "FORCES| Atom x y z |f|" header row - keep waiting
    if not forces:
        raise ValueError(f"No 'FORCES| Atomic forces' block found in {logfile}")
    arr = np.array(forces, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Non-finite forces parsed from {logfile} (NaN/Inf)")
    return arr


def read_cp2k_total_energy(logfile):
    """
    Read the final total energy: the LAST "ENERGY| Total FORCE_EVAL" line, Hartree.

    NOTE: when &DFT &EXCITED_STATES is active, this already INCLUDES the
    excitation energy (E = E_GS + omega_state) — verified against the
    'Excited State Energy' block and against E_GS + omega explicitly.
    """
    e = None
    with open(logfile) as f:
        for line in f:
            if line.lstrip().startswith("ENERGY| Total FORCE_EVAL"):
                e = float(line.split()[-1])
    if e is None:
        raise ValueError(f"No 'ENERGY| Total FORCE_EVAL' line found in {logfile}")
    return e


def read_cp2k_excitation_energies(logfile):
    """
    R-TDDFPT excitation energies and oscillator strengths.

    Thin wrapper around cp2k_find_excitation_energies (which parses the
    "TDDFPT|" states table and returns plain eV lists).

    Returns:
        omegas (np.ndarray, nstates): excitation energies in Hartree
        fosc   (np.ndarray, nstates): oscillator strengths
    """
    E, F = cp2k_find_excitation_energies(logfile)
    if len(E) == 0:
        raise ValueError(f"No TDDFPT states table found in {logfile}")
    return np.array(E) * units.ev2Ha, np.array(F)


def read_cp2k_excited_state_info(logfile):
    """
    Parse the 'Excited State Energy'/forces block printed by &EXCITED_STATES.

    Returns:
        dict with keys 'omega' (Hartree, precise 10-decimal value for the
        active state) and, for energy-only runs, 'etot'; or None if the block
        is absent (ground-state run).
    """
    omega, etot = None, None
    with open(logfile) as f:
        for line in f:
            if "Excitation Energy [Hartree]" in line:
                omega = float(line.split()[-1])
            elif "Total Energy [Hartree]" in line:
                etot = float(line.split()[-1])
    if omega is None:
        return None
    return {"omega": omega, "etot": etot}


def compute_state_energy_and_force(coords_bohr, params):
    """
    One full electronic-structure evaluation for the non-NBRA driver:
    run CP2K at `coords_bohr` for state params["act_state"], parse everything.

    Returns:
        dict:
            E_total (float): total energy of the ACTIVE state, Hartree
                             (E_GS + omega for excited states)
            E_gs (float): ground-state total energy, Hartree
            forces (np.ndarray (nat,3)): active-state forces, Ha/Bohr
            omegas (np.ndarray): TDDFPT excitation energies, Hartree
            fosc (np.ndarray): oscillator strengths
            E_adi (np.ndarray, nstates+1): total energies of ALL states
                                           [E_gs, E_gs+omega_1, ...]
            logfile (str): path to the CP2K log (for further parsing, e.g.
                           CI amplitudes via read_cp2k_tddfpt_log_file)

    With params["time_dependent"] = False (ground-state-only mode, template
    without &PROPERTIES &TDDFPT and without &EXCITED_STATES): one GS run,
    E_adi = [E_gs], omegas/fosc empty, forces = ground-state forces.
    """
    if not params.get("time_dependent", True):
        run_params = dict(params)
        run_params["act_state"] = 0
        logfile = run_cp2k(coords_bohr, run_params)
        E_gs = read_cp2k_total_energy(logfile)
        if params.get("force_source", "active_state") == "none":
            forces = np.zeros((len(params["atom_labels"]), 3))
        else:
            forces = read_cp2k_forces(logfile)
        return {"E_total": E_gs, "E_gs": E_gs, "forces": forces,
                "omegas": np.array([]), "fosc": np.array([]),
                "E_adi": np.array([E_gs]), "logfile": logfile}

    act_state = int(params.get("act_state", 0))

    if params.get("force_source", "active_state") == "all_states":
        # ---- mechanism B: one GS run + Z-vector solves for ALL listed states.
        # Template must have NO &EXCITED_STATES (main forces = ground state)
        # and must carry &TDDFPT &PRINT &FORCES with a LIST line (rewritten
        # from number_of_states). Returns the extra key forces_all (nstates+1, nat, 3).
        logfile = run_cp2k(coords_bohr, params)
        E_gs = read_cp2k_total_energy(logfile)          # no omega included here
        f_gs = read_cp2k_forces(logfile)
        omegas, fosc = read_cp2k_excitation_energies(logfile)
        wd = params.get("working_directory", "wd")
        project = params.get("project", "libra")
        tdfrc = os.path.join(wd, f"{project}-TDFORCE-1_0.tdfrc")
        f_ex, _, _ = read_cp2k_tdfrc(tdfrc)
        nexc = len(omegas)
        missing = [k for k in range(1, nexc + 1) if k not in f_ex]
        if missing:
            raise ValueError(f"states {missing} have no forces in {tdfrc} - "
                             f"check the LIST line and keep THRESHOLD 0.0 "
                             f"(dark states are screened otherwise)")
        forces_all = np.stack([f_gs] + [f_ex[k] for k in range(1, nexc + 1)])
        E_adi = np.concatenate(([E_gs], E_gs + omegas))
        return {"E_total": float(E_adi[act_state]), "E_gs": E_gs,
                "forces": forces_all[act_state], "forces_all": forces_all,
                "omegas": omegas, "fosc": fosc, "E_adi": E_adi, "logfile": logfile}

    logfile = run_cp2k(coords_bohr, params)

    E_total = read_cp2k_total_energy(logfile)
    if params.get("force_source", "active_state") == "none":
        forces = np.zeros((len(params["atom_labels"]), 3))
    else:
        forces = read_cp2k_forces(logfile)
    omegas, fosc = read_cp2k_excitation_energies(logfile)

    if act_state == 0:
        E_gs = E_total
    else:
        info = read_cp2k_excited_state_info(logfile)
        if info is not None:
            # &EXCITED_STATES template: the FORCE_EVAL total includes the
            # active state's omega (10-decimal in the excited-state block)
            E_gs = E_total - info["omega"]
        else:
            # no Excited State Energy block (all_states / forces-free
            # templates run the ground state): the total IS E_gs -
            # subtracting omegas[act_state-1] here shifted every diagonal
            # by the geometry-dependent omega_act
            E_gs = E_total

    E_adi = np.concatenate(([E_gs], E_gs + omegas))
    return {"E_total": E_total, "E_gs": E_gs, "forces": forces,
            "omegas": omegas, "fosc": fosc, "E_adi": E_adi, "logfile": logfile}


def fd_force_check(coords_bohr, params, atom, dim, h=0.01):
    """
    Finite-difference validation of the analytic force on one component.

    Runs two displaced CP2K calculations (+h/-h Bohr on atom/dim) in separate
    subdirectories and returns the central-difference force of the ACTIVE
    state's total energy. Compare with read_cp2k_forces()[atom, dim].

    Args:
        coords_bohr (array-like, 3N): geometry, Bohr
        params (dict): same as run_cp2k (act_state selects the state)
        atom (int): 0-based atom index
        dim (int): 0=x, 1=y, 2=z
        h (float): displacement, Bohr

    Returns:
        float: -(E(+h) - E(-h)) / (2h), Hartree/Bohr
    """
    wd0 = params.get("working_directory", "wd")
    E = {}
    for sgn, tag in ((+1.0, "p"), (-1.0, "m")):
        qd = np.array(coords_bohr, dtype=float).copy()
        qd[3 * atom + dim] += sgn * h
        pd = dict(params)
        pd["working_directory"] = f"{wd0}_fd_{atom}{'xyz'[dim]}{tag}"
        E[tag] = compute_state_energy_and_force(qd, pd)["E_total"]
    return -(E["p"] - E["m"]) / (2.0 * h)


def _two_spinor(S):
    """alpha (+) beta block-diagonal spinor matrix. S is either a single array
    (restricted: the block is duplicated, = data_conv.form_block_matrix(S, 0, 0, S))
    or an (S_alpha, S_beta) pair (UKS)."""
    Sa, Sb = S if isinstance(S, (tuple, list)) else (S, S)
    return data_conv.form_block_matrix(Sa, np.zeros((Sa.shape[0], Sb.shape[1])),
                                       np.zeros((Sb.shape[0], Sa.shape[1])), Sb)


def molden_spin_counts(molden_file):
    """
    Per-spin MO counts from a CP2K molden file. The channels are UNEQUAL in
    general: CP2K stores nelec_spin + ADDED_MOS orbitals per spin, so e.g. a
    triplet has two more alpha MOs than beta - a half-split of the stacked
    [MO] section misassigns the channels.

    Returns:
        (n_alpha, n_beta) - n_beta is 0 for restricted files
    """
    with open(molden_file) as f:
        txt = f.read()
    return txt.count("Spin= Alpha"), txt.count("Spin= Beta")


def load_molden_data(molden_file, is_spherical=True, isUKS=False):
    """
    Read a CP2K molden file into libint shells + AO-resorted MO coefficients.

    UKS molden files carry the alpha and beta MOs stacked (alpha first); with
    isUKS=True the stack is split and the beta channel returned alongside as
    C_b / E_b (same AO resorting - the AO order is a basis property).

    Returns:
        dict: shell (libint shells), C (nmo x nao, resorted to libint AO order),
              E (nmo, orbital energies in Hartree), resort_idx, nspin
              (+ C_b / E_b for isUKS=True)
    """
    na, nb = molden_spin_counts(molden_file)
    if (nb > 0) != bool(isUKS):
        raise ValueError(F"{molden_file} carries nspin={2 if nb else 1} but isUKS={isUKS} - "
                         F"the CP2K template's UKS/MULTIPLICITY must match the isUKS parameter")
    shell, l_vals = molden_methods.molden_file_to_libint_shell(molden_file, is_spherical)
    C, E = molden_methods.eigenvectors_molden(molden_file, nbasis(shell), l_vals)
    idx = resort_molog_eigenvectors(
        l_vals, molden_methods.molden_purity(molden_file, l_vals))
    C = np.array([c[idx] for c in C])
    E = np.array(E)
    out = {"shell": shell, "resort_idx": np.array(idx), "nspin": 2 if isUKS else 1}
    if isUKS:
        out.update({"C": C[:na], "E": E[:na],
                    "C_b": C[na:na + nb], "E_b": E[na:na + nb]})
    else:
        out.update({"C": C, "E": E})
    return out


def mo_overlap(data1, data2, lowest_orbital, highest_orbital, nprocs=1):
    """
    Windowed MO overlap  S_ij = <phi_i(R1)|phi_j(R2)>  (one spin channel).

    data1/data2 come from load_molden_data (possibly the same object for
    same-time overlaps). The AO overlap is computed with libint for the pair
    of geometries; the MO matrix is sliced to the 1-based orbital window
    [lowest_orbital, highest_orbital].

    Returns:
        np.ndarray (norb_w, norb_w)

    Raises ValueError when the window does not fit the stored MOs (python
    slicing would silently truncate or wrap).
    """
    for d in (data1, data2):
        nmo = d["C"].shape[0]
        if not 1 <= lowest_orbital <= highest_orbital <= nmo:
            raise ValueError(F"orbital window [{lowest_orbital}, {highest_orbital}] does "
                             F"not fit the {nmo} stored MOs (1-based) - increase ADDED_MOS "
                             F"in the template or adjust lowest_orbital/highest_orbital")
    S_ao = data_conv.MATRIX2nparray(compute_overlaps(data1["shell"], data2["shell"], nprocs))
    lo, hi = lowest_orbital - 1, highest_orbital
    S = data1["C"][lo:hi] @ S_ao @ data2["C"][lo:hi].T   # window first: O(win*nao^2)
    return np.ascontiguousarray(S.real)


def align_mo_phases(C_prev_aligned, data_curr, St_mo):
    """
    Fix the random MO phases of the current step against the previous one:
    flip the sign of every current MO whose time-overlap diagonal is negative.

    Args:
        C_prev_aligned: unused except for interface symmetry (phases are
            defined by St_mo, which must be <prev_aligned | curr_raw>)
        data_curr (dict): unused here - the CALLER applies the returned
            phases to its MO matrix; kept for call-site symmetry
        St_mo (np.ndarray): time-overlap computed with the RAW current MOs,
            windowed, one spin channel; fixed in place (columns flipped)

    Returns:
        phases (np.ndarray, norb_w): the +-1 factors applied (window orbitals)
    """
    phases = np.sign(np.diag(St_mo))
    phases[phases == 0] = 1.0
    St_mo *= phases[np.newaxis, :]
    return phases


# ========================= TDDFPT CI data ==================================


def read_ci_data(logfile, nstates, ci_threshold=0.01, lowest_orbital=1, highest_orbital=int(1e10),
                 isUKS=False):
    """
    Parse TDDFPT excitation energies + CI expansions from a CP2K log.

    Thin wrapper around read_cp2k_tddfpt_log_file (master format). The cut
    retains configs by |C|^2 > ci_threshold; the log itself only prints
    |C| >= 5e-2, the floor of any log-based read.

    Returns:
        dict: E (np.ndarray nexc, Hartree), configs (list per state of [occ, vir]
              ABSOLUTE 1-based orbital pairs, per spin channel), amps (list per
              state of np.ndarray, NORMALIZED after tolerance truncation),
              spins (list per state, 'alp'/'bet' - beta appears for isUKS logs,
              whose excitation analysis labels every config with its spin),
              window_drops (np.ndarray nexc, the TOTAL CI weight each state
              loses to the read chain - the log's print floor, the tolerance
              cut and the [lowest_orbital, highest_orbital] clip, composed
              against the TDA unit norm - the renormalization below hides it
              from any later norm)

    UKS caveat: the CP2K analysis print (2026.1-dev) decodes the alpha-channel
    flat indices with the beta-channel dimensions, so the alpha [occ, vir]
    labels come out corrupted whenever the channels differ (beta labels and
    all amplitudes/energies are fine) - prefer the NAMD_PRINT vectors for UKS.
    """
    pp = {"logfile_name": logfile, "number_of_states": nstates,
          "ci_threshold": ci_threshold,
          "isUKS": bool(isUKS), "lowest_orbital": lowest_orbital, "highest_orbital": highest_orbital}
    out = read_cp2k_tddfpt_log_file(pp)
    if len(out) == 2:
        # devel branch: (info, [E_Ha, configs, amplitudes, spins])
        info, data = out
        E = np.asarray(data[0], dtype=float)[:nstates]
        basis = [np.asarray(c, dtype=int).tolist() for c in data[1][:nstates]]
        coeff = [np.asarray(a, dtype=float) for a in data[2][:nstates]]
        spin = (list(data[3][:nstates]) if len(data) > 3
                else [["alp"] * len(b) for b in basis])
    else:
        # master branch: (E_eV, configs, amplitudes, spins)
        E_ev, basis, coeff, spin = out
        E = np.array(E_ev[:nstates]) * units.ev2Ha
        basis = [list(b) for b in basis[:nstates]]
        coeff = coeff[:nstates]
        spin = spin[:nstates]
    # the parser reads only orbital_space - clip the configs to the
    # requested window here, as ci_data_from_newtonx does (out-of-window
    # amplitudes cannot be represented in the windowed overlap machinery)
    lo_w, hi_w = int(lowest_orbital), int(highest_orbital)
    amps = []
    drops = np.zeros(len(basis))
    for k in range(len(basis)):
        keep = [j for j, c in enumerate(basis[k])
                if lo_w <= int(c[0]) and int(c[1]) <= hi_w]
        if not keep:
            raise ValueError(F"read_ci_data: state {k + 1} retains no configs "
                             F"inside orbitals [{lo_w}, {hi_w}] of {logfile}")
        total2 = float(np.sum(np.asarray(coeff[k], dtype=float) ** 2))
        if len(keep) < len(basis[k]):
            basis[k] = [basis[k][j] for j in keep]
            spin[k] = [spin[k][j] for j in keep]
        a = np.array([coeff[k][j] for j in keep], dtype=float)
        drops[k] = 1.0 - float(np.sum(a ** 2)) / total2
        if drops[k] > 1e-6:
            print(F"read_ci_data: state {k + 1}: dropped CI weight {drops[k]:.2e} to "
                  F"the orbital window [{lo_w}, {hi_w}] (vector renormalized)")
        amps.append(a / np.linalg.norm(a))
    rd = pp.get("ci_read_drops")
    if rd is not None:
        n = min(len(rd), len(drops))
        drops[:n] = 1.0 - (1.0 - rd[:n]) * (1.0 - drops[:n])
    return {"E": E, "configs": basis, "amps": amps, "spins": spin,
            "window_drops": drops}


# ===================== Backend A: Slater determinants ======================


def sd_overlaps(sd_states, St_mo, S_mo_prev, S_mo_curr, orthonormalize=True,
                sd_states_prev=None, kernel="auto", cache=None):
    """
    SD overlap matrices from windowed one-spin MO overlaps.

    Args:
        sd_states: signed-spin-orbital SD list (GS first), window-relative;
            this is the CURRENT (ket) basis
        St_mo, S_mo_prev, S_mo_curr: (norb_w x norb_w) MO matrices; a single
            array is used for both spin blocks (restricted), an
            (alpha, beta) tuple/list gives each channel its own block (UKS -
            both channels over the same orbital window)
        orthonormalize: normalization of the SD overlaps:
            - "lowdin" or True: Lowdin S^-1/2 sandwiches, both factors
              recomputed from this step's same-time overlaps (the step3
              convention, apply_orthonormalization_scipy) [default]
            - "lowdin_cached": the same Lowdin sandwich, but the bra factor
              S_prev^-1/2 is reused from the previous call via `cache` (the
              cp2k_compute_adi_nbra convention). The cache is guarded by a
              digest of the source overlaps, SD list and kernel (the step3
              guard) and recomputes on any mismatch, so results are
              bit-identical to "lowdin"; a hit is one fractional matrix
              power cheaper - a saving that matters in file-based
              post-processing, not next to a CP2K call
            - "none" or False: raw determinant overlaps (the historical
              dftb_compute_adi behavior; upstream moved that driver to
              Lowdin in 2026-04)
        sd_states_prev: previous-step (bra) SD list if it differs from the
            current one (e.g. per-step energy-sorted bases) [default: sd_states]
        kernel: SD determinant kernel; same names as step3's
            sd_overlap_kernel: "auto" [default] | "batched" | "lemma" | "per_pair"
        cache: mutable dict for the "lowdin_cached" mode; reads and updates
            cache["s_ihalf"] / cache["digest"] (S^-1/2 of this step's ket
            basis and its content key)

    Returns:
        st_sd (nSD x nSD), s_sd_curr (nSD x nSD)
    """
    mode = orthonormalize
    if mode is True:
        mode = "lowdin"
    elif mode is False or mode is None:
        mode = "none"
    if mode not in ("none", "lowdin", "lowdin_cached"):
        raise ValueError(f"unknown orthonormalize mode: {orthonormalize}")
    if mode == "lowdin_cached" and cache is None:
        raise ValueError("orthonormalize='lowdin_cached' requires a cache dict")
    ovlp = mapping.sd_overlap_kernel(kernel)
    if sd_states_prev is None:
        sd_states_prev = sd_states
    n_half = (St_mo[0] if isinstance(St_mo, (tuple, list)) else St_mo).shape[0]
    # the kernels return listed-order overlaps - consistent with the SD2CI
    # amplitudes (built without parities) as-is
    st = ovlp(sd_states_prev, sd_states, _two_spinor(St_mo))
    s1 = ovlp(sd_states_prev, sd_states_prev, _two_spinor(S_mo_prev))
    s2 = ovlp(sd_states, sd_states, _two_spinor(S_mo_curr))
    if mode == "lowdin":
        # two factorizations, not four: the composed helper calls
        # re-factorized s2 three times
        u1 = ortho.get_Lowdin_scipy(s1)
        u2 = ortho.get_Lowdin_scipy(s2)
        st = np.linalg.multi_dot([u1.conj().T, st, u2])
        s2 = np.linalg.multi_dot([u2.conj().T, s2, u2])
    elif mode == "lowdin_cached":
        i1 = None
        if cache.get("digest") == ortho.lowdin_digest(S_mo_prev, repr(sd_states_prev) + kernel):
            i1 = cache.get("s_ihalf")
        if i1 is None:
            i1 = ortho.get_Lowdin_scipy(s1)
        i2 = ortho.get_Lowdin_scipy(s2)
        st = np.linalg.multi_dot([i1.conj().T, st, i2])
        s2 = np.linalg.multi_dot([i2.conj().T, s2, i2])
        cache.update(s_ihalf=i2,
                     digest=ortho.lowdin_digest(S_mo_curr, repr(sd_states) + kernel))
    return st, s2


# ======================= Backend B: citools CSFs ===========================


def ci_overlap_citools(St_mo, ci_prev, ci_curr, homo, lowest_orbital, highest_orbital, nstates):
    """
    CI time-overlap via the citools CSF machinery (spin-adapted singlets).

    Args:
        St_mo: windowed one-spin MO time-overlap (norb_w x norb_w)
        ci_prev, ci_curr: read_ci_data dicts (ABSOLUTE orbital indices)
        homo: absolute 1-based HOMO index
        nstates: total number of states INCLUDING the ground state

    Returns:
        np.ndarray (nstates x nstates)
    """
    data_prev = [ci_prev["E"], [np.array(c) for c in ci_prev["configs"]], ci_prev["amps"]]
    data_curr = [ci_curr["E"], [np.array(c) for c in ci_curr["configs"]], ci_curr["amps"]]
    orbital_space = list(range(lowest_orbital, highest_orbital + 1))
    ovlp_params = {"homo_indx": homo,
                   "nocc": homo - lowest_orbital,
                   "nvirt": highest_orbital - homo,
                   "nelec": (homo - lowest_orbital + 1) * 2,
                   "nstates": nstates,
                   "active_space": orbital_space,
                   "orbital_space": orbital_space}
    st_ci = ci.overlap(_two_spinor(St_mo), data_prev, data_curr, ovlp_params)
    return np.array(st_ci).real


def make_overlap_deltat_input(template_file, csr_write=True):
    """
    Derives the OVERLAP_DELTAT companion input from a regular CP2K template,
    so ao_overlap_method="cp2k" does not require a hand-written
    es_overlap_deltat_* file.

    Transformation rules (everything else - basis sets, XC, SCF, CELL, KINDs -
    is kept verbatim so the overlap run matches the parent electronic
    structure):
        - GLOBAL RUN_TYPE -> ENERGY
        - the &PROPERTIES (TDDFPT), &VDW_POTENTIAL and &MOTION sections are
          dropped (dispersion diverges on the doubled geometry)
        - the &EXCITED_STATES section is replaced by the OVERLAP_DELTAT switch
          (added when the parent has none)
        - csr_write=True adds the &S_CSR_WRITE print (the sparse fast path of
          the Libra CP2K patch); csr_write=False adds a dense &AO_MATRICES
          OVERLAP print instead - stock-CP2K-compatible, read from the log by
          the same caller
        - CONNECTIVITY OFF is enforced in every &TOPOLOGY section (the doubled
          xyz would otherwise trip the topology generation)
        - the electronic state is re-posed for the doubled system: UKS / LSD /
          SPIN_POLARIZED and MULTIPLICITY are dropped (doubling makes the
          electron count even, so an open-shell parent multiplicity becomes
          inconsistent - the AO overlap is basis-only and spin-independent
          anyway) and any CHARGE is doubled to keep the parity correct

    Args:
        template_file ( string ): path to the parent CP2K input template
        csr_write ( Boolean ): include the &S_CSR_WRITE print [ default: True ]

    Returns:
        string: the text of the derived OVERLAP_DELTAT input
    """
    with open(template_file) as f:
        lines = f.read().splitlines()

    DROP = ("PROPERTIES", "VDW_POTENTIAL", "MOTION")
    # the overlap must be printed somewhere: sparse &S_CSR_WRITE (the Libra
    # CP2K patch) or the dense &AO_MATRICES log print (stock CP2K). The dense
    # print is OWNED by this generator: the source's &AO_MATRICES section (any
    # matrix selection, e.g. the KOHN_SHAM_MATRIX print of a POD template) is
    # dropped and one OVERLAP T section emitted - a substring test on the section
    # name once passed a KOHN_SHAM-only section as an overlap request
    have_out = _scan_overlap_template("\n".join(lines))[0] if csr_write else False
    out = []
    stack = []
    skip_until = None      # stack depth at which a dropped section ends
    did_excited = False

    def emit_excited(indent):
        out.append(F"{indent}&EXCITED_STATES T")
        out.append(F"{indent}  OVERLAP_DELTAT T")
        out.append(F"{indent}&END EXCITED_STATES")

    def emit_overlap_print(indent):
        if csr_write:
            out.append(F"{indent}&S_CSR_WRITE ON")
            out.append(F"{indent}  THRESHOLD 1.0E-12")
            out.append(F"{indent}&END S_CSR_WRITE")
        else:
            out.append(F"{indent}&AO_MATRICES")
            out.append(F"{indent}  OVERLAP T")
            out.append(F"{indent}  NDIGITS 12")
            out.append(F"{indent}&END AO_MATRICES")

    for ln in lines:
        word = ln.strip()
        up = word.upper()
        indent = ln[:len(ln) - len(ln.lstrip())]
        if up.startswith("&END"):
            if skip_until is not None:
                if len(stack) == skip_until:
                    skip_until = None
                stack.pop()
                continue
            if stack and stack[-1] == "DFT":
                if not did_excited:
                    emit_excited(indent + "  ")
                    did_excited = True
                if not have_out:
                    out.append(F"{indent}  &PRINT")
                    emit_overlap_print(indent + "    ")
                    out.append(F"{indent}  &END PRINT")
                    have_out = True
            stack.pop()
            out.append(ln)
            continue
        if up.startswith("&"):
            name = up[1:].split()[0] if len(up) > 1 else ""
            if skip_until is not None:
                stack.append(name)
                continue
            if name in DROP or (name == "EXCITED_STATES" and "DFT" in stack) \
                    or (name == "AO_MATRICES" and not csr_write):
                stack.append(name)
                skip_until = len(stack)
                if name == "EXCITED_STATES" and not did_excited:
                    emit_excited(indent)
                    did_excited = True
                continue
            stack.append(name)
            out.append(ln)
            if name == "PRINT" and stack[-2:-1] == ["DFT"] and not have_out:
                emit_overlap_print(indent + "  ")
                have_out = True
            if name == "TOPOLOGY":
                pass    # CONNECTIVITY enforced in the per-section walk below
            continue
        if skip_until is not None:
            continue
        if up.startswith("RUN_TYPE") and stack and stack[-1] == "GLOBAL":
            out.append(F"{indent}RUN_TYPE ENERGY")
            continue
        if stack and stack[-1] == "DFT":
            key = up.split()[0] if up else ""
            if key in ("UKS", "LSD", "SPIN_POLARIZED", "MULTIPLICITY"):
                continue    # doubled system: re-posed as closed shell (S is spin-independent)
            if key == "CHARGE":
                out.append(F"{indent}CHARGE {2 * int(word.split()[1])}")
                continue
        out.append(ln)

    # every &TOPOLOGY section ends up with exactly CONNECTIVITY OFF:
    # existing directives are replaced, absent ones injected
    fixed = []
    in_topo = False
    topo_sub = []          # subsection names inside the current &TOPOLOGY
    topo_done = False
    for ln in out:
        word = ln.strip()
        up = word.upper()
        indent = ln[:len(ln) - len(ln.lstrip())]
        if in_topo and up.startswith("&END"):
            if topo_sub:
                topo_sub.pop()
            else:
                if not topo_done:
                    fixed.append(F"{indent}  CONNECTIVITY OFF")
                in_topo = False
            fixed.append(ln)
            continue
        if up.startswith("&TOPOLOGY"):
            in_topo = True
            topo_done = False
            fixed.append(ln)
            continue
        if in_topo and up.startswith("&"):
            topo_sub.append(up[1:].split()[0] if len(up) > 1 else "")
            fixed.append(ln)
            continue
        if in_topo and not topo_sub and up.startswith("CONNECTIVITY"):
            fixed.append(F"{indent}CONNECTIVITY OFF")
            topo_done = True
            continue
        fixed.append(ln)
    return "\n".join(fixed) + "\n"



_TEMPLATE_TEXTS = {}


def _template_text(path):
    """The template file's text, cached per (mtime, size): the pre-run
    validators run every trajectory-step and the file rarely changes."""
    st = os.stat(path)
    key = (st.st_mtime_ns, st.st_size)
    hit = _TEMPLATE_TEXTS.get(path)
    if hit is None or hit[0] != key:
        with open(path) as f:
            hit = (key, f.read())
        _TEMPLATE_TEXTS[path] = hit
    return hit[1]


def _template_has_section(text, name):
    """Comment-stripped scan: an active &NAME section (argument not OFF)."""
    for l in text.splitlines():
        ls = l.split("#")[0].split("!")[0].strip().upper()
        if ls.startswith(F"&{name}"):
            t = ls.split()
            return not (len(t) > 1 and t[1] == "OFF")
    return False


def _scan_overlap_template(text):
    """Comment-stripped scan of a CP2K input: does it request an overlap
    output - an active &S_CSR_WRITE block, or an OVERLAP keyword directly
    inside &AO_MATRICES (nested subsections like &EACH tracked by depth,
    so a keyword after them still counts). A print-key section argument
    OFF disables the section - it must not count as a request."""
    active = [l.split("#")[0].split("!")[0].strip().upper()
              for l in text.splitlines()]

    def sect_off(l):
        t = l.split()
        return len(t) > 1 and t[1] == "OFF"

    has_csr = any(l.startswith("&S_CSR_WRITE") and not sect_off(l)
                  for l in active)
    has_dense, ao_depth, ao_off = False, 0, False
    for l in active:
        if l.startswith("&AO_MATRICES"):
            ao_depth = 1
            ao_off = sect_off(l)
        elif l.startswith("&END"):
            ao_depth = max(0, ao_depth - 1)
        elif l.startswith("&") and ao_depth:
            ao_depth += 1
        elif ao_depth == 1 and not ao_off and re.match(r"OVERLAP(\s+(T|TRUE|\.TRUE\.|ON|YES|1))?\s*$", l):
            has_dense = True
    return has_csr, has_dense

def mulliken_fragment_operators(C_win, S_ao, frag_cols):
    """
    The Mulliken fragment operators over an MO window: P_f = 1/2 (C_f (S C^T)_f + transpose), C_f the
    window MOs restricted to the AO columns of fragment f - the symmetrized oblique projector onto the
    fragment's AO span, whose expectation value is the Mulliken population of the fragment. The
    fragments' operators sum to the window's MO overlap C S C^T.

    Args:
        C_win (array (nwin, n_ao)): window MO coefficients (rows) in the AO order of S_ao
        S_ao (array (n_ao, n_ao)): the AO overlap
        frag_cols (list of index arrays): the AO columns of every fragment

    Returns:
        list over fragments of arrays (nwin, nwin)
    """
    C_m = data_conv.nparray2MATRIX(np.asarray(C_win))
    S_m = data_conv.nparray2MATRIX(np.asarray(S_ao))
    nwin, n_ao = C_m.num_of_rows, C_m.num_of_cols
    SC = S_m * C_m.T()                                                   # (n_ao, nwin), DGEMM
    rows = intList()
    rows.extend(list(range(nwin)))
    out = []
    for cols in frag_cols:
        ids = intList()
        ids.extend([int(i) for i in cols])
        nf = len(ids)
        Cf = MATRIX(nwin, nf)
        pop_submatrix(C_m, Cf, rows, ids)                                # the fragment's AO columns
        SCf = MATRIX(nf, nwin)
        pop_submatrix(SC, SCf, ids, rows)
        P = data_conv.MATRIX2nparray(Cf * SCf, np.float64)              # (nwin, nwin), DGEMM
        out.append(np.ascontiguousarray(0.5 * (P + P.T)))
    return out


def mo_overlaps_via_cp2k(mo_prev, mo_curr, coords_prev_bohr, coords_curr_bohr,
                         lowest_orbital, highest_orbital, params, fragment_atom_lists=None,
                         return_ao_overlap=False):
    """
    All three windowed MO overlap matrices from ONE CP2K-native OVERLAP_DELTAT
    run on the doubled geometry (prev atoms + curr atoms) - no libint needed.

    The doubled AO matrix supplies S(prev,prev), S(prev,curr) and S(curr,curr)
    as blocks; they are mapped from CP2K (MOLog) to libint AO order with the
    molden-derived resort index and contracted with the cached (phase-aligned)
    previous MOs and the raw current MOs. ~1 s per call even for DFT.

    Args:
        mo_prev, mo_curr: load_molden_data dicts (mo_prev["C"] may be aligned)
        coords_*_bohr (array-like, 3N): the two geometries
        lowest_orbital, highest_orbital: 1-based MO window
        params (dict): atom_labels, overlap_template (an es_overlap_deltat_* input
            with CONNECTIVITY OFF and no VDW section), working_directory,
            cp2k_exe / cp2k_data_dir / omp_num_threads / mpi_executable / nprocs
        fragment_atom_lists (list of lists of int, optional): 1-based atom indices
            of disjoint fragments. When given, the Mulliken fragment operators of
            the current geometry are returned as well: P_f(k,l) = sum_{mu in f}
            C_k,mu (S C)_l,mu, symmetrized, over the MO window - their
            eigenvectors are the MOs localized on fragment f, and the fragments
            covering all atoms sum to S_mo_curr. Needs mo_curr["ao_atom"] (the
            wfn loader supplies it).
        return_ao_overlap (bool): also return the AO overlap of the current geometry (the
            S(curr, curr) block of the doubled run, in the MO coefficients' AO order)

    Returns:
        St_mo, S_mo_prev, S_mo_curr  (each (norb_w, norb_w)). When the mo
        dicts carry a beta channel (C_b, from the UKS loaders), each of the
        three is instead a [alpha, beta] list - the AO overlap is
        spin-independent, so the single doubled-geometry run serves both
        channels at no extra CP2K cost. With fragment_atom_lists a fourth item
        follows: the list over fragments of P_f, each shaped like S_mo_curr. With
        return_ao_overlap the AO overlap S22 (n_ao, n_ao) is the last item.

    Raises ValueError when the window does not fit the stored MOs (python
    slicing would silently truncate or wrap), and for tabulated/ZDO
    electronic structures (DFTB, NDDO semi-empirics), whose parameterization
    cannot produce the doubled-geometry CROSS overlaps.
    """
    tmpl_text = _template_text(params["overlap_template"])
    tmpl_methods = set(re.findall(r"^\s*METHOD\s+(\S+)", tmpl_text.upper(), re.M))
    bad = tmpl_methods & {"DFTB", "SE", "AM1", "MNDO", "MNDOD", "PDG", "PDDG",
                          "PM3", "PM6", "PM6-FM", "PNNL", "RM1"}
    if bad:
        raise ValueError(
            F"the OVERLAP_DELTAT route cannot serve tabulated/ZDO electronic "
            F"structures (overlap template METHOD {'/'.join(sorted(bad))}): the "
            F"doubled-geometry run needs the CROSS overlaps <phi(R1)|phi(R2)> at "
            F"near-zero interatomic distances, but Slater-Koster tables are "
            F"zero-padded below their physical range (demonstrated on DFTB/CH2O: "
            F"same-copy blocks exact at 1.3e-14, cross diagonals ~0.27 instead of "
            F"~1) and NDDO never materializes basis functions at all - use xTB "
            F"(analytic GTO-fit overlaps, validated end-to-end) or a GTO method")

    # the run must REQUEST an overlap output, or CP2K completes cleanly and
    # emits nothing (a stock-CP2K template with the patch-only &S_CSR_WRITE
    # block commented out and no dense print is the observed failure)
    has_csr, has_dense = _scan_overlap_template(tmpl_text)
    if not (has_csr or has_dense):
        raise ValueError(
            F"the overlap_template {params['overlap_template']} requests NO "
            F"overlap output - the OVERLAP_DELTAT run would complete cleanly "
            F"and emit nothing (neither an active &S_CSR_WRITE block [needs "
            F"the Libra cp2k patch] nor an active &AO_MATRICES OVERLAP dense "
            F"print [stock CP2K]). Add one of the two to the template, or "
            F"drop overlap_template and set overlap_csr_write=False to "
            F"auto-generate the stock-compatible input")
    for d in (mo_prev, mo_curr):
        for suff in ("", "_b"):
            if F"C{suff}" not in d:
                continue
            nmo = d[F"C{suff}"].shape[0]
            if not 1 <= lowest_orbital <= highest_orbital <= nmo:
                raise ValueError(F"orbital window [{lowest_orbital}, {highest_orbital}] does "
                                 F"not fit the {nmo} stored MOs (1-based, "
                                 F"{'beta' if suff else 'alpha'} channel) - increase ADDED_MOS "
                                 F"in the template or adjust lowest_orbital/highest_orbital")
    atom_labels = params["atom_labels"]
    run_params = dict(params)
    run_params.update({"atom_labels": list(atom_labels) + list(atom_labels),
                       "input_template": params["overlap_template"],
                       "act_state": 0})
    for k in ("ot_template", "force_source", "number_of_states"):
        run_params.pop(k, None)
    q2 = np.concatenate([np.asarray(coords_prev_bohr, dtype=float),
                         np.asarray(coords_curr_bohr, dtype=float)])
    project = run_params.get("project", "libra")
    csr = os.path.join(run_params.get("working_directory", "wd"),
                       f"{project}-S_SPIN_1-1_0.csr")
    # consume only a csr written by THIS run - a stale one (e.g. left
    # by a pre-rebuild patched CP2K) would freeze the AO overlap
    if os.path.exists(csr):
        os.remove(csr)
    logfile = run_cp2k(q2, run_params)
    # a supplied process pool runs the parse outside the GIL - in a chain
    # thread the parse would block every other prefetched chain for its
    # full duration; .result() releases the GIL while the worker parses
    pool = params.get("_parse_pool", None)
    if os.path.exists(csr):        # patched CP2K + &S_CSR_WRITE template: O(nnz)
        # the doubled-geometry overlap is (2 nao)^2 - declaring it makes
        # out-of-range triplets raise at parse time; csr_chunk_rows caps
        # the parse transient (whole-file when unset)
        n2 = 2 * mo_curr["C"].shape[1]
        crows = params.get("csr_chunk_rows", None)
        S = pool.submit(read_cp2k_s_csr, csr, n2, crows).result() if pool \
            else read_cp2k_s_csr(csr, n2, crows)
    else:                          # fallback: dense &AO_MATRICES log print
        S = pool.submit(read_cp2k_ao_overlap_print, logfile).result() if pool \
            else read_cp2k_ao_overlap_print(logfile)
    idx = mo_curr["resort_idx"]
    if idx is None:                       # MOs already in MOLog AO order
        idx = np.arange(S.shape[0] // 2)
    n = len(idx)
    if S.shape[0] != 2 * n:
        raise ValueError(f"doubled overlap is {S.shape[0]}x, expected {2*n}")
    ix = np.ix_(idx, idx)
    S11, S12, S22 = S[:n, :n][ix], S[:n, n:][ix], S[n:, n:][ix]
    lo, hi = lowest_orbital - 1, highest_orbital
    frag_cols = None
    if fragment_atom_lists is not None:
        ao_atom = mo_curr.get("ao_atom", None)
        if ao_atom is None:
            raise ValueError("fragment operators need the AO->atom map of the wfn "
                             "loader (mo_source='wfn') - the molden route has none")
        natom = int(np.max(ao_atom)) + 1
        seen = set()
        frag_cols = []
        for f, atoms in enumerate(fragment_atom_lists):
            atoms = [int(a) for a in atoms]
            if not atoms or min(atoms) < 1 or max(atoms) > natom:
                raise ValueError(F"fragment {f} atom list {atoms} is empty or outside "
                                 F"1..{natom} (1-based atom indices)")
            if seen & set(atoms):
                raise ValueError(F"fragment {f} shares atoms {sorted(seen & set(atoms))} "
                                 F"with an earlier fragment - fragments must be disjoint")
            seen |= set(atoms)
            # column j of C is the AO idx[j] of the file order
            frag_cols.append(np.flatnonzero(np.isin(ao_atom[idx] + 1, atoms)))
    # window BEFORE contracting: O(win * nao^2) instead of O(nao^3) -
    # 20-100x cheaper already at a few thousand AOs. The AO overlap is
    # spin-independent, so the same S blocks serve every spin channel.
    # MATRIX products (DGEMM) through the bulk transfer, the real parts of
    # the inputs as before
    S11_m, S12_m, S22_m = (data_conv.nparray2MATRIX(np.asarray(b)) for b in (S11, S12, S22))
    St, S_prev, S_curr, P_frag = [], [], [], []
    for suff in ("", "_b"):
        if F"C{suff}" not in mo_prev or F"C{suff}" not in mo_curr:
            continue
        Cp, Cc = mo_prev[F"C{suff}"][lo:hi], mo_curr[F"C{suff}"][lo:hi]
        Cp_m, Cc_m = data_conv.nparray2MATRIX(np.asarray(Cp)), data_conv.nparray2MATRIX(np.asarray(Cc))
        St.append(data_conv.MATRIX2nparray((Cp_m * S12_m) * Cc_m.T(), np.float64))
        S_prev.append(data_conv.MATRIX2nparray((Cp_m * S11_m) * Cp_m.T(), np.float64))
        S_curr.append(data_conv.MATRIX2nparray((Cc_m * S22_m) * Cc_m.T(), np.float64))
        if frag_cols is not None:
            P_frag.append(mulliken_fragment_operators(Cc, S22, frag_cols))
    if frag_cols is not None:
        # fragment-major, each entry shaped like S_curr
        P_frag = [[P_spin[f] for P_spin in P_frag] for f in range(len(frag_cols))]
        out = (St[0], S_prev[0], S_curr[0], [P[0] for P in P_frag]) if len(St) == 1 else (St, S_prev, S_curr, P_frag)
    else:
        out = (St[0], S_prev[0], S_curr[0]) if len(St) == 1 else (St, S_prev, S_curr)
    if return_ao_overlap:
        out = out + (np.ascontiguousarray(S22),)
    return out


def load_newtonx_mo_data(filename):
    """
    MO data from the &NAMD_PRINT file - the molden-free source (requires
    PRINT_VIRTUALS T and PRINT_PHASES T in the template).

    Returns the same dict layout as load_molden_data minus libint shells:
    C (nmo x nao, CP2K MOLog AO order - pairs directly with the OVERLAP_DELTAT
    matrix, no reordering), E (Hartree), homo (= the occupied count of the
    print, exactly the SCF occupation), resort_idx = None (signals MOLog
    order to mo_overlaps_via_cp2k), and the parsed "newtonx" dict so the CI
    amplitudes need not be re-parsed. For UKS files the beta channel comes
    back as C_b/E_b/homo_b with nspin = 2 (the load_wfn_mo_data convention).
    """
    nx = read_cp2k_newtonx_output(filename)
    if nx["C_occ"] is None or nx["C_virt"] is None:
        raise ValueError(f"{filename} has no MO blocks - set PRINT_VIRTUALS T")
    C = np.vstack([nx["C_occ"].T, nx["C_virt"].T])      # (nmo, nao), MOLog order
    E = np.concatenate([nx["E_occ"], nx["E_virt"]])
    out = {"C": C, "E": E, "homo": nx["C_occ"].shape[1],
           "resort_idx": None, "newtonx": nx, "nspin": nx["nspin"]}
    if nx["nspin"] == 2:
        out["C_b"] = np.vstack([nx["C_occ_b"].T, nx["C_virt_b"].T])
        out["E_b"] = np.concatenate([nx["E_occ_b"], nx["E_virt_b"]])
        out["homo_b"] = nx["C_occ_b"].shape[1]
    return out


def _wfn_ao_atom_map(basis):
    """
    0-based atom index of every AO, in the CP2K-native AO order, from the
    RESTART.wfn header (natom, nspin, nao, nset_max, nshell_max, nset_info,
    nshell_info, nso_info - read_wfn_file). The Fortran arrays are stored
    column-major: nshell_info(nset_max, natom), nso_info(nshell_max, nset_max, natom).
    """
    natom, nao, nset_max, nshell_max = int(basis[0]), int(basis[2]), int(basis[3]), int(basis[4])
    nset_info = np.asarray(basis[5], dtype=int)
    nshell_info = np.asarray(basis[6], dtype=int).reshape(nset_max, natom, order="F")
    nso_info = np.asarray(basis[7], dtype=int).reshape(nshell_max, nset_max, natom, order="F")
    ao_atom = []
    for a in range(natom):
        for iset in range(nset_info[a]):
            for ishell in range(nshell_info[iset, a]):
                ao_atom += [a] * int(nso_info[ishell, iset, a])
    if len(ao_atom) != nao:
        raise ValueError(F"wfn header inconsistency: the shell structure yields "
                         F"{len(ao_atom)} AOs but nao = {nao}")
    return np.asarray(ao_atom, dtype=int)


def _fortran_records(path):
    """The records of a gfortran unformatted sequential file (4-byte length markers)."""
    data = open(path, "rb").read()
    out, pos = [], 0
    while pos < len(data):
        n = int(np.frombuffer(data[pos:pos + 4], dtype="<i4")[0])
        out.append(data[pos + 4:pos + 4 + n])
        pos += 8 + n
    return out


def _records_to_matrix(records):
    """A matrix written by cp_fm_write_unformatted: one record per column."""
    return np.stack([np.frombuffer(r, dtype="<f8") for r in records], axis=1)


def read_pod_cp2k_files(prefix, need_tmf=False):
    """
    The binary files of CP2K's projection-operator diabatization (Z. Futera's &PROPERTIES
    &ET_COUPLING &PROJECTION with the HAMILTONIAN, WAVEFUNCTION and TRANSF_MATRIX_R/F prints):
    gfortran unformatted sequential records, matrices as one record per column.

    Args:
        prefix (str): the files' path without extension, e.g. "Diag_50-POD-RESTART-1_0"
        need_tmf (bool): also read the forward transformation S^-1/2 (the inverse of tmr)

    Returns:
        dict: n_spins, n_blocks, n_ao_blk (AOs per block), tmr (S^+1/2, (n_ao, n_ao)), tmf (or
            None), S (the AO overlap tmr^2), energies {(block, spin): block-state energies
            [Ha]}, couplings {(block_i, block_j, spin): H'_ij in the block-state basis [Ha],
            (nmo_i, nmo_j)}, coeffs {(block, spin): block MOs in the AO basis, (n_ao, nmo)}
    """
    hab = _fortran_records(F"{prefix}.hab")
    wfn = _fortran_records(F"{prefix}.wfn")
    tmr = _records_to_matrix(_fortran_records(F"{prefix}.tmr"))
    tmf = _records_to_matrix(_fortran_records(F"{prefix}.tmf")) if need_tmf else None
    n_spins = int(np.frombuffer(hab[0], "<i4")[0])
    n_blocks = int(np.frombuffer(hab[1], "<i4")[0])
    n_ao_blk = np.frombuffer(hab[2], "<i4").tolist()
    if np.frombuffer(wfn[2], "<i4").tolist() != n_ao_blk:
        raise ValueError(F"{prefix}: the .hab and .wfn headers disagree on the block sizes")
    k = 3
    energies = {}
    for ib in range(n_blocks):
        for sp in range(n_spins):
            energies[(ib, sp)] = np.frombuffer(hab[k], "<f8").copy()
            k += 1
    couplings = {}
    for i in range(n_blocks):
        for j in range(i + 1, n_blocks):
            for sp in range(n_spins):
                nmo_j = len(energies[(j, sp)])
                couplings[(i, j, sp)] = _records_to_matrix(hab[k:k + nmo_j])
                k += nmo_j
    if k != len(hab):
        raise ValueError(F"{prefix}.hab: {len(hab)} records, {k} expected from the header")
    k = 3
    coeffs = {}
    for ib in range(n_blocks):
        for sp in range(n_spins):
            nmo = len(energies[(ib, sp)])
            coeffs[(ib, sp)] = _records_to_matrix(wfn[k:k + nmo])
            k += nmo
    if k != len(wfn):
        raise ValueError(F"{prefix}.wfn: {len(wfn)} records, {k} expected from the header")
    return {"n_spins": n_spins, "n_blocks": n_blocks, "n_ao_blk": n_ao_blk, "tmr": tmr,
            "tmf": tmf, "S": tmr @ tmr, "energies": energies, "couplings": couplings,
            "coeffs": coeffs}


def pod_cp2k_fragment_operators(pod, C_win, spin=0):
    """The POD fragment projectors over an MO window - libra_py.pod.pod_fragment_operators
    on the read_pod_cp2k_files output (the same dict keys as the native pod_block_states)."""
    from libra_py import pod as _pod
    return _pod.pod_fragment_operators(pod, C_win, spin)


def load_wfn_mo_data(filename):
    """
    MO data from CP2K's binary RESTART.wfn - the zero-cost native source
    (the file is already written every step for restart chaining), valid for
    BOTH ground-state-only and TDDFPT runs, at binary precision.

    Coefficients are in CP2K-native AO order (resort_idx = None -> pairs with
    the OVERLAP_DELTAT matrix; requires ao_overlap_method="cp2k"). Verified:
    C S C^T = I to 3.5e-12 against the native overlap. Per-MO global signs may
    differ from molden/NAMD_PRINT MOs - irrelevant, the driver's cumulative
    phase alignment handles them.

    Returns:
        dict: C (nmo, nao), E (Hartree, exact), occ (occupation numbers),
              homo (1-based index from the file itself), resort_idx = None,
              nspin. UKS files (nspin=2) return the beta channel alongside as
              C_b / E_b / occ_b / homo_b (alpha and beta HOMOs differ in
              general - the file stores both), ao_atom (0-based atom index of
              every AO in the file's AO order - the fragment operators of
              mo_overlaps_via_cp2k need it).
    """
    basis, spin, evocc, moc = read_wfn_file(filename)
    out = {"resort_idx": None, "nspin": len(spin), "ao_atom": _wfn_ao_atom_map(basis)}
    for i, suff in enumerate(("", "_b")[:len(spin)]):
        nmo = int(spin[i][0])
        vals = np.asarray(evocc[i], dtype=np.float64)
        out[F"C{suff}"] = np.array(moc[i], dtype=float)   # (nmo, nao), CP2K order
        out[F"E{suff}"] = vals[:nmo].copy()
        out[F"occ{suff}"] = vals[nmo:2 * nmo].copy()
        out[F"homo{suff}"] = int(spin[i][1])
    return out


def ci_data_from_newtonx(newtonx, omegas_Ha, ci_threshold=1e-6,
                         occ_min=None, vir_max=None):
    """
    Convert NAMD_PRINT excitation matrices into the read_ci_data dict layout
    (configs with ABSOLUTE 1-based orbital indices, thresholded + renormalized
    amplitudes) so it can feed the SD/citools machinery unchanged. For UKS
    parses (newtonx["X_b"] present) each state merges its alpha and beta
    configs ('alp'/'bet' spin tags, per-channel absolute orbital indices) and
    is renormalized over the combined vector - the convention of the CP2K
    excitation vectors themselves.

    Args:
        newtonx: read_cp2k_newtonx_output result
        omegas_Ha (array): excitation energies, Hartree (from the log table)
        ci_threshold: weight cutoff |C|^2 > ci_threshold, applied to
            the exact vectors
        occ_min, vir_max: clip configs to the MO window [occ_min, vir_max]
            (absolute 1-based; out-of-window amplitudes are dropped before
            renormalization - they cannot be represented in the windowed
            overlap machinery anyway)

    The returned dict also carries window_drops (np.ndarray, per state): the
    CI weight lost to the threshold + window clip relative to the full exact
    vector - the renormalization hides it from any later norm.
    """
    configs, amps, spins = [], [], []
    drops = np.zeros(len(newtonx["X"]))
    X_b = newtonx.get("X_b")
    for ist, M in enumerate(newtonx["X"]):
        channels = [(M, "alp")] + ([(X_b[ist], "bet")] if X_b is not None else [])
        conf, a, sp = [], [], []
        total2 = 0.0
        for Mc, tag in channels:
            nvirt, nocc = Mc.shape
            total2 += float(np.sum(Mc ** 2))
            sel = np.argwhere(Mc ** 2 > ci_threshold)
            if occ_min is not None:
                sel = sel[sel[:, 1] + 1 >= occ_min]
            if vir_max is not None:
                sel = sel[nocc + sel[:, 0] + 1 <= vir_max]
            conf += [[int(c) + 1, int(nocc + r) + 1] for r, c in sel]  # [occ, vir] absolute
            a += [Mc[r, c] for r, c in sel]
            sp += [tag] * len(sel)
        a = np.array(a, dtype=float)
        if a.size == 0:
            raise ValueError(F"ci_data_from_newtonx: state {ist + 1} retains no "
                             F"configs above the threshold inside the window")
        drops[ist] = 1.0 - float(np.sum(a ** 2)) / total2
        if drops[ist] > 1e-6:
            print(F"ci_data_from_newtonx: state {ist + 1}: dropped CI weight "
                  F"{drops[ist]:.2e} to the threshold/MO-window clip "
                  F"(vector renormalized)")
        a /= np.linalg.norm(a)
        configs.append(conf)
        amps.append(a)
        spins.append(sp)
    return {"E": np.asarray(omegas_Ha)[:len(configs)],
            "configs": configs, "amps": amps, "spins": spins,
            "window_drops": drops}


class tmp:
    pass


def cp2k_compute_adi_nbra(q, params, full_id):
    """
    Compute adiabatic electronic properties and derivative couplings
    for a single trajectory using CP2K TDDFPT outputs and CI overlap analysis.

    This function performs a single electronic structure evaluation step in a
    trajectory-based nonadiabatic dynamics simulation. It constructs adiabatic
    and vibronic Hamiltonians, evaluates time-overlap matrices between electronic
    states at consecutive time steps, and estimates derivative couplings using
    finite-difference time overlaps.

    The implementation assumes that CP2K TDDFPT calculations have already been
    executed externally and that all required output files (log files and MO
    overlap matrices) are available.

    Parameters
    ----------
    q : MATRIX
        Nuclear coordinates for all trajectories.
        Shape: (3*N_atoms, N_trajectories)
        Units: Bohr

    params : dict
        Dictionary containing simulation parameters and trajectory-specific data.

        Required keys
        -------------
        nstates : int
            Number of adiabatic electronic states.

        logfile_name : str
            Path to CP2K TDDFPT log file. May carry the format fields {timestep}
            / {timestep_st}, filled from the engine's per-step model_params;
            a fixed path is honored (the caller rewrites it per step) but its
            reuse across two engine steps raises.

        time_overlap_filename : str
            Path to sparse MO overlap matrix file (.npz format); same
            {timestep}/{timestep_st} templating and reuse guard as logfile_name
            ({timestep_st} = the interval start the St pairs with).

        overlap_filename : str
            Path to the sparse MO self-overlap file (.npz); same templating and
            reuse guard.

        atom_labels : list of str
            Atomic symbols (currently not used directly in this function).

        lowest_orbital : int
            Lowest orbital index included in the overlap matrix (1-based indexing).

        Optional keys
        -------------
        dt : float, default=41.0
            Nuclear time step (atomic units).

        ci_threshold : float, default=1e-6
            CI truncation, |C|^2 > ci_threshold.

        nelec_act_space : int, optional
            Number of electrons in the active space. If not provided,
            the full orbital space is used.

        working_directory_prefix : str, default="wd"
            Prefix for trajectory-specific working directories.

        Internal state (updated in-place)
        --------------------------------
        data_prev : dict
            Previous CI/electronic structure data per trajectory.

        coordinates_prev : dict
            Previous nuclear coordinates per trajectory.

        is_first_time : dict
            Flags indicating first step per trajectory.

        act_state : dict
            Active electronic state per trajectory (currently only used in commented sections).

    full_id : int or object
        Encoded trajectory identifier. The last element corresponds to `itraj`.

    Returns
    -------
    obj : tmp
        Container object with computed electronic properties.

        Attributes
        ----------
        ham_adi : CMATRIX (nstates, nstates)
            Adiabatic Hamiltonian (diagonal energies).

        hvib_adi : CMATRIX (nstates, nstates)
            Vibronic Hamiltonian including derivative coupling terms:
                H_ij = E_i δ_ij - i d_ij

        time_overlap_adi : CMATRIX (nstates, nstates)
            Time-overlap matrix:
                S_ij = <Ψ_i(t) | Ψ_j(t + dt)>

        basis_transform : CMATRIX (nstates, nstates)
            Basis transformation matrix (identity).

        nac_adi : CMATRIX (nstates, nstates)
            Placeholder for nonadiabatic couplings (currently unused).

        d1ham_adi : CMATRIXList (length = ndof)
            Derivatives of Hamiltonian w.r.t. nuclear coordinates
            (currently zero-filled; forces code is commented out).

        dc1_adi : CMATRIXList (length = ndof)
            Derivative couplings per nuclear degree of freedom
            (currently zero-filled unless NACV block is enabled).

    Notes
    -----
    - Ground state energy is set to zero; excited-state energies are averaged
      between previous and current time steps.

    - Time-overlaps are computed at the CI level using molecular orbital overlaps
      and CI coefficients from consecutive steps.

    - Derivative couplings are approximated via:
          d_ij = (S_ij - S_ji) / (2 * dt)

      and included in the vibronic Hamiltonian as:
          H_ij = -i * d_ij  (off-diagonal terms)

    - Active space selection:
        * If `nelec_act_space` is None → full orbital space is used.
        * Otherwise, a truncated space around HOMO is constructed.

    - The function assumes:
        * Closed-shell (restricted) calculations (`isUKS=False`)
        * Real-valued CI overlaps

    - NACVs (analytic derivative couplings) and forces are currently disabled
      but can be enabled via the commented sections.

    - All trajectory-specific data is stored in `params` to support
      multi-trajectory simulations.

    Limitations
    -----------
    - No phase correction is applied to CI states.
    - Overlap matrix is assumed to be correctly aligned (no orbital reordering).
    - No error handling for missing files (e.g., overlap matrix).
    - Vibronic Hamiltonian neglects diagonal derivative terms.

    Example
    -------
    >>> obj = cp2k_compute_adi(q, params, full_id)
    >>> print(obj.ham_adi)
    >>> print(obj.time_overlap_adi)
    >>> print(obj.hvib_adi)
    """
    # ================= Decode trajectory index =================
    Id = Cpp2Py(full_id)
    itraj = Id[-1]

    # ================= Extract coordinates =================
    coords = q.col(itraj)
    coordinates = data_conv.MATRIX2nparray(coords, float)

    ndof = coords.num_of_rows
    nat = ndof // 3

    # ================= Safe param access =================
    params.setdefault("MO_prev", {})
    params.setdefault("data_prev", {})
    params.setdefault("coordinates_prev", {})
    params.setdefault("s_ci_inv_prev", {})
    params.setdefault("is_first_time", {})
    params.setdefault("act_state", {})

    is_first_time = params["is_first_time"].get(itraj, True)
    act_state = params["act_state"].get(itraj, 0)

    if not is_first_time:
        missing = [key for key in ("data_prev", "coordinates_prev", "s_ci_inv_prev") if itraj not in params[key]]
        if missing:
            raise RuntimeError(F"in cp2k_compute_adi_nbra - trajectory {itraj} is past its first step but the "
                               F"carried state {missing} is gone: the same params dict must persist across all "
                               F"calls of a trajectory")

    nelec_act_space = params.get("nelec_act_space", None)
    lowest_orbital = params["lowest_orbital"]

    # ================= Read parameters =================
    dt = float(params.get("dt", 41.0))
    atom_labels = params["atom_labels"]

    wd_prefix = params.get("working_directory_prefix", "wd")
    wd = f"{wd_prefix}_itraj{itraj}"

    # ================= Run CP2K =================
    # Currently - we just take the external files

    # ================= Previous coordinates =================
    if is_first_time:
        coordinates_prev = coordinates.copy()
    else:
        coordinates_prev = params["coordinates_prev"][itraj].copy()

    # ================= Read electronic structure =================
    nstates = params["nstates"]

    logfile_name = _nbra_step_filename(params, "logfile_name", itraj)
    time_overlap_filename = _nbra_step_filename(params, "time_overlap_filename", itraj)

    read_params = {
        "number_of_states": nstates,
        "logfile_name": logfile_name,
        "ci_threshold": params.get("ci_threshold", 1e-6),
        "isUKS": False,
    }

    info, data_curr = read_cp2k_tddfpt_log_file(read_params)
    # print(f"info = {info}")

    # ================= Previous electronic data =================
    if is_first_time:
        data_prev = copy.deepcopy(data_curr)
    else:
        data_prev = params["data_prev"][itraj]

    # ================= Build object =================
    obj = tmp()
    obj.ham_adi = CMATRIX(nstates, nstates)
    obj.nac_adi = CMATRIX(nstates, nstates)
    obj.hvib_adi = CMATRIX(nstates, nstates)
    obj.basis_transform = CMATRIX(nstates, nstates)
    obj.time_overlap_adi = CMATRIX(nstates, nstates)
    obj.overlap_adi = CMATRIX(nstates, nstates)

    # ================= Compute overlaps =================
    st_mo = sp.load_npz(time_overlap_filename)
    
    if sp.issparse(st_mo):
        st_mo = st_mo.toarray()  # FIX: no dtype argument allowed
    else:
        st_mo = np.asarray(st_mo)
    st_mo = st_mo.astype(np.float64, copy=False)

    s_mo = sp.load_npz(_nbra_step_filename(params, "overlap_filename", itraj)).todense()

    
    ndim = st_mo.shape[0] // 2
    
    #print(f"ndim = {ndim}")

    orbital_space = list(range(lowest_orbital, lowest_orbital + ndim))

    # ================= Construct active space =================
    if nelec_act_space is None:
        active_space = list(orbital_space)
    else:
        min_indx = info["nocc"] - nelec_act_space // 2 + 1

        if min_indx > info["min_occ"]:
            min_elec = ((info["nocc"] - info["min_occ"]) + 1) * 2
            raise ValueError(f"The `nelec_act_space` should be at least {min_elec}")

        active_space = list(range(min_indx, info["max_vir"] + 1))

    # ================= Compute CI time-overlaps =================
    highest_orbital = lowest_orbital + ndim - 1

    ovlp_params = {
        "homo_indx": info["nocc"],
        "nocc": info["nocc"] - lowest_orbital,
        "nvirt": highest_orbital - info["nocc"],
        "nelec": (info["nocc"] - lowest_orbital + 1) * 2,
        "nstates": nstates,
        "active_space": active_space,
        "orbital_space": orbital_space,
    }

    # print(f"orbital_space: {orbital_space}")
    # print(f"active_space: {active_space}")

    st_ci = ci.overlap(st_mo, data_prev, data_curr, ovlp_params)

    s_ci = ci.overlap(s_mo, data_curr, data_curr, ovlp_params)
    
    s_ci_inv_curr = ortho.lowdin_inverse_sqrt(s_ci)
    s_ci_inv_prev = None
    if is_first_time:
        s_ci_inv_prev = copy.deepcopy(s_ci_inv_curr)
    else:
        stored = params["s_ci_inv_prev"][itraj]
        if stored["digest"] != ortho.carried_state_digest(data_prev):
            raise RuntimeError(F"in cp2k_compute_adi_nbra - the stored Lowdin factor for trajectory {itraj} was "
                               F"not built alongside this call's previous-step data: the carried params state is "
                               F"out of sync between calls")
        s_ci_inv_prev = stored["s_ci_inv"]
    
    s_ci = s_ci_inv_curr @ s_ci @ s_ci_inv_curr
    st_ci = s_ci_inv_prev @ st_ci @ s_ci_inv_curr



    # ================= Populate Hamiltonian =================
    for i in range(nstates):

        energy = 0.0 if i == 0 else 0.5 * (
            data_prev[0][i - 1] + data_curr[0][i - 1]
        )

        obj.ham_adi.set(i, i, energy)
        obj.hvib_adi.set(i, i, energy)
        obj.basis_transform.set(i, i, 1.0)

        for j in range(nstates):
            obj.time_overlap_adi.set(i, j, float(st_ci[i, j]))
            obj.overlap_adi.set(i,  j, float(s_ci[i,j]) )

    # ================= Forces =================
    obj.d1ham_adi = CMATRIXList()
    for idof in range(ndof):
        obj.d1ham_adi.append(CMATRIX(nstates, nstates))

    # for iatom in range(nat):
    #     obj.d1ham_adi[3 * iatom + 0].set(act_state, act_state, -forces[0, iatom] * (1.0 + 0.0j))
    #     obj.d1ham_adi[3 * iatom + 1].set(act_state, act_state, -forces[1, iatom] * (1.0 + 0.0j))
    #     obj.d1ham_adi[3 * iatom + 2].set(act_state, act_state, -forces[2, iatom] * (1.0 + 0.0j))

    # ================= Derivative couplings containers =================
    obj.dc1_adi = CMATRIXList()
    for idof in range(ndof):
        obj.dc1_adi.append(CMATRIX(nstates, nstates))

    # if os.path.exists(f"{wd}/NACV.DAT"):
    #     nacv = read_nacv(f"{wd}/NACV.DAT")
    #     for iatom in range(nat):
    #         for i in range(nstates):
    #             for j in range(nstates):
    #                 obj.dc1_adi[3 * iatom + 0].set(i, j, nacv[i, j, 0, iatom] * (1.0 + 0.0j))
    #                 obj.dc1_adi[3 * iatom + 1].set(i, j, nacv[i, j, 1, iatom] * (1.0 + 0.0j))
    #                 obj.dc1_adi[3 * iatom + 2].set(i, j, nacv[i, j, 2, iatom] * (1.0 + 0.0j))

    # ================= Compute derivative couplings =================
    for i in range(nstates):
        for j in range(i + 1, nstates):

            dij = (
                obj.time_overlap_adi.get(i, j)
                - obj.time_overlap_adi.get(j, i)
            ) / (2.0 * dt)

            obj.hvib_adi.set(i, j, -1j * dij)
            obj.hvib_adi.set(j, i, +1j * dij)

    # ================= Store state =================
    # params["MO_prev"][itraj] = MO_curr.copy()
    params["data_prev"][itraj] = copy.deepcopy(data_curr)
    params["coordinates_prev"][itraj] = coordinates.copy()
    params["s_ci_inv_prev"][itraj] = {"digest": ortho.carried_state_digest(data_curr),
                                      "s_ci_inv": copy.deepcopy(s_ci_inv_curr)}
    params["is_first_time"][itraj] = False

    return obj


def _nbra_step_filename(params, key, itraj):
    """Per-step input path of the NBRA driver: the {timestep}/{timestep_st}
    fields are filled from the engine's model_params; a fixed path may not be
    consumed twice in a row for one trajectory (the file would be stale)."""
    template = params[key]
    fields = {k: params[k] for k in ("timestep", "timestep_st") if k in params}
    name = template.format(**fields) if ("{" in template) else template
    seen = params.setdefault("_nbra_last_input", {}).setdefault(key, {})
    if seen.get(itraj) == name and "{" not in template:
        raise RuntimeError(F"{key} = {name!r} was already consumed for trajectory "
                           F"{itraj} on the previous step - a fixed path must be "
                           F"rewritten per step, or carry {{timestep}}/{{timestep_st}}")
    seen[itraj] = name
    return name


def _carried(params, key, itraj, current, is_first_time):
    """Per-trajectory carried state; a missing entry after the first step
    means the params dict was rebuilt between calls - substituting the
    current step would silently zero the time-overlap channel."""
    if is_first_time:
        return current
    store = params.get(key)
    if store is None or itraj not in store:
        raise RuntimeError(
            F"cp2k_compute_adi: carried state '{key}' for trajectory {itraj} "
            F"is missing at timestep {params.get('timestep', '?')} - params "
            F"must persist between driver calls")
    return store[itraj]


def cp2k_compute_adi_impl(q, params, full_id):
    """
    Compute adiabatic properties for one trajectory at one time step, on the fly.

    Args:
        q (MATRIX 3N x ntraj): coordinates of all trajectories, Bohr
        params (dict): model parameters + per-trajectory caches (mutated in place):

            Required:
                atom_labels (list of str), input_template (str), cp2k_exe (str),
                nstates (int): TOTAL number of states incl. ground state
                lowest_orbital, highest_orbital (int): MO window (1-based, absolute)
                num_occ_states, num_unocc_states (int): SD basis window around HOMO
                                                (backend A; must cover the CI configs)
            Optional:
                apply_orthonormalization: normalization of the SD-level
                    overlaps (same name/meaning as in step3's
                    run_step3_sd_nacs_libint): "lowdin"/True [default],
                    "lowdin_cached" (bra factor carried between steps, the
                    cp2k_compute_adi_nbra convention - numerically identical
                    here, one fractional matrix power cheaper), "none"/False
                    (raw determinants - the historical dftb_compute_adi
                    behavior; upstream moved that driver to Lowdin in 2026-04)
                sd_overlap_kernel (str): SD determinant kernel, same values
                    as step3: "auto" [default] | "batched" | "lemma" | "per_pair"
                align_mo_phases (bool): cumulative MO-phase alignment of the
                    orbitals between steps [default: True]. Disabling is
                    incompatible with ci_source="namd_print" (raises) - the
                    NAMD_PRINT amplitudes are re-signed against aligned MOs.
                    NOTE the step3/on-the-fly vocabulary correspondence:
                    step3's apply_phase_correction (state signs of the St
                    sequence) corresponds to the DYNAMICS-level
                    dyn_params["do_phase_correction"], NOT to this flag;
                    step3's do_state_reordering corresponds to the dynamics
                    dyn_params["state_tracking_algo"]; step3's ci_threshold
                    is the same keyword here (unified weight cut, shared
                    with dftb_compute_adi and cp2k_compute_adi_nbra)
                time_dependent (bool): True (default) = TDDFPT CI states with
                    per-state forces. False = ground-state-only mode: states are
                    the fixed KS-SD basis, energies E_gs + orbital gaps, GS
                    forces on every diagonal (needs a template WITHOUT TDDFPT /
                    EXCITED_STATES; nstates must be 1 + num_occ_states*num_unocc_states;
                    sd_state_ordering: "energy" (default) or "identity")
                ao_overlap_method (str): "cp2k" (default) = OVERLAP_DELTAT
                    doubled-geometry run per step (uses overlap_template; when
                    absent, one is auto-generated from input_template via
                    make_overlap_deltat_input - set overlap_csr_write=False for
                    stock CP2K; exact AO overlaps, periodic-capable; the CSR
                    fast path needs the cp2k patch, stock CP2K falls back to
                    the dense log print); "libint" = molden+libint in-process
                    (no extra CP2K run, nonperiodic)
                ot_first_call_only (bool): run the ot_template pre-convergence
                    only on each trajectory's first call - later calls restart
                    the diagonalization from their own previous-step wfn
                    [default: False]
                prefetch_trajectories (int): 1 = submit every trajectory's
                    chain to a thread pool on the step's first model call and
                    harvest per engine call, overlapping the CP2K runs
                    [default: 0]. Needs per-trajectory engine calls (raises
                    under isNBRA); the pools live in a module registry keyed
                    by working_directory_prefix
                max_concurrent_chains (int): concurrent chain cap; 0 [default]
                    = allocated physical cores // (nprocs * omp_num_threads)
                parse_pool_workers (int): fork-process pool size for the
                    overlap parses under prefetch; 0 disables
                    [default: the chain cap]
                csr_chunk_rows (int): parse the ODT CSR overlap in chunks of
                    this many triplet rows, capping the parse transient
                    [default: None = whole file]
                force_source (str): what forces CP2K supplies (the engine-side
                    dyn_params["force_method"] decides how they are used):
                    "active_state" [default] - one Z-vector run at the hopping
                    state (pairs with force_method=1, TSH);
                    "all_states" - mechanism B: template without
                    &EXCITED_STATES but with &TDDFPT &PRINT &FORCES (LIST is
                    rewritten from number_of_states); forces for ALL states parsed
                    from the .tdfrc file and placed on every d1ham_adi diagonal
                    (pairs with force_method=2 - Ehrenfest - or
                    force-difference rescaling);
                    "none" - RUN_TYPE is rewritten to ENERGY and no forces are
                    computed (pairs with force_method=0)
                    "gs" - CP2K always runs at the GROUND state (any
                    &EXCITED_STATES section is dropped) while TDDFPT
                    energies/overlaps are computed as usual; the GS forces go
                    on EVERY d1ham_adi diagonal, so nuclei follow S0
                    regardless of the hopping state (classical-path NBRA;
                    pairs with force_method=1 + the S0 forces)
                mo_source (str): "wfn" (default) = binary RESTART.wfn -
                    zero cost, binary precision, works in GS-SD mode too;
                    "namd_print" (TDDFPT print file); "molden" (ASCII molden
                    print - needs &MO_MOLDEN in the template, MO precision
                    capped by the print digits). wfn/namd_print need
                    ao_overlap_method="cp2k"
                ci_source (str): "namd_print" (default) = full-precision
                    X vectors from the &TDDFPT &PRINT &NAMD_PRINT file (the
                    template must carry that print with PRINT_VIRTUALS T);
                    "log" = excitation-analysis block (~5 digits, works with
                    any TDDFPT template)
                UKS (open shell): detected from the CP2K output itself (two
                    spin channels) - no flag. The SD basis becomes per-channel
                    electron-only excitations around each spin's own HOMO,
                    sized by num_occ_alpha/num_unocc_alpha (default
                    num_occ_states/num_unocc_states) and num_occ_beta/
                    num_unocc_beta (default = alpha) - the step3 SD
                    vocabulary; in GS-SD mode nstates = 1 + the summed
                    per-channel excitation counts. Requires
                    ao_overlap_method="cp2k" and overlap_backend="sd"
                overlap_backend (str): "sd" (default) or "citools"
                    (spin-adapted singlet CSFs, restricted-only)
                dia_basis (str): "none" (default) or "sd" - declare the SD
                    basis as the DIABATIC basis of the dynamics: the driver
                    then returns basis_transform = the sd2ci matrix (TDDFPT;
                    the energy-sort permutation in GS-SD mode) and
                    ovlp_dia = identity instead of the identity transform.
                    Enables engine-side diabatic initialization
                    (elec_params rep:0 with istate = an SD index: 0 = the
                    reference determinant, k = sd_unique[k-1]; set
                    elec_params ndia = 1 + num_occ*num_unocc SDs, nadi =
                    nstates) and per-step SD populations (Cdia, se_pop_dia).
                    The initial adiabatic surface is sampled from
                    |T[istate, :]|^2 (auto-renormalized); the neglected
                    weight equals the SD's CI-space leakage. Requires
                    overlap_backend="sd"
                dia_basis_max_drop (float): with dia_basis="sd", the largest
                    CI weight a PROPAGATED state may carry outside the SD
                    basis (the sd2ci dropped weight, checked at every step) -
                    beyond it the renormalized T column is noise and the
                    driver raises; widen num_occ_states/num_unocc_states or
                    reduce nstates to satisfy the contract    [default 0.5]
                occupation_threshold (float): a population up to this fraction
                    of the full occupation on an orbital above the electron-
                    count reference (or missing from one below it) is treated
                    as zero: CP2K's printed partition is re-mapped onto the
                    reference and the excitations from the over-counted orbital
                    become reported dropped weight (into sd_capture_max_drop).
                    A larger population is a fractional reference and the
                    driver raises. Set the template's EPS_FERMI_DIRAC to the
                    same value so CP2K's excitation space keeps HOMO->LUMO for
                    populations libra treats as zero; the population check
                    runs only where the log prints MO occupation numbers (&DFT
                    &PRINT &MO OCCUPATION_NUMBERS T)              [default 0.01]
                sd_capture_max_drop (float): the largest total CI weight a
                    PROPAGATED state may lose to the truncation chain the CI
                    overlaps are built from (the read-stage threshold/
                    MO-window clip combined with the sd2ci dropped weight;
                    read-stage only for overlap_backend="citools"; checked
                    at every step) - beyond it the renormalized state
                    poisons st_ci and the driver raises; widen
                    num_occ_states/num_unocc_states (and the MO window) or
                    reduce nstates to satisfy the contract    [default 0.5]
                ci_threshold (float): config retention cut,
                    |amp|^2 > ci_threshold on every route (the log route
                    additionally sits on the 5e-2 print floor)
                                                             [default 1e-6]
                act_state (dict itraj->int): synced by run_dynamics each step
                working_directory_prefix (str)                   [default "wd"]
                cp2k_data_dir, omp_num_threads, mpi_executable, nprocs,
                number_of_states: passed to the execution layer
                    (number_of_states defaults to nstates-1)
                dia_basis_hvib (str): "none" [default] or "corrected" -
                    with dia_basis="sd", also supply obj.hvib_dia =
                    H_SD - i D_SD (the SD-frame HST drift correction);
                    consume with hvib_update_method 0 +
                    ham_transform_method 3


            Caches (auto-created): mo_prev, ci_prev, E_adi_prev, S_prev,
                homo, sd_basis, coordinates_prev, is_first_time

        full_id: Libra trajectory id; last element = itraj

    Returns:
        obj with CMATRIX fields: ham_adi, hvib_adi (diagonals = mid-point total
        energies), time_overlap_adi (CI-level), overlap_adi (identity),
        basis_transform (identity), nac_adi (zeros), d1ham_adi (list over DOFs,
        -force at the active-state diagonal), dc1_adi (zeros - CP2K TDDFPT has
        no NACVs; use energy-based hop acceptance/rescaling).
    """
    if "tolerance" in params:
        raise ValueError('tolerance was unified into "ci_threshold" (the same '
                         "|C|^2 cut) - rename the key, the value is unchanged")
    # ---------------- trajectory index and coordinates ----------------
    Id = Cpp2Py(full_id)
    itraj = Id[-1]

    coords = q.col(itraj)
    coordinates = data_conv.MATRIX2nparray(coords, float).flatten()
    ndof = coords.num_of_rows
    nat = ndof // 3

    # ---------------- per-trajectory caches ----------------
    for key in ("mo_prev", "ci_prev", "E_adi_prev", "S_prev", "sd_prev",
                "coordinates_prev", "is_first_time", "act_state", "homo", "nelec"):
        params.setdefault(key, {})
    is_first_time = params["is_first_time"].get(itraj, True)
    act_state = int(params["act_state"].get(itraj, 0))

    nstates = int(params["nstates"])
    nexc = nstates - 1
    LO = int(params["lowest_orbital"])
    HI = int(params["highest_orbital"])
    backend = params.get("overlap_backend", "sd")
    ci_threshold = params.get("ci_threshold", 1e-6)
    force_source = params.get("force_source", "active_state")
    if force_source not in ("active_state", "all_states", "gs", "none"):
        raise ValueError(F"force_source must be 'active_state', 'all_states', 'gs' or 'none', "
                         F"got {force_source!r}")
    # the KS-SD mode has no excited-state machinery: the frozen-orbital SD
    # energies are orbital-energy differences whose state forces would need
    # orbital-energy gradients CP2K does not provide, the template carries
    # no &EXCITED_STATES block (the rewriter only rewrites, never injects)
    # and every run returns GROUND-state forces regardless of the hopping
    # state - "active_state"/"all_states" would silently mislabel a
    # classical-path run
    if not params.get("time_dependent", True) and force_source in ("active_state", "all_states"):
        raise ValueError(
            F'force_source="{force_source}" needs TDDFPT excited states, but '
            'time_dependent=False (the KS-SD mode) computes GROUND-state '
            'forces for every state. Use force_source="gs" (the same forces, '
            'declared honestly as classical-path) or "none", or '
            'time_dependent=True for real state-dependent forces')
    dia_basis = params.get("dia_basis", "none")
    if dia_basis not in ("none", "sd"):
        raise ValueError(F"dia_basis must be 'none' or 'sd', got {dia_basis!r}")
    if dia_basis == "sd" and params.get("overlap_backend", "sd") != "sd":
        raise ValueError('dia_basis="sd" declares the SD basis as the diabatic basis '
                         'via the sd2ci matrix - it requires overlap_backend="sd"')
    dia_max_drop = float(params.get("dia_basis_max_drop", 0.5))
    cap_max_drop = float(params.get("sd_capture_max_drop", 0.5))
    dia_hvib = params.get("dia_basis_hvib", "none")
    if dia_hvib not in ("none", "corrected"):
        raise ValueError(F"dia_basis_hvib must be 'none' or 'corrected', got {dia_hvib!r}")
    if dia_hvib == "corrected" and dia_basis != "sd":
        raise ValueError('dia_basis_hvib="corrected" needs dia_basis="sd"')
    st_sd = None
    if not 0.0 <= dia_max_drop <= 1.0:
        raise ValueError(F"dia_basis_max_drop must be in [0, 1], got {dia_max_drop}")
    if not 0.0 <= cap_max_drop <= 1.0:
        raise ValueError(F"sd_capture_max_drop must be in [0, 1], got {cap_max_drop}")
    # tblite runs (&XTB &TBLITE): CP2K's TDDFPT excited-state force assembly
    # calls the native-xTB Coulomb Hessian (xtb_ehess), whose neighbor lists a
    # tblite run never populates - hard segfault, verified
    # (qs_tddfpt2_forces.F:1358 -> xtb_ehess.F:146; unfixed upstream). Refuse
    # BEFORE any CP2K time is spent. (OE_CORR had the same class of defect in
    # tddfpt_oecorr - fixed by the local CP2K patch dispatching to
    # build_tblite_ks_matrix, so it is not guarded here.)
    if params.get("time_dependent", True) and force_source in ("active_state", "all_states"):
        if "&TBLITE" in _template_text(params["input_template"]).upper():
            raise ValueError(
                "CP2K's TDDFPT excited-state forces call the native-xTB Coulomb "
                "Hessian (xtb_ehess), whose neighbor lists tblite runs never "
                "populate - segfault, verified (qs_tddfpt2_forces.F:1358 -> "
                "xtb_ehess.F:146). Use force_source='gs'/'none' (spectrum + GS "
                "forces work), time_dependent=False, or native xTB")
    # the file-fed MO sources need their print in the template - the run
    # would otherwise complete and leave no MO file (a commented-out
    # &MO_MOLDEN block is the observed case)
    mo_source = params.get("mo_source", "wfn")
    if mo_source in ("molden", "namd_print"):
        need = "MO_MOLDEN" if mo_source == "molden" else "NAMD_PRINT"
        if not _template_has_section(_template_text(params["input_template"]), need):
            raise ValueError(
                    F"mo_source='{mo_source}' needs an active &{need} print in "
                    F"{params['input_template']} - the run would complete and "
                    F"leave no MO file")
    wd = f"{params.get('working_directory_prefix', 'wd')}_itraj{itraj}"

    # ---------------- run CP2K (execution layer) ----------------
    # a fresh trajectory must not inherit restart files from an older run in
    # the same directory (a stale .tdwfn from a different system hard-aborts
    # CP2K: "Incompatible number of atomic orbitals")
    if is_first_time:
        for f in glob.glob(os.path.join(wd, "*-RESTART.wfn*")) + \
                 glob.glob(os.path.join(wd, "*-RESTART.tdwfn*")):
            os.remove(f)

    run_params = {k: params[k] for k in
                  ("atom_labels", "input_template", "ot_template", "time_dependent",
                   "force_source", "cp2k_exe", "cp2k_data_dir",
                   "omp_num_threads", "mpi_executable", "nprocs", "project",
                   "retry_templates", "allow_unconverged_scf")
                  if k in params}
    # the OT pre-convergence run matters where the SCF starts cold - the
    # trajectory's first call; later calls restart the diagonalization from
    # their own previous-step wfn
    if params.get("ot_first_call_only", False) and not is_first_time:
        run_params.pop("ot_template", None)
    # force_source="gs": always run at the GS (drops any &EXCITED_STATES
    # section), so the parsed FORCES| block is guaranteed to be the S0 forces
    run_params.update({"act_state": 0 if force_source == "gs" else act_state,
                       "working_directory": wd,
                       "number_of_states": params.get("number_of_states", nexc)})
    res = compute_state_energy_and_force(coordinates, run_params)
    E_adi_curr = res["E_adi"][:nstates]
    forces = res["forces"]

    # ---------------- MO overlaps ----------------
    project = run_params.get("project", "libra")
    if mo_source == "wfn":
        # zero-cost binary-precision MOs from the restart file (GS and TD modes)
        if params.get("ao_overlap_method", "cp2k") != "cp2k":
            raise ValueError('mo_source="wfn" requires ao_overlap_method="cp2k"')
        mo_curr = load_wfn_mo_data(f"{wd}/{project}-RESTART.wfn")
    elif mo_source == "namd_print":
        # fully CP2K-native pipeline: MOs from the &NAMD_PRINT file (MOLog AO
        # order) - requires the cp2k AO backend (no libint shells available)
        if params.get("ao_overlap_method", "cp2k") != "cp2k":
            raise ValueError('mo_source="namd_print" requires '
                             'ao_overlap_method="cp2k" (no libint shells)')
        if not params.get("time_dependent", True):
            raise ValueError('mo_source="namd_print" needs a TDDFPT run '
                             '(&NAMD_PRINT is a TDDFPT print)')
        mo_curr = load_newtonx_mo_data(f"{wd}/{project}-CP2K_NEWTONX-1_0.inp")
    else:
        molden_file = f"{wd}/{project}-1_0.molden"
        mo_curr = load_molden_data(molden_file,
                                   isUKS=molden_spin_counts(molden_file)[1] > 0)

    isUKS = mo_curr.get("nspin", 1) == 2
    if itraj not in params["homo"]:
        # the reference occupation comes from the ELECTRON COUNT of the log,
        # never from a printed partition (under &SMEAR CP2K counts every orbital
        # populated beyond EPS_FERMI_DIRAC as occupied); the wfn's own aufbau
        # homo must agree with it
        ref = reference_occupation(res["logfile"], isUKS=isUKS)
        if "homo" in mo_curr:
            wfn_h = [mo_curr["homo"], mo_curr["homo_b"]] if isUKS else [mo_curr["homo"]]
            if wfn_h != (list(ref) if isUKS else [ref]):
                raise ValueError(
                    F"the wfn restart's HOMO {wfn_h} disagrees with the electron-count "
                    F"reference {ref} of {res['logfile']} - the wfn and the log describe "
                    F"different electron counts or spin treatments")
        params["homo"][itraj] = tuple(ref) if isUKS else ref
    homo = params["homo"][itraj]        # int, or (homo_alpha, homo_beta) for UKS
    homos = list(homo) if isUKS else [homo]

    mo_prev = _carried(params, "mo_prev", itraj, mo_curr, is_first_time)
    if params.get("ao_overlap_method", "cp2k") == "cp2k":
        # CP2K-native AO overlaps (OVERLAP_DELTAT doubled-geometry run)
        # a user template is shared; the auto-generated one is per working
        # directory and stays call-local - under prefetch_trajectories the
        # impl runs concurrently for several trajectories and a shared-params
        # write would race
        overlap_template = params.get("overlap_template", None)
        if overlap_template is None:
            auto = os.path.join(wd, "overlap_deltat_auto.inp")
            if not os.path.exists(auto):
                with open(auto, "w") as f:
                    f.write(make_overlap_deltat_input(params["input_template"],
                                                      params.get("overlap_csr_write", True)))
                print(F"overlap_template not provided - auto-generated from "
                      F"{params['input_template']} -> {auto}")
            overlap_template = auto
        coords_prev = _carried(params, "coordinates_prev", itraj,
                               coordinates, is_first_time)
        odt = {k: params[k] for k in ("atom_labels", "cp2k_exe", "cp2k_data_dir",
                                      "omp_num_threads", "mpi_executable", "nprocs",
                                      "csr_chunk_rows")
               if k in params}
        odt["overlap_template"] = overlap_template
        odt["working_directory"] = f"{wd}/odt"
        _prefetch_abort_check(params)
        parse_pool = _prefetch_parse_pool(params)
        if parse_pool is not None:
            odt["_parse_pool"] = parse_pool
        St_mo, S_prev, S_curr = mo_overlaps_via_cp2k(mo_prev, mo_curr, coords_prev,
                                                     coordinates, LO, HI, odt)
    else:
        if isUKS:
            raise ValueError('ao_overlap_method="libint" is restricted-only in '
                             'the driver - UKS runs need ao_overlap_method="cp2k"')
        S_curr = mo_overlap(mo_curr, mo_curr, LO, HI)
        S_prev = _carried(params, "S_prev", itraj, S_curr, is_first_time)
        St_mo = mo_overlap(mo_prev, mo_curr, LO, HI)

    # per-channel views (the overlap builders return [alpha, beta] lists for
    # UKS); the restricted path is the one-channel special case
    St_ch = list(St_mo) if isUKS else [St_mo]
    Sp_ch = list(S_prev) if isUKS else [S_prev]
    Sc_ch = list(S_curr) if isUKS else [S_curr]
    # a freshly produced MO self-overlap must be a clean identity -
    # anything else means the producing chain is broken NOW (the same
    # gate step2 applies to the same producer)
    for _k, _sc in enumerate(Sc_ch):
        orthogonalizations.validate_overlap_matrix(
            np.asarray(_sc), F"S_mo(t) channel {_k} (cp2k driver)",
            require_identity=0.05)
    C_keys = ("C", "C_b")[:len(St_ch)]

    # cumulative MO phase alignment: flip current MOs (in place, so the cache
    # carries aligned orbitals forward) and remember the phases for the CI data
    if params.get("align_mo_phases", True):
        phases_ch = []
        for k, ck in enumerate(C_keys):
            ph = align_mo_phases(mo_prev[ck], mo_curr, St_ch[k])
            mo_curr[ck][LO - 1:HI, :] *= ph[:, np.newaxis]
            Sc_ch[k] = ph[:, np.newaxis] * Sc_ch[k] * ph[np.newaxis, :]
            phases_ch.append(ph)
    else:
        # raw MO gauge: only safe when nothing downstream assumes aligned MOs
        if params.get("ci_source", "namd_print") == "namd_print":
            raise ValueError("align_mo_phases=False is incompatible with "
                             "ci_source='namd_print': the NAMD_PRINT amplitudes "
                             "are re-signed against the aligned MOs")
        phases_ch = [np.ones(HI - LO + 1) for _ in C_keys]

    # spinor-ready SD-machinery inputs ((alpha, beta) pairs for UKS)
    St_in = tuple(St_ch) if isUKS else St_ch[0]
    Sp_in = tuple(Sp_ch) if isUKS else Sp_ch[0]
    Sc_in = tuple(Sc_ch) if isUKS else Sc_ch[0]

    # SD-overlap options; apply_orthonormalization and sd_overlap_kernel use
    # the step3 vocabulary (run_step3_sd_nacs_libint) with the same meanings
    sd_orthon = params.get("apply_orthonormalization", "lowdin")
    sd_kernel = params.get("sd_overlap_kernel", "auto")
    sd_cache = None
    if str(sd_orthon) == "lowdin_cached":
        if is_first_time:
            print("WARNING: apply_orthonormalization='lowdin_cached' is not recommended in the "
                  "on-the-fly driver: it only saves one small matrix power per step, negligible "
                  "next to the CP2K call, and ties each step to the previous one's cache. "
                  "Consider 'lowdin' (the default). Results are identical - the cache is "
                  "digest-guarded and recomputes on any mismatch.")
            params.setdefault("s_ihalf_prev", {})[itraj] = {}
        sd_cache = params.setdefault("s_ihalf_prev", {}).setdefault(itraj, {})

    time_dependent = params.get("time_dependent", True)
    if "sd_basis" not in params:
        # per-spin excitation counts follow the step3 SD vocabulary: the alpha
        # window defaults to num_occ_states/num_unocc_states, beta to alpha
        noa = int(params.get("num_occ_alpha", params["num_occ_states"]))
        nua = int(params.get("num_unocc_alpha", params["num_unocc_states"]))
        if isUKS:
            params["sd_basis"] = make_sd_basis(
                homos[0] - LO + 1, noa, nua, HI - LO + 1,
                homo_beta_w=homos[1] - LO + 1,
                num_occ_beta=int(params.get("num_occ_beta", noa)),
                num_unocc_beta=int(params.get("num_unocc_beta", nua)))
        else:
            params["sd_basis"] = make_sd_basis(homos[0] - LO + 1, noa, nua,
                                               HI - LO + 1)
    sd_unique, sd_states = params["sd_basis"]

    if time_dependent:
        # ---------------- CI data, phase-consistent ----------------
        if params.get("ci_source", "namd_print") == "namd_print":
            # full-precision unthresholded X vectors from &NAMD_PRINT
            # (reuse the parse if the MOs came from the same file)
            nx_file = f"{wd}/{params.get('project', 'libra')}-CP2K_NEWTONX-1_0.inp"
            if mo_curr.get("newtonx") is None and not os.path.isfile(nx_file):
                raise ValueError('ci_source="namd_print" (the default) needs the '
                                 '&TDDFPT &PRINT &NAMD_PRINT section (PRINT_VIRTUALS '
                                 'T) in the input template; set ci_source="log" to '
                                 'parse the excitation-analysis block instead')
            nx = mo_curr.get("newtonx") or read_cp2k_newtonx_output(nx_file)
            # the SD basis is built ONCE for the electron-count reference. A printed
            # partition wider than it (orbitals above the reference populated
            # beyond EPS_FERMI_DIRAC under &SMEAR) is re-partitioned onto the
            # reference: the extra orbitals become virtuals, excitations FROM them
            # are counted as dropped weight (named, fed to the capture check),
            # excitations INTO them are absent from the smeared vectors. Where the
            # log prints MO occupations they are checked against
            # occupation_threshold - a population beyond it is a fractional
            # reference and raises. The electron count itself may never change
            theta = float(params.get("occupation_threshold", 0.01))
            if not 0.0 <= theta < 1.0:
                raise ValueError(F"occupation_threshold must be in [0, 1), got {theta}")
            ecount = read_cp2k_electron_count(res["logfile"])
            if params["nelec"].get(itraj) is None:
                params["nelec"][itraj] = list(ecount["nelec_printed"])
            elif list(ecount["nelec_printed"]) != params["nelec"][itraj]:
                raise ValueError(
                    F"the electron count changed along trajectory {itraj}: "
                    F"{params['nelec'][itraj]} at the first step, {ecount['nelec_printed']} "
                    F"now ({res['logfile']}) - a changed charge or multiplicity, nothing "
                    F"downstream is valid")
            nx, repart_drops, moved = repartition_newtonx(
                nx, list(homos) if isUKS else homos[0], theta,
                occupations=read_cp2k_mo_occupations(res["logfile"]),
                nelec=ecount["nelec_printed"])
            if moved and "repartition" not in _warned_smeared_partition:
                _warned_smeared_partition.add("repartition")
                print(F"cp2k_compute_adi: {nx_file} counts orbital(s) {moved} as occupied "
                      F"beyond the electron-count reference {homos} (&SMEAR population "
                      F"above EPS_FERMI_DIRAC) - re-partitioned onto the reference, the "
                      F"excitations from them are counted as dropped weight and reported "
                      F"through the capture check (once per process)")
            if isUKS and nx.get("nspin", 1) != 2:
                raise ValueError("the run is UKS but the NAMD_PRINT file has no "
                                 "per-spin MO blocks - set PRINT_VIRTUALS T (the "
                                 "spin channels cannot be told apart without them)")
            amps_prealigned = False
            if params.get("mo_source", "wfn") == "wfn":
                # X follows the NAMD-print MO sign convention; the wfn MOs
                # carry arbitrary per-MO signs. Re-signing against the ALIGNED
                # current C converts X to the aligned-MO convention in one shot
                # (relative raw signs AND the cumulative alignment phases), so
                # the phase loop below must be skipped for these amplitudes.
                # the wfn may hold fewer MOs than the NAMD print (nmo =
                # nocc + ADDED_MOS under diagonalization) - re-sign over the
                # shared MOs and truncate X to the wfn virtual range (the
                # dropped rows reference MOs the wfn route cannot overlap);
                # per spin channel for UKS (independent MO sets and counts)
                nx = dict(nx)
                trunc2 = 0.0
                total2 = 0.0
                for ck, xk, ok, vk in (("C", "X", "C_occ", "C_virt"),
                                       ("C_b", "X_b", "C_occ_b", "C_virt_b"))[:len(C_keys)]:
                    nocc = nx[ok].shape[1]
                    nmo = mo_curr[ck].shape[0]
                    C_namd = np.vstack([nx[ok].T, nx[vk].T])[:nmo]
                    s = np.sign(np.sum(mo_curr[ck] * C_namd, axis=1))
                    trunc2 = trunc2 + np.array([np.sum(M[nmo - nocc:] ** 2)
                                                for M in nx[xk]])
                    total2 = total2 + np.array([np.sum(M ** 2) for M in nx[xk]])
                    nx[xk] = [M[:nmo - nocc] * s[np.newaxis, :nocc] * s[nocc:, np.newaxis]
                              for M in nx[xk]]
                amps_prealigned = True
            ci_curr = ci_data_from_newtonx(nx, res["omegas"][:nexc], ci_threshold,
                                           occ_min=LO, vir_max=HI)
            if np.any(repart_drops > 0.0):
                # the weight lost to excitations from the re-partitioned orbitals
                # belongs in the capture metric like every other truncation, and
                # is reported per state like every other truncation - a large
                # loss under the cap must not pass in silence
                rd = repart_drops[:len(ci_curr["window_drops"])]
                for k in np.nonzero(rd > 1e-6)[0]:
                    print(F"cp2k_compute_adi: state {k + 1}: dropped CI weight {rd[k]:.2e} to "
                          F"excitations from the re-partitioned orbital(s) {moved} (a smeared "
                          F"population counted as occupied by CP2K)")
                ci_curr["window_drops"] = 1.0 - (1.0 - rd) * (1.0 - ci_curr["window_drops"])
            if amps_prealigned:
                # fold the wfn-virtual-range truncation into the capture
                # metric - those X rows were cut before ci_data_from_newtonx
                # could count them, and the renormalization hides the loss
                tw = (trunc2 / total2)[:len(ci_curr["window_drops"])]
                for k in np.nonzero(tw > 1e-6)[0]:
                    print(F"cp2k_compute_adi: state {k + 1}: dropped CI weight "
                          F"{tw[k]:.2e} to the wfn virtual range (fewer wfn MOs "
                          F"than NAMD-print rows)")
                ci_curr["window_drops"] = 1.0 - (1.0 - tw) * (1.0 - ci_curr["window_drops"])
        else:
            amps_prealigned = False
            if isUKS:
                raise ValueError(
                    'ci_source="log" is unreliable for UKS: the CP2K excitation-'
                    'analysis print decodes the alpha-channel configurations with '
                    'the beta-channel dimensions (verified on CP2K 2026.1-dev: a '
                    '9a->10a amplitude printed as "8 (alp) -> 15 (alp)" = the '
                    'alpha flat index re-decoded with nocc/nvirt of beta), so the '
                    'alpha labels are corrupted whenever the channels differ - '
                    'use ci_source="namd_print" (exact, unaffected)')
            ci_curr = read_ci_data(res["logfile"], nexc, ci_threshold, LO, HI,
                                   isUKS=isUKS)
        if not amps_prealigned:
            for confs, amps, csp in zip(ci_curr["configs"], ci_curr["amps"],
                                        ci_curr["spins"]):
                for k, (c, sp) in enumerate(zip(confs, csp)):
                    ph = phases_ch[0] if sp == "alp" else phases_ch[1]
                    amps[k] *= ph[c[0] - LO] * ph[c[1] - LO]
        ci_prev = _carried(params, "ci_prev", itraj, ci_curr, is_first_time)

        # ---------------- CI time-overlap: backend A or B ----------------
        if backend == "sd":
            st_sd, _ = sd_overlaps(sd_states, St_in, Sp_in, Sc_in,
                                   orthonormalize=sd_orthon, kernel=sd_kernel,
                                   cache=sd_cache)
            T_prev, _ = sd2ci_matrix(ci_prev, sd_unique, LO)
            T_curr, ci_drops = sd2ci_matrix(ci_curr, sd_unique, LO)
            # fold in the read-stage clip loss - the readers renormalize
            # after the threshold/MO-window cut, so sd2ci alone undercounts
            rd = ci_curr.get("window_drops")
            if rd is not None:
                n = min(len(rd), len(ci_drops) - 1)
                ci_drops[1:1 + n] = 1.0 - (1.0 - ci_drops[1:1 + n]) * (1.0 - rd[:n])
            # every propagated state's overlaps are built from its window
            # remnant - a mostly-dropped state yields well-formed but
            # meaningless couplings, so fail here rather than propagate them
            check_sd_capture(ci_drops, nstates, cap_max_drop,
                             knob="sd_capture_max_drop")
            st_ci = T_prev.T @ st_sd @ T_curr
            align_ci_state_signs(st_ci, ci_curr, T_curr)
            # the CI data may carry more excitations than the propagated
            # nstates (the template's NSTATES); the diabatic transform must be
            # exactly (1+nSD, nstates) - columns = the propagated states
            T_dia = T_curr[:, :nstates]
            if dia_basis == "sd":
                check_sd_capture(ci_drops, nstates, dia_max_drop)
        elif backend == "citools":
            if isUKS:
                raise ValueError('overlap_backend="citools" builds spin-adapted '
                                 'singlet CSFs - restricted-only; use '
                                 'overlap_backend="sd" for UKS')
            # the CSF overlaps consume the read-clipped renormalized vectors -
            # the read-stage loss is the whole capture metric on this backend
            rd = ci_curr.get("window_drops")
            if rd is not None:
                check_sd_capture(np.concatenate(([0.0], rd)), nstates,
                                 cap_max_drop, knob="sd_capture_max_drop")
            st_ci = ci_overlap_citools(St_mo, ci_prev, ci_curr, homo, LO, HI, nstates)
            align_ci_state_signs(st_ci, ci_curr)
        else:
            raise ValueError(f"unknown overlap_backend '{backend}'")
    else:
        # ------- ground-state-only mode: KS-SD (electron-only) states -------
        # States = the fixed single-particle SD basis (GS first); energies =
        # E_gs + orbital-energy gaps; forces = GS forces for every state
        # (classical-path picture, like the libra-template KS/SD workflows).
        if nstates != 1 + len(sd_unique):
            raise ValueError(f"time_dependent=False requires nstates = "
                             f"1 + num_occ_states*num_unocc_states = {1 + len(sd_unique)}, "
                             f"got {nstates}")
        eps_ch = [mo_curr["E"]] + ([mo_curr["E_b"]] if isUKS else [])
        E_gs = res["E_gs"]
        E_states = np.array([E_gs] + [E_gs + eps[v + LO - 2] - eps[o + LO - 2]
                                      for (o, v), spin in sd_unique
                                      for eps in [eps_ch[0 if spin == "alp" else 1]]])
        if params.get("sd_state_ordering", "energy") == "energy":
            order = np.argsort(E_states)
        else:
            order = np.arange(nstates)
        E_adi_curr = E_states[order]
        sd_sorted = [sd_states[k] for k in order]
        # GS-SD "adiabatic" states ARE the energy-sorted SDs, so the dia->adi
        # transform is the sort permutation over the fixed sd_states order
        T_dia = np.zeros((nstates, nstates))
        for j in range(nstates):
            T_dia[order[j], j] = 1.0
        sd_prev_sorted = _carried(params, "sd_prev", itraj, sd_sorted,
                                  is_first_time)
        st_ci, _ = sd_overlaps(sd_sorted, St_in, Sp_in, Sc_in,
                               orthonormalize=sd_orthon, kernel=sd_kernel,
                               cache=sd_cache, sd_states_prev=sd_prev_sorted)
        params["sd_prev"][itraj] = sd_sorted

    # ---------------- mid-point energies ----------------
    E_adi_prev = _carried(params, "E_adi_prev", itraj, E_adi_curr, is_first_time)
    E_mid = 0.5 * (E_adi_prev + E_adi_curr)

    # ---------------- build the returned object ----------------
    obj = tmp()
    obj.ham_adi = CMATRIX(nstates, nstates)
    obj.hvib_adi = CMATRIX(nstates, nstates)
    obj.nac_adi = CMATRIX(nstates, nstates)
    obj.time_overlap_adi = CMATRIX(nstates, nstates)
    obj.overlap_adi = CMATRIX(nstates, nstates)

    if dia_basis == "sd":
        # the SD basis is declared as the DIABATIC basis: U[dia, adi] =
        # <SD_i|state_j> - the sd2ci matrix (TDDFPT states) or the energy-sort
        # permutation (GS-SD mode). Row indices follow the fixed sd_states
        # order: 0 = the reference determinant, k = sd_unique[k-1]. Together
        # with the identity diabatic metric this enables rep = 0 (diabatic)
        # initialization in the engine and per-step SD populations
        # (Cdia = U Cadi); the engine's elec_params must carry
        # ndia = 1 + len(sd_unique) and nadi = nstates
        obj.basis_transform = data_conv.nparray2CMATRIX(np.asarray(T_dia, dtype=complex))
        ndia_sd = T_dia.shape[0]
        obj.ovlp_dia = CMATRIX(ndia_sd, ndia_sd)
        for i in range(ndia_sd):
            obj.ovlp_dia.set(i, i, 1.0 + 0.0j)
        # frame-corrected diabatic vibronic Hamiltonian: H_SD - i D_SD,
        # with the HST drift D_SD = (St_SD - St_SD^T)/(2 dt) from the SD
        # time-overlaps - removes the SD-frame-drift part of the
        # quasi-diabatic model error (the span part is irreducible for a
        # CI-projected Hamiltonian). Consumed by the engine under
        # hvib_update_method 0 + ham_transform_method 3; a
        # hvib_update_method 1 rebuild (ham_dia - i*nac_dia, nac_dia = 0)
        # disagrees by construction and fail-louds
        if dia_hvib == "corrected":
            if st_sd is None:
                raise ValueError('dia_basis_hvib="corrected" needs the SD '
                                 'time-overlap path (overlap_backend="sd" '
                                 'with TDDFPT states)')
            dt_c = float(params.get("dt", 41.0))
            H_sd = T_dia @ np.diag(np.asarray(E_mid[:nstates], dtype=float)) @ T_dia.T
            D_sd = (np.asarray(st_sd) - np.asarray(st_sd).T) / (2.0 * dt_c)
            obj.hvib_dia = data_conv.nparray2CMATRIX(H_sd - 1j * D_sd)
    else:
        obj.basis_transform = CMATRIX(nstates, nstates)
        for i in range(nstates):
            obj.basis_transform.set(i, i, 1.0 + 0.0j)

    for i in range(nstates):
        obj.ham_adi.set(i, i, float(E_mid[i]) * (1.0 + 0.0j))
        obj.hvib_adi.set(i, i, float(E_mid[i]) * (1.0 + 0.0j))
        obj.overlap_adi.set(i, i, 1.0 + 0.0j)
        for j in range(nstates):
            obj.time_overlap_adi.set(i, j, float(st_ci[i, j]) * (1.0 + 0.0j))
    dt = float(params.get("dt", 41.0))    # the NBRA twin's default
    # off-diagonal couplings -i (St_ij - St_ji)/(2 dt), the NBRA-twin
    # convention: a diagonal-only supplied hvib armed the overwrite guard
    # against any engine rebuild and carried no couplings to consume
    for i in range(nstates):
        for j in range(i + 1, nstates):
            dij = float(st_ci[i, j] - st_ci[j, i]) / (2.0 * dt)
            # the same couplings in nac_adi: read directly by the LZ
            # detectors, the NAC phase correction and the nac_adi saver
            obj.nac_adi.set(i, j, dij * (1.0 + 0.0j))
            obj.nac_adi.set(j, i, -dij * (1.0 + 0.0j))
            obj.hvib_adi.set(i, j, -1j * dij)
            obj.hvib_adi.set(j, i, +1j * dij)

    obj.d1ham_adi = CMATRIXList()
    obj.dc1_adi = CMATRIXList()
    for _ in range(ndof):
        obj.d1ham_adi.append(CMATRIX(nstates, nstates))
        obj.dc1_adi.append(CMATRIX(nstates, nstates))
    forces_all = res.get("forces_all", None)   # mechanism B: per-state forces
    for iat in range(nat):
        for k in range(3):
            if forces_all is not None:          # all diagonals, state-specific
                for ist in range(nstates):
                    obj.d1ham_adi[3 * iat + k].set(ist, ist,
                                                   -forces_all[ist, iat, k] * (1.0 + 0.0j))
            elif time_dependent and force_source == "active_state":   # mechanism A: active state only
                obj.d1ham_adi[3 * iat + k].set(act_state, act_state,
                                               -forces[iat, k] * (1.0 + 0.0j))
            else:  # GS mode / force_source "gs" or "none": GS forces on all diagonals
                for ist in range(nstates):
                    obj.d1ham_adi[3 * iat + k].set(ist, ist,
                                                   -forces[iat, k] * (1.0 + 0.0j))

    # ---------------- store per-trajectory state ----------------
    params["mo_prev"][itraj] = mo_curr
    if time_dependent:
        params["ci_prev"][itraj] = ci_curr
    params["E_adi_prev"][itraj] = E_adi_curr.copy()
    params["S_prev"][itraj] = Sc_ch if isUKS else Sc_ch[0]
    params["coordinates_prev"][itraj] = coordinates.copy()
    params["is_first_time"][itraj] = False

    return obj


# Live prefetch state (executors, in-flight futures) keyed by
# working_directory_prefix: model_params is deepcopied at every
# generic_recipe entry and executors are not picklable, and the rep-0
# init phase must find the SAME pools as the production loop.
# Concurrent runs in one process already need distinct prefixes (they
# would share run directories otherwise).
_PREFETCH_STATES = {}


def _allocation_cores():
    """Physical cores available to this process: the affinity mask with
    SMT siblings collapsed (sysfs topology). Logical count when the
    topology is unreadable; cpu_count where affinity is absent."""
    try:
        cpus = os.sched_getaffinity(0)
    except AttributeError:
        return os.cpu_count() or 1
    cores = set()
    for c in cpus:
        try:
            base = F"/sys/devices/system/cpu/cpu{c}/topology"
            with open(F"{base}/core_id") as f:
                cid = f.read().strip()
            with open(F"{base}/physical_package_id") as f:
                pid = f.read().strip()
            cores.add((pid, cid))
        except OSError:
            return len(cpus)
    return max(1, len(cores))


def _prefetch_key(params):
    return str(params.get("working_directory_prefix", "wd"))


def _prefetch_state(params):
    """This driver instance's live prefetch state (created empty)."""
    return _PREFETCH_STATES.setdefault(_prefetch_key(params), {})


def _prefetch_parse_pool(params):
    """The live parse pool, if the dispatcher created one (never creates)."""
    state = _PREFETCH_STATES.get(_prefetch_key(params))
    return state.get("parse_pool") if state else None


def _prefetch_abort_check(params):
    """Between chain stages: stop promptly once a sibling chain failed."""
    st = _PREFETCH_STATES.get(_prefetch_key(params))
    if st is not None and st.get("abort"):
        raise RuntimeError("prefetch batch aborted after a failed chain")


def _shutdown_prefetch_pools():
    for state in _PREFETCH_STATES.values():
        for key in ("pool", "parse_pool"):
            ex = state.pop(key, None)
            if ex is not None:
                ex.shutdown(wait=False, cancel_futures=True)
    _PREFETCH_STATES.clear()


atexit.register(_shutdown_prefetch_pools)


def _cp2k_prefetch_dispatch(q, params, full_id):
    """One batch per timestep: the first call of a step submits EVERY
    trajectory's chain (the CP2K runs + the python assembly) to a thread
    pool held in a module registry keyed by working_directory_prefix
    (model_params is deepcopied at generic_recipe entry - live executors
    must stay out of it) and returns its own result; the
    engine-sequenced calls for the remaining trajectories harvest their
    futures. subprocess waits release the GIL, so the CP2K chains overlap
    while the python parts interleave. Rests on what the engine already
    guarantees: all trajectories' coordinates are final before the step's
    first model call, act_state is synced from the previous step, and the
    per-trajectory carried state is keyed by itraj. A failed chain
    re-raises at its own trajectory's harvest."""
    Id = Cpp2Py(full_id)
    itraj = Id[-1]
    timestep = int(params.get("timestep", -1))
    state = _prefetch_state(params)
    if state.get("timestep") != timestep:
        leftovers = state.get("futures") or {}
        if leftovers:
            # the engine harvested only part of the batch - it is not
            # calling the model once per trajectory (isNBRA shares one
            # child Hamiltonian): every step would submit ntraj chains
            # and run ~ntraj-fold CP2K work
            for f in leftovers.values():
                f.cancel()
            raise RuntimeError(
                F"prefetch step {state['timestep']}: {len(leftovers)} of the "
                F"submitted chains were never harvested - the engine does "
                F"not call the model per trajectory here; set "
                F"prefetch_trajectories=0")
        ntraj = q.num_of_cols
        nprocs = max(1, int(params.get("nprocs", 1)))
        omp = max(1, int(params.get("omp_num_threads", 1)))
        sig = (ntraj, nprocs, omp,
               int(params.get("max_concurrent_chains", 0)),
               int(params.get("parse_pool_workers", -1)))
        if state.get("sig") != sig:
            # sizing inputs changed (a new run reuses the prefix):
            # recycle the executors
            for key in ("pool", "parse_pool"):
                ex = state.pop(key, None)
                if ex is not None:
                    ex.shutdown(wait=False, cancel_futures=True)
            state["sig"] = sig
        if "pool" not in state:
            # one CP2K run occupies nprocs x omp_num_threads PHYSICAL
            # cores - the default cap packs the allocation with exactly
            # that footprint (affinity-masked, SMT siblings collapsed;
            # cpu_count would report the whole node's logical CPUs).
            # Riding on top, uncounted: the parse workers (a full core
            # each, ~1-2 s bursts when chains finish together) and the
            # ~1-core GIL-serial python tail - brief time-sharing beats
            # reserving cores that would idle between bursts
            cap = int(params.get("max_concurrent_chains", 0)) or \
                max(1, _allocation_cores() // (nprocs * omp))
            state["pool"] = ThreadPoolExecutor(max_workers=min(cap, ntraj))
            # the overlap parses hold the GIL for seconds when run inside a
            # chain thread, stalling every other chain - a fork-context
            # process pool runs them truly in parallel (parse_pool_workers:
            # 0 disables). Forked and warmed up HERE, in the main thread,
            # BEFORE any chain thread exists - forking a threaded process
            # is not safe
            pworkers = int(params.get("parse_pool_workers", min(cap, ntraj)))
            if pworkers > 0 and multiprocessing.current_process().daemon:
                # a daemonic worker (mp.Pool batch parallelism) cannot fork
                # children - parses run inline in the chain threads, and the
                # outer process pool already covers cross-batch parallelism
                pworkers = 0
            if pworkers > 0:
                state["parse_pool"] = ProcessPoolExecutor(
                    max_workers=pworkers,
                    mp_context=multiprocessing.get_context("fork"))
                warm = [state["parse_pool"].submit(time.sleep, 0.05)
                        for _ in range(pworkers)]
                for w in warm:
                    w.result()
        state["timestep"] = timestep
        state["abort"] = False
        state["futures"] = {
            j: state["pool"].submit(cp2k_compute_adi_impl, q, params,
                                    Py2Cpp_int(list(Id[:-1]) + [j]))
            for j in range(ntraj)}
    fut = state["futures"].pop(itraj, None)
    if fut is None:                     # an unexpected repeat call - run directly
        return cp2k_compute_adi_impl(q, params, full_id)
    try:
        return fut.result()
    except BaseException:
        # the failed chain aborts the run - cancel the queued sibling
        # chains and flag the running ones to stop between stages
        state["abort"] = True
        for f in state["futures"].values():
            f.cancel()
        state["futures"] = {}
        raise


def cp2k_compute_adi(q, params, full_id):
    """Traceback-printing wrapper: C++ callers swallow Python exceptions."""
    try:
        if int(params.get("prefetch_trajectories", 0)):
            return _cp2k_prefetch_dispatch(q, params, full_id)
        return cp2k_compute_adi_impl(q, params, full_id)
    except Exception:
        import traceback, sys
        traceback.print_exc()
        sys.stdout.flush(); sys.stderr.flush()
        raise
