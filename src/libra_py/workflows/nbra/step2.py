# ***********************************************************
# * Copyright (C) 2017-2019 Brendan A. Smith, Wei Li, and Alexey V. Akimov
# * This file is distributed under the terms of the
# * GNU General Public License as published by the
# * Free Software Foundation; either version 3 of the
# * License, or (at your option) any later version.
# * http://www.gnu.org/copyleft/gpl.txt
# ***********************************************************/

"""
.. module:: step2
   :platform: Unix, Windows
   :synopsis: This module implements functions for doing NAC calculaitons in the 1-electron
       basis of KS orbitals, withe the QE package

.. moduleauthor:: Brendan A. Smith, Wei Li, and Alexey V. Akimov

"""

import json
import os
import glob
import sys
# The sparse matrix library for storing the MO overlaps
# with a high number of states
import numpy as np
import scipy.sparse
import time
# Fisrt, we add the location of the library to test to the PYTHON path
if sys.platform == "cygwin":
    from cyglibra_core import *
elif sys.platform == "linux" or sys.platform == "linux2":
    from liblibra_core import *
import libra_py.packages.qe.methods as QE_methods
import libra_py.packages.qe.utils as QE_utils
import libra_py.packages.cp2k.methods as CP2K_methods
from libra_py import pod as pod_lib
from libra_py import cube_file_methods
from libra_py import molden_methods
from libra_py import data_conv
from libra_py import data_read
from libra_py import orthogonalizations
from libra_py import units

# import libra_py.common_utils as comn
import util.libutil as comn

from . import compute_properties
from . import compute_hprime


def run(params):
    """This function is the main driver for the NAC calculations withing the workflows/nbra

    Args:
        params ( dictionary ): Simulation control parameters

        * **params["start_indx"]** ( int ): index of the starting datapoint in the trajectory to compute
            the NACs [default: 0]
        * **params["stop_indx"]** ( int ): index of the final datapoint in the trajectory to compute
            the NACs [default: 1]
        * **params["dt"]** ( double ): time step between two adjacent datapoint a the trajectory [units: a.u.; default: 41.0]
        * **params["wd"]** ( string ): the name of a "working directory (can be removed once the calculatons
            are done)" that will be created during this function execution - this is where the temporary files
            are written to
        * **params["rd"]** ( string ): the name of a "results directory" - this is where all the output files
            files will be written to [ default: "res" ]
        * **params["nac_method"]** ( int ): selects the type of output to analyze:

            - 0 : non-spin-polarized calculations [default]
            - 1 : spin-polarized calculations
            - 2 : non-collinear calculation (SOC) only
            - 3 : spin-polarized and non-collinear calculation (SOC)

        * **params["compute_Hprime"]** ( Boolean ): the flag to compute the <i|H'|j> matrices [ default: False]
        * **params["verbosity"]** ( int ): the verbosity level regarding the execution of the current function [default: 0]


        ..seealso:: parameters of ::funct::```run_qe``` function
            * **params["BATCH_SYSTEM"]**
            * **params["NP"]**
            * **params["EXE"]**
            * **params["EXE_EXPORT"]**
            * **params["prefix0"]**
            * **params["prefix1"]**

        ..seealso:: parameters of ::funct::```read_all``` function

            * **params["read_wfc"]**
            * **params["read_grid"]**
            * **params["verb0"]**
            * **params["verb1"]**
            * **params["verb2"]**

        ..seealso:: parameters of ::funct::```read_wfc_grid``` function
            * **params["maxband"]**
            * **params["minband"]**
            * **params["maxband_soc"]**
            * **params["minband_soc"]**



    Returns:
        None: but generates the files with couplings, energies and transition density matrices
            The matrices are of  2N x 2N dimensions, where N = maxband - minband + 1

    """

    # Now try to get parameters from the input
    critical_params = ["rd"]
    default_params = {"start_indx": 0, "stop_indx": 1, "dt": 1.0 * units.fs2au,
                      "wd": "wd",
                      "nac_method": 0,
                      "pdos_flg": 0,
                      "compute_Hprime": False,
                      "verbosity": 0
                      }
    comn.check_input(params, default_params, critical_params)

    rd = params["rd"]

    start_indx = int(params["start_indx"])
    stop_indx = int(params["stop_indx"])
    dt = params["dt"]
    wd = params["wd"]
    nac_method = params["nac_method"]
    compute_Hprime = params["compute_Hprime"]
    verbosity = params["verbosity"]
    pdos_flg = params["pdos_flg"]

    if verbosity > 0:
        print("Starting trajectory.run")
    tim = Timer()

    # Sanity/Convention check

    if nac_method == 0:
        if verbosity > 0:
            print("non-relativistic, non-spin-polarized calculations \n")
    elif nac_method == 1:
        if verbosity > 0:
            print("non-relativistic, spin-polarized calculations \n")
    elif nac_method == 2:
        if verbosity > 0:
            print("fully relativistic, non-collinear calculations \n")
    elif nac_method == 3:
        if verbosity > 0:
            print("same as 2 + 1, followed by the projections  \n")
    else:
        raise ValueError("nac_method must be one of the values in [0, 1, 2, 3]")

    # Initialize variables
    curr_index = start_indx - 1
    t = start_indx

    if verbosity > 0:
        print("In trajectory.run: current working directory for python: ", os.getcwd())
        print("In trajectory.run: current working directory for sh:", os.system("echo $(pwd)"))

    # Create the working directory where all output files will be written
    # The results directory should already exist
    os.system("mkdir %s" % wd)
    while t <= stop_indx:

        if verbosity > 0:
            print(">>>>>>>>>>>>>>>>>>>>  t= ", t, " <<<<<<<<<<<<<<<<<<<<<")

            print("stop_indx", stop_indx)
            print("start_indx", start_indx)

        dirname = ""
        if t == start_indx:
            if verbosity > 0:
                print("Starting first point in this batch")
            dirname0, dirname1 = "curr0", "curr1"

        if t > start_indx:
            if verbosity > 0:
                print("Continuing with other points in this batch")
            dirname0, dirname1 = "next0", "next1"

        # Run the QE calculations
        QE_methods.run_qe(params, t, dirname0, dirname1)

        if curr_index >= start_indx:

            # Update curr_index in params - this may be useful
            params["curr_index"] = curr_index

            # First see wether the calculation is what we wanted
            info0, all_e_dum0, info1, all_e_dum1 = QE_methods.read_info(params)

            # Read present and next electronic structure (es) in a formatted way
            es_curr, es_next = QE_methods.read_wfc_grid(params)

            # Finally,  using the current and the next wavefunctions to compute the properties of interest

            # For the non-relativistic and either spin or non-spin-polarized case

            if nac_method == 0 or nac_method == 1 or nac_method == 3:
                if (info0["nspin"] == 1 and info0["nk"] == 1) or (
                        info0["nspin"] == 2 and info0["nk"] == 2):  # Non-SOC case

                    # Only one k-point. For the spin-polarized case, the beta orbtials are as if they were computed using 2 K-points
                    # (nk = 2). However, they are not. This is just the QE format for expressing alpha and beta orbitals at a single K-point
                    # if info0["nk"]==1 or info0["nk"] == 2:

                    if verbosity > 0:
                        print("Computing various properies of the spin-diabatic (non-relativistic) KS orbitals using a single K-point")

                    # Compute various properties of the spin-diabatic (non-relativistic) KS
                    # orbitals at a single K-point
                    compute_properties.compute_properties_onekpt(params, es_curr, es_next, curr_index)

                else:

                    # Compute various properies of the spin-diabatic (non-relativistic) KS
                    # orbitals using multiple K-point
                    if verbosity > 0:
                        print("Computing various properies of the spin-diabatic (non-relativistic) KS orbitals using multiple K-points")
                        print("Warning: This capabilitiy is under development and not fully tested")

                    raise ValueError("multiple K-points are temporarily disabled - "
                                     "use a single K-point")
                    # compute_properties.compute_properties_general(params, es_curr, es_next, curr_index)

                if compute_Hprime:

                    # Compute the transition dipole moment along the nuclear trajectory
                    if verbosity > 0:
                        print("Computing the transition dipole moment along the nuclear trajectory")
                        print("Warning: This capabilitiy is under development and not fully tested")

                    # C++ implementation
                    compute_hprime.compute_hprime_dia(es_curr, info0, "%s/0_Hprime_%d" % (rd, curr_index))

                    # Python implementation
                    # compute_hprime.hprime_py(es_curr, info0, "%s/0_Hprime_%d" % (rd, curr_index) )

            # For the relativistic case
            if nac_method == 2 or nac_method == 3:

                # Only one k-point
                if info1["nk"] == 1:

                    if verbosity > 0:
                        print("Computing various properies of the spin-adiabatic (relativistic) KS orbitals using a single K-point")

                    # Compute various properties of the spin-adiabatic (relativistic) KS orbitals at a single K-point
                    compute_properties.compute_properties_onekpt(params, es_curr, es_next, curr_index)
                    # compute_properties.compute_properties_adi_gamma(params, es_curr, es_next, curr_index)

                else:
                    raise ValueError("multiple k-points with SOC are not yet implemented")

            # Checking if spin-adiabtic and spin-diabatic cases were using the same sized P\plane wave basis
            if nac_method == 3:

                print("nac_method == 3: Entering check for whether the adiabatic and diabatic basis have the same number of plane waves")

                # Check whether the adiabatic and diabatic basis have the same number of plane waves
                # the reason why I used the read_qe_wfc_info is because I will need the ngw
                # to check the consistency
                # But the read_qe_index does not read it, so in order to avoid the changes in the Libra code,
                # I use the read_qe_wfc_info.

                info_wfc0 = QE_methods.read_qe_wfc_info("%s/curr0/x0.export/wfc.1" % wd, 0)
                info_wfc1 = QE_methods.read_qe_wfc_info("%s/curr1/x1.export/wfc.1" % wd, 0)

                if info_wfc0["ngw"] != info_wfc1["ngw"]:
                    raise ValueError("the numbers of plane waves of the diabatic and "
                                     "adiabatic functions must be equal")
                else:
                    print("Pass: The number of plane waves of diabatic and adiabatic functions are equal")

                params1 = {"do_orth": 0, "root_directory": rd, "curr_index": curr_index, "print_overlaps": 1, "dt": dt}
                compute_ovlps(
                    coeff_curr0,
                    coeff_next0,
                    coeff_curr1,
                    coeff_next1,
                    e_curr0,
                    e_next0,
                    e_curr1,
                    e_next1,
                    params1)

            # -----------------------------------------------------------------

            # Remove current run, make next run to be the current one
            if nac_method == 0 or nac_method == 1 or nac_method == 3:
                os.system("rm -rf %s/curr0" % wd)
                os.system("mv %s/next0 %s/curr0" % (wd, wd))

            if nac_method == 2 or nac_method == 3:
                os.system("rm -rf %s/curr1" % wd)
                os.system("mv %s/next1 %s/curr1" % (wd, wd))

            if verbosity > 0:
                print("old files deleted, new have become old")

        # Move the index files to the results directory, just don't copy the very last one - it is a repetition
        # if t<stop_indx:
        if nac_method == 0 or nac_method == 1 or nac_method == 3:
            # The input file template defined in params["prefix0"] should have   prefix = 'x0' !
            os.system("cp %s/curr0/x0.export/index.xml %s/x0_index_%i.xml" % (wd, rd, t))
            os.system("cp %s/curr0/x0.save/data-file-schema.xml %s/x0_data-file-schema_%i.xml" % (wd, rd, t))
            os.system("cp %s/curr0/%s.%d.out %s/%s.%d.out" % (wd, params["prefix0"], t, rd, params["prefix0"], t))

            if pdos_flg == 1:
                print("Entering pdos_flag == 1: Outputting files necessary for computing pDOS calculations with projwfc.x")
                print("Warning: Crashes at this point may result from QE versioning. This flag is currently compatable with QE v6.2.1")
                print("Warning: This flag is tested only for non-relativisitic cases")

                os.system("cp %s/curr0/x0.save/charge-density.dat %s/x0_charge-density_%i.dat" % (wd, rd, t))
                os.system("cp %s/curr0/x0.save/wfcdw1.dat %s/x0_wfcdw1_%i.dat" % (wd, rd, t))
                os.system("cp %s/curr0/x0.save/wfcup1.dat %s/x0_wfcup1_%i.dat" % (wd, rd, t))

        if nac_method == 2 or nac_method == 3:
            # The input file template defined in params["prefix1"] should have   prefix = 'x1' !
            os.system("cp %s/curr1/x1.export/index.xml %s/x1_index_%i.xml" % (wd, rd, t))
            os.system("cp %s/curr1/x1.save/data-file-schema.xml %s/x1_data-file-schema_%i.xml" % (wd, rd, t))
            os.system("cp %s/curr1/%s.%d.out %s/%s.%d.out" % (wd, params["prefix1"], t, rd, params["prefix1"], t))

        # ACHTUNG!!! Restoring wfc makes some complications, so we might need to destroy wfc objects
        # after each round of operations and create new objects from the beginning - thia may be safer!

        curr_index = curr_index + 1

        if verbosity > 1:
            print("End of step t=", t)

        t = t + 1


def component_to_index(components):
    """
    This function returns the index of the emultipole integration components.
    Args:
        components (list)
    Returns:
        emultipole_index (list): The indices of the emultipole components
        emultipole_labels (list): The names of the emultipole components
    """
    emultipole_index = []
    emultipole_labels = []
    # S is the overlap
    all_components = ['S', 'x', 'y', 'z', 'x2', 'xy', 'xz', 'y2', 'yz', 'z2',
                      'x3', 'x2y', 'x2z', 'xy2', 'xyz', 'xz2', 'y3', 'y2z', 'yz2', 'z3']
    for component in components:
        for i in range(len(all_components)):
            if all_components[i] == component:
                emultipole_index.append(i)
                emultipole_labels.append(component)

    return emultipole_index, emultipole_labels


def _read_xyz_frame(filename, step):
    """Labels and coordinates (Angstrom, shape (N, 3)) of the step-th frame
    of an xyz trajectory - data_read.read_atom_coords with its lines cache."""
    return data_read.read_atom_coords(filename, step)


def _check_homo_bracket(step, isxTB, isUKS, lowest_orbital, highest_orbital):
    """A window that does not bracket the HOMO produces npz data the NAC
    steps reject (make_active_space) - fail before the production run is
    burned. Runs on every overlap route."""
    homo_logfile = F'OUT-diag_{step}.log' if isxTB else F'step_{step}.log'
    if not os.path.exists(homo_logfile):
        return
    if isUKS:
        homo_a, homo_b = CP2K_methods.read_homo_index(homo_logfile, isUKS=True)
        bracket_ok = (lowest_orbital <= homo_a < highest_orbital
                      and lowest_orbital <= homo_b < highest_orbital)
        homo_txt = F"alpha={homo_a}, beta={homo_b}"
    else:
        homo_a = CP2K_methods.read_homo_index(homo_logfile)
        bracket_ok = lowest_orbital <= homo_a < highest_orbital
        homo_txt = F"{homo_a}"
    if not bracket_ok:
        raise ValueError(F"the orbital window [{lowest_orbital}, "
                         F"{highest_orbital}] does not bracket the HOMO "
                         F"({homo_txt}) - the NAC steps downstream need "
                         F"lowest_orbital <= HOMO < highest_orbital")


def run_cp2k_libint_step2(params):
    """
    This function runs the step2 for computing the MO overlaps for both DFT and xTB calculations and saves them as sparse
    matrices using scipy.sparse library. In order to load the saved sparse matrices you need to use
    scipy.sparse.load_npz command.

    Args:

        params (dictionary): A dictionary containing the following parameters:

                             res_dir (string): The directory for saving the MO overlaps.

                             all_logfiles (string): The directory to save all log files.

                             all_pdosfiles (string): The directory to save all pdos files.

                             istep (integer): The initial step.

                             fstep (integer): The final step.

                             lowest_orbital (integer): The lowest orbital considered in the calculations.

                             highest_orbital (integer): The highest orbital considered in the calcualtions.

                             is_spherical (bool): Flag for spherical coordinates.

                             isxTB (bool): Flag for xTB calculations.

                             isUKS (bool): Flag for unrestricted spin calculations. Note for xTB:
                                           GFN1-xTB's SCC depends on the total charges only, so
                                           xTB-UKS orbitals are spin-DEGENERATE (identical alpha and
                                           beta channels) - only the occupations and per-spin HOMOs
                                           differ.

                             remove_molden (bool): Flag for removing the molden files after computing the MO overlaps.

                             nprocs (integer): Number of processors (MPI ranks for CP2K, threads for libint).

                             omp_num_threads (integer): OMP threads per CP2K rank - exported to every CP2K
                                                        subprocess - total CP2K load is nprocs x omp_num_threads
                                                        [default: 1].

                             ao_overlap_method (string): "cp2k" (default) computes the AO overlaps with a CP2K
                                                         OVERLAP_DELTAT run on the doubled geometry (one call per step
                                                         yields both S and St - periodic cells are handled natively by
                                                         the overlap template - no translational_vectors needed -
                                                         requires overlap_template) - "libint" computes them in-process
                                                         with libint from the molden shells (no extra CP2K run).

                             mo_source (string): "wfn" (default) - read the MO coefficients, eigenvalues
                                                 and HOMO from the binary RESTART.wfn file (molden-free, binary
                                                 precision - requires ao_overlap_method="cp2k") - or "molden" - the
                                                 ASCII molden print (needs the &MO_MOLDEN print, precision capped
                                                 by the printed digits).

                             overlap_template (string): CP2K input template for the OVERLAP_DELTAT runs
                                                        (used by ao_overlap_method="cp2k"): &EXCITED_STATES
                                                        OVERLAP_DELTAT T, &S_CSR_WRITE (or &AO_MATRICES OVERLAP),
                                                        CONNECTIVITY OFF and no &VDW_POTENTIAL. When empty, it is
                                                        auto-generated from cp2k_diag_input_template via
                                                        CP2K_methods.make_overlap_deltat_input (set
                                                        overlap_csr_write=False there for stock CP2K without the
                                                        S_CSR_WRITE patch).

                             fragment_atom_lists (list of lists of int): 1-based atom indices of disjoint
                                                                          fragments. When set, the Mulliken fragment
                                                                          operators of every frame are saved as
                                                                          P_frag{f}_ks_{step}.npz (same 2-block layout
                                                                          as S_ks) for the step-3 diabatization -
                                                                          needs mo_source "wfn" with
                                                                          ao_overlap_method "cp2k" [default: None].

                             compute_pod (bool): libra's own projection-operator diabatization (libra_py.pod) of every
                                                 frame with the fragment_atom_lists as the blocks: the AO Kohn-Sham matrix
                                                 comes from the diagonalization run's &DFT &PRINT &AO_MATRICES
                                                 KOHN_SHAM_MATRIX print, or is rebuilt from a COMPLETE MO set (ADDED_MOS =
                                                 n_ao - n_occ) when the print is absent, the AO overlap from the OVERLAP_DELTAT
                                                 run. Saves P_pod{f}_ks_{step}.npz (the block projectors over the MO window,
                                                 step3 diabatization_scheme "pod") and POD_hab_{step}.npz (block energies and
                                                 couplings). Needs mo_source "wfn" [default: False].

                             pod_schemes (list of str): the schemes compute_pod evaluates - "pod" (libra's POD, the global
                                                 Lowdin orthogonalization), "pod2" (fragment-first orthogonalization, the coupling
                                                 read between the non-orthogonal fragment states), "pod2l" and "pod2gs" (the
                                                 donor-acceptor pairs of fragment states orthogonalized by Lowdin or by
                                                 Gram-Schmidt first). Every scheme saves {SCHEME}_hab_{step}.npz (block energies
                                                 E_b*, couplings V_b*, and for the pod2 family the raw couplings V_raw_* and the
                                                 state overlaps S_*) - "pod", "pod2l" and "pod2gs" also save the
                                                 projectors
                                                 P_{scheme}{f}_ks_{step}.npz (step3 diabatization_scheme) - "pod2" cannot, its
                                                 states are not orthogonal across fragments [default: ["pod"]].

                             pod2gs_kept_block (int): POD2GS - the fragment whose states stay unchanged in the Gram-Schmidt
                                                 orthogonalization [default: 0].

                             read_pod_cp2k_files (bool): the diag template carries Futera's &PROPERTIES &ET_COUPLING
                                                    &PROJECTION (blocks = fragments) with the HAMILTONIAN,
                                                    WAVEFUNCTION and TRANSF_MATRIX_R prints: read the POD files of
                                                    every diagonalization run and save the POD fragment projectors
                                                    P_pod_cp2k{f}_ks_{step}.npz over the MO window (the analogue of P_frag,
                                                    step3 diabatization_scheme "pod_cp2k"), POD_cp2k_hab_{step}.npz (block
                                                    energies and couplings) and St_pod_cp2k_{step-1}.npz (the time overlaps
                                                    of all block states between the frames step-1 and step, block order,
                                                    2-spin-block layout - the input of step3.run_step3_pod). Needs
                                                    mo_source "wfn" [default: False].

                             cp2k_ot_input_template (string): The CP2K OT input template file name (just for the xTB).
                                                              None/absent skips the OT stage: the diagonalization runs then
                                                              chain their own wfn restarts (see CP2K_methods.run_cp2k_xtb) -
                                                              the diag template must carry full standalone SCF settings.
                                                              DFT runs never use it. [default: None]

                             ot_first_call_only (bool): with an OT template given, run the OT stage only
                                                        where the diagonalization chain starts cold (the previous
                                                        step's Diag restart does not exist - the first frame of a
                                                        job slice). [default: False]

                             cp2k_diag_input_template (string): The CP2K diagonalization input template file name. For
                                                                DFT calculations, this is the only parameter that we need.

                             trajectory_xyz_filename (string): The trajectory xyz file name.

                             cp2k_exe (string): The full path to CP2K executable.

                             mpi_executable (string): The mpi run executable for CP2K. Used only for
                                                      nprocs > 1 (a single rank skips the launcher) - for hybrid
                                                      MPI+OpenMP runs add binding flags here (e.g. "mpirun
                                                      --map-by slot:PE=4" or "--bind-to none") - Open MPI's
                                                      default bind-to-core pins each rank's OMP threads onto
                                                      a single core. It can be mpiexe, mpirun, or srun.

                             cube_visualization (bool): Flag for plotting the molecular orbitals using VMD.

                             vmd_input_template (string): The name of the VMD input template for visualizing the cube files.

                             states_to_plot (list): The list of states that need to be plot.

                             plot_phase_corrected (bool): Flag for plotting the molecular orbitals phase-corrected for the running job.

                             vmd_exe (string): The VMD executable.

                             tachyon_exe (string): The VMD Tachyon executable for rendering high-quality images.

                             x_pixels (integer): Number of pixels in the X direction of the image.

                             y_pixels (integer): Number of pixels in the Y direction of the image.

                             remove_cube (bool): Flag for removing the cube files after plotting the molecular orbitals.

    """

    # Now try to get parameters from the input
    critical_params = ["cp2k_diag_input_template", "trajectory_xyz_filename"]
    default_params = {
        "cp2k_ot_input_template": None,
        "res_dir": os.getcwd() +
        "/res",
        "all_logfiles": os.getcwd() +
        "/all_logfiles",
        "all_pdosfiles": os.getcwd() +
        "/all_pdosfiles",
        "all_images": os.getcwd() +
        "/all_images",
        "image_format": 'bmp',
        "istep": 0,
        "fstep": 2,
        "lowest_orbital": 1,
        "highest_orbital": 2,
        "is_spherical": True,
        "isxTB": False,
        "isUKS": False,
        "is_periodic": False,
        "remove_molden": True,
        "nprocs": 2,
        "cp2k_exe": "cp2k.psmp",
        "mpi_executable": "mpirun",
        "cube_visualization": False,
        "vmd_input_template": "vmd.tcl",
        "states_to_plot": [1],
        "plot_phase_corrected": True,
        "vmd_exe": "vmd",
        "tachyon_exe": "tachyon_LINIXAMD64",
        "x_pixels": 1024,
        "y_pixels": 1024,
        "remove_cube": True,
        'together_mode': False,
        'restart_file': True,
        'ao_overlap_method': 'cp2k',
        'mo_source': 'wfn',
        'fragment_atom_lists': None,
        'read_pod_cp2k_files': False,
        'compute_pod': False,
        'pod_schemes': ['pod'],
        'pod2gs_kept_block': 0,
        'overlap_template': '',
        'cp2k_data_dir': '',
        'omp_num_threads': 1,
        'use_emultipole': False,
        # the multipole reference point (a.u.) - moments about any other
        # origin are exact binomial combinations of these plus S
        'emultipole_origin': [0.0, 0.0, 0.0],
        'emultipole_components': [
            'S',
            'x',
            'y',
            'z']}
    # every read of the xTB flag uses 'isxTB' - a dict carrying the capital
    # form can only be a typo and would silently fall back to the DFT branch
    if 'isXTB' in params:
        raise ValueError("params['isXTB'] is not read by this workflow - use 'isxTB'")
    comn.check_input(params, default_params, critical_params)

    # Making the required directories for gatherig all the information needed
    # including the overlap data and pdos files. The logfiles do not contain
    # specific data but we finally move them to all_logfiles
    if not os.path.exists(params['res_dir']):
        os.system(F"mkdir {params['res_dir']}")
    # the stored orbital window, for step-3 consumers to verify their claim
    with open(os.path.join(params['res_dir'], 'orbital_window.json'), 'w') as f:
        json.dump({'lowest_orbital': int(params['lowest_orbital']),
                   'highest_orbital': int(params['highest_orbital'])}, f)
    if not os.path.exists(params['all_logfiles']):
        os.system(F"mkdir {params['all_logfiles']}")
    if not os.path.exists(params['all_pdosfiles']):
        os.system(F"mkdir {params['all_pdosfiles']}")
    if not os.path.exists(params['all_images']):
        os.system(F"mkdir {params['all_images']}")

    # setting up the initial step and final step
    istep = params['istep']
    fstep = params['fstep']

    # spherical or cartesian GTOs
    is_spherical = params['is_spherical']

    # number of processors
    nprocs = params['nprocs']

    # the multipole reference point for every emultipole call of this run
    _eo = params['emultipole_origin']
    emultipole_origin = VECTOR(float(_eo[0]), float(_eo[1]), float(_eo[2]))
    if params['use_emultipole'] and (_eo[0] != 0.0 or _eo[1] != 0.0 or _eo[2] != 0.0):
        print(F'emultipole reference point: {_eo}')

    # CP2K executable
    cp2k_exe = params['cp2k_exe']

    # mpirun executable
    mpi_executable = params['mpi_executable']

    # Unrestricted spin calculations
    isUKS = params['isUKS']

    # Extended tight-binding calculations
    isxTB = params['isxTB']

    # Periodic calculation falg
    is_periodic = params['is_periodic']

    # Alternative overlap routes
    ao_overlap_method = params['ao_overlap_method'].lower()
    mo_source = params['mo_source'].lower()
    if mo_source == 'wfn' and ao_overlap_method != 'cp2k':
        raise ValueError('mo_source="wfn" requires ao_overlap_method="cp2k" (no libint shells from a wfn)')
    fragment_atom_lists = params['fragment_atom_lists']
    if fragment_atom_lists is not None and (mo_source != 'wfn' or ao_overlap_method != 'cp2k'):
        raise ValueError('fragment_atom_lists needs mo_source="wfn" with ao_overlap_method="cp2k" - '
                         'the fragment operators use the wfn AO->atom map and the CP2K AO overlap')
    compute_pod = params['compute_pod']
    if compute_pod and fragment_atom_lists is None:
        raise ValueError('compute_pod needs fragment_atom_lists - the POD blocks are the fragments')
    pod_schemes = [str(s).lower() for s in params['pod_schemes']]
    if compute_pod and (not pod_schemes or any(s not in ('pod', 'pod2', 'pod2l', 'pod2gs') for s in pod_schemes)):
        raise ValueError(F'pod_schemes {pod_schemes} - use "pod", "pod2", "pod2l" and/or "pod2gs"')
    read_pod_cp2k_files = params['read_pod_cp2k_files']
    if read_pod_cp2k_files and (mo_source != 'wfn' or ao_overlap_method != 'cp2k'):
        raise ValueError('read_pod_cp2k_files needs mo_source="wfn" with ao_overlap_method="cp2k" - the POD '
                         'block MOs pair with the wfn MOs in the CP2K AO order')
    if ao_overlap_method == 'cp2k':
        if params['use_emultipole']:
            raise ValueError('use_emultipole requires the libint machinery (the multipole '
                             'integrals have no CP2K OVERLAP_DELTAT analog) - set '
                             'ao_overlap_method="libint"')
        if not params['overlap_template']:
            auto = "overlap_deltat_auto.inp"
            with open(auto, "w") as f:
                f.write(CP2K_methods.make_overlap_deltat_input(params['cp2k_diag_input_template'],
                                                               params.get('overlap_csr_write', True)))
            print(F"overlap_template not provided - auto-generated from "
                  F"{params['cp2k_diag_input_template']} -> {auto}")
            params['overlap_template'] = auto

    # Cube visualization
    cube_visualization = params['cube_visualization']

    if cube_visualization:
        # Read the tcl file data
        states_to_plot = params['states_to_plot']
        if isUKS:
            phase_factors_alpha = np.ones((len(states_to_plot), 1))
            phase_factors_beta = np.ones((len(states_to_plot), 1))
        else:
            phase_factors_alpha = np.ones((len(states_to_plot), 1))

    # the counter for a job steps, this counter is needed for not reading
    # the data of a molden file twice
    counter = 0
    print('-----------------------Start-----------------------')
    for step in range(istep, fstep):
        # a timer for all the procedure
        t1_all = time.time()
        print('-----------------------Step %d-----------------------' % step)
        params['step'] = step
        # timer for CP2K
        t1 = time.time()
        if isxTB:
            # CP2K_methods.make_ot_input(params)
            CP2K_methods.run_cp2k_xtb(params)
            molden_filename = F'Diag_{step}-libra-1_0.molden'
        else:
            CP2K_methods.CP2K_input_static(
                params['cp2k_diag_input_template'],
                'Diag_libra',
                params['trajectory_xyz_filename'],
                step)
            if params["restart_file"]:
                os.system(f'cp {params["restart_file"]} ./Diag_libra-{step}-RESTART.wfn')
            CP2K_methods.execute_cp2k(cp2k_exe, F'Diag_libra-{step}.inp',
                                      F'step_{step}.log',
                                      omp_num_threads=params["omp_num_threads"],
                                      mpi_executable=mpi_executable,
                                      nprocs=nprocs)
            molden_filename = F'Diag_libra-{step}-1_0.molden'
        print('Done with step', step, 'Elapsed time:', time.time() - t1)
#        if params['is_many_body']:
#            tddft_params = {"logfile_name": f'step_{step}.log', "number_of_states": params['number_of_tddft_states'],
#                            "isUKS": params['isUKS'], "ci_threshold": params['min_amplitude']}
#            tddft_data = CP2K_methods.read_cp2k_tddfpt_log_file(tddft_params)
#            # Only tested for isUKS=False
#            max_state = 0
#            min_state = 1e10 # A sufficiently large number for generating min state present in the TDDFT data
#            if not params['isUKS']:
#                for k_1 in range(len(tddft_data[1][0])): # Loop over all single-particle SDs
#                    max_state = int(max(max_state,tddft_data[1][0][k_1][1]))
#                    min_state = int(min(min_state,tddft_data[1][0][k_1][0]))
#            else:
#            for k_0 in range(2): # Loop over spin
#                for k_1 in range(len(tddft_data[1][k_0])): # Loop over all single-particle SDs
#                    max_state = int(max(max_state,tddft_data[1][k_0][k_1][1]))
#                    min_state = int(min(min_state,tddft_data[1][k_0][k_1][0]))
#            lowest_orbital = min_state
#            highest_orbital = max_state
#        else:
        lowest_orbital = params['lowest_orbital']
        highest_orbital = params['highest_orbital']

        if ao_overlap_method == 'cp2k':
            # ---- CP2K-native route: one OVERLAP_DELTAT run on the doubled geometry per
            # step yields both the time-overlap (prev|curr block) and the current-step
            # overlap - the overlap template carries the cell, so periodic systems need
            # no translational-vector summation ----
            t1 = time.time()
            print('Computing MO overlaps via a CP2K OVERLAP_DELTAT run...')
            labels_step, R_ang = _read_xyz_frame(params['trajectory_xyz_filename'], step)
            R_curr = R_ang.flatten() * units.Angst
            if mo_source == 'wfn':
                # molden-free: MOs, exact eigenvalues and HOMO from the binary restart file
                wfn_file = F'Diag_{step}-RESTART.wfn' if isxTB else F'Diag_libra-{step}-RESTART.wfn'
                mo_curr = CP2K_methods.load_wfn_mo_data(wfn_file)
                if read_pod_cp2k_files:
                    pod_hits = glob.glob(wfn_file[:-len('-RESTART.wfn')] + '-POD-RESTART-*.hab')
                    if len(pod_hits) != 1:
                        raise ValueError(F'read_pod_cp2k_files: {len(pod_hits)} POD files match '
                                         F'{wfn_file[:-len("-RESTART.wfn")]}-POD-RESTART-*.hab - the diag '
                                         F'template needs &PROPERTIES &ET_COUPLING &PROJECTION with the '
                                         F'HAMILTONIAN, WAVEFUNCTION and TRANSF_MATRIX_R prints')
                    pod_prefix = pod_hits[0][:-4]
                    pod_curr = CP2K_methods.read_pod_cp2k_files(pod_prefix)
                    for ext in ('.hab', '.wfn', '.tmf', '.tmr'):
                        if os.path.exists(pod_prefix + ext):
                            os.remove(pod_prefix + ext)
                if isUKS != (mo_curr['nspin'] == 2):
                    raise ValueError(F'isUKS={isUKS} but {wfn_file} carries nspin='
                                     F'{mo_curr["nspin"]} - the diag template UKS/MULTIPLICITY '
                                     F'must match the isUKS parameter')
            else:
                mo_curr = CP2K_methods.load_molden_data(molden_filename, is_spherical, isUKS)
            odt_params = {'atom_labels': labels_step,
                          'overlap_template': params['overlap_template'],
                          'working_directory': 'odt-wd',
                          'cp2k_exe': cp2k_exe,
                          'mpi_executable': mpi_executable,
                          'nprocs': nprocs,
                          'omp_num_threads': params['omp_num_threads']}
            if params['cp2k_data_dir']:
                odt_params['cp2k_data_dir'] = params['cp2k_data_dir']
            if counter == 0:
                mo_prev, R_prev = mo_curr, R_curr
                if read_pod_cp2k_files:
                    pod_prev = pod_curr
            # the same bracket guard as the libint route
            _check_homo_bracket(step, isxTB, isUKS, lowest_orbital,
                                highest_orbital)
            odt_out = CP2K_methods.mo_overlaps_via_cp2k(mo_prev, mo_curr, R_prev, R_curr,
                                                        lowest_orbital, highest_orbital,
                                                        odt_params, fragment_atom_lists,
                                                        return_ao_overlap=compute_pod,
                                                        return_ao_time_overlap=read_pod_cp2k_files)
            St, S_prev, S = odt_out[:3]
            # the optional trailing items of the doubled run, in the order the flags add them
            odt_extra = list(odt_out[4 if fragment_atom_lists is not None else 3:])
            S_ao = odt_extra.pop(0) if compute_pod else None
            S12_ao = odt_extra.pop(0) if read_pod_cp2k_files else None
            # 2-block npz convention (alpha | beta on the diagonal): restricted runs
            # duplicate the single channel, UKS carries genuinely distinct blocks
            S_a, S_b = S if isUKS else (S, S)
            E_a = np.diag(mo_curr['E'][lowest_orbital - 1:highest_orbital])
            E_b = np.diag((mo_curr['E_b'] if isUKS else mo_curr['E'])[lowest_orbital - 1:highest_orbital])
            zero_mat = np.zeros_like(S_a)
            S_step_sparse = scipy.sparse.csc_matrix(data_conv.form_block_matrix(S_a, zero_mat, zero_mat, S_b))
            # a freshly produced MO self-overlap must be a clean identity -
            # anything else means the producing chain is broken NOW
            orthogonalizations.validate_overlap_matrix(
                np.asarray(S_step_sparse.todense()), F'S_ks_{step} (production)',
                require_identity=0.05)
            scipy.sparse.save_npz(params['res_dir'] + F'/S_ks_{step}.npz', S_step_sparse)
            E_step_sparse = scipy.sparse.csc_matrix(data_conv.form_block_matrix(E_a, zero_mat, zero_mat, E_b))
            scipy.sparse.save_npz(params['res_dir'] + F'/E_ks_{step}.npz', E_step_sparse)
            if compute_pod:
                # the AO Kohn-Sham matrices: the diag run's print, else rebuilt from a complete MO set
                diag_log = F'OUT-diag_{step}.log' if isxTB else F'step_{step}.log'
                try:
                    H_ao = CP2K_methods.read_cp2k_ao_ks_print(diag_log, 2 if isUKS else 1)
                except RuntimeError:
                    H_ao = [pod_lib.ks_matrix_from_mos(mo_curr['C'], mo_curr['E'], S_ao)]
                    if isUKS:
                        H_ao.append(pod_lib.ks_matrix_from_mos(mo_curr['C_b'], mo_curr['E_b'], S_ao))
                lo, hi = lowest_orbital - 1, highest_orbital
                # the fragment eigenproblems and the pairwise blocks are one computation for the three POD2 schemes
                pod2_core = None
                if any(scheme != 'pod' for scheme in pod_schemes):
                    pod2_core = pod_lib.pod2_core(H_ao, S_ao, mo_curr['ao_atom'], fragment_atom_lists)
                for scheme in pod_schemes:
                    if scheme == 'pod':
                        pod_nat = pod_lib.pod_block_states(H_ao, S_ao, mo_curr['ao_atom'], fragment_atom_lists)
                    else:
                        pod_nat = pod_lib.pod2_block_states(H_ao, S_ao, mo_curr['ao_atom'], fragment_atom_lists,
                                                            {'pod2': 'none', 'pod2l': 'lowdin', 'pod2gs': 'gram_schmidt'}[scheme],
                                                            params['pod2gs_kept_block'], core=pod2_core)
                    if scheme != 'pod2':
                        # the fragment states of these schemes form one orthonormal set: the projectors sum to S
                        # (the POD2 states are not orthogonal across fragments - no projectors, the hab file only)
                        P_nat_a = pod_lib.pod_fragment_operators(pod_nat, mo_curr['C'][lo:hi], 0)
                        P_nat_b = pod_lib.pod_fragment_operators(pod_nat, mo_curr['C_b'][lo:hi], 1) if isUKS else P_nat_a
                        resid = float(np.max(np.abs(sum(P_nat_a) - S_a)))
                        if not np.isfinite(resid) or resid > 1e-8:
                            raise ValueError(F'step {step}: the {scheme} projectors sum to S only within {resid:.2e}')
                        for f in range(pod_nat['n_blocks']):
                            scipy.sparse.save_npz(params['res_dir'] + F'/P_{scheme}{f}_ks_{step}.npz',
                                                  scipy.sparse.csc_matrix(data_conv.form_block_matrix(P_nat_a[f], zero_mat, zero_mat, P_nat_b[f])))
                    extras = {}
                    if scheme != 'pod':
                        extras.update({F'V_raw_b{i}_b{j}_s{sp}': v for (i, j, sp), v in pod_nat['couplings_raw'].items()})
                        extras.update({F'S_b{i}_b{j}_s{sp}': v for (i, j, sp), v in pod_nat['state_overlaps'].items()})
                    np.savez(params['res_dir'] + F'/{scheme.upper()}_hab_{step}.npz',
                             **{F'E_b{ib}_s{sp}': e for (ib, sp), e in pod_nat['energies'].items()},
                             **{F'V_b{i}_b{j}_s{sp}': v for (i, j, sp), v in pod_nat['couplings'].items()}, **extras)
            if read_pod_cp2k_files:
                # the POD blocks partition the orthogonalized AO space: the projectors sum to S
                lo, hi = lowest_orbital - 1, highest_orbital
                P_pod_a = CP2K_methods.pod_cp2k_fragment_operators(pod_curr, mo_curr['C'][lo:hi], 0)
                P_pod_b = CP2K_methods.pod_cp2k_fragment_operators(pod_curr, mo_curr['C_b'][lo:hi], 1) if isUKS else P_pod_a
                resid = float(np.max(np.abs(sum(P_pod_a) - S_a)))
                if not np.isfinite(resid) or resid > 1e-8:
                    raise ValueError(F'step {step}: the POD fragment projectors sum to S only within '
                                     F'{resid:.2e} - the POD blocks do not cover all atoms or the AO orders differ')
                for f in range(pod_curr['n_blocks']):
                    scipy.sparse.save_npz(params['res_dir'] + F'/P_pod_cp2k{f}_ks_{step}.npz',
                                          scipy.sparse.csc_matrix(data_conv.form_block_matrix(P_pod_a[f], zero_mat, zero_mat, P_pod_b[f])))
                np.savez(params['res_dir'] + F'/POD_cp2k_hab_{step}.npz',
                         **{F'E_b{ib}_s{sp}': e for (ib, sp), e in pod_curr['energies'].items()},
                         **{F'V_b{i}_b{j}_s{sp}': v for (i, j, sp), v in pod_curr['couplings'].items()})
            if fragment_atom_lists is not None:
                # the fragments covering all atoms sum to S - a residual there is a
                # broken AO->atom map or a broken window - checked before anything is written
                P_sum = np.zeros_like(S_a)
                for P in odt_out[3]:
                    P_sum = P_sum + (P[0] if isUKS else P)
                if len(set().union(*[set(a) for a in fragment_atom_lists])) == len(labels_step):
                    resid = float(np.max(np.abs(P_sum - S_a)))
                    if not np.isfinite(resid) or resid > 1e-8:
                        raise ValueError(F'step {step}: the fragment operators of all atoms sum to S '
                                         F'only within {resid:.2e} - the AO->atom map does not match the AO overlap')
                for f, P in enumerate(odt_out[3]):
                    P_a, P_b = P if isUKS else (P, P)
                    block = scipy.sparse.csc_matrix(data_conv.form_block_matrix(P_a, zero_mat, zero_mat, P_b))
                    scipy.sparse.save_npz(params['res_dir'] + F'/P_frag{f}_ks_{step}.npz', block)
            if counter > 0:
                St_a, St_b = St if isUKS else (St, St)
                St_step_sparse = scipy.sparse.csc_matrix(data_conv.form_block_matrix(St_a, zero_mat, zero_mat, St_b))
                scipy.sparse.save_npz(params['res_dir'] + F'/St_ks_{step-1}.npz', St_step_sparse)
                if read_pod_cp2k_files:
                    # the block states of the previous frame against the current one - the input of the POD route
                    # of step 3 (their signs are the printed ones, step 3 follows them)
                    St_pod_a = pod_lib.pod_time_overlaps(pod_prev, pod_curr, S12_ao, 0)
                    St_pod_b = pod_lib.pod_time_overlaps(pod_prev, pod_curr, S12_ao, 1) if isUKS else St_pod_a
                    zero_pod = np.zeros_like(St_pod_a)
                    St_pod_block = data_conv.form_block_matrix(St_pod_a, zero_pod, zero_pod, St_pod_b)
                    scipy.sparse.save_npz(params['res_dir'] + F'/St_pod_cp2k_{step-1}.npz',
                                          scipy.sparse.csc_matrix(St_pod_block))
            mo_prev, R_prev = mo_curr, R_curr
            if read_pod_cp2k_files:
                pod_prev = pod_curr
            print('Done with computing molecular orbital overlaps. Elapsed time:', time.time() - t1)
            if params['remove_molden'] and os.path.exists(molden_filename):
                os.system(F'rm {molden_filename}')
            if counter > 0:
                if isxTB:
                    print('Removing unnecessary wfn files...')
                    if params['cp2k_ot_input_template'] is not None:
                        os.system(F'rm OT_{step-1}-RESTART*')
                    os.system(F'rm Diag_{step-1}-RESTART*')
                else:
                    os.system(F'rm Diag_libra-{step-1}-RESTART*')
            print(F'Done with step {step}.', 'Elapsed time:', time.time() - t1_all)
            counter += 1
            CP2K_methods.collect_pdos_files(params["all_pdosfiles"])
            continue

        # now if the counter is equal to zero
        # just compute the MO overlap of that step.
        if counter == 0:
            t1 = time.time()
            print('Creating shell...')
            shell_1, l_vals = molden_methods.molden_file_to_libint_shell(molden_filename,
                                                                         is_spherical)
            print('Done with creating shell. Elapsed time:', time.time() - t1)
            t1 = time.time()
            print('Reading energies and eigenvectors....')
            eig_vect_1, energies_1 = molden_methods.eigenvectors_molden(molden_filename,
                                                                        nbasis(shell_1), l_vals)
            print('Done with reading energies and eigenvectors. Elapsed time:', time.time() - t1)
            print('Computing atomic orbital overlap matrix...')
            t1 = time.time()
            if params['use_emultipole']:
                AO_c_all = compute_emultipole3(shell_1, shell_1, nprocs, emultipole_origin)
                emultipole_index, emultipole_labels = component_to_index(params['emultipole_components'])
            else:
                AO_S = compute_overlaps(shell_1, shell_1, nprocs)
            if is_periodic:
                cell = []
                cell.append(params['A_cell_vector'])
                cell.append(params['B_cell_vector'])
                cell.append(params['C_cell_vector'])
                cell = np.array(cell) * units.Angst
                translational_vectors = params['translational_vectors']
                for i1 in range(len(translational_vectors)):
                    translational_vector = np.array(translational_vectors[i1])
                    print(
                        F'Computing the AO overlaps between R({translational_vector[0]},{translational_vector[1]},{translational_vector[2]}) and R(0,0,0)')
                    shell_1p, l_vals = molden_methods.molden_file_to_libint_shell(
                        molden_filename, is_spherical, is_periodic, cell, translational_vector)
                    if params['use_emultipole']:
                        AO_c_all_p = compute_emultipole3(shell_1, shell_1p, nprocs, emultipole_origin)
                        for counter1 in range(20):
                            AO_c_all[counter1] += AO_c_all_p[counter1]
                    else:
                        AO_S += compute_overlaps(shell_1, shell_1p, nprocs)

            print('Done with computing atomic orbital overlaps. Elapsed time:', time.time() - t1)
            # the AO matrices stay MATRIX - the window contractions below are MATRIX products
            # scipy.sparse.save_npz(params['res_dir']+'/AO_S.npz', scipy.sparse.csc_matrix(data_conv.MATRIX2nparray(AO_S)))
            # Now, we need to resort the eigenvectors based on the new indices
            print('Resorting eigenvectors elements...')
            t1 = time.time()
            new_indices = CP2K_methods.resort_molog_eigenvectors(
                l_vals, molden_methods.molden_purity(molden_filename, l_vals))
            eigenvectors_1 = []

            for j in range(len(eig_vect_1)):
                # the new and sorted eigenvector
                eigenvector_1 = eig_vect_1[j]
                eigenvector_1 = eigenvector_1[new_indices]
                # append it to the eigenvectors list
                eigenvectors_1.append(eigenvector_1)
            eigenvectors_1 = np.array(eigenvectors_1)
            # the window slices below would silently truncate (HI > nmo) or wrap
            # (LO < 1) - validate against the MOs actually read from the molden
            if isUKS:
                # the channels are UNEQUAL in general (nelec_spin + ADDED_MOS
                # per spin) - never split the stacked molden MOs in half
                nmo_alpha, nmo_beta = CP2K_methods.molden_spin_counts(molden_filename)
                nmo_spin = min(nmo_alpha, nmo_beta)
            else:
                nmo_spin = eigenvectors_1.shape[0]
            if not 1 <= lowest_orbital <= highest_orbital <= nmo_spin:
                raise ValueError(F"orbital window [{lowest_orbital}, {highest_orbital}] does "
                                 F"not fit the {nmo_spin} MOs per spin read from the molden "
                                 F"file (1-based) - increase ADDED_MOS in the template or "
                                 F"adjust lowest_orbital/highest_orbital")
            _check_homo_bracket(step, isxTB, isUKS, lowest_orbital,
                                highest_orbital)
            if isUKS:
                # Alpha spin channel
                # alpha_eigenvectors_1 = eigenvectors_1[0::2]
                # alpha_energies_1 = energies_1[0::2]
                # Beta spin channel
                # beta_eigenvectors_1 = eigenvectors_1[1::2]
                # beta_energies_1 = energies_1[1::2]
                # Alpha spin channel
                alpha_eigenvectors_1 = eigenvectors_1[0:nmo_alpha]
                alpha_energies_1 = energies_1[0:nmo_alpha]
                # Beta spin channel
                beta_eigenvectors_1 = eigenvectors_1[nmo_alpha:nmo_alpha + nmo_beta]
                beta_energies_1 = energies_1[nmo_alpha:nmo_alpha + nmo_beta]

            print('Done with resorting eigenvectors elements. Elapsed time:', time.time() - t1)
            ##
            t1 = time.time()
            print('Computing and saving molecular orbital overlaps...')
            # Note that we choose the data from lowest_orbital to highest_orbital
            # the values for lowest_orbital and highest_orbital start from 1
            if isUKS:
                if params['use_emultipole']:
                    # per spin channel over every requested component - the 'S'
                    # component writes the standard S_ks file name, so the
                    # emultipole route stays a superset of the plain one
                    for counter1 in range(len(emultipole_labels)):
                        A_alpha = CP2K_methods.mo_window_matrix(alpha_eigenvectors_1, AO_c_all[emultipole_index[counter1]], alpha_eigenvectors_1, lowest_orbital, highest_orbital)
                        A_beta = CP2K_methods.mo_window_matrix(beta_eigenvectors_1, AO_c_all[emultipole_index[counter1]], beta_eigenvectors_1, lowest_orbital, highest_orbital)
                        zero_mat = np.zeros((len(A_alpha), len(A_beta)))
                        A_step = data_conv.form_block_matrix(A_alpha, zero_mat, zero_mat.T, A_beta)
                        scipy.sparse.save_npz(params['res_dir'] +
                                              F'/{emultipole_labels[counter1]}_ks_{step}.npz',
                                              scipy.sparse.csc_matrix(A_step))
                else:
                    S_alpha = CP2K_methods.mo_window_matrix(alpha_eigenvectors_1, AO_S, alpha_eigenvectors_1, lowest_orbital, highest_orbital)
                    S_beta = CP2K_methods.mo_window_matrix(beta_eigenvectors_1, AO_S, beta_eigenvectors_1, lowest_orbital, highest_orbital)
                    # creating zero matrix
                    mat_block_size = (len(S_alpha), len(S_beta))
                    zero_mat = np.zeros(mat_block_size)
                    S_step = data_conv.form_block_matrix(S_alpha, zero_mat, zero_mat.T, S_beta)
                    # Since a lot of the data are zeros we save them as sparse matrices
                    S_step_sparse = scipy.sparse.csc_matrix(S_step)
                    # a freshly produced MO self-overlap must be a clean identity -
                    # anything else means the producing chain is broken NOW
                    orthogonalizations.validate_overlap_matrix(
                        np.asarray(S_step_sparse.todense()), F'S_ks_{step} (production)',
                        require_identity=0.05)
                    scipy.sparse.save_npz(params['res_dir'] + F'/S_ks_{step}.npz', S_step_sparse)
                E_step = data_conv.form_block_matrix(
                    np.diag(alpha_energies_1)[
                        lowest_orbital -
                        1:highest_orbital,
                        lowest_orbital -
                        1:highest_orbital],
                    zero_mat,
                    zero_mat.T,
                    np.diag(beta_energies_1)[
                        lowest_orbital -
                        1:highest_orbital,
                        lowest_orbital -
                        1:highest_orbital])
                E_step_sparse = scipy.sparse.csc_matrix(E_step)
                scipy.sparse.save_npz(params['res_dir'] + F'/E_ks_{step}.npz', E_step_sparse)

            else:
                if params['use_emultipole']:
                    for counter1 in range(len(emultipole_labels)):
                        # I'll keep the same naming for matrices as it is more convenient
                        S = CP2K_methods.mo_window_matrix(eigenvectors_1, AO_c_all[emultipole_index[counter1]], eigenvectors_1, lowest_orbital, highest_orbital)
                        mat_block_size = len(S)
                        zero_mat = np.zeros((mat_block_size, mat_block_size))
                        S_step = data_conv.form_block_matrix(S, zero_mat, zero_mat, S)
                        # Since a lot of the data are zeros we save them as sparse matrices
                        S_step_sparse = scipy.sparse.csc_matrix(S_step)
                        scipy.sparse.save_npz(params['res_dir'] +
                                              F'/{emultipole_labels[counter1]}_ks_{step}.npz', S_step_sparse)
                else:
                    S = CP2K_methods.mo_window_matrix(eigenvectors_1, AO_S, eigenvectors_1, lowest_orbital, highest_orbital)
                    # creating zero matrix
                    mat_block_size = len(S)
                    zero_mat = np.zeros((mat_block_size, mat_block_size))
                    S_step = data_conv.form_block_matrix(S, zero_mat, zero_mat, S)
                    # Since a lot of the data are zeros we save them as sparse matrices
                    S_step_sparse = scipy.sparse.csc_matrix(S_step)
                    # a freshly produced MO self-overlap must be a clean identity -
                    # anything else means the producing chain is broken NOW
                    orthogonalizations.validate_overlap_matrix(
                        np.asarray(S_step_sparse.todense()), F'S_ks_{step} (production)',
                        require_identity=0.05)
                    scipy.sparse.save_npz(params['res_dir'] + F'/S_ks_{step}.npz', S_step_sparse)
                E_step = data_conv.form_block_matrix(
                    np.diag(energies_1)[
                        lowest_orbital -
                        1:highest_orbital,
                        lowest_orbital -
                        1:highest_orbital],
                    zero_mat,
                    zero_mat,
                    np.diag(energies_1)[
                        lowest_orbital -
                        1:highest_orbital,
                        lowest_orbital -
                        1:highest_orbital])
                E_step_sparse = scipy.sparse.csc_matrix(E_step)

                scipy.sparse.save_npz(params['res_dir'] + F'/E_ks_{step}.npz', E_step_sparse)
            print('Done with computing molecular orbital overlaps. Elapsed time:', time.time() - t1)
            print(F'Done with step {step}.', 'Elapsed time:', time.time() - t1_all)
            if params['remove_molden']:
                os.system(F'rm {molden_filename}')

        else:
            # The same procedure as above but now we compute time-overlaps as well
            t1 = time.time()
            print('Creating shell...')
            shell_2, l_vals = molden_methods.molden_file_to_libint_shell(molden_filename,
                                                                         is_spherical)
            print('Done with creating shell. Elapsed time:', time.time() - t1)
            t1 = time.time()
            print('Reading energies and eigenvectors....')
            eig_vect_2, energies_2 = molden_methods.eigenvectors_molden(molden_filename,
                                                                        nbasis(shell_2), l_vals)
            print('Done with reading energies and eigenvectors. Elapsed time:', time.time() - t1)
            print('Computing atomic orbital overlap matrix...')
            t1 = time.time()
            if params['use_emultipole']:
                AO_c_all = compute_emultipole3(shell_2, shell_2, nprocs, emultipole_origin)
                AO_t_all = compute_emultipole3(shell_1, shell_2, nprocs, emultipole_origin)
                emultipole_index, emultipole_labels = component_to_index(params['emultipole_components'])
            else:
                AO_S = compute_overlaps(shell_2, shell_2, nprocs)
                AO_St = compute_overlaps(shell_1, shell_2, nprocs)
            if is_periodic:
                cell = []
                cell.append(params['A_cell_vector'])
                cell.append(params['B_cell_vector'])
                cell.append(params['C_cell_vector'])
                cell = np.array(cell) * units.Angst
                translational_vectors = params['translational_vectors']
                for i1 in range(len(translational_vectors)):
                    translational_vector = np.array(translational_vectors[i1])
                    print(
                        F'Computing the AO overlaps between R({translational_vector[0]},{translational_vector[1]},{translational_vector[2]}) and R(0,0,0)')
                    shell_2p, l_vals = molden_methods.molden_file_to_libint_shell(
                        molden_filename, is_spherical, is_periodic, cell, translational_vector)
                    if params['use_emultipole']:
                        AO_c_all_p = compute_emultipole3(shell_2, shell_2p, nprocs, emultipole_origin)
                        AO_t_all_p = compute_emultipole3(shell_1, shell_2p, nprocs, emultipole_origin)
                        for counter1 in range(20):
                            AO_c_all[counter1] += AO_c_all_p[counter1]
                            AO_t_all[counter1] += AO_t_all_p[counter1]
                    else:
                        AO_S += compute_overlaps(shell_2, shell_2p, nprocs)
                        AO_St += compute_overlaps(shell_1, shell_2p, nprocs)

            print('Done with computing atomic orbital overlaps. Elapsed time:', time.time() - t1)
            # the AO matrices stay MATRIX - the window contractions below are MATRIX products
            # Now, we need to resort the eigenvectors based on the new indices
            print('Resorting eigenvectors elements...')
            t1 = time.time()
            new_indices = CP2K_methods.resort_molog_eigenvectors(
                l_vals, molden_methods.molden_purity(molden_filename, l_vals))
            eigenvectors_2 = []

            for j in range(len(eig_vect_2)):
                # the new and sorted eigenvector
                eigenvector_2 = eig_vect_2[j]
                eigenvector_2 = eigenvector_2[new_indices]
                # append it to the eigenvectors list
                eigenvectors_2.append(eigenvector_2)
            eigenvectors_2 = np.array(eigenvectors_2)
            if isUKS:
                # per-spin counts re-read each step: channels are unequal in
                # general and the counter-0 capture must not go stale
                nmo_alpha, nmo_beta = CP2K_methods.molden_spin_counts(molden_filename)
                nmo_spin = min(nmo_alpha, nmo_beta)
            else:
                nmo_spin = eigenvectors_2.shape[0]
            if not 1 <= lowest_orbital <= highest_orbital <= nmo_spin:
                raise ValueError(F"orbital window [{lowest_orbital}, {highest_orbital}] does "
                                 F"not fit the {nmo_spin} MOs per spin read from the molden "
                                 F"file (1-based) - increase ADDED_MOS in the template or "
                                 F"adjust lowest_orbital/highest_orbital")

            if isUKS:
                # Alpha spin channel
                # alpha_eigenvectors_2 = eigenvectors_2[0::2]
                # alpha_energies_2 = energies_2[0::2]
                # Beta spin channel
                # beta_eigenvectors_2 = eigenvectors_2[1::2]
                # beta_energies_2 = energies_2[1::2]
                # unequal channels in general - split by the true per-spin counts
                # Alpha spin channel
                alpha_eigenvectors_1 = eigenvectors_1[0:nmo_alpha]
                alpha_energies_1 = energies_1[0:nmo_alpha]
                # Beta spin channel
                beta_eigenvectors_1 = eigenvectors_1[nmo_alpha:nmo_alpha + nmo_beta]
                beta_energies_1 = energies_1[nmo_alpha:nmo_alpha + nmo_beta]

                # Alpha spin channel
                alpha_eigenvectors_2 = eigenvectors_2[0:nmo_alpha]
                alpha_energies_2 = energies_2[0:nmo_alpha]
                # Beta spin channel
                beta_eigenvectors_2 = eigenvectors_2[nmo_alpha:nmo_alpha + nmo_beta]
                beta_energies_2 = energies_2[nmo_alpha:nmo_alpha + nmo_beta]

            print('Done with resorting eigenvectors elements. Elapsed time:', time.time() - t1)
            ##
            t1 = time.time()
            print('Computing and saving molecular orbital overlaps...')

            if isUKS:
                if params['use_emultipole']:
                    # per spin channel over every requested component: same-time
                    # matrices from the CURRENT MOs, time-cross matrices with the
                    # previous step's channel as the bra - the 'S' component
                    # writes the standard S_ks/St_ks file names
                    for counter1 in range(len(emultipole_labels)):
                        A_alpha = CP2K_methods.mo_window_matrix(alpha_eigenvectors_2, AO_c_all[emultipole_index[counter1]], alpha_eigenvectors_2, lowest_orbital, highest_orbital)
                        A_beta = CP2K_methods.mo_window_matrix(beta_eigenvectors_2, AO_c_all[emultipole_index[counter1]], beta_eigenvectors_2, lowest_orbital, highest_orbital)
                        At_alpha = CP2K_methods.mo_window_matrix(alpha_eigenvectors_1, AO_t_all[emultipole_index[counter1]], alpha_eigenvectors_2, lowest_orbital, highest_orbital)
                        At_beta = CP2K_methods.mo_window_matrix(beta_eigenvectors_1, AO_t_all[emultipole_index[counter1]], beta_eigenvectors_2, lowest_orbital, highest_orbital)
                        zero_mat = np.zeros((len(A_alpha), len(A_beta)))
                        A_step = data_conv.form_block_matrix(A_alpha, zero_mat, zero_mat.T, A_beta)
                        At_step = data_conv.form_block_matrix(At_alpha, zero_mat, zero_mat.T, At_beta)
                        scipy.sparse.save_npz(params['res_dir'] +
                                              F'/{emultipole_labels[counter1]}_ks_{step}.npz',
                                              scipy.sparse.csc_matrix(A_step))
                        scipy.sparse.save_npz(params['res_dir'] +
                                              F'/{emultipole_labels[counter1]}t_ks_{step-1}.npz',
                                              scipy.sparse.csc_matrix(At_step))
                else:
                    S_alpha = CP2K_methods.mo_window_matrix(alpha_eigenvectors_2, AO_S, alpha_eigenvectors_2, lowest_orbital, highest_orbital)
                    St_alpha = CP2K_methods.mo_window_matrix(alpha_eigenvectors_1, AO_St, alpha_eigenvectors_2, lowest_orbital, highest_orbital)

                    S_beta = CP2K_methods.mo_window_matrix(beta_eigenvectors_2, AO_S, beta_eigenvectors_2, lowest_orbital, highest_orbital)
                    St_beta = CP2K_methods.mo_window_matrix(beta_eigenvectors_1, AO_St, beta_eigenvectors_2, lowest_orbital, highest_orbital)

                    # its own zero block: the emultipole branch's zero_mat is not
                    # defined on this path (NameError, or a stale shape from a
                    # previous iteration)
                    zero_mat = np.zeros((S_alpha.shape[0], S_beta.shape[1]))
                    S_step = data_conv.form_block_matrix(S_alpha, zero_mat, zero_mat.T, S_beta)
                    St_step = data_conv.form_block_matrix(St_alpha, zero_mat, zero_mat.T, St_beta)
                    S_step_sparse = scipy.sparse.csc_matrix(S_step)
                    St_step_sparse = scipy.sparse.csc_matrix(St_step)
                    # a freshly produced MO self-overlap must be a clean identity -
                    # anything else means the producing chain is broken NOW
                    orthogonalizations.validate_overlap_matrix(
                        np.asarray(S_step_sparse.todense()), F'S_ks_{step} (production)',
                        require_identity=0.05)
                    scipy.sparse.save_npz(params['res_dir'] + F'/S_ks_{step}.npz', S_step_sparse)
                    scipy.sparse.save_npz(params['res_dir'] + F'/St_ks_{step-1}.npz', St_step_sparse)

                E_step = data_conv.form_block_matrix(
                    np.diag(alpha_energies_2)[
                        lowest_orbital -
                        1:highest_orbital,
                        lowest_orbital -
                        1:highest_orbital],
                    zero_mat,
                    zero_mat.T,
                    np.diag(beta_energies_2)[
                        lowest_orbital -
                        1:highest_orbital,
                        lowest_orbital -
                        1:highest_orbital])

                E_step_sparse = scipy.sparse.csc_matrix(E_step)
            else:
                if params['use_emultipole']:
                    for counter1 in range(len(emultipole_labels)):
                        # I'll keep the same naming for matrices as it is more convenient
                        # same-time matrices belong to the CURRENT step: AO_c_all is
                        # computed over shell_2, so the contraction must use the current
                        # MOs (eigenvectors_1 still holds the previous step's here -
                        # it rotates only at the end of the loop)
                        S = CP2K_methods.mo_window_matrix(eigenvectors_2, AO_c_all[emultipole_index[counter1]], eigenvectors_2, lowest_orbital, highest_orbital)
                        mat_block_size = len(S)
                        zero_mat = np.zeros((mat_block_size, mat_block_size))
                        S_step = data_conv.form_block_matrix(S, zero_mat, zero_mat, S)
                        # Since a lot of the data are zeros we save them as sparse matrices
                        S_step_sparse = scipy.sparse.csc_matrix(S_step)
                        scipy.sparse.save_npz(params['res_dir'] +
                                              F'/{emultipole_labels[counter1]}_ks_{step}.npz', S_step_sparse)
                        St = CP2K_methods.mo_window_matrix(eigenvectors_1, AO_t_all[emultipole_index[counter1]], eigenvectors_2, lowest_orbital, highest_orbital)
                        St_step = data_conv.form_block_matrix(St, zero_mat, zero_mat, St)
                        St_step_sparse = scipy.sparse.csc_matrix(St_step)
                        scipy.sparse.save_npz(params['res_dir'] +
                                              F'/{emultipole_labels[counter1]}t_ks_{step-1}.npz', St_step_sparse)

                else:
                    S = CP2K_methods.mo_window_matrix(eigenvectors_2, AO_S, eigenvectors_2, lowest_orbital, highest_orbital)
                    St = CP2K_methods.mo_window_matrix(eigenvectors_1, AO_St, eigenvectors_2, lowest_orbital, highest_orbital)
                    S_step = data_conv.form_block_matrix(S, zero_mat, zero_mat, S)
                    St_step = data_conv.form_block_matrix(St, zero_mat, zero_mat, St)
                    S_step_sparse = scipy.sparse.csc_matrix(S_step)
                    St_step_sparse = scipy.sparse.csc_matrix(St_step)
                    # a freshly produced MO self-overlap must be a clean identity -
                    # anything else means the producing chain is broken NOW
                    orthogonalizations.validate_overlap_matrix(
                        np.asarray(S_step_sparse.todense()), F'S_ks_{step} (production)',
                        require_identity=0.05)
                    scipy.sparse.save_npz(params['res_dir'] + F'/S_ks_{step}.npz', S_step_sparse)
                    scipy.sparse.save_npz(params['res_dir'] + F'/St_ks_{step-1}.npz', St_step_sparse)
                E_step = data_conv.form_block_matrix(
                    np.diag(energies_2)[
                        lowest_orbital -
                        1:highest_orbital,
                        lowest_orbital -
                        1:highest_orbital],
                    zero_mat,
                    zero_mat,
                    np.diag(energies_2)[
                        lowest_orbital -
                        1:highest_orbital,
                        lowest_orbital -
                        1:highest_orbital])
                E_step_sparse = scipy.sparse.csc_matrix(E_step)

            scipy.sparse.save_npz(params['res_dir'] + F'/E_ks_{step}.npz', E_step_sparse)

            print('Done with computing molecular orbital overlaps. Elapsed time:', time.time() - t1)

            shell_1 = shell_2
            energies_1 = energies_2
            eigenvectors_1 = eigenvectors_2

            if params['remove_molden']:
                os.system(F'rm {molden_filename}')

            if isxTB:
                print('Removing unnecessary wfn files...')
                if params['cp2k_ot_input_template'] is not None:
                    os.system(F'rm OT_{step-1}-RESTART*')
                os.system(F'rm Diag_{step-1}-RESTART*')
            else:
                os.system(F'rm Diag_libra-{step-1}-RESTART*')

            if params['cube_visualization']:
                print('Plotting cube files using VMD...')
                num_orbitals = E_step.shape[0]
                for c_plot, state_to_plot in enumerate(states_to_plot):
                    if isUKS:
                        state_index = state_to_plot - params['lowest_orbital']
                        if not 0 <= state_index < num_orbitals // 2:
                            raise ValueError(
                                F"states_to_plot entry {state_to_plot} is outside the stored "
                                F"window [{params['lowest_orbital']}, "
                                F"{params['lowest_orbital'] + num_orbitals // 2 - 1}] - a "
                                F"negative index would silently read the wrong phase factor")
                        # Alpha orbitals
                        if isxTB:
                            cube_file_name = F'Diag_{step-1}-WFN_{str(state_to_plot).zfill(5)}_1-1_0.cube'
                        else:
                            cube_file_name = F'Diag_libra-{step-1}-WFN_{str(state_to_plot).zfill(5)}_1-1_0.cube'
                        cube_file_methods.plot_cube_v2(params, cube_file_name, phase_factors_alpha[c_plot, 0])
                        # Beta orbitals
                        if isxTB:
                            cube_file_name = F'Diag_{step-1}-WFN_{str(state_to_plot).zfill(5)}_2-1_0.cube'
                        else:
                            cube_file_name = F'Diag_libra-{step-1}-WFN_{str(state_to_plot).zfill(5)}_2-1_0.cube'
                        cube_file_methods.plot_cube_v2(params, cube_file_name, phase_factors_beta[c_plot, 0])

                        if params['plot_phase_corrected']:
                            if St_step[state_index, state_index] > 0:
                                f_alpha = 1
                            else:
                                f_alpha = -1

                            if St_step[state_index + num_orbitals, state_index + num_orbitals] > 0:
                                f_beta = 1
                            else:
                                f_beta = -1
                            phase_factors_alpha[c_plot, 0] *= f_alpha
                            phase_factors_alpha[c_plot, 0] *= f_beta

                    else:
                        # Only alpha orbitals
                        state_index = state_to_plot - params['lowest_orbital']
                        if not 0 <= state_index < num_orbitals // 2:
                            raise ValueError(
                                F"states_to_plot entry {state_to_plot} is outside the stored "
                                F"window [{params['lowest_orbital']}, "
                                F"{params['lowest_orbital'] + num_orbitals // 2 - 1}] - a "
                                F"negative index would silently read the wrong phase factor")
                        if isxTB:
                            cube_file_name = F'Diag_{step-1}-WFN_{str(state_to_plot).zfill(5)}_1-1_0.cube'
                        else:
                            cube_file_name = F'Diag_libra-{step-1}-WFN_{str(state_to_plot).zfill(5)}_1-1_0.cube'
                        cube_file_methods.plot_cube_v2(params, cube_file_name, phase_factors_alpha[c_plot, 0])

                        if params['plot_phase_corrected']:
                            if St_step[state_index, state_index] > 0:
                                f_alpha = 1
                            else:
                                f_alpha = -1
                            phase_factors_alpha[c_plot, 0] *= f_alpha
                    if params['together_mode']:
                        break
            print(F'Done with step {step}.', 'Elapsed time:', time.time() - t1_all)
        counter += 1
        # Move all the pdos files to all_pdosfiles for the step that is done!
        CP2K_methods.collect_pdos_files(params["all_pdosfiles"])
    # Move all the logfiles after the job is done. This is important in case one requires to restart the calculations
    os.system(F'mv *log {params["all_logfiles"]}/.')
    os.system(F'mv *{params["image_format"].lower()} {params["all_images"]}/.')
    os.system('rm *.wfn* *.tdwfn*')
    if params["remove_cube"]:
        os.system("rm *.cube")
    print('Done with the job!!!')
