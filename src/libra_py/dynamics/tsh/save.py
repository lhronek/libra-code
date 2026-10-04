# *********************************************************************************
# * Copyright (C) 2019-2020 Alexey V. Akimov
# *
# * This file is distributed under the terms of the GNU General Public License
# * as published by the Free Software Foundation, either version 3 of
# * the License, or (at your option) any later version.
# * See the file LICENSE in the root directory of this distribution
# * or <http://www.gnu.org/licenses/>.
# ***********************************************************************************
"""
.. module:: savers
   :platform: Unix, Windows
   :synopsis: This module implements functions for initializing and storing data
       computed/produced during TSH/Ehrenfest/Adiabatic MD calculations. If you need to print out more
       data, you shall add the corresponding info in these modules

       List of functions:

.. moduleauthor:: Alexey V. Akimov



"""

__author__ = "Alexey V. Akimov"
__copyright__ = "Copyright 2019 Alexey V. Akimov"
__credits__ = ["Alexey V. Akimov"]
__license__ = "GNU-3"
__version__ = "1.0"
__maintainer__ = "Alexey V. Akimov"
__email__ = "alexvakimov@gmail.com"
__url__ = "https://quantum-dynamics-hub.github.io/libra/index.html"


import os
import sys
import math
import copy
import ast
import re

if sys.platform == "cygwin":
    from cyglibra_core import *
elif sys.platform == "linux" or sys.platform == "linux2":
    from liblibra_core import *

import util.libutil as comn
# import libra_py.units as units
# import libra_py.data_outs as data_outs
import libra_py.data_savers as data_savers


# ===================== TSH calculations output ====================

def _has_slot(saver, name):
    """The mem/txt savers stage rows in np_data (allocated per
    output level) - the hdf5 saver writes datasets directly and
    carries no np_data - keyword membership is its only gate."""
    if name not in saver.keywords:
        return False
    np_data = getattr(saver, "np_data", None)
    return True if np_data is None else name in np_data


def init_tsh_data(saver, output_level, _nsteps, _ntraj, _ndof, _nadi, _ndia):
    """
    saver - can be either hdf5_saver or mem_saver

    """
    # print(F"In init_tsh_data with:nsteps = {_nsteps}, ntraj = {_ntraj}, ndof = {_ndof}, nadi = {_nadi}, ndia = {_ndia} ")
    # print(F"output_level = {output_level}")

    if output_level >= 1:

        # Time axis (integer steps)
        if "timestep" in saver.keywords:
            saver.add_dataset("timestep", (_nsteps,), "I")

        # Time axis
        if "time" in saver.keywords:
            saver.add_dataset("time", (_nsteps,), "R")

        # Average kinetic energy
        if "Ekin_ave" in saver.keywords:
            saver.add_dataset("Ekin_ave", (_nsteps,), "R")

        # Average potential energy
        if "Epot_ave" in saver.keywords:
            saver.add_dataset("Epot_ave", (_nsteps,), "R")

        # Average total energy
        if "Etot_ave" in saver.keywords:
            saver.add_dataset("Etot_ave", (_nsteps,), "R")

        # Fluctuation of average kinetic energy
        if "dEkin_ave" in saver.keywords:
            saver.add_dataset("dEkin_ave", (_nsteps,), "R")

        # Fluctuation of average potential energy
        if "dEpot_ave" in saver.keywords:
            saver.add_dataset("dEpot_ave", (_nsteps,), "R")

        # Fluctuation of average total energy
        if "dEtot_ave" in saver.keywords:
            saver.add_dataset("dEtot_ave", (_nsteps,), "R")

        # Thermostat energy
        if "Etherm" in saver.keywords:
            saver.add_dataset("Etherm", (_nsteps,), "R")

        # System + thermostat energy
        if "E_NHC" in saver.keywords:
            saver.add_dataset("E_NHC", (_nsteps,), "R")

        # TC-NBRA kinetic energy - averaged over all trajectories
        if "tcnbra_ekin" in saver.keywords:
            saver.add_dataset("tcnbra_ekin", (_nsteps,), "R")

        # TC-NBRA thermostat energy
        if "tcnbra_thermostat_energy" in saver.keywords:
            saver.add_dataset("tcnbra_thermostat_energy", (_nsteps,), "R")

        # Average kinetic energy in QTSH
        if "Ekin_ave_qtsh" in saver.keywords:
            saver.add_dataset("Ekin_ave_qtsh", (_nsteps,), "R")

        # KC-RPMD auxiliary electronic variable kinetic energy
        if "ekin_aux_var" in saver.keywords:
            saver.add_dataset("ekin_aux_var", (_nsteps,), "R")

    if output_level >= 2:

        # Trajectory-resolved instantaneous adiabatic states
        if "states" in saver.keywords:  # _has_slot(saver, "states"):
            saver.add_dataset("states", (_nsteps, _ntraj), "I")

        # Trajectory-resolved instantaneous diabatic states
        if "states_dia" in saver.keywords:  # _has_slot(saver, "states_dia"):
            saver.add_dataset("states_dia", (_nsteps, _ntraj), "I")

        # Average adiabatic SE populations
        if "se_pop_adi" in saver.keywords:  # _has_slot(saver, "se_pop_adi"):
            saver.add_dataset("se_pop_adi", (_nsteps, _nadi), "R")

        # Average diabatic SE populations
        if "se_pop_dia" in saver.keywords:  # _has_slot(saver, "se_pop_dia"):
            saver.add_dataset("se_pop_dia", (_nsteps, _ndia), "R")

        # Average adiabatic SH populations
        if "sh_pop_adi" in saver.keywords:  # _has_slot(saver, "sh_pop_adi"):
            saver.add_dataset("sh_pop_adi", (_nsteps, _nadi), "R")

        # Average diabatic SH populations
        if "sh_pop_dia" in saver.keywords:  # _has_slot(saver, "sh_pop_dia"):
            saver.add_dataset("sh_pop_dia", (_nsteps, _ndia), "R")

        # Average adiabatic SH populations from the Tempelaar and Reichman's method
        if "sh_pop_adi_TR" in saver.keywords:  # _has_slot(saver, "sh_pop_adi_TR"):
            saver.add_dataset("sh_pop_adi_TR", (_nsteps, _nadi), "R")

        # Average diabatic SH populations from the Tempelaar and Reichman's method
        if "sh_pop_dia_TR" in saver.keywords:  # _has_slot(saver, "sh_pop_dia_TR"):
            saver.add_dataset("sh_pop_dia_TR", (_nsteps, _nadi), "R")

        # Average adiabatic MASH populations
        if "mash_pop_adi" in saver.keywords:  # _has_slot(saver, "mash_pop_adi"):
            saver.add_dataset("mash_pop_adi", (_nsteps, _nadi), "R")

        # Average diabatic MASH populations
        if "mash_pop_dia" in saver.keywords:  # _has_slot(saver, "mash_pop_dia"):
            saver.add_dataset("mash_pop_dia", (_nsteps, _ndia), "R")

        # SQC windowed, renormalized populations (window/gamma recorded at init)
        if "sqc_pop_adi" in saver.keywords:
            saver.add_dataset("sqc_pop_adi", (_nsteps, _nadi), "R")
        if "sqc_pop_dia" in saver.keywords:
            saver.add_dataset("sqc_pop_dia", (_nsteps, _ndia), "R")

        # Average errors from FSSH3
        if "fssh3_average_errors" in saver.keywords:  # _has_slot(saver, "fssh3_average_errors"):
            saver.add_dataset("fssh3_average_errors", (_nsteps, 5), "R")

        # Trajectory-resolved auxiliary electronic variable coordinate
        if "y_aux_var" in saver.keywords:  # _has_slot(saver, "y_aux_var"):
            saver.add_dataset("y_aux_var", (_nsteps, 1), "R")

        # Trajectory-resolved auxiliary electronic variable momentum
        if "p_aux_var" in saver.keywords:  # _has_slot(saver, "p_aux_var"):
            saver.add_dataset("p_aux_var", (_nsteps, 1), "R")

        # Trajectory-resolved auxiliary electronic variable force
        if "f_aux_var" in saver.keywords:  # _has_slot(saver, "f_aux_var"):
            saver.add_dataset("f_aux_var", (_nsteps, 1), "R")

    if output_level >= 3:

        # Average adiabatic SH populations (dynamically-consistent)
        if "SH_pop" in saver.keywords:  # _has_slot(saver, "SH_pop"):
            saver.add_dataset("SH_pop", (_nsteps, _nadi, 1), "R")

        # Average adiabatic SH populations (raw)
        if "SH_pop_raw" in saver.keywords:  # _has_slot(saver, "SH_pop_raw"):
            saver.add_dataset("SH_pop_raw", (_nsteps, _nadi, 1), "R")

        # Average adiabatic density matrices (dynamically-consistent)
        if "D_adi" in saver.keywords:  # _has_slot(saver, "D_adi"):
            saver.add_dataset("D_adi", (_nsteps, _nadi, _nadi), "C")

        # Average adiabatic density matrices (raw)
        if "D_adi_raw" in saver.keywords:  # _has_slot(saver, "D_adi_raw"):
            saver.add_dataset("D_adi_raw", (_nsteps, _nadi, _nadi), "C")

        # Average diabatic density matrices (dynamically-consistent)
        if "D_dia" in saver.keywords:  # _has_slot(saver, "D_dia"):
            saver.add_dataset("D_dia", (_nsteps, _ndia, _ndia), "C")

        # Average diabatic density matrices (raw)
        if "D_dia_raw" in saver.keywords:  # _has_slot(saver, "D_dia_raw"):
            saver.add_dataset("D_dia_raw", (_nsteps, _ndia, _ndia), "C")

        # Average coherence indicator matrices
        if "coherence_adi" in saver.keywords:  # _has_slot(saver, "coherence_adi"):
            saver.add_dataset("coherence_adi", (_nsteps, _nadi, _nadi), "R")

        if "coherence_dia" in saver.keywords:  # _has_slot(saver, "coherence_dia"):
            saver.add_dataset("coherence_dia", (_nsteps, _ndia, _ndia), "R")

        # Trajectory-resolved coordinates
        if "q" in saver.keywords:  # _has_slot(saver, "q"):
            saver.add_dataset("q", (_nsteps, _ntraj, _ndof), "R")

        # Trajectory-resolved momenta
        if "p" in saver.keywords:  # _has_slot(saver, "p"):
            saver.add_dataset("p", (_nsteps, _ntraj, _ndof), "R")

        # Trajectory-resolved active forces
        if "f" in saver.keywords:  # _has_slot(saver, "f"):
            saver.add_dataset("f", (_nsteps, _ntraj, _ndof), "R")

        # Trajectory-resolved adiabatic TD-SE amplitudes
        if "Cadi" in saver.keywords:  # _has_slot(saver, "Cadi"):
            saver.add_dataset("Cadi", (_nsteps, _ntraj, _nadi), "C")

        # Trajectory-resolved diabatic TD-SE amplitudes
        if "Cdia" in saver.keywords:  # _has_slot(saver, "Cdia"):
            saver.add_dataset("Cdia", (_nsteps, _ntraj, _ndia), "C")

        # Trajectory-resolved adiabatic MMST electronic "coordinates"
        if "q_mm" in saver.keywords:  # _has_slot(saver, "q_mm"):
            saver.add_dataset("q_mm", (_nsteps, _ntraj, _nadi), "R")

        # Trajectory-resolved adiabatic MMST electronic "momenta"
        if "p_mm" in saver.keywords:  # _has_slot(saver, "p_mm"):
            saver.add_dataset("p_mm", (_nsteps, _ntraj, _nadi), "R")

        # Trajectory-resolved width parameters for quantum momenta
        if "wp_width" in saver.keywords:  # _has_slot(saver, "wp_width"):
            saver.add_dataset("wp_width", (_nsteps, _ntraj, _ndof), "R")

        # Trajectory-resolved quantum momenta
        if "p_quant" in saver.keywords:  # _has_slot(saver, "p_quant"):
            saver.add_dataset("p_quant", (_nsteps, _ntraj, _ndof), "R")

        # Trajectory-resolved exact vector potential
        if "VP" in saver.keywords:  # _has_slot(saver, "VP"):
            saver.add_dataset("VP", (_nsteps, _ntraj, _ndof), "R")

        # Trajectory-resolved decoherence forces based on XF
        if "f_xf" in saver.keywords:  # _has_slot(saver, "f_xf"):
            saver.add_dataset("f_xf", (_nsteps, _ntraj, _ndof), "R")

        # Trajectory-resolved nonclassical forces in QTSH
        if "qtsh_f_nc" in saver.keywords:  # _has_slot(saver, "qtsh_f_nc"):
            saver.add_dataset("qtsh_f_nc", (_nsteps, _ntraj, _ndof), "R")

        # Trajectory-averaged decoherence rates
        if "ave_decoherence_rates" in saver.keywords:  # decoherence time:
            saver.add_dataset("ave_decoherence_rates", (_nsteps, _nadi, _nadi), "R")

        
    if output_level >= 4:

        # Trajectory-resolved vibronic Hamiltonians in the adiabatic representation
        if "hvib_adi" in saver.keywords:  # _has_slot(saver, "hvib_adi"):
            saver.add_dataset("hvib_adi", (_nsteps, _ntraj, _nadi, _nadi), "C")

        # Trajectory-resolved vibronic Hamiltonians in the diabatic representation
        if "hvib_dia" in saver.keywords:  # _has_slot(saver, "hvib_dia"):
            saver.add_dataset("hvib_dia", (_nsteps, _ntraj, _ndia, _ndia), "C")

        # Trajectory-resolved time-overlaps of the adiabatic states
        if "St" in saver.keywords:  # _has_slot(saver, "St"):
            saver.add_dataset("St", (_nsteps, _ntraj, _nadi, _nadi), "C")

        # Trajectory-resolved diabatic-to-adiabatic transformation matrices
        if "basis_transform" in saver.keywords:  # _has_slot(saver, "basis_transform"):
            saver.add_dataset("basis_transform", (_nsteps, _ntraj, _ndia, _nadi), "C")

        # Trajectory-resolved projector matrices (from the raw adiabatic to consistent adiabatic)
        if "projector" in saver.keywords:  # _has_slot(saver, "projector"):
            saver.add_dataset("projector", (_nsteps, _ntraj, _nadi, _nadi), "C")

        # Trajectory-resolved auxiliary coordinates for XF methods
        if "q_aux" in saver.keywords:  # _has_slot(saver, "q_aux"):
            saver.add_dataset("q_aux", (_nsteps, _ntraj, _nadi, _ndof), "R")

        # Trajectory-resolved auxiliary momenta for XF methods
        if "p_aux" in saver.keywords:  # _has_slot(saver, "p_aux"):
            saver.add_dataset("p_aux", (_nsteps, _ntraj, _nadi, _ndof), "R")

        # Trajectory-resolved nabla_phase for XF methods
        if "nab_phase" in saver.keywords:  # _has_slot(saver, "nab_phase"):
            saver.add_dataset("nab_phase", (_nsteps, _ntraj, _nadi, _ndof), "R")


        # Trajectory-resolved energy gaps
        if "energy_gaps" in saver.keywords: # energy gaps
            saver.add_dataset("energy_gaps", (_nsteps, _ntraj, _nadi, _nadi), "R")

        # Trajectory-resolved running-average (mean) energy gaps
        if "mean_energy_gaps" in saver.keywords: # running-average (mean) energy gaps
            saver.add_dataset("mean_energy_gaps", (_nsteps, _ntraj, _nadi, _nadi), "R")

        # Trajectory-resolved energy gaps squared
        if "energy_gaps2" in saver.keywords: # energy gaps squared
            saver.add_dataset("energy_gaps2", (_nsteps, _ntraj, _nadi, _nadi), "R")

        # Trajectory-resolved running-average (mean) energy gaps squared
        if "mean_energy_gaps2" in saver.keywords: # running-average (mean) energy gaps squared
            saver.add_dataset("mean_energy_gaps2", (_nsteps, _ntraj, _nadi, _nadi), "R")

        # Trajectory-resolved energy gap fluctuations
        if "energy_gap_fluctuations" in saver.keywords: # energy gap fluctuations
            saver.add_dataset("energy_gap_fluctuations", (_nsteps, _ntraj, _nadi, _nadi), "R")

        # Trajectory-resolved gap correlation functions
        if "energy_gap_correlations" in saver.keywords: # energy gap correlation functions
            saver.add_dataset("energy_gap_correlations", (_nsteps, _ntraj, _nadi, _nadi), "R")



    if output_level >= 5:
        # Trajectory-resolved derivative coupling vectors
        if "dc1_adi" in saver.keywords:  # _has_slot(saver, "dc1_adi"):
            saver.add_dataset("dc1_adi", (_nsteps, _ntraj, _ndof, _nadi, _nadi), "R")


def read_params_record(path):
    """The top-level scalars of a _dyn_params.txt / _model_params.txt record (the str(dict) dump written by
    init_tsh_savers below): int, float, bool and string values keyed by name. The record is read as a python
    literal after three rewrites - a numpy scalar repr (np.float64(1.0), np.int64(5) under numpy 2) becomes its
    value, a range repr (range(0, 11) - a runner's state selection) its list, an object repr
    (<liblibra_core.MATRIX object at ...>) a placeholder string - so a key of a nested dict stays inside it and never
    overrides the top-level one. Nested containers and objects are skipped."""
    with open(path) as f:
        txt = f.read()
    txt = re.sub(r"np\.(?:float|int|uint|bool_?|complex)\w*\(([^()]*)\)", r"\1", txt)
    txt = re.sub(r"range\(([-\d, ]*)\)", lambda m: str(list(range(*[int(a) for a in m.group(1).split(",")]))), txt)
    txt = re.sub(r"<[^<>]*>", "'<object>'", txt)
    try:
        record = ast.literal_eval(txt)
    except (ValueError, SyntaxError) as e:
        raise ValueError(F"{path} is not a readable parameter record: {e}") from None
    if not isinstance(record, dict):
        raise ValueError(F"{path} does not hold a parameter dict")
    return {k: v for k, v in record.items() if isinstance(v, (bool, int, float, str)) and v != "<object>"}


def init_tsh_savers(params, model_params, nsteps, ntraj, nnucl, nadi, ndia):

    # ================ Create savers ==================
    if params["txt_output_level"] > 0 or params["hdf5_output_level"] > 0 or params["mem_output_level"] > 0:

        prefix = params["prefix"]

        # Create an output directory, if not present
        if not os.path.isdir(prefix):
            os.mkdir(prefix)

        # Simulation parameters
        f = open(F"{prefix}/_dyn_params.txt", "w")
        f.write(str(params))
        f.close()

        f = open(F"{prefix}/_model_params.txt", "w")
        f.write(str(model_params))
        f.close()

    if params["txt2_output_level"] > 0:

        prefix2 = params["prefix2"]

        # Create an output directory, if not present
        if not os.path.isdir(prefix2):
            os.mkdir(prefix2)

        # Simulation parameters
        f = open(F"{prefix2}/_dyn_params.txt", "w")
        f.write(str(params))
        f.close()

        f = open(F"{prefix2}/_model_params.txt", "w")
        f.write(str(model_params))
        f.close()

    properties_to_save = params["properties_to_save"]

    _savers = {"hdf5_saver": None, "txt_saver": None, "mem_saver": None, "txt2_saver": None}

    # ====== HDF5 ========
    hdf5_output_level = params["hdf5_output_level"]

    if hdf5_output_level > 0:
        _savers["hdf5_saver"] = data_savers.hdf5_saver(F"{prefix}/data.hdf", properties_to_save)
        _savers["hdf5_saver"].set_compression_level(params["use_compression"], params["compression_level"])
        init_tsh_data(_savers["hdf5_saver"], hdf5_output_level, nsteps, ntraj, nnucl, nadi, ndia)

    # ====== TXT ========
    txt_output_level = params["txt_output_level"]

    if params["txt_output_level"] > 0:
        _savers["txt_saver"] = data_savers.mem_saver(properties_to_save)
        # _savers["txt_saver"].set_compression_level(params["use_compression"], params["compression_level"])
        init_tsh_data(_savers["txt_saver"], txt_output_level, nsteps, ntraj, nnucl, nadi, ndia)

    # ====== TXT2: No intermediate memory allocation ========
    txt2_output_level = params["txt2_output_level"]

    if params["txt2_output_level"] > 0:
        _savers["txt2_saver"] = data_savers.mem_saver(properties_to_save)

        # Here, nsteps is set to 1 since in this type of saver, we only care about the current values,
        # not all the timesteps - that would be too consuming
        init_tsh_data(_savers["txt2_saver"], txt2_output_level, 1, ntraj, nnucl, nadi, ndia)

    # ====== MEM =========
    mem_output_level = params["mem_output_level"]

    if mem_output_level > 0:
        _savers["mem_saver"] = data_savers.mem_saver(properties_to_save)
        init_tsh_data(_savers["mem_saver"], mem_output_level, nsteps, ntraj, nnucl, nadi, ndia)

    return _savers


def save_hdf5_1D(saver, i, dt, Ekin, Epot, Etot, dEkin, dEpot, dEtot, Etherm, E_NHC, txt_type=0):
    """
    saver - can be either hdf5_saver or mem_saver

    txt_type ( int ): 0 - standard, all the timesteps, 1 - only the current one

    """

    t = 0
    if txt_type == 0:
        t = i

    # Timestep
    saver.save_scalar(t, "timestep", i)

    # Actual time
    saver.save_scalar(t, "time", dt * i)

    # Average kinetic energy
    saver.save_scalar(t, "Ekin_ave", Ekin)

    # Average potential energy
    saver.save_scalar(t, "Epot_ave", Epot)

    # Average total energy
    saver.save_scalar(t, "Etot_ave", Etot)

    # Fluctuation of average kinetic energy
    saver.save_scalar(t, "dEkin_ave", dEkin)

    # Fluctuation of average potential energy
    saver.save_scalar(t, "dEpot_ave", dEpot)

    # Fluctuation average total energy
    saver.save_scalar(t, "dEtot_ave", dEtot)

    # Thermostat energy
    saver.save_scalar(t, "Etherm", Etherm)

    # System + thermostat energy
    saver.save_scalar(t, "E_NHC", E_NHC)


def save_hdf5_1D_new(saver, i, params, dyn_var, ham, txt_type=0):
    """
    saver - can be either hdf5_saver or mem_saver

    txt_type ( int ): 0 - standard, all the timesteps, 1 - only the current one

    """

    dt = params["dt"]
    nprint = params["nprint"]

    Ekin, Epot, Etot = 0.0, 0.0, 0.0
    dEkin, dEpot, dEtot, Etherm = 0.0, 0.0, 0.0, 0.0
    E_NHC = 0.0

    t = 0
    if txt_type == 0:
        t = i

    # Timestep
    #saver.save_scalar(t, "timestep", i)
    saver.save_scalar(t, "timestep", nprint * i)

    # Actual time
    #saver.save_scalar(t, "time", dt * i)
    saver.save_scalar(t, "time", dt * nprint * i)

    # Average kinetic energy
    Ekin = dyn_var.compute_average_kinetic_energy()
    saver.save_scalar(t, "Ekin_ave", Ekin)

    # Average potential energy
    Epot = average_potential_energy(params, dyn_var, ham)
    saver.save_scalar(t, "Epot_ave", Epot)

    # Average total energy
    Etot = Ekin + Epot
    saver.save_scalar(t, "Etot_ave", Etot)

    # Fluctuation of average kinetic energy
    saver.save_scalar(t, "dEkin_ave", dEkin)

    # Fluctuation of average potential energy
    saver.save_scalar(t, "dEpot_ave", dEpot)

    # Fluctuation average total energy
    saver.save_scalar(t, "dEtot_ave", dEtot)

    # Thermostat energy
    saver.save_scalar(t, "Etherm", Etherm)

    # System + thermostat energy
    saver.save_scalar(t, "E_NHC", E_NHC)

    # TC-NBRA average kinetic energy
    if _has_slot(saver, "tcnbra_ekin"):
        tcnbra_ekin = dyn_var.compute_tcnbra_ekin()
        saver.save_scalar(t, "tcnbra_ekin", tcnbra_ekin)

    # TC-NBRA average energy of thermostat
    if _has_slot(saver, "tcnbra_thermostat_energy"):
        tcnbra_thermostat_energy = dyn_var.compute_tcnbra_thermostat_energy()
        saver.save_scalar(t, "tcnbra_thermostat_energy", tcnbra_thermostat_energy)

    # KC-RPMD auxiliary electronic variable kinetic energy
    if _has_slot(saver, "ekin_aux_var"):
        ekin_aux_var = dyn_var.compute_kcrpmd_ekin()
        saver.save_scalar(t, "ekin_aux_var", ekin_aux_var)


def save_hdf5_2D(saver, i, states, txt_type=0):
    """
    saver - can be either hdf5_saver or mem_saver

    txt_type ( int ): 0 - standard, all the timesteps, 1 - only the current one

    """

    t = 0
    if txt_type == 0:
        t = i

    # Trajectory-resolved instantaneous adiabatic states
    # Format: saver.add_dataset("states", (_nsteps, _ntraj), "I")
    saver.save_vector(t, "states", states)


def save_hdf5_2D_new(saver, i, dyn_var, ham, txt_type=0):
    """
    saver - can be either hdf5_saver or mem_saver

    txt_type ( int ): 0 - standard, all the timesteps, 1 - only the current one

    """

    t = 0
    if txt_type == 0:
        t = i

    if _has_slot(saver, "states"):
        # Trajectory-resolved instantaneous adiabatic states
        # Format: saver.add_dataset("states", (_nsteps, _ntraj), "I")
        saver.save_vector(t, "states", dyn_var.act_states)

    if _has_slot(saver, "states_dia"):
        # Trajectory-resolved instantaneous diabatic states
        # Format: saver.add_dataset("states", (_nsteps, _ntraj), "I")
        saver.save_vector(t, "states_dia", dyn_var.act_states_dia)

    if _has_slot(saver, "se_pop_dia"):
        # Average diabatic SE populations
        # Format: saver.add_dataset("se_pop_dia", (_nsteps, _ndia), "R")
        pops_se0 = dyn_var.compute_average_se_pop(0)
        saver.save_vector(t, "se_pop_dia", pops_se0)

    if _has_slot(saver, "se_pop_adi"):
        # Average adiabatic SE populations
        # Format: saver.add_dataset("se_pop_adi", (_nsteps, _nadi), "R")
        pops_se1 = dyn_var.compute_average_se_pop(1)
        saver.save_vector(t, "se_pop_adi", pops_se1)

    if _has_slot(saver, "sh_pop_dia"):
        # Average diabatic SH populations
        # Format: saver.add_dataset("sh_pop_dia", (_nsteps, _ndia), "R")
        pops_sh0 = dyn_var.compute_average_sh_pop(0)
        saver.save_vector(t, "sh_pop_dia", pops_sh0)

    if _has_slot(saver, "sh_pop_adi"):
        # Average adiabatic SH populations
        # Format: saver.add_dataset("sh_pop_adi", (_nsteps, _nadi), "R")
        pops_sh1 = dyn_var.compute_average_sh_pop(1)
        saver.save_vector(t, "sh_pop_adi", pops_sh1)

    if _has_slot(saver, "sh_pop_dia_TR"):
        # Average diabatic SH populations
        # Format: saver.add_dataset("sh_pop_dia", (_nsteps, _ndia), "R")
        pops_sh0 = dyn_var.compute_average_sh_pop_TR(0)
        saver.save_vector(t, "sh_pop_dia_TR", pops_sh0)

    if _has_slot(saver, "sh_pop_adi_TR"):
        # Average adiabatic SH populations
        # Format: saver.add_dataset("sh_pop_adi", (_nsteps, _nadi), "R")
        pops_sh1 = dyn_var.compute_average_sh_pop_TR(1)
        saver.save_vector(t, "sh_pop_adi_TR", pops_sh1)

    if _has_slot(saver, "mash_pop_dia"):
        # Average diabatic MASH populations
        # Format: saver.add_dataset("mash_pop_dia", (_nsteps, _ndia), "R")
        pops_sh0 = dyn_var.compute_average_mash_pop(0)
        saver.save_vector(t, "mash_pop_dia", pops_sh0)

    if _has_slot(saver, "mash_pop_adi"):
        # Average adiabatic MASH populations
        # Format: saver.add_dataset("mash_pop_adi", (_nsteps, _nadi), "R")
        pops_sh1 = dyn_var.compute_average_mash_pop(1)
        saver.save_vector(t, "mash_pop_adi", pops_sh1)

    if _has_slot(saver, "sqc_pop_dia"):
        # SQC populations, diabatic windows
        pops_sh0 = dyn_var.compute_average_sqc_pop(0)
        saver.save_vector(t, "sqc_pop_dia", pops_sh0)

    if _has_slot(saver, "sqc_pop_adi"):
        # SQC populations, adiabatic windows
        pops_sh1 = dyn_var.compute_average_sqc_pop(1)
        saver.save_vector(t, "sqc_pop_adi", pops_sh1)

    if _has_slot(saver, "fssh3_average_errors"):
        # Average errors in FSSH3
        # Format: saver.add_dataset("fssh3_average_errors", (_nsteps, 5), "R")
        fssh3_average_errors = dyn_var.get_fssh3_average_errors()
        for k in range(5):
            saver.save_multi_scalar(t, k, "fssh3_average_errors", fssh3_average_errors[k])

    if _has_slot(saver, "y_aux_var"):
        # Trajectory-resolved auxiliary electronic variable coordinate
        # Format: saver.add_dataset("y_aux_var", (_nsteps, 1), "R")
        y_aux_var = dyn_var.get_y_aux_var()
        saver.save_multi_scalar(t, 0, "y_aux_var", y_aux_var[0])

    if _has_slot(saver, "p_aux_var"):
        # Trajectory-resolved auxiliary electronic variable coordinate
        # Format: saver.add_dataset("p_aux_var", (_nsteps, 1), "R")
        p_aux_var = dyn_var.get_p_aux_var()
        saver.save_multi_scalar(t, 0, "p_aux_var", p_aux_var[0])

    if _has_slot(saver, "f_aux_var"):
        # Trajectory-resolved auxiliary electronic variable coordinate
        # Format: saver.add_dataset("f_aux_var", (_nsteps, 1), "R")
        f_aux_var = dyn_var.get_f_aux_var()
        saver.save_multi_scalar(t, 0, "f_aux_var", f_aux_var[0])


def save_hdf5_3D(saver, i, pops, pops_raw, dm_adi, dm_adi_raw, dm_dia, dm_dia_raw, q, p, Cadi, Cdia, txt_type=0):
    """
    saver - can be either hdf5_saver or mem_saver

    txt_type ( int ): 0 - standard, all the timesteps, 1 - only the current one

    """

    t = 0
    if txt_type == 0:
        t = i

    # Average adiabatic SH populations (dynamically-consistent)
    # Format: saver.add_dataset("SH_pop", (_nsteps, _nadi, 1), "R")
    if _has_slot(saver, "SH_pop"):
        saver.save_matrix(t, "SH_pop", pops)

    # Average adiabatic SH populations (raw)
    # Format: saver.add_dataset("SH_pop_raw", (_nsteps, _nadi, 1), "R")
    if _has_slot(saver, "SH_pop_raw"):
        saver.save_matrix(t, "SH_pop_raw", pops_raw)

    # Average adiabatic density matrices (dynamically-consistent)
    # Format: saver.add_dataset("D_adi", (_nsteps, _nadi, _nadi), "C")
    if _has_slot(saver, "D_adi"):
        saver.save_matrix(t, "D_adi", dm_adi)

    # Average adiabatic density matrices (raw)
    # Format: saver.add_dataset("D_adi_raw", (_nsteps, _nadi, _nadi), "C")
    if _has_slot(saver, "D_adi_raw"):
        saver.save_matrix(t, "D_adi_raw", dm_adi_raw)

    # Average diabatic density matrices (dynamically-consistent)
    # Format: saver.add_dataset("D_dia", (_nsteps, _ndia, _ndia), "C")
    if _has_slot(saver, "D_dia"):
        saver.save_matrix(t, "D_dia", dm_dia)

    # Average diabatic density matrices (raw)
    # Format: saver.add_dataset("D_dia_raw", (_nsteps, _ndia, _ndia), "C")
    if _has_slot(saver, "D_dia_raw"):
        saver.save_matrix(t, "D_dia_raw", dm_dia_raw)

    # Trajectory-resolved coordinates
    # Format: saver.add_dataset("q", (_nsteps, _ntraj, _dof), "R")
    if _has_slot(saver, "q"):
        saver.save_matrix(t, "q", q.T())

    # Trajectory-resolved momenta
    # Format: saver.add_dataset("p", (_nsteps, _ntraj, _dof), "R")
    if _has_slot(saver, "p"):
        saver.save_matrix(t, "p", p.T())

    # Trajectory-resolved active forces
    # Format: saver.add_dataset("f", (_nsteps, _ntraj, _dof), "R")
#    if _has_slot(saver, "f"):
#        saver.save_matrix(t, "f", f.T())

    # Trajectory-resolved adiabatic TD-SE amplitudes
    # Format: saver.add_dataset("C_adi", (_nsteps, _ntraj, _nadi), "C")
    if _has_slot(saver, "Cadi"):
        saver.save_matrix(t, "Cadi", Cadi.T())

    # Trajectory-resolved diabatic TD-SE amplitudes
    # Format: saver.add_dataset("C_dia", (_nsteps, _ntraj, _ndia), "C")
    if _has_slot(saver, "Cdia"):
        saver.save_matrix(t, "Cdia", Cdia.T())



def save_hdf5_3D_new(saver, i, dyn_var, txt_type=0):
    """
    saver - can be either hdf5_saver or mem_saver

    txt_type ( int ): 0 - standard, all the timesteps, 1 - only the current one

    """

    t = 0
    if txt_type == 0:
        t = i

    # Average adiabatic density matrices
    # Format: saver.add_dataset("D_adi", (_nsteps, _nadi, _nadi), "C")
    if _has_slot(saver, "D_adi"):
        dm_adi = dyn_var.compute_average_dm(1)
        saver.save_matrix(t, "D_adi", dm_adi)

    # Average diabatic density matrices
    # Format: saver.add_dataset("D_dia", (_nsteps, _ndia, _ndia), "C")
    if _has_slot(saver, "D_dia"):
        dm_dia = dyn_var.compute_average_dm(0)
        saver.save_matrix(t, "D_dia", dm_dia)

    # Trajectory-resolved coordinates
    # Format: saver.add_dataset("q", (_nsteps, _ntraj, _dof), "R")
    if _has_slot(saver, "q"):
        q = dyn_var.get_coords()
        saver.save_matrix(t, "q", q.T())

    # Trajectory-resolved momenta
    # Format: saver.add_dataset("p", (_nsteps, _ntraj, _dof), "R")
    if _has_slot(saver, "p"):
        p = dyn_var.get_momenta()
        saver.save_matrix(t, "p", p.T())

    # Trajectory-resolved active forces
    # Format: saver.add_dataset("f", (_nsteps, _ntraj, _dof), "R")
    if _has_slot(saver, "f"):
        f = dyn_var.get_forces()
        saver.save_matrix(t, "f", f.T())

    # Trajectory-resolved adiabatic TD-SE amplitudes
    # Format: saver.add_dataset("C_adi", (_nsteps, _ntraj, _nadi), "C")
    if _has_slot(saver, "Cadi"):
        Cadi = dyn_var.get_ampl_adi()
        saver.save_matrix(t, "Cadi", Cadi.T())

    # Trajectory-resolved diabatic TD-SE amplitudes
    # Format: saver.add_dataset("C_dia", (_nsteps, _ntraj, _ndia), "C")
    if _has_slot(saver, "Cdia"):
        Cdia = dyn_var.get_ampl_dia()
        saver.save_matrix(t, "Cdia", Cdia.T())

    # Trajectory-resolved MMST electronic "coordinate"
    # Format: saver.add_dataset("q_mm", (_nsteps, _ntraj, _nadi), "C")
    if _has_slot(saver, "q_mm"):
        q_mm = dyn_var.get_q_mm()
        saver.save_matrix(t, "q_mm", q_mm.T())

    # Trajectory-resolved MMST electronic "momentum"
    # Format: saver.add_dataset("p_mm", (_nsteps, _ntraj, _nadi), "C")
    if _has_slot(saver, "p_mm"):
        p_mm = dyn_var.get_p_mm()
        saver.save_matrix(t, "p_mm", p_mm.T())

    # Average adiabatic coherence indicator matrices
    # Format: saver.add_dataset("coherence_adi", (_nsteps, _nadi, _nadi), "R")
    if _has_slot(saver, "coherence_adi"):
        coh_adi = dyn_var.compute_coherence_indicator(1)
        saver.save_matrix(t, "coherence_adi", coh_adi)

    # Average diabatic coherence indicator matrices
    # Format: saver.add_dataset("coherence_dia", (_nsteps, _ndia, _ndia), "R")
    if _has_slot(saver, "coherence_dia"):
        coh_dia = dyn_var.compute_coherence_indicator(0)
        saver.save_matrix(t, "coherence_dia", coh_dia)

    # Wavepacket width
    # Format: saver.add_dataset("wp_width", (_nsteps, _ntraj, _dof), "R")
    if _has_slot(saver, "wp_width"):
        wp_width = dyn_var.get_wp_width()
        saver.save_matrix(t, "wp_width", wp_width.T())

    # Trajectory-resolved quantum momenta
    # Format: saver.add_dataset("p_quant", (_nsteps, _ntraj, _dof), "R")
    if _has_slot(saver, "p_quant"):
        p_quant = dyn_var.get_p_quant()
        saver.save_matrix(t, "p_quant", p_quant.T())

    # Trajectory-resolved exact vector potential
    # Format: saver.add_dataset("VP", (_nsteps, _ntraj, _dof), "R")
    if _has_slot(saver, "VP"):
        VP = dyn_var.get_VP()
        saver.save_matrix(t, "VP", VP.T())

    # Trajectory-resolved XF-based decoherence force
    # Format: saver.add_dataset("f_xf", (_nsteps, _ntraj, _dof), "R")
    if _has_slot(saver, "f_xf"):
        f_xf = dyn_var.get_f_xf()
        saver.save_matrix(t, "f_xf", f_xf.T())

    # Nonclassical force in QTSH
    # Format: saver.add_dataset("qtsh_f_nc", (_nsteps, _ntraj, _dof), "R")
    if _has_slot(saver, "qtsh_f_nc"):
        qtsh_f_nc = dyn_var.get_qtsh_f_nc()
        saver.save_matrix(t, "qtsh_f_nc", qtsh_f_nc.T())

    # Average decoherence rate
    # Format: saver.add_dataset("ave_decoherence_rates", (_nsteps, _nadi, _nadi), "R") 
    if _has_slot(saver, "ave_decoherence_rates"):
        ave_decoherence_rates = dyn_var.get_ave_decoherence_rates()
        saver.save_matrix(t, "ave_decoherence_rates", ave_decoherence_rates)



def save_hdf5_4D(
        saver,
        dyn_var,
        i,
        tr,
        hvib_adi,
        hvib_dia,
        St,
        U,
        projector,
        q_aux=None,
        p_aux=None,
        nab_phase=None,
        txt_type=0):
    """
    saver - can be either hdf5_saver or mem_saver

    txt_type ( int ): 0 - standard, all the timesteps, 1 - only the current one

    """

    t = 0
    if txt_type == 0:
        t = i

    # Trajectory-resolved vibronic Hamiltoninans in the adiabatic representation
    # Format: saver.add_dataset("hvib_adi", (_nsteps, _ntraj, _nadi, _nadi), "C")
    if _has_slot(saver, "hvib_adi"):
        saver.save_multi_matrix(t, tr, "hvib_adi", hvib_adi)

    # Trajectory-resolved vibronic Hamiltoninans in the diabatic representation
    # Format: saver.add_dataset("hvib_dia", (_nsteps, _ntraj, _ndia, _ndia), "C")
    if _has_slot(saver, "hvib_dia"):
        saver.save_multi_matrix(t, tr, "hvib_dia", hvib_dia)

    # Trajectory-resolved time-overlaps of the adiabatic states
    # Format: saver.add_dataset("St", (_nsteps, _ntraj, _nadi, _nadi), "C")
    if St is not None and _has_slot(saver, "St"):
        saver.save_multi_matrix(t, tr, "St", St)

    # Trajectory-resolved diabatic-to-adiabatic transformation matrices
    # Format: saver.add_dataset("basis_transform", (_nsteps, _ntraj, _ndia, _nadi), "C")
    if _has_slot(saver, "basis_transform"):
        saver.save_multi_matrix(t, tr, "basis_transform", U)

    # Trajectory-resolved projector matrices (from the raw adiabatic to consistent adiabatic)
    # Format: saver.add_dataset("projector", (_nsteps, _ntraj, _nadi, _nadi), "C")
    if _has_slot(saver, "projector"):
        saver.save_multi_matrix(t, tr, "projector", projector)

    # Trajectory-resolved auxiliary coordinates
    # Format: saver.add_dataset("q_aux", (_nsteps, _ntraj, _nadi, _ndof), "R")
    if _has_slot(saver, "q_aux"):
        saver.save_multi_matrix(t, tr, "q_aux", q_aux)

    # Trajectory-resolved auxiliary momenta
    # Format: saver.add_dataset("p_aux", (_nsteps, _ntraj, _nadi, _ndof), "R")
    if _has_slot(saver, "p_aux"):
        saver.save_multi_matrix(t, tr, "p_aux", p_aux)

    # Trajectory-resolved spatial derivatives of coefficient phase
    # Format: saver.add_dataset("nab_phase", (_nsteps, _ntraj, _nadi, _ndof), "R")
    if _has_slot(saver, "nab_phase"):
        saver.save_multi_matrix(t, tr, "nab_phase", nab_phase)



    # Instantaneous energy gaps
    # Format: saver.add_dataset("energy_gaps", (_nsteps, _ntraj, _nadi, _nadi), "R")
    if _has_slot(saver, "energy_gaps"):
        x = dyn_var.get_energy_gaps(tr)
        saver.save_multi_matrix(t, tr, "energy_gaps", x)

    # Running-average (mean) energy gaps
    # Format: saver.add_dataset("mean_energy_gaps", (_nsteps, _ntraj, _nadi, _nadi), "R")
    if _has_slot(saver, "mean_energy_gaps"):
        x = dyn_var.get_mean_energy_gaps(tr)
        saver.save_multi_matrix(t, tr, "mean_energy_gaps", x)

    # Instantaneous energy gaps squared
    # Format: saver.add_dataset("energy_gaps2", (_nsteps, _ntraj, _nadi, _nadi), "R")
    if _has_slot(saver, "energy_gaps2"):
        x = dyn_var.get_energy_gaps2(tr)
        saver.save_multi_matrix(t, tr, "energy_gaps2", x)

    # Running-average (mean) energy gaps squared
    # Format: saver.add_dataset("mean_energy_gaps2", (_nsteps, _ntraj, _nadi, _nadi), "R")
    if _has_slot(saver, "mean_energy_gaps2"):
        x = dyn_var.get_mean_energy_gaps2(tr)
        saver.save_multi_matrix(t, tr, "mean_energy_gaps2", x)

    # Instantaneous energy gap fluctuations
    # Format: saver.add_dataset("energy_gap_fluctuations", (_nsteps, _ntraj, _nadi, _nadi), "R")
    if _has_slot(saver, "energy_gap_fluctuations"):
        x = dyn_var.get_energy_gap_fluctuations(tr)
        saver.save_multi_matrix(t, tr, "energy_gap_fluctuations", x)

    # Instantaneous energy gap correlation functions
    # Format: saver.add_dataset("energy_gap_correlations", (_nsteps, _ntraj, _nadi, _nadi), "R")
    if _has_slot(saver, "energy_gap_correlations"):
        x = dyn_var.get_energy_gap_correlations(tr)
        saver.save_multi_matrix(t, tr, "energy_gap_correlations", x)



def save_hdf5_5D(saver, i, tr, idof, dc1_adi, txt_type=0):
    """
    saver - can be either hdf5_saver or mem_saver

    txt_type ( int ): 0 - standard, all the timesteps, 1 - only the current one

    """

    t = 0
    if txt_type == 0:
        t = i

    # Trajectory-resolved vibronic Hamiltoninans in the adiabatic representation
    # Format: saver.add_dataset("dc1_adi", (_nsteps, _ntraj, _nadi, _nadi, _ndof), "R")
    if _has_slot(saver, "dc1_adi"):
        saver.save_multi_matrix2(t, tr, idof, "dc1_adi", dc1_adi)


def save_tsh_data_123(_savers, params,
                      i, dt, Ekin, Epot, Etot, dEkin, dEpot, dEtot, Etherm, E_NHC, states,
                      pops, pops_raw, dm_adi, dm_adi_raw, dm_dia, dm_dia_raw, q, p, Cadi, Cdia
                      ):

    hdf5_output_level = params["hdf5_output_level"]
    mem_output_level = params["mem_output_level"]
    txt_output_level = params["txt_output_level"]
    txt2_output_level = params["txt2_output_level"]

    nsteps = params["nsteps"]
    progress_frequency = params["progress_frequency"]

    if progress_frequency is not None and progress_frequency > 0:
        if i % max(1, int(progress_frequency * nsteps)) == 0:
            print(F" step= {i}")

    if hdf5_output_level >= 1 and _savers["hdf5_saver"] is not None:
        save_hdf5_1D(_savers["hdf5_saver"], i, dt, Ekin, Epot, Etot, dEkin, dEpot, dEtot, Etherm, E_NHC)

    if mem_output_level >= 1 and _savers["mem_saver"] is not None:
        save_hdf5_1D(_savers["mem_saver"], i, dt, Ekin, Epot, Etot, dEkin, dEpot, dEtot, Etherm, E_NHC)

    if txt_output_level >= 1 and _savers["txt_saver"] is not None:
        save_hdf5_1D(_savers["txt_saver"], i, dt, Ekin, Epot, Etot, dEkin, dEpot, dEtot, Etherm, E_NHC)

    if txt2_output_level >= 1 and _savers["txt2_saver"] is not None:
        save_hdf5_1D(_savers["txt2_saver"], i, dt, Ekin, Epot, Etot, dEkin, dEpot, dEtot, Etherm, E_NHC, 1)

    if hdf5_output_level >= 2 and _savers["hdf5_saver"] is not None:
        save_hdf5_2D(_savers["hdf5_saver"], i, states)

    if mem_output_level >= 2 and _savers["mem_saver"] is not None:
        save_hdf5_2D(_savers["mem_saver"], i, states)

    if txt_output_level >= 2 and _savers["txt_saver"] is not None:
        save_hdf5_2D(_savers["txt_saver"], i, states)

    if txt2_output_level >= 2 and _savers["txt2_saver"] is not None:
        save_hdf5_2D(_savers["txt2_saver"], i, states, 1)

    if hdf5_output_level >= 3 and _savers["hdf5_saver"] is not None:
        save_hdf5_3D(_savers["hdf5_saver"], i, pops, pops_raw, dm_adi,
                     dm_adi_raw, dm_dia, dm_dia_raw, q, p, Cadi, Cdia)

    if mem_output_level >= 3 and _savers["mem_saver"] is not None:
        save_hdf5_3D(_savers["mem_saver"], i, pops, pops_raw, dm_adi, dm_adi_raw, dm_dia, dm_dia_raw, q, p, Cadi, Cdia)

    if txt_output_level >= 3 and _savers["txt_saver"] is not None:
        save_hdf5_3D(_savers["txt_saver"], i, pops, pops_raw, dm_adi, dm_adi_raw, dm_dia, dm_dia_raw, q, p, Cadi, Cdia)

    if txt2_output_level >= 3 and _savers["txt2_saver"] is not None:
        save_hdf5_3D(_savers["txt2_saver"], i, pops, pops_raw, dm_adi,
                     dm_adi_raw, dm_dia, dm_dia_raw, q, p, Cadi, Cdia, 1)


def save_time_overlaps(_savers, params, i, dyn_var, ham):
    """
    Saves the adiabatic time-overlap rows AFTER step i has been propagated,
    so frame i carries the interval [t_i, t_i+1] that step i crossed - under
    BOTH time_overlap_method conventions (the engine computes exactly this
    interval under method 1 - the model supplies it under method 0 via
    timestep_st). The pre-step savers would instead capture the PREVIOUS
    interval left in the slot, with a method-dependent frame 0.
    """
    hdf5_output_level = params["hdf5_output_level"]
    mem_output_level = params["mem_output_level"]
    txt_output_level = params["txt_output_level"]
    txt2_output_level = params["txt2_output_level"]
    is_nbra = params["is_nbra"]
    nprint = params["nprint"]

    # frame j of the strided grid carries <t_{j nprint} | t_{(j+1) nprint}>:
    # the PRODUCT of the raw intervals it spans, accumulated per call and
    # written when the span closes (a raw-dt sub-sample would make the
    # standard (St - St^H)/(2 dt_saved) post-processing off by nprint) -
    # the final frame closes early when nprint does not divide nsteps
    nsteps = params["nsteps"]
    acc = _savers.setdefault("_St_accum", {})
    close = ((i + 1) % nprint == 0) or (i == nsteps - 1)

    tr_range = [0] if is_nbra == 1 else range(dyn_var.ntraj)
    for tr in tr_range:
        St_step = ham.get_time_overlap_adi(Py2Cpp_int([0, tr]))
        St = St_step if tr not in acc else acc[tr] * St_step
        if not close:
            acc[tr] = St
            continue
        acc.pop(tr, None)
        frame = i // nprint
        for key, lvl, txt_type in (("hdf5_saver", hdf5_output_level, 0),
                                   ("mem_saver", mem_output_level, 0),
                                   ("txt_saver", txt_output_level, 0),
                                   ("txt2_saver", txt2_output_level, 1)):
            saver = _savers[key]
            if lvl >= 4 and saver is not None:
                if _has_slot(saver, "St"):
                    t = 0 if txt_type == 1 else frame
                    saver.save_multi_matrix(t, tr, "St", St)


def save_tsh_data_1234_new(_savers, params, i, dyn_var, ham):

    hdf5_output_level = params["hdf5_output_level"]
    mem_output_level = params["mem_output_level"]
    txt_output_level = params["txt_output_level"]
    txt2_output_level = params["txt2_output_level"]

    nsteps = params["nsteps"]
    nprint = params["nprint"]
    progress_frequency = params["progress_frequency"]

    if progress_frequency is not None and progress_frequency > 0:
        if i % max(1, int(progress_frequency * nsteps)) == 0:
            print(F" step= {i}")

    if i % nprint == 0:
        i = int(i / nprint)
    else:
        return None

    # =========== Using: def save_hdf5_1D_new(saver, i, params, dyn_var, ham, txt_type=0) =========

    if hdf5_output_level >= 1 and _savers["hdf5_saver"] is not None:
        save_hdf5_1D_new(_savers["hdf5_saver"], i, params, dyn_var, ham)

    if mem_output_level >= 1 and _savers["mem_saver"] is not None:
        # save_hdf5_1D(_savers["mem_saver"], i, dt, Ekin, Epot, Etot, dEkin, dEpot, dEtot, Etherm, E_NHC)
        save_hdf5_1D_new(_savers["mem_saver"], i, params, dyn_var, ham)

    if txt_output_level >= 1 and _savers["txt_saver"] is not None:
        # save_hdf5_1D(_savers["txt_saver"], i, dt, Ekin, Epot, Etot, dEkin, dEpot, dEtot, Etherm, E_NHC)
        save_hdf5_1D_new(_savers["txt_saver"], i, params, dyn_var, ham)

    if txt2_output_level >= 1 and _savers["txt2_saver"] is not None:
        # save_hdf5_1D(_savers["txt2_saver"], i, dt, Ekin, Epot, Etot, dEkin, dEpot, dEtot, Etherm, E_NHC, 1)
        save_hdf5_1D_new(_savers["txt2_saver"], i, params, dyn_var, ham, 1)

    # =========== Using : def save_hdf5_2D_new(saver, i, dyn_var, txt_type=0) =====================
    if hdf5_output_level >= 2 and _savers["hdf5_saver"] is not None:
        # save_hdf5_2D(_savers["hdf5_saver"], i, states)
        save_hdf5_2D_new(_savers["hdf5_saver"], i, dyn_var, ham)

    if mem_output_level >= 2 and _savers["mem_saver"] is not None:
        # save_hdf5_2D(_savers["mem_saver"], i, states)
        save_hdf5_2D_new(_savers["mem_saver"], i, dyn_var, ham)

    if txt_output_level >= 2 and _savers["txt_saver"] is not None:
        # save_hdf5_2D(_savers["txt_saver"], i, states)
        save_hdf5_2D_new(_savers["txt_saver"], i, dyn_var, ham)

    if txt2_output_level >= 2 and _savers["txt2_saver"] is not None:
        # save_hdf5_2D(_savers["txt2_saver"], i, states, 1)
        save_hdf5_2D_new(_savers["txt2_saver"], i, dyn_var, ham, 1)

    # ============ Using: def save_hdf5_3D_new(saver, i, dyn_var, txt_type=0) ==========================

    if hdf5_output_level >= 3 and _savers["hdf5_saver"] is not None:
        # save_hdf5_3D(_savers["hdf5_saver"], i, pops, pops_raw, dm_adi, dm_adi_raw, dm_dia, dm_dia_raw, q, p, Cadi, Cdia)
        save_hdf5_3D_new(_savers["hdf5_saver"], i, dyn_var)

    if mem_output_level >= 3 and _savers["mem_saver"] is not None:
        # save_hdf5_3D(_savers["mem_saver"], i, pops, pops_raw, dm_adi, dm_adi_raw, dm_dia, dm_dia_raw, q, p, Cadi, Cdia)
        save_hdf5_3D_new(_savers["mem_saver"], i, dyn_var)

    if txt_output_level >= 3 and _savers["txt_saver"] is not None:
        # save_hdf5_3D(_savers["txt_saver"], i, pops, pops_raw, dm_adi, dm_adi_raw, dm_dia, dm_dia_raw, q, p, Cadi, Cdia)
        save_hdf5_3D_new(_savers["txt_saver"], i, dyn_var)

    if txt2_output_level >= 3 and _savers["txt2_saver"] is not None:
        # save_hdf5_3D(_savers["txt2_saver"], i, pops, pops_raw, dm_adi, dm_adi_raw, dm_dia, dm_dia_raw, q, p, Cadi, Cdia, 1)
        save_hdf5_3D_new(_savers["txt2_saver"], i, dyn_var, 1)

    # ============ Using: def save_hdf5_4D_new(saver, i, tr, dyn_var, ham, txt_type=0) ====================

    is_nbra = params["is_nbra"]
    time_overlap_method = params["time_overlap_method"]

    tr_range = []
    if is_nbra == 1:
        tr_range = [0]
    else:
        tr_range = range(dyn_var.ntraj)

    for tr in tr_range:

        U = ham.get_basis_transform(Py2Cpp_int([0, tr]))
        # St is saved AFTER the step by save_time_overlaps: at this point the
        # slot still holds the PREVIOUS interval (or the init ingestion)
        St = None
        T = dyn_var.get_proj_adi(tr)

        q_aux, p_aux, nab_phase = None, None, None
        if dyn_var.shxf_vars_status == 1 or dyn_var.mqcxf_vars_status == 1:
            q_aux = dyn_var.get_coords_aux(tr)
            p_aux = dyn_var.get_momenta_aux(tr)
            nab_phase = dyn_var.get_nab_phase(tr)

        if hdf5_output_level >= 4 and _savers["hdf5_saver"] is not None:
            hvib_adi = ham.get_hvib_adi(Py2Cpp_int([0, tr]))
            hvib_dia = ham.get_hvib_dia(Py2Cpp_int([0, tr]))
            save_hdf5_4D(_savers["hdf5_saver"], dyn_var, i, tr, hvib_adi, hvib_dia, St, U, T, q_aux, p_aux, nab_phase)

        if mem_output_level >= 4 and _savers["mem_saver"] is not None:
            hvib_adi = ham.get_hvib_adi(Py2Cpp_int([0, tr]))
            hvib_dia = ham.get_hvib_dia(Py2Cpp_int([0, tr]))
            save_hdf5_4D(_savers["mem_saver"], dyn_var, i, tr, hvib_adi, hvib_dia, St, U, T, q_aux, p_aux, nab_phase)

        if txt_output_level >= 4 and _savers["txt_saver"] is not None:
            hvib_adi = ham.get_hvib_adi(Py2Cpp_int([0, tr]))
            hvib_dia = ham.get_hvib_dia(Py2Cpp_int([0, tr]))
            save_hdf5_4D(_savers["txt_saver"], dyn_var, i, tr, hvib_adi, hvib_dia, St, U, T, q_aux, p_aux, nab_phase)

        if txt2_output_level >= 4 and _savers["txt2_saver"] is not None:
            hvib_adi = ham.get_hvib_adi(Py2Cpp_int([0, tr]))
            hvib_dia = ham.get_hvib_dia(Py2Cpp_int([0, tr]))
            save_hdf5_4D(_savers["txt2_saver"], dyn_var, i, tr, hvib_adi, hvib_dia, St, U, T, q_aux, p_aux, nab_phase, 1)

    # ============= Using save_hdf5_5D(saver, i, tr, idof, dc1) =========================
    for tr in tr_range:
        for idof in range(dyn_var.ndof):

            dc1 = ham.get_dc1_adi(idof, (Py2Cpp_int([0, tr])))  # .real()
            dc1 = dc1.real()
            # T = dyn_var.get_proj_adi(tr)
            # dc1 = (T.H() * dc1 * T).real()

            if hdf5_output_level >= 5 and _savers["hdf5_saver"] is not None:
                save_hdf5_5D(_savers["hdf5_saver"], i, tr, idof, dc1)

            if mem_output_level >= 5 and _savers["mem_saver"] is not None:
                save_hdf5_5D(_savers["mem_saver"], i, tr, idof, dc1)

            if txt_output_level >= 5 and _savers["txt_saver"] is not None:
                save_hdf5_5D(_savers["txt_saver"], i, tr, idof, dc1)

            if txt2_output_level >= 5 and _savers["txt2_saver"] is not None:
                # the txt2 saver holds the current step only (txt_type 1)
                save_hdf5_5D(_savers["txt2_saver"], i, tr, idof, dc1, 1)


def print_results12(i, dt, res, prefix, file_output_level):
    """
    Print out the properties at given timestep i to the files

    Args:
        i ( int ): index of the timestep
        dt ( double ): time step size [ units: a.u. ]
        res ( tuple ): contains the properties (at given time) to print out, the order is as follows:

            * q = res[0] ( MATRIX(ndof, ntraj) ): coordiantes of all trajectories
            * p = res[1] ( MATRIX(ndof, ntraj) ): momenta of all trajectories
            * Ekin = res[2] ( double ): average kinetic energy of the ensemble
            * Epot = res[3] ( double ): average potential energy of the ensemble
            * Etot = res[4] ( double ): average total energy of the ensemble
            * dEkin = res[5] ( double ): fluctuation of the average kinetic energy of the ensemble
            * dEpot = res[6] ( double ): fluctuation of the average potential energy of the ensemble
            * dEtot = res[7] ( double ): fluctuation of the average total energy of the ensemble
            * Cadi = res[8] ( CMATRIX(nadi, ntraj) ): amplitudes of adiabatic states for all trajectories
            * Cdia = res[9] ( CMATRIX(ndia, ntraj) ): amplitudes of diabatic states for all trajectories
            * dm_adi = res[10] ( CMATRIX(nadi, nadi) ): adiabatic density matrix averaged over all trajectories
            * dm_dia = res[11] ( CMATRIX(ndia, ndia) ): diabatic density matrix averaged over all trajectories
            * pops = res[12] ( list of nadi doubles ): average populations of all adiabatic states
            * states = res[13] ( list of ntraj ints ): electronic states of all trajectories

        prefix ( string ): the common prefix of the output (usually the directory name)
        file_output_level ( int ): the verbosity of the output

    Returns:
        None

    """

    q, p = res[0], res[1]
    Ekin, Epot, Etot = res[2], res[3], res[4]
    dEkin, dEpot, dEtot = res[5], res[6], res[7]
    Cadi, Cdia = res[8], res[9]
    dm_adi, dm_dia = res[10], res[11]
    pops = res[12]
    states = res[13]

    # File output
    if file_output_level >= 1:
        add_doublelist2file("%s/energies.txt" % (prefix), i * dt, [Ekin, Epot, Etot, dEkin, dEpot, dEtot])
        add_cmatrix2file("%s/D_adi.txt" % (prefix), i * dt, dm_adi)
        add_cmatrix2file("%s/D_dia.txt" % (prefix), i * dt, dm_dia)
        add_matrix2file("%s/SH_pop.txt" % (prefix), i * dt, pops)

    # File output
    if file_output_level >= 2:
        add_matrix2file("%s/q.txt" % (prefix), i * dt, q)
        add_matrix2file("%s/p.txt" % (prefix), i * dt, p)
        add_cmatrix2file("%s/C_adi.txt" % (prefix), i * dt, Cadi)
        add_cmatrix2file("%s/C_dia.txt" % (prefix), i * dt, Cdia)
        add_intlist2file("%s/states.txt" % (prefix), i * dt, states)


def print_results3(i, dt, res, prefix, file_output_level, tr):
    """
    Print out the properties at given timestep i to the files

    Args:
        i ( int ): index of the timestep
        dt ( double ): time step size [ units: a.u. ]
        res ( tuple ): contains the properties (at given time) to print out, the order is as follows:

            * hvib_adi = res[0] ( CMATRIX(nadi, nadi) ): adiabatic Hamiltonian for a given trajectory
            * hvib_dia = res[1] ( CMATRIX(ndia, ndia) ): diabatic Hamiltonian for a given trajectory
            * St = res[2] ( CMATRIX(nadi, nadi) ): time-overlap of the adiabatic states for a given trajectory
            * U = res[3] ( CMATRIX(ndia, nadi) ): dia-to-adi transformation matrix for a given trajectory

        prefix ( string ): the common prefix of the output (usually the directory name)
        file_output_level ( int ): the verbosity of the output
        tr ( int ): the index of the trajectory that we handle

    Returns:
        None

    """

    hvib_adi = res[0]
    hvib_dia = res[1]
    St = res[2]
    U = res[3]

    # File output
    if file_output_level >= 3:
        add_cmatrix2file("%s/Hvib_adi_%i.txt" % (prefix, tr), i * dt, hvib_adi)
        add_cmatrix2file("%s/Hvib_dia_%i.txt" % (prefix, tr), i * dt, hvib_dia)
        add_cmatrix2file("%s/St_%i.txt" % (prefix, tr), i * dt, St)
        add_cmatrix2file("%s/basis_transform_%i.txt" % (prefix, tr), i * dt, U)


def read_results(prefix, file_output_level, nadi, ndia, ndof, ntraj):
    """
    This function
    """

    obs_T = None
    obs_q, obs_p = None, None
    obs_Ekin, obs_Epot, obs_Etot = None, None, None
    obs_dEkin, obs_dEpot, obs_dEtot = None, None, None
    obs_Cadi, obs_Cdia = None, None
    obs_dm_adi, obs_dm_dia = None, None
    obs_pop, obs_states = None, None
    obs_hvib_adi, obs_hvib_dia, obs_St, obs_U = None, None, None, None

    if file_output_level >= 1:
        res = data_read.get_data_from_file2("%s/energies.txt" % (prefix), range(0, 7))
        obs_T = res[0]
        obs_Ekin = res[1]
        obs_Epot = res[2]
        obs_Etot = res[3]
        obs_dEkin = res[4]
        obs_dEpot = res[5]
        obs_dEtot = res[6]

        obs_dm_adi = file2cmatrix("%s/D_adi.txt" % (prefix), nadi, nadi)
        obs_dm_dia = file2cmatrix("%s/D_dia.txt" % (prefix), ndia, ndia)
        obs_pop = file2matrix("%s/SH_pop.txt" % (prefix), nadi, 1)

    if file_output_level >= 2:
        obs_q = file2matrix("%s/q.txt" % (prefix), ndof, ntraj)
        obs_p = file2matrix("%s/p.txt" % (prefix), ndof, ntraj)
        obs_C_adi = file2cmatrix("%s/C_adi.txt" % (prefix), nadi, ntraj)
        obs_C_dia = file2cmatrix("%s/C_dia.txt" % (prefix), ndia, ntraj)
        obs_states = file2intlist("%s/states.txt" % (prefix))

    if file_output_level >= 3:
        obs_hvib_adi, obs_hvib_dia, obs_St, obs_U = [], [], [], []
        for tr in range(ntraj):
            hvib_adi = file2cmatrix("%s/Hvib_adi_%i.txt" % (prefix, tr), nadi, nadi)
            hvib_dia = file2cmatrix("%s/Hvib_dia_%i.txt" % (prefix, tr), nadi, nadi)
            St = file2cmatrix("%s/St_%i.txt" % (prefix, tr), nadi, nadi)
            U = file2cmatrix("%s/basis_transform_%i.txt" % (prefix, tr), ndia, nadi)

            obs_hvib_adi.append(hvib_adi)
            obs_hvib_dia.append(hvib_dia)
            obs_St.append(St)
            obs_U.append(U)

    return obs_T, obs_q, obs_p, obs_Ekin, obs_Epot, obs_Etot, obs_dEkin, obs_dEpot, obs_dEtot, \
        obs_Cadi, obs_Cdia, obs_dm_adi, obs_dm_dia, obs_pop, obs_states, obs_hvib_adi, obs_hvib_dia, obs_St, obs_U
