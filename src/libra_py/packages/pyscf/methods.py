# *********************************************************************************
# * Copyright (C) 2026  Jieyang Gu and Alexey V. Akimov
# *
# * This file is distributed under the terms of the GNU General Public License
# * as published by the Free Software Foundation, either version 3 of
# * the License, or (at your option) any later version.
# * See the file LICENSE in the root directory of this distribution
# * or <http://www.gnu.org/licenses/>.
# *
# *********************************************************************************/
"""
.. module:: methods
   :platform: Unix, Windows
   :synopsis: This module implements the Libra/PySCF interface function

.. moduleauthor::
       Alexey V. Akimov, Jieyang Gu

"""


import os, sys, math, re, struct, copy, subprocess
import numpy as np
from liblibra_core import MATRIX, CMATRIX, CMATRIXList, Py2Cpp_int, Cpp2Py, Random

# Fisrt, we add the location of the library to test to the PYTHON path
from libra_py.packages.pyscf.implementations.cisd import CISD
from libra_py.packages.pyscf.implementations.casscf import CASSCF
from libra_py.packages.pyscf.interfaces import ElectronicStructureStrategy, MolecularGeometry

from libra_py.packages.cp2k import methods as cp2k
from libra_py import data_conv
from libra_py import units
from libra_py.packages.driver_forces import fill_diagonal_forces, force_diagonals
from libra_py.packages.driver_fragments import carrier_operators_from_densities, single_excitation_rows

class tmp:
    pass


def pyscf_compute_adi(q, params, full_id):
    """
    The PySCF model function of generic_recipe: the SA-CASSCF (or CISD) states of every trajectory's geometry, their
    energies, every root's gradient and the time overlaps between consecutive geometries - the engine derives the
    couplings (nac_update_method 2). Optional keys of params:

        dia_basis (str): "none" [default] - the identity as basis_transform. "fragment" - the fragment-localized states
            of the step as the diabatic basis (step 3's vocabulary, as the CP2K driver): localized_carrier ('hole' /
            'electron' / 'both'), diabatization_scheme ('mulliken' - the Mulliken operators of the step's AO overlap,
            'pod' - libra's own POD block projectors of the step's Fock matrix) over fragment_atom_lists (1-based,
            disjoint, every atom assigned), do_diabatic_state_reordering (0 / 1 / 2 / 22 / 5 / 52), the thresholds
            diabatic_state_reordering_alpha / _min_overlap, occupancy_tolerance. The carriers are the hole and the
            electron of every excited state relative to the ground state (state 0) from the one-particle transition
            densities of the CASSCF states (carrier_operators_from_densities). The SINGLE excitations - the states that
            remove fewer than 1.5 electrons from the ground state's occupied space - form the localized manifold, the
            doubles and the ground state stay energy eigenstates labelled ("ci", rank) when a double is among the
            states, otherwise the ground state keeps the ("gs",) label and every excited state is localized. The
            method must be 'casscf' (the CISD implementation has no transition densities). The
            diabatic index of the first step is written to <working_directory_prefix>_itraj{i}/dia-record.txt (the
            localized states first, hole-major then electron for 'both', then the complement) and the following
            state to dia-state.npz there. The hand-over is the square basis_transform U[dia, adi], the identity
            ovlp_dia and dia_groups, as the CP2K driver's.
        working_directory_prefix (str): the records' folder stem [default "wd"]
    """

    # ================= Decode trajectory index =================
    Id = Cpp2Py(full_id)
    itraj = Id[-1]

    # ================= Extract coordinates =================
    coords = q.col(itraj)
    coordinates = data_conv.MATRIX2nparray(coords, float).reshape(-1,3)/units.Angst # in Angstrom

    ndof = coords.num_of_rows
    nat = ndof // 3

    # ================= Safe param access =================
    params.setdefault("is_first_time", {})
    params.setdefault("act_state", {})
    params.setdefault("pyscf_obj", {})
    params.setdefault("coords_prev", {})
    params.setdefault("dia_state", {})
    dia_basis = params.get("dia_basis", "none")
    if dia_basis not in ("none", "fragment"):
        raise ValueError(F"dia_basis must be 'none' or 'fragment', got {dia_basis!r}")
    frag = cp2k.fragment_options(params) if dia_basis == "fragment" else None
    if frag is not None and frag["scheme"] == "pod_cp2k":
        raise ValueError("diabatization_scheme 'pod_cp2k' reads CP2K's projection prints - the PySCF driver builds the "
                         "'mulliken' operators of the step's AO overlap or libra's own 'pod' projectors of its Fock matrix")

    is_first_time = params["is_first_time"].get(itraj, True)
    act_state = params["act_state"].get(itraj, 0)
    
    _basis = params.get("basis", "sto-3g")
    _charge = params.get("charge", 0)
    nstates = params.get("nstates", 2)
    method = params.get("method", "casscf")
    # The default is CAS(2,2) 
    _norbcas = params.get("norbcas", 2)
    _nelecas = params.get("nelecas", 2)
    if frag is not None and method != "casscf":
        raise ValueError(F"dia_basis 'fragment' needs method 'casscf' - the {method!r} implementation has no transition "
                         F"densities or fragment operators")

    # ================= Read parameters =================
    #dt = float(params.get("dt", 41.0)) # no coupling is derived here, the time step is the engine's
    _atom_labels = params["atom_labels"]
    
    # ================= Previous coordinates =================

    pyscf_obj = None
    if is_first_time:
        coords_prev = copy.deepcopy(coordinates)
        
        if method=="casscf":
            pyscf_obj = CASSCF(norbcas=_norbcas, nelecas=_nelecas, nroots=nstates, basis=_basis, charge=_charge)
        elif method=="cisd":
            pyscf_obj = CISD(nroots=nstates, basis=_basis, charge=_charge)     

    else:
        #coords_prev = copy.deepcopy(params["coords_prev"].get(itraj))
        pyscf_obj = copy.deepcopy(params["pyscf_obj"].get(itraj))
        #pyscf_obj = params["pyscf_obj"].get(itraj)

    #geom_prev = MolecularGeometry(atom_labels = _atom_labels, coords_angstrom=np.array(coords_prev) )
    geom = MolecularGeometry(atom_labels = _atom_labels, coords_angstrom=np.array(coordinates) )
    
    
    #pyscf_obj.set_geom_and_run_hf(geom_prev)
    #_ = [pyscf_obj.compute_energy(root) for root in range(nstates)]
    #_ = [pyscf_obj.compute_gradient(root) for root in range(nstates)]

    pyscf_obj.set_geom_and_run_hf(geom)
    energies = [pyscf_obj.compute_energy(root) for root in range(nstates)]
    grad = [pyscf_obj.compute_gradient(root) for root in range(nstates)]
    if is_first_time:
        # the first step has no previous geometry: its own solution is the previous one, the time overlap the identity
        pyscf_obj.cache_current_as_previous()

        
    #print(F"coords_prev = {coords_prev}")

    # ================= Compute overlaps =================
    st_ci = pyscf_obj.time_overlap_matrix(nstates)

    # ================= Build object =================
    obj = tmp()
    obj.ham_adi = CMATRIX(nstates, nstates)
    obj.basis_transform = CMATRIX(nstates, nstates)
    obj.time_overlap_adi = CMATRIX(nstates, nstates)

    # ================= Populate Hamiltonian =================
    for i in range(nstates):
        obj.ham_adi.set(i, i, complex(energies[i]) )
        obj.basis_transform.set(i, i, 1.0+0.0j)

        for j in range(nstates):
            obj.time_overlap_adi.set(i, j, complex(float(st_ci[i, j])) )

    if frag is not None:
        # the fragment-localized states as the DIABATIC basis: the hole and the electron of every excited CASSCF state
        # relative to the ground state over the states themselves, the single excitations localized (the doubles and
        # the ground state energy eigenstates of the complement), followed and aligned against the previous step, the
        # state of the following persisted per trajectory - the CP2K driver's hand-over
        wd = F"{params.get('working_directory_prefix', 'wd')}_itraj{itraj}"
        os.makedirs(wd, exist_ok=True)
        gamma_a, gamma_b = pyscf_obj.transition_densities(nstates)
        P = pyscf_obj.fragment_operators(frag["scheme"], frag["atoms"])
        O_hole, O_elec, _ = carrier_operators_from_densities(gamma_a, gamma_b, P, P)
        singles, doubles, holes = single_excitation_rows(O_hole)
        timestep = int(params.get("timestep", 0))
        state_file = F"{wd}/dia-state.npz"
        if is_first_time and timestep > 0 and os.path.exists(state_file):
            params["dia_state"][itraj] = cp2k.fragment_state_load(state_file)
            print(F"dia_basis fragment: the following of trajectory {itraj} resumed from {state_file} "
                  F"(saved at step {params['dia_state'][itraj]['step']})")
        elif is_first_time:
            params["dia_state"][itraj] = None
        St_frag = None if params["dia_state"].get(itraj) is None else np.asarray(st_ci, dtype=float)
        E_adi = np.asarray(energies, dtype=float)
        U_frag, dia_labels, params["dia_state"][itraj], rec = cp2k.fragment_diabatic_step(
            "ci_ops", np.eye(nstates), None, None, E_adi, St_frag, frag["kind"], params["dia_state"].get(itraj),
            frag["reordering"], frag["alpha"], frag["min_overlap"], frag["tol"], None, timestep,
            ci_ops=(O_hole, O_elec), ci_manifold=singles if doubles else False)
        cp2k.fragment_state_save(state_file, params["dia_state"][itraj])
        with open(F"{wd}/dia-record.txt", "a") as rf:
            if rec["O"] is None:
                rf.write(F"# step perm swap angle drift min_continuity edges unresolved - the diabatic index {rec['dia_index']}"
                         F"{' - the doubles ' + str(doubles) + ' stay energy eigenstates' if doubles else ''}\n")
            rf.write(F"{timestep} {list(int(c) for c in params['dia_state'][itraj]['perm'])} {int(rec['swap'])} {rec['angle']:.4f} "
                     F"{rec['drift']:.4f} {rec['min_continuity']:.6f} {rec['edges']} {int(rec['unresolved'])}\n")
        with open(F"{wd}/cas-record.txt", "a") as rf:
            rf.write(F"step {timestep}  E (Ha) {np.round(E_adi, 6).tolist()}  holes relative to S0 {np.round(holes, 3).tolist()}  "
                     F"singles {singles}  doubles {doubles}\n")
            for k in range(1, nstates):
                rf.write(F"  state {k}  hole on the fragments {np.round([O[k, k] for O in O_hole], 3).tolist()}  "
                         F"electron {np.round([O[k, k] for O in O_elec], 3).tolist()}\n")
        obj.basis_transform = data_conv.nparray2CMATRIX(np.asarray(U_frag, dtype=complex))
        obj.ovlp_dia = CMATRIX(nstates, nstates)
        for i in range(nstates):
            obj.ovlp_dia.set(i, i, 1.0 + 0.0j)
        obj.dia_groups = [int(x) for x in dia_labels]

    # ================== Forces ===============================
    # the gradients (nstates, nat, 3) are minus the forces the shared helper takes
    fill_diagonal_forces(obj, force_diagonals(nstates, ndof, nat, None, forces_all=-np.asarray(grad[:nstates])))

    # no coupling is handed over: the engine derives the couplings of the followed states from time_overlap_adi
    # with its own projector (nac_update_method 2) and builds hvib_adi = H - i D (hvib_update_method 1) - a
    # driver's own follower could disagree with the engine's at an ambiguous crossing
            
    # ================= Store state =================
    params["pyscf_obj"][itraj] = copy.deepcopy(pyscf_obj)
    #params["pyscf_obj"][itraj] = pyscf_obj
    #params["coords_prev"][itraj] = copy.deepcopy(coordinates)
    params["is_first_time"][itraj] = False

    return obj



