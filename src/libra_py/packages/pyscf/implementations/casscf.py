# *********************************************************************************
# * Copyright (C) 2026 Jieyang Gu <jieyanggu792@gmail.com>
# * This file is distributed under the terms of the GNU General Public License
# * as published by the Free Software Foundation, either version 3 of
# * the License, or (at your option) any later version.
# * See the file LICENSE in the root directory of this distribution
# * or <http://www.gnu.org/licenses/>.
# *
# *********************************************************************************/
"""
.. module:: pyscf.implementations.casscf
   :platform: Unix, Windows
   :synopsis: PySCF CASSCF implementation for Libra ES strategy.
.. moduleauthor::
       Jieyang Gu <jieyanggu792@gmail.com>

"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union, Sequence
import numpy as np
from pyscf import fci, gto, mcscf, scf
from libra_py.packages.pyscf.interfaces import ElectronicStructureStrategy, MolecularGeometry


@dataclass
class Cache:
    prev_mol: Optional[Any] = None
    prev_mc: Optional[Any] = None
    prev_mf: Optional[Any] = None

class CASSCF(ElectronicStructureStrategy):
    """PySCF-based CASSCF backend for the universal ES interface."""

    # 1) when a geom is set the HF is run; the MF is written to the member attribute
    #    `_mf`
    # 2) when the CASSCF energy is requested for a certain root, the CASSCF is run
    #    (number of roots requested = self._nroots). The resulting MC object is
    #    stored in the member attribute `mc`, which also contains energies of other
    #    states.

    def __init__(
        self,
        mol: Optional[Any] = None,
        norbcas: int = 0,
        nelecas:int = 0, 
        nroots: int = 1,
        basis: str = "sto-3g",
        unit: str = "Angstrom",
        charge: int = 0,
        cas_list: Optional[List[int]] = None,
    ) -> None:
        #setting up the initial state 
        super().__init__(mol=mol, nroots=nroots, basis=basis, unit=unit, charge=int(charge))
        self._norbcas: int = norbcas
        self._nelecas: Union[int, Tuple[int, int]] = nelecas  
        self._nroots: int = nroots
        self._basis: str = basis
        self._unit: str = unit  # default to Angstrom 
        self._charge: int = int(charge)
        self._mf: Optional[Any] = None
        self._mc: Optional[Any] = None
        self._ao_overlap: Optional[np.ndarray] = None  # AO overlap between consecutive geoms for time-overlap computation
        self._cache: Cache = Cache()
        self._cas_list: Optional[List[int]] = cas_list  # list of active space orbital indices (0-based); if None, use the first `n

    def save_cache(self) -> None:
        # Cache the current state directly (None is fine for first geometry).
        self._cache.prev_mol = self._mol
        self._cache.prev_mc = self._mc
        self._cache.prev_mf = self._mf

    def run_hf(self) -> None:
        # Clear the current state.
        self._mc = None
        self._ao_overlap = None

        # Set up the new molecule and run HF.
        charge: int = self._charge
        self._mol = gto.M(
            atom=";".join(
                f"{label} {coord[0]} {coord[1]} {coord[2]}"
                for label, coord in zip(self._geom.atom_labels, self._geom.coords_angstrom)
            ),
            basis=self._basis,
            unit=self._unit,
            charge=charge,
            spin=0,
        )
        #compute the AO overlap between new geom and the previous one 
        if self._cache.prev_mol is not None:
            self._ao_overlap = gto.intor_cross("int1e_ovlp", self._cache.prev_mol, self._mol)

        self._mf = scf.RHF(self._mol).run(verbose=0)

    def compute_energy(self, root: int) -> float:
        if self._mf is None:
            raise ValueError("HF must be run before computing CASSCF energies.")

        if self._mc is None:
            self._mc = mcscf.CASSCF(self._mf, self._norbcas, self._nelecas)
            self._mc.fcisolver = fci.direct_spin0.FCI(self._mol)
            self._mc.fcisolver.nroots = self._nroots
            if self._nroots > 1:
                self._mc = self._mc.state_average_([1.0 / self._nroots] * self._nroots)  # equal weights as default; required for gradients in pyscf

            # Correctly use currently converged HF spatial orbitals as initial guess for SA-CASSCF
            # Sort orbitals if cas_list is provided
            mo_coeff = self._mf.mo_coeff
            if self._cas_list is not None:
                # Notice: In PySCF mcscf.sort_mo, cas_list must be a list of 1-based indices
                # if your cas_list values passed in __init__ are 1-based, directly use it:
                mo_coeff = mcscf.sort_mo(self._mc, mo_coeff, self._cas_list)
             
            # Prepare CI coefficients from previous step (only if use_prev_ci is True)
            ci0 = None
            if getattr(self, '_use_prev_ci', False) and self._cache.prev_mc is not None:
                ci0 = getattr(self._cache.prev_mc, 'ci', None)

            if ci0 is not None:
                self._mc.kernel(mo_coeff, ci0=ci0)
            else:
                self._mc.kernel(mo_coeff)

        e_states: Optional[Sequence[float]] = getattr(self._mc, "e_states", None)
        if e_states is not None:
            if root < 0 or root >= len(e_states):
                raise IndexError(f"Requested root {root}, but only {len(e_states)} roots are available.")
            return float(np.asarray(e_states)[root])
        if root != 0:
            raise IndexError("Only root 0 is available for a single-state CASSCF calculation.")
        return float(self._mc.e_tot)

    def compute_gradient(self, root: int) -> np.ndarray:
        if self._mc is None:
            raise ValueError("CASSCF must be run before computing gradients.")
        e_states: Optional[Sequence[float]] = getattr(self._mc, "e_states", None)
        if e_states is not None:
            if root < 0 or root >= len(e_states):
                raise IndexError(f"Requested root {root}, but only {len(e_states)} roots are available.")
            return np.asarray(self._mc.nuc_grad_method(state=root).kernel())
        if root != 0:
            raise IndexError("Only root 0 is available for a single-state CASSCF calculation.")
        return np.asarray(self._mc.nuc_grad_method().kernel())

    def time_overlap_matrix(self, nroots: int) -> np.ndarray:
        """The overlaps <Psi_i(previous geometry)|Psi_j(this geometry)> of the SA-CASSCF states the dynamics runs on -
        the CI vectors and active orbitals of the cached previous CASSCF and of this step's (not a CASCI re-solved on
        the HF orbitals, whose states can reorder against the CASSCF ones), through the cross AO overlap of the two
        geometries. The signs are the CI vectors' own: a negative diagonal element marks a state whose vector changed
        sign between the geometries - the engine's phase correction and the fragment basis read it, a row flipped to a
        positive diagonal would hide it from both."""
        if self._mc is None:
            raise ValueError("CASSCF must be run before computing time-overlap matrix.")
        if self._cache is None or self._cache.prev_mc is None or self._cache.prev_mol is None:
            raise ValueError("Previous and current CASSCF states are required for time-overlap computation.")
        if self._ao_overlap is None:
            raise ValueError("Cached AO overlap between consecutive geometries is not available.")
        if nroots <= 0:
            raise ValueError(f"nroots must be positive, got {nroots}")
        nroots = int(nroots)
        prev_mc, curr_mc = self._cache.prev_mc, self._mc
        prev_roots = self._ci_vectors(prev_mc, nroots)
        curr_roots = self._ci_vectors(curr_mc, nroots)
        mo_prev_act = np.asarray(prev_mc.mo_coeff)[:, prev_mc.ncore:prev_mc.ncore + prev_mc.ncas]
        mo_curr_act = np.asarray(curr_mc.mo_coeff)[:, curr_mc.ncore:curr_mc.ncore + curr_mc.ncas]
        s12_mo = mo_prev_act.T @ self._ao_overlap @ mo_curr_act
        overlap = np.zeros((nroots, nroots), dtype=float)
        for i in range(nroots):
            for j in range(nroots):
                overlap[i, j] = fci.addons.overlap(prev_roots[i], curr_roots[j], curr_mc.ncas, curr_mc.nelecas, s=s12_mo)
        return np.asarray(np.real_if_close(overlap))

    @staticmethod
    def _ci_vectors(mc, nroots: int) -> List[np.ndarray]:
        """The CI vectors of a CASSCF object as a list of arrays - one vector for a single-state run."""
        ci = mc.ci
        if isinstance(ci, np.ndarray) and ci.ndim == 2 and nroots == 1:
            ci = [ci]
        if len(ci) < nroots:
            raise ValueError(f"Requested {nroots} roots, but only {len(ci)} are available.")
        return [np.asarray(v) for v in ci[:nroots]]

    def transition_densities(self, nroots: int) -> Tuple[np.ndarray, np.ndarray]:
        """The one-particle transition densities of the CASSCF states per spin over the MO basis of this step
        (mo_coeff): the core orbitals carry one electron per spin in every diagonal block, the active block half the
        spin-summed transition density of the FCI solver (the states are singlets), the virtuals nothing - so that
        tr(gamma[k, l] O) is the matrix element of a one-body operator O between the states. Returns (gamma_a, gamma_b),
        equal arrays (nroots, nroots, nmo, nmo)."""
        if self._mc is None:
            raise ValueError("CASSCF must be run before computing the transition densities.")
        mc = self._mc
        ci = self._ci_vectors(mc, nroots)
        nmo = np.asarray(mc.mo_coeff).shape[1]
        ncore, ncas = mc.ncore, mc.ncas
        gamma = np.zeros((nroots, nroots, nmo, nmo))
        core = np.arange(ncore)
        for k in range(nroots):
            gamma[k, k, core, core] = 2.0
            for l in range(nroots):
                gamma[k, l, ncore:ncore + ncas, ncore:ncore + ncas] = mc.fcisolver.trans_rdm1(ci[k], ci[l], ncas, mc.nelecas)
        gamma_a = 0.5 * gamma
        return gamma_a, gamma_a.copy()

    def ao_atom_map(self) -> np.ndarray:
        """The 0-based atom of every AO of this step's molecule (the spherical AO order of PySCF)."""
        mol = self._mol
        return np.array([mol.bas_atom(b) for b in range(mol.nbas)
                         for _ in range((2 * mol.bas_angular(b) + 1) * mol.bas_nctr(b))], dtype=int)

    def fragment_operators(self, scheme: str, fragment_atom_lists, kept_block: int = 0) -> List[np.ndarray]:
        """The fragment operators of this step in the MO basis of the CASSCF orbitals (one per fragment, (nmo, nmo)):
        'mulliken' - the Mulliken operators of the AO overlap over the fragments' AOs, 'pod2l' / 'pod2gs' - libra's
        second-generation POD of the HF Fock matrix (the fragments orthogonalized first, the fragment states then
        orthonormalized as a whole by Lowdin or by Gram-Schmidt keeping kept_block's), 'pod' - libra's own POD block
        projectors of the HF Fock matrix in the Lowdin-orthogonalized AO basis (the projector of a block is its Lowdin
        projector whatever the Fock matrix)."""
        if self._mc is None or self._mf is None:
            raise ValueError("CASSCF must be run before building the fragment operators.")
        from libra_py import pod as _pod
        from libra_py.packages.cp2k.methods import mulliken_fragment_operators
        S = self._mol.intor("int1e_ovlp")
        C = np.asarray(self._mc.mo_coeff)                          # (nao, nmo)
        ao_atom = self.ao_atom_map()
        if scheme == "mulliken":
            frag_cols = [np.asarray(b, dtype=int) for b in _pod.ao_blocks(ao_atom, fragment_atom_lists)]
            return mulliken_fragment_operators(C.T, S, frag_cols)
        if scheme == "pod":
            pod = _pod.pod_block_states([np.asarray(self._mf.get_fock())], S, ao_atom, fragment_atom_lists)
            return _pod.pod_fragment_operators(pod, C.T, 0)
        if scheme in ("pod2l", "pod2gs"):
            pod = _pod.pod2_block_states([np.asarray(self._mf.get_fock())], S, ao_atom, fragment_atom_lists,
                                         orthogonalization="lowdin" if scheme == "pod2l" else "gram_schmidt",
                                         kept_block=int(kept_block))
            return _pod.pod_fragment_operators(pod, C.T, 0)
        raise ValueError(f"diabatization_scheme {scheme!r} - the PySCF driver builds 'mulliken', 'pod', 'pod2l' or 'pod2gs' "
                         f"operators")

    def compute_nac_vectors(self, use_etfs: bool = True) -> np.ndarray:
        if self._mc is None:
            raise ValueError("CASSCF must be run before computing NAC vectors.")

        nstates = self._nroots
        natm = int(self._mol.natm)
        
        mc_nacs = self._mc.nac_method()
        nacv = np.zeros((nstates, nstates, natm, 3), dtype=np.float64)

        for ket in range(nstates):
            for bra in range(nstates):
                if bra == ket:
                    continue
                nacv[bra, ket] = np.asarray(
                    mc_nacs.kernel(state=(ket, bra), use_etfs=use_etfs, mult_ediff=False),
                    dtype=np.float64,
                )

        return nacv

