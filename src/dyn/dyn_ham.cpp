/*********************************************************************************
 * Copyright (C) 2022 Alexey V. Akimov
 *
 * This file is distributed under the terms of the GNU General Public License
 * as published by the Free Software Foundation, either version 3 of
 * the License, or (at your option) any later version.
 * See the file LICENSE in the root directory of this distribution
 * or <http://www.gnu.org/licenses/>.
 *
 *********************************************************************************/
/**
  \file dyn_ham.cpp
  \brief The file implements the methods that update internal Hamiltonian
  variables in response to change of the dynamical variables
*/

#include "dyn_ham.h"
#include <exception>
#include <stdexcept>
#include "dyn_projectors.h"

/// liblibra namespace
namespace liblibra {

using namespace libcalculators;

/// libdyn namespace
namespace libdyn {

namespace bp = boost::python;

// the consume side of the model-attribute contract: an option that reads a model-supplied
// quantity refuses a model that hands over none of it - the engine's slot would otherwise keep
// its zeros or the previous step's values and the run would look healthy
static void check_model_handover(dyn_control_params &prms, nHamiltonian &ham) {
  int ntraj = ham.children.size();
  // the time overlap is read by the HST/NPI couplings, the state tracking, the phase correction
  // and the local-diabatization frame
  int reads_st = (prms.nac_update_method == 2 || prms.ham_transform_method == 4 ||
                  prms.state_tracking_algo != 0 || prms.do_phase_correction == 1);
  // the adiabatic propagation reads hvib_adi, the diabatic one hvib_dia
  int adi_tdse = (prms.rep_tdse == 1 || prms.rep_tdse == 3);
  for (int traj = 0; traj < ntraj; traj++) {
    nHamiltonian *ch = ham.children[traj];
    const char *missing = nullptr;
    const char *reader = nullptr;
    const char *cure = nullptr;
    if (prms.ham_update_method == 1 && !ch->ham_dia_present) {
      missing = "ham_dia"; reader = "ham_update_method = 1"; cure = "hand it over";
    } else if (prms.ham_update_method == 2 && !ch->ham_adi_present) {
      missing = "ham_adi"; reader = "ham_update_method = 2"; cure = "hand it over";
    } else if (prms.force_method != 0 && prms.ham_update_method == 2 && !ch->d1ham_adi_present) {
      missing = "d1ham_adi"; reader = "force_method other than 0 (the forces of an adiabatic-native model)";
      cure = "hand it over or use force_method = 0";
    } else if (prms.force_method != 0 && prms.ham_update_method == 1 && !ch->d1ham_dia_present) {
      missing = "d1ham_dia";
      reader = "force_method other than 0 (the forces of a diabatic model, the adiabatic ones follow from them)";
      cure = "hand it over or use force_method = 0";
    } else if (prms.time_overlap_method == 0 && reads_st && !ch->time_overlap_adi_present) {
      missing = "time_overlap_adi";
      reader = "time_overlap_method = 0 with a reader of the time overlap (nac_update_method 2, the state "
               "tracking, the phase correction or ham_transform_method 4)";
      cure = "hand it over or use time_overlap_method = 1 (St from the basis transforms)";
    } else if (prms.tsh_method == 3 || prms.tsh_method == 4) {
      ; // the Landau-Zener and Zhu-Nakamura hops read the gap, not the coupling
    } else if (adi_tdse && prms.hvib_update_method == 0 && !ch->hvib_adi_present) {
      missing = "hvib_adi"; reader = "hvib_update_method = 0 (the adiabatic propagation)";
      cure = "hand it over or use hvib_update_method = 1 (ham_adi - i nac_adi)";
    } else if (adi_tdse && prms.hvib_update_method == 1 && prms.nac_update_method == 0 &&
               !ch->nac_adi_present) {
      missing = "nac_adi";
      reader = "nac_update_method = 0 (hvib_update_method 1 builds hvib_adi = ham_adi - i nac_adi)";
      cure = "hand it over or use nac_update_method = 2 (the coupling derived from time_overlap_adi)";
    } else if (adi_tdse && prms.nac_update_method == 1 && prms.ham_update_method == 2 && !ch->dc1_adi_present) {
      // a diabatic model's dc1_dia may stay zero (strictly diabatic) - its adiabatic couplings follow from
      // d1ham_dia through the derivative tier, an adiabatic-native model has no other source
      missing = "dc1_adi";
      reader = "nac_update_method = 1 (the coupling of an adiabatic-native model from its derivative couplings and "
               "the momenta)";
      cure = "hand it over or use nac_update_method = 2 (the coupling derived from time_overlap_adi)";
    } else if (!adi_tdse && prms.hvib_update_method == 0 && prms.ham_transform_method != 3 &&
               prms.ham_transform_method != 4 && !ch->hvib_dia_present) {
      // transforms 3 and 4 build hvib_dia from the rotated ham_dia, and nac_dia stays zero under
      // nac_update_method 0 by the strictly diabatic contract
      missing = "hvib_dia"; reader = "hvib_update_method = 0 (the diabatic propagation)";
      cure = "hand it over or use hvib_update_method = 1";
    }
    if (missing != nullptr) {
      cout << "Error in update_Hamiltonian_variables(): " << reader << " reads the model's obj." << missing
           << ", but the model hands over none - " << cure << "\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }
  }
}

void align_dia_groups(dyn_control_params &prms, nHamiltonian &ham, nHamiltonian &ham_prev) {
  /**
    dia_tracking_algo 5: the model-supplied diabatic basis of every trajectory continued from the previous step
    within the groups of its dia_groups - for a group g the block B = (U_prev St U^H)[g, g] of the diabatic time
    overlap (U_prev the aligned transform stored last step, St the supplied adiabatic time overlap, U the transform
    handed over now), R = V_s U_s^H its polar factor (B = U_s S V_s^H, the BDCSVD of state_tracking_algo 5), the
    rows g of U become R^H U[g, :] - the aligned U is stored back into the child, so ham_dia, the diabatic amplitudes
    and the next step's U_prev all see it. The singletons (a negative label) and the first step (no supplied U
    before) are left as handed over. The step-3 twin: _follow_diabatic_interval of libra_py.workflows.nbra.step3
  */
  int ntraj = ham.children.size();
  for (int traj = 0; traj < ntraj; traj++) {
    nHamiltonian *child = ham.children[traj];
    nHamiltonian *prev = ham_prev.children[traj];
    if (!child->dia_groups_supplied) {
      std::cout << "Error in align_dia_groups(): dia_tracking_algo = 5 needs the model's dia_groups (the tracking group"
                << " of every diabatic state) - the CP2K driver hands them over under dia_basis 'fragment'\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }
    if (!child->basis_transform_supplied) {
      std::cout << "Error in align_dia_groups(): dia_tracking_algo = 5 aligns the model-supplied basis_transform - none"
                << " was handed over this step\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }
    if (child->ndia != child->nadi) {
      std::cout << "Error in align_dia_groups(): dia_tracking_algo = 5 needs a square basis_transform (ndia = nadi) -"
                << " the diabatic time overlap U_prev St U^H - got ndia = " << child->ndia << ", nadi = " << child->nadi << "\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }
    if (!child->time_overlap_adi_supplied) {
      std::cout << "Error in align_dia_groups(): dia_tracking_algo = 5 aligns by the model's time_overlap_adi - none was"
                << " handed over this step\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }
    if (child->hvib_dia_supplied) {
      std::cout << "Error in align_dia_groups(): the model's hvib_dia is expressed in its own diabatic basis, which"
                << " dia_tracking_algo = 5 rotates after the hand-over - align in the driver (do_diabatic_state_reordering"
                << " 5 / 52 with dia_tracking_algo 0) or hand over no hvib_dia (hvib_update_method 1)\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }
    if (!prev->basis_transform_supplied) { continue; }   // the first step - nothing to align against
    int n = child->ndia;
    CMATRIX U(child->get_basis_transform());
    CMATRIX O(n, n);
    O = prev->get_basis_transform() * child->get_time_overlap_adi() * U.H();
    vector<int> labels(child->dia_groups);
    for (int lab = 0; lab < n; lab++) {
      vector<int> g;
      for (int i = 0; i < n; i++) { if (labels[i] == lab) { g.push_back(i); } }
      int m = g.size();
      if (m < 2) { continue; }
      CMATRIX B(m, m), svd_u(m, m), svd_s(m, m), svd_v(m, m);
      for (int a = 0; a < m; a++) { for (int b = 0; b < m; b++) { B.set(a, b, O.get(g[a], g[b])); } }
      BDCSVD_decomposition(B, svd_u, svd_s, svd_v);
      CMATRIX R(svd_v * svd_u.H());
      // U[g, :] <- R^H U[g, :]
      CMATRIX Ug(m, n);
      for (int a = 0; a < m; a++) { for (int I = 0; I < n; I++) { Ug.set(a, I, U.get(g[a], I)); } }
      CMATRIX Ug_new(R.H() * Ug);
      for (int a = 0; a < m; a++) { for (int I = 0; I < n; I++) { U.set(g[a], I, Ug_new.get(a, I)); } }
    }
    child->set_basis_transform_by_val(U);
  }
}


void update_Hamiltonian_variables(dyn_control_params &prms,
                                  dyn_variables &dyn_var, nHamiltonian &ham,
                                  nHamiltonian &ham_prev, bp::object py_funct,
                                  bp::object model_params, int update_type) {
  /**
    This is a new function to replace all other "update_Hamiltonian functions"

    update_type:
       - 0: in response to changed q
       - 1: in response to changed p
       - 2: in response to changed electronic amplitudes

    The key parameters in the `prms` are:

    time_overlap_method:
       - 0: don't update - time_overlap_adi is supplied externally
       - 1: compute from the basis_transform pair U_prev^H U

    ham_update_method:
       - 0: don't update any Hamiltonians
       - 1: recompute only diabatic Hamiltonian [ default ]
       - 2: recompute only adiabatic Hamiltonian

    ham_transform_method:
       - 0: don't do any transforms
       - 1: diabatic->adiabatic according to internal diagonalization [ default
    ]
       - 2: diabatic->adiabatic according to internally stored basis
    transformation matrix [ requires ham_update_method == 1 and a
    model-supplied obj.basis_transform - the model's gauge replaces the
    internal diagonalization ]
       - 3: adiabatic->diabatic according to internally stored basis
    transformation matrix [ requires ham_update_method == 2: the loop
    never re-diagonalizes under this option, so a diabatic model's
    adiabatic side would stay frozen at its initial values -
    sanity_check rejects the combination ]
       - 4: adiabatic->diabatic according to local diabatization method
    [ requires ham_update_method == 2, ndia == nadi and supplied
    time-overlaps: ham_dia = V^H ham_adi V in the cumulative trajectory
    frame V(t_n) = T_n V(t_n-1). The frame OWNS the diabatic slot:
    basis_transform becomes V^H (also in the saved dataset - analysis
    scripts read the frame, not a model transform), and the frame is
    not persisted across restarts - a restarted run re-anchors its
    diabatic basis at its own first step ]
    Option 1 is skipped when ham_update_method == 2: an adiabatic-native
    model never populates ham_dia, so there is nothing to transform from.


    nac_update_method:
       - 0: don't update them (e.g. for simplest NAC)
       - 1: update according to changed momentum and existing derivative
    couplings [ default ]
       - 2: update according to time-overlaps (only time-derivative NACs)

    hvib_update_method:
       - 0: don't recompute Hvib (e.g. we read it)
       - 1: update according to changed NACs and energies

  */

  int nadi = ham.nadi;
  int ndof = dyn_var.ndof;
  int ntraj = ham.children.size();

  MATRIX &q = *dyn_var.q;
  MATRIX &p = *dyn_var.p;
  MATRIX &iM = *dyn_var.iM;

  // isNBRA carries ONE child Hamiltonian shared by all trajectories - only
  // parameter combinations that never demand per-trajectory children can
  // run: force_method 0, nac_update_method 0 or 2, decoherence rates from
  // given/EDC/schwartz_2/Gu_Franco. Every combination rejected here dies
  // deeper in the engine anyway, with a dimension error that never names
  // isNBRA - this funnel names the convention instead
  if (prms.isNBRA == 1 && dyn_var.ntraj > 1 && ham.children.size() == 1) {
    if (prms.nac_update_method == 1) {
      cout << "Error in update_Hamiltonian_variables(): nac_update_method = "
              "1 rebuilds couplings from per-trajectory momenta, but isNBRA "
              "keeps a single child Hamiltonian with one NAC slot. Use "
              "nac_update_method = 0 (consume supplied couplings, the NBRA "
              "convention) or 2 (HST/NPI from the shared time-overlap), or "
              "run with isNBRA = 0 for per-trajectory children\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }
    // the transform-4 frame rotates the shared child's ham_dia by ONE
    // frame - a tracking algorithm whose projector reads per-trajectory
    // data (4: the momenta) diverges the trajectories' frames
    if (prms.ham_transform_method == 4 && prms.state_tracking_algo == 4) {
      cout << "Error in update_Hamiltonian_variables(): state_tracking_algo = 4 "
              "builds per-trajectory projectors from the momenta, but isNBRA "
              "with ham_transform_method = 4 rotates the single shared child "
              "Hamiltonian by one frame. Use a tracking algorithm that reads "
              "the shared Hamiltonian alone (-1, 0, 1, 2, 21, 22, 5, 6), or "
              "run with isNBRA = 0 for per-trajectory children\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }
    // the shared-child peninsula: state-enforced, rescaling-free live
    // forces are trajectory-invariant, so one force column serves all
    // trajectories (classical-path on-the-fly NBRA - the cp2k driver's
    // force_source "gs" pairs with it)
    int nbra_live_forces = (prms.force_method == 1 &&
                            prms.enforce_state_following == 1 &&
                            prms.momenta_rescaling_algo == 0 &&
                            prms.ensemble == 0);
    if (prms.force_method != 0 && !nbra_live_forces) {
      cout << "Error in update_Hamiltonian_variables(): force_method = "
           << prms.force_method
           << " computes per-trajectory forces, but isNBRA keeps a single "
              "child Hamiltonian. Use force_method = 0 (precomputed "
              "trajectory, the file-based NBRA convention), the "
              "state-enforced live-force combination force_method = 1 + "
              "enforce_state_following = 1 + momenta_rescaling_algo = 0 + "
              "ensemble = 0 (classical-path on-the-fly NBRA), or run with "
              "isNBRA = 0 for per-trajectory children\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }
    // the collapses that edit the momenta per trajectory (MFSD and the
    // exact-factorization family) and the XF forces split the trajectories
    // off the shared path - the divergence sentinel below reads q alone and
    // would see it a step late, after a wasted shared CP2K step
    if (prms.force_method != 0 &&
        (prms.decoherence_algo == 4 || prms.decoherence_algo == 5 ||
         prms.decoherence_algo == 6 || prms.use_xf_force == 1)) {
      cout << "Error in update_Hamiltonian_variables(): decoherence_algo = "
           << prms.decoherence_algo << " / use_xf_force = " << prms.use_xf_force
           << " edits the momenta per trajectory, but isNBRA with live forces "
              "keeps every trajectory on the shared child Hamiltonian's path. "
              "Use decoherence_algo -1, 0, 1, 2 or 3 and use_xf_force = 0 "
              "there, or run with isNBRA = 0 for per-trajectory children\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }
    if (prms.decoherence_times_type == 2 ||
        prms.decoherence_times_type == 4) {
      cout << "Error in update_Hamiltonian_variables(): "
              "decoherence_times_type = " << prms.decoherence_times_type
           << " (Schwartz) builds per-trajectory mean-field forces, but "
              "isNBRA keeps a single child Hamiltonian. Use "
              "decoherence_times_type = 1 (energy-based EDC, the NBRA "
              "convention) or 0 (given rates), or run with isNBRA = 0\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }
    // the shared child evaluates the Hamiltonian at trajectory 0's
    // geometry - live-force runs stay valid only while every trajectory
    // rides the same nuclear path (identical initial conditions, no
    // per-trajectory momentum edits) - any spread across the q columns
    // means the shared Hamiltonian is wrong for the divergent
    // trajectories
    if (prms.force_method != 0) {
      double q_spread = 0.0;
      for (int idof = 0; idof < dyn_var.ndof; idof++) {
        double q0 = dyn_var.q->get(idof, 0);
        for (int itraj = 1; itraj < dyn_var.ntraj; itraj++) {
          double dq = fabs(dyn_var.q->get(idof, itraj) - q0);
          if (dq > q_spread) { q_spread = dq; }
        }
      }
      if (q_spread > 1e-8) {
        cout << "Error in update_Hamiltonian_variables(): the trajectories "
                "diverged under the shared child Hamiltonian (max coordinate "
                "spread " << q_spread << ") - isNBRA = 1 with live forces "
                "requires identical initial conditions (no q/p sampling) and "
                "trajectory-invariant propagation. Sample initial conditions "
                "with isNBRA = 0\n";
        std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
      }
    }
  }

  if (update_type == 0) {
    //============================= Energies =========================
    // How to compute electronic Hamiltonian - this may already read a lot of
    // other variables, such as time-overlaps, NAC, and Hvib
    if (prms.ham_update_method == 0) {
      ;
      ;
    } else if (prms.ham_update_method == 1) {
      //      cout<<" "<<q.n_cols<<"  "<<q.n_rows<<endl;
      //      cout<<" "<<iM.n_cols<<"  "<<iM.n_rows<<endl;
      //      cout<<" "<<p.n_cols<<"  "<<p.n_rows<<endl;
      // exit(0);
      ham.compute_diabatic(py_funct, q, model_params, 1);
      //      exit(0);
    } else if (prms.ham_update_method == 2) {
      ham.compute_adiabatic(py_funct, q, model_params, 1);
    }
    else {
      std::cout<<"Error in update_Hamiltonian_variables(): ham_update_method = "<<prms.ham_update_method<<" has no branch here (implemented: 0, 1, 2)\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }
    check_model_handover(prms, ham);
    //    exit(0);
    // Do the additional transformation between any reps, if needed
    if (prms.ham_transform_method == 0) {
      ;
      ;
    } else if (prms.ham_transform_method == 1) {
      // an adiabatic-native model (ham_update_method == 2) never populates
      // ham_dia - diagonalizing that zero matrix would overwrite the ingested
      // energies and basis_transform, so there is nothing to transform from
      if (prms.ham_update_method != 2) {
        ham.compute_adiabatic(1, 1); // do internal diagonalization
      }
    } else if (prms.ham_transform_method == 2) {
      // diabatic -> adiabatic by the MODEL-SUPPLIED transformation: the
      // model's own gauge replaces the internal diagonalization (no
      // LAPACK branch switches) - values + the shared Hellmann-Feynman
      // derivative tier
      ham.transform_dia2adi(1, 1);
    } else if (prms.ham_transform_method == 3) {
      // adiabatic -> diabatic by the stored basis transformation:
      // ham_dia = S U ham_adi U^H S (transform_adi2dia) - runs before the
      // time-overlap/NAC/Hvib updates so every diabatic consumer of this
      // step sees the rotated Hamiltonian - the engine-side alignment of the
      // supplied basis (dia_tracking_algo 5) comes first, so the rotated basis
      // is the aligned one
      if (prms.dia_tracking_algo == 5) { align_dia_groups(prms, ham, ham_prev); }
      ham.transform_adi2dia(1);
    } else if (prms.ham_transform_method == 4) {
      // the adi->dia local-diabatization rotation itself lives in the
      // type-1 hook below (update_proj_adi refreshes the frame between
      // the type-0 and the first type-1 call) - ham_adi was just
      // harvested - mark the frame rotation stale
      dyn_var.ld_frame_stale = 1;
    }
    else {
      std::cout<<"Error in update_Hamiltonian_variables(): ham_transform_method = "<<prms.ham_transform_method<<" has no branch here (implemented: 0, 1, 2, 3, 4)\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }

    // Internal diagonalization always sorts - only a supplied or externally
    // transformed adiabatic ladder can arrive unsorted, and whether it does
    // is a property of the model's output at this geometry, so this is
    // detectable only at runtime. Two origins look the same here. Labels that
    // follow the states' identity through their crossings (NBRA files of a
    // tracked or diabatized step 3): E_j - E_i of the labelled states is then
    // the true gap and every consumer stays right - the hop acceptance and
    // the momentum rescaling read the energies of the old and the new state,
    // the Landau-Zener target filter takes adjacency from the energies. Or a
    // linear-response excited-state method whose reference breaks down: a
    // negative excitation energy puts the excited state below its reference,
    // and a relaxation across that pair is bookkept as uphill. Checked before
    // the entanglement options, which add their own terms to the diagonal
    {
      static int warned_unsorted_adi = 0;
      int nchildren = ham.children.size();
      for (int itraj = 0; itraj < nchildren && !warned_unsorted_adi; itraj++) {
        nHamiltonian *ch = ham.children[itraj];
        if (ch->ham_adi_mem_status == 0) { continue; }
        for (int i = 0; i + 1 < ch->nadi; i++) {
          double gap = ch->ham_adi->get(i + 1, i + 1).real() -
                       ch->ham_adi->get(i, i).real();
          if (gap < -1e-10) {
            cout << "WARNING in update_Hamiltonian_variables(): the adiabatic "
                 << "energies are not in ascending order - state " << i + 1
                 << " lies " << -gap << " Ha (" << -gap * evolt
                 << " eV) BELOW state " << i << " for trajectory " << itraj
                 << ". Two origins: (1) supplied states labelled by identity "
                 << "through their crossings (NBRA files of a tracked or "
                 << "diabatized step 3) - expected then and harmless: the hop "
                 << "acceptance, the momentum rescaling and the Landau-Zener "
                 << "target filter read the energies of the labelled states, "
                 << "so E_j - E_i is the true gap. (2) a linear-response "
                 << "excited-state method whose reference breaks down - a "
                 << "negative excitation energy inverts the two lowest states "
                 << "near an S1/S0 conical intersection, a relaxation across "
                 << "this pair is then bookkept as uphill and the rescaling "
                 << "drains kinetic energy instead of releasing it - check the "
                 << "excitation energies the driver reports at this geometry. "
                 << "Warned once per process\n";
            warned_unsorted_adi = 1;
            break;
          }
        } // for i
      } // for itraj
    }

    // Also add entanglement options - this is done in the adiabatic rep, so
    // we'd need to compute all the adiabatic energies first
    if (prms.entanglement_opt == 0) { /* Nothing to do */
    } else if (prms.entanglement_opt == 1) {
      ham.add_ethd_adi(q, iM, 1);
    } else if (prms.entanglement_opt == 2) {
      ham.add_ethd3_adi(q, iM, prms.ETHD3_alpha, 1);
    } else if (prms.entanglement_opt == 22) {
      ham.add_ethd3_adi(q, p, iM, prms.ETHD3_alpha, prms.ETHD3_beta, 1);
    } else {
      cout << "ERROR in update_Hamiltonian_q_ethd: The entanglement option = "
           << prms.entanglement_opt << " is not avaialable\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }

    //============================== Time-overlaps ======================
    // the adiabatic time overlaps through the diabatic metric, St = U_prev^H S U - one owner for the
    // step loop and the python export (the former in-loop copy left S out, a non-identity metric gave a
    // non-unitary St: scaled HST couplings, NaN in the NPI arccos)
    update_time_overlaps(prms, dyn_var, ham, ham_prev);

    // Copy the transformation matrices to the dynamics variable
    dyn_var.update_basis_transform(ham);

  } // update_type == 0

  // exit(0);

  if (update_type == 0 || update_type == 1) {

    // Local-diabatization frame (ham_transform_method == 4): the
    // t0-anchored transported basis V(t_n) = T_n V(t_n-1), ham_dia =
    // V^H ham_adi V. Runs on the type-1 passes - update_proj_adi sits
    // between the type-0 and the first type-1 call, so T_n and hence V
    // are fresh here (the type-0 dispatch would see the previous
    // step's frame) - the Hvib update below consumes the rotated ham_dia
    if (update_type == 1 && prms.ham_transform_method == 4 &&
        dyn_var.ld_frame_stale) {
      // the frame accumulation lives in update_proj_adi (exactly once
      // per step) - the loop makes several type-1 calls per step
      // (post-half-step, post-rescaling) - the stale flag limits the
      // rotation to the passes where V or ham_adi actually changed
      int nchildren = ham.children.size();
      // one shared child (isNBRA) takes trajectory 0's frame - every
      // trajectory's frame must be that one, or the amplitudes of the
      // others are converted with a frame their Hamiltonian is not in
      if (nchildren == 1 && dyn_var.ntraj > 1) {
        for (int itraj = 1; itraj < dyn_var.ntraj; itraj++) {
          double spread = abs((*dyn_var.ld_frame[itraj] - *dyn_var.ld_frame[0]).max_elt());
          if (spread > 1e-10) {
            cout << "Error in the local-diabatization transform: the frame of trajectory "
                 << itraj << " differs from trajectory 0's (max |dV| = " << spread
                 << ") while isNBRA keeps a single shared child Hamiltonian, rotated by "
                    "one frame - the projectors must be trajectory-invariant under isNBRA = 1\n";
            std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
          }
        }
      }
      for (int traj = 0; traj < nchildren; traj++) {
        nHamiltonian *ch = ham.children[traj];
        if (ch->ham_adi_mem_status == 0 || ch->ham_dia_mem_status == 0) {
          cout << "Error in the local-diabatization transform: ham_adi or ham_dia \
  is not allocated for trajectory " << traj << "\n";
          std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
        }
        // the frame OWNS the diabatic slot: a model-supplied non-identity
        // basis_transform or a ham_dia would define a SECOND diabatic
        // basis (model diabats vs the trajectory frame) and the
        // amplitude machinery can serve only one - the identity the
        // on-the-fly drivers supply is the t0 frame itself and passes
        if (ch->basis_transform_supplied) {
          CMATRIX U_id(ch->get_basis_transform());
          U_id.identity();
          if (abs((ch->get_basis_transform() - U_id).max_elt()) > 1e-12) {
            cout << "Error in the local-diabatization transform: the model supplies a \
  non-identity obj.basis_transform, but ham_transform_method = 4 defines the diabatic \
  basis as the trajectory frame - the two conventions cannot coexist. Supply the \
  identity (the t0 frame) or no basis_transform, or use ham_transform_method = 3 for \
  a model-defined diabatic basis\n";
            std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
          }
        }
        if (ch->ham_dia_supplied) {
          cout << "Error in the local-diabatization transform: the model supplies \
  obj.ham_dia, which the frame rotation would overwrite every step - the two are \
  different diabatic bases. Drop obj.ham_dia, or run without ham_transform_method = 4\n";
          std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
        }
        // the same ownership for the vibronic slot: a supplied obj.hvib_dia
        // is read in the model's diabatic basis (hvib_update_method 0
        // consumes it as is, 1 validates the rebuild against it) while
        // ham_dia here is the frame rotation - under this transform the
        // engine builds hvib_dia from the rotated ham_dia itself
        if (ch->hvib_dia_supplied) {
          cout << "Error in the local-diabatization transform: the model supplies \
  obj.hvib_dia, but ham_transform_method = 4 defines the diabatic basis as the \
  trajectory frame and builds hvib_dia from the rotated ham_dia - the supplied one is \
  in another basis. Drop obj.hvib_dia, or use ham_transform_method = 3 for a \
  model-defined diabatic basis\n";
          std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
        }
        // chained in-place product - CBLAS zgemm-backed
        ch->ham_dia->product(adjoint(*dyn_var.ld_frame[traj]),
                             *ch->ham_adi, *dyn_var.ld_frame[traj]);
        // the frame is the engine's diabatic basis: c_dia = V^H c_adi
        // means basis_transform = V^H on BOTH sides - the child (the
        // init-time conversions) and dyn_var (the in-loop amplitude
        // and density readouts, copied before this hook runs)
        *ch->basis_transform = (*dyn_var.ld_frame[traj]).H();
      }
      for (int traj = 0; traj < dyn_var.ntraj; traj++) {
        *dyn_var.basis_transform[traj] =
            (*dyn_var.ld_frame[traj]).H();
      }
      dyn_var.ld_frame_stale = 0;
    }

    // Derivative NAC correction option:
    if (prms.do_nac_phase_correction ==
        2) { // Experimental option to fix the phase of NACVs

      vector<int> traj_id(1, 0);
      for (int traj = 0; traj < ntraj; traj++) {
        traj_id[0] = traj;
        CMATRIX Eadi(ham.children[traj]->get_ham_adi());
        MATRIX e_curr(ham.children[traj]->get_ham_adi().real());
        MATRIX e_prev(ham_prev.children[traj]->get_ham_adi().real());
        CMATRIX f_curr = ham.children[traj]->all_forces_adi(traj_id);
        CMATRIX f_prev = ham_prev.children[traj]->all_forces_adi(traj_id);
        // ham->get_hvib_adi().show_matrix();
        MATRIX pp(dyn_var.ndof, 1);
        double dt = prms.dt;
        pp = p.col(traj);
        int act_state = dyn_var.act_states[traj];

        // vector<MATRIX> T_new(compute_F_cost_matrix_dof_resolved(f_curr,
        // f_prev, e_curr, e_prev, pp, iM, dt, act_state));
        MATRIX T_new(compute_F_cost_matrix(f_curr, f_prev, e_curr, e_prev, pp,
                                           iM, dt, act_state)
                         .real());

        for (int k = 0; k < ndof; k++) {
          if (ham.children[traj]->dc1_adi_mem_status[k] == 0) {
            cout << "Error in update_Hamiltonian_variables(): the derivative "
                    "coupling w.r.t. the nuclear DOF " << k << " is not "
                    "allocated for trajectory " << traj << " but is needed "
                    "for its sign correction\n";
            std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
          }
          for (int i = 0; i < nadi; i++) {
            for (int j = i + 1; j < nadi; j++) {
              // double sgn = T_new[k].get(i,j);
              double sgn = T_new.get(i, j);
              if (sgn > 0.5) {
                sgn = -1;
              } // change sign
              else {
                sgn = 1.0;
              }

              ham.children[traj]->dc1_adi[k]->scale(i, j, sgn);
              ham.children[traj]->dc1_adi[k]->scale(j, i, sgn);
            } // for j
          } // for i

        } // for k
      } // for traj
    } // if correction

    //========================== Couplings ===============================
    // Don't update NACs - they may have been read in step 1
    if (prms.nac_update_method == 0) {
      ;
      ;
    }

    // Compute NACs explicitly
    else if (prms.nac_update_method == 1) {

      // For the purpose of updating the NACs and Hvibs for just the quantum
      // DOFs, we'll reset the momenta for all other DOFs to zero, to
      // effectively turn of the effect of classical momenta on the NAC
      // calculations (in case those derivative couplings have been computed)
      int ndof = dyn_var.ndof;
      int ntraj = dyn_var.ntraj;

      MATRIX p_quantum_dof(ndof, ntraj);
      for (auto dof : prms.quantum_dofs) {
        for (int itraj = 0; itraj < ntraj; itraj++) {
          p_quantum_dof.set(dof, itraj, p.get(dof, itraj));
        }
      }

      // an adiabatic-native model (ham_update_method == 2) never populates
      // the diabatic derivative couplings - there is nothing to build the
      // diabatic NACs from
      if (prms.ham_update_method != 2) {
        ham.compute_nac_dia(p_quantum_dof, iM, 0, 1);
      }
      ham.compute_nac_adi(p_quantum_dof, iM, 0, 1);
    }

    // Compute NAC from the time-overlaps
    else if (prms.nac_update_method == 2 && update_type == 0) {

      // Don't update NACs and time-overlaps in response to p
      // if they are computed via the time-overlaps - only in response to
      // change of q

      double dt = prms.dt;
      // int nst = ham.nadi;
      // int ntraj = ham.children.size();
      CMATRIX st(nadi, nadi);
      MATRIX st_re(nadi, nadi);
      MATRIX st_im(nadi, nadi);

      CMATRIX nac(nadi, nadi);
      MATRIX nac_re(nadi, nadi);
      MATRIX nac_im(nadi, nadi);

      for (int traj = 0; traj < ntraj; traj++) {
        // flag-only, no compare: the HST/NPI derivatives legitimately
        // differ from supplied NACs at O(dt), so no consistent-redundancy
        // case exists - the combination itself is the contradiction
        if (ham.children[traj]->nac_adi_supplied) {
          cout << "Error in the NAC update: the model supplies obj.nac_adi, but "
                  "nac_update_method = 2 replaces it with time-overlap (HST/NPI) "
                  "derivatives - the supplied couplings would be discarded. Use "
                  "nac_update_method = 0 (consume the supplied NACs), or stop "
                  "supplying nac_adi\n";
          std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
        }
        st = ham.children[traj]->get_time_overlap_adi();
        // a per-state phase flip of the new eigenvectors makes St the
        // product of the physical time-overlap with a diagonal sign
        // matrix, whose antisymmetric part loses the flipped pair's
        // coupling for this one step (a reflection has no antisymmetric
        // part). On the relabel-projector routes the amplitudes are
        // re-signed by exactly those diagonal signs, so undo the flips
        // before differentiating and fold the result back into that
        // gauge. Not on the routes that keep the raw gauge: the St^-1
        // rotation projectors carry the flip themselves, T = I keeps the
        // raw amplitudes, and do_nac_phase_correction re-signs the NAC
        // downstream - a fold there double-corrects
        // rotation projectors (St^-1) carry the flip in T itself - the raw
        // St is what they must see - every other route wants the flip
        // undone in St (else the pair's coupling is zero for this step),
        // and only the relabel routes need it folded back into the
        // re-signed amplitude gauge afterwards
        int rotation_proj = prms.rotation_projector();
        int relabel_gauge = (prms.assume_always_consistent == 0 && !rotation_proj);
        // the relabel routes swap the labels of a pair that crosses within the step
        // (an unresolved crossing) and move the amplitudes along. The raw St of a
        // swapped pair turns by nearly pi/2: its antisymmetric part is 1/dt or 0 by
        // the arbitrary sign of a fresh diagonalization, and the amplitudes would be
        // turned twice or not at all. The coupling is taken between the FOLLOWED
        // states - the new states ordered by the projector's own permutation - and
        // goes back under the raw labels of the new states at the end. The pairing is
        // the projector's own: the time overlap under 1/2/21/22, the force-cost matrix
        // of the two frames under the force-based tracking 4 (as update_proj_adi)
        CMATRIX perm(nadi, nadi);
        perm.identity();
        if (relabel_gauge && (prms.state_tracking_algo == 1 || prms.state_tracking_algo == 2 ||
                              prms.state_tracking_algo == 21 || prms.state_tracking_algo == 22 ||
                              prms.state_tracking_algo == 4)) {
          CMATRIX Eadi(ham.children[traj]->get_ham_adi());
          CMATRIX pairing(st);
          if (prms.state_tracking_algo == 4) {
            vector<int> traj_id(1, traj);
            MATRIX e_curr(ham.children[traj]->get_ham_adi().real());
            MATRIX e_prev(ham_prev.children[traj]->get_ham_adi().real());
            CMATRIX f_curr(ham.children[traj]->all_forces_adi(traj_id));
            CMATRIX f_prev(ham_prev.children[traj]->all_forces_adi(traj_id));
            MATRIX iM(dyn_var.get_imass());
            MATRIX momenta(dyn_var.get_momenta());
            MATRIX p(momenta.col(traj));
            pairing = compute_F_cost_matrix(f_curr, f_prev, e_curr, e_prev, p, iM, prms.dt,
                                            dyn_var.act_states[traj]);
          }
          CMATRIX T_follow(compute_projector(prms, Eadi, pairing));
          for (int i = 0; i < nadi; i++) {
            for (int j = 0; j < nadi; j++) {
              perm.set(i, j, (std::abs(T_follow.get(i, j)) > 0.5) ? complex<double>(1.0, 0.0)
                                                                   : complex<double>(0.0, 0.0));
            }
          }
          st = st * perm;
        } else if (!relabel_gauge || rotation_proj || prms.state_tracking_algo == 0) {
          ; // the raw gauge, the rotation projectors (-1, 5, 6 and their label twins) and no reordering (0) take the new states as
            // stored
        } else {
          std::cout<<"Error in update_Hamiltonian_variables(): state_tracking_algo = "<<prms.state_tracking_algo<<" has no branch here (implemented: -12, -1, 0, 1, 2, 4, 5, 6, 21, 22, 52, 62)\n";
          std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
        }
        vector<double> flip(nadi, 1.0);
        if (!rotation_proj) {
          for (int i = 0; i < nadi; i++) {
            if (st.get(i, i).real() < 0.0) {
              flip[i] = -1.0;
              for (int k = 0; k < nadi; k++) { st.scale(k, i, -1.0); }
            }
          }
        }

        if (prms.nac_algo == 0) {
          nac = (0.5 / dt) * (st - st.H());
        } else if (prms.nac_algo == 1) {
          st_re = st.real();
          nac_re = nac_npi(st_re, dt);
          nac = CMATRIX(nac_re, nac_im);
        }
        else {
          std::cout<<"Error in update_Hamiltonian_variables(): nac_algo = "<<prms.nac_algo<<" has no branch here (implemented: 0, 1)\n";
          std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
        }
        // back into the re-signed amplitude gauge on the relabel routes:
        // nac -> F nac F with F = diag(flip), then under the raw labels of the new
        // states: nac -> perm nac perm^H
        if (relabel_gauge) {
          for (int i = 0; i < nadi; i++) {
            for (int j = 0; j < nadi; j++) {
              if (flip[i] * flip[j] < 0.0) { nac.scale(i, j, -1.0); }
            }
          }
          nac = perm * nac * perm.H();
        }

        ham.children[traj]->set_nac_adi_by_val(nac);

      } // for traj

    } // for nac_update_method == 2
    else if (prms.nac_update_method == 2) { ; } // update_type != 0: the time-overlap NACs are built on the type-0 pass only
    else {
      std::cout<<"Error in update_Hamiltonian_variables(): nac_update_method = "<<prms.nac_update_method<<" has no branch here (implemented: 0, 1, 2)\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }

    if (prms.do_nac_phase_correction ==
        1) { // Experimental option to fix the phase of NACs:
      for (int traj = 0; traj < ntraj; traj++) {
        for (int i = 0; i < nadi; i++) {
          for (int j = i + 1; j < nadi; j++) {
            double x1 = ham.children[traj]->nac_adi->get(i, j).real();
            double x2 = ham_prev.children[traj]->nac_adi->get(i, j).real();
            // flip only on definite sign disagreement: SIGN(0) = 0
            // would scale a first-step or zero-crossing NAC to zero
            // permanently (an HST NAC starts from St = I, so zero)
            if (SIGN(x1) * SIGN(x2) < 0.0) {
              ham.children[traj]->nac_adi->scale(i, j, -1.0);
              ham.children[traj]->nac_adi->scale(j, i, -1.0);
              // the flip invalidates the supplied-hvib comparison
              ham.children[traj]->hvib_adi_supplied = 0;
            }
          } // for j
        } // for i
      } // for traj
    } // if correction

    //========================== Vibronic Hamiltonian
    //===============================
    // a diabatic vibronic supply has consumers only under the adi->dia
    // transforms (consumed as-is at hvib_update_method 0, validated against
    // the rebuild at 1) - anywhere else on the adiabatic-native route it
    // would sit ingested and unread
    if (prms.ham_update_method == 2 && prms.ham_transform_method != 3 &&
        prms.ham_transform_method != 4) {
      int ntraj_h = ham.children.size();
      for (int traj = 0; traj < ntraj_h; traj++) {
        if (ham.children[traj]->hvib_dia_supplied) {
          cout << "Error in update_Hamiltonian_variables(): the model supplies "
                  "obj.hvib_dia, but with ham_update_method = 2 and no adi->dia "
                  "transform (ham_transform_method 3 or 4) nothing consumes it - "
                  "the supplied vibronic Hamiltonian would be silently unused. "
                  "Use ham_transform_method = 3, or drop the attribute\n";
          std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
        }
      }
    }
    // Don't update Hvib - perhaps because they are read from files in step 1
    if (prms.hvib_update_method == 0) {
      // the supplied/precomputed data covers the ADI side only - under
      // transform 3 the diabatic picture (H_SD readout, rep-0
      // propagation) needs hvib_dia built from the rotated ham_dia -
      // unless the model supplied obj.hvib_dia itself (e.g. the
      // frame-corrected H_SD - i D_SD), which is consumed as-is
      if (prms.ham_transform_method == 3 || prms.ham_transform_method == 4) {
        int ntraj_h = ham.children.size();
        for (int traj = 0; traj < ntraj_h; traj++) {
          if (!ham.children[traj]->hvib_dia_supplied) {
            ham.children[traj]->compute_hvib_dia();
          }
        }
      }
    }

    // Explicitly update Hvib
    else if (prms.hvib_update_method == 1) {
      // the diabatic Hvib of an adiabatic-native model would be built from
      // the never-populated ham_dia/nac_dia - all-zero dead output -
      // unless transform 3 rotates ham_adi into ham_dia (nac_dia stays
      // zero: hvib_dia = ham_dia is the strictly-diabatic contract)
      if (prms.ham_update_method != 2 || prms.ham_transform_method == 3 ||
          prms.ham_transform_method == 4) {
        ham.compute_hvib_dia(1);
      }
      ham.compute_hvib_adi(1);
    }
    else {
      std::cout<<"Error in update_Hamiltonian_variables(): hvib_update_method = "<<prms.hvib_update_method<<" has no branch here (implemented: 0, 1)\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }

  } // update_type==0 || update_type==1

  if (update_type == 2) { // in response to updated electronic variables

  } // update_type==2

  // Now apply the projectors to reflect any state switches that may have
  // occurred in the meantime

  // cout<<"In update_Hamiltonian_variables... St = \n";
  // ham.children[0]->get_time_overlap_adi().show_matrix();
}

void update_Hamiltonian_variables(bp::dict prms, dyn_variables &dyn_var,
                                  nHamiltonian &ham, nHamiltonian &ham_prev,
                                  bp::object py_funct, bp::object model_params,
                                  int update_type) {

  dyn_control_params _prms;
  _prms.set_parameters(prms);

  update_Hamiltonian_variables(_prms, dyn_var, ham, ham_prev, py_funct,
                               model_params, update_type);
}

void update_time_overlaps(dyn_control_params &prms, dyn_variables &dyn_var,
                          nHamiltonian &ham, nHamiltonian &ham_prev) {

  int nadi = ham.nadi;
  int ntraj = ham.children.size();

  /// Don't update time-overlaps
  if (prms.time_overlap_method == 0) {
    ;
    ;
  } // maybe it is already pre-computed and stored

  /// Compute the time-overlap directly, using previous MO vectors
  else if (prms.time_overlap_method == 1) { // Explicitly compute it:

    CMATRIX st(nadi, nadi);
    for (int traj = 0; traj < ntraj; traj++) {
      // the adiabatic bra-ket runs through the diabatic metric: a static
      // diabatic basis has <dia(t)|dia(t+dt)> = ovlp_dia (identity for
      // orthonormal diabats) - time-dependent diabats supply their own
      // overlaps through time_overlap_method 0
      st = ham_prev.children[traj]->get_basis_transform().H() *
           ham.children[traj]->get_ovlp_dia() *
           ham.children[traj]->get_basis_transform();
      // a model that supplies its own time-overlaps must not have them
      // silently overwritten by a DISAGREEING recomputation (identity
      // basis transforms make st = I and freeze the electronic dynamics) -
      // at the init call ham_prev is ham itself: no interval exists yet, the
      // recomputed identity is no disagreement with the supplied interval overlap
      if (&ham_prev != &ham && ham.children[traj]->time_overlap_adi_supplied &&
          abs((st - ham.children[traj]->get_time_overlap_adi()).max_elt()) > 1e-8) {
        cout << "Error in update_time_overlaps(): the model supplies "
                "obj.time_overlap_adi, but time_overlap_method = 1 recomputes the "
                "time-overlap from the basis transforms and the two disagree - the "
                "supplied data would be overwritten. Use time_overlap_method = 0 "
                "(consume the supplied time-overlaps), or supply basis transforms "
                "consistent with them\n";
        std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
      }
      ham.children[traj]->set_time_overlap_adi_by_val(st);
    }
  } // 1
  else {
    std::cout<<"Error in update_time_overlaps(): time_overlap_method = "<<prms.time_overlap_method<<" has no branch here (implemented: 0, 1)\n";
    std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
  }
}

void update_proj_adi(dyn_control_params &prms, dyn_variables &dyn_var,
                     nHamiltonian &Ham, nHamiltonian &Ham_prev) {
  /**
    Just re-compute the proj_adi matrices
  */

  //======= Parameters of the dyn variables ==========
  int ntraj = dyn_var.ntraj;

  MATRIX iM(dyn_var.get_imass());
  MATRIX momenta(dyn_var.get_momenta());
  double dt = prms.dt;

  // NBRA runs (isNBRA==1) carry a single child Hamiltonian shared by all
  // trajectories - never index children beyond what actually exists
  int nchildren = Ham.children.size();
  // an eigenvector branch switch (negative time-overlap diagonal) with the
  // phase correction opted out leaves the sign flip uncorrected in every
  // adiabatic-representation consumer - the run would complete with
  // silently wrong populations - whether a flip occurs is a property of the
  // data, so this is detectable only at runtime - say so once. The
  // opt-out stays legitimate for flip-free sources (driver-side sign
  // alignment, a model-supplied smooth transformation, the rotation
  // projectors, which fold the flip into T)
  if (prms.do_phase_correction == 0 && !prms.rotation_projector()) {
    static int warned_flip_nopc = 0;
    if (!warned_flip_nopc) {
      for (int itraj = 0; itraj < ntraj && !warned_flip_nopc; itraj++) {
        nHamiltonian *ch = Ham.children[(itraj < nchildren) ? itraj : 0];
        if (ch->time_overlap_adi_mem_status != 0) {
          for (int i = 0; i < ch->nadi; i++) {
            if (ch->time_overlap_adi->get(i, i).real() < 0.0) {
              cout << "WARNING in update_proj_adi(): the adiabatic time-overlap \
  diagonal went negative (an eigenvector branch switch) while \
  do_phase_correction = 0 - the relabel projector will not fold the sign \
  flip into the amplitudes, and adiabatic-representation consumers \
  (Hvib-family integrators, hop proposals) will mishandle this crossing. \
  Use do_phase_correction = 1, or supply a smooth gauge (driver-side sign \
  alignment, or a model basis_transform with ham_transform_method = 2)\n";
              warned_flip_nopc = 1;
              break;
            }
          } // for i
        }
      } // for itraj
    }
  }
  // Deterministic per-trajectory projector updates - parallel-safe (each
  // iteration writes only proj_adi[itraj]) - identical for any thread count
  // exceptions cannot leave an OMP-outlined region (even a serialized
  // one) - capture and rethrow outside so the guards stay translatable
  // the label twins of the rotation projectors take the assignment's permutation for the active state's label
  dyn_var.label_assignment = (prms.state_tracking_algo == -12 || prms.state_tracking_algo == 52 ||
                              prms.state_tracking_algo == 62) ? 1 : 0;
  std::exception_ptr omp_eptr = nullptr;
  #pragma omp parallel for num_threads(prms.omp_num_threads) \
      if (prms.omp_num_threads > 1)
  for (int itraj = 0; itraj < ntraj; itraj++) {
    try {
    vector<int> traj_id(1, 0);
    CMATRIX f_prev(dyn_var.nadi, dyn_var.ndof);
    CMATRIX f_curr(dyn_var.nadi, dyn_var.ndof);
    MATRIX p(dyn_var.ndof, 1);
    int traj1 = itraj;
    if (traj1 >= nchildren) { traj1 = 0; }
    traj_id[0] = traj1;

    nHamiltonian *ham = Ham.children[traj1];
    nHamiltonian *ham_prev = Ham_prev.children[traj1];
    // the clamp serves the ham index only - momenta and the active
    // state are per-trajectory
    p = momenta.col(itraj);
    int act_state = dyn_var.act_states[itraj];

    //================= Basis re-expansion ===================
    CMATRIX P(ham->nadi, ham->nadi);
    CMATRIX T_new(*dyn_var.proj_adi[itraj]);

    P = ham->get_time_overlap_adi(); // U_old.H() * U

    if (prms.state_tracking_algo == -1 || prms.state_tracking_algo == -12) { // This is LD
      // More consistent with the new derivations:
      FullPivLU_inverse(P, T_new);

      if (fabs((P * T_new).tr().real() - P.n_cols) > 0.1) {
        cout << "Problem inverting time-overlap matrix\n";
        P.show_matrix("p_matrix.txt");
        std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
      }
      T_new = orthogonalized_T(T_new);
    } else if (prms.state_tracking_algo == 0 ||
               prms.state_tracking_algo == 1 || prms.state_tracking_algo == 2 ||
               prms.state_tracking_algo == 21 || prms.state_tracking_algo == 22) { // This is based on reordering + phase correction
      // algo 0 = no reordering: compute_projector keeps the identity
      // permutation and still applies the phase corrections
      // (do_phase_correction-gated) - without them the diabatic readout
      // Cdia = U C_adi jumps at eigenvector sign flips
      CMATRIX Eadi(ham->get_ham_adi());
      T_new = P;
      T_new = compute_projector(prms, Eadi, T_new);
    } else if (prms.state_tracking_algo ==
               4) { // new experimental approach - based on forces
      // exit(0);
      CMATRIX Eadi(ham->get_ham_adi());
      MATRIX e_curr(ham->get_ham_adi().real());
      MATRIX e_prev(ham_prev->get_ham_adi().real());
      f_curr = ham->all_forces_adi(traj_id);
      f_prev = ham_prev->all_forces_adi(traj_id);
      // ham->get_hvib_adi().show_matrix();
      T_new = compute_F_cost_matrix(f_curr, f_prev, e_curr, e_prev, p, iM, dt,
                                    act_state);
      // T_new.show_matrix();
      // exit(0);
      T_new = compute_projector(
          prms, Eadi, T_new); // CMATRIX compute_projector(dyn_control_params&
                              // prms, CMATRIX& Eadi, CMATRIX& St){
      //  T_new = orthogonalized_T( T_new );
    } else if (prms.state_tracking_algo == 5 || prms.state_tracking_algo == 52) { // This is SVD-based LD
      CMATRIX svd_u(dyn_var.nadi, dyn_var.nadi);
      CMATRIX svd_s(dyn_var.nadi, dyn_var.nadi);
      CMATRIX svd_v(dyn_var.nadi, dyn_var.nadi);

      BDCSVD_decomposition(P, svd_u, svd_s, svd_v);
      // the reprojection runs OPPOSITE to St (the -1 branch builds
      // orth(St^-1)): the polar factor of St needs the adjoint,
      // V U^H, not U V^H - self-inverse sign-flip scripts hide the
      // difference, genuine rotations do not
      T_new = svd_v * svd_u.H();
    }// 5 - SVD-based LD

    else if (prms.state_tracking_algo == 6 || prms.state_tracking_algo == 62) { // adaptive SVD-based LD
      CMATRIX svd_u(dyn_var.nadi, dyn_var.nadi);
      CMATRIX svd_s(dyn_var.nadi, dyn_var.nadi);
      CMATRIX svd_v(dyn_var.nadi, dyn_var.nadi);
      CMATRIX f(dyn_var.nadi, dyn_var.nadi);
      CMATRIX eye(dyn_var.nadi, dyn_var.nadi); eye.identity();
      CMATRIX T_tilde(dyn_var.nadi, dyn_var.nadi);

      BDCSVD_decomposition(P, svd_u, svd_s, svd_v);

      double eps = 1e-12;
      for(int i=0; i<dyn_var.nadi; i++){
        double sigma_i = svd_s.get(i,i).real();
        // an explicit complex value: a plain double binds to the
        // flat-index set(int, double re, double im) overload (int ->
        // double outranks double -> complex), writing complex(i,
        // sigma) into element [i] and leaving the diagonal zero
        f.set(i, i, complex<double>(sigma_i/(sigma_i + eps), 0.0));
      }// for i

      CMATRIX Proj(dyn_var.nadi, dyn_var.nadi);
      
      Proj = svd_v * f * svd_v.H();

      T_tilde = svd_u * f * svd_v.H() + (eye - Proj);

      // We are going to re-use the storage, but the meaning
      // is different
      BDCSVD_decomposition(T_tilde, svd_u, svd_s, svd_v);
      T_new = svd_u * svd_v.H();

      T_new = T_new.H();
   

    }// 6 adaptive SVD-based LD
    else {
      std::cout<<"Error in update_proj_adi(): state_tracking_algo = "<<prms.state_tracking_algo<<" has no branch here (implemented: -12, -1, 0, 1, 2, 4, 5, 6, 21, 22, 52, 62 - the stochastic 3, 32, 33 are refused by sanity_check)\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }

    // the label twins -12 / 52 / 62: the same transform, the labels by the Jonker-Volgenant assignment of |T|^2 -
    // T carries old state i into new state j with the weight |T(j, i)|^2, so the assignment reads T^H as the
    // time overlap <old i|new j> with the step's energies for the MK penalty, as the relabel projectors do. The
    // active state then follows the permutation (update_active_states) instead of the largest weight of its own
    // moved vector, which can send two states to one label at a near 50/50 mixing
    if (dyn_var.label_assignment) {
      CMATRIX Eadi(ham->get_ham_adi());
      CMATRIX TH(T_new.H());
      dyn_var.proj_perm[itraj] = jonker_volgenant(TH, Eadi, prms.MK_alpha, prms.MK_scaling_function);
    }
    *dyn_var.proj_adi[itraj] = T_new;
    } catch (...) {
      #pragma omp critical
      { if (!omp_eptr) { omp_eptr = std::current_exception(); } }
    }

  } // for ntraj
  if (omp_eptr) { std::rethrow_exception(omp_eptr); }

  // the local-diabatization frame accumulates the per-step projector
  // HERE - this function runs exactly once per nuclear step, while the
  // type-1 Hamiltonian update (which rotates ham_dia by the frame) can
  // run several times per step
  if (prms.ham_transform_method == 4) {
    for (int itraj = 0; itraj < ntraj; itraj++) {
      dyn_var.ld_frame[itraj]->product(*dyn_var.proj_adi[itraj],
                                       *dyn_var.ld_frame[itraj]);
    }
    dyn_var.ld_frame_stale = 1;
  }

} // reproject_basis

void update_proj_adi(dyn_control_params &prms, dyn_variables &dyn_var,
                     nHamiltonian &Ham) {
  update_proj_adi(prms, dyn_var, Ham, Ham);
}

} // namespace libdyn
} // namespace liblibra
