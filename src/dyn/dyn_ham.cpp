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
    transformation matrix
       - 3: adiabatic->diabatic according to internally stored basis
    transformation matrix
       - 4: adiabatic->diabatic according to local diabatization method
    The dia->adi options are skipped when ham_update_method == 2: an
    adiabatic-native model never populates ham_dia, so there is nothing
    to transform from.


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
      // TO DO: Add dia->adi transformation according to the basis_transform
      // matrix
    } else if (prms.ham_transform_method == 3) {
      // adiabatic -> diabatic by the stored basis transformation:
      // ham_dia = S U ham_adi U^H S (transform_adi2dia); runs before the
      // time-overlap/NAC/Hvib updates so every diabatic consumer of this
      // step sees the rotated Hamiltonian
      ham.transform_adi2dia(1);
    } else if (prms.ham_transform_method == 4) {
      // TO DO: Add adi->dia transformation according to local diabatization
      // method
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
    /// Don't update time-overlaps
    if (prms.time_overlap_method == 0) {
      ;
      ;
    } // maybe it is already pre-computed and stored

    /// Compute the time-overlap directly, using previous MO vectors
    else if (prms.time_overlap_method == 1) { // Explicitly compute it:

      CMATRIX st(nadi, nadi);
      for (int traj = 0; traj < ntraj; traj++) {
        st = ham_prev.children[traj]->get_basis_transform().H() *
             ham.children[traj]->get_basis_transform();
        // a model that supplies its own time-overlaps must not have them
        // silently overwritten by a DISAGREEING recomputation (identity
        // basis transforms make st = I and freeze the electronic dynamics)
        if (ham.children[traj]->time_overlap_adi_supplied &&
            abs((st - ham.children[traj]->get_time_overlap_adi()).max_elt()) > 1e-8) {
          cout << "Error in the time-overlap update: the model supplies "
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

    // Copy the transformation matrices to the dynamics variable
    dyn_var.update_basis_transform(ham);

  } // update_type == 0

  // exit(0);

  if (update_type == 0 || update_type == 1) {

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
        // St is what they must see; every other route wants the flip
        // undone in St (else the pair's coupling is zero for this step),
        // and only the relabel routes need it folded back into the
        // re-signed amplitude gauge afterwards
        int rotation_proj = (prms.state_tracking_algo == -1 ||
                             prms.state_tracking_algo == 5 ||
                             prms.state_tracking_algo == 6);
        int relabel_gauge = (prms.assume_always_consistent == 0 && !rotation_proj);
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
        // back into the re-signed amplitude gauge on the relabel routes:
        // nac -> F nac F with F = diag(flip)
        if (relabel_gauge) {
          for (int i = 0; i < nadi; i++) {
            for (int j = 0; j < nadi; j++) {
              if (flip[i] * flip[j] < 0.0) { nac.scale(i, j, -1.0); }
            }
          }
        }

        ham.children[traj]->set_nac_adi_by_val(nac);

      } // for traj

    } // for nac_update_method == 2

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
    // Don't update Hvib - perhaps because they are read from files in step 1
    if (prms.hvib_update_method == 0) {
      // the supplied/precomputed data covers the ADI side only; under
      // transform 3 the diabatic picture (H_SD readout, rep-0
      // propagation) needs hvib_dia built from the rotated ham_dia -
      // unless the model supplied obj.hvib_dia itself (e.g. the
      // frame-corrected H_SD - i D_SD), which is consumed as-is
      if (prms.ham_transform_method == 3) {
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
      if (prms.ham_update_method != 2 || prms.ham_transform_method == 3) {
        ham.compute_hvib_dia(1);
      }
      ham.compute_hvib_adi(1);
    }

  } // update_type==0 || update_type==1

  if (update_type == 2) { // in responce to updated electronic variables

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
      st = ham_prev.children[traj]->get_basis_transform().H() *
           ham.children[traj]->get_basis_transform();
      // a model that supplies its own time-overlaps must not have them
      // silently overwritten by a DISAGREEING recomputation (identity
      // basis transforms make st = I and freeze the electronic dynamics)
      if (ham.children[traj]->time_overlap_adi_supplied &&
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
  // Deterministic per-trajectory projector updates - parallel-safe (each
  // iteration writes only proj_adi[itraj]); identical for any thread count
  // exceptions cannot leave an OMP-outlined region (even a serialized
  // one) - capture and rethrow outside so the guards stay translatable
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

    if (prms.state_tracking_algo == -1) { // This is LD
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
               prms.state_tracking_algo == 21 ||
               prms.state_tracking_algo == 3 ||
               prms.state_tracking_algo == 32 ||
               prms.state_tracking_algo ==
                   33) { // This is based on reordering + phase correction
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
    } else if (prms.state_tracking_algo == 5) { // This is SVD-based LD
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

    else if (prms.state_tracking_algo == 6) { // adaptive SVD-based LD
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

    *dyn_var.proj_adi[itraj] = T_new;
    } catch (...) {
      #pragma omp critical
      { if (!omp_eptr) { omp_eptr = std::current_exception(); } }
    }

  } // for ntraj
  if (omp_eptr) { std::rethrow_exception(omp_eptr); }

} // reproject_basis

void update_proj_adi(dyn_control_params &prms, dyn_variables &dyn_var,
                     nHamiltonian &Ham) {
  update_proj_adi(prms, dyn_var, Ham, Ham);
}

} // namespace libdyn
} // namespace liblibra
