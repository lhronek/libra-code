/*********************************************************************************
 * Copyright (C) 2019-2023 Alexey V. Akimov
 *
 * This file is distributed under the terms of the GNU General Public License
 * as published by the Free Software Foundation, either version 3 of
 * the License, or (at your option) any later version.
 * See the file LICENSE in the root directory of this distribution
 * or <http://www.gnu.org/licenses/>.
 *
 *********************************************************************************/
/**
  \file dynamics.cpp
  \brief The file implements the general framework to run:
   * adiabatic dynamics (Verlet)
   * nonadiabatic Ehrenfest dynamics
   * nonadiabatic TSH dynamics
   * include thermostat, if needed
   * include decoherence
   * include quantum nuclear effect (ETHD)
   * include phase corrections/state tracking in NA-MD
   * the same framework for multiple trajectories
   * include coupled-trajectories methods (planned)
   * enable the NBRA-like calculations as well as non-NBRA
*/

#include <cmath>
#include "../calculators/NPI.h"
#include "../math_meigen/mEigen.h"
#include <exception>
#include <stdexcept>
#include <memory>
#include <deque>
#include "Dynamics.h"
#include "Energy_and_Forces.h"
#include "Surface_Hopping.h"
#include "dyn_control_params.h"
#include "dyn_ham.h"
#include "dyn_variables.h"
#include "electronic/libelectronic.h"

/// liblibra namespace
namespace liblibra {

using namespace libcalculators;

/// libdyn namespace
namespace libdyn {

namespace bp = boost::python;

void aux_get_transforms(CMATRIX **Uprev, nHamiltonian &ham) {

  // For adiabatic representation only:
  // Save the previous orbitals info - in case we need to
  // do either phase correction of state tracking

  int ntraj = ham.children.size();

  for (int traj = 0; traj < ntraj; traj++) {
    *Uprev[traj] = ham.children[traj]->get_basis_transform();
  }
}

vector<CMATRIX> compute_St(nHamiltonian &ham, nHamiltonian &ham_prev,
                           int isNBRA) {

  return compute_St(&ham, &ham_prev, isNBRA);
}

vector<CMATRIX> compute_St(nHamiltonian *ham, nHamiltonian *ham_prev,
                           int isNBRA) {
  /**
    This function computes the time-overlap matrices for all trajectories

  */

  int nst = ham->nadi;
  int ntraj = ham->children.size();

  vector<CMATRIX> St(ntraj, CMATRIX(nst, nst));

  if (isNBRA == 1) {
    St[0] = ham_prev->children[0]->get_basis_transform().H() *
            ham->children[0]->get_basis_transform();
    // AVA: temporarily comment on 11/25/2022
    ham->children[0]->set_time_overlap_adi_by_val(St[0]);
  } else {
    for (int traj = 0; traj < ntraj; traj++) {
      St[traj] = ham_prev->children[traj]->get_basis_transform().H() *
                 ham->children[traj]->get_basis_transform();
      ham->children[traj]->set_time_overlap_adi_by_val(St[traj]);
    }
  }
  return St;
}

vector<CMATRIX> compute_St(nHamiltonian &ham, nHamiltonian &ham_prev) {

  int is_nbra = 0;
  return compute_St(&ham, &ham_prev, is_nbra);
}

vector<CMATRIX> compute_St(nHamiltonian *ham, int isNBRA) {
  /**
    This function computes the time-overlap matrices for all trajectories
  */

  int nst = ham->nadi;
  int ntraj = ham->children.size();

  vector<CMATRIX> St(ntraj, CMATRIX(nst, nst));
  if (isNBRA == 1) {
    St[0] = ham->children[0]->get_time_overlap_adi();
  } else {
    for (int traj = 0; traj < ntraj; traj++) {
      St[traj] = ham->children[traj]->get_time_overlap_adi();
    }
  }
  return St;
}

vector<CMATRIX> compute_St(nHamiltonian &ham, int isNBRA) {

  return compute_St(&ham, isNBRA);
}

vector<CMATRIX> compute_St(nHamiltonian &ham) {
  int is_nbra = 0;

  return compute_St(&ham, is_nbra);
}

void apply_afssh(dyn_variables &dyn_var, CMATRIX &C, vector<int> &act_states,
                 MATRIX &invM, nHamiltonian &ham, bp::dict &dyn_params,
                 Random &rnd) {

  // cout<<"In apply_afssh\n";

  dyn_control_params prms;
  prms.set_parameters(dyn_params);

  int i;
  int ndof = invM.n_rows;
  int nst = C.n_rows;
  int ntraj = C.n_cols;
  int traj, idof;
  int num_el = prms.num_electronic_substeps;
  double dt_el = prms.dt / num_el;

  // A-FSSH

  CMATRIX hvib_curr(nst, nst);
  CMATRIX force_full(nst, nst);
  CMATRIX force_diag(nst, nst);
  CMATRIX c_traj(nst, 1);
  CMATRIX dR_afssh(nst, nst);
  CMATRIX dP_afssh(nst, nst);

  // cout<<"Propagating moments...\n";
  //=========================== Propagate moments ===============
  for (traj = 0; traj < ntraj; traj++) {

    // the shared NBRA child serves every trajectory
    nHamiltonian *chld = ham.child(traj);
    hvib_curr = chld->get_hvib_adi();
    c_traj = C.col(traj);

    for (idof = 0; idof < ndof; idof++) {

      force_full = -1.0 * chld->get_d1ham_adi(idof);

      for (i = 0; i < nst; i++) {
        if (std::isnan(force_full.get(i, i).real())) {
          std::cout << "Error in apply_afssh(): the force of state " << i << " along DOF " << idof << " is NaN - "
                    << "A-FSSH reads every state's force, the model handed over its active state's alone "
                    << "(force_source all_states hands over every state's)\n";
          std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
        }
        force_diag.set(i, i, force_full.get(i, i));
      }

      dR_afssh = *dyn_var.dR[traj][idof];
      dP_afssh = *dyn_var.dP[traj][idof];

      integrate_afssh_moments(dR_afssh, dP_afssh, hvib_curr, force_diag, c_traj,
                              1.0 / invM.get(idof, 0), act_states[traj], dt_el,
                              num_el);

      *dyn_var.dR[traj][idof] = dR_afssh;
      *dyn_var.dP[traj][idof] = dP_afssh;

    } // for idof
  } // for traj

  // cout<<"Computing reset and collapse probabilities...\n";

  //======================== Compute reset and collapse probabilities =========

  MATRIX gamma_reset(nst, ntraj);
  MATRIX gamma_collapse(nst, ntraj);

  for (traj = 0; traj < ntraj; traj++) {
    for (i = 0; i < nst; i++) {

      double gamma_reset_i = 0.0;
      double gamma_collapse_i = 0.0;

      for (idof = 0; idof < ndof; idof++) {

        double dx_ii = dR_afssh.get(i, i).real();
        int as = act_states[traj];
        double f_i = -ham.child(traj)->get_d1ham_adi(idof).get(i, i).real();
        double f_as =
            -ham.child(traj)->get_d1ham_adi(idof).get(as, as).real();

        gamma_reset_i -= 0.5 * (f_i - f_as) * dx_ii;

        double f_ji =
            -ham.child(traj)->get_d1ham_adi(idof).get(as, i).real();
        gamma_collapse_i += f_ji * dx_ii;

      } // for idof

      gamma_reset.set(i, traj, gamma_reset_i * prms.dt);
      gamma_collapse.set(
          i, traj, (gamma_reset_i - 2.0 * fabs(gamma_collapse_i)) * prms.dt);

    } // for nst
  } // for traj

  // cout<<"Doing the collapses and resets...\n";
  //======================== Do the collapse and resets =======================

  complex<double> zero(0.0, 0.0);

  for (traj = 0; traj < ntraj; traj++) {

    for (i = 0; i < nst; i++) {

      if (i != act_states[traj]) {

        // Reset
        double ksi = rnd.uniform(0.0, 1.0);

        if (ksi < gamma_reset.get(i, traj)) {
          for (idof = 0; idof < ndof; idof++) {
            dyn_var.dR[traj][idof]->scale(-1, i, zero);
            dyn_var.dR[traj][idof]->scale(i, -1, zero);
            dyn_var.dP[traj][idof]->scale(-1, i, zero);
            dyn_var.dP[traj][idof]->scale(i, -1, zero);
          } // for j
        } // if ksi < gamma_reset

        // Collapse
        ksi = rnd.uniform(0.0, 1.0);
        if (ksi < gamma_collapse.get(i, traj)) {
          collapse(C, traj, act_states[traj], prms.collapse_option);
        } // if ksi < gamma_collapse

      } // all non-active states

    } // for i
  } // for traj

  // cout<<"Done\n";
}

void apply_projectors(CMATRIX &C, vector<CMATRIX> &proj) {

  int ntraj = proj.size();
  int nst = C.n_rows;

  int i;
  vector<int> t2(1, 0);
  vector<int> t3(nst, 0);
  for (i = 0; i < nst; i++) {
    t3[i] = i;
  }
  CMATRIX c_tmp(nst, 1);

  for (int traj = 0; traj < ntraj; traj++) {
    t2[0] = traj;
    pop_submatrix(C, c_tmp, t3, t2);
    c_tmp = proj[traj] * c_tmp;
    push_submatrix(C, c_tmp, t3, t2);
  }
}

CMATRIX Zhu_Liouvillian(double Etot, CMATRIX &Ham, CMATRIX &rho) {
  /**
    This is the Liouvillian based on Eqs. 3.34 and 3.35 of my Chapter, but
    withing the local diabatization approach, that is when NACs are dropped and
    Hvib is replaced by Hadi.
  */

  int nst = Ham.n_cols;
  int sz = nst * nst;

  CMATRIX L(sz, sz);
  int ij;

  double Eeff = 0.0;
  for (int i = 0; i < nst; i++) {
    Eeff += Ham.get(i, i).real() * rho.get(i, i).real();
  }
  Eeff = Etot - Eeff;
  if (Eeff < 0.0) {
    Eeff = 0.0;
  } else {
    Eeff = sqrt(Eeff);
  }

  vector<double> sqE_Ei(nst, 0.0);
  for (int i = 0; i < nst; i++) {
    sqE_Ei[i] = Etot - Ham.get(i, i).real();
    if (sqE_Ei[i] < 0.0) {
      sqE_Ei[i] = 0.0;
    } else {
      sqE_Ei[i] = sqrt(sqE_Ei[i]);
    }
    sqE_Ei[i] *= Eeff;
  }

  ij = 0;
  for (int i = 0; i < nst; i++) {
    for (int j = 0; j < nst; j++) {

      L.set(ij, ij, complex<double>(2.0 * (sqE_Ei[j] - sqE_Ei[i]), 0.0));

      ij++;
    } // for j
  } // for i

  return L;
}

MATRIX momenta_on_excited_states(dyn_variables &dyn_var, nHamiltonian *ham,
                                 int itraj) {
  /**
    Compute the momenta on all excited states given the momentum on the current
    active state such that the total energy is conserved. All done for a given
    trajectory `itraj`. This is done in the adiabatic basis.

    The resulting momenta are taken to be parallel to the current momentum

    Returns: MATRIX(ndof, nadi)
  */

  int nst = dyn_var.nadi;
  int ndof = dyn_var.ndof;
  CMATRIX Ham(nst, nst);
  Ham = ham->get_ham_adi();

  int st_indx = dyn_var.act_states[itraj];
  double Ekin = dyn_var.compute_kinetic_energy(itraj);
  double Epot = Ham.get(st_indx, st_indx).real();

  MATRIX res(ndof, nst);
  MATRIX p_i(ndof, 1);
  MATRIX p(ndof, 1);
  p = dyn_var.p->col(itraj);

  for (int i = 0; i < nst; i++) {

    if (Ekin > 0.0) {
      double Epot_i = Ham.get(i, i).real();
      double Ekin_i = Ekin + Epot - Epot_i;
      if (Ekin_i < 0.0) {
        Ekin_i = 0.0;
      }

      p_i = p * sqrt(Ekin_i / Ekin);
    } else {
      p_i = 0.0;
    }

    for (int j = 0; j < ndof; j++) {
      res.set(j, i, p_i.get(j, 0));
    }

  } // for i - all states

  return res;
}

MATRIX momenta_on_excited_states(dyn_variables &dyn_var, nHamiltonian &ham,
                                 int itraj) {
  return momenta_on_excited_states(dyn_var, &ham, itraj);
}

void SSY_correction(CMATRIX &Ham, dyn_variables &dyn_var, nHamiltonian *ham,
                    int itraj) {
  /**
    This is a correction of a Hamiltonian according to the phase-correction of
    Shenvi-Subotnik-Yang, 2011

    See my Chapter, Eq. 3.27
  */

  int ndof = dyn_var.ndof;
  int nst = dyn_var.nadi;

  MATRIX p(ndof, nst);
  p = momenta_on_excited_states(dyn_var, ham, itraj);
  int st_indx = dyn_var.act_states[itraj]; // active state index
  MATRIX p_act(ndof, 1);
  p_act = p.col(st_indx);
  MATRIX p_tmp(ndof, 1);

  for (int i = 0; i < nst; i++) {
    p_tmp = p.col(i);
    p_tmp.dot_product(p_tmp, *dyn_var.iM);
    p_tmp.dot_product(p_tmp, p_act);
    double sm = p_tmp.sum();

    Ham.set(i, i, complex<double>(-sm, 0.0));

  } // for i
}

void SSY_correction(CMATRIX &Ham, dyn_variables &dyn_var, nHamiltonian &ham,
                    int itraj) {
  SSY_correction(Ham, dyn_var, &ham, itraj);
}

// the diabatic metric changed over the step (any element, exactly) - the rep-0
// propagators then take the Lowdin-frame step instead of their constant-metric form,
// and compute_dynamics demands a model dS = dc1_dia + dc1_dia^H where the frame's
// derivative is read
static int metric_moved(CMATRIX &S_old, CMATRIX &S_new) {
  for (int i = 0; i < S_old.n_rows; i++) {
    for (int j = 0; j < S_old.n_cols; j++) {
      if (S_old.get(i, j) != S_new.get(i, j)) { return 1; }
    }
  }
  return 0;
}

// the factorizations of a step's metric, memoized by content: compute_dynamics calls
// propagate_electronic once per electronic substep, and the substeps of one nuclear step -
// and every trajectory under one shared Hamiltonian - come with the same metric ends
// S_first, S_last. The inverse and the Lowdin frame of a metric that does not move over
// the step, the square roots of the ends of one that does, are computed once per distinct
// pair, each on its first use. A few entries per thread, the least recently used replaced
struct dia_metric_factors {
  CMATRIX S_first;  CMATRIX S_last;
  int moving;
  int has_inverse;  CMATRIX invS;                  // S_last^-1 (a metric that does not move)
  int lowdin_state;  CMATRIX X;  CMATRIX Y;        // its Lowdin frame X = S^-1/2, Y = S^1/2: -1 unknown, 0 identity, 1 set
  int has_sqrt;  CMATRIX Y0;  CMATRIX X0;  CMATRIX Y1;  CMATRIX X1;   // S^1/2, S^-1/2 of the ends (a metric that moves)
  long stamp;
};
static dia_metric_factors &dia_metric(CMATRIX &S_first, CMATRIX &S_last) {
  thread_local std::deque<dia_metric_factors> memo;   // a deque: a reference handed out stays valid when a later lookup adds an entry
  thread_local long clock = 0;
  int sz = S_last.n_cols;
  for (size_t k = 0; k < memo.size(); k++) {
    dia_metric_factors &e = memo[k];
    if (e.S_last.n_cols == sz && e.S_first.n_cols == S_first.n_cols &&
        !metric_moved(e.S_first, S_first) && !metric_moved(e.S_last, S_last)) {
      e.stamp = ++clock;
      return e;
    }
  }
  size_t slot = memo.size();
  if (slot < 8) {
    memo.push_back(dia_metric_factors());
  } else {
    slot = 0;
    for (size_t k = 1; k < memo.size(); k++) { if (memo[k].stamp < memo[slot].stamp) { slot = k; } }
  }
  dia_metric_factors &e = memo[slot];
  e.S_first = S_first;  e.S_last = S_last;
  e.moving = metric_moved(S_first, S_last);
  e.has_inverse = 0;  e.lowdin_state = -1;  e.has_sqrt = 0;
  e.stamp = ++clock;
  return e;
}
static CMATRIX &dia_metric_inverse(dia_metric_factors &e) {
  if (!e.has_inverse) {
    e.invS = CMATRIX(e.S_last.n_rows, e.S_last.n_cols);
    FullPivLU_inverse(e.S_last, e.invS);
    e.has_inverse = 1;
  }
  return e.invS;
}
static int dia_metric_lowdin(dia_metric_factors &e) {
  if (e.lowdin_state < 0) {
    int sz = e.S_last.n_cols;
    e.X = CMATRIX(sz, sz);  e.Y = CMATRIX(sz, sz);
    e.lowdin_state = libmeigen::lowdin_frame(e.S_last, e.X, e.Y);
  }
  return e.lowdin_state;
}
static void dia_metric_sqrt(dia_metric_factors &e) {
  if (!e.has_sqrt) {
    int sz = e.S_last.n_cols;
    e.Y0 = CMATRIX(sz, sz);  e.X0 = CMATRIX(sz, sz);  e.Y1 = CMATRIX(sz, sz);  e.X1 = CMATRIX(sz, sz);
    libmeigen::sqrt_matrix(e.S_first, e.Y0, e.X0);
    libmeigen::sqrt_matrix(e.S_last, e.Y1, e.X1);
    e.has_sqrt = 1;
  }
}

// the square roots Y = S^1/2, X = S^-1/2 of one metric, memoized by its elements: the ends
// and midpoints of the Lowdin steps repeat (a substep's end is the next substep's start, a
// step's end the next step's start, every trajectory of a shared Hamiltonian brings the
// same metric). Eight entries per thread, the least recently used replaced
struct dia_sqrt_factors {
  CMATRIX S;  CMATRIX Y;  CMATRIX X;
  long stamp;
};
static dia_sqrt_factors &dia_sqrt(CMATRIX &S) {
  thread_local std::deque<dia_sqrt_factors> memo;   // a deque: a reference handed out stays valid when a later lookup adds an entry
  thread_local long clock = 0;
  int sz = S.n_cols;
  for (size_t k = 0; k < memo.size(); k++) {
    dia_sqrt_factors &e = memo[k];
    if (e.S.n_cols == sz && e.S.n_rows == S.n_rows && !metric_moved(e.S, S)) {
      e.stamp = ++clock;
      return e;
    }
  }
  size_t slot = memo.size();
  if (slot < 8) {
    memo.push_back(dia_sqrt_factors());
  } else {
    slot = 0;
    for (size_t k = 1; k < memo.size(); k++) { if (memo[k].stamp < memo[slot].stamp) { slot = k; } }
  }
  dia_sqrt_factors &e = memo[slot];
  e.S = S;
  e.Y = CMATRIX(sz, sz);  e.X = CMATRIX(sz, sz);
  libmeigen::sqrt_matrix(S, e.Y, e.X);
  e.stamp = ++clock;
  return e;
}
// the generator of lowdin_frame_generator (Electronic_Dynamics1.cpp) with the roots from the
// memo - the same G = Xm Hvib Xm + i (Y1 - Y0)/dt Xm, Hermitian part, and the frame factors
// Y0, X1 - a lookup may replace the least recently used entry, so each root is copied out
// before the next lookup
static void dia_lowdin_generator(double dt, CMATRIX &Hvib, CMATRIX &S_old, CMATRIX &S_new,
                                 CMATRIX &G, CMATRIX &Y0, CMATRIX &X1) {
  int sz = Hvib.n_cols;
  CMATRIX Sm(sz, sz);
  Sm = 0.5 * (S_old + S_new);
  Y0 = dia_sqrt(S_old).Y;
  dia_sqrt_factors &f1 = dia_sqrt(S_new);
  CMATRIX Y1(f1.Y);
  X1 = f1.X;
  CMATRIX Xm(dia_sqrt(Sm).X);
  G = Xm * Hvib * Xm + complex<double>(0.0, 1.0 / dt) * ((Y1 - Y0) * Xm);
  G = 0.5 * (G + G.H());
}
// the Lowdin-frame step of propagate_electronic_lowdin on the memoized roots: b = Y0 c,
// exp(-i G dt) (rot: the sequential rotations), c = X1 b
static void dia_lowdin_step(double dt, CMATRIX &C, CMATRIX &Hvib, CMATRIX &S_old, CMATRIX &S_new,
                            int rot) {
  int sz = Hvib.n_cols;
  CMATRIX G(sz, sz);  CMATRIX Y0(sz, sz);  CMATRIX X1(sz, sz);
  dia_lowdin_generator(dt, Hvib, S_old, S_new, G, Y0, X1);
  CMATRIX b(Y0 * C);
  if (rot) { propagate_electronic_rot(dt, b, G); }
  else { propagate_electronic_eig(dt, b, G); }
  C = X1 * b;
}
// one exact step of i S dc/dt = Hvib c over dt with the metric moving from S_old to
// S_new: the Lowdin-frame step when it moves, exp(-i S^-1 Hvib dt) when it does not -
// the constant-metric form of the rep-0 method 3, shared by the 2-point steps 4-8
static void dia_exp_step(double dt, CMATRIX &C, CMATRIX &Hvib, CMATRIX &S_old,
                         CMATRIX &S_new) {
  if (metric_moved(S_old, S_new)) {
    dia_lowdin_step(dt, C, Hvib, S_old, S_new, 0);
  } else {
    CMATRIX G(dia_metric_inverse(dia_metric(S_new, S_new)) * Hvib);
    propagate_electronic_nonHermitian(dt, C, G);
  }
}

// the same step by sequential rotations (the rep-0 methods 10-15 and 19): a splitting
// of the pairwise couplings, second order in |coupling| dt - it follows the exp_ step
// only while the couplings of Hvib_dia stay far below hbar / dt. The rotations need
// a Hermitian generator and an orthonormal basis, so a non-identity metric is taken
// to its Lowdin frame b = S^1/2 c - a moving metric by propagate_electronic_lowdin with
// the rotation step inside its frame
static void dia_rot_step(double dt, CMATRIX &C, CMATRIX &Hvib, CMATRIX &S_old,
                         CMATRIX &S_new) {
  if (metric_moved(S_old, S_new)) {
    dia_lowdin_step(dt, C, Hvib, S_old, S_new, 1);
  } else {
    dia_metric_factors &fac = dia_metric(S_new, S_new);
    if (dia_metric_lowdin(fac)) {
      CMATRIX G(fac.X * Hvib * fac.X);
      CMATRIX b(fac.Y * C);
      propagate_electronic_rot(dt, b, G);
      C = fac.X * b;
    } else {
      propagate_electronic_rot(dt, C, Hvib);
    }
  }
}

// the second-Magnus generator of a step whose Hamiltonian moves linearly from A to B over
// the nuclear step dt_step: (A + B)/2 + (i dt_step/12) [A, B] - Hermitian for Hermitian A and
// B, so exp(-i dt G) is the step's propagator (the whole-step commutator, the electronic
// substeps dt multiply up to the one-step exponent). Under a metric S the generators are
// S^-1 H and the commutator reads A S^-1 B - B S^-1 A, S^-1 applied outside
static CMATRIX magnus2_generator(CMATRIX &A, CMATRIX &B, double dt_step) {
  return 0.5 * (A + B) + complex<double>(0.0, dt_step / 12.0) * (A * B - B * A);
}
static CMATRIX magnus2_generator(CMATRIX &A, CMATRIX &B, CMATRIX &invS,
                                 double dt_step) {
  return 0.5 * (A + B) + complex<double>(0.0, dt_step / 12.0) * (A * invS * B - B * invS * A);
}
// the plain density C C^H over a metric that moves within the step, in the Lowdin frame of the
// generator G (lowdin_frame_generator or the Magnus generator of two frames): Y0 rho Y0^H
// into the frame, the Liouvillian of G over dt, X1 rho X1^H out of it - the density-matrix
// image of the amplitude step b = Y0 c, exp(-i G dt), c = X1 b
static void dia_dm_lowdin_step(double dt, CMATRIX &RHO, CMATRIX &G, CMATRIX &Y0, CMATRIX &X1) {
  int sz = G.n_cols;
  CMATRIX rho_b(Y0 * RHO * Y0.H());
  CMATRIX L(make_Liouvillian(G));
  CMATRIX v(vectorize_density_matrix(&rho_b));
  v = exp_(L, complex<double>(0.0, -dt)) * v;
  rho_b = unvectorize_density_matrix(v);
  RHO = X1 * rho_b * X1.H();
}
// the rep-0 step with the second Magnus term (methods 9 and 19, by rotations when rot is
// set): exp(-i (A + B) dt/2 + (dt^2/12) [A, B]) with A, B the generators of the old and
// the new frame - exact for a generator linear over the nuclear step. dt is the
// electronic substep and dt_step the nuclear step: the commutator carries the whole
// step, so the substeps multiply up to the one-step exponent. S_first -> S_last is the
// metric over the nuclear step, S_old -> S_new over this substep. A constant metric
// keeps the generators S^-1 Hvib, whose Magnus exponent is S^-1 applied to the Hermitian
// (A + B)/2 + (i dt/12) (A S^-1 B - B S^-1 A) - a moving metric takes the Lowdin frame
// b = S^1/2 c, where the two generators are X Hvib X + i (dY/dt) X at the ends of the
// nuclear step, Hermitian parts as propagate_electronic_lowdin takes them
static void dia_magnus_step(double dt, double dt_step, CMATRIX &C, CMATRIX &Hvib_old,
                            CMATRIX &Hvib_new, CMATRIX &S_first, CMATRIX &S_last,
                            CMATRIX &S_old, CMATRIX &S_new, int rot) {
  int sz = Hvib_old.n_cols;
  if (metric_moved(S_first, S_last)) {
    dia_metric_factors &fac = dia_metric(S_first, S_last);
    dia_metric_sqrt(fac);
    CMATRIX dY(complex<double>(0.0, 1.0 / dt_step) * (fac.Y1 - fac.Y0));
    CMATRIX GA(fac.X0 * Hvib_old * fac.X0 + dY * fac.X0);
    CMATRIX GB(fac.X1 * Hvib_new * fac.X1 + dY * fac.X1);
    GA = 0.5 * (GA + GA.H());
    GB = 0.5 * (GB + GB.H());
    CMATRIX G(magnus2_generator(GA, GB, dt_step));
    // the ends of this substep - the step's own ends for a single substep
    CMATRIX Ya(sz, sz);  CMATRIX Xa(sz, sz);
    CMATRIX Yb(sz, sz);  CMATRIX Xb(sz, sz);
    if (metric_moved(S_old, S_first)) { libmeigen::sqrt_matrix(S_old, Ya, Xa); } else { Ya = fac.Y0;  Xa = fac.X0; }
    if (metric_moved(S_new, S_last)) { libmeigen::sqrt_matrix(S_new, Yb, Xb); } else { Yb = fac.Y1;  Xb = fac.X1; }
    CMATRIX b(Ya * C);
    if (rot) { propagate_electronic_rot(dt, b, G); }
    else { propagate_electronic_eig(dt, b, G); }
    C = Xb * b;
  } else {
    CMATRIX &invS = dia_metric_inverse(dia_metric(S_new, S_new));
    CMATRIX Heff(magnus2_generator(Hvib_old, Hvib_new, invS, dt_step));
    if (rot) { dia_rot_step(dt, C, Heff, S_new, S_new); }
    else { dia_exp_step(dt, C, Heff, S_new, S_new); }
  }
}

void propagate_electronic(dyn_variables &dyn_var, nHamiltonian &ham,
                          nHamiltonian &ham_prev, dyn_control_params &prms) {

  // a single external call covers the whole step - act as its last
  // substep so the reprojection T applies
  propagate_electronic(dyn_var, &ham, &ham_prev, prms,
                       prms.num_electronic_substeps - 1);
}

void propagate_electronic(dyn_variables &dyn_var, nHamiltonian &ham,
                          nHamiltonian &ham_prev, dyn_control_params &prms,
                          int isub) {

  propagate_electronic(dyn_var, &ham, &ham_prev, prms, isub);
}

void apply_thermal_correction(dyn_variables &dyn_var, nHamiltonian &ham,
                              nHamiltonian &ham_old, vector<int> old_states,
                              dyn_control_params &prms, Random &rnd) {
  /**
    Computes the thermal corrections the adiabatic NACs according to the
    experimental method and scales NACs according to it
  */

  //======= Parameters of the dyn variables ==========
  int ntraj = dyn_var.ntraj;
  int nadi = dyn_var.nadi;
  int ndia = dyn_var.ndia;

  int nst, a, b;
  if (prms.rep_tdse == 0 || prms.rep_tdse == 2) {
    nst = ndia;
  } else if (prms.rep_tdse == 1 || prms.rep_tdse == 3 || prms.rep_tdse == 4) {
    nst = nadi;
  } else {
    nst = -1;
  } // to throw an error

  // per-trajectory scale factors cannot live on a shared NBRA child -
  // every trajectory's factor would compound on the same matrices
  if ((int)ham.children.size() < ntraj) {
    cout << "Error in apply_thermal_correction(): TC-NBRA scales per-trajectory "
         << "couplings, but the hierarchy carries " << ham.children.size()
         << " children for " << ntraj << " trajectories - per-trajectory "
         << "child Hamiltonians are required\n";
    std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
  }

  for (int itraj = 0; itraj < ntraj; itraj++) {

    int i = dyn_var.act_states[itraj];
    double E_tot = prms.total_energy;
    double T0 = ham.children[itraj]->gs_kinetic_energy;
    double Ei = ham.children[itraj]->get_ham_adi().get(i, i).real();

    // the first call of the thermal correction function after initialization
    if (fabs(dyn_var.tcnbra_ekin[itraj] - (-1000.0)) < 1e-4) {
      dyn_var.tcnbra_ekin[itraj] = (E_tot - Ei);
      dyn_var.tcnbra_thermostats[itraj].nu_therm = prms.tcnbra_nu_therm;
      dyn_var.tcnbra_thermostats[itraj].NHC_size = prms.tcnbra_nhc_size;

      dyn_var.tcnbra_thermostats[itraj].Nf_t =
          dyn_var.ndof -
          prms.constrained_dofs
              .size(); // For now, we assume  prms.quantum_dofs.size();

      dyn_var.tcnbra_thermostats[itraj].init_nhc();
    }

    int prev_i = old_states[itraj];
    double E_prev_i =
        ham_old.children[itraj]->get_ham_adi().get(prev_i, prev_i).real();
    double dE = Ei - E_prev_i;        // adiabatic change of potential energy
    dyn_var.tcnbra_ekin[itraj] -= dE; // reflect it in the kinetic energy

    if (dyn_var.tcnbra_ekin[itraj] > 0.0) {

      double scl = dyn_var.tcnbra_thermostats[itraj].vel_scale(0.5 * prms.dt);
      dyn_var.tcnbra_ekin[itraj] *= (scl * scl);

      dyn_var.tcnbra_thermostats[itraj].propagate_nhc(
          prms.dt, dyn_var.tcnbra_ekin[itraj], 0.0, 0.0);

      scl = dyn_var.tcnbra_thermostats[itraj].vel_scale(0.5 * prms.dt);
      dyn_var.tcnbra_ekin[itraj] *= (scl * scl);
    }

    if (prms.tcnbra_do_nac_scaling == 1) {
      double alp = dyn_var.tcnbra_ekin[itraj] / T0;

      if (alp > 0.0) {
        alp = sqrt(alp);
      } else {
        alp = 1.0;
      }

      dyn_var.thermal_correction_factors[itraj] = alp;

      // Scale hvib_adi, and nac_adi by the alp parameter
      for (a = 0; a < nst; a++) {
        for (b = 0; b < nst; b++) {

          if (a != b) {
            ham.children[itraj]->nac_adi->scale(a, b, alp);
            ham.children[itraj]->hvib_adi->scale(a, b, alp);
            ham.children[itraj]->time_overlap_adi->scale(a, b, alp);
          }
        } // for b
      } // for a
    } // if do scaling
    else {
      dyn_var.thermal_correction_factors[itraj] = 1.0;
    }

  } // for itraj
}

void remove_thermal_correction(dyn_variables &dyn_var, nHamiltonian &ham,
                               dyn_control_params &prms) {

  int ntraj = dyn_var.ntraj;
  int nadi = dyn_var.nadi;
  int ndia = dyn_var.ndia;

  int nst, a, b;
  if (prms.rep_tdse == 0 || prms.rep_tdse == 2) {
    nst = ndia;
  } else if (prms.rep_tdse == 1 || prms.rep_tdse == 3) {
    nst = nadi;
  }
  else {
    std::cout<<"Error in remove_thermal_correction(): rep_tdse = "<<prms.rep_tdse<<" has no branch here (implemented: 0, 1, 2, 3)\n";
    std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
  }

  // per-trajectory scale factors cannot live on a shared NBRA child -
  // every trajectory's factor would compound on the same matrices
  if ((int)ham.children.size() < ntraj) {
    cout << "Error in remove_thermal_correction(): TC-NBRA scales per-trajectory "
         << "couplings, but the hierarchy carries " << ham.children.size()
         << " children for " << ntraj << " trajectories - per-trajectory "
         << "child Hamiltonians are required\n";
    std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
  }

  for (int itraj = 0; itraj < ntraj; itraj++) {

    double alp = dyn_var.thermal_correction_factors[itraj];

    // Scale hvib_adi, and nac_adi by the alp parameter
    for (a = 0; a < nst; a++) {
      for (b = 0; b < nst; b++) {

        if (a != b) {
          ham.children[itraj]->nac_adi->scale(a, b, 1.0 / alp);
          ham.children[itraj]->hvib_adi->scale(a, b, 1.0 / alp);
          ham.children[itraj]->time_overlap_adi->scale(a, b, 1.0 / alp);
          //          ham->children[itraj]->ovlp_adi->scale(a, b, alp)
        }
      } // for b
    } // for a

  } // for itraj
}

void update_wp_width(dyn_variables &dyn_var, dyn_control_params &prms) {
  /**
    Updates the wave packet width in XF based on the Gaussian approximation
  */

  //======= Parameters of the dyn variables ==========
  int ndof = dyn_var.ndof;
  int ntraj = dyn_var.ntraj;

  if (prms.use_td_width == 0) {
    for (int itraj = 0; itraj < ntraj; itraj++) {
      for (int idof = 0; idof < ndof; idof++) {
        dyn_var.wp_width->set(idof, itraj, prms.wp_width->get(idof, 0));
      }
    }
  } else if (prms.use_td_width == 1) {
    double elapsed_time, s2;

    elapsed_time = prms.dt * dyn_var.timestep;

    for (int itraj = 0; itraj < ntraj; itraj++) {
      for (int idof = 0; idof < ndof; idof++) {
        s2 = pow(prms.wp_width->get(idof, 0), 2.0) +
             pow(prms.wp_v->get(idof, 0) * elapsed_time, 2.0);
        dyn_var.wp_width->set(idof, itraj, sqrt(s2));
      }
    }
  } else if (prms.use_td_width == 2) {
    double s2;

    for (int itraj = 0; itraj < ntraj; itraj++) {
      for (int idof = 0; idof < ndof; idof++) {
        s2 = pow(prms.wp_width->get(idof, 0), 2.0);
        dyn_var.wp_width->set(idof, itraj,
                              4 * M_PI / (s2 * dyn_var.p->get(idof, itraj)));
      }
    }
  } else if (prms.use_td_width == 3) {
    int nadi = dyn_var.nadi;

    // wp_width is computed by pairwise widths based on auxiliary trajectories
    dyn_var.wp_width->set(-1, -1, 0.0);
    MATRIX sum_inv_w2(ndof, 1);
    MATRIX w2_temp(ndof, 1);

    for (int traj = 0; traj < ntraj; traj++) {
      vector<int> &is_mixed = dyn_var.is_mixed[traj];
      vector<int> &is_first = dyn_var.is_first[traj];

      sum_inv_w2.set(-1, 0, 0.0);
      w2_temp.set(-1, 0, 0.0);

      int check_mixing = 0;
      for (int i = 0; i < nadi; i++) {
        for (int j = 0; j < nadi; j++) {
          if (i >= j) {
            continue;
          }

          if (is_mixed[i] == 0 or is_mixed[j] == 0) {
            continue;
          }
          check_mixing = 1;

          if (is_first[i] == 1 or is_first[j] == 1) {
            w2_temp.set(-1, 0, 1.0e+10); // At initial, an auxiliary pair does
                                         // not contribute to wp_width
          } else {
            for (int idof = 0; idof < ndof; idof++) {
              w2_temp.set(idof, 0,
                          fabs(dyn_var.q_aux[traj]->get(i, idof) -
                               dyn_var.q_aux[traj]->get(j, idof)) /
                              fabs(dyn_var.p_aux[traj]->get(i, idof) -
                                   dyn_var.p_aux[traj]->get(j, idof)));
            }
          }

          for (int idof = 0; idof < ndof; idof++) {
            sum_inv_w2.add(idof, 0, 1.0 / w2_temp.get(idof));
          }

        } // j
      } // i

      if (check_mixing == 1) {
        for (int idof = 0; idof < ndof; idof++) {
          dyn_var.wp_width->set(
              idof, traj, sqrt((nadi - 1) * 1.0 / sum_inv_w2.get(idof, 0)));
        }
      }

    } // traj
  }
  else {
    std::cout<<"Error in update_wp_width(): use_td_width = "<<prms.use_td_width<<" has no branch here (implemented: 0, 1, 2, 3)\n";
    std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
  }
}

int need_active_states_diff_rep(dyn_control_params &prms) {
  int res = 0;

  int list_size = bp::len(prms.properties_to_save);

  // every saved property that reads the active states of the OTHER
  // representation: the plain SH populations, the Tempelaar-Reichman
  // TR populations (compute_average_sh_pop_TR(1) reads act_states_dia,
  // TR(0) reads act_states) and the raw states_dia list
  if (prms.rep_sh == 0) {
    for (int i = 0; i < list_size; ++i) {
      // Extract each element and convert it to a std::string
      std::string property =
          bp::extract<std::string>(prms.properties_to_save[i]);
      if (property == "sh_pop_adi" || property == "sh_pop_dia_TR" ||
          property == "states") {
        res = 1;
      }
    }
  } else if (prms.rep_sh == 1) {
    for (int i = 0; i < list_size; ++i) {
      // Extract each element and convert it to a std::string
      std::string property =
          bp::extract<std::string>(prms.properties_to_save[i]);
      if (property == "sh_pop_dia" || property == "sh_pop_adi_TR" ||
          property == "states_dia") {
        res = 1;
      }
    }
  }
  else {
    std::cout<<"Error in need_active_states_diff_rep(): rep_sh = "<<prms.rep_sh<<" has no branch here (implemented: 0, 1)\n";
    std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
  }

  return res;
}

static void propagate_amplitudes_adi_exp(int method, double dt, CMATRIX &C,
                                          CMATRIX &T_new, nHamiltonian *ham,
                                          nHamiltonian *ham_prev, int is_ssy,
                                          dyn_variables &dyn_var, int itraj,
                                          int t_identity, int t_ampl_skip,
                                          double dt_step) {
  /**
    One step of the adiabatic-amplitude propagation for the exp_-based
    integrators (electronic_integrator = -1..9 and 100..109), extracted
    verbatim from propagate_electronic. C holds amplitudes as columns: a
    single trajectory when called from the per-trajectory loop, or the whole
    (nadi x ntraj) batch at once on the NBRA fast path, where the one-step
    propagator is common to all trajectories. The rotation-based integrators
    (10-15) remain in the per-trajectory loop - propagate_electronic_rot
    works on single columns by construction.
  */
  // Thread-local workspaces (resized on nadi change): the helper runs both
  // once per step (NBRA fast path) and per trajectory inside OpenMP regions -
  // per-thread reuse removes four nadi^2 allocations+zeroings per call.
  // Every branch assigns these before reading them. t_identity signals that
  // T_new was loaded as the identity (assume_always_consistent), so its
  // factors drop out of the chains exactly - t_ampl_skip suppresses only the
  // amplitude-side T factors (early electronic substeps) while the Hvib
  // dressings keep aligning data frames with the real T. dt_step is the full
  // nuclear step: the commutator of integrator 9 takes it, so that the
  // electronic substeps multiply up to the exponent of the one step.
  thread_local std::unique_ptr<CMATRIX> wHvib, wHvib_old, wA, wB;
  if (!wHvib || wHvib->n_rows != ham->nadi) {
    wHvib.reset(new CMATRIX(ham->nadi, ham->nadi));
    wHvib_old.reset(new CMATRIX(ham->nadi, ham->nadi));
    wA.reset(new CMATRIX(ham->nadi, ham->nadi));
    wB.reset(new CMATRIX(ham->nadi, ham->nadi));
  }
  CMATRIX &Hvib = *wHvib;
  CMATRIX &Hvib_old = *wHvib_old;
  CMATRIX &A = *wA;      /// this is A(t)
  CMATRIX &B = *wB;      /// this is actually A(t+dt)

      if (method == -1) {
        ;
        ;
      } // No evolution
      else if (method == 0 || method == 100) {
        // A crude factorization of the Hamiltonian operator
        Hvib = ham_prev->get_ham_adi_ref(); // T is the identity matrix
        if (is_ssy) {
          SSY_correction(Hvib, dyn_var, ham_prev, itraj);
        }
        exp_(A, Hvib, complex<double>(0.0, -0.5 * dt));

        Hvib = ham->get_ham_adi_ref();
        if (is_ssy) {
          SSY_correction(Hvib, dyn_var, ham, itraj);
        }
        exp_(B, Hvib, complex<double>(0.0, -0.5 * dt));

        if (t_identity) { C.product(B, A, C); }
        // early substeps stay in the old frame: dress the new-frame
        // half-step, defer the relabel to the last substep
        else if (t_ampl_skip) { C.product(adjoint(T_new), B, T_new, A, C); }
        else { C.product(B, T_new, A, C); }
      } // method == 0 && 100

      else if (method == 1 || method == 101) {
        // Trotter-based symmetric splitting
        Hvib = ham_prev->get_ham_adi_ref(); // T is the identity matrix
        if (is_ssy) {
          SSY_correction(Hvib, dyn_var, ham_prev, itraj);
        }
        exp_(B, Hvib, complex<double>(0.0, -0.25 * dt));

        Hvib = ham->get_ham_adi_ref();
        if (is_ssy) {
          SSY_correction(Hvib, dyn_var, ham, itraj);
        }
        exp_(A, Hvib, complex<double>(0.0, -0.5 * dt));

        if (t_identity) { C.product(B, A, B, C); }
        // dressed old-frame step - the leading relabel waits for the
        // last substep
        else if (t_ampl_skip) { C.product(B, adjoint(T_new), A, T_new, B, C); }
        else { C.product(T_new, B, adjoint(T_new), A, T_new, B, C); }
      } // method == 1 && 101

      else if (method == 2 || method == 102) {
        // The local diabatization approach
        Hvib_old = ham_prev->get_ham_adi_ref(); // T is the identity matrix
        if (is_ssy) {
          SSY_correction(Hvib_old, dyn_var, ham_prev, itraj);
        }

        Hvib = ham->get_ham_adi_ref();
        if (is_ssy) {
          SSY_correction(Hvib, dyn_var, ham, itraj);
        }
        if (!t_identity) { Hvib.product(adjoint(T_new), Hvib, T_new); }
        Hvib += Hvib_old;

        exp_(A, Hvib, complex<double>(0.0, -0.5 * dt));
        if (t_identity || t_ampl_skip) { C.product(A, C); }
        else { C.product(T_new, A, C); }
        //      C = A * C;

      } // method == 2 && 102

      else if (method == 3 || method == 103) {
        // 1-point with vibronic Hamiltonian
        Hvib = ham_prev->get_hvib_adi_ref();

        if (is_ssy) {
          SSY_correction(Hvib, dyn_var, ham_prev, itraj);
        }

        exp_(A, Hvib, complex<double>(0.0, -dt));
        if (t_identity || t_ampl_skip) { C.product(A, C); }
        else { C.product(T_new, A, C); }
      } // method == 3 && 103

      else if (method == 4 || method == 104) {
        // 2-point with vibronic Hamiltonian
        Hvib_old = ham_prev->get_hvib_adi_ref();
        if (is_ssy) {
          SSY_correction(Hvib_old, dyn_var, ham_prev, itraj);
        }

        Hvib = ham->get_hvib_adi_ref();
        if (is_ssy) {
          SSY_correction(Hvib, dyn_var, ham, itraj);
        }

        Hvib += Hvib_old;

        exp_(A, Hvib, complex<double>(0.0, -0.5 * dt));
        if (t_identity || t_ampl_skip) { C.product(A, C); }
        else { C.product(T_new, A, C); }
      } // method == 4 && 104

      else if (method == 5 || method == 105) {
        // 2-point Hvib with vibronic Hamiltonian and correction
        Hvib_old = ham_prev->get_hvib_adi_ref(); // T is the identity matrix
        if (is_ssy) {
          SSY_correction(Hvib_old, dyn_var, ham_prev, itraj);
        }

        Hvib = ham->get_hvib_adi_ref();
        if (!t_identity) { Hvib.product(adjoint(T_new), Hvib, T_new); }
        if (is_ssy) {
          SSY_correction(Hvib, dyn_var, ham, itraj);
        }

        Hvib += Hvib_old;
        exp_(A, Hvib, complex<double>(0.0, -0.5 * dt));
        if (t_identity || t_ampl_skip) { C.product(A, C); }
        else { C.product(T_new, A, C); }
      } // method == 5 && 105

      else if (method == 6 || method == 106) {
        // 2-point with vibronic Hamiltonian, no reordering
        Hvib_old = ham_prev->get_hvib_adi_ref();
        if (is_ssy) {
          SSY_correction(Hvib_old, dyn_var, ham_prev, itraj);
        }

        Hvib = ham->get_hvib_adi_ref();
        if (is_ssy) {
          SSY_correction(Hvib, dyn_var, ham, itraj);
        }

        Hvib += Hvib_old;

        // the 2-point propagator exponentiates the SUM (dt/2 x the sum =
        // dt x the midpoint average) - exponentiating Hvib_old advanced
        // only half a step with the stale Hamiltonian
        exp_(A, Hvib, complex<double>(0.0, -0.5 * dt));
        C.product(A, C);

        // dyn_var.proj_adi[itraj]->load_identity();

        // T_new.identity();
      } // method == 6 && 106

      else if (method == 7 || method == 107) {
        // 2-point Hvib with vibronic Hamiltonian and correction - same as 5,
        // but different order Propagate old coefficients using old Hamiltonian,
        // only half-way
        Hvib_old = ham_prev->get_hvib_adi_ref(); // T is the identity matrix
        if (is_ssy) {
          SSY_correction(Hvib_old, dyn_var, ham_prev, itraj);
        }
        exp_(A, Hvib_old, complex<double>(0.0, -0.5 * dt));
        C.product(A, C);

        // Propagate the new coefficients using the transformed new Hamiltonian,
        // half-way
        Hvib = ham->get_hvib_adi_ref();
        if (!t_identity) { Hvib.product(adjoint(T_new), Hvib, T_new); }
        if (is_ssy) {
          SSY_correction(Hvib, dyn_var, ham, itraj);
        }

        exp_(A, Hvib, complex<double>(0.0, -0.5 * dt));
        // T applies once, after both half-steps: a mid-step adjoint(T)
        // reprojection between the two half-exponentials puts them in
        // opposite gauges at an eigenvector branch switch - their
        // couplings cancel and the crossing step transfers nothing
        if (t_identity || t_ampl_skip) { C.product(A, C); }
        else { C.product(T_new, A, C); }

      } // method == 7 && 107

      else if (method == 8 || method == 108) {
        // new LD: 2-point Ham - same as 2 but different order

        // Propagate with the old Ham
        Hvib_old = ham_prev->get_hvib_adi_ref(); // T is the identity matrix
        if (is_ssy) {
          SSY_correction(Hvib_old, dyn_var, ham_prev, itraj);
        }
        exp_(A, Hvib_old, complex<double>(0.0, -0.5 * dt));
        C.product(A, C);

        // Propagate with the new Ham
        Hvib = ham->get_hvib_adi_ref();
        if (is_ssy) {
          SSY_correction(Hvib, dyn_var, ham, itraj);
        }
        if (!t_identity) { Hvib.product(adjoint(T_new), Hvib, T_new); }
        exp_(A, Hvib, complex<double>(0.0, -0.5 * dt));
        // the T-dressed propagation ends in the previous step's frame -
        // without a final reprojection the amplitudes fall out of the
        // basis bookkeeping at every eigenvector branch switch
        if (t_identity || t_ampl_skip) { C.product(A, C); }
        else { C.product(T_new, A, C); }

      } // method == 8 && 108

      else if (method == 9 || method == 109) {
        // The local diabatization approach with the second Magnus term. With
        // A the old Hamiltonian and B the new one in the old frame, linear in
        // between, the propagator of the step is
        // exp(-i (A + B) dt/2 + (dt^2/12) [A, B]). The averaged form (2)
        // leaves the commutator out - error constant -1/12 - and the split
        // form (0) carries (dt^2/8) [A, B] - error constant +1/24
        Hvib_old = ham_prev->get_ham_adi_ref(); // T is the identity matrix
        if (is_ssy) {
          SSY_correction(Hvib_old, dyn_var, ham_prev, itraj);
        }

        Hvib = ham->get_ham_adi_ref();
        if (is_ssy) {
          SSY_correction(Hvib, dyn_var, ham, itraj);
        }
        if (!t_identity) { Hvib.product(adjoint(T_new), Hvib, T_new); }
        Hvib = magnus2_generator(Hvib_old, Hvib, dt_step);

        exp_(A, Hvib, complex<double>(0.0, -dt));
        if (t_identity || t_ampl_skip) { C.product(A, C); }
        else { C.product(T_new, A, C); }

      } // method == 9 && 109

      else {
        // sanity_check refuses these earlier - a silent fall-through would
        // leave the amplitudes untouched for the whole run
        cout << "Error in propagate_electronic(): electronic_integrator = " << method
             << " has no adiabatic branch (rep_tdse = 1) - use -1, 0-9, 10-15, 19 or their "
                "+100 variants\n";
        std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
      }
}

void propagate_electronic(dyn_variables &dyn_var, nHamiltonian *Ham,
                          nHamiltonian *Ham_prev, dyn_control_params &prms) {

  // a single external call covers the whole step - act as its last
  // substep so the reprojection T applies
  propagate_electronic(dyn_var, Ham, Ham_prev, prms,
                       prms.num_electronic_substeps - 1);
}

void propagate_electronic(dyn_variables &dyn_var, nHamiltonian *Ham,
                          nHamiltonian *Ham_prev, dyn_control_params &prms,
                          int isub) {

  int itraj;

  int num_el = prms.num_electronic_substeps;
  double dt = prms.dt / num_el;
  int rep = prms.rep_tdse;
  int method = prms.electronic_integrator;
  int is_ssy = prms.do_ssy;

  //======= Parameters of the dyn variables ==========
  // int ndof = dyn_var.ndof;
  int ntraj = dyn_var.ntraj;
  int nadi = dyn_var.nadi;
  int ndia = dyn_var.ndia;

  int nst;
  if (prms.rep_tdse == 0 || prms.rep_tdse == 2) {
    nst = ndia;
  } else if (prms.rep_tdse == 1 || prms.rep_tdse == 3 || prms.rep_tdse == 4) {
    nst = nadi;
  } else {
    nst = -1;
  } // to throw an error

  // int ampl_transformation_method = prms.ampl_transformation_method;

  // NOTE: the amplitude column C is a per-trajectory local (see the loop),
  // and the Liouvillian-space work matrices vRHO/RHO/L are declared inside
  // the density-matrix branches (rep >= 2) - L alone is (nst^2 x nst^2),
  // a large pointless allocation for the common amplitude representations
  // Operate directly on the propagated amplitude matrix in the wavefunction
  // representations - the (nst x ntraj) round-trip copies were pure churn.
  // Nothing inside this function reads the amplitudes besides the propagation
  // itself (SSY corrections read momenta only) - rep >= 2 propagates the
  // density matrices instead and never touches Coeff (dummy target).
  thread_local std::unique_ptr<CMATRIX> w_coeff_dummy;
  if (!w_coeff_dummy) { w_coeff_dummy.reset(new CMATRIX(1, 1)); }
  CMATRIX &Coeff = (prms.rep_tdse == 0) ? *dyn_var.ampl_dia :
                   ((prms.rep_tdse == 1) ? *dyn_var.ampl_adi : *w_coeff_dummy);

  //  dyn_var.compute_kinetic_energies()

  ///======================== Now do the integration of the TD-SE
  ///===================

  // ============================ NBRA fast path ============================
  // Under isNBRA there is a single shared child Hamiltonian and
  // update_proj_adi builds every proj_adi[i] from it, so the one-step
  // propagator is identical for all trajectories: build it once and
  // propagate the whole amplitude batch with one matrix product, instead of
  // ntraj matrix exponentials per step. Falls back to the per-trajectory
  // loop for the rotation-based integrators (single-column by construction),
  // SSY corrections (trajectory-dependent) and non-amplitude representations.
  // state_tracking_algo 4 builds proj_adi from per-trajectory momenta -
  // trajectory 0's projector does not serve the whole batch there
  if (prms.isNBRA == 1 && is_ssy == 0 && rep == 1 && ntraj > 0 &&
      prms.state_tracking_algo != 4 &&
      (method <= 9 || (method >= 100 && method <= 109))) {
    CMATRIX T_new(*dyn_var.proj_adi[0]);
    // proj_adi is the per-nuclear-step reprojection: the amplitude-side
    // factors apply on the last electronic substep only - reapplying them
    // every substep multiplies the basis rotation num_el times (and a
    // sign-flip relabel self-cancels at even substep counts) - while the
    // Hvib dressings align data frames with the real T on every substep
    int t_skip = (isub < num_el - 1);
    propagate_amplitudes_adi_exp(method, dt, Coeff, T_new, Ham->children[0],
                                 Ham_prev->children[0], 0, dyn_var, 0,
                                 prms.assume_always_consistent, t_skip,
                                 prms.dt);
  } else {

  // Deterministic per-trajectory propagation: every iteration works only on
  // its own amplitude column and Hamiltonian child, so the loop parallelizes
  // cleanly and the results are identical for any thread count. The
  // density-matrix representations (rep >= 2) stay serial - their per-
  // iteration Liouvillian workspaces would multiply a large memory footprint.
  // exceptions cannot leave an OMP-outlined region (even a serialized
  // one) - capture and rethrow outside so the guards stay translatable
  std::exception_ptr omp_eptr = nullptr;
  #pragma omp parallel for num_threads(prms.omp_num_threads) \
      if (prms.omp_num_threads > 1 && rep <= 1)
  for (int itraj = 0; itraj < ntraj; itraj++) {
    try {

    // rep >= 2 propagates the density matrices - Coeff is the 1x1
    // dummy there, so the amplitude round-trip stays in the
    // wavefunction representations
    CMATRIX C(nst, 1);
    if (rep <= 1) { C = Coeff.col(itraj); }

    int traj1 = itraj;
    if (method >= 100 && method < 200) {
      traj1 = 0;
    }
    // NBRA runs (isNBRA==1) carry a single child Hamiltonian shared by all
    // trajectories - never index children beyond what actually exists
    if (traj1 >= (int)Ham->children.size()) {
      traj1 = 0;
    }

    nHamiltonian *ham = Ham->children[traj1];
    nHamiltonian *ham_prev = Ham_prev->children[traj1];
    /*

        //================= Basis re-expansion ===================
        CMATRIX P(ham->nadi, ham->nadi);
        //proj[itraj]->load_identity();
        CMATRIX T(*dyn_var.proj_adi[itraj]);  T.load_identity();
        CMATRIX T_new(ham->nadi, ham->nadi);

        P = ham->get_time_overlap_adi();  // U_old.H() * U;

    //    if(prms.thermally_corrected_nbra){
    //      P = thermally_corrected_time_overlap(dyn_var, ham, prms, itraj);
    //    }

        // More consistent with the new derivations:
        FullPivLU_inverse(P, T_new);
        T_new = orthogonalized_T( T_new );

        *dyn_var.proj_adi[itraj] = T_new;

    */
    CMATRIX T_new(*dyn_var.proj_adi[itraj]);
    if (prms.assume_always_consistent) {
      T_new.identity();
    }
    // per-nuclear-step reprojection: the amplitude-side factors apply
    // on the last substep only, while the Hvib dressings align data
    // frames with the real T on every substep (see the NBRA fast path
    // note)
    int t_skip = (isub < num_el - 1);
    CMATRIX T_ampl(T_new);
    if (t_skip) {
      T_ampl.identity();
    }

    //    if ampl_transformation_method

    if (rep == 0) { // diabatic
      CMATRIX Hvib(ham->ndia, ham->ndia);
      // the metric over this electronic substep: S moves linearly over the nuclear
      // step and a substep carries the amplitudes across its own share of that
      // move alone - the whole move in every substep would apply the change of
      // frame num_electronic_substeps times. The ends of the nuclear step are the
      // very matrices of the Hamiltonians - a single substep, or a metric that does
      // not move, is the plain step
      CMATRIX Sdia_old(ham_prev->get_ovlp_dia());
      CMATRIX Sdia(ham->get_ovlp_dia());
      double w_old = double(isub) / num_el;
      double w_new = double(isub + 1) / num_el;
      CMATRIX S_first(Sdia_old);
      CMATRIX S_last(Sdia);
      if (num_el > 1 && metric_moved(Sdia_old, Sdia)) {
        if (isub > 0) { Sdia_old = (1.0 - w_old) * S_first + w_old * S_last; }
        if (isub < num_el - 1) { Sdia = (1.0 - w_new) * S_first + w_new * S_last; }
      }

      if (method == -1) {
        ;
        ;
      } // No evolution
      else if (method == 0 || method == 100) {
        // Based on Lowdin transformations, using mid-point Hvib
        Hvib = 0.5 * (ham->get_hvib_dia() + ham_prev->get_hvib_dia());
        if (metric_moved(Sdia_old, Sdia)) {
          // a moving metric makes Hvib non-Hermitian - the generalized eigen
          // route cannot take it, the Lowdin-frame step is its exact form
          dia_lowdin_step(dt, C, Hvib, Sdia_old, Sdia, 0);
        } else {
          propagate_electronic_eig(dt, C, Hvib,
                                   Sdia); // in this case C - diabatic coeffs
        }
      } else if (method == 1 || method == 101) {
        Hvib = 0.5 * (ham->get_hvib_dia() + ham_prev->get_hvib_dia());
        if (metric_moved(Sdia_old, Sdia)) {
          // the QTAG step solves Hvib Z = S Z E with one metric and keeps c^H S c for
          // that metric alone - across a metric that moves the norm is not kept, the
          // Lowdin-frame step carries the amplitudes over
          dia_lowdin_step(dt, C, Hvib, Sdia_old, Sdia, 0);
        } else {
          propagate_electronic_qtag(dt, C, Hvib,
                                    Sdia); // in this case C - diabatic coeffs
        }
      } else if (method == 2 || method == 102) {
        Hvib = ham->get_ham_dia();
        CMATRIX Hvib_old(ham->ndia, ham->ndia);
        Hvib_old = ham_prev->get_ham_dia();
        // the transfer Z(old) -> Z(new) belongs to the nuclear step: a substep
        // takes its own share of the move of the Hamiltonian as well, the whole
        // move in every substep repeats the transfer num_electronic_substeps times
        if (num_el > 1) {
          CMATRIX H_first(Hvib_old);
          CMATRIX H_last(Hvib);
          if (isub > 0) { Hvib_old = (1.0 - w_old) * H_first + w_old * H_last; }
          if (isub < num_el - 1) { Hvib = (1.0 - w_new) * H_first + w_new * H_last; }
        }

        propagate_electronic_qtag2(dt, C, Hvib, Hvib_old, Sdia, Sdia_old);
      } else if (method == 3 || method == 103) {
        // exp(S^-1 Hvib_dia dt) with the mid-point Hvib - the Lowdin-frame step when the
        // metric moves (exp(S^-1 Hvib dt) with S taken at one end conserves c^H S c only
        // to O(dt) then)
        Hvib = 0.5 * (ham->get_hvib_dia() + ham_prev->get_hvib_dia());
        dia_exp_step(dt, C, Hvib, Sdia_old, Sdia);
      }

      // The diabatic basis needs no reprojection - the coupling sits in
      // Hvib_dia itself - so the adiabatic variants that differ only by the
      // T dressing coincide here - every step takes the Lowdin-frame form
      // when the metric moves within the step
      else if (method == 4 || method == 104 || method == 5 || method == 105 ||
               method == 6 || method == 106) {
        // 2-point Hvib_dia: the exact midpoint step of method 3 (the
        // adiabatic 4 undressed, 5 with the new point T-dressed, 6 without
        // the projector)
        Hvib = 0.5 * (ham->get_hvib_dia() + ham_prev->get_hvib_dia());
        dia_exp_step(dt, C, Hvib, Sdia_old, Sdia);
      } else if (method == 7 || method == 107 || method == 8 || method == 108) {
        // split 2-point Hvib_dia: half a step with the old Hamiltonian, half
        // with the new, the metric moving half-way in each (the adiabatic 7
        // and 8 differ only in the reprojection, absent here)
        CMATRIX Sdia_mid(0.5 * (Sdia_old + Sdia));
        Hvib = ham_prev->get_hvib_dia();
        dia_exp_step(0.5 * dt, C, Hvib, Sdia_old, Sdia_mid);
        Hvib = ham->get_hvib_dia();
        dia_exp_step(0.5 * dt, C, Hvib, Sdia_mid, Sdia);
      } else if (method == 9 || method == 109 || method == 19 || method == 119) {
        // 2-point Hvib_dia with the second Magnus term, 19 by rotations - for
        // Hvib_dia linear in the step the 2-point average (0, 3-6) has the error
        // constant -1/12 and the split step (7, 8) +1/24, this one neither
        CMATRIX Hvib_old(ham_prev->get_hvib_dia());
        Hvib = ham->get_hvib_dia();
        dia_magnus_step(dt, prms.dt, C, Hvib_old, Hvib, S_first, S_last, Sdia_old, Sdia,
                        (method == 19 || method == 119));
      } else if (method == 10 || method == 110) {
        // crude splitting by rotations: old half, new half
        CMATRIX Sdia_mid(0.5 * (Sdia_old + Sdia));
        Hvib = ham_prev->get_hvib_dia();
        dia_rot_step(0.5 * dt, C, Hvib, Sdia_old, Sdia_mid);
        Hvib = ham->get_hvib_dia();
        dia_rot_step(0.5 * dt, C, Hvib, Sdia_mid, Sdia);
      } else if (method == 11 || method == 111) {
        // symmetric splitting by rotations: old quarter, new half, old quarter
        CMATRIX Sdia_q1(0.75 * Sdia_old + 0.25 * Sdia);
        CMATRIX Sdia_q3(0.25 * Sdia_old + 0.75 * Sdia);
        CMATRIX Hvib_old(ham_prev->get_hvib_dia());
        Hvib = ham->get_hvib_dia();
        dia_rot_step(0.25 * dt, C, Hvib_old, Sdia_old, Sdia_q1);
        dia_rot_step(0.5 * dt, C, Hvib, Sdia_q1, Sdia_q3);
        dia_rot_step(0.25 * dt, C, Hvib_old, Sdia_q3, Sdia);
      } else if (method == 12 || method == 112 || method == 14 || method == 114 ||
                 method == 15 || method == 115) {
        // 2-point Hvib_dia by rotations (the adiabatic 12 local
        // diabatization, 14 undressed and 15 dressed coincide here)
        Hvib = 0.5 * (ham->get_hvib_dia() + ham_prev->get_hvib_dia());
        dia_rot_step(dt, C, Hvib, Sdia_old, Sdia);
      } else if (method == 13 || method == 113) {
        // 1-point Hvib_dia by rotations: the old Hamiltonian over the step
        Hvib = ham_prev->get_hvib_dia();
        dia_rot_step(dt, C, Hvib, Sdia_old, Sdia);
      } else {
        // sanity_check refuses these earlier - a silent fall-through would
        // leave the amplitudes untouched for the whole run
        cout << "Error in propagate_electronic(): electronic_integrator = " << method
             << " has no diabatic branch (rep_tdse = 0) - use -1, 0-15, 19 or their "
                "+100 variants\n";
        std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
      }

    } // rep == 0 // diabatic

    else if (rep == 1) { // adiabatic
      /**
          -1              -  No

           0              -  ld, with crude splitting,  with exp_
           1              -  ld, with symmetric splitting, with exp_
           2              -  ld, original, with exp_
           9              -  ld, original with the second Magnus term, with exp_
           3              -  1-point, Hvib integration, with exp_
           4              -  2-points, Hvib integration, with exp_
           5              -  2-points, Hvib, integration with the second-point
         correction of Hvib, with exp_ 6              -  2-points, Hvib, no
         reordering

          10              -  same as 0, but with rotations
          11              -  same as 1, but with rotations
          12              -  same as 2, but with rotations
          13              -  same as 3, but with rotations
          14              -  same as 4, but with rotations
          15              -  same as 5, but with rotations
          19              -  same as 9, but with rotations

      */
      // Workspaces for the rotation-based integrators below (thread-local:
      // this loop body runs inside an OpenMP region) - the exp-family helper
      // manages its own workspaces. Every branch assigns before reading.
      thread_local std::unique_ptr<CMATRIX> wHvib_rot, wHvib_old_rot;
      if (!wHvib_rot || wHvib_rot->n_rows != ham->nadi) {
        wHvib_rot.reset(new CMATRIX(ham->nadi, ham->nadi));
        wHvib_old_rot.reset(new CMATRIX(ham->nadi, ham->nadi));
      }
      CMATRIX &Hvib = *wHvib_rot;
      CMATRIX &Hvib_old = *wHvib_old_rot;

      if (method <= 9 || (method >= 100 && method <= 109)) {
        propagate_amplitudes_adi_exp(method, dt, C, T_new, ham, ham_prev,
                                     is_ssy, dyn_var, itraj,
                                     prms.assume_always_consistent,
                                     t_skip, prms.dt);
      }

      //========================== Rotation-based ============================
      else if (method == 10 || method == 110) {
        // Same as 0 or 100, but with rotations

        Hvib = ham_prev->get_ham_adi(); // T is the identity matrix
        if (is_ssy) {
          SSY_correction(Hvib, dyn_var, ham_prev, itraj);
        }
        propagate_electronic_rot(0.5 * dt, C, Hvib);
        C = T_new * C;

        Hvib = ham->get_ham_adi();
        if (is_ssy) {
          SSY_correction(Hvib, dyn_var, ham, itraj);
        }
        propagate_electronic_rot(0.5 * dt, C, Hvib);
        // early substeps: undo the relabel, keep the dressing
        if (t_skip) { C = T_new.H() * C; }

      } // method == 10 && 110

      else if (method == 11 || method == 111) {
        // Trotter-based symmetric splitting, but with rotations
        Hvib_old = ham_prev->get_ham_adi(); // T is the identity matrix
        if (is_ssy) {
          SSY_correction(Hvib_old, dyn_var, ham_prev, itraj);
        }
        Hvib = ham->get_ham_adi();
        if (is_ssy) {
          SSY_correction(Hvib, dyn_var, ham, itraj);
        }

        propagate_electronic_rot(0.25 * dt, C, Hvib_old);
        C = T_new * C;
        propagate_electronic_rot(0.5 * dt, C, Hvib);
        C = T_new.H() * C;
        propagate_electronic_rot(0.25 * dt, C, Hvib_old);
        // the leading relabel waits for the last substep
        C = T_ampl * C;

      } // method == 11 && 111

      else if (method == 12 || method == 112) {
        // The local diabatization approach, but with rotation
        Hvib_old = ham_prev->get_ham_adi(); // T is the identity matrix
        if (is_ssy) {
          SSY_correction(Hvib_old, dyn_var, ham_prev, itraj);
        }

        Hvib = ham->get_ham_adi();
        if (is_ssy) {
          SSY_correction(Hvib, dyn_var, ham, itraj);
        }
        Hvib = T_new.H() * Hvib * T_new;
        Hvib += Hvib_old;

        propagate_electronic_rot(0.5 * dt, C, Hvib);
        C = T_ampl * C;

      } // method == 12 && 112

      else if (method == 13 || method == 113) {
        // 1-point with vibronic Hamiltonian
        Hvib = ham_prev->get_hvib_adi();
        if (is_ssy) {
          SSY_correction(Hvib, dyn_var, ham_prev, itraj);
        }

        propagate_electronic_rot(dt, C, Hvib);
        C = T_ampl * C;
      } // method == 13 && 113

      else if (method == 14 || method == 114) {
        // 2-point with vibronic Hamiltonian
        Hvib_old = ham_prev->get_hvib_adi();
        if (is_ssy) {
          SSY_correction(Hvib_old, dyn_var, ham_prev, itraj);
        }

        Hvib = ham->get_hvib_adi();
        if (is_ssy) {
          SSY_correction(Hvib, dyn_var, ham, itraj);
        }

        Hvib += Hvib_old;

        propagate_electronic_rot(0.5 * dt, C, Hvib);
        C = T_ampl * C;
      } // method == 14 && 114

      else if (method == 15 || method == 115) {
        // 2-point Hvib with vibronic Hamiltonian and correction
        Hvib_old = ham_prev->get_hvib_adi(); // T is the identity matrix
        if (is_ssy) {
          SSY_correction(Hvib_old, dyn_var, ham_prev, itraj);
        }

        Hvib = ham->get_hvib_adi();
        Hvib = T_new.H() * Hvib * T_new;
        if (is_ssy) {
          SSY_correction(Hvib, dyn_var, ham, itraj);
        }

        Hvib += Hvib_old;

        propagate_electronic_rot(0.5 * dt, C, Hvib);
        C = T_ampl * C;

      } // method == 15 && 115

      else if (method == 19 || method == 119) {
        // Same as 9 or 109, but with rotations
        Hvib_old = ham_prev->get_ham_adi(); // T is the identity matrix
        if (is_ssy) {
          SSY_correction(Hvib_old, dyn_var, ham_prev, itraj);
        }

        Hvib = ham->get_ham_adi();
        if (is_ssy) {
          SSY_correction(Hvib, dyn_var, ham, itraj);
        }
        Hvib = T_new.H() * Hvib * T_new;
        Hvib = magnus2_generator(Hvib_old, Hvib, prms.dt);

        propagate_electronic_rot(dt, C, Hvib);
        C = T_ampl * C;

      } // method == 19 && 119

      else {
        cout << "Error in propagate_electronic(): electronic_integrator = " << method
             << " has no adiabatic branch (rep_tdse = 1) - use -1, 0-9, 10-15, 19 or their "
                "+100 variants\n";
        std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
      }

    } // rep == 1 - adiabatic

    else if (rep == 2) { // diabatic, density matrix formalism
      CMATRIX vRHO(nst * nst, 1);      // vectorized DM
      CMATRIX RHO(nst, nst);           // DM
      CMATRIX L(nst * nst, nst * nst); // Liouvillian
      // the metric over the nuclear step and over this electronic substep, as the
      // amplitude route takes them (S moves linearly, a substep carries its own share) -
      // a metric that moves takes the density into the Lowdin frame, one that does not
      // keeps the S^-1 Hvib Liouvillian bit-identical
      int with_metric = ham->ovlp_dia_mem_status;
      CMATRIX S_first(nst, nst);  CMATRIX S_last(nst, nst);
      CMATRIX S_old(nst, nst);  CMATRIX S_new(nst, nst);
      S_first.identity();  S_last.identity();
      if (with_metric) {
        S_last = ham->get_ovlp_dia();
        S_first = ham_prev->ovlp_dia_mem_status ? ham_prev->get_ovlp_dia() : S_last;
      }
      int moving = with_metric && metric_moved(S_first, S_last);
      double w_old = double(isub) / num_el;
      double w_new = double(isub + 1) / num_el;
      S_old = (1.0 - w_old) * S_first + w_old * S_last;
      S_new = (1.0 - w_new) * S_first + w_new * S_last;

      if (method == 0 || method == 100) {
        // Based on Lowdin transformations, using mid-point Hvib
        CMATRIX Hvib(ham->ndia, ham->ndia);
        Hvib = 0.5 * (ham->get_hvib_dia() + ham_prev->get_hvib_dia());
        if (moving) {
          CMATRIX G(nst, nst);  CMATRIX Y0(nst, nst);  CMATRIX X1(nst, nst);
          dia_lowdin_generator(dt, Hvib, S_old, S_new, G, Y0, X1);
          dia_dm_lowdin_step(dt, *dyn_var.dm_dia[itraj], G, Y0, X1);
        } else {
          // non-orthogonal diabats: the generator of i S dC/dt = Hvib C is S^-1 Hvib,
          // and its Liouvillian (non-Hermitian, real spectrum E_i - E_j) is the image
          // d rho/dt = -i (A rho - rho A^H) of that equation for the plain C C^H density
          // the rep-sync layer stores. An identity metric leaves Hvib bit-identical
          if (with_metric) {
            Hvib = dia_metric_inverse(dia_metric(S_last, S_last)) * Hvib;
          }
          L = make_Liouvillian(Hvib);
          vRHO = vectorize_density_matrix(dyn_var.dm_dia[itraj]);
          vRHO = exp_(L, complex<double>(0.0, -dt)) * vRHO;
          //      propagate_electronic_rot(dt, vRHO, L);
          *dyn_var.dm_dia[itraj] = unvectorize_density_matrix(vRHO);
        }
      } // method == 0
      else if (method == 9 || method == 109) {
        // mid-point Hvib with the second Magnus term - the diabatic amplitude step 9
        // for the density matrix: the generator is S^-1 applied to the Hermitian
        // (A + B)/2 + (i dt/12) (A S^-1 B - B S^-1 A), the commutator carrying the
        // whole nuclear step so the substeps multiply up to the one-step exponent
        CMATRIX Hvib_old(ham_prev->get_hvib_dia());
        CMATRIX Hvib(ham->get_hvib_dia());
        if (moving) {
          // the two generators of the Lowdin frame at the ends of the nuclear step, as
          // dia_magnus_step builds them, the frame change over this substep
          dia_metric_factors &fac = dia_metric(S_first, S_last);
          dia_metric_sqrt(fac);
          CMATRIX dY(complex<double>(0.0, 1.0 / prms.dt) * (fac.Y1 - fac.Y0));
          CMATRIX GA(fac.X0 * Hvib_old * fac.X0 + dY * fac.X0);
          CMATRIX GB(fac.X1 * Hvib * fac.X1 + dY * fac.X1);
          GA = 0.5 * (GA + GA.H());
          GB = 0.5 * (GB + GB.H());
          CMATRIX G(magnus2_generator(GA, GB, prms.dt));
          // the ends of this substep - the step's own ends for a single substep
          CMATRIX Ya(nst, nst);  CMATRIX Xa(nst, nst);
          CMATRIX Yb(nst, nst);  CMATRIX Xb(nst, nst);
          if (metric_moved(S_old, S_first)) { libmeigen::sqrt_matrix(S_old, Ya, Xa); } else { Ya = fac.Y0;  Xa = fac.X0; }
          if (metric_moved(S_new, S_last)) { libmeigen::sqrt_matrix(S_new, Yb, Xb); } else { Yb = fac.Y1;  Xb = fac.X1; }
          dia_dm_lowdin_step(dt, *dyn_var.dm_dia[itraj], G, Ya, Xb);
        } else {
          CMATRIX invS(ham->ndia, ham->ndia);
          invS.identity();
          if (with_metric) { invS = dia_metric_inverse(dia_metric(S_last, S_last)); }
          CMATRIX Heff(invS * magnus2_generator(Hvib_old, Hvib, invS, prms.dt));
          L = make_Liouvillian(Heff);
          vRHO = vectorize_density_matrix(dyn_var.dm_dia[itraj]);
          vRHO = exp_(L, complex<double>(0.0, -dt)) * vRHO;
          *dyn_var.dm_dia[itraj] = unvectorize_density_matrix(vRHO);
        }
      } // method == 9

      else {
        cout << "Error in propagate_electronic(): electronic_integrator = " << method
             << " has no diabatic density-matrix branch (rep_tdse = 2) - use 0, 9 or their +100 variants\n";
        std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
      }

    } // rep == 2 - diabatic, density matrix

    else if (rep == 3) { // adiabatic, density matrix formalism
      CMATRIX vRHO(nst * nst, 1);      // vectorized DM
      CMATRIX RHO(nst, nst);           // DM
      CMATRIX L(nst * nst, nst * nst); // Liouvillian
      /**
           0              -  mid-point Hvib with the second-point correction of
         Hvib 1              -  Zhu Liouvillian

          10              -  same as 0, but with rotations
           9              -  same as 0 with the second Magnus term
          19              -  same as 9, but with rotations
      */

      if (method == 0 || method == 100) {
        // Based on Lowdin transformations, using mid-point Hvib
        CMATRIX Hvib(ham->nadi, ham->nadi);
        CMATRIX Hvib_prev(ham->nadi, ham->nadi);
        Hvib = T_new.H() * ham->get_ham_adi() * T_new; // "raw" to dyn-const
        Hvib_prev = ham_prev->get_ham_adi();
        // SSY REPLACES the diagonal - correct each endpoint with its
        // own step, then take the midpoint (the m4/m5 convention)
        if (is_ssy) {
          SSY_correction(Hvib, dyn_var, ham, itraj);
          SSY_correction(Hvib_prev, dyn_var, ham_prev, itraj);
        }
        Hvib = 0.5 * (Hvib + Hvib_prev);

        L = make_Liouvillian(Hvib);

        // RHO = T_new.H() * (*dyn_var.dm_adi[itraj]) * T; // "raw" to dyn-const
        RHO = *dyn_var.dm_adi[itraj];
        vRHO = vectorize_density_matrix(RHO);
        vRHO = exp_(L, complex<double>(0.0, -dt)) * vRHO;

        RHO = unvectorize_density_matrix(vRHO);
        RHO = T_ampl * RHO * T_ampl.H(); // dyn-const to "raw"
        *dyn_var.dm_adi[itraj] = RHO;
      } // method == 0 or 100

      else if (method == 1 || method == 101) {
        // Here goes the Zhu's method, Eqs. 3.34-3.35 of my Chapter
        // run this with a TSH versions, not Ehrenfest, cause the total energies
        // are different in these groups of methods

        CMATRIX Hvib(ham->nadi, ham->nadi);
        Hvib = ham_prev->get_ham_adi(); // + ham_prev->get_ham_adi());

        int st_indx = dyn_var.act_states[itraj];
        double Epot = Hvib.get(st_indx, st_indx).real();
        double Ekin = dyn_var.compute_kinetic_energy(itraj);

        L = Zhu_Liouvillian(Epot + Ekin, Hvib, *dyn_var.dm_adi[itraj]);
        //      L = make_Liouvillian(Hvib);
        vRHO = vectorize_density_matrix(dyn_var.dm_adi[itraj]);
        //      propagate_electronic_rot(dt, vRHO, L);
        vRHO = exp_(L, complex<double>(0.0, -dt)) * vRHO;

        RHO = unvectorize_density_matrix(vRHO);
        // dyn-const to "raw", the same direction as methods 0/10:
        // the Liouvillian is built in the previous step's frame
        RHO = T_ampl * RHO * T_ampl.H();
        *dyn_var.dm_adi[itraj] = RHO;

      } // method == 1 or 101

      else if (method == 10 || method == 110) {
        // Same as 0, but with rotations
        CMATRIX Hvib(ham->nadi, ham->nadi);
        CMATRIX Hvib_prev(ham->nadi, ham->nadi);
        Hvib = T_new.H() * ham->get_ham_adi() * T_new; // "raw" to dyn-const
        Hvib_prev = ham_prev->get_ham_adi();
        // SSY REPLACES the diagonal - correct each endpoint with its
        // own step, then take the midpoint (the m4/m5 convention)
        if (is_ssy) {
          SSY_correction(Hvib, dyn_var, ham, itraj);
          SSY_correction(Hvib_prev, dyn_var, ham_prev, itraj);
        }
        Hvib = 0.5 * (Hvib + Hvib_prev);

        L = make_Liouvillian(Hvib);
        // RHO = T_new.H() * (*dyn_var.dm_adi[itraj]) * T; // "raw" to dyn-const
        RHO = *dyn_var.dm_adi[itraj];
        vRHO = vectorize_density_matrix(RHO);

        propagate_electronic_rot(dt, vRHO, L);

        RHO = unvectorize_density_matrix(vRHO);
        RHO = T_ampl * RHO * T_ampl.H(); // dyn-const to "raw"
        *dyn_var.dm_adi[itraj] = RHO;
      } // method == 10 or 110

      else if (method == 9 || method == 109 || method == 19 || method == 119) {
        // Same as 0 (9) or 10 (19) with the second Magnus term: the
        // mid-point Hamiltonian plus (i dt/12) [A, B], A the old Hamiltonian
        // and B the new one in the old frame - Hermitian, so the step is the
        // Liouvillian of exp(-i (A + B) dt/2 + (dt^2/12) [A, B]). The
        // commutator takes the full nuclear step, the electronic substeps
        // multiply up to the exponent of the one step
        CMATRIX Hvib(ham->nadi, ham->nadi);
        CMATRIX Hvib_prev(ham->nadi, ham->nadi);
        Hvib = T_new.H() * ham->get_ham_adi() * T_new; // "raw" to dyn-const
        Hvib_prev = ham_prev->get_ham_adi();
        if (is_ssy) {
          SSY_correction(Hvib, dyn_var, ham, itraj);
          SSY_correction(Hvib_prev, dyn_var, ham_prev, itraj);
        }
        Hvib = magnus2_generator(Hvib_prev, Hvib, prms.dt);

        L = make_Liouvillian(Hvib);
        RHO = *dyn_var.dm_adi[itraj];
        vRHO = vectorize_density_matrix(RHO);
        if (method == 9 || method == 109) {
          vRHO = exp_(L, complex<double>(0.0, -dt)) * vRHO;
        } else {
          propagate_electronic_rot(dt, vRHO, L);
        }

        RHO = unvectorize_density_matrix(vRHO);
        RHO = T_ampl * RHO * T_ampl.H(); // dyn-const to "raw"
        *dyn_var.dm_adi[itraj] = RHO;
      } // method == 9, 109, 19 or 119

      else {
        cout << "Error in propagate_electronic(): electronic_integrator = " << method
             << " has no adiabatic density-matrix branch (rep_tdse = 3) - use 0, 1, 9, 10, 19 or "
                "their +100 variants\n";
        std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
      }

    } // rep == 3 - adiabatic, density matrix
    else {
      std::cout<<"Error in propagate_electronic(): rep = "<<rep<<" has no branch here (implemented: 0, 1, 2, 3)\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }

    //  *dyn_var.proj_adi[itraj] = T_new;

    // Insert the propagated result back
    if (rep <= 1) {
      for (int st = 0; st < nst; st++) {
        Coeff.set(st, itraj, C.get(st, 0));
      }
    }

    } catch (...) {
      #pragma omp critical
      { if (!omp_eptr) { omp_eptr = std::current_exception(); } }
    }
  } // for itraj - all trajectories
  if (omp_eptr) { std::rethrow_exception(omp_eptr); }

  } // NBRA fast path vs per-trajectory loop
}

// Learn * vs &
void propagate_electronic_kcrpmd(dyn_variables &dyn_var, nHamiltonian &Ham,
                                 dyn_control_params &prms, Random &rnd) {

  int num_el = prms.num_electronic_substeps;
  double dt = prms.dt / num_el;
  double beta = (hartree / boltzmann) / prms.Temperature;
  double a = prms.kcrpmd_a;
  double b = prms.kcrpmd_b;
  double c = prms.kcrpmd_c;
  double eta = prms.kcrpmd_eta;
  double gam = prms.kcrpmd_gamma;
  double gamKP = prms.kcrpmd_gammaKP;
  double sigma;
  double xi;
  double theta;

  vector<double> &m_aux_var = dyn_var.m_aux_var;
  vector<double> &y_aux_var = dyn_var.y_aux_var;
  vector<double> &p_aux_var = dyn_var.p_aux_var;
  vector<double> &f_aux_var = dyn_var.f_aux_var;

  if (y_aux_var[0] >= -0.5 && y_aux_var[0] <= 0.5) {
    if (gamKP == 0.0) {
      p_aux_var[0] += 0.5 * f_aux_var[0] * dt;
      y_aux_var[0] += p_aux_var[0] / m_aux_var[0] * dt;
      f_aux_var[0] = (Ham.kcrpmd_effective_auxiliary_force(y_aux_var, beta, eta,
                                                           a, b, c))[0];
      p_aux_var[0] += 0.5 * f_aux_var[0] * dt;
    } else {
      sigma = sqrt(2.0 * gamKP / (beta * m_aux_var[0]));
      xi = rnd.normal();
      theta = rnd.normal();
      p_aux_var[0] +=
          0.5 * xi * sigma * m_aux_var[0] * sqrt(dt) +
          0.5 * (f_aux_var[0] - gamKP * p_aux_var[0]) * dt -
          0.25 * (xi / 2.0 + theta / sqrt(3.0)) * gamKP * sigma * m_aux_var[0] *
              sqrt(pow(dt, 3)) -
          0.125 * gamKP * (f_aux_var[0] - gamKP * p_aux_var[0]) * pow(dt, 2);
      y_aux_var[0] += p_aux_var[0] / m_aux_var[0] * dt +
                      0.5 * theta / sqrt(3.0) * sigma * sqrt(pow(dt, 3));
      f_aux_var[0] = (Ham.kcrpmd_effective_auxiliary_force(y_aux_var, beta, eta,
                                                           a, b, c))[0];
      p_aux_var[0] +=
          0.5 * xi * sigma * m_aux_var[0] * sqrt(dt) +
          0.5 * (f_aux_var[0] - gamKP * p_aux_var[0]) * dt -
          0.25 * (xi / 2.0 + theta / sqrt(3.0)) * gamKP * sigma * m_aux_var[0] *
              sqrt(pow(dt, 3)) -
          0.125 * gamKP * (f_aux_var[0] - gamKP * p_aux_var[0]) * pow(dt, 2);
    }
  } else {
    if (gam == 0.0) {
      p_aux_var[0] += 0.5 * f_aux_var[0] * dt;
      y_aux_var[0] += p_aux_var[0] / m_aux_var[0] * dt;
      f_aux_var[0] = (Ham.kcrpmd_effective_auxiliary_force(y_aux_var, beta, eta,
                                                           a, b, c))[0];
      p_aux_var[0] += 0.5 * f_aux_var[0] * dt;
    } else {
      sigma = sqrt(2.0 * gam / (beta * m_aux_var[0]));
      xi = rnd.normal();
      theta = rnd.normal();
      p_aux_var[0] +=
          0.5 * xi * sigma * m_aux_var[0] * sqrt(dt) +
          0.5 * (f_aux_var[0] - gam * p_aux_var[0]) * dt -
          0.25 * (xi / 2.0 + theta / sqrt(3.0)) * gam * sigma * m_aux_var[0] *
              sqrt(pow(dt, 3)) -
          0.125 * gam * (f_aux_var[0] - gam * p_aux_var[0]) * pow(dt, 2);
      y_aux_var[0] += p_aux_var[0] / m_aux_var[0] * dt +
                      0.5 * theta / sqrt(3.0) * sigma * sqrt(pow(dt, 3));
      f_aux_var[0] = (Ham.kcrpmd_effective_auxiliary_force(y_aux_var, beta, eta,
                                                           a, b, c))[0];
      p_aux_var[0] +=
          0.5 * xi * sigma * m_aux_var[0] * sqrt(dt) +
          0.5 * (f_aux_var[0] - gam * p_aux_var[0]) * dt -
          0.25 * (xi / 2.0 + theta / sqrt(3.0)) * gam * sigma * m_aux_var[0] *
              sqrt(pow(dt, 3)) -
          0.125 * gam * (f_aux_var[0] - gam * p_aux_var[0]) * pow(dt, 2);
    }
  }
}

void renormalize_hopping_probabilities(
    vector<vector<double>> &g,
    vector<vector<vector<double>>> &coherence_factors,
    vector<int> &act_states) {
  /**
    Scale the hopping probabilities by the factors and ensure the sum of all
    hopping probabilities is still 1

    g[itraj][i] - hopping probability from the current active state for the
    trajectory itraj to the state i

    act_states[itraj] - the active state for the trajectory itraj

    coherence_factors[itraj][i][j] - the scaling of the hopping probability from
    state i to state j for the trajectory itraj


    Result: the g vector is modified
  */
  int ntraj = g.size();
  int nstates = coherence_factors[0].size();

  for (int itraj = 0; itraj < ntraj; itraj++) {
    int a = act_states[itraj];
    double sum = 0.0;
    for (int j = 0; j < nstates; j++) {
      if (j != a) {
        double val = g[itraj][j] * coherence_factors[itraj][a][j]; // a -> j hopping probability
        sum += val;
        g[itraj][j] = val;
      }
    } // for j
    // The self-element:
    g[itraj][a] = 1.0 - sum;
    if (g[itraj][a] < 0) {
      g[itraj][a] = 0.0;
    }

  } // for itraj
}

void reset_coherence_factors(dyn_variables &dyn_var, vector<int> &act_states, vector<int> &old_states, double gap_correlation_time) {

  int ntraj = dyn_var.coherence_factors.size();
  int nstates = dyn_var.coherence_factors[0].size();

  for (int itraj = 0; itraj < ntraj; itraj++){


    // This means the hop i -> j  has happened
    if (act_states[itraj] != old_states[itraj]) {
      int i = old_states[itraj];
      int j = act_states[itraj];

      for (int k = 0; k < nstates; k++) {
        dyn_var.coherence_factors[itraj][k][j] = dyn_var.coherence_factors[itraj][j][k] = 1.0;
      }


      // This means the hop i -> j  has happened
//      dyn_var.coherence_factors[itraj][i][j] = dyn_var.coherence_factors[itraj][j][i] = 1.0;
//      dyn_var.coherence_clocks[itraj][i][j] = dyn_var.coherence_clocks[itraj][j][i] = 0.0;


      // Reinitialize running averages for all gaps involving j
      for (int k = 0; k < nstates; k++) {
//        dyn_var.is_first_gap[itraj][i][k] = dyn_var.is_first_gap[itraj][k][i] = 1;
//        dyn_var.is_first_gap[itraj][j][k] = dyn_var.is_first_gap[itraj][k][j] = 1;
      } // for k 

/*
      // Coherence transfer:
      for (int k = 0; k < nstates; k++) {
        if(k==i || k==j){ ;; }
        else{
          //if(itraj==0){ 
          //cout<<"Doing coherence transfer\n";
          dyn_var.coherence_factors[itraj][j][k] = dyn_var.coherence_factors[itraj][i][k]; 
          dyn_var.coherence_factors[itraj][k][j] = dyn_var.coherence_factors[itraj][k][j];

          // Also transfer clocks:
          dyn_var.coherence_clocks[itraj][j][k] = dyn_var.coherence_clocks[itraj][i][k];
          dyn_var.coherence_clocks[itraj][k][j] = dyn_var.coherence_clocks[itraj][k][j];
        } 
      }// for k
*/
      // The rest of coherence factors - leave unchanged
     
    } // if  - hop happened
    else{

/*
      int j = act_states[itraj];
      for (int k = 0; k < nstates; k++) {
        if(dyn_var.coherence_clocks[itraj][k][j] > gap_correlation_time){

          dyn_var.coherence_factors[itraj][k][j] = dyn_var.coherence_factors[itraj][j][k] = 1.0;
          dyn_var.is_first_gap[itraj][j][k] = dyn_var.is_first_gap[itraj][k][j] = 1;
          dyn_var.coherence_clocks[itraj][k][j] = dyn_var.coherence_clocks[itraj][k][j] = 0.0; 

        }// if coherence is longer thant correlation time 
      }// k
*/
    } // no hop 
   
  } // for itraj
}

static void check_hop_force_rows(dyn_control_params &prms, dyn_variables &dyn_var,
                                 nHamiltonian &ham, vector<int> &old_states,
                                 vector<int> &new_states) {
  /**
    After a change of the active state - a surface hop or a relabel at a
    crossing - the force row of the new state must not be NaN (the sentinel
    of a row the model did not hand over) while the old state's row is not:
    that is a model handing over the force of its active state alone (the
    on-the-fly drivers under force_source active_state), whose next half kick
    would run force-free. A zero force is a force. Under hop_force_refresh both
    rows are filled inside the step (refresh_force_rows: a relabel's before the
    second half kick, a hop's after the hops), so a row still NaN here is a
    call-back that returned none.
  */
  if (prms.force_method != 1 || prms.rep_sh != 1) { return; }
  int ndof = ham.nnucl;
  for (int traj = 0; traj < dyn_var.ntraj; traj++) {
    if (old_states[traj] == new_states[traj] || (int)ham.children.size() <= traj) { continue; }
    bool missing = false;
    bool old_carries = false;
    for (int idof = 0; idof < ndof; idof++) {
      CMATRIX d1(prms.rep_force == 0 ? ham.children[traj]->get_d1ham_dia(idof)
                                     : ham.children[traj]->get_d1ham_adi(idof));
      if (std::isnan(d1.get(new_states[traj], new_states[traj]).real())) { missing = true; }
      if (!std::isnan(d1.get(old_states[traj], old_states[traj]).real())) { old_carries = true; }
    }
    if (missing && old_carries) {
      std::cout << "Error in compute_dynamics(): the active state of trajectory " << traj << " changed "
                << old_states[traj] << " -> " << new_states[traj] << " (a hop or a relabel at a crossing) and "
                << "the force of the new state was not handed over (NaN) while the old "
                << "state's row carries one - the model hands over the force of its active state alone (the "
                << "on-the-fly drivers under force_source active_state), so the next half kick would run "
                << "force-free: set hop_force_refresh = 1 (the model is called back at the same geometry for "
                << "the new state's force) or hand over every state's force (force_source all_states)\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }
  }
}

static void refresh_force_rows(dyn_control_params &prms, dyn_variables &dyn_var,
                               nHamiltonian &ham, bp::object py_funct, bp::dict params,
                               vector<int> &old_states, vector<int> &new_states) {
  /**
    The force of the new active state at this geometry for every trajectory
    whose active state changed - in the relabel of this step (called before
    the second half kick) or in its hops (called after them) - while the new
    row of the force hand-over is NaN (not handed over) and the old row is
    not (a model handing over the force of its active state alone). The model
    is called back once with params["force_refresh"] = {trajectory: new state}
    and params["act_state"] updated - the drivers return the forces of the
    listed trajectories and nothing else. A row still NaN after the call-back
    stops the run. The one owner of the refresh: the python step loop has none.
  */
  if (prms.force_method != 1 || prms.rep_sh != 1) { return; }
  int ndof = ham.nnucl;
  bp::dict refresh;
  bp::dict act;
  int any = 0;
  for (int traj = 0; traj < dyn_var.ntraj; traj++) {
    act[traj] = new_states[traj];
    if (old_states[traj] == new_states[traj] || (int)ham.children.size() <= traj) { continue; }
    bool missing = false;
    bool old_carries = false;
    for (int idof = 0; idof < ndof; idof++) {
      CMATRIX d1(prms.rep_force == 0 ? ham.children[traj]->get_d1ham_dia(idof)
                                     : ham.children[traj]->get_d1ham_adi(idof));
      if (std::isnan(d1.get(new_states[traj], new_states[traj]).real())) { missing = true; }
      if (!std::isnan(d1.get(old_states[traj], old_states[traj]).real())) { old_carries = true; }
    }
    if (missing && old_carries) { refresh[traj] = new_states[traj]; any = 1; }
  }
  if (any == 0) { return; }
  params["act_state"] = act;
  params["force_refresh"] = refresh;
  try {
    ham.compute_adiabatic(py_funct, *dyn_var.q, params, 1);
  } catch (...) {
    params.attr("pop")("force_refresh");
    throw;
  }
  params.attr("pop")("force_refresh");
  for (int traj = 0; traj < dyn_var.ntraj; traj++) {
    if (!refresh.has_key(traj)) { continue; }
    bool missing = false;
    for (int idof = 0; idof < ndof && !missing; idof++) {
      CMATRIX d1(prms.rep_force == 0 ? ham.children[traj]->get_d1ham_dia(idof)
                                     : ham.children[traj]->get_d1ham_adi(idof));
      if (std::isnan(d1.get(new_states[traj], new_states[traj]).real())) { missing = true; }
    }
    if (missing) {
      std::cout << "Error in compute_dynamics(): the active state of trajectory " << traj << " changed "
                << old_states[traj] << " -> " << new_states[traj] << " (a hop or a relabel) and the model called "
                << "back for the new state's force (hop_force_refresh) returned none - the next half kick would "
                << "run force-free\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }
  }
}

void compute_dynamics(dyn_variables &dyn_var, bp::dict dyn_params,
                      nHamiltonian &ham, nHamiltonian &ham_aux,
                      bp::object py_funct, bp::dict params, Random &rnd,
                      vector<Thermostat> &therm) {
  /**
    \brief One step of the TSH algorithm for electron-nuclear DOFs for one
    trajectory

    \param[in] Integration time step
    \param[in,out] q [Ndof x Ntraj] nuclear coordinates. Change during the
    integration.
    \param[in,out] p [Ndof x Ntraj] nuclear momenta. Change during the
    integration.
    \param[in] invM [Ndof  x 1] inverse nuclear DOF masses.
    \param[in,out] C [nadi x ntraj]  or [ndia x ntraj] matrix containing the
    electronic coordinates. The amplitudes are assumed to be
    dynamically-consistent
    \param[in,out] projectors [ntraj CMATRIX(nadi, nadi)] - the projector
    matrices that account for the state tracking and phase correction. These
    matrices should be considered as the dynamical varibles, similar to quantum
    amplitudes. Except their evolution does not necessarily follow from some
    equations of motion, but rather from various ad hoc schemes.
    \param[in,out] act_states - vector of ntraj indices of the physical states
    in which each of the trajectories initially is (active states).
    \param[in] ham Is the Hamiltonian object that works as a functor (takes care
    of all calculations of given type)
    - its internal variables (well, actually the variables it points to) are
    changed during the compuations
    \param[in] py_funct Python function object that is called when this
    algorithm is executed. The called Python function does the necessary
    computations to update the diabatic Hamiltonian matrix (and derivatives),
    stored externally.
    \param[in] params The Python object containing any necessary parameters
    passed to the "py_funct" function when it is executed.
    \param[in] params1 The Python dictionary containing the control parameters
    passed to this function
    \param[in] rnd The Random number generator object

    Return: propagates C, q, p and updates state variables

  */

  //cout<<"In compute_dynamics\n";
  //======== General variables =======================
  int i, j, traj, dof, idof, ntraj1;

  //========= Control parameters variables ===========
  dyn_control_params prms;
  prms.set_parameters(dyn_params);

  int num_el = prms.num_electronic_substeps;

  //======= Parameters of the dyn variables ==========
  int ndof = dyn_var.ndof;
  int ntraj = dyn_var.ntraj;
  int nadi = dyn_var.nadi;
  int ndia = dyn_var.ndia;

  int nst;
  if (prms.rep_tdse == 0 || prms.rep_tdse == 2) {
    nst = ndia;
  } else if (prms.rep_tdse == 1 || prms.rep_tdse == 3 || prms.rep_tdse == 4) {
    nst = nadi;
  } else {
    nst = -1;
  } // to throw an error

  vector<int> act_states(dyn_var.act_states); // = dyn_var.act_states;
  vector<int> act_states_dia(dyn_var.act_states_dia);
  MATRIX p(*dyn_var.p);
  MATRIX &invM = *dyn_var.iM;

  //======== General variables =======================
  // Defining ntraj1 as a reference for making these matrices
  // ntraj is defined as q.n_cols as since it would be large in NBRA
  // we can define another variable like ntraj1 and build the matrices based on
  // that. We can make some changes where q is generated but this seems to be a
  // bit easier
  if (prms.isNBRA == 1) {
    ntraj1 = 1;
  } else {
    ntraj1 = ntraj;
  }

  // Defining matrices based on ntraj1
  vector<CMATRIX> Eadi(ntraj1, CMATRIX(nadi, nadi));          // the adiabatic energies and rates, whatever rep_tdse propagates
  vector<MATRIX> decoherence_rates(ntraj1, MATRIX(nadi, nadi));
  vector<double> Ekin(ntraj1, 0.0);
  MATRIX gamma(ndof, ntraj);
  MATRIX p_traj(ndof, 1);
  vector<int> t1(ndof, 0);
  for (dof = 0; dof < ndof; dof++) {
    t1[dof] = dof;
  }
  vector<int> t2(1, 0);
  vector<int> t3(nst, 0);
  for (i = 0; i < nst; i++) {
    t3[i] = i;
  }

  //============ Sanity checks ==================
  int n_therm_dofs = 1; // default
  if (prms.ensemble == 1) {
    n_therm_dofs = therm[0].Nf_t + therm[0].Nf_r;
    if (static_cast<std::size_t>(n_therm_dofs) != prms.thermostat_dofs.size()) {
      cout << "Error in compute_dynamics: The number of thermostat DOFs ( "
              "currently "
           << n_therm_dofs << ") must be \
      equal to the number of thermostat dofs set up by the `thermostat_dofs` parameter ( currently "
           << prms.thermostat_dofs.size() << ")\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }
  }

  //***************************** Coherent dynamics
  //*******************************
  //============== Nuclear propagation ===================
  //  // NVT dynamics
  //  if(prms.ensemble==1){
  //    for(idof=0; idof<n_therm_dofs; idof++){
  //      dof = prms.thermostat_dofs[idof];
  //      for(traj=0; traj<ntraj; traj++){
  //        dyn_var.p->scale(dof, traj, therm[traj].vel_scale(0.5*prms.dt));
  //      }// traj
  //    }// idof
  //  }
  //
  //  *dyn_var.p = *dyn_var.p + 0.5 * prms.dt * (*dyn_var.f);

  // NVT dynamics
  double beta = (hartree / boltzmann) / prms.Temperature;
  MATRIX sigma_therm(ndof, ntraj);
  MATRIX xi_therm(ndof, ntraj);
  MATRIX theta_therm(ndof, ntraj);
  MATRIX extra_force(ndof, ntraj);
  MATRIX extra_momentum(ndof, ntraj);
  if (prms.ensemble == 1) {
    if (therm[0].thermostat_type == "Nose-Hoover") {
      for (idof = 0; idof < n_therm_dofs; idof++) {
        dof = prms.thermostat_dofs[idof];
        for (traj = 0; traj < ntraj; traj++) {
          dyn_var.p->scale(dof, traj, therm[traj].vel_scale(0.5 * prms.dt));
        } // traj
      } // idof
    } // Nose-Hoover
    else if (therm[0].thermostat_type == "Langevin") {
      for (idof = 0; idof < n_therm_dofs; idof++) {
        dof = prms.thermostat_dofs[idof];
        for (traj = 0; traj < ntraj; traj++) {
          sigma_therm.set(
              dof, traj,
              sqrt(2 * invM.get(dof, traj) * therm[traj].nu_therm / beta));
          xi_therm.set(dof, traj, rnd.normal());
          theta_therm.set(dof, traj, rnd.normal());
          extra_force.set(
              dof, traj,
              0.5 * xi_therm.get(dof, traj) * sigma_therm.get(dof, traj) /
                      invM.get(dof, traj) * sqrt(prms.dt) -
                  0.5 * therm[traj].nu_therm * dyn_var.p->get(dof, traj) *
                      prms.dt -
                  0.25 *
                      (xi_therm.get(dof, traj) / 2.0 +
                       theta_therm.get(dof, traj) / sqrt(3.0)) *
                      therm[traj].nu_therm * sigma_therm.get(dof, traj) /
                      invM.get(dof, traj) * sqrt(pow(prms.dt, 3)) -
                  0.125 * therm[traj].nu_therm *
                      (dyn_var.f->get(dof, traj) -
                       therm[traj].nu_therm * dyn_var.p->get(dof, traj)) *
                      pow(prms.dt, 2));
          extra_momentum.set(dof, traj,
                             theta_therm.get(dof, traj) *
                                 sigma_therm.get(dof, traj) /
                                 (2.0 * sqrt(3.0)) * sqrt(pow(prms.dt, 3)));
        } // traj
      } // idof
    } // Langevin
  }

  *dyn_var.p = *dyn_var.p + 0.5 * prms.dt * (*dyn_var.f) + extra_force;
  // if(prms.use_qtsh==1){ *dyn_var.p = *dyn_var.p + 0.5 *
  // compute_dkinemat(dyn_var, ham); }
  if (prms.use_qtsh == 1) {
    *dyn_var.p = *dyn_var.p + 0.5 * prms.dt * (*dyn_var.qtsh_f_nc);
  }

  // Kinetic constraint
  for (auto cdof = 0u; cdof < prms.constrained_dofs.size(); cdof++) {
    dyn_var.p->scale(prms.constrained_dofs[cdof], -1, 0.0);
  }

  if (prms.entanglement_opt == 22) {
    gamma = ETHD3_friction(*dyn_var.q, *dyn_var.p, invM, prms.ETHD3_alpha,
                           prms.ETHD3_beta);
  }
  // Update coordinates of nuclei for all trajectories
  for (traj = 0; traj < ntraj; traj++) {
    for (dof = 0; dof < ndof; dof++) {
      dyn_var.q->add(dof, traj,
                     invM.get(dof, 0) * dyn_var.p->get(dof, traj) * prms.dt +
                         extra_momentum.get(dof, traj));

      if (prms.entanglement_opt == 22) {
        dyn_var.q->add(dof, traj,
                       invM.get(dof, 0) * gamma.get(dof, traj) * prms.dt);
      }
    }
  }

  // Use the current as the old
  ham_aux.copy_content(ham);

  // Recompute diabatic/adiabatic states, time-overlaps, NAC, Hvib, etc. in
  // response to change of q
  update_Hamiltonian_variables(prms, dyn_var, ham, ham_aux, py_funct, params,
                               0);

  // Copy diabatic-to-adiabatic basis transformation to the dynamical variable
  dyn_var.update_basis_transform(ham);

  // the Lowdin frame's derivative reads the metric's motion from dS = dc1_dia + dc1_dia^H
  // (the contract of lowdin_dia_derivative - init_all allocates dc1_dia as zeros for a model
  // without them): a metric that moved over the step while dS vanishes for every DOF has no
  // derivative source, and its readers - the diabatic state and QTSH forces (force_method 1
  // or 3 under rep_force 0) and the Lowdin hopping generator (any hop method under rep_sh 0)
  // - would take the frame as static. Refused where both metrics of the step are at hand -
  // a run without those readers (no forces, no hops - the pure propagation on a moving
  // metric) needs no dc1_dia
  int reads_frame_derivative = ((prms.force_method == 1 || prms.force_method == 3) && prms.rep_force == 0) ||
                               (prms.tsh_method != -1 && prms.rep_sh == 0);
  if (reads_frame_derivative) {
    for (int c = 0; c < (int)ham.children.size(); c++) {
      if (dyn_var.ovlp_dia_is_identity[c]) { continue; }
      nHamiltonian* now = ham.children[c];
      nHamiltonian* was = ham_aux.children[c];
      if (now->ovlp_dia_mem_status == 0 || was->ovlp_dia_mem_status == 0) { continue; }
      if (!metric_moved(*was->ovlp_dia, *now->ovlp_dia)) { continue; }
      int moving = 0;
      for (int n = 0; n < now->nnucl && !moving; n++) {
        if (now->dc1_dia_mem_status[n] == 0) { continue; }
        CMATRIX dS(*now->dc1_dia[n] + (*now->dc1_dia[n]).H());
        for (int i = 0; i < dS.n_rows && !moving; i++) {
          for (int j = 0; j < dS.n_cols; j++) {
            if (std::abs(dS.get(i, j)) > 0.0) { moving = 1; break; }
          }
        }
      }
      if (!moving) {
        std::cout << "Error in compute_dynamics(): the diabatic metric of trajectory " << c
                  << " moved over the step, but the model's dc1_dia give dS = dc1_dia + dc1_dia^H = 0 "
                  << "for every nuclear DOF (absent, or without a Hermitian part) - the Lowdin frame's "
                  << "derivative (the diabatic forces under rep_force 0, the Lowdin hopping generator "
                  << "under rep_sh 0) reads the metric's motion from dS and would take the frame as "
                  << "static. Supply dc1_dia consistent with the metric, keep the metric constant, or "
                  << "run without forces and hops\n";
        std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
      }
    }
  }

  // Recompute the orthogonalized reprojection matrices, stored in
  // dyn_var.proj_adi this calculations used ham.children[i].time_overlap
  // matrix, updated in the previous step
  update_proj_adi(prms, dyn_var, ham, ham_aux);

  // Recompute NAC, Hvib, etc. in response to change of p
  update_Hamiltonian_variables(prms, dyn_var, ham, ham_aux, py_funct, params,
                               1);

  // Update wave packet width in XF algorithm
  if (prms.decoherence_algo == 5 or prms.decoherence_algo == 6) {
    dyn_var.get_current_timestep(params);
    update_wp_width(dyn_var, prms);
  }

  // Propagate electronic coefficients in the [t, t + dt] interval, this also
  // updates the basis re-projection matrices
  for (i = 0; i < num_el; i++) {
    if (prms.decoherence_algo == 5 or prms.decoherence_algo == 6) {
      // the XF half step acts on the propagated adiabatic variable - the density under rep_tdse 3, the amplitudes
      // under 1 (the diabatic representations are refused by sanity_check: interleaved with the propagation, the
      // half step would see the old state in the new basis there)
      if (prms.rep_tdse == 3) {
        update_ham_xf_dm(dyn_var);
        propagate_half_xf_dm(dyn_var, ham, prms);
      } else {
        update_ham_xf(dyn_var);
        propagate_half_xf(dyn_var, ham, prms);
      }
    }

    propagate_electronic(dyn_var, ham, ham_aux, prms, i);

    if (prms.decoherence_algo == 5 or prms.decoherence_algo == 6) {
      rotate_nab_phase(dyn_var, ham, prms);
      // the XF half step acts on the propagated adiabatic variable - the density under rep_tdse 3, the amplitudes
      // under 1 (the diabatic representations are refused by sanity_check: interleaved with the propagation, the
      // half step would see the old state in the new basis there)
      if (prms.rep_tdse == 3) {
        update_ham_xf_dm(dyn_var);
        propagate_half_xf_dm(dyn_var, ham, prms);
      } else {
        update_ham_xf(dyn_var);
        propagate_half_xf(dyn_var, ham, prms);
      }
    }
  }

  // Recompute density matrices in response to the updated amplitudes
  dyn_var.update_amplitudes(prms);
  dyn_var.update_density_matrix(prms);

  if (prms.use_kcrpmd == 1) {
    propagate_electronic_kcrpmd(dyn_var, ham, prms, rnd);
  }

  vector<int> old_states(dyn_var.act_states);

  // In the interval [t, t + dt], we may have experienced the basis reordering,
  // so we need to change the active adiabatic state
  if (prms.tsh_method == 3 or prms.tsh_method == 4 or prms.rep_sh == 0) {
    // Don't update states based on amplitudes, in the LZ method
    ;
    ;
  } else {
    dyn_var.update_active_states(1, 0); // 1 - forward; 0 - only active state
  }


  // a relabel is a change of label, not of surface: the trajectory stays on the state that carries it, so
  // there is no hop to accept and no momentum to rescale (the hops proposed below have both) - the acceptance
  // test here could refuse the relabel and return the trajectory to the OTHER surface, and the rescaling by
  // the gap of the two labels changed the total energy at every relabel
  act_states = dyn_var.act_states;
  // Check if rescaling is needed:
  //act_states = dyn_var.act_states; // these are proposed states
  //act_states = accept_hops(dyn_var, ham, act_states, old_states, prms, rnd);

  //====================== Momenta adjustment after successful/frustrated hops
  //===================
  // Velocity rescaling: however here we may be changing velocities
  //act_states = dyn_var.act_states;
  //if (prms.rep_sh == 1) {
      //cout<<"Rescaling (adiabatic): \n";
      //cout<<"initial -> final: "<<old_states[0]<<" -> "<<act_states[0]<<" ";
      //double av_Epot = average_potential_energy(prms, dyn_var, ham);
      //cout<<"Epot (before) = "<<av_Epot<<endl;

      //handle_hops_nuclear(dyn_var, ham, act_states, old_states, prms);
      //cout<<"final state = "<<act_states[0]<<endl;
      //dyn_var.act_states = act_states;
      //cout<<"energies: \n"; ham.children[0]->ham_adi->show_matrix();

      //av_Epot = average_potential_energy(prms, dyn_var, ham);
      //cout<<"Epot (after) = "<<av_Epot<<endl;
  //}



  // For now, this function also accounts for the kinetic energy adjustments to
  // reflect the adiabatic evolution
  if (prms.thermally_corrected_nbra == 1) {
    apply_thermal_correction(dyn_var, ham, ham_aux, old_states, prms, rnd);
  }

  //cout<<"act_states: "; for(int a=0; a<ntraj; a++){ cout<<dyn_var.act_states[a]<<" "; }
  //
  //cout<<"forces before: \n";
  //dyn_var.f->show_matrix();
  //

  // a relabel at a crossing changed the active state before the second half kick - the new label's force must be
  // in its row before the kick reads it
  if (prms.hop_force_refresh == 1) {
    refresh_force_rows(prms, dyn_var, ham, py_funct, params, old_states, dyn_var.act_states);
  }
  // the check names the remedy, so it runs before the forces are rebuilt from the new row
  check_hop_force_rows(prms, dyn_var, ham, old_states, dyn_var.act_states);
  update_forces(prms, dyn_var, ham);

  //cout<<"forces after: \n";
  //dyn_var.f->show_matrix();
  //

  if (prms.decoherence_algo == 6 and prms.use_xf_force == 1) {
    if (prms.rep_tdse == 3) { update_forces_xf_dm(dyn_var, ham, ham_aux); }
    else { update_forces_xf(dyn_var, ham, ham_aux); }
  }

  //  // NVT dynamics
  //  if(prms.ensemble==1){
  //    for(traj=0; traj<ntraj; traj++){
  //      t2[0] = traj;
  //      pop_submatrix(p, p_traj, t1, t2);
  //      double ekin = compute_kinetic_energy(p_traj, invM,
  //      prms.thermostat_dofs); therm[traj].propagate_nhc(prms.dt, ekin, 0.0,
  //      0.0);
  //    }
  //
  //  }
  //
  //  *dyn_var.p = *dyn_var.p + 0.5*prms.dt* (*dyn_var.f);

  // NVT dynamics
  if (prms.ensemble == 1) {
    if (therm[0].thermostat_type == "Nose-Hoover") {
      for (traj = 0; traj < ntraj; traj++) {
        t2[0] = traj;
        pop_submatrix(p, p_traj, t1, t2);
        double ekin =
            compute_kinetic_energy(p_traj, invM, prms.thermostat_dofs);
        therm[traj].propagate_nhc(prms.dt, ekin, 0.0, 0.0);
      } // traj
    } // Nose-Hoover
    else if (therm[0].thermostat_type == "Langevin") {
      for (idof = 0; idof < n_therm_dofs; idof++) {
        dof = prms.thermostat_dofs[idof];
        for (traj = 0; traj < ntraj; traj++) {
          extra_force.set(
              dof, traj,
              0.5 * xi_therm.get(dof, traj) * sigma_therm.get(dof, traj) /
                      invM.get(dof, traj) * sqrt(prms.dt) -
                  0.5 * therm[traj].nu_therm * dyn_var.p->get(dof, traj) *
                      prms.dt -
                  0.25 *
                      (xi_therm.get(dof, traj) / 2.0 +
                       theta_therm.get(dof, traj) / sqrt(3.0)) *
                      therm[traj].nu_therm * sigma_therm.get(dof, traj) /
                      invM.get(dof, traj) * sqrt(pow(prms.dt, 3)) -
                  0.125 * therm[traj].nu_therm *
                      (dyn_var.f->get(dof, traj) -
                       therm[traj].nu_therm * dyn_var.p->get(dof, traj)) *
                      pow(prms.dt, 2));
        } // traj
      } // idof
    } // Langevin
  }

  *dyn_var.p = *dyn_var.p + 0.5 * prms.dt * (*dyn_var.f) + extra_force;
  // if(prms.use_qtsh==1){ *dyn_var.p = *dyn_var.p + 0.5*
  // compute_dkinemat(dyn_var, ham); }
  if (prms.use_qtsh == 1) {
    *dyn_var.p = *dyn_var.p + 0.5 * prms.dt * (*dyn_var.qtsh_f_nc);
  }

  // Kinetic constraint
  for (auto cdof = 0u; cdof < prms.constrained_dofs.size(); cdof++) {
    dyn_var.p->scale(prms.constrained_dofs[cdof], -1, 0.0);
  }

  //  // NVT dynamics
  //  if(prms.ensemble==1){
  //    for(idof=0; idof<n_therm_dofs; idof++){
  //      dof = prms.thermostat_dofs[idof];
  //      for(traj=0; traj<ntraj; traj++){
  //        dyn_var.p->scale(dof, traj, therm[traj].vel_scale(0.5*prms.dt));
  //      }// traj
  //    }// idof
  //  }

  // NVT dynamics
  if (prms.ensemble == 1) {
    if (therm[0].thermostat_type == "Nose-Hoover") {
      for (idof = 0; idof < n_therm_dofs; idof++) {
        dof = prms.thermostat_dofs[idof];
        for (traj = 0; traj < ntraj; traj++) {
          dyn_var.p->scale(dof, traj, therm[traj].vel_scale(0.5 * prms.dt));
        } // traj
      } // idof
    } // Nose-Hoover
  }

  // ham_aux.copy_content(ham);
  update_Hamiltonian_variables(prms, dyn_var, ham, ham_aux, py_funct, params,
                               1);

  // Update amplitudes and density matrices in response to regular evolution
  dyn_var.update_amplitudes(prms);
  dyn_var.update_density_matrix(prms);

  //============== Begin the TSH part ===================
  //exit(0);

  //================= Update decoherence rates & times ================
  /// Effectively turn off decoherence effects
  if (prms.decoherence_times_type == -1) {
    for (traj = 0; traj < ntraj1; traj++) {
      decoherence_rates[traj] = 0.0;
    }
  }
  /// Just use the plain times given from the input, usually the mSDM formalism
  else if (prms.decoherence_times_type == 0) {
    for (traj = 0; traj < ntraj1; traj++) {
      decoherence_rates[traj] = *prms.decoherence_rates;
    }
  }
  /// Compute the dephasing rates according the original energy-based formalism
  else if (prms.decoherence_times_type == 1) {
    Eadi = get_Eadi(ham);
    Ekin = dyn_var.compute_kinetic_energies();
    decoherence_rates = edc_rates(Eadi, Ekin, prms.decoherence_C_param,
                                  prms.decoherence_eps_param, prms.isNBRA);
  }

  else if (prms.decoherence_times_type == 2) {
    // the density-matrix representations (rep_tdse 2 / 3) hand the propagated densities - their amplitudes
    // keep the t0 values
    if (prms.rep_tdse == 2 || prms.rep_tdse == 3) {
      decoherence_rates = schwartz_1_dm(prms, dyn_var.dm_adi, ham, *prms.schwartz_decoherence_inv_alpha);
    } else {
      decoherence_rates = schwartz_1(prms, *dyn_var.ampl_adi, ham,
                                     *prms.schwartz_decoherence_inv_alpha);
    }
  }

  else if (prms.decoherence_times_type == 3) {
    decoherence_rates =
        schwartz_2(prms, ham, *prms.schwartz_decoherence_inv_alpha);
  }

  else if (prms.decoherence_times_type == 4) {
    if (prms.rep_tdse == 2 || prms.rep_tdse == 3) {
      decoherence_rates = schwartz_1_dm(prms, dyn_var.dm_adi, *dyn_var.p, ham, *prms.schwartz_interaction_width);
    } else {
      decoherence_rates = schwartz_1(prms, *dyn_var.ampl_adi, *dyn_var.p, ham,
                                     *prms.schwartz_interaction_width);
    }
  }

  else if (prms.decoherence_times_type == 5) {
    if (prms.rep_tdse == 2 || prms.rep_tdse == 3) {
      decoherence_rates = Gu_Franco_dm(prms, dyn_var.dm_adi);
    } else {
      decoherence_rates = Gu_Franco(prms, *dyn_var.ampl_adi);
    }
  }
  else {
    std::cout<<"Error in compute_dynamics(): decoherence_times_type = "<<prms.decoherence_times_type<<" has no branch here (implemented: -1, 0, 1, 2, 3, 4, 5)\n";
    std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
  }

  ///== Optionally, apply the dephasing-informed correction ==
  if (prms.dephasing_informed == 1) {
    Eadi = get_Eadi(ham);
    MATRIX ave_gaps(*prms.ave_gaps);
    dephasing_informed_correction(decoherence_rates, Eadi, ave_gaps,
                                  prms.isNBRA);
  }

  //============ Apply pre-TSH decoherence corrections ==================

  // SDM and alike methods - only in the adiabatic rep
  if (prms.decoherence_algo == 0) {
    if (prms.rep_tdse == 0 || prms.rep_tdse == 1) {
      *dyn_var.ampl_adi =
          sdm(*dyn_var.ampl_adi, prms.dt, dyn_var.act_states, decoherence_rates,
              prms.sdm_norm_tolerance, prms.isNBRA);
      // rep_tdse 0 derives its adiabatic amplitudes - the corrected ones are written through into the propagated diabatic ones
      if (prms.rep_tdse == 0) { dyn_var.adopt_adiabatic(prms); }
    } else if (prms.rep_tdse == 2 || prms.rep_tdse == 3) {
      sdm_dm(dyn_var.dm_adi, prms.dt, dyn_var.act_states, decoherence_rates, prms.sdm_norm_tolerance, prms.isNBRA);
      // rep_tdse 2 derives its adiabatic density - the corrected one is written through into the propagated diabatic one
      if (prms.rep_tdse == 2) { dyn_var.adopt_adiabatic(prms); }
    } else {
      std::cout<<"Error in compute_dynamics(): rep_tdse = "<<prms.rep_tdse<<" has no branch here (implemented: 0, 1, 2, 3)\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }
  } else if (prms.decoherence_algo ==
             1) { /* ID-A. Nothing to do here, this is done in the surface
                     hopping section */
  }
  // decoherence_algo 2 (A-FSSH) is refused by sanity_check - its application is commented out upstream
  //else if (prms.decoherence_algo ==
  //           2) { /* A-FSSH. Nothing to do here, this is done in the surface
  //                   hopping section */
  //}
  // BCSH
  else if (prms.decoherence_algo == 3) {
    if (prms.rep_tdse == 0 || prms.rep_tdse == 1) {
      wp_reversal_events(dyn_var, ham, prms.dt);
      *dyn_var.ampl_adi = bcsh(*dyn_var.ampl_adi, prms.dt, dyn_var.act_states,
                               *dyn_var.reversal_events);
      // rep_tdse 0 derives its adiabatic amplitudes - the corrected ones are written through into the propagated diabatic ones
      if (prms.rep_tdse == 0) { dyn_var.adopt_adiabatic(prms); }
    } else if (prms.rep_tdse == 2 || prms.rep_tdse == 3) {
      wp_reversal_events(dyn_var, ham, prms.dt);
      bcsh_dm(dyn_var.dm_adi, prms.dt, dyn_var.act_states, *dyn_var.reversal_events);
      // rep_tdse 2 derives its adiabatic density - the corrected one is written through into the propagated diabatic one
      if (prms.rep_tdse == 2) { dyn_var.adopt_adiabatic(prms); }
    } else {
      std::cout<<"Error in compute_dynamics(): rep_tdse = "<<prms.rep_tdse<<" has no branch here (implemented: 0, 1, 2, 3)\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }
  }
  // MFSD
  else if (prms.decoherence_algo == 4) {
    if (prms.rep_tdse == 0 || prms.rep_tdse == 1) {
      p = *dyn_var.p;
      // cout<<"p before mfsd\n"; dyn_var.p->show_matrix();
      *dyn_var.ampl_adi =
          mfsd(p, *dyn_var.ampl_adi, *dyn_var.iM, prms.dt, dyn_var.act_states,
               decoherence_rates, ham, rnd, prms.isNBRA);
      *dyn_var.p = p;
      // rep_tdse 0 derives its adiabatic amplitudes - the corrected ones are written through into the propagated diabatic ones
      if (prms.rep_tdse == 0) { dyn_var.adopt_adiabatic(prms); }
      // cout<<"p after mfsd\n"; dyn_var.p->show_matrix();

      // Recompute NAC, Hvib, etc. in response to change of p
      update_Hamiltonian_variables(prms, dyn_var, ham, ham_aux, py_funct,
                                   params, 1);
    } else if (prms.rep_tdse == 2 || prms.rep_tdse == 3) {
      p = *dyn_var.p;
      mfsd_dm(p, dyn_var.dm_adi, *dyn_var.iM, prms.dt, dyn_var.act_states, decoherence_rates, ham, rnd, prms.isNBRA);
      *dyn_var.p = p;
      // rep_tdse 2 derives its adiabatic density - the corrected one is written through into the propagated diabatic one
      if (prms.rep_tdse == 2) { dyn_var.adopt_adiabatic(prms); }
      update_Hamiltonian_variables(prms, dyn_var, ham, ham_aux, py_funct, params, 1);
    } else {
      std::cout<<"Error in compute_dynamics(): rep_tdse = "<<prms.rep_tdse<<" has no branch here (implemented: 0, 1, 2, 3)\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }
  }
  // SHXF
  else if (prms.decoherence_algo == 5) {
    if (prms.rep_tdse == 1) {
      shxf(dyn_var, ham, ham_aux, prms);
    } else if (prms.rep_tdse == 3) {
      shxf_dm(dyn_var, ham, ham_aux, prms);
    } else {
      std::cout<<"Error in compute_dynamics(): rep_tdse = "<<prms.rep_tdse<<" has no branch here (implemented: 1, 3 - the XF half step is interleaved with the adiabatic propagation, sanity_check refuses 0 and 2)\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }
  }
  // MQCXF
  else if (prms.decoherence_algo == 6) {
    if (prms.rep_tdse == 1) {
      mqcxf(dyn_var, ham, ham_aux, prms);
    } else if (prms.rep_tdse == 3) {
      mqcxf_dm(dyn_var, ham, ham_aux, prms);
    } else {
      std::cout<<"Error in compute_dynamics(): rep_tdse = "<<prms.rep_tdse<<" has no branch here (implemented: 1, 3 - the XF half step is interleaved with the adiabatic propagation, sanity_check refuses 0 and 2)\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }
  }

  // DISH, rev2023
  else if (prms.decoherence_algo == 7) {
    // the density representations act on the adiabatic density, the amplitude ones on the adiabatic amplitudes -
    // the diabatic ones (0, 2) write the correction through into their propagated variable
    if (prms.rep_tdse == 2 || prms.rep_tdse == 3) {
      dish_rev2023_dm(dyn_var, ham, decoherence_rates, prms, rnd);
      if (prms.rep_tdse == 2) { dyn_var.adopt_adiabatic(prms); }
    } else {
      dish_rev2023(dyn_var, ham, decoherence_rates, prms, rnd);
      if (prms.rep_tdse == 0) { dyn_var.adopt_adiabatic(prms); }
    }
  }


  // Simple decoherence
  else if (prms.decoherence_algo == 9) {

    double alpha = prms.dt/prms.gap_correlation_time; 

    for (traj = 0; traj < ntraj; traj++) {
      

      // Compute the gaps and running averages of the gaps and their squares
      for (i = 0; i < nadi; i++) {
        for (j = 0; j < nadi; j++) {
            // dE_ij = E_i - E_j
            // the shared NBRA child serves every trajectory
            double E_i = ham.child(traj)->hvib_adi->get(i,i).real();
            double E_j = ham.child(traj)->hvib_adi->get(j,j).real();
            double dE_ij = E_i - E_j;
            dyn_var.gaps_curr[traj][i][j] = dE_ij;
            dyn_var.coherence_clocks[traj][i][j] += prms.dt;

            // Running averages
            if(dyn_var.is_first_gap[traj][i][j] == 1){
              dyn_var.mean_gap[traj][i][j] = dE_ij;
              dyn_var.mean_gap2[traj][i][j] = dE_ij * dE_ij;

              dyn_var.is_first_gap[traj][i][j] = 0;
              dyn_var.coherence_clocks[traj][i][j] = 0.0;

              dyn_var.gap_fluctuations[traj][i][j]  = dyn_var.gap_fluctuations_prev[traj][i][j] = 0.0;
              dyn_var.gap_correlations[traj][i][j] = 0.0;

              dyn_var.averaging_steps[traj][i][j] = 1.0;
            }
            else{
              dyn_var.averaging_steps[traj][i][j] += 1.0;
              alpha = 1.0/dyn_var.averaging_steps[traj][i][j];

              dyn_var.mean_gap[traj][i][j] = (1.0 - alpha) * dyn_var.mean_gap[traj][i][j] + alpha * dE_ij;
              dyn_var.mean_gap2[traj][i][j] = (1.0 - alpha) * dyn_var.mean_gap2[traj][i][j] + alpha * (dE_ij * dE_ij);

              dyn_var.gap_fluctuations[traj][i][j] = dyn_var.gaps_curr[traj][i][j] - dyn_var.mean_gap[traj][i][j];
              //dyn_var.mean_gap2[traj][i][j] - dyn_var.mean_gap[traj][i][j] * dyn_var.mean_gap[traj][i][j];
               
              //dyn_var.gap_correlations[traj][i][j] = (1.0 - alpha) * dyn_var.gap_correlations[traj][i][j] + alpha * (dE_ij * dyn_var.gaps_prev[traj][i][j]);
              dyn_var.gap_correlations[traj][i][j] = (1.0 - alpha) * dyn_var.gap_correlations[traj][i][j] + 
                                                     alpha * dyn_var.gap_fluctuations[traj][i][j] * dyn_var.gap_fluctuations_prev[traj][i][j];

              
            }

        }// for j
      }// for i
  
      // Compute rates and decoherence correction
      for (i = 0; i < nadi; i++) {
        for (j = 0; j < nadi; j++) {

          // Compute decoherence rates 
          double mean = dyn_var.mean_gap[traj][i][j];
          double mean2 = dyn_var.mean_gap2[traj][i][j];

          double var = mean2 - mean * mean;
          if(var < 0.0){ var = 0.0; }

          double sigma = sqrt(var); // estimate of the decoherence rate!

          double argg = sigma * prms.dt;
          double pw = 2.0; // 1 - exponential;  2 - Gaussian
          dyn_var.coherence_factors[traj][i][j] *= exp( - pow(argg, pw)/pw);

          // Make current old:
          dyn_var.gaps_prev[traj][i][j] = dyn_var.gaps_curr[traj][i][j];
          dyn_var.gap_fluctuations_prev[traj][i][j] = dyn_var.gap_fluctuations[traj][i][j];;
           
        } // for j
      } // for i

      //cout<<"coherence factors: "<<dyn_var.coherence_factors[traj][0][1]<<"  "<<dyn_var.coherence_factors[traj][1][0]<<endl;
      double err = dyn_var.coherence_factors[traj][0][1] - dyn_var.coherence_factors[traj][1][0];
      if(fabs(err)>1e-10){ cout<<"Inconsistent coherence factors "<<err<<endl; }

    } // for traj
  } // simple decoherence
  // -1: no decoherence - 8 (diabatic IDA) acts after the hops
  else if (prms.decoherence_algo == -1 || prms.decoherence_algo == 8) { ; }
  else {
    std::cout<<"Error in compute_dynamics(): decoherence_algo = "<<prms.decoherence_algo<<" has no branch here (implemented: -1, 0, 1, 3, 4, 5, 6, 7, 8, 9)\n";
    std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
  }

  // Update amplitudes and density matrices in response to decoherence
  // corrections
  dyn_var.update_amplitudes(prms);
  dyn_var.update_density_matrix(prms);



  //************************************ TSH options
  //****************************************
  // Adiabatic dynamics
  if (prms.tsh_method == -1) {
    ;
    ;
  }

  // FSSH, GFSH, MSSH, LZ, ZN, DISH, MASH, FSSH2, FSSH3
  else if (prms.tsh_method == 0 || prms.tsh_method == 1 ||
           prms.tsh_method == 2 || prms.tsh_method == 3 ||
           prms.tsh_method == 4 || prms.tsh_method == 5 ||
           prms.tsh_method == 6 || prms.tsh_method == 7 ||
           prms.tsh_method == 8 || prms.tsh_method == 9) {

    vector<int> old_states(dyn_var.act_states);
    vector<int> old_states_dia(dyn_var.act_states_dia);
    vector<int> prop_states(ntraj, 0);
    //========================== Hop proposal and acceptance
    //================================

    // FSSH (0), GFSH (1), MSSH (2), LZ(3), ZN (4), MASH(6), FSSH2(7), FSSH3(8),
    // GFSH original (9)
    if (prms.tsh_method == 0 || prms.tsh_method == 1 || prms.tsh_method == 2 ||
        prms.tsh_method == 3 || prms.tsh_method == 4 || prms.tsh_method == 6 ||
        prms.tsh_method == 7 || prms.tsh_method == 8 || prms.tsh_method == 9) {

      /// Compute hop proposal probabilities from the active state of each
      /// trajectory to all other states of that trajectory
      vector<vector<double>> g;
      g = hop_proposal_probabilities(prms, dyn_var, ham, ham_aux);

      // simple decoherence correction
      if (prms.decoherence_algo ==
          9) { // intended only for adiabatic states for now
        renormalize_hopping_probabilities(g, dyn_var.coherence_factors,
                                          dyn_var.act_states);
      }

      // Propose new discrete states for all trajectories
      if (prms.rep_sh == 1) {
        prop_states = propose_hops(g, dyn_var.act_states, rnd);

        // Decide if to accept the transitions (and then which)
        act_states = accept_hops(dyn_var, ham, prop_states, dyn_var.act_states,
                                 prms, rnd);
      } else if (prms.rep_sh == 0) {
        prop_states = propose_hops(g, dyn_var.act_states_dia, rnd);
        act_states_dia = accept_hops(dyn_var, ham, prop_states,
                                     dyn_var.act_states_dia, prms, rnd);
      }
      else {
        std::cout<<"Error in compute_dynamics(): rep_sh = "<<prms.rep_sh<<" has no branch here (implemented: 0, 1)\n";
        std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
      }

      //=== Post-hop decoherence options ===

      // Instantaneous decoherence
      if (prms.decoherence_algo == 1) {
        if (prms.rep_tdse == 0 || prms.rep_tdse == 1) {
          instantaneous_decoherence(
              *dyn_var.ampl_adi, act_states, prop_states, old_states,
              prms.instantaneous_decoherence_variant, prms.collapse_option);
          if (prms.rep_tdse == 0) { dyn_var.adopt_adiabatic(prms); }
        } else if (prms.rep_tdse == 2 || prms.rep_tdse == 3) {
          instantaneous_decoherence_dm(dyn_var.dm_adi, act_states, prop_states, old_states,
                                       prms.instantaneous_decoherence_variant, prms.collapse_option);
          if (prms.rep_tdse == 2) { dyn_var.adopt_adiabatic(prms); }
        } else {
        std::cout<<"Error in compute_dynamics(): rep_tdse = "<<prms.rep_tdse<<" has no branch here (implemented: 0, 1, 2, 3)\n";
          std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
        }
      }
      // decoherence_algo 2 (A-FSSH) is refused by sanity_check - its application is commented out upstream
      //else if (prms.decoherence_algo == 2) {
      //  /// Temporarily commented AVA 11/7/2022
      //  /// apply_afssh(dyn_var, Coeff, act_states, invM, ham, dyn_params, rnd);
      //} // AFSSH
      else if (prms.decoherence_algo == 3) {
        ;
        ;
      } // BCSH of Linjun Wang, nothing to do here
      else if (prms.decoherence_algo == 4) {
        ;
        ;
      } // MF-SD of Bedard-Hearn, Larsen, Schwartz, nothing to do here
      else if (prms.decoherence_algo == 5 or prms.decoherence_algo == 6) {
        if (prms.rep_sh == 1) {
          xf_hop_reset(dyn_var, act_states, old_states);
        } else {
          cout << "ERROR: Independent-trajectory XF methods require rep_sh = "
                  "1\n";
          std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
        }
      } // SHXF or MQCXF
      else if (prms.decoherence_algo == 7) {
        ;
        ;
      } // DISH rev 2023, nothing to do here

      // the instantaneous decoherence in the diabatic basis - for the diabatic hopping (rep_sh 0, sanity_check
      // pairs them): the diabatic lists (the accepted act_states_dia, the proposals, the initial act_states_dia)
      // and the diabatic variable - the amplitude representations act on ampl_dia (derived under rep_tdse 1, the
      // correction written through into the propagated ampl_adi), the density ones on dm_dia (derived under 3)
      else if (prms.decoherence_algo == 8) {
        if (prms.rep_tdse == 0 || prms.rep_tdse == 1) {
          instantaneous_decoherence_dia(
              *dyn_var.ampl_dia, ham, act_states_dia, prop_states, old_states_dia,
              prms.instantaneous_decoherence_variant, prms.collapse_option);
          if (prms.rep_tdse == 1) { dyn_var.adopt_diabatic(prms); }
        } else if (prms.rep_tdse == 2 || prms.rep_tdse == 3) {
          instantaneous_decoherence_dia_dm(dyn_var.dm_dia, ham, act_states_dia, prop_states, old_states_dia,
                                           prms.instantaneous_decoherence_variant, prms.collapse_option);
          if (prms.rep_tdse == 3) { dyn_var.adopt_diabatic(prms); }
        } else {
        std::cout<<"Error in compute_dynamics(): rep_tdse = "<<prms.rep_tdse<<" has no branch here (implemented: 0, 1, 2, 3)\n";
          std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
        }
      } // algo == 8

      else if (prms.decoherence_algo == 9) { // simple decoherence method 
        reset_coherence_factors(dyn_var, act_states, old_states, prms.gap_correlation_time);
      } // algo == 9
      // -1: no decoherence - 0 (MSDM / EDC) acts during the propagation
      else if (prms.decoherence_algo == -1 || prms.decoherence_algo == 0) { ; }
      else {
        std::cout<<"Error in compute_dynamics(): decoherence_algo = "<<prms.decoherence_algo<<" has no branch here (implemented: -1, 0, 1, 3, 4, 5, 6, 7, 8, 9)\n";
        std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
      }

    }
    // DISH - the old one
    else if (prms.tsh_method == 5) {
      if (prms.decoherence_algo == -1) {
        ;
        ;
      } else {
        cout << "ERROR: DISH method should be used only with decoherence_algo "
                "= -1\n";
        std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
      }
      // the density representations collapse and project the adiabatic density, the amplitude ones the adiabatic
      // amplitudes - the diabatic ones (0, 2) write the result through into their propagated variable
      if (prms.rep_tdse == 2 || prms.rep_tdse == 3) {
        act_states = dish_dm(dyn_var, ham, decoherence_rates, prms, rnd);
        if (prms.rep_tdse == 2) { dyn_var.adopt_adiabatic(prms); }
      } else {
        act_states = dish(dyn_var, ham, decoherence_rates, prms, rnd);
        if (prms.rep_tdse == 0) { dyn_var.adopt_adiabatic(prms); }
      }
      // the proposals DISH refused stay inside dish - no frustrated hop reaches the rescaling
      prop_states = act_states;
    } // DISH

    //====================== Momenta adjustment after successful/frustrated hops
    //===================
    // Velocity rescaling: however here we may be changing velocities
    if (prms.rep_sh == 1) {
      //cout<<"In TSH rescaling:\n";
      //cout<<"initial -> final state: "<<old_states[0]<<" ->"<<act_states[0];
      //double av_Epot = average_potential_energy(prms, dyn_var, ham);
      //cout<<"Epot (before) = "<<av_Epot<<endl;

      handle_hops_nuclear(dyn_var, ham, act_states, old_states, prop_states, prms);
      //cout<<" active state after: "<<act_states[0]<<endl;
      dyn_var.act_states = act_states;

      //av_Epot = average_potential_energy(prms, dyn_var, ham);
      //cout<<"Epot (after) = "<<av_Epot<<endl;

    } else {
      handle_hops_nuclear(dyn_var, ham, act_states_dia, old_states_dia, prop_states, prms);
      dyn_var.act_states_dia = act_states_dia;
    }

    // Set the active states in the other representation through the
    // tranformation matrix
    if (need_active_states_diff_rep(prms) == 1) {
      dyn_var.set_active_states_diff_rep(prms.rep_sh, rnd);
    }

    // Re-scale (back) couplings and time-overlaps, if the TC-NBRA was used
    if (prms.thermally_corrected_nbra == 1 && prms.tcnbra_do_nac_scaling == 1) {
      remove_thermal_correction(dyn_var, ham, prms);
    }

    // Update vib Hamiltonian to reflect the change of the momentum
    update_Hamiltonian_variables(prms, dyn_var, ham, ham_aux, py_funct, params,
                                 1);

  } // tsh_method == 0, 1, 2, 3, 4, 5, 6, 7, 8, 9

  else {
    cout << "tsh_method == " << prms.tsh_method
         << " is undefined.\n";
    std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
  }

  // Update the amplitudes and DM, in response to state hopping and other
  // changes in the TSH part so that we have them consistent in the output
  dyn_var.update_amplitudes(prms);
  dyn_var.update_density_matrix(prms);

  // a hop changed the row the forces are read from - under hop_force_refresh the
  // model is called back for the new row here, as it is for a relabel before the
  // second half kick (one owner: compute_dynamics)
  if (prms.hop_force_refresh == 1) {
    refresh_force_rows(prms, dyn_var, ham, py_funct, params, old_states, act_states);
  }
  // old_states holds the active states at the start of the step - a relabel at a
  // crossing and a surface hop both change the row the forces are read from - the
  // check names the remedy, so it runs before the forces are rebuilt from that row
  check_hop_force_rows(prms, dyn_var, ham, old_states, act_states);
  update_forces(prms, dyn_var, ham);

  // Saves the current density matrix into the previous - needed for FSSH2 and
  // GFSH (original)
  dyn_var.save_curr_dm_into_prev();

  // NBRA runs (isNBRA==1) size decoherence_rates with a single shared entry
  int nrates = decoherence_rates.size();
  for (traj = 0; traj < ntraj; traj++) {
    *dyn_var.ave_decoherence_rates += decoherence_rates[(traj < nrates) ? traj : 0];
  }
  *dyn_var.ave_decoherence_rates /= ntraj;


}

} // namespace libdyn
} // namespace liblibra
