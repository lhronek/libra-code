/*********************************************************************************
 * Copyright (C) 2017-2023 Alexey V. Akimov
 *
 * This file is distributed under the terms of the GNU General Public License
 * as published by the Free Software Foundation, either version 3 of
 * the License, or (at your option) any later version.
 * See the file LICENSE in the root directory of this distribution
 * or <http://www.gnu.org/licenses/>.
 *
 *********************************************************************************/
/**
  \file nHamiltonian_compute_QTSH_forces.cpp
  \brief The file implements the calculations of various variants of the QTSH
  forces
*/

#include <stdexcept>
#if defined(USING_PCH)
#include "../pch.h"
#else
#include <stdlib.h>
#endif

#include "../math_meigen/libmeigen.h"
#include "nHamiltonian.h"

/// liblibra namespace
namespace liblibra {

/// libnhamiltonian namespace
namespace libnhamiltonian {

using namespace liblinalg;
using namespace libmeigen;

CMATRIX nHamiltonian::QTSH_forces_dia_unit(CMATRIX &ampl_dia, int option) {
  /**
    \param[in] ampl_dia: MATRIX(ndia, 1) diabatic amplitudes for one trajectory

    Returns:
    MATRIX(ndof, 1) - QTSH off-diagonal Hellmann-Feynman forces in diabatic
    representation, for a single trajectory

    The off-diagonal Hellmann-Feynman force in QTSH has the form of the
    off-diagonal terms in the Ehrenfest force.

  */

  // an unlisted option would leave res at zero - a silent zero force
  if (!(option == 0 || option == 1)) {
    cout << "Error in QTSH_forces_dia_unit(): option = " << option
         << " is not allowed (0 - first-order off-diagonal force, 1 - with the second-order term)\n";
    std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
  }
  if (ovlp_dia_mem_status == 0) {
    cout
        << "Error in QTSH_forces_dia_unit(): the overlap matrix in the diabatic basis is not allocated \
  but it is needed for the calculations\n";
    std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
  }

  if (ham_dia_mem_status == 0) {
    cout
        << "Error in QTSH_forces_dia_unit(): the diabatic Hamiltonian matrix is not allocated \
  but it is needed for the calculations\n";
    std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
  }

  CMATRIX res(nnucl, 1);
  CMATRIX X(ndia, ndia);
  CMATRIX Y(ndia, ndia);
  if (lowdin_dia(X, Y)) {
    // the Lowdin frame of the diabats, the frame of the QTSH state energy and coherence energy
    // (get_ham_dia_lowdin, QTSH_energy_dia with the momenta): -d^H offdiag(dH_L) d / (d^H d) on
    // d = Y ampl_dia, dH_L the total derivative of X ham_dia X along the DOF (get_d1ham_dia_lowdin,
    // the frame's own motion inside it) for option 0 and the Hamiltonian's derivative in the frame,
    // X d1ham_dia X, for option 1 - the frame terms the raw-frame expressions below carry through
    // S^-1 and dc1_dia
    CMATRIX d(Y * ampl_dia);
    complex<double> norm_l = (d.H() * d).M[0];
    for (int n = 0; n < nnucl; n++) {
      if (d1ham_dia_mem_status[n] == 0) {
        cout << "Error in QTSH_forces_dia_unit(): the derivatives of the Hamiltonian matrix in the \
      diabatic basis w.r.t. the nuclear DOF " << n << " is not allocated but is needed for the calculations \n";
        std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
      }
      CMATRIX dH((option == 0) ? get_d1ham_dia_lowdin(n) : X * (*d1ham_dia[n]) * X);
      for (int i = 0; i < ndia; i++) { dH.set(i, i, complex<double>(0.0, 0.0)); }
      res.M[n] = -(d.H() * dH * d).M[0];
    }
    res /= norm_l;
    return res;
  }
  CMATRIX *dtilda;
  dtilda = new CMATRIX(ndia, ndia);
  CMATRIX *invS;
  invS = new CMATRIX(ndia, ndia);

  FullPivLU_inverse(*ovlp_dia, *invS);

  complex<double> norm = (ampl_dia.H() * (*ovlp_dia) * ampl_dia).M[0];

  CMATRIX *temp_diag = new CMATRIX(ndia, ndia);

  for (int n = 0; n < nnucl; n++) {

    if (d1ham_dia_mem_status[n] == 0) {
      cout
          << "Error in QTSH_forces_dia_unit(): the derivatives of the Hamiltonian matrix in the \
      diabatic basis w.r.t. the nuclear DOF "
          << n << " is not allocated but is needed for the calculations \n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }

    if (dc1_dia_mem_status[n] == 0) {
      cout
          << "Error in QTSH_forces_dia_unit(): the derivatives couplings matrix in the diabatic \
      basis w.r.t. the nuclear DOF "
          << n << " is not allocated but is needed for the calculations \n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }

    if (option == 0) {
      *dtilda = ((*dc1_dia[n]).H() * (*invS) * (*ham_dia) +
                 (*ham_dia) * (*invS) * (*dc1_dia[n]));

      for (int i = 0; i < ndia; i++) {
        temp_diag->set(i, i, (*d1ham_dia[n] - *dtilda).get(i, i));
      }
      res.M[n] =
          -(ampl_dia.H() * (*d1ham_dia[n] - *dtilda - *temp_diag) * ampl_dia)
               .M[0];
    } else if (option == 1) {
      for (int i = 0; i < ndia; i++) {
        temp_diag->set(i, i, d1ham_dia[n]->get(i, i));
      }
      res.M[n] = -(ampl_dia.H() * (*d1ham_dia[n] - *temp_diag) * ampl_dia).M[0];
    }

  } // for n

  res /= norm;

  delete dtilda;
  delete invS;
  delete temp_diag;

  return res;
}

CMATRIX nHamiltonian::QTSH_forces_dia_unit(CMATRIX &ampl_dia) {
  return QTSH_forces_dia_unit(ampl_dia, 0);
}

CMATRIX nHamiltonian::QTSH_forces_dia(CMATRIX &ampl_dia, int lvl, int option) {
  /**
    \brief Computes the QTSH forces in the diabatic basis

    \param[in] ampl_dia [ndia x ntraj] matrix of diabatic amplitudes for
    one or many (ntraj) trajectories
    \param[in] lvl [0 or 1] - 0 - use the present level for all trajectories, 1
    - use the next level for the trajectories (one sub-Hamiltonian per each
    trajectory)

    There are 2 possible use cases:
    a) lvl = 0 ampl_dia is a ndia x ntraj matrix and is meant to
    be handled by the current Hamiltonian (e.g. like the NBRA)

    b) lvl = 1 ampl_adi is a ndia x ntraj matrix (ntraj trajectories) and
    each trajectory is meant to be handled by a separate sub-Hamiltonian

    Returns:
    MATRIX(ndof, ntraj) - QTSH forces in diabatic representation, for multiple
    trajectories


  */
  int i;

  if (lvl == 1) {
    // Check whether we have enough sub-Hamiltonians
    if (children.size() != static_cast<std::size_t>(ampl_dia.n_cols)) {
      cout << "ERROR in CMATRIX nHamiltonian::QTSH_forces_dia(CMATRIX& "
              "ampl_dia):\n";
      cout << "The number of columns of the ampl_dia (" << ampl_dia.n_cols
           << ")";
      cout << " should be equal to the number of children Hamiltonians ("
           << children.size() << ")\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }
  }

  CMATRIX ampl_tmp(ampl_dia.n_rows, 1);
  CMATRIX frc_tmp(nnucl, 1);
  CMATRIX F(nnucl, ampl_dia.n_cols);

  vector<int> stenc_ampl(ampl_dia.n_rows, 0);
  vector<int> stenc_frc(nnucl, 0);
  vector<int> stenc_col(1, 0);

  // The indicies in stenc_frc reflects which indicies are to be updated in the
  // final F matrix
  for (i = 0; i < nnucl; i++) {
    stenc_frc[i] = i;
  }

  for (i = 0; i < ampl_dia.n_rows; i++) {
    stenc_ampl[i] = i;
  }

  for (i = 0; i < ampl_dia.n_cols; i++) {
    stenc_col[0] = i;

    pop_submatrix(ampl_dia, ampl_tmp, stenc_ampl, stenc_col);

    if (lvl == 0) {
      frc_tmp = QTSH_forces_dia_unit(ampl_tmp, option);
    }
    if (lvl == 1) {
      frc_tmp = children[i]->QTSH_forces_dia_unit(ampl_tmp, option);
    }

    push_submatrix(F, frc_tmp, stenc_frc, stenc_col);

  } // for all children

  return F;
}

CMATRIX nHamiltonian::QTSH_forces_dia(CMATRIX &ampl_dia, int lvl) {
  return QTSH_forces_dia(ampl_dia, lvl, 0);
}

/*
CMATRIX nHamiltonian::QTSH_forces_dia(CMATRIX& ampl_dia, vector<int>& id_){
//
//  See the description of the QTSH_forces_dia(CMATRIX& ampl_dia) function
//
  if(id_.size()==1){
    if(id_[0]==id){   return QTSH_forces_dia(ampl_dia);    }
    else{ cout<<"ERROR in QTSH_forces_dia: No Hamiltonian matching the requested
id\n"; std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above"); }
  }
  else{
    vector<int> next(id_.begin()+1,id_.end());
    return children[id_[1]]->QTSH_forces_dia(ampl_dia, next);
  }
}
*/

CMATRIX nHamiltonian::QTSH_forces_adi_unit(CMATRIX &ampl_adi, int option) {
  /**
    \param[in] ampl_adi: MATRIX(nadi, 1) diabatic amplitudes for one trajectory

    \params[in] option [0 or 1] - option 0 keeps all the terms in the
    Ehrenfest-like force expression, including NAC option 1 removes all the
    derivative NACs - this is to enforce the local diabatization approximation,
    to be consistent with it

    The off-diagonal Hellmann-Feynman force in QTSH has the form of the
    off-diagonal terms in the Ehrenfest force.

    Returns:
    MATRIX(ndof, 1) - QTSH forces in adiabatic representation, for single
    trajectory


  */

  // an unlisted option would leave res at zero - a silent zero force
  if (!(option == 0 || option == 1 || option == 2)) {
    cout << "Error in QTSH_forces_adi_unit(): option = " << option
         << " is not allowed (0 - first-order off-diagonal force, 1 - with the second-order term, 2 - no derivative-coupling force)\n";
    std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
  }
  if (ham_adi_mem_status == 0) {
    cout
        << "Error in QTSH_forces_adi(): the adiabatic Hamiltonian matrix is not allocated \
  but it is needed for the calculations\n";
    std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
  }

  complex<double> norm = (ampl_adi.H() * ampl_adi).M[0];

  CMATRIX res(nnucl, 1);
  CMATRIX tmp(nadi, nadi);

  //  CMATRIX& T = transform;
  // the amplitudes arrive in the basis of ham_adi - the step's projector moved them in the propagation, no transform here

  for (int n = 0; n < nnucl; n++) {

    if (d1ham_adi_mem_status[n] == 0) {
      cout
          << "Error in QTSH_forces_adi_unit(): the derivatives of the Hamiltonian matrix in the \
    adiabatic basis w.r.t. the nuclear DOF "
          << n << " is not allocated but is needed for the calculations \n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }

    if (dc1_adi_mem_status[n] == 0) {
      cout
          << "Error in QTSH_forces_adi_unit(): the derivatives couplings matrix in the adiabatic \
    basis w.r.t. the nuclear DOF "
          << n << " is not allocated but is needed for the calculations \n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }

    /**  Ehrenfest force for reference:

        if(option==0){ // Original formulation with NACs - for non-LD
       integrators tmp = (T.H() * (*dc1_adi[n]) * T ).H() *  (T.H() * (*ham_adi)
       * T); tmp = tmp + tmp.H(); res.M[n] = -( ampl_adi.H() * ( T.H() *
       (*d1ham_adi[n]) * T - tmp ) * ampl_adi ).M[0];
        }
        else if(option==1){ // Options that disregard the NACs - appropriate for
       the LD integrators res.M[n] = -( ampl_adi.H() * T.H() * ( *d1ham_adi[n] )
       * T * ampl_adi ).M[0];
        }

    */

    if (option ==
        0) { // Original formulation with NACs - for non-LD integrators

      tmp = (*dc1_adi[n]).H() * (*ham_adi);
      tmp = tmp + tmp.H();

      res.M[n] = (ampl_adi.H() * tmp * ampl_adi).M[0];

      // for(int i=0; i<nadi; i++){
      //   for(int j=0; j<nadi; j++){
      //     if(i<=j){continue;}
      //
      //     res.M[n] += 2.0* (T.H() * (*dc1_adi[n]) * T).get(i,j).real() *
      //     ((*ham_adi).get(i,i) - (*ham_adi).get(j,j)).real() *
      //       (T * ampl_adi * ampl_adi.H() * T.H() ).get(i,j).real();
      //   }
      // }
    } else if (option == 1) { // Including the 2nd order force

      tmp = (*dc1_adi[n]).H() * (*hvib_adi);
      tmp = tmp + tmp.H();

      res.M[n] = (ampl_adi.H() * tmp * ampl_adi).M[0];
    } else if (option == 2) { // Options that disregard the NACs - appropriate
                              // for the LD integrators
      res.M[n] = 0.0;
    }

  } // for n

  res /= norm;
  // delete tmp;

  return res;
}


CMATRIX nHamiltonian::QTSH_forces_adi_unit(CMATRIX &ampl_adi) {
  return QTSH_forces_adi_unit(ampl_adi, 0);
}

CMATRIX nHamiltonian::QTSH_forces_adi(CMATRIX &ampl_adi, int lvl, int option) {
  /**
    \brief Computes the QTSH forces in the adiabatic basis

    \param[in] ampl_adi [nadi x ntraj] matrix of adiabatic amplitudes for
    one of many (ntraj) trajectories
    \param[in] lvl [0 or 1] - 0 - use the present level for all trajectories, 1
    - use the next level for the trajectories (one sub-Hamiltonian per each
    trajectory)

    There are 2 possible use cases:
    a) lvl = 0 ampl_adi is a nadi x ntraj matrix and is meant to
    be handled by the current Hamiltonian (e.g. like the NBRA)

    b) lvl = 1 ampl_adi is a nadi x ntraj matrix (ntraj trajectories) and
    each trajectory is meant to be handled by a separate sub-Hamiltonian

    \params[in] option [0 or 1] - option 0 keeps all the terms in the QTSH force
    expression, including NAC option 1 removes all the derivative NACs - this is
    to enforce the local diabatization approximation, to be consistent with it

    Returns:
    MATRIX(ndof, ntraj) - QTSH forces in adiabatic representation, for multiple
    trajectories

  */
  int i;

  if (lvl == 1) {
    // Check whether we have enough sub-Hamiltonians
    if (children.size() != static_cast<std::size_t>(ampl_adi.n_cols)) {
      cout << "ERROR in CMATRIX nHamiltonian::QTSH_forces_adi(CMATRIX& "
              "ampl_adi):\n";
      cout << "The number of columns of the ampl_adi (" << ampl_adi.n_cols
           << ")";
      cout << " should be equal to the number of children Hamiltonians ("
           << children.size() << ")\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }
  }

  CMATRIX ampl_tmp(ampl_adi.n_rows, 1);
  CMATRIX frc_tmp(nnucl, 1);
  CMATRIX F(nnucl, ampl_adi.n_cols);

  vector<int> stenc_ampl(ampl_adi.n_rows, 0);
  vector<int> stenc_frc(nnucl, 0);
  vector<int> stenc_col(1, 0);

  // The indicies in stenc_frc reflects which indicies are to be updated in the
  // final F matrix
  for (i = 0; i < nnucl; i++) {
    stenc_frc[i] = i;
  }

  for (i = 0; i < ampl_adi.n_rows; i++) {
    stenc_ampl[i] = i;
  }

  for (i = 0; i < ampl_adi.n_cols; i++) {
    stenc_col[0] = i;

    pop_submatrix(ampl_adi, ampl_tmp, stenc_ampl, stenc_col);

    if (lvl == 0) {
      frc_tmp = QTSH_forces_adi_unit(ampl_tmp, option);
    }
    if (lvl == 1) {
      frc_tmp = children[i]->QTSH_forces_adi_unit(ampl_tmp, option);
    }

    push_submatrix(F, frc_tmp, stenc_frc, stenc_col);

  } // for all children

  return F;
}


CMATRIX nHamiltonian::QTSH_forces_adi(CMATRIX &ampl_adi, int lvl) {
  return QTSH_forces_adi(ampl_adi, lvl, 0);
}

/*
CMATRIX nHamiltonian::QTSH_forces_adi(CMATRIX& ampl_adi, vector<int>& id_){
//
//  See the description of the QTSH_forces_adi(CMATRIX& ampl_adi) function
//
  if(id_.size()==1){
    if(id_[0]==id){   return QTSH_forces_adi(ampl_adi);    }
    else{ cout<<"ERROR in QTSH_forces_adi: No Hamiltonian matching the requested
id\n"; std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above"); }
  }
  else{
    vector<int> next(id_.begin()+1,id_.end());
    return children[id_[1]]->QTSH_forces_adi(ampl_adi, next);
  }
}
*/


CMATRIX nHamiltonian::QTSH_forces_dia_unit_dm(CMATRIX &dm_dia, int option) {
  /**
    The density-matrix form of QTSH_forces_dia_unit: every C^H A C of the amplitude form is tr(A rho) for the
    plain-coefficient density rho = C_dia C_dia^H of one trajectory (dyn_variables::dm_dia), the Lowdin amplitudes
    d = Y C give the Lowdin density Y rho Y^H - equal to the amplitude form for a pure state
  */
  if (!(option == 0 || option == 1)) {
    cout << "Error in QTSH_forces_dia_unit_dm(): option = " << option
         << " is not allowed (0 - first-order off-diagonal force, 1 - with the second-order term)\n";
    std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
  }
  if (ovlp_dia_mem_status == 0) {
    cout << "Error in QTSH_forces_dia_unit_dm(): the overlap matrix in the diabatic basis is not allocated \
  but it is needed for the calculations\n";
    std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
  }
  if (ham_dia_mem_status == 0) {
    cout << "Error in QTSH_forces_dia_unit_dm(): the diabatic Hamiltonian matrix is not allocated \
  but it is needed for the calculations\n";
    std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
  }
  CMATRIX res(nnucl, 1);
  CMATRIX X(ndia, ndia);
  CMATRIX Y(ndia, ndia);
  if (lowdin_dia(X, Y)) {
    // the Lowdin frame of the diabats (QTSH_forces_dia_unit): -tr(offdiag(dH_L) rho_L) / tr(rho_L) on
    // rho_L = Y rho Y^H, dH_L the total derivative of X ham_dia X along the DOF for option 0 and X d1ham_dia X
    // for option 1
    CMATRIX rho_l(Y * dm_dia * Y.H());
    complex<double> norm_l = rho_l.tr();
    for (int n = 0; n < nnucl; n++) {
      if (d1ham_dia_mem_status[n] == 0) {
        cout << "Error in QTSH_forces_dia_unit_dm(): the derivatives of the Hamiltonian matrix in the \
      diabatic basis w.r.t. the nuclear DOF " << n << " is not allocated but is needed for the calculations \n";
        std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
      }
      CMATRIX dH((option == 0) ? get_d1ham_dia_lowdin(n) : X * (*d1ham_dia[n]) * X);
      for (int i = 0; i < ndia; i++) { dH.set(i, i, complex<double>(0.0, 0.0)); }
      res.M[n] = -(dH * rho_l).tr();
    }
    res /= norm_l;
    return res;
  }
  CMATRIX dtilda(ndia, ndia);
  CMATRIX invS(ndia, ndia);
  FullPivLU_inverse(*ovlp_dia, invS);
  complex<double> norm = ((*ovlp_dia) * dm_dia).tr();
  CMATRIX temp_diag(ndia, ndia);
  for (int n = 0; n < nnucl; n++) {
    if (d1ham_dia_mem_status[n] == 0) {
      cout << "Error in QTSH_forces_dia_unit_dm(): the derivatives of the Hamiltonian matrix in the \
      diabatic basis w.r.t. the nuclear DOF " << n << " is not allocated but is needed for the calculations \n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }
    if (dc1_dia_mem_status[n] == 0) {
      cout << "Error in QTSH_forces_dia_unit_dm(): the derivatives couplings matrix in the diabatic \
      basis w.r.t. the nuclear DOF " << n << " is not allocated but is needed for the calculations \n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }
    if (option == 0) {
      dtilda = ((*dc1_dia[n]).H() * invS * (*ham_dia) + (*ham_dia) * invS * (*dc1_dia[n]));
      for (int i = 0; i < ndia; i++) { temp_diag.set(i, i, (*d1ham_dia[n] - dtilda).get(i, i)); }
      res.M[n] = -((*d1ham_dia[n] - dtilda - temp_diag) * dm_dia).tr();
    } else if (option == 1) {
      for (int i = 0; i < ndia; i++) { temp_diag.set(i, i, d1ham_dia[n]->get(i, i)); }
      res.M[n] = -((*d1ham_dia[n] - temp_diag) * dm_dia).tr();
    }
  } // for n
  res /= norm;
  return res;
}

CMATRIX nHamiltonian::QTSH_forces_dia_dm(vector<CMATRIX *> &dm_dia, int lvl, int option) {
  /**
    The density-matrix form of QTSH_forces_dia: one ndia x ndia density per trajectory (dyn_variables::dm_dia),
    lvl 0 - this Hamiltonian serves every trajectory, lvl 1 - children[i] serves trajectory i
    Returns MATRIX(ndof, ntraj) - the QTSH off-diagonal Hellmann-Feynman forces in the diabatic representation
  */
  int ntraj = dm_dia.size();
  if (lvl == 1 && children.size() != static_cast<std::size_t>(ntraj)) {
    cout << "ERROR in CMATRIX nHamiltonian::QTSH_forces_dia_dm(): the number of densities (" << ntraj
         << ") should be equal to the number of children Hamiltonians (" << children.size() << ")\n";
    std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
  }
  CMATRIX frc_tmp(nnucl, 1);
  CMATRIX F(nnucl, ntraj);
  for (int i = 0; i < ntraj; i++) {
    if (lvl == 0) { frc_tmp = QTSH_forces_dia_unit_dm(*dm_dia[i], option); }
    if (lvl == 1) { frc_tmp = children[i]->QTSH_forces_dia_unit_dm(*dm_dia[i], option); }
    for (int n = 0; n < nnucl; n++) { F.set(n, i, frc_tmp.get(n, 0)); }
  }
  return F;
}

CMATRIX nHamiltonian::QTSH_forces_adi_unit_dm(CMATRIX &dm_adi, int option) {
  /**
    The density-matrix form of QTSH_forces_adi_unit: every C^H A C of the amplitude form is tr(A rho) for
    rho = C_adi C_adi^H of one trajectory (dyn_variables::dm_adi), T the identity as the amplitude form takes it -
    equal to the amplitude form for a pure state
  */
  if (!(option == 0 || option == 1 || option == 2)) {
    cout << "Error in QTSH_forces_adi_unit_dm(): option = " << option
         << " is not allowed (0 - first-order off-diagonal force, 1 - with the second-order term, 2 - no derivative-coupling force)\n";
    std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
  }
  if (ham_adi_mem_status == 0) {
    cout << "Error in QTSH_forces_adi_unit_dm(): the adiabatic Hamiltonian matrix is not allocated \
  but it is needed for the calculations\n";
    std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
  }
  complex<double> norm = dm_adi.tr();
  CMATRIX res(nnucl, 1);
  CMATRIX tmp(nadi, nadi);
  // the amplitudes arrive in the basis of ham_adi - the step's projector moved them in the propagation, no transform here
  for (int n = 0; n < nnucl; n++) {
    if (d1ham_adi_mem_status[n] == 0) {
      cout << "Error in QTSH_forces_adi_unit_dm(): the derivatives of the Hamiltonian matrix in the \
    adiabatic basis w.r.t. the nuclear DOF " << n << " is not allocated but is needed for the calculations \n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }
    if (dc1_adi_mem_status[n] == 0) {
      cout << "Error in QTSH_forces_adi_unit_dm(): the derivatives couplings matrix in the adiabatic \
    basis w.r.t. the nuclear DOF " << n << " is not allocated but is needed for the calculations \n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }
    if (option == 0) { // the first-order off-diagonal force
      tmp = (*dc1_adi[n]).H() * (*ham_adi);
      tmp = tmp + tmp.H();
      res.M[n] = (tmp * dm_adi).tr();
    } else if (option == 1) { // with the second-order term
      tmp = (*dc1_adi[n]).H() * (*hvib_adi);
      tmp = tmp + tmp.H();
      res.M[n] = (tmp * dm_adi).tr();
    } else if (option == 2) { // no derivative-coupling force - the local-diabatization-consistent variant
      res.M[n] = 0.0;
    }
  } // for n
  res /= norm;
  return res;
}

CMATRIX nHamiltonian::QTSH_forces_adi_dm(vector<CMATRIX *> &dm_adi, int lvl, int option) {
  /**
    The density-matrix form of QTSH_forces_adi - one nadi x nadi density per trajectory
    (dyn_variables::dm_adi), lvl 0 - this Hamiltonian serves every trajectory, lvl 1 - children[i] serves trajectory i
    Returns MATRIX(ndof, ntraj) - the QTSH forces in the adiabatic representation
  */
  int ntraj = dm_adi.size();
  if (lvl == 1 && children.size() != static_cast<std::size_t>(ntraj)) {
    cout << "ERROR in CMATRIX nHamiltonian::QTSH_forces_adi_dm(): the number of densities (" << ntraj
         << ") should be equal to the number of children Hamiltonians (" << children.size() << ")\n";
    std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
  }
  CMATRIX frc_tmp(nnucl, 1);
  CMATRIX F(nnucl, ntraj);
  for (int i = 0; i < ntraj; i++) {
    if (lvl == 0) { frc_tmp = QTSH_forces_adi_unit_dm(*dm_adi[i], option); }
    if (lvl == 1) { frc_tmp = children[i]->QTSH_forces_adi_unit_dm(*dm_adi[i], option); }
    for (int n = 0; n < nnucl; n++) { F.set(n, i, frc_tmp.get(n, 0)); }
  }
  return F;
}

} // namespace libnhamiltonian
} // namespace liblibra
