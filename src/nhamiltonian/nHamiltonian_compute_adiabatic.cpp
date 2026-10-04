/*********************************************************************************
 * Copyright (C) 2017-2022 Alexey V. Akimov
 *
 * This file is distributed under the terms of the GNU General Public License
 * as published by the Free Software Foundation, either version 3 of
 * the License, or (at your option) any later version.
 * See the file LICENSE in the root directory of this distribution
 * or <http://www.gnu.org/licenses/>.
 *
 *********************************************************************************/
/**
  \file nHamiltonian_compute_adiabatic.cpp
  \brief The file implements the calculations of the adiabatic Hamiltonian (e.g.
  as a transformation)

*/

#include <exception>
#include <stdexcept>
#if defined(USING_PCH)
#include "../pch.h"
#else
#include <omp.h>
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

void nHamiltonian::update_ordering(vector<int> &perm_t, int lvl) {

  if (level == lvl) {

    update_permutation(perm_t, ordering_adi);
    // ordering_adi[0] = perm_t;

    // Apply the re-ordering to all the underlying properties:
    if (basis_transform_mem_status) {
      basis_transform->permute_cols(perm_t);
    }

    if (ham_adi_mem_status) {
      ham_adi->permute_cols(perm_t);
      ham_adi->permute_rows(perm_t);
    }

    if (nac_adi_mem_status) {
      nac_adi->permute_cols(perm_t);
      nac_adi->permute_rows(perm_t);
    }

    if (hvib_adi_mem_status) {
      hvib_adi->permute_cols(perm_t);
      hvib_adi->permute_rows(perm_t);
    }

    for (auto i = 0u; i < dc1_adi.size(); i++) {
      if (dc1_adi_mem_status[i]) {
        dc1_adi[i]->permute_cols(perm_t);
        dc1_adi[i]->permute_rows(perm_t);
      }
    }

    for (auto i = 0u; i < d1ham_adi.size(); i++) {
      if (d1ham_adi_mem_status[i]) {
        d1ham_adi[i]->permute_cols(perm_t);
        d1ham_adi[i]->permute_rows(perm_t);
      }
    }

    for (auto i = 0u; i < d2ham_adi.size(); i++) {
      if (d2ham_adi_mem_status[i]) {
        d2ham_adi[i]->permute_cols(perm_t);
        d2ham_adi[i]->permute_rows(perm_t);
      }
    }

  } // level==lvl

  else if (lvl > level) {
    require_children(lvl, __func__);

    for (auto i = 0u; i < children.size(); i++) {
      children[i]->update_ordering(perm_t, lvl);
    }

  } // lvl >level

  else {
    cout << "WARNING in void nHamiltonian::update_order(vector<int>& perm_t, "
            "int lvl):\n";
    cout << "Can not run the function in the parent Hamiltonian from the\
     child node\n";
  }
}

void nHamiltonian::update_ordering(vector<int> &perm_t) {

  update_ordering(perm_t, 0);
}

/**

void nHamiltonian::apply_phase_corrections(CMATRIX* phase_corr, int lvl){

//  phase_corr - CMATRIX(nadi, 1) - the changes of cumulative phases of all
eigenvectors.



  int i, j;
  complex<double> fji;

  if(level==lvl){

    if(basis_transform_mem_status){
      if(phase_corr->n_rows!=basis_transform->n_cols){
        cout<<"ERROR in void nHamiltonian::update_phases(vector<double>& phases,
int lvl): \
        the number of elements in the <phases> input should be equal to the
number of columns of \ the basis transformation matrix\n"; std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
      }

      for(i=0; i<basis_transform->n_cols; i++){
        for(j=0; j<basis_transform->n_rows; j++){
          basis_transform->scale(j,i, std::conj(phase_corr->get(i,0)) );
        }
      }

    }// basis_transform


    if(nac_adi_mem_status){
      if(phase_corr->n_rows!=nac_adi->n_cols){
        cout<<"ERROR in void nHamiltonian::update_phases(vector<double>& phases,
int lvl): \
        the number of elements in the <phases> input should be equal to the
number of columns of \ the nac_adi matrix\n"; std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
      }

      for(i=0; i<nac_adi->n_cols; i++){
        for(j=0; j<nac_adi->n_rows; j++){
          fji = phase_corr->get(j,0) * std::conj(phase_corr->get(i,0));
          nac_adi->scale(j,i, fji);
        }
      }
    }// nac_adi


    if(hvib_adi_mem_status){
      if(phase_corr->n_rows!=hvib_adi->n_cols){
        cout<<"ERROR in void nHamiltonian::update_phases(vector<double>& phases,
int lvl): \
        the number of elements in the <phases> input should be equal to the
number of columns of \ the hvib_adi matrix\n"; std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
      }

      for(i=0; i<hvib_adi->n_cols; i++){
        for(j=0; j<hvib_adi->n_rows; j++){
          fji = phase_corr->get(j,0) * std::conj(phase_corr->get(i,0));
          hvib_adi->scale(j,i, fji);
        }
      }
    }// nac_adi

  }// level==lvl

  else if(lvl>level){
    require_children(lvl, __func__);

    for(int i=0;i<children.size();i++){
      children[i]->apply_phase_corrections(phase_corr, lvl);
    }

  }// lvl >level

  else{
    cout<<"WARNING in void nHamiltonian::update_phases(vector<double>& phases,
int lvl) :\n";
    cout<<"Can not run the function in the parent Hamiltonian from the\
     child node\n";
  }

}

void nHamiltonian::apply_phase_corrections(CMATRIX& phase_corr, int lvl){

  apply_phase_corrections(&phase_corr, lvl);
}


void nHamiltonian::apply_phase_corrections(CMATRIX* phase_corr){

  apply_phase_corrections(phase_corr, 0);
}

void nHamiltonian::apply_phase_corrections(CMATRIX& phase_corr){

  apply_phase_corrections(&phase_corr, 0);
}


*/

/**

CMATRIX nHamiltonian::update_phases(CMATRIX& U_prev, int lvl){

//  This function computes the phase corrections to all current eigenvectors
//  w.r.t. those in U_prev.
//  It also applies the correction to the present eigenvectors.
//  Finally, this function computes the phases by which the adiabatic amplitudes
//  should be updated and returns these corrections as a column-vector.
//
//  basis_transform - the |psi(t')>
//  U_prev = |psi(t)>
//
//  t' > t



  if(lvl==level){

    // Memorize the cumulative phase correction up to this step
    CMATRIX* cum_phase_corr_prev;
    cum_phase_corr_prev = new CMATRIX(nadi, 1);
    *cum_phase_corr_prev = *cum_phase_corr;

    // Compute cumulative phase corrections
    *cum_phase_corr = compute_phase_corrections1(*basis_transform, U_prev,
phase_corr_ovlp_tol);

    // Update the wavefunction and wavefunction-dependent properties with
    // the new phase corrections
    apply_phase_corrections(cum_phase_corr, lvl);

    // Compute the phase correction to the evolving amplitudes
    CMATRIX ampl_corr(nadi, 1);

    for(int i=0;i<nadi; i++){
      complex<double> scl(1.0, 0.0);

      if( abs(cum_phase_corr_prev->get(i,0) ) > 0.0 ){
        scl = cum_phase_corr->get(i,0)/cum_phase_corr_prev->get(i,0);
      }
      ampl_corr.set(i,0, scl);
    }
    delete cum_phase_corr_prev;

    return ampl_corr;

  }// level==lvl

  else if(lvl>level){
    require_children(lvl, __func__);

    for(int i=0;i<children.size();i++){
      children[i]->apply_phase_corrections(U_prev, lvl);
    }

  }// lvl >level

  else{
    cout<<"WARNING in void nHamiltonian::update_phases(CMATRIX& U_prev, int lvl)
:\n"; cout<<"Can not run the function in the parent Hamiltonian from the child
node\n";
  }
}

CMATRIX nHamiltonian::update_phases(CMATRIX& U_prev){

  return update_phases(U_prev, 0);
}
*/

void nHamiltonian::compute_adiabatic(int der_lvl) {

  compute_adiabatic(der_lvl, 0);
}

void nHamiltonian::compute_adiabatic_derivatives() {
  /**
    The Hellmann-Feynman derivative tier shared by the internal
    diagonalization (compute_adiabatic) and the stored-transformation
    path (transform_dia2adia): adiabatic gradients and derivative
    couplings from (basis_transform, ham_adi, d1ham_dia, dc1_dia) -
    exact when basis_transform diagonalizes ham_dia. Extracted verbatim -
    int i, j locals replace the member-function scope of the caller.
  */
  int i, j;

        // Now compute the derivative couplings (off-diagonal, multiplied by
        // energy difference) and adiabatic gradients (diagonal)
        CMATRIX *tmp;
        tmp = new CMATRIX(nadi, nadi);
        CMATRIX *dtilda;
        dtilda = new CMATRIX(nadi, nadi);

        for (int n = 0; n < nnucl; n++) {

          if (d1ham_dia_mem_status[n] == 0) {
            cout
                << "Error in compute_adiabatic(): the derivatives of the diabatic Hamiltonian \
        matrix w.r.t. the nuclear DOF "
                << n << " is not allocated but is needed for calculations \n";
            std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
          }

          if (d1ham_adi_mem_status[n] == 0) {
            cout
                << "Error in compute_adiabatic(): the derivatives of the adiabatic Hamiltonian \
        matrix w.r.t. the nuclear DOF "
                << n
                << " is not allocated but is needed for collecting results \n";
            std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
          }

          if (dc1_dia_mem_status[n] == 0) {
            cout
                << "Error in compute_adiabatic(): the derivatives couplings matrix in the diabatic \
        basis w.r.t. the nuclear DOF "
                << n
                << " is not allocated but is needed for collecting results \n";
            std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
          }

          if (dc1_adi_mem_status[n] == 0) {
            cout
                << "Error in compute_adiabatic(): the derivatives couplings matrix in the adiabatic \
        basis w.r.t. the nuclear DOF "
                << n
                << " is not allocated but is needed for collecting results \n";
            std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
          }

          // E.g. see the derivations here:
          // https://github.com/alexvakimov/Derivatory/blob/master/theory_NAC.pdf
          // also:
          // http://www.theochem.ruhr-uni-bochum.de/~nikos.doltsinis/nic_10_doltsinis.pdf
          *tmp = (*basis_transform).H() * (*d1ham_dia[n]) * (*basis_transform);

          //        *dtilda = (*basis_transform) * (*dc1_dia[n]).H() *
          //        (*basis_transform).H() * (*ham_adi);
          // the derivative of the generalized eigenproblem H U = S U E for a moving
          // basis with dS = D + D^H: tmp_kl = (U^H dH U)_kl - E_k (U^H D U)_kl
          // - E_l (U^H D^H U)_kl, the energies multiplying from the left - the
          // diagonal is the Hellmann-Feynman force, the off-diagonals give
          // <phi_k|d phi_l> = (U^H S dU)_kl + (U^H D U)_kl, and a rigidly rotating
          // orthonormal basis keeps a zero adiabatic NAC
          *dtilda = (*ham_adi) * (*basis_transform).H() * (*dc1_dia[n]) *
                    (*basis_transform);
          *dtilda = (*dtilda + (*dtilda).H());

          *tmp -= *dtilda;

          // Adiabatic "forces"
          *d1ham_adi[n] = 0.0;
          for (i = 0; i < nadi; i++) {
            d1ham_adi[n]->set(i, i, tmp->get(i, i));
          }

          // Adiabatic derivative couplings
          *dc1_adi[n] = 0.0;

          for (i = 0; i < nadi; i++) {
            for (j = i + 1; j < nadi; j++) {

              // if(i==j){  dc1_adi[n]->set(i,j, 0.0, 0.0); }
              // else{

              double dE = (ham_adi->get(j, j) - ham_adi->get(i, i)).real();

              if (fabs(dE) < 1e-25) {
                dE = 1e-25;
              }
              complex<double> val = tmp->get(i, j) / dE;

              dc1_adi[n]->set(i, j, val);
              dc1_adi[n]->set(j, i, -val);

              /*
              if(fabs(dE)<1e-100){
                //dE = 1e-10 * (dE>0.0 ? 1.0 : -1.0);
                dE = 1e-100;
                dc1_adi[n]->set(i,j, tmp->get(i,j)/dE );
                //dc1_adi[n]->set(i,j, 0.0, 0.0 );
              }else{

                dc1_adi[n]->set(i,j, tmp->get(i,j)/dE );
              }
              */

              //}

            } // for j
          } // for i
        } // for n

        delete tmp;
        delete dtilda;

}

void nHamiltonian::compute_adiabatic(int der_lvl, int lvl) {
  /**
    Compute the adiabatic Hamiltonian

    |psi_adi> = |psi_dia> * U

    Stationary SE:  H |psi_adi> =  |psi_adi> * E

    project on <psi_adi|, keeping in mind that <psi_adi|psi_adi> = I

    <psi_dia| H |psi_adi> =  <psi_dia|psi_adi> E

    <psi_dia| H |psi_dia> * U =  <psi_dia|psi_dia> * U * E

    H_dia * U = ovlp_dia * U * H_adi

    Here, U = basis_transform

    Assume: H does not depend on U

    der_lvl >= 0 - only diabatic - to - adiabatic transform
    der_lvl >= 1 - forces and derivative couplings
    der_lvl >= 2 - Hessian

  */

  int i, j;

  if (level == lvl) {

    if (der_lvl >= 0) {

      if (ham_dia_mem_status == 0) {
        cout
            << "Error in compute_adiabatic(): the diabatic Hamiltonian matrix is not allocated \
    but it is needed for the calculations\n";
        std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
      }
      if (ovlp_dia_mem_status == 0) {
        cout
            << "Error in compute_adiabatic(): the overlap matrix of the diabatic states is not allocated \
    but it is needed for the calculations\n";
        std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
      }

      if (ham_adi_mem_status == 0) {
        cout
            << "Error in compute_adiabatic(): the adiabatic Hamiltonian matrix is not allocated \
    but it is used to collect the results of the calculations\n";
        std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
      }
      if (basis_transform_mem_status == 0) {
        cout
            << "Error in compute_adiabatic(): the basis_transform (eigenvector) matrix is\
    not allocated but it is used to collect the results of the calculations\n";
        std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
      }

      eigen_algo = 0; /// Should be this option, otherwise the static
                      /// calculations are incorrect

      if (nadi == 1 && ndia == 1) {
        *ham_adi = *ham_dia;
        basis_transform->set(0, 0, 1.0, 0.0);
      } else {
        if (eigen_algo == 0) {
          solve_eigen(ham_dia, ovlp_dia, ham_adi, basis_transform, 1);
        } else if (eigen_algo == 1) {
          solve_eigen_nosort(ham_dia, ham_adi, basis_transform,
                             1); ///< references
        }
      }

      if (der_lvl >= 1) {
        compute_adiabatic_derivatives();
      } // der_lvl>=1
    } // der_lvl>=0

  } // level == lvl

  else if (lvl > level) {
    require_children(lvl, __func__);

    for (auto i = 0u; i < children.size(); i++) {
      children[i]->compute_adiabatic(der_lvl, lvl);
    }

  } // lvl >level

  else {
    cout << "WARNING in nHamiltonian::compute_adiabatic\n";
    cout
        << "Can not run evaluation of function in the parent Hamiltonian from the\
     child node\n";
  }
}

void nHamiltonian::transform_dia2adi(int der_lvl, int lvl) {
  /**
    Diabatic-to-adiabatic transform by the MODEL-SUPPLIED basis
    transformation (ham_transform_method == 2): ham_adi = U^H ham_dia U
    in the model's own gauge - no internal diagonalization, so no LAPACK
    eigenvector branch switches. der_lvl >= 1 adds the shared
    Hellmann-Feynman derivative tier (exact when U diagonalizes ham_dia -
    a warning names the residual otherwise).
  */
  if (level == lvl) {
    if (ham_dia_mem_status == 0) {
      cout << "Error in transform_dia2adi(): the diabatic Hamiltonian matrix is not \
  allocated but is needed for the calculations\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }
    if (basis_transform_mem_status == 0) {
      cout << "Error in transform_dia2adi(): the basis_transform matrix is not \
  allocated but is needed for the calculations\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }
    if (ham_adi_mem_status == 0) {
      cout << "Error in transform_dia2adi(): the adiabatic Hamiltonian matrix is not \
  allocated but is used to collect the results of the calculations\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }
    // the square identity DEFAULT must not silently alias dia == adi: the
    // stored-U transform is meaningful only for a transformation the model
    // actually supplied (an explicit identity supply passes)
    if (!basis_transform_supplied) {
      cout << "Error in transform_dia2adi(): ham_transform_method = 2 consumes the \
  model-supplied basis transformation, but obj.basis_transform was never supplied\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }
    // chained in-place product - CBLAS zgemm-backed, no per-call temporaries
    ham_adi->product(adjoint(*basis_transform), *ham_dia, *basis_transform);
    if (der_lvl >= 1) {
      compute_adiabatic_derivatives();
      // the Hellmann-Feynman tier is exact only in the eigenbasis - a U that
      // does not diagonalize ham_dia leaves off-diagonals in U^H H U
      static int warned_offdiag = 0;
      if (!warned_offdiag) {
        double offmax = 0.0;
        for (int i = 0; i < nadi; i++) {
          for (int j = 0; j < nadi; j++) {
            if (i != j) {
              double a = abs(ham_adi->get(i, j));
              if (a > offmax) { offmax = a; }
            }
          } // for j
        } // for i
        if (offmax > 1e-8) {
          cout << "WARNING in transform_dia2adi(): the supplied basis_transform does not \
  diagonalize ham_dia (max off-diagonal " << offmax << ") - the propagation basis is \
  quasi-diabatic and the Hellmann-Feynman derivative tier is approximate\n";
          warned_offdiag = 1;
        }
      }
    } // der_lvl >= 1
  } // level == lvl

  else if (lvl > level) {
    require_children(lvl, __func__);
    for (auto i = 0u; i < children.size(); i++) {
      children[i]->transform_dia2adi(der_lvl, lvl);
    }
  } // lvl > level

  else {
    cout << "WARNING in nHamiltonian::transform_dia2adi\n";
    cout
        << "Can not run evaluation of function in the parent Hamiltonian from the\
     child node\n";
  }
}

void nHamiltonian::transform_adi2dia(int lvl) {
  /**
    Back-transform the adiabatic Hamiltonian into the diabatic basis
    using the stored basis transformation (ham_transform_method == 3).
    From H_dia U = S U E with U^H S U = I follows U^{-1} = U^H S, so
    ham_dia = S U ham_adi U^H S
    The derivative tier is NOT transformed - that would need dU/dR.
  */
  if (level == lvl) {
    if (ham_adi_mem_status == 0) {
      cout << "Error in transform_adi2dia(): the adiabatic Hamiltonian matrix is not \
  allocated but is needed for the calculations\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }
    if (basis_transform_mem_status == 0) {
      cout << "Error in transform_adi2dia(): the basis_transform matrix is not \
  allocated but is needed for the calculations\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }
    if (ovlp_dia_mem_status == 0) {
      cout << "Error in transform_adi2dia(): the overlap matrix of the diabatic states is not \
  allocated but is needed for the calculations\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }
    if (ham_dia_mem_status == 0) {
      cout << "Error in transform_adi2dia(): the diabatic Hamiltonian matrix is not \
  allocated but is used to collect the results of the calculations\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }
    // the square identity DEFAULT must not silently alias dia == adi (ham_dia = ham_adi): the
    // transform is meaningful only for a transformation the model actually supplied - an
    // explicit identity supply passes, as in transform_dia2adi
    if (!basis_transform_supplied) {
      cout << "Error in transform_adi2dia(): the adiabatic->diabatic transform consumes the \
  model-supplied basis transformation, but obj.basis_transform was never supplied\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }
    // S U first (ndia x nadi), then (S U H) (S U)^H: every product of the
    // chain stays ndia x nadi, where the left-to-right association ended
    // in an ndia x ndia product (ndia^3) - the ndia x nadi temporaries are
    // the price
    CMATRIX SU(ndia, nadi);
    SU.product(*ovlp_dia, *basis_transform);
    CMATRIX SUH(ndia, nadi);
    SUH.product(SU, *ham_adi);
    // a model-supplied obj.ham_dia lives in the SAME U-basis the
    // back-transform rebuilds into - consistent redundancy passes, a
    // disagreeing supply would be silently overwritten every step
    // (the rebuilt matrix is then the result itself)
    if (ham_dia_supplied) {
      CMATRIX rebuilt(ndia, ndia);
      rebuilt.product(SUH, adjoint(SU));
      if (abs((rebuilt - *ham_dia).max_elt()) > 1e-8) {
        cout << "Error in transform_adi2dia(): the model supplies obj.ham_dia, but \
  the adiabatic->diabatic transform rebuilds it as S U ham_adi U^H S and the two \
  disagree - the supplied data would be overwritten. Supply consistent data, or \
  drop obj.ham_dia\n";
        std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
      }
      *ham_dia = rebuilt;
    }
    else {
      ham_dia->product(SUH, adjoint(SU));
    }
  } // level == lvl

  else if (lvl > level) {
    require_children(lvl, __func__);
    for (auto i = 0u; i < children.size(); i++) {
      children[i]->transform_adi2dia(lvl);
    }
  } // lvl > level

  else {
    cout << "WARNING in nHamiltonian::transform_adi2dia\n";
    cout
        << "Can not run evaluation of function in the parent Hamiltonian from the\
     child node\n";
  }
}

// void nHamiltonian::compute_adiabatic(bp::object py_funct, bp::object q,
// bp::object params){
void nHamiltonian::compute_adiabatic(bp::object py_funct, MATRIX &q,
                                     bp::object params) {
  /**
    Performs the adiabatic properties calculation at the top-most level of the
    Hamiltonians hierarchy. See the description of the more general function
    prototype for more info.
  */

  compute_adiabatic(py_funct, q, params, 0);
}

// void nHamiltonian::compute_adiabatic(bp::object py_funct, bp::object q,
// bp::object params, int lvl){
void nHamiltonian::compute_adiabatic(bp::object py_funct, MATRIX &q,
                                     bp::object params, int lvl) {
  /**
    This function will call the <py_funct> function defined in Python and taking
    the signature:

    def py_funct(q, params, full_id)
        return obj

    The object <obj> returned by the function should contain variables named:
    "ham_adi" (CMATRIX), "d1ham_adi" (list of CMATRIX), "dc1_adi" (list of
    CMATRIX) of the specified types

    q - is the object containing coordinates of nuclei. Since almost any C++
    type exposed to Python is seen by Python as a Python object, you can use any
    convenient Libra type to store the coordinates, for instance a MATRIX
    object.

    params - is an object containing any parameters needed by the <py_funct>
    Python function to perform the specified calculations. Again, there are many
    options to pass the parameters on the Python side, but it may be convenient
    to use Python dictionary, since the parameters will be mostly used at the
    Python level of calculations.

    full_id - is vector<int> object containing the "path" of the calling node in
    the hierarchy of Hamiltonians this information may be needed by the py_funct
    function in order to do the proper calculations

    lvl - is the level of the Hamiltonians in the hierarchy of Hamiltonians to
    be executed by this call You can only request to perform computations at the
    same of higher (children) levels that the level of the Hamiltonian from
    which you call this function.

    Contentwise, this function computes the adiabatic properties and populates
    the corresponding storage.

  */

  if (level == lvl) {

    // Call the Python function with such arguments
    bp::object obj = py_funct(q, params, get_full_id());

    // Extract all the computed properties
    int has_attr = 0;
    has_attr = (int)hasattr(obj, "ham_adi");
    ham_adi_present = has_attr;
    if (has_attr) {

      check_cmatrix(obj, "ham_adi", nadi, nadi);
      // a model may supply any family the hierarchy was not initialized
      // for - allocate on demand (the raw-pointer writes would otherwise
      // dereference NULL) - same policy for every ingested family below
      if (ham_adi_mem_status == 0) { init_ham_adi(); }
      //*ham_adi = extract<CMATRIX>(obj.attr("ham_adi"));
      try {
        *ham_adi = extract<CMATRIX>(obj.attr("ham_adi"));
      }
      catch (bp::error_already_set const &) {
        std::cout << "Error extracting ham_adi\n";
        PyErr_Print();
        throw;
      }
    }

    has_attr = (int)hasattr(obj, "nac_adi");
    nac_adi_present = has_attr;
    nac_adi_supplied = 0;
    if (has_attr) {

      check_cmatrix(obj, "nac_adi", nadi, nadi);
      if (nac_adi_mem_status == 0) { init_nac_adi(); }
      //*nac_adi = extract<CMATRIX>(obj.attr("nac_adi"));
      try {
        *nac_adi = extract<CMATRIX>(obj.attr("nac_adi"));
      }
      catch (bp::error_already_set const &) {
        std::cout << "Error extracting nac_adi\n";
        PyErr_Print();
        throw;
      }
      // an all-zero supply is the ecosystem's placeholder convention - not data
      nac_adi_supplied = (abs(nac_adi->max_elt()) != 0.0);
    }

    has_attr = (int)hasattr(obj, "hvib_adi");
    hvib_adi_present = has_attr;
    hvib_adi_supplied = 0;
    if (has_attr) {

      check_cmatrix(obj, "hvib_adi", nadi, nadi);
      if (hvib_adi_mem_status == 0) { init_hvib_adi(); }
      //*hvib_adi = extract<CMATRIX>(obj.attr("hvib_adi"));
      try {
        *hvib_adi = extract<CMATRIX>(obj.attr("hvib_adi"));
      }
      catch (bp::error_already_set const &) {
        std::cout << "Error extracting hvib_adi\n";
        PyErr_Print();
        throw;
      }
      // an all-zero supply is the ecosystem's placeholder convention - not data
      hvib_adi_supplied = (abs(hvib_adi->max_elt()) != 0.0);
    }

    has_attr = (int)hasattr(obj, "hvib_dia");
    hvib_dia_present = has_attr;
    hvib_dia_supplied = 0;
    if (has_attr) {

      // an adiabatic-native model may supply the DIABATIC vibronic
      // Hamiltonian (e.g. the frame-corrected H_SD - i D_SD) to be
      // consumed under hvib_update_method 0 with transform 3
      check_cmatrix(obj, "hvib_dia", ndia, ndia);
      if (hvib_dia_mem_status == 0) { init_hvib_dia(); }
      *hvib_dia = extract<CMATRIX>(obj.attr("hvib_dia"));
      // an all-zero supply is the ecosystem's placeholder convention - not data
      hvib_dia_supplied = (abs(hvib_dia->max_elt()) != 0.0);
    }

    basis_transform_supplied = 0;
    has_attr = (int)hasattr(obj, "basis_transform");
    if (has_attr) {

      // the transformation matrix is (ndia x nadi) by allocation - rectangular
      // whenever the model declares more diabats than adiabatic states (e.g.
      // an SD basis as the diabatic basis with the CI matrix as the transform)
      check_cmatrix(obj, "basis_transform", ndia, nadi);
      if (basis_transform_mem_status == 0) { init_basis_transform(); }
      //*basis_transform = extract<CMATRIX>(obj.attr("basis_transform"));
      try {
        *basis_transform = extract<CMATRIX>(obj.attr("basis_transform"));
      }
      catch (bp::error_already_set const &) {
        std::cout << "Error extracting basis_transform\n";
        PyErr_Print();
        throw;
      }
      // an all-zero supply is the ecosystem's placeholder convention - not data
      basis_transform_supplied = (abs(basis_transform->max_elt()) != 0.0);
    }

    ham_dia_supplied = 0;
    has_attr = (int)hasattr(obj, "ham_dia");
    ham_dia_present = has_attr;
    if (has_attr) {

      // the diabatic-representation hop proposals (LZ with rep_lz 0/1, ZN)
      // detect crossings and read couplings in ham_dia, which
      // compute_diabatic never populates for adiabatic-native models - this
      // is their way to supply it
      check_cmatrix(obj, "ham_dia", ndia, ndia);
      if (ham_dia_mem_status == 0) { init_ham_dia(); }
      try {
        *ham_dia = extract<CMATRIX>(obj.attr("ham_dia"));
      }
      catch (bp::error_already_set const &) {
        std::cout << "Error extracting ham_dia\n";
        PyErr_Print();
        throw;
      }
      // an all-zero supply is the ecosystem's placeholder convention - not data
      ham_dia_supplied = (abs(ham_dia->max_elt()) != 0.0);
    }

    has_attr = (int)hasattr(obj, "ovlp_dia");
    if (has_attr) {

      // adiabatic-native models never run compute_diabatic, so this is their
      // only way to declare the diabatic metric (identity for orthonormal
      // diabats) - without it ampl_dia2adi multiplies by an all-zero S_dia
      check_cmatrix(obj, "ovlp_dia", ndia, ndia);
      if (ovlp_dia_mem_status == 0) { init_ovlp_dia(); }
      try {
        *ovlp_dia = extract<CMATRIX>(obj.attr("ovlp_dia"));
      }
      catch (bp::error_already_set const &) {
        std::cout << "Error extracting ovlp_dia\n";
        PyErr_Print();
        throw;
      }
      ovlp_dia_supplied = 1;
    }

    has_attr = (int)hasattr(obj, "time_overlap_adi");
    time_overlap_adi_present = has_attr;
    time_overlap_adi_supplied = 0;
    if (has_attr) {

      check_cmatrix(obj, "time_overlap_adi", nadi, nadi);
      if (time_overlap_adi_mem_status == 0) { init_time_overlap_adi(); }
      //*time_overlap_adi = extract<CMATRIX>(obj.attr("time_overlap_adi"));
      try {
        *time_overlap_adi = extract<CMATRIX>(obj.attr("time_overlap_adi"));
      }
      catch (bp::error_already_set const &) {
        std::cout << "Error extracting time_overlap_adi\n";
        PyErr_Print();
        throw;
      }
      // an all-zero supply is the ecosystem's placeholder convention - not data
      time_overlap_adi_supplied = (abs(time_overlap_adi->max_elt()) != 0.0);
    }

    has_attr = (int)hasattr(obj, "dc1_adi");
    dc1_adi_present = has_attr;
    if (has_attr) {

      check_cmatrix_list(obj, "dc1_adi", nadi, nadi, nnucl);
      if (dc1_adi_mem_status[0] == 0) { init_dc1_adi(); }

      vector<CMATRIX> _dc1_adi(nnucl, CMATRIX(nadi, nadi));
      //_dc1_adi = extract<CMATRIXList>(obj.attr("dc1_adi"));

      try {
        _dc1_adi = extract<CMATRIXList>(obj.attr("dc1_adi"));
        for (int i = 0; i < nnucl; i++) {   *dc1_adi[i] = _dc1_adi[i];   }
      }
      catch (bp::error_already_set const &) {
        std::cout << "Error extracting dc1_adi\n";
        PyErr_Print();
        throw;
      }

      //for (int i = 0; i < nnucl; i++) {
      //  *dc1_adi[i] = _dc1_adi[i];
      //}
    }

    has_attr = (int)hasattr(obj, "d1ham_adi");
    d1ham_adi_present = has_attr;
    d1ham_dia_present = 0;
    if (has_attr) {

      check_cmatrix_list(obj, "d1ham_adi", nadi, nadi, nnucl);
      if (d1ham_adi_mem_status[0] == 0) { init_d1ham_adi(); }

      vector<CMATRIX> _d1ham_adi(nnucl, CMATRIX(nadi, nadi));
      _d1ham_adi = extract<CMATRIXList>(obj.attr("d1ham_adi"));

      for (int i = 0; i < nnucl; i++) {
        *d1ham_adi[i] = _d1ham_adi[i];
      }
    }

    has_attr = (int)hasattr(obj, "d2ham_adi");
    if (has_attr) {

      check_cmatrix_list(obj, "d2ham_adi", nadi, nadi, nnucl * nnucl);

      // a model may supply second derivatives while the hierarchy was
      // initialized to der_lvl < 2 - allocate on demand (the raw-pointer
      // copy below would otherwise dereference NULL)
      if (d2ham_adi_mem_status[0] == 0) { init_d2ham_adi(); }
      vector<CMATRIX> _d2ham_adi(nnucl * nnucl, CMATRIX(nadi, nadi));
      _d2ham_adi = extract<CMATRIXList>(obj.attr("d2ham_adi"));

      for (int i = 0; i < nnucl * nnucl; i++) {
        *d2ham_adi[i] = _d2ham_adi[i];
      }
    }

    // Additional properties that may be needed in some NBRA calculations:
    has_attr = (int)hasattr(obj, "gs_kinetic_energy");
    if (has_attr) {

      // check_cmatrix(obj, "gs_kinetic_energy", 1, 1);
      gs_kinetic_energy = extract<double>(obj.attr("gs_kinetic_energy"));
    }

    // attributes of the OTHER representation are NOT read by this harvest -
    // silently dropping a supplied channel hides wrong models (ham_dia and
    // ovlp_dia are exempt, harvested above for the crossing detectors -
    // basis_transform is dual-harvested), so fail loud instead
    {
      // hvib_dia is NOT in this list - the supplied-hvib workflows ingest
      // it on this harvest (dia_basis_hvib and the transform-3 route)
      const char* dia_names[] = {"nac_dia", "dc1_dia",
                                 "d1ham_dia", "d2ham_dia",
                                 "time_overlap_dia"};
      for (int a = 0; a < 5; a++) {
        if ((int)hasattr(obj, dia_names[a])) {
          cout << "Error in compute_adiabatic(): the model supplies obj."
               << dia_names[a] << ", which the adiabatic harvest"
               << " (ham_update_method = 2) does not read - the supplied data"
               << " would be silently dropped. Use ham_update_method = 1"
               << " (diabatic harvest) or drop the attribute\n";
          std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
        }
      }
    }

    // any attribute matching NO harvest name in EITHER representation is
    // silently dropped - report each unknown name once per process
    {
      static bool warned_unknown = false;
      if (!warned_unknown && (int)hasattr(obj, "__dict__")) {
        // the union of both harvests - overlap_adi is the deliberate
        // python-side auxiliary channel (drivers/certifiers read it
        // from the model object directly - never ingested)
        const char* known[] = {"ham_dia", "ovlp_dia", "nac_dia",
            "hvib_dia", "dc1_dia", "d1ham_dia", "d2ham_dia",
            "time_overlap_dia", "ham_adi", "nac_adi", "hvib_adi",
            "time_overlap_adi", "basis_transform", "dc1_adi",
            "d1ham_adi", "d2ham_adi", "gs_kinetic_energy",
            "overlap_adi"};
        boost::python::dict dct =
            extract<boost::python::dict>(obj.attr("__dict__"));
        boost::python::list keys = dct.keys();
        for (int a = 0; a < len(keys); a++) {
          std::string name = extract<std::string>(keys[a]);
          if (name.rfind("_", 0) == 0) { continue; }
          bool match = false;
          for (int b = 0; b < 18; b++) {
            if (name == known[b]) { match = true; break; }
          }
          if (!match) {
            cout << "WARNING in compute_adiabatic(): the model supplies obj."
                 << name << ", which matches NO ingested attribute name in"
                 << " either representation - it is silently dropped (check"
                 << " the spelling against the harvest lists in"
                 << " nHamiltonian_compute_adiabatic/diabatic.cpp)\n";
            warned_unknown = true;
          }
        }
      }
    }

  } // if lvl == level

  else if (lvl > level) {
    require_children(lvl, __func__);

#ifdef __OPENMP
    int nthreads = omp_get_num_threads();
#else
    int nthreads = 1;
#endif
    // exceptions (including python errors from the model callback) cannot
    // leave an OMP-outlined region - capture and rethrow outside
    std::exception_ptr omp_eptr = nullptr;
#pragma omp parallel num_threads(nthreads)
    {
#pragma omp for
      for (auto i = 0u; i < children.size(); i++) {
        try { children[i]->compute_adiabatic(py_funct, q, params, lvl); }
        catch (...) {
          #pragma omp critical
          { if (!omp_eptr) { omp_eptr = std::current_exception(); } }
        }
      }
    } // pragma
    if (omp_eptr) { std::rethrow_exception(omp_eptr); }

  }

  else {
    cout << "WARNING in nHamiltonian::compute_adiabatic\n";
    cout
        << "Can not run evaluation of function in the parent Hamiltonian from the\
     child node\n";
  }
}

} // namespace libnhamiltonian
} // namespace liblibra
