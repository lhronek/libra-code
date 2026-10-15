/*********************************************************************************
* Copyright (C) 2022-2024 Alexey V. Akimov
*
* This file is distributed under the terms of the GNU General Public License
* as published by the Free Software Foundation, either version 3 of
* the License, or (at your option) any later version.
* See the file LICENSE in the root directory of this distribution
* or <http://www.gnu.org/licenses/>.
*
*********************************************************************************/
/**
  \file dyn_variables_electronic.cpp
  \brief The file implements the methods for electronic dyn vars
*/

#include "../util/libutil.h"
#include <exception>
#include <stdexcept>
#include "../converters/libconverters.h"
#include "dyn_variables.h"
#include "dyn_ham.h"
#include "../math_meigen/mEigen.h"


/// liblibra namespace
namespace liblibra{

using namespace liblinalg;
using namespace libutil;
using namespace libconverters;


/// libdyn namespace
namespace libdyn{

namespace bp = boost::python;



CMATRIX transform_amplitudes(int rep_in, int rep_out, CMATRIX& C, nHamiltonian& ham){
/**
  This function converts the amplitudes from one representation to another

  The reason: we may be solving TD-SE (computing forces) in one representation
  but compute the hopping probabilities in another one.

  This function assumes we already have the basis transformation matrix in ham object 
  computed/updated

*/

  int nst = C.n_rows;    
  int ntraj = C.n_cols;

  CMATRIX Coeff(nst,ntraj); 

  /// Depending on the basis, select which   
  /// C - the basis in which the electron-nuclear propagation is done
  /// Coeff - the basis in which SH is done

  // Input in the diabatic basis
  if(rep_in==0){                   
    // Output in the diabatic basis too
    if(rep_out==0){    Coeff = C;   }

    // Output in the adiabatic basis
    else if(rep_out==1){  ham.ampl_dia2adi(C, Coeff, 0, 1);  }
  }

  // Input in the adiabatic basis 
  else if(rep_in==1){   
    // Output in the diabatic basis 
    if(rep_out==0){  ham.ampl_adi2dia(Coeff, C, 0, 1);   } 

    // Output in the diabatic basis too
    else if(rep_out==1){     Coeff = C;    } 
  }

  return Coeff;
}


void dyn_variables::update_mm_variables(){
  /**
    Meyer-Miller mapping variables from the adiabatic amplitudes, ampl_adi =
    (q_mm + i p_mm)/sqrt2 - kept in step with every amplitude update
  */
  for(int traj=0; traj<ntraj; traj++){
    for(int i=0; i<nadi; i++){
      complex<double> c = ampl_adi->get(i, traj);
      q_mm->set(i, traj, sqrt(2.0) * c.real());
      p_mm->set(i, traj, sqrt(2.0) * c.imag());
    }
  }
}

void dyn_variables::update_amplitudes(dyn_control_params& dyn_params, nHamiltonian& ham){

  // The diabatic rep is the major, so we update the adiabatic amplitudes
  if(dyn_params.rep_tdse==0){   ham.ampl_dia2adi(ampl_dia, ampl_adi, 0, 1);  }

  // The adiabatic rep is the major, so we update the diabatic amplitudes
  else if(dyn_params.rep_tdse==1){   ham.ampl_adi2dia(ampl_dia, ampl_adi, 0, 1);  }
  else if(dyn_params.rep_tdse==2 || dyn_params.rep_tdse==3){ ; } // density-matrix representations: no amplitudes to convert
  else{
    std::cout<<"Error in dyn_variables::update_amplitudes(): rep_tdse = "<<dyn_params.rep_tdse<<" has no branch here (implemented: 0, 1; 2 and 3 carry density matrices, no amplitudes)\n";
    std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
  }

  update_mm_variables();
}

void dyn_variables::update_amplitudes(bp::dict dyn_params, nHamiltonian& ham){

  dyn_control_params _prms;
  _prms.set_parameters(dyn_params);

  update_amplitudes(_prms, ham);

}


void dyn_variables::update_amplitudes(dyn_control_params& dyn_params, bp::object compute_model, bp::dict model_params){
/**
    """
    Args:
        dyn_params ( Python dictionary ): control of the dynamics
        compute_model ( Python function ): the function that does the calculations
        model_params ( Python dictionary ): parameters of the computational model

    Returns:
        None: updates ampl_dia or ampl_adi, depending on `dyn_params.rep_tdse` variable:

        rep

    """
*/

    //# Prepare the Hamiltonian's hierarchy
    nHamiltonian ham(ndia, nadi, ndof);
    ham.add_new_children(ndia, nadi, ndof, ntraj);
    ham.init_all(1,1);

    //# Compute the Hamiltonian transformation
    //update_Hamiltonian_q(dyn_params, *q, ham, compute_model, model_params);
    update_Hamiltonian_variables(dyn_params, *this, ham, ham, compute_model, model_params, 0);


    // The diabatic rep is the major, so we update the adiabatic amplitudes
    if(dyn_params.rep_tdse==0){   ham.ampl_dia2adi(ampl_dia, ampl_adi, 0, 1);  }

    // The adiabatic rep is the major, so we update the diabatic amplitudes
    else if(dyn_params.rep_tdse==1){   ham.ampl_adi2dia(ampl_dia, ampl_adi, 0, 1);  }
    else if(dyn_params.rep_tdse==2 || dyn_params.rep_tdse==3){ ; } // density-matrix representations: no amplitudes to convert
    else{
      std::cout<<"Error in dyn_variables::update_amplitudes(): rep_tdse = "<<dyn_params.rep_tdse<<" has no branch here (implemented: 0, 1; 2 and 3 carry density matrices, no amplitudes)\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }

    update_mm_variables();
}


void dyn_variables::update_amplitudes(bp::dict dyn_params, bp::object compute_model, bp::dict model_params){

  dyn_control_params _prms;
  _prms.set_parameters(dyn_params);

  update_amplitudes(_prms, compute_model, model_params);

}

void dyn_variables::update_basis_transform(nHamiltonian& Ham){
/**

*/
  // NBRA runs (isNBRA==1) carry a single child Hamiltonian shared by all
  // trajectories - never index children beyond what actually exists
  int nchildren = Ham.children.size();
  for(int itraj=0; itraj<ntraj; itraj++){
    nHamiltonian* ham = Ham.children[ (itraj < nchildren) ? itraj : 0 ];
    *basis_transform[itraj] = *(ham->basis_transform);
    // a supplied metric is copied and scanned once over its storage (the
    // conversions skip their metric products on an identity), a child never
    // given a metric holds the identity of its allocation, and the identity
    // slot is refilled only when it held a metric before
    if(ham->ovlp_dia_mem_status && ham->ovlp_dia_supplied){
      *ovlp_dia[itraj] = *(ham->ovlp_dia);
      int identity = 1;
      complex<double>* Sm = ovlp_dia[itraj]->M;
      for(int k=0; k<ndia*ndia && identity; k++){
        double re = Sm[k].real() - ((k % (ndia + 1) == 0) ? 1.0 : 0.0);
        if(std::abs(re) > 1e-12 || std::abs(Sm[k].imag()) > 1e-12){ identity = 0; }
      }
      ovlp_dia_is_identity[itraj] = identity;
    }
    else{
      if(!ovlp_dia_is_identity[itraj]){ ovlp_dia[itraj]->load_identity(); }
      ovlp_dia_is_identity[itraj] = 1;
    }
  }// for itraj
}


void dyn_variables::update_amplitudes(dyn_control_params& dyn_params){
/**
           |\psi_adi > = | \psi_dia > U,  U^H S U = I

   |\Psi> = |\psi_adi > C_adi = |\psi_dia> U C_adi = |\psi_dia> C_dia, so:

   C_dia = U C_adi

   C_adi = U.H() S C_dia   (S = the cached diabatic metric - identity for
                            orthonormal diabats)
*/
  // Deterministic per-trajectory transforms - parallel-safe (each iteration
  // reads/writes only its own column) - results identical for any thread count
  // exceptions cannot leave an OMP-outlined region (even serialized) -
  // capture and rethrow outside so the guards stay translatable
  std::exception_ptr omp_eptr = nullptr;
  #pragma omp parallel for num_threads(dyn_params.omp_num_threads) \
      if (dyn_params.omp_num_threads > 1)
  for(int traj=0; traj<ntraj; traj++){
    try {

    CMATRIX cd(ndia, 1);
    CMATRIX ca(nadi, 1);
    CMATRIX& U = (*basis_transform[traj]);

    /// Diabatic wfc representation
    if(dyn_params.rep_tdse==0){
      cd = ampl_dia->col(traj);
      if(ovlp_dia_is_identity[traj]){ ca.product(adjoint(U), cd); }
      else{ ca.product(adjoint(U), (*ovlp_dia[traj]) * cd); }
      for(int i=0;i<nadi;i++){ ampl_adi->set(i, traj, ca.get(i,0) ); }
    }
    /// Adiabatic wfc representation
    else if(dyn_params.rep_tdse==1){
      ca = ampl_adi->col(traj);
      cd.product(U, ca);
      for(int i=0;i<ndia;i++){ ampl_dia->set(i, traj, cd.get(i,0) ); }
    }
    else if(dyn_params.rep_tdse==2 || dyn_params.rep_tdse==3){ ; } // density-matrix representations: no amplitudes to convert
    else{
      std::cout<<"Error in dyn_variables::update_amplitudes(): rep_tdse = "<<dyn_params.rep_tdse<<" has no branch here (implemented: 0, 1; 2 and 3 carry density matrices, no amplitudes)\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }

    } catch (...) {
      #pragma omp critical
      { if (!omp_eptr) { omp_eptr = std::current_exception(); } }
    }
  }// for traj
  if (omp_eptr) { std::rethrow_exception(omp_eptr); }
  update_mm_variables();

}


static void check_shape(const char* fn, CMATRIX& x, int nr, int nc){
  if(x.n_rows != nr || x.n_cols != nc){
    std::cout<<"Error in dyn_variables::"<<fn<<"(): the matrix is "<<x.n_rows<<" x "<<x.n_cols<<", "<<nr<<" x "<<nc<<" expected\n";
    std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
  }
}
static void check_traj(const char* fn, int i, int ntraj){
  if(i < 0 || i >= ntraj){
    std::cout<<"Error in dyn_variables::"<<fn<<"(): trajectory "<<i<<" of "<<ntraj<<"\n";
    std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
  }
}
void dyn_variables::set_ampl_adi(CMATRIX& x){ check_shape("set_ampl_adi", x, nadi, ntraj); *ampl_adi = x; }
void dyn_variables::set_ampl_dia(CMATRIX& x){ check_shape("set_ampl_dia", x, ndia, ntraj); *ampl_dia = x; }
void dyn_variables::set_dm_adi(int i, CMATRIX& x){ check_traj("set_dm_adi", i, ntraj); check_shape("set_dm_adi", x, nadi, nadi); *dm_adi[i] = x; }
void dyn_variables::set_dm_dia(int i, CMATRIX& x){ check_traj("set_dm_dia", i, ntraj); check_shape("set_dm_dia", x, ndia, ndia); *dm_dia[i] = x; }

void dyn_variables::adopt_adiabatic(dyn_control_params& dyn_params){
/**
  The write-through of a correction made on the adiabatic variable (a collapse, a projection, an SDM scaling, an
  XF half step): under rep_tdse 0 the propagated diabatic amplitudes take the corrected adiabatic ones
  (C_dia = U C_adi, the rep_tdse 1 direction of update_amplitudes), under rep_tdse 2 the propagated diabatic density
  takes the corrected adiabatic one (rho_dia = U rho_adi U^H, the rep_tdse 3 direction of update_density_matrix) -
  called right after the correction, before the end-of-step recompute of the adiabatic variable from the propagated
  one would discard it. Under rep_tdse 1 and 3 the adiabatic variable is the propagated one and nothing is done
*/
  int rep = dyn_params.rep_tdse;
  if(rep==1 || rep==3){ return; }
  if(rep==0){
    CMATRIX cd(ndia, 1);
    CMATRIX ca(nadi, 1);
    for(int traj=0; traj<ntraj; traj++){
      CMATRIX& U = (*basis_transform[traj]);
      ca = ampl_adi->col(traj);
      cd.product(U, ca);
      for(int i=0;i<ndia;i++){ ampl_dia->set(i, traj, cd.get(i,0) ); }
    }
  }
  else if(rep==2){
    for(int traj=0; traj<ntraj; traj++){
      CMATRIX& U = (*basis_transform[traj]);
      dm_dia[traj]->product(U, *dm_adi[traj], adjoint(U));
    }
  }
  else{
    std::cout<<"Error in dyn_variables::adopt_adiabatic(): rep_tdse = "<<rep<<" has no branch here (implemented: 0, 1, 2, 3)\n";
    std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
  }
}

void dyn_variables::derive_adiabatic(dyn_control_params& dyn_params, nHamiltonian& frame){
/**
  The adiabatic variable derived from the propagated diabatic one in the basis of the given Hamiltonian (its
  per-trajectory children): C_adi = U^H S C_dia under rep_tdse 0, rho_adi = (S U)^H rho_dia (S U) under 2 - the frame
  of the amplitude propagation within a step is the previous step's Hamiltonian until the last substep's relabel.
  Under rep_tdse 1 and 3 the adiabatic variable is the propagated one and nothing is done
*/
  int rep = dyn_params.rep_tdse;
  if(rep==1 || rep==3){ return; }
  if(rep!=0 && rep!=2){
    std::cout<<"Error in dyn_variables::derive_adiabatic(): rep_tdse = "<<rep<<" has no branch here (implemented: 0, 1, 2, 3)\n";
    std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
  }
  for(int traj=0; traj<ntraj; traj++){
    nHamiltonian* child = frame.child(traj);
    CMATRIX SU(child->get_basis_transform());
    if(child->ovlp_dia_supplied){ SU = child->get_ovlp_dia() * SU; }
    if(rep==0){
      CMATRIX ca(nadi, 1);
      ca.product(adjoint(SU), ampl_dia->col(traj));
      for(int i=0;i<nadi;i++){ ampl_adi->set(i, traj, ca.get(i,0)); }
    }
    else{
      dm_adi[traj]->product(adjoint(SU), *dm_dia[traj], SU);
    }
  }
}

void dyn_variables::adopt_adiabatic(dyn_control_params& dyn_params, nHamiltonian& frame){
/**
  The write-through of adopt_adiabatic(dyn_params) in the basis of the given Hamiltonian (its per-trajectory
  children): C_dia = U C_adi under rep_tdse 0, rho_dia = U rho_adi U^H under 2 - the frame the adiabatic variable was
  derived in (derive_adiabatic). Under rep_tdse 1 and 3 nothing is done
*/
  int rep = dyn_params.rep_tdse;
  if(rep==1 || rep==3){ return; }
  if(rep!=0 && rep!=2){
    std::cout<<"Error in dyn_variables::adopt_adiabatic(): rep_tdse = "<<rep<<" has no branch here (implemented: 0, 1, 2, 3)\n";
    std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
  }
  for(int traj=0; traj<ntraj; traj++){
    CMATRIX U(frame.child(traj)->get_basis_transform());
    if(rep==0){
      CMATRIX cd(ndia, 1);
      cd.product(U, ampl_adi->col(traj));
      for(int i=0;i<ndia;i++){ ampl_dia->set(i, traj, cd.get(i,0)); }
    }
    else{
      dm_dia[traj]->product(U, *dm_adi[traj], adjoint(U));
    }
  }
}

void dyn_variables::adopt_diabatic(dyn_control_params& dyn_params){
/**
  The write-through of a correction made on the diabatic variable (the diabatic instantaneous decoherence): under
  rep_tdse 1 the propagated adiabatic amplitudes take the corrected diabatic ones (C_adi = U^H S C_dia, the rep_tdse 0
  direction of update_amplitudes), under rep_tdse 3 the propagated adiabatic density takes the corrected diabatic one
  (rho_adi = (S U)^H rho_dia (S U), the rep_tdse 2 direction of update_density_matrix) - the mirror of adopt_adiabatic.
  Under rep_tdse 0 and 2 the diabatic variable is the propagated one and nothing is done
*/
  int rep = dyn_params.rep_tdse;
  if(rep==0 || rep==2){ return; }
  if(rep==1){
    CMATRIX cd(ndia, 1);
    CMATRIX ca(nadi, 1);
    for(int traj=0; traj<ntraj; traj++){
      CMATRIX& U = (*basis_transform[traj]);
      cd = ampl_dia->col(traj);
      if(ovlp_dia_is_identity[traj]){ ca.product(adjoint(U), cd); }
      else{ ca.product(adjoint(U), (*ovlp_dia[traj]) * cd); }
      for(int i=0;i<nadi;i++){ ampl_adi->set(i, traj, ca.get(i,0) ); }
    }
  }
  else if(rep==3){
    for(int traj=0; traj<ntraj; traj++){
      CMATRIX& U = (*basis_transform[traj]);
      if(ovlp_dia_is_identity[traj]){ dm_adi[traj]->product(adjoint(U), *dm_dia[traj], U); }
      else{
        CMATRIX su((*ovlp_dia[traj]) * U);
        dm_adi[traj]->product(adjoint(su), *dm_dia[traj], su);
      }
    }
  }
  else{
    std::cout<<"Error in dyn_variables::adopt_diabatic(): rep_tdse = "<<rep<<" has no branch here (implemented: 0, 1, 2, 3)\n";
    std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
  }
}

void dyn_variables::update_density_matrix(dyn_control_params& dyn_params){
/**
    """

    Update the density matrices in diabatic and adiabatic representations

    Args:
        * **dyn_params** - contains the parameters of transformation

        rep ( int ): a selector of which representation is considered main (being propagated)
            E.g. if rep = 0 - that means we propagate the diabatic coefficients, that is the calculation
            of the diabatic density matrix is straightforward, but we need to involve some transformations
            to compute the adiabatic density matrix and vice versa, if rep = 1, the propagation is done according
            to the adiabatic properties and we'd need to convert to the diabatic representation in the end

            - 0: diabatic
            - 1: adiabatic

    Notes:
           Since the basis_transform corresponding to the transformation of the diabatic-to-raw-adiabatic states: 
           |\psi_adi (t) > = | \psi_dia (t)> U
           |\psi\tilde_adi (t) > = |\psi_adi (t) > T =  | \psi_dia (t)> U * T, so the effective U is U_eff = U * T
          
           T is the basis re-projection matrix

    This function should be used only after the proj_adi = T has been computed and the basis_transform has been set

    """
*/

  int rep = dyn_params.rep_tdse;
  // In the adiabatic representations (rep_tdse 1 and 3) dm_dia is a derived
  // quantity costing two nadi x nadi matrix products per trajectory - skip it
  // when nothing consumes it: compute_dm_dia == 0 - the "auto" value (-1) is
  // resolved to 0/1 in dynamics/tsh/compute.py from rep_tdse, rep_sh and the
  // requested outputs ("D_dia")
  int skip_dia = (dyn_params.compute_dm_dia == 0) && (rep == 1 || rep == 3);

  // dm_dia is the PLAIN-coefficient density C C^H and its diagonal the
  // plain |C_i|^2 - exact populations for orthonormal diabats - with
  // ovlp_dia != I the S-metric variants derive in post (Re diag(dm S)),
  // and the dia->adi directions here carry the cached metric (C_adi = U^H S C_dia,
  // rho_adi = (S U)^H rho_dia (S U)). The diabatic hop machinery (rep_sh 0) works
  // in the Lowdin frame S^-1/2 of the diabats (nHamiltonian::lowdin_dia)
  // Deterministic per-trajectory updates - parallel-safe (each iteration
  // writes only its own dm slots) - results identical for any thread count
  // exceptions cannot leave an OMP-outlined region (even serialized) -
  // capture and rethrow outside so the guards stay translatable
  std::exception_ptr omp_eptr = nullptr;
  #pragma omp parallel for num_threads(dyn_params.omp_num_threads) \
      if (dyn_params.omp_num_threads > 1)
  for(int traj=0; traj<ntraj; traj++){
    try {

    CMATRIX cd(ndia, 1);
    CMATRIX ca(nadi, 1);
    CMATRIX& U = (*basis_transform[traj]); // * (*proj_adi[indx]); // U_eff - let's wait with this

    /// Diabatic wfc representation
    // dm_dia is the PLAIN-coefficient density C C^H in every branch - the
    // cached diabatic metric makes the dia->adi directions exact for
    // S != I (identity metric = the historical bare-U form)
    if(rep==0){
      cd = ampl_dia->col(traj);
      dm_dia[traj]->product(cd, adjoint(cd));
      if(ovlp_dia_is_identity[traj]){ ca.product(adjoint(U), cd); }
      else{ ca.product(adjoint(U), (*ovlp_dia[traj]) * cd); }
      dm_adi[traj]->product(ca, adjoint(ca));
    }
    /// Adiabatic wfc representation
    else if(rep==1){
      ca = ampl_adi->col(traj);
      dm_adi[traj]->product(ca, adjoint(ca));
      if(!skip_dia){
        dm_dia[traj]->product(U, *dm_adi[traj], adjoint(U));
      }
    }
    /// Diabatic DM representation
    else if(rep==2){
      if(ovlp_dia_is_identity[traj]){ dm_adi[traj]->product(adjoint(U), *dm_dia[traj], U); }
      else{
        CMATRIX su0((*ovlp_dia[traj]) * U);
        dm_adi[traj]->product(adjoint(su0), *dm_dia[traj], su0);
      }
    }
    /// Adiabatic DM representation
    else if(rep==3){
      if(!skip_dia){
        dm_dia[traj]->product(U, *dm_adi[traj], adjoint(U));
      }
    }
    else{
      std::cout<<"Error in dyn_variables::update_density_matrix(): rep = "<<rep<<" has no branch here (implemented: 0, 1, 2, 3)\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }

    } catch (...) {
      #pragma omp critical
      { if (!omp_eptr) { omp_eptr = std::current_exception(); } }
    }
  }// for traj
  if (omp_eptr) { std::rethrow_exception(omp_eptr); }

}


void dyn_variables::update_density_matrix(dyn_control_params& dyn_params, nHamiltonian& ham, int lvl){
/**
    """

    Compute the trajectory-averaged density matrices in diabatic
    or adiabatic representations

    Args: 
        ham ( nHamiltonian ): object that handles Hamiltonian-related calculations with many trajectories
        Cdia ( CMATRIX(ndia, ntraj) ): amplitudes of diabatic states in the TD wavefunction expansion
        Cadi ( CMATRIX(ndia, ntraj) ): amplitudes of adiabatic states in the TD wavefunction expansion
        projectors ( list of CMATRIX(nst, nst)): dynamically-consistent corrections   
        rep ( int ): a selector of which representation is considered main (being propagated)
            E.g. if rep = 0 - that means we propagate the diabatic coefficients, that is the calculation 
            of the diabatic density matrix is straightforward, but we need to involve some transformations 
            to compute the adiabatic density matrix and vice versa, if rep = 1, the propagation is done according
            to the adiabatic properties and we'd need to convert to the diabatic representation in the end
             
            - 0: diabatic
            - 1: adiabatic

        lvl ( int ): The level of the Hamiltonian that treats the transformations:
            - 0: ham is the actual Hamiltonian to use (use with single trajectory),
            - 1: ham is the parent of the Hamiltonians to use (use with multiple trajectories)

        isNBRA ( int ): The flag for NBRA type calculations:
            - 0: the Hamiltonian related properties are computed for all of the trajectories [default]
            - 1: the Hamiltonian related properties are computed only for one trajectory 

    Returns:
        tuple: ( dm_dia, dm_adi ):

            * dm_dia ( CMATRIX(ndia, ndia) ): the trajectory-averaged density matrix in
                the diabatic representation. Here, ndia - is the number of diabatic basis
                states
            * dm_adi ( CMATRIX(nadi, nadi) ): the trajectory-averaged density matrix in
                the adiabatic representation. Here, nadi - is the number of adiabatic basis
                states

    """
*/

  CMATRIX su(ndia, nadi);
  CMATRIX S(ndia, ndia);
  CMATRIX U(ndia, nadi);
  CMATRIX cd(ndia, 1);
  CMATRIX ca(nadi, 1);

  //cout<<"In update density matrix\n";
  vector<int> indx;
  if(lvl==0){ indx = vector<int>(1, 0); }
  else if(lvl==1) { indx = vector<int>(2, 0); }

  int rep = dyn_params.rep_tdse;
  // In the adiabatic representations (rep_tdse 1 and 3) dm_dia is a derived
  // quantity costing two nadi x nadi matrix products per trajectory per step.
  // Skip it when nothing consumes it: compute_dm_dia == 0 - the "auto" value
  // (-1) is resolved to 0/1 in dynamics/tsh/compute.py from rep_tdse, rep_sh
  // and the requested outputs ("D_dia")
  int skip_dia = (dyn_params.compute_dm_dia == 0) && (rep == 1 || rep == 3);
  // With a single child Hamiltonian (NBRA) S, U and su are the same for
  // every trajectory - fetch and multiply them once
  int nchildren = ham.children.size();
  int shared = (lvl == 0) || (nchildren == 1);
  if(shared && !skip_dia){
    S = ham.get_ovlp_dia(indx);
    U = ham.get_basis_transform(indx);
    su = S * U;
  }

  for(int traj=0; traj<ntraj; traj++){

    if(!shared && !skip_dia){
      // NBRA runs carry a single child Hamiltonian shared by all trajectories
      if(lvl==1){  indx[1] = (traj < nchildren) ? traj : 0; }
      S = ham.get_ovlp_dia(indx);
      U = ham.get_basis_transform(indx);
      su = S * U;
    }

    /// Diabatic wfc representation
    // dm_dia is the PLAIN-coefficient density C C^H in every branch (its
    // diagonal = |C_i|^2, exact populations for orthonormal diabats - the
    // S-metric variants derive in post) - S enters only FUNCTIONALLY in
    // the dia<->adi conversions: C_adi = (S U)^H C_dia, C_dia = U C_adi
    if(rep==0){
      cd = ampl_dia->col(traj);
      dm_dia[traj]->product(cd, adjoint(cd));
      ca.product(adjoint(su), cd);
      dm_adi[traj]->product(ca, adjoint(ca));
    }
    /// Adiabatic wfc representation
    else if(rep==1){
      ca = ampl_adi->col(traj);
      dm_adi[traj]->product(ca, adjoint(ca));
      if(!skip_dia){
        dm_dia[traj]->product(U, *dm_adi[traj], adjoint(U));
      }
    }
    /// Diabatic DM representation
    else if(rep==2){
      dm_adi[traj]->product(adjoint(su), *dm_dia[traj], su);
    }
    /// Adiabatic DM representation
    else if(rep==3){
      if(!skip_dia){
        dm_dia[traj]->product(U, *dm_adi[traj], adjoint(U));
      }
    }
    else{
      std::cout<<"Error in dyn_variables::update_density_matrix(): rep = "<<rep<<" has no branch here (implemented: 0, 1, 2, 3)\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }

  }// for traj

  
}

void dyn_variables::update_density_matrix(bp::dict dyn_params, nHamiltonian& ham, int lvl){

  dyn_control_params _prms;
  _prms.set_parameters(dyn_params);

  update_density_matrix(_prms, ham, lvl);

  //exit(0);
}



void dyn_variables::update_density_matrix(dyn_control_params& dyn_params, bp::object compute_model, bp::dict model_params, int lvl){

  //# Prepare the Hamiltonian's hierarchy
  nHamiltonian ham(ndia, nadi, ndof);
  ham.add_new_children(ndia, nadi, ndof, ntraj);
  ham.init_all(1,1);

  //# Compute the Hamiltonian transformation
  update_Hamiltonian_variables(dyn_params, *this, ham, ham, compute_model, model_params, 0);
  //update_Hamiltonian_q(dyn_params, *q, ham, compute_model, model_params);
  update_density_matrix(dyn_params, ham, lvl);

}


void dyn_variables::update_density_matrix(bp::dict dyn_params, bp::object compute_model, bp::dict model_params, int lvl){

  dyn_control_params _prms;
  _prms.set_parameters(dyn_params);


  //# Prepare the Hamiltonian's hierarchy
  nHamiltonian ham(ndia, nadi, ndof);
  ham.add_new_children(ndia, nadi, ndof, ntraj);
  ham.init_all(1,1);

  //# Compute the Hamiltonian transformation
  //update_Hamiltonian_q(_prms, *q, ham, compute_model, model_params);
  update_Hamiltonian_variables(_prms, *this, ham, ham, compute_model, model_params, 0);

  update_density_matrix(_prms, ham, lvl);

}


//bp::dict dyn_variables::init_electronic_dyn_var(bp::dict& _params, Random& rnd){
void dyn_variables::init_amplitudes(bp::dict _params, Random& rnd){
/**
    """
    Args:

        params ( dictionary ): control parameters
 
            * **params["init_type"]** ( int ): the type of sampling of electronic DOFs
     
                - 0 : initialize all states according to "istate" and sets all
                    amplitudes to 1.0  [ default ]
 
                - 1 : initialize all states according to "istate" but sets 
                    amplitudes to exp(-2*pi*i*rnd), where rnd is a random 
                    number uniformly distributed on the [0, 1] interval

                - 2 : initialize all states according to "istates" - the 
                    integer indices are selected randomly according to populations provided in
                    variable "istates", the amplitudes are set to be sqrt(istates[i]), but
                    their phases are identical (like in the option 0)

                - 3 : initialize all states according to "istates" - the 
                    integer indices are selected randomly according to populations provided in
                    variable "istates", the amplitudes are set to be sqrt(istates[i]), but
                    their phases are set to exp(-2*pi*i*rnd), where rnd is a random 
                    number uniformly distributed on the [0, 1] interval (line in option 1)

                - 4 : initialize all states according to Voronoi-style sampling 
                    Designed for MASH dynamics
                    Uses "istates" instead of the "istate", similar to the types 2 and 3
                - 5 : SQC square-window sampling of the Meyer-Miller actions around "istate"
                    (Cotton-Miller 2013): |c_k|^2 = e_k, occupied e in [1, 1 + 2 gamma], the
                    others in [0, 2 gamma], random phases - gamma = 0.366 unless "sqc_gamma" is given.
                    NOT normalized - use with force_method 2, ehrenfest_normalized 0 and the same sqc_gamma
                - 6 : SQC triangle-window sampling (Cotton-Miller 2016/2019): occupied e in [1, 2],
                    every other e_k in [0, 2 - e_occ], uniform over the region - gamma = 1/3 unless given

            * **params["nstates"]** ( int ): the number of electronic states in the basis
                [ default: 1 ]

            * **params["istate"]** ( int ): the index of the initial electronic state, used 
                only when **params["init_type"]** is 0 or 1, in which case it defines on 
                which state the amplitudes will be initialized [ default: 0]

            * **params["sqc_gamma"]** ( double ): the Meyer-Miller gamma of the SQC windows (init_type 5/6) -
                -1 (default) selects the window's standard value (0.366 square, 1/3 triangle)

            * **params["istates"]** ( list of ints ): the list of the populations on all electronic
                states included in the basis, used only when **params["init_type"]** is 2 or 3, 
                in which case it defines on which states the amplitudes will be initialized.
                The length of this list should be consistent with ```nstates``` variable. And the sum
                of all entries should be 1.0 [ default: [1.0], meaning that only the lowest state
                is occupied ]

            * **params["rep"]** ( int ): defines for which repersentation we generate the amplitudes
 
                - 0 : diabatic, the corresponding matrix is initialized, the adiabatic amplitudes are zeroes
                - 1 : adiabatic, the corresponding matrix is initialized, the diabatic amplitudes are zeroes [ default ]

            * **params["ntraj"]** ( int ): the number of trajectories - the parameter defines the
                number of columns the output matrices will have [ default: 1]

            * **params["is_nbra"]** (int): A flag for NBRA type calculations. If it is set to 1 then
                                          the Hamiltonian related properties are only computed for one trajectory [ default : 0]

        rnd ( Random ): random numbers generator object


    Returns:
        Cdia, Cadi, states:  where:

            * Cdia ( CMATRIX(nstates, ntraj) ) : amplitudes on all diabatic states for all trajectories
            * Cadi ( CMATRIX(nstates, ntraj) ) : amplitudes on all adiabatic states for all trajectories
            * states ( list of ints ) : state indices for each trajectory

    """
*/

  //# Read the parameters
  bp::list critical_params; 
  bp::dict default_params;
  bp::dict params(_params);

  default_params["init_type"] = 0;
  default_params["istate"] = 0;
  bp::list lst; lst.append(1.0);
  default_params["istates"] = lst;
  default_params["rep"] = 1;
  default_params["is_nbra"] = 0;
  default_params["verbosity"] = 0;
  default_params["sqc_window"] = 0;
  default_params["sqc_gamma"] = -1.0;

  check_input(params, default_params, critical_params);


  int i, traj;
  double ksi;
  int init_type = -1;
  int istate = -1;
  vector<double> istates;
  int rep = -1;
  //int is_nbra = -1; 
  int verbosity = 0;
  int sqc_window = 0;
  double sqc_gamma = -1.0;

  std::string key;
  for(int i=0;i<len(params.values());i++){
    key = bp::extract<std::string>(params.keys()[i]);

    ///================= Computing Hamiltonian-related properties ====================
    if(key=="init_type") {  init_type = bp::extract<int>(params.values()[i]); }
    else if(key=="istate") {  istate = bp::extract<int>(params.values()[i]); }
    else if(key=="istates") {  istates = liblibra::libconverters::Py2Cpp<double>( bp::extract< bp::list >(params.values()[i]) ); }
    else if(key=="rep") {  rep = bp::extract<int>(params.values()[i]); }
    //else if(key=="is_nbra") {  is_nbra = bp::extract<int>(params.values()[i]); }
    else if(key=="verbosity") {  verbosity = bp::extract<int>(params.values()[i]); }
    else if(key=="sqc_window") {  sqc_window = bp::extract<int>(params.values()[i]); }
    else if(key=="sqc_gamma") {  sqc_gamma = bp::extract<double>(params.values()[i]); }

  }


  ///# Sanity check
  if(! ((rep==0) || (rep==1)) ){
    cout<<"WARNINIG in init_amplitudes:\
           the rep = "<<rep<<" is not known. Allowed values are: [0, 1]\n";
  }

  if(! (init_type==0 || init_type==1 || init_type==2 || init_type==3 || init_type==4 ||
        init_type==5 || init_type==6)){
    cout<<"WARNINIG in init_amplitudes:\
           the init_type = "<<init_type<<" is not known. Allowed values are: [0, 1, 2, 3, 4, 5, 6]\n";
  }

  // SQC window sampling: the window fixes gamma unless given explicitly
  if(init_type==5 || init_type==6){
    if(sqc_gamma < 0.0){ sqc_gamma = (init_type==5) ? 0.366 : 1.0/3.0; }
    sqc_window = (init_type==5) ? 0 : 1;
    this->sqc_window = sqc_window;
    this->sqc_gamma = sqc_gamma;
  }

  if(init_type==0 || init_type==1 || init_type==5 || init_type==6){
    if(istate>=ndia && rep==0){
      cout<<"ERROR in init_amplitudes: the istate is= "<<istate<<", but should be less than "<<ndia<<"\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }
    if(istate>=nadi && rep==1){
      cout<<"ERROR in init_amplitudes: the istate is= "<<istate<<", but should be less than "<<nadi<<"\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }
  }

  if(init_type==2 || init_type==3){
    if(istates.size()!=ndia && rep==0){
      cout<<"ERROR in init_amplitudes: the istates array is of length = "<<istates.size()<<", but should be of length "<<ndia<<"\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }
    if(istates.size()!=nadi && rep==1){
      cout<<"ERROR in init_amplitudes: the istates array is of length = "<<istates.size()<<", but should be of length "<<nadi<<"\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }

    double summ = 0.0;
    for(i=0; i<istates.size(); i++){  summ += istates[i]; }
    if( fabs(summ - 1.0) > 1e-5 ){
      cout<<"ERROR in init_amplitudes: the sum of the entries in the istates array is "<<summ<<", but should be 1.0\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }
  }

  //# Dynamical variables
  for(traj=0; traj<ntraj; traj++){

    //========================== First, let's set up the amplitudes =======================================
    if(init_type==0){
      if(verbosity > 0){
        cout<<"======= Initialization type is "<<init_type<<" ========\n";
        cout<<"setting representation "<<rep<<" coefficient C_"<<istate<<" to 1.0\n";
      }

      if(rep==0){ ampl_dia->set(istate, traj, complex<double>(1.0, 0.0) ); }  
      else if(rep==1){ ampl_adi->set(istate, traj, complex<double>(1.0, 0.0) );  }

    }// init_type==0

    else if(init_type==1){
      if(verbosity > 0){
        cout<<"======= Initialization type is "<<init_type<<" ========\n";
        cout<<"setting representation "<<rep<<" coefficient to complex C_"<<istate<<" such that |C_"<<istate<<"}|^2 = 1.0\n";
      }

      ksi = rnd.uniform(0.0, 1.0);
      complex<double> ampl(cos(2.0*M_PI*ksi), sin(2.0*M_PI*ksi) );

      if(rep==0){ ampl_dia->set(istate, traj, ampl );   }
      else if(rep==1){ ampl_adi->set(istate, traj, ampl );  }

    }// init_type==1

    else if(init_type==2){
      if(verbosity > 0){
        cout<<"======= Initialization type is "<<init_type<<" ========\n";
        cout<<"setting representation "<<rep<<" coefficients C_i for all i to sqrt( target populations) \n";

      }

      for(i=0; i<istates.size(); i++){
          double ampl = sqrt( istates[i] );

          if(rep==0){ ampl_dia->set(i, traj, complex<double>(ampl, 0.0) );  }
          else if(rep==1){ ampl_adi->set(i, traj, complex<double>(ampl, 0.0) );  }
      }

    }// init_type==2

    else if(init_type==3){
      if(verbosity > 0){
        cout<<"======= Initialization type is "<<init_type<<" ========\n";
        cout<<"setting representation "<<rep<<" coefficients C_i for all i to complex numbers such that |C_i|^2  = target populations \n";

      }

      for(i=0; i<istates.size(); i++){
          ksi = rnd.uniform(0.0, 1.0);
          complex<double> ampl(cos(2.0*M_PI*ksi), sin(2.0*M_PI*ksi) );
          ampl = ampl * sqrt( istates[i] );

          if(rep==0){ ampl_dia->set(i, traj, ampl );  }
          else if(rep==1){ ampl_adi->set(i, traj, ampl);  }
      }

    }// init_type==3

    else if(init_type==4){
      if(verbosity > 0){
        cout<<"======= Initialization type is "<<init_type<<" ========\n";
        cout<<"setting representation "<<rep<<" coefficients C_i for all i to complex numbers such that populations are sampled evenly \n";
      }

      istates = rnd.voron(istates.size(), istate);
      for(i=0; i<istates.size(); i++){
        cout<<"Voron_mag["<<i<<"]= "<<istates[i]<<endl;
        ksi = rnd.uniform(0.0, 1.0);
        complex<double> ampl(cos(2.0*M_PI*ksi), sin(2.0*M_PI*ksi) );
        ampl = ampl * sqrt( istates[i] );
        cout<<"Voron_amp["<<i<<"]= "<<ampl<<endl;

        if(rep==0){ ampl_dia->set(i, traj, ampl );  }
        else if(rep==1){ ampl_adi->set(i, traj, ampl);  }
      }
    }

    else if(init_type==5 || init_type==6){
      // Cotton-Miller SQC window sampling of the Meyer-Miller actions: the
      // shifted actions e_k = n_k + gamma = |c_k|^2 (c = (x + i p)/sqrt2), phases
      // uniform on [0, 2 pi). Square (init_type 5, gamma 0.366): occupied
      // e in [1, 1 + 2 gamma], the others in [0, 2 gamma]. Triangle (init_type 6,
      // gamma 1/3): occupied e in [1, 2], every other e_k in [0, 2 - e_occ] -
      // uniform over the region, sampled by rejection from the box. The
      // amplitudes are NOT normalized (|c|^2 = 1 + N gamma) - pair with
      // ehrenfest_normalized 0 and the same sqc_gamma in the dynamics
      int nst = (rep==0) ? ndia : nadi;
      if(istate < 0 || istate >= nst){
        cout<<"ERROR in init_amplitudes: the istate is= "<<istate<<", but should be in [0, "<<nst<<")\n";
        std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
      }
      if(verbosity > 0){
        cout<<"======= Initialization type is "<<init_type<<" ========\n";
        cout<<"SQC window sampling ("<<(sqc_window==0 ? "square" : "triangle")<<", gamma = "<<sqc_gamma<<") around state "<<istate<<"\n";
      }
      vector<double> e(nst, 0.0);
      if(sqc_window==0){
        for(i=0; i<nst; i++){
          double lo = (i==istate) ? 1.0 : 0.0;
          e[i] = lo + 2.0 * sqc_gamma * rnd.uniform(0.0, 1.0);
        }
      }
      else{
        int accepted = 0;
        while(!accepted){
          e[istate] = 1.0 + rnd.uniform(0.0, 1.0);
          accepted = 1;
          for(i=0; i<nst; i++){
            if(i==istate){ continue; }
            e[i] = rnd.uniform(0.0, 1.0);
            if(e[i] > 2.0 - e[istate]){ accepted = 0; }
          }
        }
      }
      for(i=0; i<nst; i++){
        ksi = rnd.uniform(0.0, 1.0);
        complex<double> ampl(cos(2.0*M_PI*ksi), sin(2.0*M_PI*ksi) );
        ampl = ampl * sqrt( e[i] );
        if(rep==0){ ampl_dia->set(i, traj, ampl );  }
        else if(rep==1){ ampl_adi->set(i, traj, ampl);  }
      }
    }// init_type==5 || init_type==6

  }// for traj

  update_mm_variables();

  if(verbosity > 1){
      cout<<"========== ampl_dia ===============\n";
      ampl_dia->show_matrix();
      cout<<"========== ampl_adi ===============\n";
      ampl_adi->show_matrix();
      cout<<"========== states ===============\n";
  }



  //return params;

}



void dyn_variables::init_density_matrix(bp::dict _params){

  //# Read the parameters
  bp::list critical_params; 
  bp::dict default_params;
  bp::dict params(_params);

  default_params["init_dm_type"] = 0; /// 0 - based on amplitudes; 1 - using direct input
  default_params["verbosity"] = 0;
  default_params["extern_dm"] = CMATRIX(nadi, nadi); /// 0 - based on amplitudes; 1 - using direct input


  check_input(params, default_params, critical_params);


  int i, traj;
  double ksi;
  int init_dm_type = -1;
  int rep = -1;
  int verbosity = 0;
  //CMATRIX extern_dm_dia();

  std::string key;
  for(int i=0;i<len(params.values());i++){
    key = bp::extract<std::string>(params.keys()[i]);

    ///================= Computing Hamiltonian-related properties ====================
    if(key=="init_dm_type") {  init_dm_type = bp::extract<int>(params.values()[i]); }
    else if(key=="rep") {  rep = bp::extract<int>(params.values()[i]); }
    else if(key=="verbosity") {  verbosity = bp::extract<int>(params.values()[i]); }

  }


  ///================= Sanity check ================================
  if(! ((rep==0) || (rep==1)) ){
    cout<<"WARNINIG in init_density_matrix:\
           the rep = "<<rep<<" is not known. Allowed values are: [0, 1]\n";
  }

  if(! (init_dm_type==0 || init_dm_type==1 )){
    cout<<"WARNINIG in init_density_matrix:\
           the init_dm_type = "<<init_dm_type<<" is not known. Allowed values are: [0, 1]\n";
  }




  ///================= Actual calculations  ================================

  //# Dynamical variables
  for(traj=0; traj<ntraj; traj++){

    // From the amplitudes
    if(init_dm_type==0){
      if(verbosity > 0){
        cout<<"======= Initialization type is "<<init_dm_type<<" ========\n";
        cout<<"setting DM in representation "<<rep<<" using wfc amplitudes \n";
      }

      if(rep==0){ 
        CMATRIX C(ndia, 1); 
        C = ampl_dia->col(traj);
        *dm_dia[traj] = C * C.H(); 
      }  
      else if(rep==1){ 
        CMATRIX C(nadi, 1); 
        C = ampl_adi->col(traj);
        *dm_adi[traj] = C * C.H(); 
      }

    }// init_dm_type==0


    if(init_dm_type==1){
      if(verbosity > 0){
        cout<<"======= Initialization type is "<<init_dm_type<<" ========\n";
        cout<<"setting DM in representation "<<rep<<" using provided input \n";
      }

      if(rep==0){ 
        for(int i=0;i<len(params.values());i++){
          key = bp::extract<std::string>(params.keys()[i]);
          if(key=="extern_dm") {  *dm_dia[traj] = bp::extract<CMATRIX>(params.values()[i]); }
        }// for i
      }// rep == 0 
      else if(rep==1){ 
        for(int i=0;i<len(params.values());i++){
          key = bp::extract<std::string>(params.keys()[i]);
          if(key=="extern_dm") {  *dm_adi[traj] = bp::extract<CMATRIX>(params.values()[i]); }
        }// for i
      }// rep == 1 

    }// init_dm_type == 1
 
  }// for traj
  

  //return params;
}


// Y = S^1/2 of a trajectory's cached metric for the hopping-frame sandwiches of the samplers
// below: 1 with Y filled for a non-identity metric, 0 (Y untouched) for the identity, where the
// callers keep the bare transform
static int lowdin_root(CMATRIX* S, int identity, CMATRIX& Y){
  if(identity){ return 0; }
  CMATRIX X_(S->n_cols, S->n_cols);
  libmeigen::sqrt_matrix(*S, Y, X_);
  return 1;
}

void dyn_variables::init_active_states(bp::dict _params, Random& rnd){

  //# Read the parameters
  bp::list critical_params; 
  bp::dict default_params;
  bp::dict params(_params);


  default_params["init_type"] = 0;
  default_params["istate"] = 0;
  bp::list lst; lst.append(1.0);
  default_params["istates"] = lst;
  default_params["rep"] = 1;
  default_params["verbosity"] = 0;

  check_input(params, default_params, critical_params);

  int i, traj;
  double ksi;
  int init_type = -1;
  int istate = -1;
  vector<double> istates;
  int rep = -1;
  int verbosity = 0;


  std::string key;
  for(int i=0;i<len(params.values());i++){
    key = bp::extract<std::string>(params.keys()[i]);

    ///================= Computing Hamiltonian-related properties ====================
    if(key=="init_type") {  init_type = bp::extract<int>(params.values()[i]); }
    else if(key=="istate") {  istate = bp::extract<int>(params.values()[i]); }
    else if(key=="istates") {  istates = liblibra::libconverters::Py2Cpp<double>( bp::extract< bp::list >(params.values()[i]) ); }
    else if(key=="rep") {  rep = bp::extract<int>(params.values()[i]); }
    else if(key=="verbosity") {  verbosity = bp::extract<int>(params.values()[i]); }

  }


  ///================= Sanity check ================================
  if(! ((rep==0) || (rep==1)) ){
    cout<<"WARNINIG in init_active_states:\
           the rep = "<<rep<<" is not known. Allowed values are: [0, 1]\n";
  }

  if(! (init_type==0 || init_type==1 || init_type==2 || init_type==3 || init_type==4 || init_type==5 || init_type==6) ){
    cout<<"WARNINIG in init_active_states:\
           the init_type = "<<init_type<<" is not known. Allowed values are: [0, 1, 2, 3, 4]\n";
  }
  
  if(init_type==0 || init_type==1 || init_type==4 || init_type==5 || init_type==6){
    if(istate>=ndia && rep==0){
      cout<<"ERROR in init_active_states: the istate is= "<<istate<<", but should be less than "<<ndia<<"\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }
    if(istate>=nadi && rep==1){
      cout<<"ERROR in init_active_states: the istate is= "<<istate<<", but should be less than "<<nadi<<"\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }
  }

  if(init_type==2 || init_type==3){
    if(istates.size()!=ndia && rep==0){
      cout<<"ERROR in init_active_states: the istates array is of length = "<<istates.size()<<", but should be of length "<<ndia<<"\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }
    if(istates.size()!=nadi && rep==1){
      cout<<"ERROR in init_active_states: the istates array is of length = "<<istates.size()<<", but should be of length "<<nadi<<"\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }

    double summ = 0.0;
    for(i=0; i<istates.size(); i++){  summ += istates[i]; }
    if( fabs(summ - 1.0) > 1e-5 ){
      cout<<"ERROR in init_active_states: the sum of the entries in the istates array is "<<summ<<", but should be 1.0\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }
  }



  ///================= Actual calculations  ================================

  act_states.clear();  

  //================ For adiabatic - set up them directly =================
  for(traj=0; traj<ntraj; traj++){
    if(init_type==0 || init_type==1 || init_type==4 || init_type==5 || init_type==6){ 
      act_states.push_back(istate);
    }
    else if(init_type==2 || init_type==3){
      ksi = rnd.uniform(0.0, 1.0);
      act_states.push_back( liblibra::libspecialfunctions::set_random_state(istates, ksi) );
    }
  }// for traj

  //============= For adiabatic: more complicated =========================
  if(rep==0){
    vector<int> dia_act_states_temp(act_states);

    CMATRIX pop_adi(nadi, nadi);
    CMATRIX pop_dia(ndia, ndia);

    for(traj=0; traj<ntraj; traj++){
      i = dia_act_states_temp[traj]; // active diabatic state
      pop_dia *= 0.0; pop_dia.set(i, i, complex<double>(1.0, 0.0) );

      // the adiabatic populations of the Lowdin state X e_i, the diabatic active state of the
      // hopping machinery: C_adi = U^H S X e_i = U^H Y e_i, the bare U^H e_i at S = I
      { CMATRIX Y_(ndia, ndia);
        int lowdin = lowdin_root(ovlp_dia[traj], ovlp_dia_is_identity[traj], Y_);
        pop_adi = lowdin ? (*basis_transform[traj]).H() * Y_ * pop_dia * Y_ * (*basis_transform[traj])
                         : (*basis_transform[traj]).H() * pop_dia * (*basis_transform[traj]); }

      vector<double> res(nadi, 0.0);
      for(i=0; i<nadi;i++){ res[i] = pop_adi.get(i,i).real(); }

      // Now select an adiabatic state according to the probabilities in the res
      ksi = rnd.uniform(0.0, 1.0);
      act_states[traj] = hop(res, ksi);  // this is the adiabatic state that corresponds to
                                         // the distribution of the diabatic populations


    }// for traj

  }// rep == 0

}


void dyn_variables::init_active_states_dia(bp::dict _params, Random& rnd){

  //# Read the parameters
  bp::list critical_params; 
  bp::dict default_params;
  bp::dict params(_params);


  default_params["init_type"] = 0;
  default_params["istate"] = 0;
  bp::list lst; lst.append(1.0);
  default_params["istates"] = lst;
  default_params["rep"] = 0;
  default_params["verbosity"] = 0;

  check_input(params, default_params, critical_params);

  int i, traj;
  double ksi;
  int init_type = -1;
  int istate = -1;
  vector<double> istates;
  int rep = -1;
  int verbosity = 0;


  std::string key;
  for(int i=0;i<len(params.values());i++){
    key = bp::extract<std::string>(params.keys()[i]);

    ///================= Computing Hamiltonian-related properties ====================
    if(key=="init_type") {  init_type = bp::extract<int>(params.values()[i]); }
    else if(key=="istate") {  istate = bp::extract<int>(params.values()[i]); }
    else if(key=="istates") {  istates = liblibra::libconverters::Py2Cpp<double>( bp::extract< bp::list >(params.values()[i]) ); }
    else if(key=="rep") {  rep = bp::extract<int>(params.values()[i]); }
    else if(key=="verbosity") {  verbosity = bp::extract<int>(params.values()[i]); }

  }


  ///================= Sanity check ================================
  if(! ((rep==0) || (rep==1)) ){
    cout<<"WARNINIG in init_active_states_dia:\
           the rep = "<<rep<<" is not known. Allowed values are: [0, 1]\n";
  }

  if(! (init_type==0 || init_type==1 || init_type==2 || init_type==3 || init_type==4 || init_type==5 || init_type==6) ){
    cout<<"WARNINIG in init_active_states_dia:\
           the init_type = "<<init_type<<" is not known. Allowed values are: [0, 1, 2, 3, 4]\n";
  }
  
  if(init_type==0 || init_type==1 || init_type==4 || init_type==5 || init_type==6){
    if(istate>=ndia && rep==0){
      cout<<"ERROR in init_active_states_dia: the istate is= "<<istate<<", but should be less than "<<ndia<<"\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }
    if(istate>=ndia && rep==1){
      cout<<"ERROR in init_active_states_dia: the istate is= "<<istate<<", but should be less than "<<ndia<<"\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }
  }

  if(init_type==2 || init_type==3){
    if(istates.size()!=ndia && rep==0){
      cout<<"ERROR in init_active_states_dia: the istates array is of length = "<<istates.size()<<", but should be of length "<<ndia<<"\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }
    if(istates.size()!=ndia && rep==1){
      cout<<"ERROR in init_active_states_dia: the istates array is of length = "<<istates.size()<<", but should be of length "<<ndia<<"\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }

    double summ = 0.0;
    for(i=0; i<istates.size(); i++){  summ += istates[i]; }
    if( fabs(summ - 1.0) > 1e-5 ){
      cout<<"ERROR in init_active_states_dia: the sum of the entries in the istates array is "<<summ<<", but should be 1.0\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }
  }



  ///================= Actual calculations  ================================

  act_states_dia.clear();  

  //================ For diabatic - set up them directly =================
  for(traj=0; traj<ntraj; traj++){
    if(init_type==0 || init_type==1 || init_type==4 || init_type==5 || init_type==6){ 
      act_states_dia.push_back(istate);
    }
    else if(init_type==2 || init_type==3){
      ksi = rnd.uniform(0.0, 1.0);
      act_states_dia.push_back( liblibra::libspecialfunctions::set_random_state(istates, ksi) );
    }
  }// for traj

  //============= For diabatic: more complicated =========================
  // the hopping states of non-orthogonal diabats are the Lowdin states S^-1/2 |i> and a pure diabat is a
  // superposition of them - the initial active state follows the Lowdin populations |(S^1/2 C_dia)_i|^2 of the
  // initial amplitudes, else the ensemble starts off the populations it is meant to track (0.023 off for a
  // 0.3 overlap) - at S = I a rep-1 request is mapped through the pure adiabatic state's diabatic populations,
  // one draw per trajectory either way
  vector<int> act_states_temp(act_states_dia);
  CMATRIX pop_adi(nadi, nadi);
  CMATRIX pop_dia(ndia, ndia);
  for(traj=0; traj<ntraj; traj++){
    CMATRIX X(ndia, ndia);
    CMATRIX Y(ndia, ndia);
    vector<double> res(ndia, 0.0);
    if(!ovlp_dia_is_identity[traj]){
      libmeigen::sqrt_matrix(*ovlp_dia[traj], Y, X);
      CMATRIX popL(Y * (*dm_dia[traj]) * Y);
      for(i=0; i<ndia; i++){ res[i] = popL.get(i,i).real(); }
    }
    else if(rep==1){
      i = act_states_temp[traj]; // active diabatic state
      pop_adi *= 0.0; pop_adi.set(i, i, complex<double>(1.0, 0.0) );
      pop_dia = (*basis_transform[traj]) * pop_adi * (*basis_transform[traj]).H();
      for(i=0; i<ndia;i++){ res[i] = pop_dia.get(i,i).real(); }
    }
    else{ continue; }
    ksi = rnd.uniform(0.0, 1.0);
    act_states_dia[traj] = hop(res, ksi);
  }// for traj
}

void dyn_variables::init_auxiliary_variables(bp::dict _params, Random& rnd){
/**
    """
    Args:
        y_aux_var ( list of doubles ): the mean values of coordinates for all DOFs [ units: a.u.]
        p_aux_var ( list of doubles ): the mean values of momenta for all DOFs [ units: a.u. ]
        m_aux_var ( list of doubles ): masses of all auxiliary DOFs [ units: a.u. ]

        params ( dictionary ): control parameters
 
            * **params["init_type"]** ( int ): the type of sampling of auxiliary DOFs
     
                - 0 : initialize identical copy of coordinates and momenta
 
        rnd ( Random ): random numbers generator object


    Returns:
        y_aux_var, p_aux_var, m_aux_var:  where:

            * y_aux_var ( vector<double>(1) ) : coordinates for all trajectories
            * p_aux_var ( vector<double>(1) ) : momenta for all trajectories
            * m_aux var ( vector<double>(1) ) : masses of all DOFs (same across the trajectories)

    """
*/

  //# Read the parameters
  bp::list critical_params; 
  bp::dict default_params;
  bp::dict params(_params);

  default_params["init_type"] = 0;
  //default_params["y_aux_var"] = vector<double>(1, 0.0);
  //default_params["p_aux_var"] = vector<double>(1, 0.0);
  //default_params["m_aux_var"] = vector<double>(1, 1.0);
  bp::list default_list1;
  default_list1.append(1.0);
  default_params["y_aux_var"] = default_list1;
  bp::list default_list2;
  default_list2.append(0.0);
  default_params["p_aux_var"] = default_list2;
  bp::list default_list3;
  default_list3.append(1.0);
  default_params["m_aux_var"] = default_list3;


  check_input(params, default_params, critical_params);


  int init_type;
  vector<double> _Y_aux_var;
  vector<double> _P_aux_var;
  vector<double> _M_aux_var;

  int idof;

  std::string key;
  for(int i=0;i<len(params.values());i++){
    key = bp::extract<std::string>(params.keys()[i]);

    ///================= Computing Hamiltonian-related properties ====================
    if(key=="init_type") {  init_type = bp::extract<int>(params.values()[i]); }
    else if(key=="y_aux_var") {  _Y_aux_var = liblibra::libconverters::Py2Cpp<double>( bp::extract< bp::list >(params.values()[i]) ); }
    else if(key=="p_aux_var") {  _P_aux_var = liblibra::libconverters::Py2Cpp<double>( bp::extract< bp::list >(params.values()[i]) ); }
    else if(key=="m_aux_var") {  _M_aux_var = liblibra::libconverters::Py2Cpp<double>( bp::extract< bp::list >(params.values()[i]) ); }
  }


  ///================= Sanity check ================================
  if(! (init_type==0) ){
    cout<<"WARNINIG in init_auxiliary_variables:\
           the init_type = "<<init_type<<" is not known. Allowed values are: [0]\n";
  }
  

  ///================= Actual calculations  ================================

  if(init_type==0){
    y_aux_var.push_back(_Y_aux_var[0]);
    p_aux_var.push_back(_P_aux_var[0]);
    m_aux_var.push_back(_M_aux_var[0]);
  }// init_type == 0

}

void dyn_variables::init_electronic_dyn_var(bp::dict _params, Random& rnd){

//  bp::dict res;

//  return res;

}


//vector<int> update_active_states(vector<int>& act_states, vector<CMATRIX*>& T){
void dyn_variables:: update_active_states(int direction, int property){
/*
  The projector T of the step (proj_adi, compute_projector on the time overlap St = <psi(t)|psi(t + dt)>) carries
  the amplitudes from the labels of the previous step to those of the new one exactly as the propagation does
  (propagate_electronic: C = T_ampl C, RHO = T_ampl RHO T_ampl^H) - the active state, a unit vector in the old
  labels, moves by the same T. T^H would move it the other way round: the same thing for a swap (T = T^T), the
  opposite cycle for three states that rotate their labels.


 
   direction = 1 - forward:  C_new = T C_old
              -1 - backward  C_old = T^H C_new

   property  = 0 - active state only
             = 1 - adiabatic amplitudes only 
             = 2 - both active states and adiabatic amplitudes

   act_states[itraj] - index of the active adiabatic state for the trajectory `itraj`
                       in terms of the energy-ordered (instantaneous) eign-states

   Under the label twins of the rotation projectors (state_tracking_algo -12, 52, 62) the active state moves by
   the assignment's permutation of the step (proj_perm) instead of the largest weight of its moved unit vector,
   which can send two states to one label at a near 50/50 mixing - the amplitudes move by T in both cases.
*/

  int i;
  int nst = nadi;
  CMATRIX coeff(nst, 1);
  MATRIX rho_tmp(nst,nst);
  complex<double> max_val;

  for(int itraj=0; itraj<ntraj; itraj++){

    //================ Active states ===================
    if(property==0 || property==2){
      if(label_assignment){
        // the assignment's permutation moves the label - forward old -> new, backward its inverse
        int act = act_states[itraj];
        if(direction==1){ act = proj_perm[itraj][act]; }
        else if(direction==-1){
          for(int j=0; j<nst; j++){ if(proj_perm[itraj][j]==act_states[itraj]){ act = j; break; } }
        }
        act_states[itraj] = act;
      }
      else{
      coeff = 0.0;
      coeff.set(act_states[itraj], 0, complex<double>(1.0, 0.0));

      if(direction==1){    coeff = (*proj_adi[itraj]) * coeff; }
      else if(direction==-1){  coeff = proj_adi[itraj]->H() * coeff; }

      //coeff.max_col_elt(0, max_val, act_states[itraj]);
      rho_tmp = (coeff * coeff.H()).real();
      //double max_val = fabs(rho_tmp.get(0,0)); 
      int max_val_indx = act_states[itraj];
      double max_val = fabs(rho_tmp.get(max_val_indx, max_val_indx));
      for(int j=0;j<nst;j++){ 
        // >= is important !!! not just >
        // on 2/15/2025:  actually, we need ">"
        if( fabs(rho_tmp.get(j,j)) > max_val){  max_val_indx = j; max_val = fabs(rho_tmp.get(j,j)); } 
      }
      act_states[itraj] = max_val_indx;
      }
    }

    //============ Amplitudes of states ===================
    if(property==1 || property==2){
      coeff = ampl_adi->col(itraj);

      if(direction==1){    coeff = (*proj_adi[itraj]) * coeff; }
      else if(direction==-1){  coeff = proj_adi[itraj]->H() * coeff; }

      for(i=0; i<nst; i++){ ampl_adi->set(i, itraj,  coeff.get(i, 0) ); }
    }

  }// for itraj
}

void dyn_variables::update_active_states(){
/**
  for backward-compatability
*/
   update_active_states(1, 0);
}



static int image_generator_seed(unsigned int s){
  /** The seed of the image generator from the main seed - mixed, so that the image draws of a run seeded k never replay the
      hop decisions of a run seeded k + 1 (the worker pools seed consecutively) */
  return (int)((s * 2654435761u + 0x9e3779b9u) & 0x7fffffffu);
}

void dyn_variables:: set_active_states_diff_rep(int rep_sh, Random& rnd){
/*
  Set the active state in the other representation through the transformation matrix

  rep_sh = 1 - Use the adiabatic active states to set the diabatic ones
           0 - Use the diabatic active states to set the adiabatic ones

  The process for determining the active state is based on the transformation of the density matrix,
  whose diagonal elements are replaced by the active state information, in similar to the following.
  Tempelaar, R.; Reichman, D. R. Generalization of Fewest-Switches Surface Hopping for Coherences. 
  The Journal of Chemical Physics 2018, 148 (10), 102309. https://doi.org/10.1063/1.5000843
*/
  int sz, i, j, traj;
  double ksi; 
  if(rep_sh==0){ sz = ndia; }
  else if(rep_sh==1){ sz = nadi; }
  else{ 
    cout<<"Wrong representation = "<<rep_sh<<"\n";
    std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
  }
  
  // Transform the active states between representations based on the transformation matrix.
  // the image draws come from their own generator, seeded once from the main generator's seed (not its stream): a saved
  // property that needs the image neither consumes nor shifts the random numbers of the hop decisions
  if(!image_rnd_seeded){
    image_rnd = Random(image_generator_seed(rnd.get_seed()));
    image_rnd_seeded = 1;
  }
  CMATRIX pop_adi(nadi, nadi);
  CMATRIX pop_dia(ndia, ndia);
  CMATRIX U(ndia, nadi);
  
  // Obtain the diabatic active states from the adiabatic ones
  if(rep_sh==1){

    for(traj=0; traj<ntraj; traj++){
      i = act_states[traj]; // active adiabatic state
      pop_adi = *dm_adi[traj];
      for(j=0;j<nadi; j++){ pop_adi.set(j,j, complex<double>(0.0, 0.0) ); }
      pop_adi.set(i, i, complex<double>(1.0, 0.0) );

      // the diabatic populations of an adiabatic state in the hopping frame: the Lowdin
      // sandwich Y U rho U^H Y for a non-orthogonal metric, the bare U rho U^H at S = I
      U = (*basis_transform[traj]);// * (*proj_adi[traj]);

      { CMATRIX Y_(ndia, ndia);
        int lowdin = lowdin_root(ovlp_dia[traj], ovlp_dia_is_identity[traj], Y_);
        pop_dia = lowdin ? Y_ * U * pop_adi * U.H() * Y_ : U * pop_adi * U.H(); }
      
      vector<double> diag_els(ndia, 0.0);
      for(i=0;i<ndia; i++){ diag_els[i] = pop_dia.get(i,i).real(); }

      ksi = image_rnd.uniform(0.0, 1.0);
      act_states_dia[traj] = hop(diag_els, ksi); 
    }
  }

  // Obtain the adiabatic active states from the diabatic ones
  else{
    for(traj=0; traj<ntraj; traj++){
      i = act_states_dia[traj]; // active diabatic state
      // the Lowdin-frame density Y rho Y carries the indicator of the active Lowdin state on its
      // diagonal - the coherences and the indicator in one frame (rho itself at S = I)
      U = (*basis_transform[traj]);// * (*proj_adi[traj]);

      { CMATRIX Y_(ndia, ndia);
        int lowdin = lowdin_root(ovlp_dia[traj], ovlp_dia_is_identity[traj], Y_);
        pop_dia = lowdin ? Y_ * (*dm_dia[traj]) * Y_ : *dm_dia[traj];
        for(j=0;j<ndia; j++){ pop_dia.set(j,j, complex<double>(0.0, 0.0) ); }
        pop_dia.set(i, i, complex<double>(1.0, 0.0) );

        // the adiabatic populations of a Lowdin state X e_i: C_adi = U^H S X e_i = U^H Y e_i,
        // the bare U^H rho U at S = I
        pop_adi = lowdin ? U.H() * Y_ * pop_dia * Y_ * U : U.H() * pop_dia * U; }
      
      vector<double> diag_els(nadi, 0.0);
      for(i=0;i<nadi; i++){ diag_els[i] = pop_adi.get(i,i).real(); }

      ksi = image_rnd.uniform(0.0, 1.0);
      act_states[traj] = hop(diag_els, ksi); 
    }
  }

}


/*
void dyn_variables::update_ampl_reproject(int option){

 changes all the adiabatic properties of the Hamiltonian by the matrix T

  option = 0 - use matrix U = T
         = 1 - use matrix U = T.H()



  CMATRIX U(*T); // option == 0
  if (option==1){  U = CMATRIX(T->H()); }
  else if(option==2 || option==4){

}

*/





CMATRIX orthogonalized_T(CMATRIX& T){
/**
   Return a unitary equivalent of the matrix T

  |psi_adi_tilde(t+dt)> = |psi_adi(t+dt)> T(t+dt)

  <psi_adi_tilde(t)|psi_adi_tilde(t+dt)> = <psi_adi(t)|psi_adi(t+dt)> T(t+dt)
        (should be close to I)                    (call it P (t, t+dt)  )

  So:

  I = P(t, t+dt) * T(t+dt)

  So, ideally T(t+dt) should be P^{-1}, but we also want it to be orthonormal

  T = P^{-1} * N

  We want:  T^+ T = N^+ (P^{-1})^+ P^(-1) N = I

  N =  (X^+ X)^-{1/2}, where X = P^(-1), indeed:

  N^+ (P^{-1})^+ P^(-1) N = ((X^+ X)^{-1/2})^+  (X^+ X) (X^+ X )^{-1/2} = 

  = [((X^+ X)^{-1/2})^+ ] [ (X^+ X )^{1/2}] = (X X^+)^{-1/2} (X^+ X)^{1/2} = 
 
  = (X^+)^{-1/2} X^{-1/2}   X^{1/2} (X^+)^{1/2} = (X^+)^{-1/2} (X^+)^{1/2} = I
  
  The input to this function should be P^{-1} 

*/

  int nst = T.n_cols;

  CMATRIX t2(nst,nst);
  CMATRIX t2_half(nst, nst);
  CMATRIX t2_i_half(nst, nst);
    
  t2 = T.H() * T;
  sqrt_matrix(t2, t2_half, t2_i_half);

  CMATRIX id(nst,nst); id.identity();
  if( std::abs( (t2_half * t2_i_half - id).max_elt()) > 1e-5 ){
    cout<<"Error in orthogonlized_T: S^{1/2} * S^{-1/2} is not an identity matrix\n";
    std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
  }
  t2 = T * t2_i_half;

  return t2;

}

CMATRIX dyn_variables::compute_average_dm(int rep){

  int sz;
  if(rep==0 || rep==2){ sz = ndia; }
  else if(rep==1 || rep==3 || rep == 4){ sz = nadi; }
  else{ cout<<"Error in dyn_variables::compute_average_dm: rep is not allowed. "; std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above"); }

  CMATRIX res(sz, sz);

  for(int traj=0; traj<ntraj; traj++){
    if(rep==0 || rep==2){   res += *dm_dia[traj]; }
    else if(rep==1 || rep==3){ res += *dm_adi[traj]; }
    else{
      std::cout<<"Error in dyn_variables::compute_average_dm(): rep = "<<rep<<" has no branch here (implemented: 0, 1, 2, 3)\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }
  }

  res = res / (float)ntraj;

  return res;
}

  
vector<double> dyn_variables::compute_average_se_pop(int rep){
/**
  The ensemble-averaged SE populations of representation rep: the diagonal of the averaged density matrix,
  except that the diabatic populations (rep 0 and 2) of non-orthogonal diabats are the Lowdin ones,
  (S^1/2 rho S^1/2)_ii - the populations of the hopping states the SH populations count (the bare diag(rho)
  does not sum to 1 under the normalization C^H S C = 1); at S = I the two agree
*/
  int sz;
  if(rep==0 || rep==2){ sz = ndia; }
  else if(rep==1 || rep==3 || rep==4){ sz = nadi; }
  else{ cout<<"Error in dyn_variables::compute_average_se_pop: rep is not allowed. "; std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above"); }

  vector<double> res(sz, 0.0);
  if(rep==0 || rep==2){
    CMATRIX X(ndia, ndia);
    CMATRIX Y(ndia, ndia);
    for(int traj=0; traj<ntraj; traj++){
      if(!ovlp_dia_is_identity[traj]){
        libmeigen::sqrt_matrix(*ovlp_dia[traj], Y, X);
        CMATRIX popL(Y * (*dm_dia[traj]) * Y);
        for(int i=0; i<sz; i++){ res[i] += popL.get(i,i).real(); }
      }
      else{
        for(int i=0; i<sz; i++){ res[i] += dm_dia[traj]->get(i,i).real(); }
      }
    }
    for(int i=0; i<sz; i++){ res[i] = res[i] / (float)ntraj; }
    return res;
  }

  MATRIX ave(sz, sz);
  ave = compute_average_dm(rep).real();
  for(int i=0; i<sz; i++){ res[i] = ave.get(i,i); }
  return res;
}


vector<double> dyn_variables::compute_average_mash_pop(int rep){
/**
  Computing the MASH population estimators based on:

  (1) E. Runeson, J.; P. Fay, T.; E. Manolopoulos, D. Exciton Dynamics from the Mapping Approach to Surface Hopping: Comparison with Förster and Redfield Theories. Physical Chemistry Chemical Physics 2024, 26 (6), 4929–4938. https://doi.org/10.1039/D3CP05926J.

*/

  //======= First, compute the SE populations ========
  int sz,i;

  if(rep==0 || rep==2){ sz = ndia; }
  else if(rep==1 || rep==3 || rep == 4){ sz = nadi; }
  else{ cout<<"Error in dyn_variables::compute_average_mash_pop: rep is not allowed. "; std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above"); }

  MATRIX ave(sz, sz);
  ave = compute_average_dm(rep).real();

  vector<double> res(sz, 0.0);

  for(i=0; i<sz; i++){ res[i] = ave.get(i,i); }

  //====== Now transform them according to MASH estimators formula =====

  double sum = 0.0;
  for(i=1; i<=sz; i++){  sum += 1.0/i; }  sum -= 1.0;
  double alpha_N = (sz - 1.0)/sum;
  double beta_N = (1.0 - alpha_N)/sz;

  for(i=0; i<sz; i++){  res[i] = alpha_N * res[i] + beta_N; }

  return res;

}



vector<double> dyn_variables::compute_average_sqc_pop(int rep, int window, double gamma){
/**
  Cotton-Miller SQC populations: each trajectory is binned by its shifted
  Meyer-Miller actions e_k = |c_k|^2 (= dm_kk) with the window of the
  requested type, the counts are averaged and renormalized over the states
  (P_k = <W_k> / sum_m <W_m>). Trajectories inside no window are dropped.
    window 0 - square: e_k in [1, 1 + 2 gamma], every other e_j in [0, 2 gamma]
    window 1 - triangle bin: e_k >= 1, every other e_j <= 1
  Refs: Cotton, Miller JPCA 117, 7190 (2013), JCP 145, 144108 (2016),
  JCP 150, 104101 (2019)
*/
  int sz;
  if(rep==0 || rep==2){ sz = ndia; }
  else if(rep==1 || rep==3){ sz = nadi; }
  else{ cout<<"Error in dyn_variables::compute_average_sqc_pop: rep is not allowed. "; std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above"); }
  if(!(window==0 || window==1)){
    cout<<"Error in dyn_variables::compute_average_sqc_pop: window = "<<window<<" is not allowed (0 - square, 1 - triangle)\n";
    std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
  }

  vector<double> res(sz, 0.0);
  vector<double> e(sz, 0.0);
  double total = 0.0;
  for(int traj=0; traj<ntraj; traj++){
    CMATRIX& dm = (rep==0 || rep==2) ? *dm_dia[traj] : *dm_adi[traj];
    for(int i=0; i<sz; i++){ e[i] = dm.get(i,i).real(); }
    for(int k=0; k<sz; k++){
      int inside = 1;
      if(window==0){
        if(e[k] < 1.0 || e[k] > 1.0 + 2.0*gamma){ inside = 0; }
        for(int j=0; j<sz && inside; j++){ if(j!=k && (e[j] < 0.0 || e[j] > 2.0*gamma)){ inside = 0; } }
      }
      else{
        if(e[k] < 1.0){ inside = 0; }
        for(int j=0; j<sz && inside; j++){ if(j!=k && e[j] > 1.0){ inside = 0; } }
      }
      if(inside){ res[k] += 1.0; total += 1.0; }
    }
  }
  if(total > 0.0){ for(int k=0; k<sz; k++){ res[k] /= total; } }
  return res;
}

vector<double> dyn_variables::compute_average_sqc_pop(int rep){
  return compute_average_sqc_pop(rep, sqc_window, sqc_gamma);
}

vector<double> dyn_variables::compute_average_sh_pop(int rep){
/**
  Computing the SH population with the active state of the input rep.
  If rep is different from rep_sh, the resultant SH population is computed indirectly through the sampling based on diagonal elments of the density matrix
*/

  int sz, i, j, traj; 
  if(rep==0){ sz = ndia; }
  else if(rep==1){ sz = nadi; }
  else{ 
    cout<<"Can not compute SH population for representation = "<<rep<<"\n";
    std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
  }

  vector<double> res(sz, 0.0);

  if(rep==0){
    vector<int> effective_states( act_states_dia );
    for(traj=0; traj<ntraj; traj++){ i = effective_states[traj]; res[i] += 1.0;  }
  }// rep == 0

  else if(rep==1){
    vector<int> effective_states( act_states );
    for(traj=0; traj<ntraj; traj++){ i = effective_states[traj]; res[i] += 1.0;  }
  }// rep == 1

  for(j=0; j<sz; j++){   res[j] = res[j] / (float)ntraj;   }

  return res;
}



vector<double> dyn_variables::compute_average_sh_pop_TR(int rep){
/**
  Computing the SH population based on the following work:
  Tempelaar, R.; Reichman, D. R. Generalization of Fewest-Switches Surface Hopping for Coherences. 
  The Journal of Chemical Physics 2018, 148 (10), 102309. https://doi.org/10.1063/1.5000843
*/
  int sz, i, j, traj; 
  if(rep==0){ sz = ndia; }
  else if(rep==1){ sz = nadi; }
  else{ 
    cout<<"Can not compute SH population for representation = "<<rep<<"\n";
    std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
  }

  vector<double> res(sz, 0.0);

  if(rep==0){
    vector<int> effective_states( act_states );

    CMATRIX pop_adi(nadi, nadi);
    CMATRIX pop_dia(ndia, ndia);
    CMATRIX U(ndia, nadi);

    for(traj=0; traj<ntraj; traj++){
      i = effective_states[traj]; // active adiabatic state
      pop_adi = *dm_adi[traj];
      for(j=0;j<nadi; j++){ pop_adi.set(j,j, complex<double>(0.0, 0.0) ); }
      pop_adi.set(i, i, complex<double>(1.0, 0.0) );

      // the diabatic populations of an adiabatic state in the hopping frame: the Lowdin
      // sandwich Y U rho U^H Y for a non-orthogonal metric, the bare U rho U^H at S = I
      U = (*basis_transform[traj]);// * (*proj_adi[traj]);

      { CMATRIX Y_(ndia, ndia);
        int lowdin = lowdin_root(ovlp_dia[traj], ovlp_dia_is_identity[traj], Y_);
        pop_dia = lowdin ? Y_ * U * pop_adi * U.H() * Y_ : U * pop_adi * U.H(); }
      for(j=0; j<ndia; j++){ res[j] += pop_dia.get(j,j).real(); }
    }
  }// rep == 0

  else if(rep==1){
    vector<int> effective_states( act_states_dia );

    CMATRIX pop_adi(nadi, nadi);
    CMATRIX pop_dia(ndia, ndia);
    CMATRIX U(ndia, nadi);

    for(traj=0; traj<ntraj; traj++){
      i = effective_states[traj]; // active diabatic state
      // the Lowdin-frame density Y rho Y carries the indicator of the active Lowdin state on its
      // diagonal - the coherences and the indicator in one frame (rho itself at S = I)
      U = (*basis_transform[traj]);// * (*proj_adi[traj]);

      { CMATRIX Y_(ndia, ndia);
        int lowdin = lowdin_root(ovlp_dia[traj], ovlp_dia_is_identity[traj], Y_);
        pop_dia = lowdin ? Y_ * (*dm_dia[traj]) * Y_ : *dm_dia[traj];
        for(j=0;j<ndia; j++){ pop_dia.set(j,j, complex<double>(0.0, 0.0) ); }
        pop_dia.set(i, i, complex<double>(1.0, 0.0) );

        // the adiabatic populations of a Lowdin state X e_i: C_adi = U^H S X e_i = U^H Y e_i,
        // the bare U^H rho U at S = I
        pop_adi = lowdin ? U.H() * Y_ * pop_dia * Y_ * U : U.H() * pop_dia * U; }
      for(j=0; j<nadi; j++){ res[j] += pop_adi.get(j,j).real(); }
    }
  }// rep == 1

  for(j=0; j<sz; j++){   res[j] = res[j] / (float)ntraj;   }

  return res;
}


MATRIX dyn_variables::compute_coherence_indicator(int rep){

  int sz;
  if(rep==0 || rep==2){ sz = ndia; }
  else if(rep==1 || rep==3 || rep==4 ){ sz = nadi; }
  else{ cout<<"Error in dyn_variables::compute_coherence_indicator: rep is not allowed. "; std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above"); }

  MATRIX res(sz, sz);
  
  CMATRIX dm(sz, sz); MATRIX temp(sz, ntraj);

  // Making a temporary matrix collecting trajectory-wise population elements
  for(int traj=0; traj<ntraj; traj++){
    if(rep==0 || rep==2){   dm = *dm_dia[traj]; }
    else if(rep==1 || rep==3){ dm = *dm_adi[traj]; }
    else{
      std::cout<<"Error in dyn_variables::compute_coherence_indicator(): rep = "<<rep<<" has no branch here (implemented: 0, 1, 2, 3)\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }
  
    for(int i=0; i<sz; i++){temp.set(i, traj, dm.get(i,i).real() );}
  }

  res = temp * temp.T() / (float)ntraj;

  return res;
}


void dyn_variables::save_curr_dm_into_prev(){

  if(fssh2_vars_status==1){
    for(int i=0; i<ntraj; i++){
      *dm_dia_prev[i] = *dm_dia[i];
      *dm_adi_prev[i] = *dm_adi[i];
    }
  }

}


double dyn_variables::compute_kcrpmd_ekin(){
  double res = 0.0;

  res += pow(p_aux_var[0], 2) / m_aux_var[0];

  return 0.5*res;
}


}// namespace libdyn
}// liblibra

