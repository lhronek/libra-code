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
  \file nHamiltonian_compute_QTSH.cpp
  \brief The file implements the calculations of the QTSH-related properties:
  energies, forces and state-resolved force contributions
    
*/

#include <stdexcept>
#if defined(USING_PCH)
#include "../pch.h"
#else
#include <stdlib.h>
#endif

#include "nHamiltonian.h"
#include "../math_meigen/libmeigen.h"


/// liblibra namespace
namespace liblibra{

/// libnhamiltonian namespace 
namespace libnhamiltonian{


using namespace liblinalg;
using namespace libmeigen;



complex<double> nHamiltonian::QTSH_energy_dia(CMATRIX& ampl_dia){
/**
  Compute the coherence energy of QTSH in the diabatic representation.
  The coherence energy in QTSH has the form of the off-diagonal terms of the Ehrenfest energy.
*/

  if(ovlp_dia_mem_status==0){ cout<<"Error in QTSH_energy_dia(): the overlap matrix in the diabatic basis is not allocated \
  but it is needed for the calculations\n"; std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above"); }

  if(ham_dia_mem_status==0){ cout<<"Error in QTSH_energy_dia(): the diabatic Hamiltonian matrix is not allocated \
  but it is needed for the calculations\n"; std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above"); }

//  if(ampl_dia_mem_status==0){ cout<<"Error in QTSH_energy_dia(): the amplitudes of the diabatic states are\
//  not allocated, but they are needed for the calculations\n"; exit(0); }

  // To obtain the off-diagonal part of the diabatic Hamiltonian matrix
  CMATRIX* H_diag; H_diag = new CMATRIX(ndia,ndia);
  CMATRIX* V; V = new CMATRIX(ndia,ndia);
  for(int i=0; i<ndia; i++){ H_diag->set(i,i, (*ham_dia).get(i,i) ); }
  *V = *ham_dia - *H_diag; 

  complex<double> norm = (ampl_dia.H() * (*ovlp_dia) * ampl_dia).M[0]; 
  complex<double> res = (ampl_dia.H() * ( (*V) + complex<double>(0.0, -1.0)* (*nac_dia) ) * ampl_dia).M[0] / norm;
  
  delete H_diag;
  delete V;

  return res;

}



complex<double> nHamiltonian::QTSH_energy_dia(CMATRIX& ampl_dia, vector<int>& id_){
/**
  See the description of the QTSH_energy_dia(CMATRIX& ampl_dia) function
*/
  if(id_.size()==1){
    if(id_[0]==id){   return QTSH_energy_dia(ampl_dia);    }
    else{ cout<<"ERROR in force_dia: No Hamiltonian matching the requested id\n"; std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above"); }
  }
  else{
    vector<int> next(id_.begin()+1,id_.end());
    return children[id_[1]]->QTSH_energy_dia(ampl_dia, next);
  }
}




complex<double> nHamiltonian::QTSH_energy_dia(CMATRIX& ampl_dia, MATRIX& p, const MATRIX& invM){
/**
  The coherence energy of QTSH in the Lowdin frame of the diabats, the frame of the QTSH state energy
  (get_ham_dia_lowdin): d^H offdiag(Hvib_L) d / (d^H d) over the Lowdin amplitudes d = Y ampl_dia, with
  Hvib_L = X hvib_dia X + i (dY/dt) X of get_hvib_dia_lowdin (p, invM: the trajectory's momenta and inverse
  masses, nnucl x 1). At S = I the frames coincide and this is QTSH_energy_dia(ampl_dia)
*/
  CMATRIX X(ndia, ndia);
  CMATRIX Y(ndia, ndia);
  if(lowdin_dia(X, Y) == 0){ return QTSH_energy_dia(ampl_dia); }
  CMATRIX d(Y * ampl_dia);
  CMATRIX V(get_hvib_dia_lowdin(p, invM));
  for(int i=0; i<ndia; i++){ V.set(i, i, complex<double>(0.0, 0.0)); }
  complex<double> norm = (d.H() * d).M[0];
  return (d.H() * V * d).M[0] / norm;
}


complex<double> nHamiltonian::QTSH_energy_dia(CMATRIX& ampl_dia, MATRIX& p, const MATRIX& invM, vector<int>& id_){
/**
  See the description of the QTSH_energy_dia(CMATRIX& ampl_dia, MATRIX& p, const MATRIX& invM) function
*/
  if(id_.size()==1){
    if(id_[0]==id){   return QTSH_energy_dia(ampl_dia, p, invM);    }
    else{ cout<<"ERROR in QTSH_energy_dia: No Hamiltonian matching the requested id\n"; std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above"); }
  }
  else{
    vector<int> next(id_.begin()+1,id_.end());
    return children[id_[1]]->QTSH_energy_dia(ampl_dia, p, invM, next);
  }
}




complex<double> nHamiltonian::QTSH_energy_adi(CMATRIX& ampl_adi, CMATRIX& transform){
/**
  Compute the coherence energy of QTSH in the adiabatic representation.
  The coherence energy in QTSH has the form of the off-diagonal terms of the Ehrenfest energy.
*/

  if(ham_adi_mem_status==0){ cout<<"Error in QTSH_energy_adi(): the adiabatic Hamiltonian matrix is not allocated \
  but it is needed for the calculations\n"; std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above"); }

//  if(ampl_adi_mem_status==0){ cout<<"Error in QTSH_energy_adi(): the amplitudes of the adiabatic states are\
//  not allocated, but they are needed for the calculations\n"; exit(0); }

//  CMATRIX& T = transform;
  CMATRIX T(transform); T.identity();
  
  complex<double> norm = (ampl_adi.H() * ampl_adi).M[0]; 
  
  return (ampl_adi.H() * ( T.H() * complex<double>(0.0, -1.0)* (*nac_adi) * T) * ampl_adi).M[0] / norm;

}

complex<double> nHamiltonian::QTSH_energy_adi(CMATRIX& ampl_adi){
  CMATRIX I(nadi, nadi); I.identity();
  return QTSH_energy_adi(ampl_adi, I);
}


complex<double> nHamiltonian::QTSH_energy_adi(CMATRIX& ampl_adi, vector<int>& id_, CMATRIX& transform){
/**
  See the description of the QTSH_energy_adi(CMATRIX& ampl_adi) function
*/
  if(id_.size()==1){
    if(id_[0]==id){   return QTSH_energy_adi(ampl_adi);    }
    else{ cout<<"ERROR in force_dia: No Hamiltonian matching the requested id\n"; std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above"); }
  }
  else{
    vector<int> next(id_.begin()+1,id_.end());
    return children[id_[1]]->QTSH_energy_adi(ampl_adi, next, transform);
  }
}

complex<double> nHamiltonian::QTSH_energy_adi(CMATRIX& ampl_adi, vector<int>& id_){
  CMATRIX I(nadi, nadi); I.identity();
  return QTSH_energy_adi(ampl_adi, id_, I);
}


complex<double> nHamiltonian::QTSH_energy_dia_dm(CMATRIX& dm_dia){
/**
  The density-matrix form of QTSH_energy_dia: tr((V - i nac_dia) rho) / tr(S rho) with V the off-diagonal part of
  the diabatic Hamiltonian and rho = C_dia C_dia^H (dyn_variables::dm_dia) - equal to the amplitude form for a pure
  state
*/
  if(ovlp_dia_mem_status==0){ cout<<"Error in QTSH_energy_dia_dm(): the overlap matrix in the diabatic basis is not allocated \
  but it is needed for the calculations\n"; std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above"); }
  if(ham_dia_mem_status==0){ cout<<"Error in QTSH_energy_dia_dm(): the diabatic Hamiltonian matrix is not allocated \
  but it is needed for the calculations\n"; std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above"); }

  CMATRIX V(*ham_dia);
  for(int i=0; i<ndia; i++){ V.set(i, i, complex<double>(0.0, 0.0)); }
  complex<double> norm = ((*ovlp_dia) * dm_dia).tr();
  return (( V + complex<double>(0.0, -1.0) * (*nac_dia) ) * dm_dia).tr() / norm;
}

complex<double> nHamiltonian::QTSH_energy_dia_dm(CMATRIX& dm_dia, MATRIX& p, const MATRIX& invM){
/**
  The density-matrix form of QTSH_energy_dia(ampl_dia, p, invM): the coherence energy in the Lowdin frame of the
  diabats, tr(offdiag(Hvib_L) rho_L) / tr(rho_L) over the Lowdin density rho_L = Y rho Y^H (the amplitude form's
  d = Y ampl_dia gives d d^H = Y rho Y^H). At S = I the frames coincide and this is QTSH_energy_dia_dm(dm_dia)
*/
  CMATRIX X(ndia, ndia);
  CMATRIX Y(ndia, ndia);
  if(lowdin_dia(X, Y) == 0){ return QTSH_energy_dia_dm(dm_dia); }
  CMATRIX rho_l(Y * dm_dia * Y.H());
  CMATRIX V(get_hvib_dia_lowdin(p, invM));
  for(int i=0; i<ndia; i++){ V.set(i, i, complex<double>(0.0, 0.0)); }
  complex<double> norm = rho_l.tr();
  return (V * rho_l).tr() / norm;
}

complex<double> nHamiltonian::QTSH_energy_dia_dm(CMATRIX& dm_dia, MATRIX& p, const MATRIX& invM, vector<int>& id_){
/**
  See the description of the QTSH_energy_dia_dm(CMATRIX& dm_dia, MATRIX& p, const MATRIX& invM) function
*/
  if(id_.size()==1){
    if(id_[0]==id){   return QTSH_energy_dia_dm(dm_dia, p, invM);    }
    else{ cout<<"ERROR in QTSH_energy_dia_dm: No Hamiltonian matching the requested id\n"; std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above"); }
  }
  else{
    if (id_[1] < 0 || id_[1] >= (int)children.size()) {
      cout << "Error in nHamiltonian::QTSH_energy_dia_dm(): the child index " << id_[1]
           << " is beyond the " << children.size() << " children of this level - a shared child "
              "(isNBRA) serves trajectory 0's index alone\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }
    vector<int> next(id_.begin()+1,id_.end());
    return children[id_[1]]->QTSH_energy_dia_dm(dm_dia, p, invM, next);
  }
}

}// namespace libnhamiltonian
}// liblibra

