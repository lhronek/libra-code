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
  \file nHamiltonian_compute_forces.cpp
  \brief The file implements computations of various types of forces and force tensors
    
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



CMATRIX nHamiltonian::forces_adi(vector<int>& act_states){
/**
  This only applies for a nHamiltonian object with a set of children whose size is 
  equal to the number of integers in `act_states` input

  Return: -d1ham_adi shaped as CMATRIX(ndof, ntraj) - for `act_states` states
*/
  int ntraj = act_states.size();   
  CMATRIX res(nnucl, ntraj);
 
  if(children.size()==ntraj){

    for(int traj=0; traj<ntraj; traj++){
      int st = act_states[traj];

      for(int dof=0; dof<nnucl; dof++){
        if(children[traj]->d1ham_adi_mem_status[dof]==0){
          cout<<"Error in forces_adi(): the derivative of the Hamiltonian w.r.t. the nuclear \
DOF "<<dof<<" is not allocated for trajectory "<<traj<<" but is needed for the calculations \n";
          std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
        }
        res.set(dof, traj, -children[traj]->d1ham_adi[dof]->get(st, st) );
      }// for dof 
    }// for traj
  }
  else{
    cout<<"ERROR: the size of the input is different from the number of children\n"; std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above"); 
  }

  return res;
}



CMATRIX nHamiltonian::forces_dia(vector<int>& act_states){
/**
  This only applies for a nHamiltonian object with a set of children whose size is 
  equal to the number of integers in `act_states` input

  Return: -d1ham_dia shaped as CMATRIX(ndof, ntraj) - for `act_states` states
*/

  int ntraj = act_states.size();   
  CMATRIX res(nnucl, ntraj);
 
  if(children.size()==ntraj){

    for(int traj=0; traj<ntraj; traj++){
      int st = act_states[traj];

      for(int dof=0; dof<nnucl; dof++){
        if(children[traj]->d1ham_dia_mem_status[dof]==0){
          cout<<"Error in forces_dia(): the derivative of the Hamiltonian w.r.t. the nuclear \
DOF "<<dof<<" is not allocated for trajectory "<<traj<<" but is needed for the calculations \n";
          std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
        }
        res.set(dof, traj, -children[traj]->d1ham_dia[dof]->get(st, st) );
      }// for dof 
    }// for traj
  }
  else{
    cout<<"ERROR: the size of the input is different from the number of children\n"; std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above"); 
  }

  return res;
}



CMATRIX nHamiltonian::all_forces_adi(vector<int>& id_){
/**
  These are forces on all states for a Hamiltonian of given level, given by its id
*/

  if(id_.size()==1){
    if(id_[0]==id){   
      // This is where the content of the function is
      CMATRIX res(nadi, nnucl); 

      for(int n=0;n<nnucl;n++){
        if(d1ham_adi_mem_status[n]==0){ 
          cout<<"Error in all_forces_adi(): the derivatives of the Hamiltonian matrix in the \
                adiabatic basis w.r.t. the nuclear DOF "<<n<<" is not allocated but is \
                needed for the calculations \n"; std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above"); 
        }

        for(int ist=0; ist<nadi; ist++){
          res.set(ist, n,  -d1ham_adi[n]->get(ist, ist) ); 
        }

      }// for n

      return res; 

    }// id_[0]==id
    else{ cout<<"ERROR in all_forces_adi: No Hamiltonian matching the requested id\n"; std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above"); }
  }
  else{
    vector<int> next(id_.begin()+1,id_.end());
    return children[id_[1]]->all_forces_adi(next);
  }

} // all_forces_adi

CMATRIX nHamiltonian::all_forces_dia(vector<int>& id_){
/**
  These are forces on all states for a Hamiltonian of given level, given by its id
*/

  if(id_.size()==1){
    if(id_[0]==id){
      // This is where the content of the function is
      CMATRIX res(ndia, nnucl);

      for(int n=0;n<nnucl;n++){
        if(d1ham_dia_mem_status[n]==0){
          cout<<"Error in all_forces_dia(): the derivatives of the Hamiltonian matrix in the \
                adiabatic basis w.r.t. the nuclear DOF "<<n<<" is not allocated but is \
                needed for the calculations \n"; std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
        }

        for(int ist=0; ist<ndia; ist++){
          res.set(ist, n,  -d1ham_dia[n]->get(ist, ist) );
        }

      }// for n

      return res;

    }// id_[0]==id
    else{ cout<<"ERROR in all_forces_dia: No Hamiltonian matching the requested id\n"; std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above"); }
  }
  else{
    vector<int> next(id_.begin()+1,id_.end());
    return children[id_[1]]->all_forces_dia(next);
  }

}  // all_forces_dia



}// namespace libnhamiltonian
}// liblibra

