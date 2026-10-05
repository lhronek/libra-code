/*********************************************************************************
* Copyright (C) 2015-2022 Alexey V. Akimov
*
* This file is distributed under the terms of the GNU General Public License
* as published by the Free Software Foundation, either version 3 of
* the License, or (at your option) any later version.
* See the file LICENSE in the root directory of this distribution
* or <http://www.gnu.org/licenses/>.
*
*********************************************************************************/
/**
  \file Energy_and_Forces.cpp
  \brief The file implements the functions for dynamics-immediate energy calculations

  The "dynamics-immediate" means the energies and forces computed and organized to 
  be used in dynamical calculations of different type - classical, quantum, quantum-classical.
    
*/

#include <cmath>
#include "Energy_and_Forces.h"
#include <stdexcept>
#include "dyn_projectors.h"

/// liblibra namespace
namespace liblibra{

/// libdyn namespace 
namespace libdyn{


double compute_kinetic_energy(MATRIX& p, MATRIX& invM, vector<int>& which_dofs){
/**
  \brief Compute kinetic energy of nuclear DOFs
  \param[in] p [ndof x ntraj] Momenta of ntraj replicas of the system with ndof 
  nuclear DOFs
  \param[in] invM [ndof x 1] Matrix of inverted masses of all DOFs

  In the case of more than 1 trajectories supplied into this function, the 
  total kinetic energy of all trajectories is returned.

  This is the classical nuclear kinetic energy
*/
  int ndof = which_dofs.size(); ///p.n_rows;
  int ntraj = p.n_cols;

  double Ekin = 0.0;
  
    
  for(int idof=0; idof < ndof; idof++){ 

    int dof = which_dofs[idof];

    double sum = 0.0;
    for(int traj=0; traj < ntraj; traj++){    
      sum = p.get(dof, traj) * p.get(dof, traj);
    }
    Ekin += sum * invM.get(dof, 0);
  }
  Ekin *= 0.5;

  return Ekin;

}

double compute_kinetic_energy(MATRIX& p, MATRIX& invM){

  int ndof = p.n_rows;
  vector<int> which_dofs(ndof);
  for(int i = 0; i < ndof; i++){  which_dofs[i] = i; }

  return compute_kinetic_energy(p, invM, which_dofs);

}


vector<double> compute_kinetic_energies(MATRIX& p, MATRIX& invM, vector<int>& which_dofs){
/**
  \brief Compute trajectory-resolved kinetic energies of nuclear DOFs

  \param[in] p [ndof x ntraj] Momenta of ntraj replicas of the system with ndof 
  nuclear DOFs
  \param[in] invM [ndof x 1] Matrix of inverted masses of all DOFs

  This is the classical nuclear kinetic energy
*/
  int ndof = which_dofs.size(); ///p.n_rows;
  int ntraj = p.n_cols;

  vector<double> Ekin(ntraj, 0.0);
  
  for(int traj=0; traj < ntraj; traj++){    

    double sum = 0.0;

    for(int idof=0; idof < ndof; idof++){ 

      int dof = which_dofs[idof];
      sum += p.get(dof, traj) * invM.get(dof, 0) * p.get(dof, traj);
    }
    sum *= 0.5;

    Ekin[traj] = sum;

  }// for traj

  return Ekin;

}



vector<double> compute_kinetic_energies(MATRIX& p, MATRIX& invM){

  int ndof = p.n_rows;
  vector<int> which_dofs(ndof);
  for(int i = 0; i < ndof; i++){  which_dofs[i] = i; }

  return compute_kinetic_energies(p, invM, which_dofs);

}





CMATRIX tsh_indx2ampl(vector<int>& res, int nstates){
/**
  Convert the active state index to the vector with occupation numbers
  This is done for 1 or many trajectories
*/

  int ntraj = res.size();
  CMATRIX ampl(nstates, ntraj);

  
  for(int traj = 0; traj < ntraj; traj++){

    ampl.set(res[traj], traj, complex<double>(1.0, 0.0));

  }

  return ampl;

}


double average_potential_energy(dyn_control_params& prms, dyn_variables& dyn_vars, nHamiltonian& ham){

  vector<double> res(dyn_vars.ntraj, 0.0);
  res = potential_energies(prms, dyn_vars, ham);

  double ave = 0.0;
  for(int itraj = 0; itraj < dyn_vars.ntraj; itraj++){  ave += res[itraj]; }
  ave /= double(dyn_vars.ntraj);
  
  return ave;
}

double average_potential_energy(bp::dict params, dyn_variables& dyn_vars, nHamiltonian& ham){

  dyn_control_params prms;
  prms.set_parameters(params);

  return average_potential_energy(prms, dyn_vars, ham);
}


/// the state whose surface drives each trajectory: the hopping state of the hopping representation (rep_sh), or
/// the enforced state under enforce_state_following
static vector<int> driving_states(dyn_control_params& prms, dyn_variables& dyn_vars){
  vector<int> states(dyn_vars.ntraj, 0);
  if(prms.rep_sh==1){ states = dyn_vars.act_states; }
  else if(prms.rep_sh==0){ states = dyn_vars.act_states_dia; }
  else{
    std::cout<<"Error in driving_states(): rep_sh = "<<prms.rep_sh<<" has no branch here (implemented: 0, 1)\n";
    std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
  }
  if(prms.enforce_state_following==1){
    for(int itraj=0; itraj<dyn_vars.ntraj; itraj++){ states[itraj] = prms.enforced_state_index; }
  }
  return states;
}

/// the Hamiltonian child of a trajectory - one shared child under isNBRA
static int child_of(nHamiltonian& ham, int itraj){
  return (itraj < (int)ham.children.size()) ? itraj : 0;
}

vector<double> potential_energies(dyn_control_params& prms, dyn_variables& dyn_vars, nHamiltonian& ham){

  int itraj;
  int ntraj = dyn_vars.ntraj;
  int nadi = dyn_vars.nadi;
  int ndia = dyn_vars.ndia;

  vector<int> id(2,0);

  vector<double> res(ntraj, 0.0);

/*
  if(prms.force_method==0){  // No forces

    // Don't compute forces at all - e.g. in NBRA
    for(itraj=0; itraj<ntraj; itraj++){ res[itraj] = 0.0; }

  }// NBRA
*/
  if(prms.force_method==0 || prms.force_method==1){   // State-specific forces

    // TSH or adiabatic (including excited states)
    // state-specific forces

    vector<int> effective_states = driving_states(prms, dyn_vars);

    // Diabatic 
    if(prms.rep_force==0){  
      for(itraj=0; itraj<ntraj; itraj++){
        id[1] = child_of(ham, itraj);
        int ist = effective_states[itraj];
        res[itraj] = ham.child(id[1])->get_ham_dia_lowdin().get(ist, ist).real();   
      }
    }
    // Adiabatic 
    else if(prms.rep_force==1){  
      for(itraj=0; itraj<ntraj; itraj++){
        id[1] = child_of(ham, itraj);
        int ist = effective_states[itraj];

        //CMATRIX& T = *dyn_vars.proj_adi[itraj];
        res[itraj] = ham.get_ham_adi(id).get(ist,ist).real();
      }// for itraj
    }// rep_force == 1
    else{
      std::cout<<"Error in potential_energies(): rep_force = "<<prms.rep_force<<" has no branch here (implemented: 0, 1)\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }

  }// TSH && adiabatic

  else if(prms.force_method==2){  // Ehrenfest forces

    // the density-matrix representations (rep_tdse 2 / 3) propagate rho alone and their amplitudes keep the t0
    // values - the Ehrenfest energy is tr(H rho) / tr(S rho) there (the amplitude form for a pure state)
    int dm_rep = (prms.rep_tdse==2 || prms.rep_tdse==3);
    if(dm_rep && prms.rep_force==0){
      for(itraj=0; itraj<ntraj; itraj++){
        id[1] = child_of(ham, itraj);
        CMATRIX& rho = *dyn_vars.dm_dia[itraj];
        res[itraj] = ham.Ehrenfest_energy_dia_dm(rho, id).real();
        if(prms.ehrenfest_normalized==0){
          double nrm = (ham.get_ovlp_dia(id) * rho).tr().real();
          res[itraj] = res[itraj] * nrm - prms.sqc_gamma * ham.get_ham_dia(id).tr().real();
        }
      }
    }
    else if(dm_rep && prms.rep_force==1){
      for(itraj=0; itraj<ntraj; itraj++){
        id[1] = child_of(ham, itraj);
        CMATRIX& rho = *dyn_vars.dm_adi[itraj];
        res[itraj] = ham.Ehrenfest_energy_adi_dm(rho, id, *(dyn_vars.proj_adi[itraj])).real();
        if(prms.ehrenfest_normalized==0){
          double nrm = rho.tr().real();
          res[itraj] = res[itraj] * nrm - prms.sqc_gamma * ham.get_ham_adi(id).tr().real();
        }
      }
    }
    // Diabatic 
    else if(prms.rep_force==0){ 
      CMATRIX coeff(ndia, 1);
      for(itraj=0; itraj<ntraj; itraj++){
        id[1] = child_of(ham, itraj);
        coeff = dyn_vars.ampl_dia->col(itraj);
        res[itraj] = ham.Ehrenfest_energy_dia(coeff, id).real();
        // the un-normalized Meyer-Miller potential, E = c^H H c - gamma tr(H) -
        // the quantity conserved together with the ehrenfest_normalized 0 force
        if(prms.ehrenfest_normalized==0){
          double nrm = (coeff.H() * ham.get_ovlp_dia(id) * coeff).get(0,0).real();
          res[itraj] = res[itraj] * nrm - prms.sqc_gamma * ham.get_ham_dia(id).tr().real();
        }
      }
    }
    // Adiabatic 
    else if(prms.rep_force==1){  
      CMATRIX coeff(nadi, 1);
      for(itraj=0; itraj<ntraj; itraj++){
        id[1] = child_of(ham, itraj);
        coeff = dyn_vars.ampl_adi->col(itraj);
        res[itraj] = ham.Ehrenfest_energy_adi(coeff, id, *(dyn_vars.proj_adi[itraj]) ).real();
        if(prms.ehrenfest_normalized==0){
          double nrm = (coeff.H() * coeff).get(0,0).real();
          res[itraj] = res[itraj] * nrm - prms.sqc_gamma * ham.get_ham_adi(id).tr().real();
        }
      }
    }
    else{
      std::cout<<"Error in potential_energies(): rep_force = "<<prms.rep_force<<" has no branch here (implemented: 0, 1)\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }
  
  }// Ehrenfest

  else if(prms.force_method==3){ // QTSH
    
    vector<int> effective_states = driving_states(prms, dyn_vars);
    
    // the density-matrix representations (rep_tdse 2 / 3): the coherence energy from rho (the amplitude form for a
    // pure state) - the adiabatic branch below adds no coherence term and serves them as it is
    int dm_rep = (prms.rep_tdse==2 || prms.rep_tdse==3);
    if(dm_rep && prms.rep_force==0){
      MATRIX p_traj(dyn_vars.ndof, 1);
      for(itraj=0; itraj<ntraj; itraj++){
        id[1] = child_of(ham, itraj);
        int ist = effective_states[itraj];
        res[itraj] = ham.child(id[1])->get_ham_dia_lowdin().get(ist, ist).real();
        p_traj = dyn_vars.p->col(itraj);
        res[itraj] += ham.QTSH_energy_dia_dm(*dyn_vars.dm_dia[itraj], p_traj, *dyn_vars.iM, id).real();
      }
    }
    // Diabatic 
    else if(prms.rep_force==0){ 
      CMATRIX coeff(ndia, 1);
      MATRIX p_traj(dyn_vars.ndof, 1);
      for(itraj=0; itraj<ntraj; itraj++){
        id[1] = child_of(ham, itraj);
        int ist = effective_states[itraj];
        res[itraj] = ham.child(id[1])->get_ham_dia_lowdin().get(ist, ist).real(); // diagonal energy
        
        coeff = dyn_vars.ampl_dia->col(itraj);
        p_traj = dyn_vars.p->col(itraj);
        // the coherence energy in the same Lowdin frame (the raw diabatic one at S = I)
        res[itraj] += ham.QTSH_energy_dia(coeff, p_traj, *dyn_vars.iM, id).real();
      }
    }
    // Adiabatic 
    else if(prms.rep_force==1){  
      //CMATRIX coeff(nadi, 1);
      for(itraj=0; itraj<ntraj; itraj++){
        id[1] = child_of(ham, itraj);
        int ist = effective_states[itraj];
        res[itraj] = ham.get_ham_adi(id).get(ist,ist).real(); // diagonal energy

        // No need to add the coherence term directly in the adiabatic representation due to the kinetic energy contribution of the kinematic momentum
        // coeff = dyn_vars.ampl_adi->col(itraj);
        // res[itraj] += ham.QTSH_energy_adi(coeff, id, *(dyn_vars.proj_adi[itraj]) ).real(); // coherence energy
      }// for itraj
    }// rep_force == 1
    else{
      std::cout<<"Error in potential_energies(): rep_force = "<<prms.rep_force<<" has no branch here (implemented: 0, 1)\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }

  }
  else if(prms.force_method==4){  // KC-RPMD force

    // Diabatic 
    if(prms.rep_force==0){ 
      double Veff;
      double beta = (hartree/boltzmann) / prms.Temperature; 
      double eta = prms.kcrpmd_eta; 
      double a = prms.kcrpmd_a; 
      double b = prms.kcrpmd_b; 
      double c = prms.kcrpmd_c; 
      Veff = ham.kcrpmd_effective_potential(dyn_vars.y_aux_var, *dyn_vars.q, *dyn_vars.iM, beta, eta, a, b, c);
      for(itraj=0; itraj<ntraj; itraj++){
        id[1] = child_of(ham, itraj);
        res[itraj] = Veff;
      }
    }
    // Adiabatic
    else if(prms.rep_force==1){
      cout<<"Error in potential_energies(): KC-RPMD works exclusively within diabatic bases\n"; std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }
    else{
      std::cout<<"Error in potential_energies(): rep_force = "<<prms.rep_force<<" has no branch here (implemented: 0, 1)\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }
  
  }// KC-RPMD
  else{
    std::cout<<"Error in potential_energies(): force_method = "<<prms.force_method<<" has no branch here (implemented: 0, 1, 2, 3, 4)\n";
    std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
  }


  return res;
}

vector<double> potential_energies(bp::dict params, dyn_variables& dyn_vars, nHamiltonian& ham){

  dyn_control_params prms;
  prms.set_parameters(params);

  return potential_energies(prms, dyn_vars, ham);

}

//MATRIX aux_get_forces(dyn_control_params& prms, dyn_variables& dyn_vars, nHamiltonian& ham){
void update_forces(dyn_control_params& prms, dyn_variables& dyn_vars, nHamiltonian& ham){
  /**
    Compute the force depending on the method used
  */

  int ndof = ham.nnucl;
  int nst = ham.nadi;
  int ntraj = dyn_vars.ntraj;

  //MATRIX F(ndof, ntraj);
  CMATRIX f_all(nst, ndof);
  CMATRIX f_diag(nst, nst);
  CMATRIX f(ndof, 1);
  vector<int> id(2, 0); 
  

  if(prms.force_method==0){  // No forces

    // Don't compute forces at all - e.g. in NBRA

  }// NBRA

  else if(prms.force_method==1){   // State-specific forces

    // TSH or adiabatic (including excited states)
    // state-specific forces   
    vector<int> effective_states = driving_states(prms, dyn_vars);

    if(prms.isNBRA==1 && (int)ham.children.size()==1 && ntraj>1){
      // the shared child: one force column serves every trajectory -
      // valid only while the forces are trajectory-invariant (the
      // update_Hamiltonian_variables funnel admits exactly the
      // state-enforced, rescaling-free combination)
      vector<int> one_state(1, effective_states[0]);
      MATRIX f1(ndof, 1);
      if(prms.rep_force==0){ f1 = ham.forces_dia(one_state).real(); }
      else if(prms.rep_force==1){ f1 = ham.forces_adi(one_state).real(); }
      else{
        std::cout<<"Error in update_forces(): rep_force = "<<prms.rep_force<<" has no branch here (implemented: 0, 1)\n";
        std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
      }
      for(int itraj=0; itraj<ntraj; itraj++){
        for(int idof=0; idof<ndof; idof++){
          dyn_vars.f->set(idof, itraj, f1.get(idof, 0));
        }
      }
    }
    // Diabatic 
    else if(prms.rep_force==0){  *dyn_vars.f = ham.forces_dia(effective_states).real();   }

    // Adiabatic 
    else if(prms.rep_force==1){  
      *dyn_vars.f = ham.forces_adi(effective_states).real(); 

    }// rep_force == 1
    else{
      std::cout<<"Error in update_forces(): rep_force = "<<prms.rep_force<<" has no branch here (implemented: 0, 1)\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }
  }// TSH && adiabatic

  else if(prms.force_method==2){  // Ehrenfest forces
    // the density-matrix representations (rep_tdse 2 / 3) propagate rho alone and their amplitudes keep the t0
    // values - the Ehrenfest forces are -tr(dH rho) / tr(S rho) there (the amplitude form for a pure state)
    int dm_rep = (prms.rep_tdse==2 || prms.rep_tdse==3);
    if(dm_rep && prms.rep_force==0){
      *dyn_vars.f = ham.Ehrenfest_forces_dia_dm(dyn_vars.dm_dia, 1, 0, prms.sqc_gamma, prms.ehrenfest_normalized).real();
    }
    else if(dm_rep && prms.rep_force==1){
      int option = prms.ehrenfest_force_option;
      *dyn_vars.f = ham.Ehrenfest_forces_adi_dm(dyn_vars.dm_adi, 1, option, prms.sqc_gamma, prms.ehrenfest_normalized).real();
    }
    // Diabatic 
    else if(prms.rep_force==0){  
     // sqc_gamma adds the Meyer-Miller ZPE term (see the adiabatic branch)
     *dyn_vars.f = ham.Ehrenfest_forces_dia(*dyn_vars.ampl_dia, 1, 0, prms.sqc_gamma, prms.ehrenfest_normalized).real(); 
 

     //cout<<"dia MF forces\n"; dyn_vars.f->show_matrix();
     //cout<<"adi MF forces\n"; ham.Ehrenfest_forces_adi(*dyn_vars.ampl_adi, 1).real().show_matrix();
    }

    // Adiabatic 
    else if(prms.rep_force==1){
      // 0 keeps the derivative-coupling terms (energy-conserving with every
      // integrator), 1 drops them - the local-diabatization-consistent
      // variant for the bare-energy integrators 0-2, 10-12
      int option = prms.ehrenfest_force_option;
      //*dyn_vars.f = ham.Ehrenfest_forces_adi(*dyn_vars.ampl_adi, 1, option, dyn_vars.proj_adi).real();
      // sqc_gamma adds the Meyer-Miller ZPE term gamma * tr(dH/dR) / |c|^2 -
      // 0 is the plain Ehrenfest force
      *dyn_vars.f = ham.Ehrenfest_forces_adi(*dyn_vars.ampl_adi, 1, option, prms.sqc_gamma, prms.ehrenfest_normalized).real();
/*
      CMATRIX& U = *ham.basis_transform;
      CMATRIX& C = *dyn_vars.ampl_adi;
      double nrm = (C.H() * U * U.H() * C).get(0,0).real();
      cout<<"nrm in Ehrenfest frcs = "<<nrm<<endl;
*/
    }     
    else{
      std::cout<<"Error in update_forces(): rep_force = "<<prms.rep_force<<" has no branch here (implemented: 0, 1)\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }
  }// Ehrenfest
  else if(prms.force_method==3){  // QTSH forces
    // Effective force determined by the TSH procedure 
    vector<int> effective_states = driving_states(prms, dyn_vars);

    // Diabatic 
    if(prms.rep_force==0){  *dyn_vars.f = ham.forces_dia(effective_states).real();   }

    // Adiabatic 
    else if(prms.rep_force==1){  
      *dyn_vars.f = ham.forces_adi(effective_states).real(); 
    }// rep_force == 1
    else{
      std::cout<<"Error in update_forces(): rep_force = "<<prms.rep_force<<" has no branch here (implemented: 0, 1)\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }

    // Off-diagonal Hellmann-Feynman force calculations
    // the density-matrix representations (rep_tdse 2 / 3) take it from rho (the amplitude form for a pure state)
    int dm_rep = (prms.rep_tdse==2 || prms.rep_tdse==3);
    if(dm_rep && prms.rep_force==0){
      *dyn_vars.qtsh_f_nc = ham.QTSH_forces_dia_dm(dyn_vars.dm_dia, 1, 0).real();
    }
    else if(dm_rep && prms.rep_force==1){
      int option = prms.qtsh_force_option;
      *dyn_vars.qtsh_f_nc = ham.QTSH_forces_adi_dm(dyn_vars.dm_adi, 1, option).real();
    }
    else if(prms.rep_force==0){  
      *dyn_vars.qtsh_f_nc = ham.QTSH_forces_dia(*dyn_vars.ampl_dia, 1).real();
    }
    else if(prms.rep_force==1){
      int option = prms.qtsh_force_option;

      *dyn_vars.qtsh_f_nc = ham.QTSH_forces_adi(*dyn_vars.ampl_adi, 1, option).real();
    }
    else{
      std::cout<<"Error in update_forces(): rep_force = "<<prms.rep_force<<" has no branch here (implemented: 0, 1)\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }

  }// QTSH
  else if(prms.force_method==4){  // KC-RPMD forces
    // Diabatic 
    if(prms.rep_force==0){  
     double beta = (hartree/boltzmann) / prms.Temperature; 
     double eta = prms.kcrpmd_eta; 
     double a = prms.kcrpmd_a; 
     double b = prms.kcrpmd_b; 
     double c = prms.kcrpmd_c;
     *dyn_vars.f = ham.kcrpmd_effective_force(dyn_vars.y_aux_var, *dyn_vars.q, *dyn_vars.iM, beta, eta, a, b, c);
    }

    // Adiabatic 
    else if(prms.rep_force==1){ cout<<"Error in update_forces(): KC-RPMD works exclusively within diabatic bases\n"; std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above"); }
    else{
      std::cout<<"Error in update_forces(): rep_force = "<<prms.rep_force<<" has no branch here (implemented: 0, 1)\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }
  }// KC-RPMD
  else{
    std::cout<<"Error in update_forces(): force_method = "<<prms.force_method<<" has no branch here (implemented: 0, 1, 2, 3, 4)\n";
    std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
  }

  // a NaN force reaching the kicks is a row the model did not hand over (the mean-field, QTSH and KC-RPMD forces
  // read every state's row)
  for(int itraj=0; itraj<ntraj; itraj++){
    for(int idof=0; idof<ndof; idof++){
      if(std::isnan(dyn_vars.f->get(idof, itraj))){
        std::cout<<"Error in update_forces(): the force along DOF "<<idof<<" of trajectory "<<itraj<<" is NaN under "
                 <<"force_method "<<prms.force_method<<" - a state's force the model did not hand over entered it "
                 <<"(force_source all_states hands over every state's, hop_force_refresh = 1 the new active state's)\n";
        std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
      }
    }
  }
//  return F;
}

//MATRIX aux_get_forces(bp::dict params, dyn_variables& dyn_vars, nHamiltonian& ham){
void update_forces(bp::dict params, dyn_variables& dyn_vars, nHamiltonian& ham){

  dyn_control_params prms;
  prms.set_parameters(params);

  update_forces(prms, dyn_vars, ham);
//  return aux_get_forces(prms, dyn_vars, ham);
  
}



vector<CMATRIX> get_Eadi(nHamiltonian& ham){

  int nst = ham.nadi;
  int ntraj = ham.children.size();

  vector<CMATRIX> Eadi(ntraj, CMATRIX(nst, nst));

  for(int traj=0; traj<ntraj; traj++){
    Eadi[traj] = ham.children[traj]->get_ham_adi(); 
  }

  return Eadi;

}

vector<CMATRIX> get_Edia(nHamiltonian& ham){
/** The diabatic twin of get_Eadi: the Lowdin-frame diabatic Hamiltonian X ham_dia X of every child (ham_dia itself for an
    identity metric) - the energies the diabatic hopping (hop_acceptance_algo 11) and the diabatic DISH rates read */
  int nst = ham.ndia;
  int ntraj = ham.children.size();
  vector<CMATRIX> Edia(ntraj, CMATRIX(nst, nst));
  for(int traj=0; traj<ntraj; traj++){
    Edia[traj] = ham.children[traj]->get_ham_dia_lowdin();
  }
  return Edia;
}

vector<CMATRIX> get_Eadi(nHamiltonian* ham){

  int nst = ham->nadi;
  int ntraj = ham->children.size();

  vector<CMATRIX> Eadi(ntraj, CMATRIX(nst, nst));

  for(int traj=0; traj<ntraj; traj++){
    Eadi[traj] = ham->children[traj]->get_ham_adi();
  }

  return Eadi;

}



}// namespace libdyn
}// liblibra
