/*********************************************************************************
* Copyright (C) 2019-2020 Alexey V. Akimov
*
* This file is distributed under the terms of the GNU General Public License
* as published by the Free Software Foundation, either version 3 of
* the License, or (at your option) any later version.
* See the file LICENSE in the root directory of this distribution
* or <http://www.gnu.org/licenses/>.
*
*********************************************************************************/
/**
  \file dyn_hop_acceptance.cpp
  \brief The file implements the functions to determine if to accept hops
*/

#include <cmath>
#include "Surface_Hopping.h"
#include <stdexcept>
#include "Energy_and_Forces.h"
#include "electronic/libelectronic.h"
///#include "Dynamics.h"
#include "dyn_control_params.h"
#include "dyn_hop_acceptance.h"
#include "../math_specialfunctions/libspecialfunctions.h"

/// liblibra namespace
namespace liblibra{

using namespace libspecialfunctions;

/// libdyn namespace
namespace libdyn{


namespace bp = boost::python;


int can_rescale_along_vector(double E_old, double E_new, MATRIX& p, MATRIX& invM, MATRIX& t, vector<int>& which_dofs){
/**
  Check if we can rescale momenta p along the vector t such that the total energy is conserved

*/

  int ndof = which_dofs.size();

  // According to Fabiano
  double a_ij = 0.0;
  double b_ij = 0.0;
  double c_ij = E_old - E_new;
  double t2 = 0.0;

  for(int idof = 0; idof < ndof; idof++){  

    int dof = which_dofs[idof];

    a_ij += t.get(dof, 0) * t.get(dof, 0) * invM.get(dof, 0); 
    b_ij += p.get(dof, 0) * t.get(dof, 0) * invM.get(dof, 0); 
    t2 += t.get(dof, 0) * t.get(dof, 0);

  }// for dof
 
  a_ij *= 0.5;

  double det = b_ij*b_ij + 4.0*a_ij*c_ij;

  int res = 0;
  // a null direction (t = 0, so a_ij = b_ij = 0) can neither absorb nor supply an
  // energy difference - det = 0 answered yes for ANY gap. Only an isoenergetic hop
  // is feasible along it
  if(t2 == 0.0){ if(fabs(c_ij) <= 1e-12){ res = 1; } }
  else if(det >= 0.0){ res = 1; }

  return res;
}

int can_rescale_along_vector(double E_old, double E_new, MATRIX& p, MATRIX& invM, MATRIX& t){

  int ndof = p.n_rows;
  vector<int> which_dofs(ndof, 0);
  for(int i = 0; i < ndof; i++){ which_dofs[i] = i; }

  return can_rescale_along_vector(E_old, E_new, p, invM, t, which_dofs);

}



void rescale_along_vector(double E_old, double E_new, MATRIX& p, MATRIX& invM, MATRIX& t, int do_reverse, vector<int>& which_dofs){
/**
  Rescale momenta p along the vector t such that the total energy is conserved

  or reverse the momentum along the vector t

*/
  int idof, dof; 
  int ndof = which_dofs.size();

  // According to Fabiano
  double a_ij = 0.0;
  double b_ij = 0.0;
  double c_ij = E_old - E_new;
  double t2 = 0.0;
    
  for(idof=0; idof < ndof; idof++){  

    dof = which_dofs[idof];

    a_ij += t.get(dof, 0) * t.get(dof, 0) * invM.get(dof, 0); 
    b_ij += p.get(dof, 0) * t.get(dof, 0) * invM.get(dof, 0); 
    t2 += t.get(dof, 0) * t.get(dof, 0);

  }// for dof
 
  a_ij *= 0.5;

  double det = b_ij*b_ij + 4.0*a_ij*c_ij;

  // Calculate the scaling factor and new state
  double gamma_ij = 0.0;

  if(det<0.0){  // Frustrated hops!

    // the reversal p -> p - (b/a) t is the reflection of p along t, defined for every direction
    // (a = t M^-1 t / 2 > 0 whenever t != 0) - a weak direction reverses as a strong one does
    if(do_reverse && a_ij > 0.0){ gamma_ij = b_ij / a_ij; }
    else{ gamma_ij = 0.0;  }

  }
  else{    // Accepted hops!

    if(fabs(a_ij)>1e-12){
      if(b_ij<0){ gamma_ij = 0.5*(b_ij + sqrt(det))/a_ij; }
      else{       gamma_ij = 0.5*(b_ij - sqrt(det))/a_ij; }
    }
    else if(t2 > 0.0){
      // the same (physical) root in its cancellation-free form -2c/(b + sign(b) sqrt(det)),
      // finite as a -> 0 (-> -c/b) where the quotient form loses it: the momentum change
      // gamma*t stays finite for a tiny direction, so the energy difference is still paid
      double sgn = (b_ij < 0.0) ? -1.0 : 1.0;
      double den = b_ij + sgn * sqrt(det);
      if(den != 0.0){ gamma_ij = -2.0 * c_ij / den; }
    }
    // t = 0: only an isoenergetic hop reaches here - nothing to scale along

  }

  //Rescale velocities and do the hop
  p -= gamma_ij * t;

/* 
  MATRIX dp(t.n_rows, t.n_cols);
  dp = gamma_ij * t;

  /// Only shift the momenta along the active dofs (`which_dofs`)
  /// p -= dp; (only for the active dofs)
  for(idof=0; idof < ndof; idof++){  
    dof = which_dofs[idof];
    p.add(dof, 0, -dp.get(dof, 0) );
  }
*/
}


void rescale_along_vector(double E_old, double E_new, MATRIX& p, MATRIX& invM, MATRIX& t, int do_reverse){

  int ndof = p.n_rows;
  vector<int> which_dofs(ndof, 0);
  for(int i = 0; i < ndof; i++){ which_dofs[i] = i; }

  rescale_along_vector(E_old, E_new, p, invM, t, do_reverse, which_dofs);

}



vector<double> Boltz_quant_prob(vector<double>& E, double T){
    /**

    Computes the quantum Boltzman probability of occupying different 
    energy levels at given temperature T:  P_i = exp(-E_i/kT) / Z
    where Z = sum_i { exp(-E_i/kT)  }

    Args:
        E ( list of doubles ): energy levels [in a.u.]
        T ( double ): temperature [K]

    Returns:
        double: prob: the probability to  find a system in a discrete state i with
            energy E_i at given temperature T, considering a number of selected energy
            levels as given by the list E
    */
    int n;
 
    double kB = boltzmann/hartree; 
    double b = 1.0/(kB * T);
    int nstates = E.size();

    double Z = 0.0;  // partition function
    vector<double> prob(nstates, 0.0);

    for(n=0; n<nstates; n++){
      prob[n] = exp(-E[n]*b);
      Z += prob[n];
    }
    
    for(n=0; n<nstates; n++){
      prob[n] = prob[n] / Z;
    }

    return prob;
}



static void force_nan_error(const char* where, int old_st, int new_st, int dof){
/**
  The force-difference hop test and the Jasper-Truhlar criterion read the forces of both states - a model handing
  over its active state's force alone leaves the other NaN
*/
  std::cout<<"Error in "<<where<<": the force of state "<<old_st<<" or "<<new_st<<" along DOF "<<dof
           <<" is NaN - the force-difference hop acceptance and momentum rescaling read the forces of both states - the"
           <<" model handed over its active state's alone (force_source all_states hands over every state's)\n";
  std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
}

int can_rescale_for_hop(dyn_variables& dyn_var, nHamiltonian& ham, int traj, int old_st, int new_st,
                        dyn_control_params& prms){
/**
  Whether the momentum change momenta_rescaling_algo configures can pay for the hop old_st -> new_st of trajectory
  traj: the uniform scalings (100 / 101 on the adiabatic, 110 / 111 on the Lowdin diabatic energies) need a positive
  kinetic energy over the quantum DOFs no smaller than the energy gap, the directional ones (200 / 201 along the
  derivative coupling, 210 / 211 along the force difference) a direction can_rescale_along_vector accepts; 0 and 40
  change no momentum here and pay for every hop. accept_hops refuses an unpayable hop before the post-hop decoherence
  collapses the amplitudes, handle_hops_nuclear reverses the momentum of such a frustrated hop under x01 / x11
*/
  int res = prms.momenta_rescaling_algo;
  if(old_st == new_st || res==0 || res==40){ return 1; }

  vector<int>& which_dofs = prms.quantum_dofs;
  int ndof = dyn_var.ndof;
  int n_active_dof = which_dofs.size();
  MATRIX p_tr(dyn_var.p->col(traj));
  MATRIX& invM = *dyn_var.iM;
  // the NBRA fast path keeps one Hamiltonian, the child of trajectory 0
  int src = (prms.isNBRA==1) ? 0 : traj;

  if(rescaling_is_uniform(res)){
    double T_i = compute_kinetic_energy(p_tr, invM, which_dofs);
    double E_i, E_f;
    if(rescaling_reads_adiabatic(res)){
      CMATRIX hvib(ham.child(src)->get_ham_adi());
      E_i = hvib.get(old_st, old_st).real();
      E_f = hvib.get(new_st, new_st).real();
    }
    else{
      CMATRIX hvib(ham.child(traj)->get_ham_dia_lowdin());
      E_i = hvib.get(old_st, old_st).real();
      E_f = hvib.get(new_st, new_st).real();
    }
    // no kinetic energy to draw on, or a zero momentum no factor can grow: unpayable unless isoenergetic
    return ((T_i > 0.0 && T_i + E_i - E_f >= 0.0) || E_i == E_f) ? 1 : 0;
  }

  if(rescaling_is_directional(res)){
    CMATRIX hvib(ham.child(src)->get_ham_adi());
    double E_i = hvib.get(old_st, old_st).real();
    double E_f = hvib.get(new_st, new_st).real();
    MATRIX t(ndof, 1);
    for(int idof = 0; idof < n_active_dof; idof++){
      int dof = which_dofs[idof];
      if(rescaling_along_nac(res)){
        CMATRIX nac(ham.child(traj)->get_dc1_adi(dof));
        t.set(dof, 0, nac.get(old_st, new_st).real());
      }
      else{
        CMATRIX df(ham.child(traj)->get_d1ham_adi(dof));
        if(std::isnan(df.get(old_st, old_st).real()) || std::isnan(df.get(new_st, new_st).real())){
          force_nan_error("can_rescale_for_hop()", old_st, new_st, dof);
        }
        t.set(dof, 0, df.get(old_st, old_st).real() - df.get(new_st, new_st).real());
      }
    }
    return can_rescale_along_vector(E_i, E_f, p_tr, invM, t, which_dofs);
  }

  std::cout<<"Error in can_rescale_for_hop(): momenta_rescaling_algo = "<<res
           <<" has no branch here (implemented: 0, 40, 100, 101, 110, 111, 200, 201, 210, 211)\n";
  std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
}


double Boltz_cl_prob(double E, double T){
    /**

    Computes the normalized classical Boltzmann probability distribution function 

    Args: 
        E ( double ): the minimum energy level [in a.u.]
        T ( double ): temperature [K]

    Returns:
        double: The probability to have kinetic energy greater than a given threshold value at
            given temperature

    See Also:
        This is essentially a Maxwell-Boltzmann distribution in the energy scale
        Used this: http://mathworld.wolfram.com/MaxwellDistribution.html

    */
 
    int n;
 
    double kB = boltzmann/hartree; 
    double b = 1.0/(kB * T);

    double Z = 0.0;  // partition function

    double x = sqrt(E * b);
    double res = 1.0;

    double D = (2.0*b/sqrt(M_PI)) * x * exp(-x);
    if( D>1.0 || D<0.0 ){
      cout<<"D = "<<D<<endl;
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }

    res = D;

    return res;

}



double Boltz_cl_prob_up(double E, double T){
    /**

    Computes the classical Boltzmann probability to have kinetic energy larger than a given 
    threshold E  at temperature T. See Eq. 7 of probabilities_theory.docx.
    The present function is related to it.

    Args: 
        E ( double ): the minimum energy level [in a.u.]
        T ( double ): temperature [K]

    Returns:
        double: The probability to have kinetic energy greater than a given threshold value at
            given temperature

    See Also:
        This is essentially a Maxwell-Boltzmann distribution in the energy scale
        Used this: http://mathworld.wolfram.com/MaxwellDistribution.html

    */

    double kB = boltzmann/hartree; 
 
    double x = sqrt(E/(kB * T));

    double D = ERF(x) - sqrt(4.0/M_PI) * x * exp(-x*x);
    if( D>1.0 || D<0.0){
      cout<<"D = "<<D<<endl;
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }

    double res = 1.0 - D;

    return res;

}



double HO_prob(vector<double>& E, vector<int>& qn, double T, vector<double>& prob){

    /**
    
    Probability that the oscillators are in the given vibrational states
    Multi-oscillator generalization of Eq. 10

    Args:
        E ( list of doubles ): all the energy levels present in the system [in a.u.]
        qn ( list of ints ): quantum numbers for each oscillator
        T ( double ): temperature 

    Returns:
        tuple: (res, prob), where:

            * res ( double ): the probability that the system of N oscillators is in a given
                state, defined by quantum numbers of each oscillator
            * prob ( list of N doubles ): probability with which each of N oscillators occupies 
                given vibrational state
    
    */

    int n_freqs = E.size();

    double kB = boltzmann/hartree; 
 
    double res = 1.0;
    prob.clear();

    for(int i=0; i<n_freqs; i++){

        double xi = exp(-E[i]/(kB*T));
        prob.push_back( pow(xi, qn[i]) * (1.0 - xi) );
        res *= prob[i];
    }

    return res;
}



double HO_prob_up(vector<double>& E, vector<int>& qn, double T, vector<double>& prob){

    /**
    
    Probability that the oscillators are in the given vibrational states
    Multi-oscillator generalization of Eq. 10

    Args:
        E ( list of doubles ): all the energy levels present in the system [in a.u.]
        qn ( list of ints ): quantum numbers for each oscillator
        T ( double ): temperature 

    Returns:
        tuple: (res, prob), where:

            * res ( double ): the probability that the system of N oscillators is in a given
                state, defined by quantum numbers of each oscillator
            * prob ( list of N doubles ): probability with which each of N oscillators occupies 
                given vibrational state
    
    */

    int n_freqs = E.size();

    double kB = boltzmann/hartree; 
 
    double res = 1.0;
    prob.clear();

    for(int i=0; i<n_freqs; i++){

        double xi = exp(-E[i]/(kB*T));
        prob.push_back( xi );
        res *= prob[i];
    }

    return res;
}




double boltz_factor(double E_new, double E_old, double T, int boltz_opt){
    /**
    Compute the Boltzmann scaling factor, but only if we consider a hop up in energy

    Args: 
        E_new ( double ): the energy of the proposed (new) state [units: a.u.]
        E_old ( double ): the energy of the current (old) state [units: a.u.]
        T ( double ): temperature of the bath [units: K]
        boltz_opt ( int ): the proposed hop acceptance criterion

            * 0: all hops are accepted
            * 1: hops are accepted according to the Boltzmann ratio of the final and initial states
            * 2: hops are accepted according to the classical Maxwell-Boltzmann distribution
            * 3: hops are accepted based on the quantum probability of the final state

    Returns:
        double: boltz_f: the probability of the proposed hop acceptance

    */

    double dE = (E_new - E_old);
    double boltz_f = 1.0;
    double kB = boltzmann/hartree; 
    double argg;

    if(boltz_opt==0){
        boltz_f = 1.0;
    }
     
    else if(boltz_opt==1){
        if(dE > 0.0){
            argg = dE/(kB*T);
            if( argg > 50.0){ boltz_f = 0.0;         }
            else{             boltz_f = exp(-argg);  }
        }
    }

    else if(boltz_opt==2){
        if(dE > 0.0){
            boltz_f = Boltz_cl_prob_up(dE, T);
        }
    }

    else if(boltz_opt==3){
        if(dE > 0.0){
            vector<double> energies(2);
            energies[0] = 0.0;
            energies[1] = dE;
            boltz_f = Boltz_quant_prob(energies, T)[1];
        }
    }

    return boltz_f;

}


//accept_hops(dyn_var, ham, prms, prop_states, rnd);
/*
vector<int> accept_hops(dyn_control_params& prms,
       MATRIX& q, MATRIX& p, MATRIX& invM, CMATRIX& C, 
       nHamiltonian& ham, vector<int>& proposed_states, vector<int>& initial_states, Random& rnd, 
       vector<int>& which_trajectories){
*/
/// the Hamiltonian whose diagonal the Boltzmann acceptances 31-33 read: the hopping representation's
/// own - the Lowdin-frame diabatic one under rep_sh 0, the adiabatic one under rep_sh 1
static CMATRIX hopping_energies(dyn_control_params& prms, nHamiltonian& ham, int traj){
  return (prms.rep_sh == 0) ? ham.child(traj)->get_ham_dia_lowdin() : ham.child(traj)->get_ham_adi();
}

vector<int> accept_hops(dyn_variables& dyn_var, nHamiltonian& ham, vector<int>& proposed_states,  vector<int>& initial_states,
                        dyn_control_params& prms,Random& rnd,  vector<int>& which_trajectories){
/**
  This function returns the new state indices if the corresponding transitions can be
  accepted according to given criteria

  C - is assumed to be dynamically-consistent

  options:
  0 - accept all

  10 - based on adiabatic energy
  11 - based on diabatic energy

  20 - derivative coupling vectors
  21 - difference of state-specific forces

  31 - quantum Boltzmann
  32 - Maxwell-Boltzmann
  33 - updated quantum Boltzmann

  40 - based on tcnbra_ekin

*/

  vector<int>& which_dofs = prms.quantum_dofs;

  MATRIX q(*dyn_var.q);
  MATRIX p(*dyn_var.p);
  MATRIX invM(*dyn_var.iM);
  CMATRIX C(*dyn_var.ampl_adi); 
//  vector<int> initial_states = dyn_var.act_states; 

  int ndof = q.n_rows;
  int ndof_active = which_dofs.size();
  int ntraj = q.n_cols;
  int ntraj_active = which_trajectories.size();
  int nst = C.n_rows;    
  int itraj, traj, dof, i, idof;
  int isNBRA = prms.isNBRA;

  vector<int> fstates(ntraj, -1); 
  MATRIX p_tr(ndof, 1);
  CMATRIX hvib(nst, nst);
  CMATRIX nac(nst, nst);
  CMATRIX df(nst, nst);
  CMATRIX U(nst, nst); // proj_adi;

//  cout<<"In accept_hops with algo = "<<prms.hop_acceptance_algo<<endl;
//  cout<<"ntraj_active = "<<ntraj_active<<endl;


  if(prms.hop_acceptance_algo==0){  // Just accept all the proposed hops

    for(itraj=0; itraj<ntraj_active; itraj++){
      traj = which_trajectories[itraj];

      fstates[traj] = proposed_states[traj];
    }

  }// algo = 0

  else if(prms.hop_acceptance_algo==10){  // Just based on the adiabatic energy levels

    for(itraj=0; itraj<ntraj_active; itraj++){
      traj = which_trajectories[itraj];

      int old_st = initial_states[traj];
      int new_st = proposed_states[traj];

      if(old_st != new_st){

        p_tr = p.col(traj);
        double T_i = compute_kinetic_energy(p_tr, invM, which_dofs); // initial kinetic energy

        // Do the calculations only for itraj=0 if isNBRA is 1
        if(isNBRA==1){
          if(itraj==0){
            hvib = ham.child(traj)->get_ham_adi();
            //hvib = projectors[traj].H() * hvib * projectors[traj];
          }
        }

        else{
          hvib = ham.child(traj)->get_ham_adi();
          //hvib = projectors[traj].H() * hvib * projectors[traj];
        }
        
        double E_i = hvib.get(old_st, old_st).real();  // initial potential energy
        double E_f = hvib.get(new_st, new_st).real();  // final potential energy  
        double T_f = T_i + E_i - E_f;             // predicted final kinetic energy

//        cout<<"E_i = "<<E_i<<" E_f = "<<E_f<<" T_i = "<<T_i<<" T_f = "<<T_f<<endl;

        if(T_f>=0.0){  // hop is possible - accept it
          fstates[traj] = proposed_states[traj];
        }
        else{
          fstates[traj] = initial_states[traj];
        }      
      }
      else{ 
        fstates[traj] = initial_states[traj];
      }

    }// for traj

  }// algo = 10

  else if(prms.hop_acceptance_algo==11){  // Just based on the diabatic energy levels

    for(itraj=0; itraj<ntraj_active; itraj++){
      traj = which_trajectories[itraj];

      int old_st = initial_states[traj];
      int new_st = proposed_states[traj];

      if(old_st != new_st){

        p_tr = p.col(traj);
        double T_i = compute_kinetic_energy(p_tr, invM, which_dofs); // initial kinetic energy
        double E_i = ham.child(traj)->get_ham_dia_lowdin().get(old_st, old_st).real();  // initial potential energy
        double E_f = ham.child(traj)->get_ham_dia_lowdin().get(new_st, new_st).real();  // final potential energy  
        double T_f = T_i + E_i - E_f;             // predicted final kinetic energy

        if(T_f>=0.0){  // hop is possible - accept it
          fstates[traj] = proposed_states[traj];
        }
        else{
          fstates[traj] = initial_states[traj];
        }      
      }
      else{ 
        fstates[traj] = initial_states[traj];
      }

    }// for traj
  }// algo = 11


  else if(prms.hop_acceptance_algo==20){  // if rescaling momenta along the derivative coupling vector

    MATRIX dNAC(ndof, 1);

    for(itraj=0; itraj<ntraj_active; itraj++){
      traj = which_trajectories[itraj];

      int old_st = initial_states[traj];
      int new_st = proposed_states[traj];

      if(old_st != new_st){

        // Do the calculations only for itraj=0 if isNBRA is 1
        if(isNBRA==1){
        if(itraj==0){
        hvib = ham.child(traj)->get_ham_adi();
        //hvib = projectors[traj].H() * hvib * projectors[traj];
        }
        }
        else{
        hvib = ham.child(traj)->get_ham_adi();
        //hvib = projectors[traj].H() * hvib * projectors[traj];
        }
        double E_i = hvib.get(old_st, old_st).real();  // initial potential energy
        double E_f = hvib.get(new_st, new_st).real();  // final potential energy  

        dNAC = 0.0;
        //for(dof = 0; dof < ndof; dof++){
        for(idof = 0; idof < ndof_active; idof++){
          dof = which_dofs[idof];
          nac = ham.child(traj)->get_dc1_adi(dof);
          U = *dyn_var.proj_adi[traj];
          //nac = U.H() * nac * U;
          //nac = projectors[traj].H() * nac * projectors[traj];

          dNAC.set(dof, 0, nac.get(old_st, new_st).real() );
        }
      
        p_tr = p.col(traj);
        if(can_rescale_along_vector(E_i, E_f, p_tr, invM, dNAC, which_dofs)){
          fstates[traj] = proposed_states[traj];
        }
        else{ fstates[traj] = initial_states[traj];   }
      }
      else{   fstates[traj] = initial_states[traj];   }
    }// for traj

  }// algo = 20

  else if(prms.hop_acceptance_algo==21){  // if rescaling momenta along the difference in forces

    MATRIX dF(ndof, 1);
/*
    MATRIX Fi(ndof, ntraj);
    MATRIX Ff(ndof, ntraj);
    CMATRIX _ampl_i(nst, ntraj);     // CMATRIX version of "initial_states"
    CMATRIX _ampl_f(nst, ntraj);     // CMATRIX version of "proposed_states"

    tsh_indx2vec(ham, _ampl_i, initial_states);
    Fi = ham.Ehrenfest_forces_adi(_ampl_i, 1).real();

    tsh_indx2vec(ham, _ampl_f, proposed_states);
    Ff = ham.Ehrenfest_forces_adi(_ampl_f, 1).real();
*/


    for(itraj=0; itraj<ntraj_active; itraj++){
      traj = which_trajectories[itraj];

      int old_st = initial_states[traj];
      int new_st = proposed_states[traj];

      if(old_st != new_st){
        // Do the calculations only for itraj=0 if isNBRA is 1
        if(isNBRA==1){
        if(itraj==0){
        hvib = ham.child(traj)->get_ham_adi();
        //hvib = projectors[traj].H() * hvib * projectors[traj];
        }
        }
        else{
        hvib = ham.child(traj)->get_ham_adi();
        //hvib = projectors[traj].H() * hvib * projectors[traj];
        }
        double E_i = hvib.get(old_st, old_st).real();  // initial potential energy
        double E_f = hvib.get(new_st, new_st).real();  // final potential energy        

        //for(dof = 0; dof < ndof; dof++){
        dF = 0.0;
        for(idof = 0; idof < ndof_active; idof++){
          dof = which_dofs[idof];
          df = ham.child(traj)->get_d1ham_adi(dof);
          //df = projectors[traj].H() * df * projectors[traj];
          if(std::isnan(df.get(old_st, old_st).real()) || std::isnan(df.get(new_st, new_st).real())){
            force_nan_error("the force-difference hop test", old_st, new_st, dof);
          }
          dF.set(dof, 0, df.get(old_st, old_st).real() - df.get(new_st, new_st).real());

        }
      
        p_tr = p.col(traj);
        if(can_rescale_along_vector(E_i, E_f, p_tr, invM, dF, which_dofs)){
          fstates[traj] = proposed_states[traj];
        }
        else{
          fstates[traj] = initial_states[traj];
        }
      }
      else{ 
        fstates[traj] = initial_states[traj];
      }

    }// for traj

  }// algo = 21


  else if(prms.hop_acceptance_algo==31){  // stochastic decision based on quantum Boltzmann factors
    // 31-33 read the energies of the HOPPING states in the hopping representation:
    // adiabatic for rep_sh 1, the Lowdin-frame diabatic energies for rep_sh 0 (ham_dia
    // at S = I) - the Boltzmann factor gives diabatic hops the same detailed balance.
    // The LZ gap-minimum correction indexes adiabatic states and stays with rep_sh 1

    for(itraj=0; itraj<ntraj_active; itraj++){
      traj = which_trajectories[itraj];

      int old_st = initial_states[traj];
      int new_st = proposed_states[traj];
      // Do the calculations only for itraj=0 if isNBRA is 1
      if(isNBRA==0 || itraj==0){ hvib = hopping_energies(prms, ham, traj); }
      //hvib = projectors[traj].H() * hvib * projectors[traj];
      double E_i = hvib.get(old_st, old_st).real();  // initial potential energy
      double E_f = hvib.get(new_st, new_st).real();  // final potential energy  

      // the LZ/ZN proposals (tsh_method 3/4) fire one step past the
      // crossing, where the gap has reopened - evaluate the thermal
      // filter at the stored at-crossing gap instead, the same geometry
      // the proposal fires for
      if(prms.rep_sh == 1 && (prms.tsh_method == 3 || prms.tsh_method == 4) &&
         dyn_var.lz_gap_min_status == 1){
        double gm = dyn_var.lz_gap_min->get(new_st, traj);
        if(gm >= 0.0 && E_f > E_i){ E_f = E_i + gm; }
      }

      double prob = boltz_factor(E_f, E_i, prms.Temperature, 1);

      double ksi = rnd.uniform(0.0, 1.0);

      if(ksi < prob ){  
        fstates[traj] = proposed_states[traj]; 
      }
      else{ 
        fstates[traj] = initial_states[traj]; 
      }

    }
  }// algo = 31

  else if(prms.hop_acceptance_algo==32){  // stochastic decision based on classical Maxwell-Boltzmann factors

    for(itraj=0; itraj<ntraj_active; itraj++){
      traj = which_trajectories[itraj];

      int old_st = initial_states[traj];
      int new_st = proposed_states[traj];
      // Do the calculations only for itraj=0 if isNBRA is 1
      if(isNBRA==0 || itraj==0){ hvib = hopping_energies(prms, ham, traj); }
      //hvib = projectors[traj].H() * hvib * projectors[traj];
      double E_i = hvib.get(old_st, old_st).real();  // initial potential energy
      double E_f = hvib.get(new_st, new_st).real();  // final potential energy  

      // the LZ/ZN proposals (tsh_method 3/4) fire one step past the
      // crossing, where the gap has reopened - evaluate the thermal
      // filter at the stored at-crossing gap instead, the same geometry
      // the proposal fires for
      if(prms.rep_sh == 1 && (prms.tsh_method == 3 || prms.tsh_method == 4) &&
         dyn_var.lz_gap_min_status == 1){
        double gm = dyn_var.lz_gap_min->get(new_st, traj);
        if(gm >= 0.0 && E_f > E_i){ E_f = E_i + gm; }
      }

      double prob = boltz_factor(E_f, E_i, prms.Temperature, 2);

      double ksi = rnd.uniform(0.0, 1.0);

      if(ksi < prob ){  
        fstates[traj] = proposed_states[traj]; 
      }
      else{ 
        fstates[traj] = initial_states[traj]; 
      }

    }

  }// algo = 32

  else if(prms.hop_acceptance_algo==33){  // stochastic decision based on quantum probabilities

    for(itraj=0; itraj<ntraj_active; itraj++){
      traj = which_trajectories[itraj];

      int old_st = initial_states[traj];
      int new_st = proposed_states[traj];
      // Do the calculations only for itraj=0 if isNBRA is 1
      if(isNBRA==0 || itraj==0){ hvib = hopping_energies(prms, ham, traj); }
      //hvib = projectors[traj].H() * hvib * projectors[traj];
      double E_i = hvib.get(old_st, old_st).real();  // initial potential energy
      double E_f = hvib.get(new_st, new_st).real();  // final potential energy  

      // the LZ/ZN proposals (tsh_method 3/4) fire one step past the
      // crossing, where the gap has reopened - evaluate the thermal
      // filter at the stored at-crossing gap instead, the same geometry
      // the proposal fires for
      if(prms.rep_sh == 1 && (prms.tsh_method == 3 || prms.tsh_method == 4) &&
         dyn_var.lz_gap_min_status == 1){
        double gm = dyn_var.lz_gap_min->get(new_st, traj);
        if(gm >= 0.0 && E_f > E_i){ E_f = E_i + gm; }
      }

      double prob = boltz_factor(E_f, E_i, prms.Temperature, 3);

      double ksi = rnd.uniform(0.0, 1.0);

      if(ksi < prob ){  
        fstates[traj] = proposed_states[traj]; 
      }
      else{ 
        fstates[traj] = initial_states[traj]; 
      }

    }

  }// algo = 33

  else if(prms.hop_acceptance_algo==40){  // based on tcnbra_ekin

    for(itraj=0; itraj<ntraj_active; itraj++){
      traj = which_trajectories[itraj];
      hvib = ham.child(traj)->get_hvib_adi();

      int old_st = initial_states[traj];
      int new_st = proposed_states[traj];

      double E_i = hvib.get(old_st, old_st).real();  // initial potential energy
      double E_f = hvib.get(new_st, new_st).real();  // final potential energy

      double ekin_new = E_i + dyn_var.tcnbra_ekin[traj] - E_f;

      if(ekin_new >= 0.0){    fstates[traj] = proposed_states[traj];   }
      else{    fstates[traj] = initial_states[traj];      }

    }// for itraj
  }// algo == 40
  else{
    std::cout<<"Error in accept_hops(): hop_acceptance_algo = "<<prms.hop_acceptance_algo<<" has no branch here (implemented: 0, 10, 11, 20, 21, 31, 32, 33, 40)\n";
    std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
  }


  // an accepted hop the configured rescaling cannot pay for is refused here, before the post-hop decoherence
  // (and DISH) collapse the amplitudes onto the states returned - handle_hops_nuclear treats it as a frustrated hop
  int res = prms.momenta_rescaling_algo;
  if(rescaling_is_uniform(res) || rescaling_is_directional(res)){
    for(itraj=0; itraj<ntraj_active; itraj++){
      traj = which_trajectories[itraj];
      if(fstates[traj] != initial_states[traj] &&
         !can_rescale_for_hop(dyn_var, ham, traj, initial_states[traj], fstates[traj], prms)){
        fstates[traj] = initial_states[traj];
      }
    }
  }

  return fstates;
}

/*
vector<int> accept_hops(dyn_control_params& prms,
       MATRIX& q, MATRIX& p, MATRIX& invM, CMATRIX& C, 
       nHamiltonian& ham, vector<int>& proposed_states, vector<int>& initial_states, Random& rnd ){
*/
vector<int> accept_hops(dyn_variables& dyn_var, nHamiltonian& ham, vector<int>& proposed_states, vector<int>& initial_states,
                        dyn_control_params& prms,Random& rnd){

    int ntraj = dyn_var.ntraj;
    vector<int> which_trajectories(ntraj);

    for(int i=0; i<ntraj; i++){ which_trajectories[i] = i; }

    return accept_hops(dyn_var, ham, proposed_states, initial_states, prms, rnd, which_trajectories);

//    return accept_hops(prms, q, p, invM, C, /*projectors,*/ ham, proposed_states, initial_states, rnd, which_trajectories);

}




/*
vector<int> where_can_we_hop(int traj, dyn_control_params& prms,
       MATRIX& q, MATRIX& p,  MATRIX& invM, CMATRIX& Coeff,
       nHamiltonian& ham, vector<int>& act_states, Random& rnd){
*/
vector<int> where_can_we_hop(int traj, dyn_variables& dyn_var, nHamiltonian& ham, dyn_control_params& prms,Random& rnd){
/**
   This function gives a list of indices of states to which a trajectory of index `traj` can
   hop to from its current active state
   We want to determine only the non-trivial transitions (that is everything  other than hopping onto itself)

   act_states = [ state_0,  state_1, ... <state_traj>, .... ] is the array of the current state indices for each trajectory
 
   So, we are interested in the particular trajectory `traj` and we agoing to check onto which other states can that trajectory hop

   We don't care about all other values in act_states, q, p, etc. ... other than corresponding to the index `traj`

*/
//  MATRIX& q = *dyn_var.q;
//  MATRIX& p = *dyn_var.p;
//  MATRIX& invM = *dyn_var.iM;
//  CMATRIX& Coeff = *dyn_var.ampl_adi;
    vector<int> act_states = dyn_var.act_states;

    int ntraj = act_states.size();
    int nstates = dyn_var.nadi; // Coeff.n_rows;

    vector<int> proposed_states(ntraj, -1);
    vector<int> new_states(ntraj, -1);
    vector<int> which_trajectories(1, traj);
    vector<int> all_possible_hops;

    /// Determine whereto we can hop from the current state:
    for(int istate=0; istate<nstates; istate++){

      if(istate != act_states[traj]){

        proposed_states[traj] = istate;

        /// Decide if we can accept the transitions, the function below only checks the hopping for a single trajectory `traj`
        /// other elements of the input and output vector<int> variables (old_states, new_states, proposed_states) are irrelevant
        /// the variable `which_trajectories` instructs to handle only the current trajectory
        //new_states = accept_hops(prms, q, p, invM, Coeff, /*projectors,*/ ham, proposed_states, act_states, rnd, which_trajectories);

        new_states = accept_hops(dyn_var, ham, proposed_states, act_states, prms, rnd, which_trajectories);

        if(new_states[traj]!=act_states[traj]){
          all_possible_hops.push_back(new_states[traj]);
        }

      }/// if not the current active state 

    }/// for istate

    return all_possible_hops;
}



/*
void handle_hops_nuclear(dyn_control_params& prms,
       MATRIX& q, MATRIX& p, MATRIX& invM, CMATRIX& C,
       nHamiltonian& ham, vector<int>& new_states, vector<int>& old_states){
*/
static int hop_target(dyn_variables& dyn_var, nHamiltonian& ham, int traj, int old_st, int new_st, int prop_st,
                      dyn_control_params& prms){
/**
  The state the momentum change of a trajectory refers to: the new state of an accepted hop, the proposed state of a
  hop accept_hops refused for the energy (a frustrated hop - the x01 / x11 reversal applies), old_st when no hop was
  proposed or a stochastic acceptance refused a payable one
*/
  if(new_st != old_st){ return new_st; }
  if(prop_st != old_st && !can_rescale_for_hop(dyn_var, ham, traj, old_st, prop_st, prms)){ return prop_st; }
  return old_st;
}

static void unpayable_accepted_hop(int traj, int old_st, int new_st, dyn_control_params& prms){
/**
  accept_hops refuses a hop its rescaling cannot pay for (can_rescale_for_hop), so an accepted hop that reaches the
  rescaling unpayable means the states or the momenta changed between the acceptance and the rescaling
*/
  std::cout<<"Error in handle_hops_nuclear(): the accepted hop "<<old_st<<" -> "<<new_st<<" of trajectory "<<traj
           <<" cannot conserve the total energy with momenta_rescaling_algo = "<<prms.momenta_rescaling_algo
           <<" - accept_hops refuses such a hop (can_rescale_for_hop), so the states or the momenta changed between"
           <<" the acceptance and the rescaling\n";
  std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
}

void handle_hops_nuclear(dyn_variables& dyn_var, nHamiltonian& ham,
vector<int>& new_states, vector<int>& old_states, vector<int>& proposed_states, dyn_control_params& prms){
/**
  This function changes the nuclear dynamical variables after successful or frustrated hops

  new_states - the states accept_hops returned, old_states - the states before the hop, proposed_states - the states
  the proposal named: a trajectory whose proposed hop accept_hops refused for the energy (can_rescale_for_hop) is a
  frustrated hop here and gets the momentum reversal of the x01 / x11 options, a trajectory a stochastic acceptance
  refused (its hop payable) keeps its momentum, and an accepted hop that cannot pay is an error - accept_hops tests
  the same payability before any collapse of the amplitudes

  options:
  0 - don't rescale

  100 - based on adiabatic energy, don't reverse on frustrated hops
  101 - based on adiabatic energy, reverse on frustrated hops
  110 - based on diabatic energy, don't reverse on frustrated hops
  111 - based on diabatic energy, reverse on frustrated hops

  200 - along derivative coupling vectors, don't reverse on frustrated hops
  201 - along derivative coupling vectors, reverse on frustrated hops
  210 - along difference of state-specific forces, don't reverse on frustrated hops
  211 - along difference of state-specific forces, reverse on frustrated hops

*/

  MATRIX q(*dyn_var.q);
  MATRIX p(*dyn_var.p);
  MATRIX invM(*dyn_var.iM);
  CMATRIX C(*dyn_var.ampl_adi);

  vector<int>& which_dofs = prms.quantum_dofs;
  int ndof = q.n_rows;
  int n_active_dof = which_dofs.size();
  int ntraj = q.n_cols;
  int nst = C.n_rows;
  int traj, idof, dof;
  int isNBRA = prms.isNBRA;
  int res = prms.momenta_rescaling_algo;

  MATRIX p_tr(ndof, 1);
  CMATRIX hvib(nst, nst);
  CMATRIX nac(nst, nst);
  CMATRIX df(nst, nst);



  if(prms.momenta_rescaling_algo==0){  // Don't rescale at all
    ; ;
  }// algo = 0

  else if(prms.momenta_rescaling_algo==100 || prms.momenta_rescaling_algo==101 ||
          prms.momenta_rescaling_algo==110 || prms.momenta_rescaling_algo==111){
    // rescale momenta uniformly, based on adiabatic (100 / 101) or Lowdin diabatic (110 / 111) energies

    for(traj=0; traj<ntraj; traj++){

      int old_st = old_states[traj];
      int target = hop_target(dyn_var, ham, traj, old_st, new_states[traj], proposed_states[traj], prms);
      if(target == old_st){ continue; }
      int accepted = (new_states[traj] != old_st);

      p_tr = p.col(traj);
      double T_i = compute_kinetic_energy(p_tr, invM, which_dofs); // initial kinetic energy
      double E_i, E_f;
      if(rescaling_reads_adiabatic(res)){
        // the NBRA fast path keeps one Hamiltonian, the child of trajectory 0
        hvib = ham.child(isNBRA==1 ? 0 : traj)->get_ham_adi();
        //hvib = projectors[traj].H() * hvib * projectors[traj];
        E_i = hvib.get(old_st, old_st).real();  // initial potential energy
        E_f = hvib.get(target, target).real();  // final potential energy
      }
      else{
        E_i = ham.child(traj)->get_ham_dia_lowdin().get(old_st, old_st).real();  // initial potential energy
        E_f = ham.child(traj)->get_ham_dia_lowdin().get(target, target).real();  // final potential energy
      }
      double T_f = T_i + E_i - E_f;             // predicted final kinetic energy

      double scl_fac = 1.0;
      if(accepted){
        if(!((T_i > 0.0 && T_f >= 0.0) || E_i == E_f)){ unpayable_accepted_hop(traj, old_st, target, prms); }
        if(T_i > 0.0){ scl_fac = std::sqrt(T_f/T_i); }
      }
      else if(res==101 || res==111){ scl_fac = -1.0; }  // the frustrated hop reverses the quantum momenta

      //for(dof = 0; dof < ndof; dof++){   p.scale(dof, traj, scl_fac);  }
      for(idof = 0; idof < n_active_dof; idof++){
        dof = which_dofs[idof];
        p.scale(dof, traj, scl_fac);
      }
    }// for traj
  }// algo = 100 || 101 || 110 || 111


  else if(prms.momenta_rescaling_algo==200 || prms.momenta_rescaling_algo==201){  // rescale momenta along the derivative coupling vector

    MATRIX dNAC(ndof, 1);
    MATRIX F_old(ndof, 1);
    MATRIX F_new(ndof, 1);
    MATRIX vel(ndof, 1);
    int do_reverse = (res==201) ? 1 : 0;

    for(traj=0; traj<ntraj; traj++){

      int old_st = old_states[traj];
      int target = hop_target(dyn_var, ham, traj, old_st, new_states[traj], proposed_states[traj], prms);
      if(target == old_st){ continue; }
      int accepted = (new_states[traj] != old_st);

      // the NBRA fast path keeps one Hamiltonian, the child of trajectory 0
      hvib = ham.child(isNBRA==1 ? 0 : traj)->get_ham_adi();

      double E_i = hvib.get(old_st, old_st).real();  // initial potential energy
      double E_f = hvib.get(target, target).real();  // final potential energy

      dNAC = 0.0;
      F_old = 0.0;
      F_new = 0.0;

      for(idof = 0; idof < n_active_dof; idof++){
        dof = which_dofs[idof];
        nac = ham.child(traj)->get_dc1_adi(dof);
        //nac = U.H() * nac * U;
        //nac = projectors[traj].H() * nac * projectors[traj];
        dNAC.set(dof, 0, nac.get(old_st, target).real() );

        df = ham.child(traj)->get_d1ham_adi(dof);
        if(std::isnan(df.get(old_st, old_st).real()) || std::isnan(df.get(target, target).real())){
          force_nan_error("handle_hops_nuclear()", old_st, target, dof);
        }
        F_old.set(dof, 0, -df.get(old_st, old_st).real() );
        F_new.set(dof, 0, -df.get(target, target).real() );
      }

      p_tr = p.col(traj);
      for(dof = 0; dof<ndof; dof++){ vel.set(dof, 0,   p_tr.get(dof, 0) * invM.get(dof, 0) ); }

      if(prms.use_Jasper_Truhlar_criterion==1){
        // the criterion needs the PROPOSED state's force - a driver that
        // fills only the ground and active diagonals leaves this row zero
        // and the products below silently disable the reversal
        int f_new_zero = 1;
        for(idof = 0; idof < n_active_dof; idof++){
          if(F_new.get(which_dofs[idof], 0) != 0.0){ f_new_zero = 0; break; }
        }
        static int warned_zero_f_new = 0;
        if(f_new_zero==1 && warned_zero_f_new==0){
          warned_zero_f_new = 1;
          std::cout<<"WARNING in handle_hops_nuclear: the Jasper-Truhlar criterion sees an"
                   <<" all-zero proposed-state force - reversal on frustrated hops stays off."
                   <<" Supply all-state forces or set use_Jasper_Truhlar_criterion = 0\n";
        }
      }

      int do_reverse_actual = do_reverse;
      if(prms.use_Jasper_Truhlar_criterion==1){
        if( (F_old.T() * dNAC).get(0,0) * (F_new.T() * dNAC).get(0,0) < 0 ){
          if( (vel.T() * dNAC).get(0,0) * (F_new.T() * dNAC).get(0,0) < 0 ){ ;; }
          else{ do_reverse_actual *= 0; }
        }else{  do_reverse_actual *= 0; }
      }

      if(accepted && !can_rescale_along_vector(E_i, E_f, p_tr, invM, dNAC, which_dofs)){
        unpayable_accepted_hop(traj, old_st, target, prms);
      }
      // the accepted hop takes the conserving root, the frustrated one the reversal along dNAC (do_reverse_actual)
      rescale_along_vector(E_i, E_f, p_tr, invM, dNAC, do_reverse_actual, which_dofs);

      for(dof = 0; dof < ndof; dof++){   p.set(dof, traj, p_tr.get(dof, 0));     }

    }// for traj

  }// algo = 200 || 201

  else if(prms.momenta_rescaling_algo==210 || prms.momenta_rescaling_algo==211 ){  // rescale momenta along the difference in forces

    MATRIX dF(ndof, 1);
/*
    MATRIX Fi(ndof, ntraj);
    MATRIX Ff(ndof, ntraj);
    CMATRIX _ampl_i(nst, ntraj);     // CMATRIX version of "initial_states"
    CMATRIX _ampl_f(nst, ntraj);     // CMATRIX version of "proposed_states"


    tsh_indx2vec(ham, _ampl_i, old_states);
    Fi = ham.Ehrenfest_forces_adi(_ampl_i, 1).real();

    tsh_indx2vec(ham, _ampl_f, new_states);
    Ff = ham.Ehrenfest_forces_adi(_ampl_f, 1).real();

*/

    int do_reverse = (res==211) ? 1 : 0;

    for(traj=0; traj<ntraj; traj++){

      int old_st = old_states[traj];
      int target = hop_target(dyn_var, ham, traj, old_st, new_states[traj], proposed_states[traj], prms);
      if(target == old_st){ continue; }
      int accepted = (new_states[traj] != old_st);

      // the NBRA fast path keeps one Hamiltonian, the child of trajectory 0
      hvib = ham.child(isNBRA==1 ? 0 : traj)->get_ham_adi();
      //hvib = projectors[traj].H() * hvib * projectors[traj];
      double E_i = hvib.get(old_st, old_st).real();  // initial potential energy
      double E_f = hvib.get(target, target).real();  // final potential energy

      dF = 0.0;
      for(idof = 0; idof < n_active_dof; idof++){
        dof = which_dofs[idof];
        df = ham.child(traj)->get_d1ham_adi(dof);
        //df = projectors[traj].H() * df * projectors[traj];
        if(std::isnan(df.get(old_st, old_st).real()) || std::isnan(df.get(target, target).real())){
          force_nan_error("the force-difference hop test", old_st, target, dof);
        }
        dF.set(dof, 0, df.get(old_st, old_st).real() - df.get(target, target).real());
      }

      p_tr = p.col(traj);
      if(accepted && !can_rescale_along_vector(E_i, E_f, p_tr, invM, dF, which_dofs)){
        unpayable_accepted_hop(traj, old_st, target, prms);
      }
      // the accepted hop takes the conserving root, the frustrated one the reversal along dF (do_reverse)
      rescale_along_vector(E_i, E_f, p_tr, invM, dF, do_reverse, which_dofs);

      for(dof = 0; dof < ndof; dof++){   p.set(dof, traj, p_tr.get(dof, 0));     }

    }// for traj

  }// algo = 210 || 211

  else if(prms.momenta_rescaling_algo==40){  // based on tcnbra_ekin
    for(traj=0; traj<ntraj; traj++){

      int old_st = old_states[traj];
      int new_st = new_states[traj];
 
      if(old_st!=new_st){

        hvib = ham.child(traj)->get_ham_adi();

        double E_i = hvib.get(old_st, old_st).real();  // initial potential energy
        double E_f = hvib.get(new_st, new_st).real();  // final potential energy

        double ekin_new = E_i + dyn_var.tcnbra_ekin[traj] - E_f;

        dyn_var.tcnbra_ekin[traj] = ekin_new;

      }// if different states
    }// for traj
  }// algo == 40
  else{
    std::cout<<"Error in handle_hops_nuclear(): momenta_rescaling_algo = "<<prms.momenta_rescaling_algo<<" has no branch here (implemented: 0, 40, 100, 101, 110, 111, 200, 201, 210, 211)\n";
    std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
  }


  *dyn_var.q = q;
  *dyn_var.p = p;
  *dyn_var.ampl_adi = C;



}// handle_hops_nuclear

void handle_hops_nuclear(dyn_variables& dyn_var, nHamiltonian& ham,
vector<int>& new_states, vector<int>& old_states, dyn_control_params& prms){
/**
  The handler without the proposed states: the hops accept_hops refused leave no trace here, so no frustrated hop is
  reversed - the step loop hands the proposals over
*/
  handle_hops_nuclear(dyn_var, ham, new_states, old_states, new_states, prms);
}// handle_hops_nuclear



}// namespace libdyn
}// liblibra

