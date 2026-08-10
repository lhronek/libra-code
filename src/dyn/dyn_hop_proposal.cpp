/*********************************************************************************
* Copyright (C) 2019-2022 Alexey V. Akimov
*
* This file is distributed under the terms of the GNU General Public License
* as published by the Free Software Foundation, either version 3 of
* the License, or (at your option) any later version.
* See the file LICENSE in the root directory of this distribution
* or <http://www.gnu.org/licenses/>.
*
*********************************************************************************/
/**
  \file dyn_hop_proposal.cpp
  \brief The file implements the functionality to compute the hopping probabilities and do the hops
*/

#include "Surface_Hopping.h"
#include <stdexcept>
#include "Energy_and_Forces.h"
#include "electronic/libelectronic.h"
///#include "Dynamics.h"
//#include "../hamiltonian/libhamiltonian.h"
//#include "../io/libio.h"
#include "../opt/libopt.h"
#include "../math_meigen/mEigen.h"

///#include "dyn_control_params.h"
#include "dyn_hop_proposal.h"

/// liblibra namespace
namespace liblibra{


using namespace libmeigen;
using namespace libopt;

/// libdyn namespace
namespace libdyn{


namespace bp = boost::python;


MATRIX hopping_probabilities_fssh(dyn_control_params& prms, CMATRIX& Coeff, CMATRIX& Hvib){
/**
  \brief This function computes the surface hopping probabilities according to Tully's FSSH prescription. 
  The surface-hopping probabilities may be Boltzmann-corrected

  See more details in:
  (1) Tully, J. C. Molecular Dynamics with Electronic Transitions. J. Chem. Phys. 1990, 93, 1061–1071. - the original paper
  (2) Fabiano, E.; Keal, T. W.; Thiel, W. Implementation of Surface Hopping Molecular Dynamics Using Semiempirical Methods. Chem. Phys. 2008, 349, 334–347.
  Here, we generalized the formula, so it works equally well for both diabatic and adiabatic representations
  (3) Akimov, A. V. Libra: An Open-Source “Methodology Discovery” Library for Quantum and Classical Dynamics Simulations.
  J. Comput. Chem. 2016, 37, 1626–1649.

  \param[in] Coeff - [ndia x 1] or a [nadi x 1] matrix of electronic basis states amplitudes in a superposition - is dynamically-consistent
  \param[in] Hvib - [ndia x ndia] or a [nadi x nadi] vibronic Hamiltonian matrix - must be dynamically-consistent
  \param[in] dt - the interval of the time step for which we compute the hopping probability
  \param[in] use_boltz_factor A flag to select the Boltzmann scaling in lieu of hop rejection/velocity rescaling scheme
  \param[in] T nuclear temperature - used in the Boltzmann factor - only if use_boltz_factor is set to 1

  Returns: A matrix with the hopping probabilities between all pairs of states is returned

*/
  const double kb = 3.166811429e-6; // Hartree/K
  int i,j,k;
  double sum,g_ij,argg;

  double dt = prms.dt;
  double T = prms.Temperature;
  int use_boltz_factor = prms.use_boltz_factor;


  int nstates = Coeff.n_rows;
  MATRIX g(nstates,nstates);
  CMATRIX denmat(nstates, nstates);   
  
  denmat = Coeff * Coeff.H();

  // Now calculate the hopping probabilities
  for(i=0;i<nstates;i++){
    sum = 0.0;
    double a_ii = denmat.get(i,i).real(); 

    for(j=0;j<nstates;j++){

      if(i!=j){ 
        /**
          dc/dt = -(i/hbar) * Hvib * c
          (dc/dt)^+ = i/hbar * c^+ * Hvib^+

          rho = c * c^+

          Then
          drho/dt = i/hbar * (rho*Hvib - Hvib*rho)

          The diagonal element:
           
          drho_ii/dt = (i/hbar) *sum_a { rho_ia * Hvib_ai - Hvib_ia * rho_ai}

          Then:  P(i->*) = -(drho_ii / rho_ii ) * dt  - probability of leaving state i

          Then:  P(i->a) = - (dt/rho_ii) * Re[ (i/hbar) *sum_a { rho_ia * Hvib_ai - Hvib_ia * rho_ai} ] 

          = (dt/(hbar*rho_ii)) * Im[ rho_ia * Hvib_ai - Hvib_ia * rho_ai ] 

          Or:

          P(i->j) = (dt/(hbar*rho_ii)) * Im[ rho_ij * Hvib_ji - Hvib_ij * rho_ji ] 

        */

        double imHaij = ( denmat.get(i,j) * Hvib.get(j,i) - Hvib.get(i,j) * denmat.get(j,i) ).imag(); 

        if(a_ii<1e-8){ g_ij = 0.0; }  // avoid division by zero
        else{
          g_ij = dt*imHaij/a_ii;  // This is a general case -

          if(use_boltz_factor){

            if(Hvib.get(j,j).real() > Hvib.get(i,i).real()){
              argg = -(Hvib.get(j,j).real() - Hvib.get(i,i).real())/(kb*T);        
              if(argg<500.0){ g_ij = g_ij * std::exp(argg); }
            }

          }// if use_boltz_factor


          if(g_ij<0.0){  g_ij = 0.0; }

        }// else

        g.set(i,j,g_ij);
        sum = sum + g_ij;
      }
      else{ g.set(i,j,0.0); }

    }// for j

    g.set(i,i,1.0 - sum);

  }// for i

  return g;

}// fssh



vector<double> hopping_probabilities_fssh(dyn_control_params& prms, CMATRIX& denmat, CMATRIX& Hvib, int act_state_indx){
/**
  \brief This function computes the surface hopping probabilities according to Tully's FSSH prescription. 
  The surface-hopping probabilities may be Boltzmann-corrected

  See more details in:
  (1) Tully, J. C. Molecular Dynamics with Electronic Transitions. J. Chem. Phys. 1990, 93, 1061–1071. - the original paper
  (2) Fabiano, E.; Keal, T. W.; Thiel, W. Implementation of Surface Hopping Molecular Dynamics Using Semiempirical Methods. Chem. Phys. 2008, 349, 334–347.
  Here, we generalized the formula, so it works equally well for both diabatic and adiabatic representations
  (3) Akimov, A. V. Libra: An Open-Source “Methodology Discovery” Library for Quantum and Classical Dynamics Simulations.
  J. Comput. Chem. 2016, 37, 1626–1649.

  \param[in] key parameters needed for this type of calculations
    - dt - integration timestep [a.u.]
    - Temperature - temperature [ K ]
    - use_boltz_factor - whether to scale the computed probabilities by a Boltzmann factor
  \param[in] denmat - [nstates x nstates] - density matrix 
  \param[in] Hvib - [nstates x nstates] - vibronic Hamiltonian matrix
  \param[in] act_state_indx - index of the initial state 

  Returns: A nstates-vector of hopping probabilities to all states from the current active state

*/
  const double kb = 3.166811429e-6; // Hartree/K
  int i,j,k;
  double sum,g_ij,argg;

  double dt = prms.dt;
  double T = prms.Temperature;
  int use_boltz_factor = prms.use_boltz_factor;

  int nstates = denmat.n_rows;
  vector<double> g(nstates, 0.0);


//  cout<<"In hopping prob fssh\n";
  // Now calculate the hopping probabilities
  i = act_state_indx;
   
  sum = 0.0;
  double a_ii = denmat.get(i,i).real(); 
//  cout<<"a_ii = "<<a_ii<<endl;

  for(j=0;j<nstates;j++){
 
    if(i!=j){ 
      /**
        dc/dt = -(i/hbar) * Hvib * c
        (dc/dt)^+ = i/hbar * c^+ * Hvib^+
 
        rho = c * c^+
 
        Then
        drho/dt = i/hbar * (rho*Hvib - Hvib*rho)
 
        The diagonal element:
         
        drho_ii/dt = (i/hbar) *sum_a { rho_ia * Hvib_ai - Hvib_ia * rho_ai}
 
        Then:  P(i->*) = -(drho_ii / rho_ii ) * dt  - probability of leaving state i
 
        Then:  P(i->a) = - (dt/rho_ii) * Re[ (i/hbar) *sum_a { rho_ia * Hvib_ai - Hvib_ia * rho_ai} ] 
 
        = (dt/(hbar*rho_ii)) * Im[ rho_ia * Hvib_ai - Hvib_ia * rho_ai ] 
 
        Or:
 
        P(i->j) = (dt/(hbar*rho_ii)) * Im[ rho_ij * Hvib_ji - Hvib_ij * rho_ji ] 
 
      */
 
      double imHaij = ( denmat.get(i,j) * Hvib.get(j,i) - Hvib.get(i,j) * denmat.get(j,i) ).imag(); 
//      cout<<"rho_ij = "<<denmat.get(i,j)<<"  Hvib_ij = "<<Hvib.get(i,j)<<endl;
 
      if(a_ii<1e-8){ g_ij = 0.0; }  // avoid division by zero
      else{
        g_ij = dt*imHaij/a_ii;  // This is a general case -
 
        if(use_boltz_factor){
 
          if(Hvib.get(j,j).real() > Hvib.get(i,i).real()){
            argg = -(Hvib.get(j,j).real() - Hvib.get(i,i).real())/(kb*T);        
            if(argg<500.0){ g_ij = g_ij * std::exp(argg); }
          }
 
        }// if use_boltz_factor
 
 
        if(g_ij<0.0){  g_ij = 0.0; }
        if(g_ij>1.0){  g_ij = 1.0; }
 
      }// else
 
      g[j] = g_ij;
      sum = sum + g_ij;
    }
    else{ g[j] = 0.0; }
 
  }// for j

  if(sum>1.0){ sum = 1.0; }
 
  g[i] = 1.0 - sum;
 
  return g;
 
}// fssh





MATRIX hopping_probabilities_gfsh(dyn_control_params& prms, CMATRIX& Coeff, CMATRIX& Hvib){

/**
  \brief Compute the GFSH surface hopping probabilities for a single trajectory

  \param[in] Coeff - [ndia x 1] or a [nadi x 1] matrix of electronic basis states amplitudes in a superposition
  \param[in] Hvib - [ndia x ndia] or a [nadi x nadi] vibronic Hamiltonian matrix
  \param[in] dt - the interval of the time step for which we compute the hopping probability
  \param[in] use_boltz_factor A flag to select the Boltzmann scaling in lieu of hop rejection/velocity rescaling scheme
  \param[in] T nuclear temperature - used in the Boltzmann factor - only if use_boltz_factor is set to 1

  Returns: A matrix with the hopping probabilities between all pairs of states is returned

  Abbreviation: GFSH - global flux surface hopping
  References: 
  (1) Wang, L.; Trivedi, D.; Prezhdo, O. V. Global Flux Surface Hopping Approach for Mixed Quantum-Classical Dynamics. J. Chem. Theory Comput. 2014, 10, 3598–3605.
  (2) Akimov, A. V. Libra: An Open-Source “Methodology Discovery” Library for Quantum and Classical Dynamics Simulations.
  J. Comput. Chem. 2016, 37, 1626–1649.


*/

  const double kb = 3.166811429e-6; // Hartree/K
  int i,j,k;
  double sum,g_ij,argg;

  double dt = prms.dt;
  double T = prms.Temperature;
  int use_boltz_factor = prms.use_boltz_factor;


  int nstates = Coeff.n_rows;
  MATRIX g(nstates,nstates);
  CMATRIX denmat(nstates, nstates);   

  //denmat = (Coeff * Coeff.H() ).conj();
  denmat = Coeff * Coeff.H();

  //CMATRIX* denmat_dot; denmat_dot = new CMATRIX(nstates, nstates);   
  CMATRIX denmat_dot(nstates, nstates);
  denmat_dot = ( denmat *  Hvib.H() - Hvib * denmat) * complex<double>(0.0, 1.0);



  // compute a_kk and a_dot_kk
  vector<double> a(nstates,0.0);
  vector<double> a_dot(nstates,0.0);
  double norm = 0.0; // normalization factor

  for(i=0;i<nstates;i++){
    a[i] = denmat.get(i,i).real();
    a_dot[i] = denmat_dot.get(i,i).real();

    if(a_dot[i]<0.0){ norm += a_dot[i]; } // total rate of population decrease in all decaying states

  }// for i


  // Now calculate the hopping probabilities
  for(i=0;i<nstates;i++){       
    double sumg = 0.0;

    for(j=0;j<nstates;j++){

      if(j!=i){  // off-diagonal = probabilities to hop to other states

        if(a[i]<1e-12){  g.set(i,j,0.0); }  // since the initial population is almost zero, so no need for hops
        else{

          if( fabs(norm) > 1e-12 ){
            g.set(i,j,  dt*(a_dot[j]/a[i]) * a_dot[i] / norm);  
          }
 
          if(g.get(i,j)<0.0){  // since norm is negative, than this condition means that a_dot[i] and a_dot[j] have same signs
                               // which is bad - so no transitions are assigned
            g.set(i,j,0.0);
          }
          else{  // here we have opposite signs of a_dot[i] and a_dot[j], but this is not enough yet
            if(a_dot[i]<0.0 & a_dot[j]>0.0){ ;; } // this is out transition probability, but it is already computed
            else{  g.set(i,j,0.0); } // wrong transition
          }

        }// a[i]>1e-12

          if(use_boltz_factor){

            if(Hvib.get(j,j).real() > Hvib.get(i,i).real()){
              argg = -(Hvib.get(j,j).real() - Hvib.get(i,i).real())/(kb*T);        
              if(argg<500.0){ g_ij = g_ij * std::exp(argg); }
            }

          }// if use_boltz_factor


        sumg += g.get(i,j);

      }
    }// for j

    g.set(i,i, 1.0 - sumg);  // probability to stay in state i

    if(g.get(i,i)<0.0){  g.set(i,i, 0.0); }

  }// for i

//  delete denmat;
//  delete denmat_dot;
  return g;

}// gfsh







vector<double> hopping_probabilities_gfsh(dyn_control_params& prms, CMATRIX& denmat, CMATRIX& Hvib, int act_state_indx){
/**
  \brief Compute the GFSH surface hopping probabilities for a single trajectory

  Abbreviation: GFSH - global flux surface hopping
  References: 
  (1) Wang, L.; Trivedi, D.; Prezhdo, O. V. Global Flux Surface Hopping Approach for Mixed Quantum-Classical Dynamics. J. Chem. Theory Comput. 2014, 10, 3598–3605.
  (2) Akimov, A. V. Libra: An Open-Source “Methodology Discovery” Library for Quantum and Classical Dynamics Simulations.
  J. Comput. Chem. 2016, 37, 1626–1649.

  \param[in] key parameters needed for this type of calculations
    - dt - integration timestep [a.u.]
    - Temperature - temperature [ K ]
    - use_boltz_factor - whether to scale the computed probabilities by a Boltzmann factor
  \param[in] denmat - [nstates x nstates] - density matrix 
  \param[in] Hvib - [nstates x nstates] - vibronic Hamiltonian matrix
  \param[in] act_state_indx - index of the initial state 

  Returns: A nstates-vector of hopping probabilities to all states from the current active state

*/

  const double kb = 3.166811429e-6; // Hartree/K
  int i,j,k;
  double sum,g_ij,argg;

  double dt = prms.dt;
  double T = prms.Temperature;
  int use_boltz_factor = prms.use_boltz_factor;


  int nstates = denmat.n_rows;
  vector<double> g(nstates, 0.0);

  CMATRIX denmat_dot(nstates, nstates);
  denmat_dot = ( denmat *  Hvib.H() - Hvib * denmat) * complex<double>(0.0, 1.0);


  // compute a_kk and a_dot_kk
  vector<double> a(nstates,0.0);
  vector<double> a_dot(nstates,0.0);
  double norm = 0.0; // normalization factor

  for(i=0;i<nstates;i++){
    a[i] = denmat.get(i,i).real();
    a_dot[i] = denmat_dot.get(i,i).real();

    if(a_dot[i]<0.0){ norm += a_dot[i]; } // total rate of population decrease in all decaying states

  }// for i


  // Now calculate the hopping probabilities
  i = act_state_indx;
  double sumg = 0.0;

  for(j=0;j<nstates;j++){

    if(j!=i){  // off-diagonal = probabilities to hop to other states

      if(a[i]<1e-12){  g[j] = 0.0; }  // since the initial population is almost zero, so no need for hops
      else{

        if( fabs(norm) > 1e-12 ){ g[j] = dt*(a_dot[j]/a[i]) * a_dot[i] / norm;  }

        // since norm is negative, than this condition means that a_dot[i] and a_dot[j] have same signs
        // which is bad - so no transitions are assigned
        if(g[j]<0.0){  g[j] = 0.0;   }

        // here we have opposite signs of a_dot[i] and a_dot[j], but this is not enough yet
        else{  
          if(a_dot[i]<0.0 && a_dot[j]>0.0){ ;; } // this is out transition probability, but it is already computed
          else{  g[j] = 0.0; } // wrong transition
        }
      }// a[i]>1e-12

        if(use_boltz_factor){
          if(Hvib.get(j,j).real() > Hvib.get(i,i).real()){
            argg = -(Hvib.get(j,j).real() - Hvib.get(i,i).real())/(kb*T);        
            if(argg<500.0){ g_ij = g_ij * std::exp(argg); }
          }
        }// if use_boltz_factor

      sumg += g[j];
    }
  }// for j

  g[i] = 1.0 - sumg;  // probability to stay in state i

  if(g[i]<0.0){  g[i] = 0.0; }

  return g;

}// gfsh





vector<double> hopping_probabilities_gfsh_orig(dyn_control_params& prms, CMATRIX& denmat, CMATRIX& denmat_old, int act_state_indx){
/**
  \brief Compute the GFSH surface hopping probabilities for a single trajectory

  Abbreviation: GFSH - global flux surface hopping
  References: 
  (1) Wang, L.; Trivedi, D.; Prezhdo, O. V. Global Flux Surface Hopping Approach for Mixed Quantum-Classical Dynamics. J. Chem. Theory Comput. 2014, 10, 3598–3605.
  (2) Akimov, A. V. Libra: An Open-Source “Methodology Discovery” Library for Quantum and Classical Dynamics Simulations.
  J. Comput. Chem. 2016, 37, 1626–1649.

  \param[in] key parameters needed for this type of calculations
    - dt - integration timestep [a.u.]
    - Temperature - temperature [ K ]
    - use_boltz_factor - whether to scale the computed probabilities by a Boltzmann factor
  \param[in] denmat - [nstates x nstates] - current density matrix
  \param[in] denmat_old - [nstates x nstates] - previous density matrix
  \param[in] act_state_indx - index of the initial state

  Returns: A nstates-vector of hopping probabilities to all states from the current active state


*/

  const double kb = 3.166811429e-6; // Hartree/K
  int i,j,k;
  double sum,g_ij,argg;

  double dt = prms.dt;
  double T = prms.Temperature;
  int use_boltz_factor = prms.use_boltz_factor;


  int nstates = denmat.n_rows;
  vector<double> g(nstates, 0.0);

  CMATRIX denmat_dot(nstates, nstates);
  denmat_dot = denmat - denmat_old;


  // compute a_kk and a_dot_kk
  vector<double> a(nstates,0.0);
  vector<double> a_dot(nstates,0.0);
  double norm = 0.0; // normalization factor

  for(i=0;i<nstates;i++){
    a[i] = denmat.get(i,i).real();
    a_dot[i] = denmat_dot.get(i,i).real();

    if(a_dot[i]<0.0){ norm += a_dot[i]; } // total rate of population decrease in all decaying states

  }// for i


  // Now calculate the hopping probabilities
  i = act_state_indx;
  double sumg = 0.0;

  for(j=0;j<nstates;j++){

    if(j!=i){  // off-diagonal = probabilities to hop to other states

      if(a[i]<1e-12){  g[j] = 0.0; }  // since the initial population is almost zero, so no need for hops
      else{

        if( fabs(norm) > 1e-12 ){ g[j] = (a_dot[j]/a[i]) * a_dot[i] / norm;  }

        // since norm is negative, than this condition means that a_dot[i] and a_dot[j] have same signs
        // which is bad - so no transitions are assigned
        if(g[j]<0.0){  g[j] = 0.0;   }

        // here we have opposite signs of a_dot[i] and a_dot[j], but this is not enough yet
        else{  
          if(a_dot[i]<0.0 && a_dot[j]>0.0){ ;; } // this is out transition probability, but it is already computed
          else{  g[j] = 0.0; } // wrong transition
        }
      }// a[i]>1e-12

      sumg += g[j];
    }
  }// for j

  g[i] = 1.0 - sumg;  // probability to stay in state i

  if(g[i]<0.0){  g[i] = 0.0; }

  return g;

}// gfsh - original version 





vector<double> hopping_probabilities_fssh2(dyn_control_params& prms, CMATRIX& denmat, CMATRIX& denmat_old, int act_state_indx){
/**
  \brief This function computes the surface hopping probabilities according to Leonardo Araujo's hopping probability 
  reformulation of FSSH recipe - this one does not need nonadiabatic couplings!

  See more details in: TBD


  P_{m->n}(t,t+dt) = min{ P_{m,out}(t,t+dt), (rho_{nn}(t+dt) - rho_{nn}(t))/rho_{mm}(t) }, 

  where 

  P_{m, out}(t, t+dt) = sum_{n=0, n\neqm}^{nstates} (  (rho_{nn}(t+dt) - rho{nn}(t)) /rho_{mm}(t) )

  \param[in] key parameters needed for this type of calculations
    - dt - integration timestep [a.u.]
    - Temperature - temperature [ K ]
    - use_boltz_factor - whether to scale the computed probabilities by a Boltzmann factor
  \param[in] denmat - [nstates x nstates] - current density matrix
  \param[in] denmat_old - [nstates x nstates] - previous density matrix
  \param[in] act_state_indx - index of the initial state

  Returns: A nstates-vector of hopping probabilities to all states from the current active state

*/

  const double kb = 3.166811429e-6; // Hartree/K
  int i,j,k;
  double sum,g_ij,argg;

  double dt = prms.dt;
  double T = prms.Temperature;
  int use_boltz_factor = prms.use_boltz_factor;

  int version = prms.fssh2_revision;

  int nstates = denmat.n_rows;
  vector<double> g(nstates, 0.0);


  // Now calculate the hopping probabilities
  i = act_state_indx;

  double a_ii = denmat.get(i,i).real();
  double a_ii_old = denmat_old.get(i,i).real();

  double P_out_ii = 0.0;
  if(a_ii_old > 0.0 ){  P_out_ii = - (a_ii - a_ii_old)/a_ii_old; }
  if(P_out_ii < 0.0){ P_out_ii = 0.0; } // same as below - assume this is implied

  sum = 0.0;  
  for(j=0;j<nstates;j++){

    if(i!=j){
      double a_jj_old = denmat_old.get(j,j).real();
      double a_jj     = denmat.get(j,j).real();

      if(a_ii_old> 0.0){ // avoid division by zero
        g_ij = (a_jj - a_jj_old)/a_ii_old;  // This is a general case 

        if(g_ij<0.0){  g_ij = 0.0; } // this is not present in the original paper,
                                     // but the populations can not be negative, so
                                     // I take that this condition was implied there
      } else{ g_ij = 0.0; }
    
      if(version==0){    g_ij = MIN(g_ij, P_out_ii);  } // original version
      else if(version==1){  ;; }                        // corrected one - don't do anything here

      g[j] = g_ij;
      sum += g_ij;
    }
    else{ g[j] = 0.0; }

  }// for j

  g[i] = 1.0 - sum;

  if(version==1){  // Revision as discussed

    if(sum>0.0){
      g[i] = P_out_ii;
      for(j=0;j<nstates;j++){  if(j!=i){  g[j] *= (1.0 - P_out_ii)/sum; }  }
    }else{
      g[i] = 1.0; 
      for(j=0;j<nstates;j++){  if(j!=i){  g[j] = 0.0; }  }
    }

  }// revised version

  return g;

}// fssh2



MATRIX hopping_probabilities_mssh(dyn_control_params& prms, CMATRIX& Coeff, CMATRIX& Hvib){
/**
   \brief Compute the MSSH surface hopping probabilities scaled by Boltzmann factor
   This is the version taking the minimal amount of input information
   \param[in] Coeff The amplitudes of different basis states in the coherent superposition. This matrix is assumed to be
   a column-vector, so of the size N x 1, where N is the number of basis excited states
   \param[in] Hvib - [ndia x ndia] or a [nadi x nadi] vibronic Hamiltonian matrix
   \param[in] use_boltz_factor A flag to select the Boltzmann scaling in lieu of hop rejection/velocity rescaling scheme
   \param[in] T nuclear temperature - used in the Boltzmann factor - only if use_boltz_factor is set to 1
   The function returns the matrix g with hopping probabilities
   Abbreviation: MSSH - Markov state surface hopping
   References:
   (1) Akimov, A. V.; Trivedi, D.; Wang, L.; Prezhdo, O. V. Analysis of the Trajectory Surface Hopping Method from the Markov State Model Perspective. J. Phys. Soc. Jpn. 2015, 84, 094002.
*/

  const double kb = 3.166811429e-6; // Hartree/K
  int i,j,k;
  double sum,g_ij,argg;

  double dt = prms.dt;
  double T = prms.Temperature;
  int use_boltz_factor = prms.use_boltz_factor;

  int nstates = Coeff.n_rows;
  MATRIX g(nstates,nstates);
  CMATRIX denmat(nstates, nstates);   


  double norm; norm = (Coeff.H() * Coeff).get(0,0).real();  // <- this is the norm <PSI|PSI>

  // Calculate the hopping probabilities
  for(int j=0;j<nstates;j++){
    double gjj = (std::conj(Coeff.get(j)) * Coeff.get(j)).real()/norm; // c_j^* * c_j

    for(int i=0;i<nstates;i++){ g.set(i,j,gjj);}
  }

  if(use_boltz_factor){
    if(Hvib.get(j,j).real() > Hvib.get(i,i).real()){
      argg = -(Hvib.get(j,j).real() - Hvib.get(i,i).real())/(kb*T);        
      if(argg<500.0){ g_ij = g_ij * std::exp(argg); }
    }
  }// if use_boltz_factor


  return g;

}


vector<double> hopping_probabilities_mssh(dyn_control_params& prms, CMATRIX& denmat, CMATRIX& Hvib, int act_state_indx){
/**
   \brief Compute the MSSH surface hopping probabilities scaled by Boltzmann factor
   This is the version taking the minimal amount of input information

   The function returns the matrix g with hopping probabilities
   Abbreviation: MSSH - Markov state surface hopping

   References:
   (1) Akimov, A. V.; Trivedi, D.; Wang, L.; Prezhdo, O. V. Analysis of the Trajectory Surface Hopping Method from the Markov State Model Perspective. J. Phys. Soc. Jpn. 2015, 84, 094002.

   \param[in] key parameters needed for this type of calculations
     - dt - integration timestep [a.u.]
     - Temperature - temperature [ K ]
     - use_boltz_factor - whether to scale the computed probabilities by a Boltzmann factor
   \param[in] denmat - [nstates x nstates] - density matrix 
   \param[in] Hvib - [nstates x nstates] - vibronic Hamiltonian matrix
   \param[in] act_state_indx - index of the initial state 

   Returns: A nstates-vector of hopping probabilities to all states from the current active state

*/

  const double kb = 3.166811429e-6; // Hartree/K
  int i,j,k;
  double sum,g_ij,argg;

  double dt = prms.dt;
  double T = prms.Temperature;
  int use_boltz_factor = prms.use_boltz_factor;

  int nstates = denmat.n_rows;
  vector<double> g(nstates, 0.0);

  double norm; norm = denmat.tr().real();  // <- this is the norm <PSI|PSI>

  i = act_state_indx; 

  // Calculate the hopping probabilities
  double summ = 0.0;
  for(int j=0;j<nstates;j++){

    if(j!=i){    
      double gjj = denmat.get(j,j).real()/norm; // |c_j|^2
    
      if(use_boltz_factor){
        if(Hvib.get(j,j).real() > Hvib.get(i,i).real()){
          argg = -(Hvib.get(j,j).real() - Hvib.get(i,i).real())/(kb*T);        
          if(argg<500.0){ gjj = gjj * std::exp(argg); }
        }
      }// if use_boltz_factor
          
      g[j] = gjj;
      summ += gjj;
    }
    else{  g[j] = 0.0; }
  }// for j

  g[i] = 1.0 - summ;

  return g;

}


//vector<double> hopping_probabilities_lz(nHamiltonian* ham, nHamiltonian* ham_prev, int act_state_indx, int rep, MATRIX& p, const MATRIX& invM, double dt, vector<double>& gap_mins){
vector<double> hopping_probabilities_lz(nHamiltonian* ham, nHamiltonian* ham_prev, int act_state_indx, int rep, MATRIX& p, const MATRIX& invM, double dt, vector<double>& gap_mins){
/**
  \brief This function computes the surface hopping probabilities according to Landau-Zener formula.

  See more details in:
  (1) Tully, J. C. Molecular Dynamics with Electronic Transitions. J. Chem. Phys. 1990, 93, 1061<96>1071. - the original paper
  (2) Belyaev, A. K.; Lebedev, O. V. Nonadiabatic nuclear dynamics of atomic collisions based on branching classical trajectories
  Phys. Rev. A 2011, 84, 014701

  \param[in] ham Is the nHamiltonian object that organizes energy-related calculations
  \param[in] ham_prev Is the nHamiltonian object of previous step or geometry - needed to locate the diabatic crossing
  \param[in] act_state_indx - index of the current adiabatic state
  \param[in] rep index selecting the type of representation to be used: 0 - diabatic, 1 - adiabatic based on the min of diabatic 
  gap, 2 - adiabatic, based on the switch of the NAC sign
  3 - diabatic, generalized-linear LZ: Eq. (3) of Belyaev, Lasser, Trigila (JCP 140, 224108 (2014))
  evaluated at the gap minimum of the linearized passage, detected inside the last step
  \param[in] p [ndof x 1] or [ndof x ntraj] matrix of nuclear momenta
  \param[in] invM [ndof x 1] matrix of inverse nuclear masses
  \param[in] dt the integration timestep (a.u.) - the finite-difference gap velocity
  when the model supplies no diabatic derivatives

  Returns:
      A nstates-vector of hopping probabilities to all states from the current active state

*/
  int i,j,k;
  int nstates;
  int ndof = ham->nnucl;

  // Determine the dimensions
  if(rep==0 || rep==3){ nstates = ham->ndia;  }
  else if(rep==1 || rep==2){  nstates = ham->nadi;  }
  else{
    cout<<"Error in hopping_probabilities_lz(): rep can be 0, 1, 2 or 3; the Belyaev-Lebedev \
gap-minimum probabilities (rep_lz = 4) come from hopping_probabilities_bllz \n";
    throw std::runtime_error("libra C++ error - see the message printed above");
  }

//  vector<double> g(nstates, 0.0);
  vector<double> g(nstates, 0.0);

  i = act_state_indx;

  /** Diabatic representation:

   p_{i->j} = exp( - 2 * pi * H_ij^2 / (hbar*v*(|H'_ii - H'_jj| )) )
 
  */
  
  // in-out channel: incoming entries may carry the dispatch's three-point
  // gap-parabola minima (rep 2); the vector is rewritten with the
  // at-crossing gaps of the channels this call detects (-1 elsewhere)
  vector<double> gap_in(gap_mins);
  if((int)gap_in.size() != nstates){ gap_in.assign(nstates, -1.0); }
  gap_mins.assign(nstates, -1.0);

  if(rep==0){

    // Diabatic Ham matrices
    MATRIX ham_dia(nstates,nstates);
    MATRIX ham_dia_prev(nstates, nstates);
    ham_dia = ham->get_ham_dia().real();
    ham_dia_prev = ham_prev->get_ham_dia().real();
    if(ham_dia.max_elt()==0.0 && ham_dia_prev.max_elt()==0.0){
      cout<<"Error in hopping_probabilities_lz(): the diabatic Hamiltonian is all-zero at two \
consecutive steps - the diabatic crossing detection is inoperative. Adiabatic-native models \
must supply obj.ham_dia, or use rep_lz = 2 (adiabatic-only detection) \n";
      throw std::runtime_error("libra C++ error - see the message printed above");
    }

    // Get denominator
    MATRIX dham(nstates,nstates); // H'_ii * v
    MATRIX dHv(nstates,nstates); // |H'_ii - H'_jj| * v

    if(ham->d1ham_dia_mem_status[0] != 0){
      // |d(H_ii - H_jj)/dt| by the chain rule: the DOF contributions sum
      // WITH signs and one magnitude is taken of the result - per-DOF
      // magnitudes would inflate the gap velocity whenever DOF motions
      // oppose each other
      for(k=0;k<ndof;k++){
        dham = ham->get_d1ham_dia(k).real() * p.get(k,0) * invM.get(k,0);

        for(j=0;j<nstates;j++){
          double dmod = dham.get(i,i) - dham.get(j,j);
          dHv.add(i,j, dmod);
          dHv.add(j,i, dmod);
        }// for j
      }// for k
      for(j=0;j<nstates;j++){
        double dmod = fabs(dHv.get(i,j));
        dHv.set(i,j, dmod);
        dHv.set(j,i, dmod);
      }// for j
    }
    else{
      // no diabatic derivatives (adiabatic-native models supplying ham_dia):
      // by the chain rule the DOF sum is d(H_ii - H_jj)/dt, available as a
      // two-point finite difference on the same steps the crossing detection
      // below compares
      for(j=0;j<nstates;j++){
        double dgap = fabs( (ham_dia.get(i,i) - ham_dia.get(j,j))
                          - (ham_dia_prev.get(i,i) - ham_dia_prev.get(j,j)) ) / dt;
        dHv.set(i,j, dgap);
        dHv.set(j,i, dgap);
      }// for j
    }


    //========== Now calculate the hopping probabilities ===========
    
    // Off-diagonal elements - only, if we meet the gap minimum
    for(j=0; j<nstates; j++){

      if(j!=i){
        double dh = ham_dia.get(i,i) - ham_dia.get(j,j);
        double dh_prev = ham_dia_prev.get(i,i) - ham_dia_prev.get(j,j);

        if(dh * dh_prev < 0.0){

          double h_ij = ham_dia.get(i,j);
          double g_ij = exp(-2.0*M_PI*h_ij*h_ij / dHv.get(i,j) );
          // the two-state adiabatic gap at the diabatic crossing is
          // exactly 2|H_ij| - interpolate the coupling to the crossing
          double frac = dh_prev/(dh_prev - dh);
          gap_mins[j] = 2.0*fabs( ham_dia_prev.get(i,j)
                        + frac*(h_ij - ham_dia_prev.get(i,j)) );

          g[j] = g_ij;
          //g[j] = g_ij;

        }// only if minimum is located 
        else{ g[j] = 0.0; }

      }// j!=i
    }// for j
  }// rep == 0 diabatic


  // Adiabatic representation
  else if(rep==1 || rep==2){

    // Update adiabatic NACs - assume those are updated 
    //ham->compute_nac_adi(p, invM);

    // Now calculate the hopping probabilities
    MATRIX ham_dia(nstates,nstates);
    MATRIX ham_adi(nstates,nstates);
    MATRIX nac_adi(nstates,nstates);
    MATRIX ham_adi_prev(nstates,nstates);
    MATRIX ham_dia_prev(nstates,nstates);
    MATRIX nac_adi_prev(nstates,nstates);

    ham_dia = ham->get_ham_dia().real();
    ham_adi = ham->get_ham_adi().real();
    nac_adi = ham->get_nac_adi().real();
    ham_adi_prev = ham_prev->get_ham_adi().real();
    ham_dia_prev = ham_prev->get_ham_dia().real();
    nac_adi_prev = ham_prev->get_nac_adi().real();
    if(rep==1 && ham_dia.max_elt()==0.0 && ham_dia_prev.max_elt()==0.0){
      cout<<"Error in hopping_probabilities_lz(): rep_lz = 1 detects crossings in the diabatic \
Hamiltonian, which is all-zero at two consecutive steps. Adiabatic-native models must supply \
obj.ham_dia, or use rep_lz = 2 (adiabatic-only detection) \n";
      throw std::runtime_error("libra C++ error - see the message printed above");
    }

    // Off-diagonal elements
    for(j=0;j<nstates;j++){
      if(j!=i){

        int nonzero = 0;
        double Z_star = -1.0;   // the at-crossing adiabatic gap, when known
        if(rep==1){
          double dh = ham_dia.get(i,i) - ham_dia.get(j,j);
          double dh_prev = ham_dia_prev.get(i,i) - ham_dia_prev.get(j,j);
          if( dh * dh_prev < 0.0){
            nonzero = 1;
            // 2|H_ij| interpolated to the diabatic crossing
            double frac = dh_prev/(dh_prev - dh);
            Z_star = 2.0*fabs( ham_dia_prev.get(i,j)
                     + frac*(ham_dia.get(i,j) - ham_dia_prev.get(i,j)) );
          }
        }
        else if(rep==2){
          double nac = nac_adi.get(i,j);
          double nac_prev = nac_adi_prev.get(i,j);
          if( nac * nac_prev < 0.0){
            nonzero = 1;
            // the dispatch pre-fills the three-point gap-parabola minimum
            // when the energy history holds one
            if(gap_in[j] >= 0.0){ Z_star = gap_in[j]; }
          }
        }

        if(nonzero==1){  // This is the condition of intersection of diabatic surfaces, alternatively, we 
                         // could be looking to locate the adiabatic surfaces minimum
          double g_ij = 0.0;
          // the probability's gap and the stored acceptance gap both
          // evaluate at the crossing when it is known
          double Z_ij = (Z_star >= 0.0) ? Z_star
                      : fabs(ham_adi.get(i,i) - ham_adi.get(j,j));
          double nac_ij = fabs(nac_adi.get(i,j));
          if(nac_ij > 0.0){ g_ij = exp(-0.25*M_PI*Z_ij/nac_ij);     }
          gap_mins[j] = Z_star;
          g[j] = g_ij;
        }
        else{ g[j] = 0.0; }

      }// j!=i
    }// for j
  }// rep == 1 adiabatic


  /** Generalized-linear diabatic LZ:

   p_{i->j} = exp( - pi * (v1*v2' - v2*v1')^2 / (v1'^2 + v2'^2)^(3/2) ),  v = ( (H_ii - H_jj)/2, H_ij )

   BLT Eq. (3) evaluated at the gap minimum of the linearized passage - exact for a linear
   diabatic model, reduces to the rep 0 formula for a constant coupling

  */

  else if(rep==3){

    // Diabatic Ham matrices
    MATRIX ham_dia(nstates,nstates);
    MATRIX ham_dia_prev(nstates, nstates);
    ham_dia = ham->get_ham_dia().real();
    ham_dia_prev = ham_prev->get_ham_dia().real();
    if(ham_dia.max_elt()==0.0 && ham_dia_prev.max_elt()==0.0){
      cout<<"Error in hopping_probabilities_lz(): the diabatic Hamiltonian is all-zero at two \
consecutive steps - the diabatic crossing detection is inoperative. Adiabatic-native models \
must supply obj.ham_dia, or use rep_lz = 2 (adiabatic-only detection) \n";
      throw std::runtime_error("libra C++ error - see the message printed above");
    }

    // the pair-local trace-free vector is v = ( (H_ii - H_jj)/2, H_ij ); its
    // time derivative comes from the diabatic derivatives when the model
    // supplies them, from the two steps otherwise
    vector<double> v1d(nstates, 0.0);
    vector<double> v2d(nstates, 0.0);
    if(ham->d1ham_dia_mem_status[0] != 0){
      MATRIX d1(nstates,nstates);
      for(k=0;k<ndof;k++){
        d1 = ham->get_d1ham_dia(k).real();
        double vk = p.get(k,0) * invM.get(k,0);
        for(j=0;j<nstates;j++){
          v1d[j] += 0.5*(d1.get(i,i) - d1.get(j,j)) * vk;
          v2d[j] += d1.get(i,j) * vk;
        }// for j
      }// for k
    }
    else{
      for(j=0;j<nstates;j++){
        v1d[j] = 0.5*( (ham_dia.get(i,i) - ham_dia.get(j,j))
                     - (ham_dia_prev.get(i,i) - ham_dia_prev.get(j,j)) )/dt;
        v2d[j] = (ham_dia.get(i,j) - ham_dia_prev.get(i,j))/dt;
      }// for j
    }

    for(j=0; j<nstates; j++){

      if(j!=i){
        double v1c = 0.5*(ham_dia.get(i,i) - ham_dia.get(j,j));
        double v2c = ham_dia.get(i,j);
        double dv1 = v1c - 0.5*(ham_dia_prev.get(i,i) - ham_dia_prev.get(j,j));
        double dv2 = v2c - ham_dia_prev.get(i,j);

        // the linearized passage attains its gap minimum inside this step
        // iff the projection of v on the step direction changes sign
        double s_prev = (v1c - dv1)*dv1 + (v2c - dv2)*dv2;
        double s_curr = v1c*dv1 + v2c*dv2;

        if(s_prev < 0.0 && s_curr >= 0.0){
          double den = v1d[j]*v1d[j] + v2d[j]*v2d[j];
          if(den > 0.0){
            // the cross product is constant along a linear passage, so the
            // probability does not depend on where in the step it is evaluated
            double cross = v1c*v2d[j] - v2c*v1d[j];
            g[j] = exp(-M_PI * cross*cross / (den*sqrt(den)) );
            // the linear passage's minimal adiabatic gap: 2|v|_min
            gap_mins[j] = 2.0*fabs(cross)/sqrt(den);
          }
          else{ g[j] = 0.0; }
        }// gap minimum inside the step
        else{ g[j] = 0.0; }

      }// j!=i
    }// for j
  }// rep == 3 generalized-linear diabatic

  // Diagonal elements - common for both reps
  double sum = 0.0;
  for(j=0;j<nstates;j++){
    if(j!=i){  sum += g[j];   }
  }// for i
  g[i] = 1.0 - sum;

  return g;

}// lz

vector<double> hopping_probabilities_lz(nHamiltonian* ham, nHamiltonian* ham_prev, int act_state_indx, int rep, MATRIX& p, const MATRIX& invM, double dt){

  vector<double> gap_mins;
  return hopping_probabilities_lz(ham, ham_prev, act_state_indx, rep, p, invM, dt, gap_mins);

}// lz, no at-crossing-gap output


//vector<double> hopping_probabilities_lz(nHamiltonian& ham, nHamiltonian& ham_prev, int act_state_indx, int rep, MATRIX& p, const MATRIX& invM, double dt){
vector<double> hopping_probabilities_lz(nHamiltonian& ham, nHamiltonian& ham_prev, int act_state_indx, int rep, MATRIX& p, const MATRIX& invM, double dt){
/**
  \brief This function computes the surface hopping probabilities according to Landau-Zener formula.

  See more details in:
  (1) Tully, J. C. Molecular Dynamics with Electronic Transitions. J. Chem. Phys. 1990, 93, 1061<96>1071. - the original paper
  (2) Belyaev, A. K.; Lebedev, O. V. Nonadiabatic nuclear dynamics of atomic collisions based on branching classical trajectories
  Phys. Rev. A 2011, 84, 014701

  \param[in] ham Is the nHamiltonian object that organizes energy-related calculations
  \param[in] rep index selecting the type of representation to be used: 0 - diabatic, 1 - adiabatic
  \param[in] p [ndof x 1] or [ndof x ntraj] matrix of nuclear momenta
  \param[in] invM [ndof x 1] matrix of inverse nuclear masses
  \param[in] dt the integration timestep (a.u.) - the finite-difference gap velocity
  when the model supplies no diabatic derivatives
  \param[in] prev_ham_dia [nstates x nstates] the matrix of previous diabatic Hamiltonian - needed to locate the diabatic crossing

  Returns:
      A matrix with the hopping probabilities between all pairs of states is returned
      Convention: P(i,j) is the probability for the i->j transition

*/

  return hopping_probabilities_lz(&ham, &ham_prev, act_state_indx, rep, p, invM, dt);

}


vector<double> hopping_probabilities_bllz(MATRIX& E_prev2, MATRIX& E_prev, MATRIX& E_curr, int act_state_indx, double dt, vector<double>& gap_mins){
/**
  \brief Belyaev-Lebedev hopping probabilities - Eq. (4) of Belyaev, Lasser, Trigila,
  JCP 140, 224108 (2014) - from a three-step adiabatic energy history of one trajectory

  \param[in] E_prev2 [nstates x 1] adiabatic energies two steps back
  \param[in] E_prev [nstates x 1] adiabatic energies at the previous step
  \param[in] E_curr [nstates x 1] adiabatic energies at the current step
  \param[in] act_state_indx - index of the current adiabatic state
  \param[in] dt the integration timestep (a.u.)

  A pair's gap history Z(t-2dt) > Z(t-dt) < Z(t) marks a local minimum; the Lagrange
  parabola through the three SQUARED gaps gives the extrapolated minimum and the
  curvature - Z^2 of a locally-linear passage is exactly quadratic, so the vertex is
  sampling-exact where a parabola on Z lands at the middle-sample scale and
  overestimates the minimum as dt grows. In the regular form
  Z_min^3/Z_dd = Q_min^2/a (a = half the Q-parabola curvature):
  P = exp( -(pi/2) Q_min / sqrt(a) ), with a negative vertex clamped to zero - the
  trivial-crossing limit, P = 1. Detailed balance is
  NOT applied here: the hop acceptance machinery owns it.

  Returns:
      A nstates-vector of hopping probabilities to all states from the current active state
*/

  int nstates = E_curr.n_rows;
  vector<double> g(nstates, 0.0);
  gap_mins.assign(nstates, -1.0);
  int i = act_state_indx;
  int j;

  double sum = 0.0;
  for(j=0; j<nstates; j++){
    if(j!=i){
      double z0 = fabs(E_prev2.get(i,0) - E_prev2.get(j,0));
      double z1 = fabs(E_prev.get(i,0) - E_prev.get(j,0));
      double z2 = fabs(E_curr.get(i,0) - E_curr.get(j,0));

      if(z0 > z1 && z1 < z2){
        // fit the SQUARED gap: Z^2 of a locally-linear passage is exactly
        // quadratic in time, so the vertex recovers the true minimum at any
        // sampling; a parabola on Z itself sits at the middle-sample scale
        // in the V-shaped wings and overestimates the minimum with dt
        double q0 = z0*z0, q1 = z1*z1, q2 = z2*z2;
        // the strict minimum makes denom > 0 and keeps |t_min| < dt/2
        double denom = q0 - 2.0*q1 + q2;
        double t_min = 0.5*(q0 - q2)*dt/denom;
        double q_min = 0.5*( t_min*(t_min - dt)*q0
                      - 2.0*(t_min + dt)*(t_min - dt)*q1
                      + t_min*(t_min + dt)*q2 )/(dt*dt);
        if(q_min < 0.0){ q_min = 0.0; }  // conical/trivial crossing

        // Z_min^3/Z_dd = Q_min^2/a, a = Qdd/2 - regular at Q_min -> 0
        double a = 0.5*denom/(dt*dt);
        gap_mins[j] = sqrt(q_min);
        g[j] = exp(-0.5*M_PI*q_min/sqrt(a));
        sum += g[j];
      }// local minimum
    }// j!=i
  }// for j

  // proposals may sum above 1 when several pairs attain minima in the same
  // step - rescale, as the NBRA post-processing does
  if(sum < 1.0){  g[i] = 1.0 - sum; }
  else{
    g[i] = 0.0;
    for(j=0; j<nstates; j++){  if(j!=i){ g[j] /= sum; }  }
  }

  return g;

}// bllz

vector<double> hopping_probabilities_bllz(MATRIX& E_prev2, MATRIX& E_prev, MATRIX& E_curr, int act_state_indx, double dt){

  vector<double> gap_mins;
  return hopping_probabilities_bllz(E_prev2, E_prev, E_curr, act_state_indx, dt, gap_mins);

}// bllz, no gap-minimum output

vector<double> hopping_probabilities_zn(nHamiltonian* ham, nHamiltonian* ham_prev, int act_state_indx, int rep, MATRIX& p, const MATRIX& invM, vector<double>& gap_mins){
/**
  \brief This function computes the surface hopping probabilities according to Zhu-Nakamura generalized to multiple dimensons 
  according to formula of Yu et al. See my Chapter, Eqs. 3.89 - 3.91

  See more details in:

  \param[in] ham Is the nHamiltonian object that organizes energy-related calculations
  \param[in] ham_prev Is the nHamiltonian object of previous step or geometry - needed to locate the diabatic crossing
  \param[in] act_state_indx - index of the current adiabatic state
  \param[in] rep index selecting the type of representation to be used: 0 - diabatic, 1 - adiabatic based on the min of diabatic
  gap, 2 - adiabatic, based on the switch of the NAC sign
  \param[in] p [ndof x 1] or [ndof x ntraj] matrix of nuclear momenta
  \param[in] invM [ndof x 1] matrix of inverse nuclear masses

  Returns:
      A nstates-vector of hopping probabilities to all states from the current active state

*/
  int i,j,k;

  int ndof = ham->nnucl;
  int nstates = ham->ndia; 
//  int ntraj = ham->children.size();

  vector<double> g(nstates, 0.0);

  i = act_state_indx;

  /** Diabatic representation:

   p_{i->j} = exp( - (pi / 4.0 * |a|) * sqrt(  2 / (b^2 + sqrt( | b^4  + sign(F_i * F_j) |)) ) )

  */

  // Diabatic Ham matrices
  MATRIX ham_dia(nstates,nstates);
  MATRIX ham_dia_prev(nstates, nstates);
  ham_dia = ham->get_ham_dia().real();
  ham_dia_prev = ham_prev->get_ham_dia().real();
  if(ham_dia.max_elt()==0.0 && ham_dia_prev.max_elt()==0.0){
    cout<<"Error in hopping_probabilities_zn(): the diabatic Hamiltonian is all-zero at two \
consecutive steps - the crossing detection and the diabatic coupling are inoperative. \
Adiabatic-native models must supply obj.ham_dia \n";
    throw std::runtime_error("libra C++ error - see the message printed above");
  }

  MATRIX F(ndof, nstates);
  MATRIX Fi(ndof, 1);
  MATRIX Fj(ndof, 1);
  MATRIX tmp(ndof, 1);
  vector<int> st(1, ham->id);

  F = ham->all_forces_adi(st).real().T();
  Fi = F.col(i);

  gap_mins.assign(nstates, -1.0);

  // Off-diagonal elements - only, if we meet the gap minimum
  for(j=0; j<nstates; j++){

    if(j!=i){
      double dh = ham_dia.get(i,i) - ham_dia.get(j,j);
      double dh_prev = ham_dia_prev.get(i,i) - ham_dia_prev.get(j,j);

      if(dh * dh_prev < 0.0){
        Fj = F.col(j);

        double sgn = (Fi.T() * Fj).get(0);
        sgn = SIGN(sgn);
        
        // |Fi * Fj| = sqrt( |Fi^T * iM * F_j |)
        tmp.dot_product( Fi,  Fj);
        tmp.dot_product( tmp, invM);
        double x = sqrt(fabs(tmp.sum()));

        // |F_i - F_j | = sqrt( (F_i - F_j)^T * iM * (F_i - F_j) )
        Fj = Fi - Fj;
        tmp.dot_product( Fj, Fj );
        tmp.dot_product( tmp, invM);
        double y = sqrt( fabs(tmp.sum() ) );

        double h = fabs(ham_dia.get(i,j));
        // the two-state adiabatic gap at the diabatic crossing is
        // exactly 2|H_ij| - interpolate the coupling to the crossing
        double frac = dh_prev/(dh_prev - dh);
        gap_mins[j] = 2.0*fabs( ham_dia_prev.get(i,j)
                      + frac*(ham_dia.get(i,j) - ham_dia_prev.get(i,j)) );

        if(h > 0.0 && x > 0.0){
          double a2 = 0.0625 * x * y / (h * h * h);
          double b2 = 0.5 * y/(x * h) ;
          double g_ij = exp( -(0.25*M_PI / sqrt(a2) ) * sqrt(2.0 / ( b2 + sqrt(b2*b2 + sgn )  )) );

          g[j] = g_ij;
        }
        else{ g[j] = 0.0; }

      }// only if minimum is located
      else{ g[j] = 0.0; }

    }// j!=i
  }// for j

  // Diagonal elements - common for both reps
  double sum = 0.0;
  for(j=0;j<nstates;j++){
    if(j!=i){  sum += g[j];   }
  }// for i
  g[i] = 1.0 - sum;

  return g;

}

vector<double> hopping_probabilities_zn(nHamiltonian* ham, nHamiltonian* ham_prev, int act_state_indx, int rep, MATRIX& p, const MATRIX& invM){

  vector<double> gap_mins;
  return hopping_probabilities_zn(ham, ham_prev, act_state_indx, rep, p, invM, gap_mins);

}// zn, no at-crossing-gap output

vector<double> hopping_probabilities_zn(nHamiltonian& ham, nHamiltonian& ham_prev, int act_state_indx, int rep, MATRIX& p, const MATRIX& invM){

  return hopping_probabilities_zn(&ham, &ham_prev, act_state_indx, rep, p, invM);

}




vector<double> hopping_probabilities_mash(dyn_control_params& prms, CMATRIX& denmat){
/**
   \brief Compute the MASH surface hopping probabilities
   This is the version taking the minimal amount of input information

   The function returns the matrix g with hopping probabilities
   Abbreviation: MASH - mapped surface hopping

   Hopping with 100% probability to the state with the largest population

   References:
   (1)Mannouch, J. R.; Richardson, J. O. A Mapping Approach to Surface Hopping. J. Chem. Phys. 2023, 158 (10), 104111. https://doi.org/10.1063/5.0139734.
   (2)Runeson, J. E.; Manolopoulos, D. E. A Multi-State Mapping Approach to Surface Hopping. The Journal of Chemical Physics 2023, 159 (9), 094115. https://doi.org/10.1063/5.0158147.
   (3)E. Runeson, J.; P. Fay, T.; E. Manolopoulos, D. Exciton Dynamics from the Mapping Approach to Surface Hopping: Comparison with Förster and Redfield Theories. Physical Chemistry Chemical Physics 2024, 26 (6), 4929–4938. https://doi.org/10.1039/D3CP05926J.

   \param[in] key parameters needed for this type of calculations
     - dt - integration timestep [a.u.]
   \param[in] denmat - [nstates x nstates] - density matrix

   Returns: A nstates-vector of hopping probabilities to all states from the current active state

*/


  int nstates = denmat.n_rows;
  vector<double> g(nstates, 0.0);

  // In MASH, we set the hopping probability to 1.0 for the state with maximal density
  int max_dens_indx = 0;
  double max_dens = denmat.get(0,0).real();
  for(int i=1; i<nstates; i++){
    double dens = denmat.get(i,i).real();
    if(dens > max_dens){  max_dens_indx = i; max_dens = dens; }
  }

  g[max_dens_indx] = 1.0;
  
  return g;
}



/*
vector<MATRIX> hop_proposal_probabilities(dyn_control_params& prms,
       MATRIX& q, MATRIX& p, MATRIX& invM, CMATRIX& C, vector<CMATRIX>& projectors,
       nHamiltonian& ham, vector<MATRIX>& prev_ham_dia){
*/
vector<MATRIX> hop_proposal_probabilities(dyn_control_params& prms,
       MATRIX& q, MATRIX& p, MATRIX& invM, CMATRIX& C, 
       nHamiltonian& ham, vector<MATRIX>& prev_ham_dia){
/**
  This function computes the hop probabilities for each trajectory to hop from any state to all states

  C - are assumed to be dynamically-consistent

  TO BE DEPRECATED

*/

  int ndof = q.n_rows;
  int ntraj = q.n_cols;
  int nst = C.n_rows;    
  int traj, dof, i;
  int isNBRA = prms.isNBRA;


  vector<int> nucl_stenc_x(ndof, 0); for(i=0;i<ndof;i++){  nucl_stenc_x[i] = i; }
  vector<int> nucl_stenc_y(1, 0); 
  vector<int> el_stenc_x(nst, 0); for(i=0;i<nst;i++){  el_stenc_x[i] = i; }
  vector<int> el_stenc_y(1, 0); 
  vector<int> full_id(2,0);


  vector<MATRIX> g(ntraj, MATRIX(nst,nst)); /// the matrices of hopping probability
  MATRIX p_traj(ndof, 1);
  CMATRIX coeff(nst, 1);
  CMATRIX Hvib(nst, nst);
  vector<int> fstates(ntraj,0); 

  //============== Begin the TSH part ===================  
  // Proposed hops probabilities

  for(traj=0; traj<ntraj; traj++){

    nucl_stenc_y[0] = traj;
    el_stenc_y[0] = traj;
    full_id[1] = traj;

    pop_submatrix(C, coeff, el_stenc_x, el_stenc_y);

    if(isNBRA==1){
    // Only compute the Hvib for one traj, traj==0
      if(traj==0){
      Hvib = ham.children[traj]->get_hvib_adi();

      //Transform Hamiltonian to the dynamically-consistent form:
      //Hvib = projectors[traj].H() * Hvib * projectors[traj];
      }
    }

    else{
    // Compute the Hvib for all traj
    Hvib = ham.children[traj]->get_hvib_adi();

    //Transform Hamiltonian to the dynamically-consistent form:
    //Hvib = projectors[traj].H() * Hvib * projectors[traj];
    }

    if(prms.tsh_method == 0){ // FSSH

      g[traj] = hopping_probabilities_fssh(prms, coeff, Hvib);

    }
    else if(prms.tsh_method == 1){ // GFSH

      g[traj] = hopping_probabilities_gfsh(prms, coeff, Hvib);

    }
    else if(prms.tsh_method == 2){ // MSSH

      g[traj] = hopping_probabilities_mssh(prms, coeff, Hvib);

    }
    else if(prms.tsh_method == 3){ // LZ

//      pop_submatrix(p, p_traj, nucl_stenc_x, nucl_stenc_y);
//      g[traj] = hopping_probabilities_lz(ham.children[traj], ham_prev.children[traj], dyn_var.act_states[traj], prms.rep_lz, p_traj, *dyn_var.iM, prms.dt);

    }

    else{
      cout<<"Error in tsh1: tsh_method can be -1, 0, 1, 2, or 3. Other values are not defined\n";
      throw std::runtime_error("libra C++ error - see the message printed above");
    }

  }// for traj

  return g;

}


vector< vector<double> > hop_proposal_probabilities(dyn_control_params& prms, dyn_variables& dyn_var, 
nHamiltonian& ham, nHamiltonian& ham_prev){
//vector<MATRIX>& prev_ham_dia){
/**

  This function computes the hop probabilities for each trajectory to hop from the current state to all states

  Returns: array of the ntraj x nstates shape
*/

  int ndof = dyn_var.ndof;
  int ntraj = dyn_var.ntraj;
  int nst = dyn_var.nadi;
  if(prms.rep_tdse==0 || prms.rep_tdse==2){ nst = dyn_var.ndia; }

  int isNBRA = prms.isNBRA;

  int traj, dof, i;

  vector<int> nucl_stenc_x(ndof, 0); for(i=0;i<ndof;i++){  nucl_stenc_x[i] = i; }
  vector<int> nucl_stenc_y(1, 0); 
  vector<int> el_stenc_x(nst, 0); for(i=0;i<nst;i++){  el_stenc_x[i] = i; }
  vector<int> el_stenc_y(1, 0); 
  vector<int> full_id(2,0);


  vector< vector<double> > g(ntraj, vector<double>(nst,0.0) ); /// the hopping probability for all trajectories
  MATRIX p_traj(ndof, 1);
  CMATRIX coeff(nst, 1);
  CMATRIX dm(nst, nst);
  CMATRIX Hvib(nst, nst);
  vector<int> fstates(ntraj,0); 

  //============== Begin the TSH part ===================  
  // Proposed hops probabilities
  for(traj=0; traj<ntraj; traj++){

    if(prms.rep_sh==1){
      dm = *dyn_var.dm_adi[traj];
      //if(prms.rep_tdse==0 || prms.rep_tdse==2){ dm = *dyn_var.dm_dia[traj]; }
      if(prms.rep_tdse==2){ dm = *dyn_var.dm_dia[traj]; }
    }
    else if(prms.rep_sh==0){
      dm = *dyn_var.dm_dia[traj];
    }

    nucl_stenc_y[0] = traj;
    el_stenc_y[0] = traj;
    full_id[1] = traj;

    //pop_submatrix(C, coeff, el_stenc_x, el_stenc_y);

    if(isNBRA==1){
      // Only compute the Hvib for one traj, traj==0
      if(traj==0){   Hvib = ham.children[traj]->get_hvib_adi();   }
    }

    else{
      // Compute the Hvib for all traj
      if(prms.rep_sh==1){
        Hvib = ham.children[traj]->get_hvib_adi();
        //if(prms.rep_tdse==0 || prms.rep_tdse==2){ Hvib = ham.children[traj]->get_hvib_dia(); }
        if(prms.rep_tdse==2){ Hvib = ham.children[traj]->get_hvib_dia(); }
      }
      else{
        Hvib = ham.children[traj]->get_hvib_dia();
      }
    }

    if(prms.tsh_method == 0){ // FSSH

      if(prms.rep_sh==1){
        g[traj] = hopping_probabilities_fssh(prms, dm, Hvib, dyn_var.act_states[traj]);
      }
      else{
        g[traj] = hopping_probabilities_fssh(prms, dm, Hvib, dyn_var.act_states_dia[traj]);
      }
    }
    else if(prms.tsh_method == 1){ // GFSH

      if(prms.rep_sh==1){
        g[traj] = hopping_probabilities_gfsh(prms, dm, Hvib, dyn_var.act_states[traj]);
      }
      else if(prms.rep_sh==0){
        g[traj] = hopping_probabilities_gfsh(prms, dm, Hvib, dyn_var.act_states_dia[traj]);
      }
    }
    else if(prms.tsh_method == 2){ // MSSH

      g[traj] = hopping_probabilities_mssh(prms, dm, Hvib, dyn_var.act_states[traj]);

    }
    else if(prms.tsh_method == 3){ // LZ

      if(prms.rep_lz == 4){
        // true Belyaev-Lebedev: adiabatic energies alone; the two-step history
        // lives in dyn_var (allocated on first use) and the first two steps
        // propose nothing while it fills
        if(dyn_var.bllz_vars_status == 0){ dyn_var.allocate_bllz(); }
        if(dyn_var.lz_gap_min_status == 0){ dyn_var.allocate_lz_gap_min(); }
        int nst_t = ham.children[traj]->nadi;
        // the acceptance reads the at-crossing gaps AFTER the history has
        // shifted - stale entries must not survive a step
        for(int st=0; st<nst_t; st++){ dyn_var.lz_gap_min->set(st, traj, -1.0); }
        vector<double> gg(nst_t, 0.0);
        gg[dyn_var.act_states[traj]] = 1.0;
        if(dyn_var.bllz_hist_size >= 2){
          MATRIX Ec(nst_t, 1);
          MATRIX Ep(nst_t, 1);
          MATRIX Ep2(nst_t, 1);
          CMATRIX Ham_t(ham.children[traj]->get_ham_adi());
          for(int st=0; st<nst_t; st++){
            Ec.set(st, 0, Ham_t.get(st, st).real());
            Ep.set(st, 0, dyn_var.bllz_E_prev->get(st, traj));
            Ep2.set(st, 0, dyn_var.bllz_E_prev2->get(st, traj));
          }// for st
          vector<double> gms(nst_t, -1.0);
          gg = hopping_probabilities_bllz(Ep2, Ep, Ec, dyn_var.act_states[traj], prms.dt, gms);
          for(int st=0; st<nst_t; st++){ dyn_var.lz_gap_min->set(st, traj, gms[st]); }
        }
        g[traj] = gg;
      }
      else{
        pop_submatrix(*dyn_var.p, p_traj, nucl_stenc_x, nucl_stenc_y);
        if(dyn_var.lz_gap_min_status == 0){ dyn_var.allocate_lz_gap_min(); }
        int nst_t = ham.children[traj]->nadi;
        vector<double> gms(nst_t, -1.0);
        if(prms.rep_lz == 2 && dyn_var.bllz_vars_status == 1 &&
           dyn_var.bllz_hist_size >= 2){
          // pre-fill the three-point gap-parabola minima so the rep-2
          // probability and the acceptance both evaluate at the crossing
          MATRIX Ec(nst_t, 1);
          MATRIX Ep(nst_t, 1);
          MATRIX Ep2(nst_t, 1);
          CMATRIX Ham_t(ham.children[traj]->get_ham_adi());
          for(int st=0; st<nst_t; st++){
            Ec.set(st, 0, Ham_t.get(st, st).real());
            Ep.set(st, 0, dyn_var.bllz_E_prev->get(st, traj));
            Ep2.set(st, 0, dyn_var.bllz_E_prev2->get(st, traj));
          }// for st
          hopping_probabilities_bllz(Ep2, Ep, Ec, dyn_var.act_states[traj], prms.dt, gms);
        }
        g[traj] = hopping_probabilities_lz(ham.children[traj], ham_prev.children[traj], dyn_var.act_states[traj], 
                  prms.rep_lz, p_traj, *dyn_var.iM, prms.dt, gms);
        for(int st=0; st<nst_t; st++){ dyn_var.lz_gap_min->set(st, traj, gms[st]); }
      }

    }
    else if(prms.tsh_method == 4){ // ZN

      pop_submatrix(*dyn_var.p, p_traj, nucl_stenc_x, nucl_stenc_y);
      if(dyn_var.lz_gap_min_status == 0){ dyn_var.allocate_lz_gap_min(); }
      int nst_t = ham.children[traj]->nadi;
      vector<double> gms(nst_t, -1.0);
      g[traj] = hopping_probabilities_zn(ham.children[traj], ham_prev.children[traj], dyn_var.act_states[traj],
                prms.rep_lz, p_traj, *dyn_var.iM, gms);
      for(int st=0; st<nst_t; st++){ dyn_var.lz_gap_min->set(st, traj, gms[st]); }

    }
 
    else if(prms.tsh_method == 5){ // DISH, but handled differently
    }

    else if(prms.tsh_method == 6){ // MASH 
      g[traj] = hopping_probabilities_mash(prms, dm);
    }

    else if(prms.tsh_method == 7){ // FSSH2

      if(prms.rep_sh==1){
        CMATRIX& dm_prev = *dyn_var.dm_adi_prev[traj];
        if(prms.rep_tdse==2){ dm = *dyn_var.dm_dia_prev[traj]; }
        g[traj] = hopping_probabilities_fssh2(prms, dm, dm_prev, dyn_var.act_states[traj]);
      }
      else if(prms.rep_sh==0){
        CMATRIX& dm_prev = *dyn_var.dm_dia_prev[traj];
        g[traj] = hopping_probabilities_fssh2(prms, dm, dm_prev, dyn_var.act_states_dia[traj]);
      }
    }
    else if(prms.tsh_method == 8){ // FSSH3

      if(prms.rep_sh==1){
        CMATRIX& dm_prev = *dyn_var.dm_adi_prev[traj];
        if(prms.rep_tdse==2){ dm = *dyn_var.dm_dia_prev[traj]; }

        //cout<<"Coordinates\n";    dyn_var.q->show_matrix();

        CMATRIX& U = *dyn_var.proj_adi[traj];
        CMATRIX dm_prev_trans(*dyn_var.dm_adi_prev[traj]);
  
        // |psi'> = |psi> T =>  Hvib' = <psi'|H\hat|psi'> = T_new.H() * Hvib * T_new;
        // |Psi> = |psi> C = |psi'> C' = |psi> T C', so C = T C', C' = T^+ C => rho' = C' C'^+ = T^+ C C^+ T = T^+ rho T
        //  dm_adi_prev is the DM computed before the current step basis update, so it would transform as
        //  dm_adi_prev -> T^+ dm_adi_prev * T
        //dm_prev_trans = U.H() * dm_prev_trans * U;// transformation of the previousely-computed density matrix according to 
                                                // new ordering found at the current time-step (which is already reflected in the current DM).
        // However, for the GFSH option to work properly, we should not be using the transformation of the DM here      
        g[traj] = hopping_probabilities_fssh3(prms, dm, dm_prev_trans, dyn_var.act_states[traj], dyn_var.fssh3_errors[traj]);
      }
      else if(prms.rep_sh==0){
        CMATRIX& dm_prev = *dyn_var.dm_dia_prev[traj];
        CMATRIX dm_prev_trans(*dyn_var.dm_dia_prev[traj]);
        g[traj] = hopping_probabilities_fssh3(prms, dm, dm_prev_trans, dyn_var.act_states_dia[traj], dyn_var.fssh3_errors[traj]);
      }
    } // FSSH-3
	
	else if(prms.tsh_method == 9){ // GFSH - original
 
      if(prms.rep_sh==1){
        CMATRIX& dm_prev = *dyn_var.dm_adi_prev[traj];
        if(prms.rep_tdse==2){ dm = *dyn_var.dm_dia_prev[traj]; }
        g[traj] = hopping_probabilities_gfsh_orig(prms, dm, dm_prev, dyn_var.act_states[traj]);
      }
      else if(prms.rep_sh==0){
        CMATRIX& dm_prev = *dyn_var.dm_dia_prev[traj];
        g[traj] = hopping_probabilities_gfsh_orig(prms, dm, dm_prev, dyn_var.act_states_dia[traj]);
      }
    }// GFSH original
		

    else{
      cout<<"Error in tsh1: tsh_method can be -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9. Other values are not defined\n";
      throw std::runtime_error("libra C++ error - see the message printed above");
    }

  }// for traj

  if(prms.tsh_method == 3 || prms.tsh_method == 4){
    // push the current energies into the history AFTER all proposals -
    // it feeds the BLLZ (rep_lz 4) parabola and the rep-2 pre-fill
    if(dyn_var.bllz_vars_status == 0){ dyn_var.allocate_bllz(); }
    for(traj=0; traj<ntraj; traj++){
      CMATRIX Ham_t(ham.children[traj]->get_ham_adi());
      int nst_t = ham.children[traj]->nadi;
      for(int st=0; st<nst_t; st++){
        dyn_var.bllz_E_prev2->set(st, traj, dyn_var.bllz_E_prev->get(st, traj));
        dyn_var.bllz_E_prev->set(st, traj, Ham_t.get(st, st).real());
      }// for st
    }// for traj
    if(dyn_var.bllz_hist_size < 2){ dyn_var.bllz_hist_size += 1; }
  }

  return g;

}






int hop(vector<double>& prob, double ksi){
/** 
  \brief Attempts a stochastic hop: basically select one of the intervals in proportion
   to their length
  \param[in] ksi A random number that determines the outcome of the "hop" procedure

  Returned value: the index of the state to which we have hopped
*/
  int i;
  int nstates = prob.size();
  double left, right; left = right = 0.0;
  int finstate = -1;
  

  // To avoid problems, lets renormalize the hopping probabilities
  double nrm = 0.0;
  if(ksi >= 1){ finstate = nstates - 1; } // hops to highest state if ksi = 1 to avoid rounding errors
  else{
    for(i=0;i<nstates;i++){  nrm += prob[i];  }

    if(nrm>0.0){
      for(i=0;i<nstates;i++){    
        if(i==0){left = 0.0; right = prob[i]/nrm; }
        else{  left = right; right = right + prob[i]/nrm; }

        if((left<=ksi) && (ksi<=right)){  finstate = i;  }    
      } // for

    } // if nrm
    else{  finstate = 0; }   // probability to hop to any other states is zero
                             // so stay on the original state
                                  
    if(finstate==-1){
      std::cout<<"Something is wrong in the hop(...) function\n";
      throw std::runtime_error("libra C++ error - see the message printed above");
    }
  } // else  ksi < 1.0

  return finstate;

}// hop





int hop(int initstate, MATRIX& g, double ksi){
/** 
  \brief Attempts a stochastic hop from the initial state "initstate"
  \param[in] initstate The index of the state from which we try to hop out 
  \param[in] g The hopping probabilities matrix (type MATRIX)
  \param[in] ksi A random number that determines the outcome of the "hop" procedure

  Returned value: the index of the state to which we have hopped
*/
  int i;
  int nstates = g.n_cols;
  double left, right; left = right = 0.0;
  int finstate = -1;
  

  // To avoid problems, lets renormalize the hopping probabilities
  double nrm = 0.0;
  if(ksi >= 1){ finstate = nstates - 1; } // hops to highest state if ksi = 1 to avoid rounding errors
  else{
    for(i=0;i<nstates;i++){  nrm += g.get(initstate, i);  }

    if(nrm>0.0){
      for(i=0;i<nstates;i++){    
        if(i==0){left = 0.0; right = g.get(initstate,i)/nrm; }
        else{  left = right; right = right + g.get(initstate,i)/nrm; }

        if((left<=ksi) && (ksi<=right)){  finstate = i;  }    
      } // for

    } // if nrm
    else{  finstate = initstate; }  // probability to hop to any other states is zero
                                    // so stay on the original state
                                  
    if(finstate==-1){
      std::cout<<"Something is wrong in the hop(...) function\n";
      throw std::runtime_error("libra C++ error - see the message printed above");
    }
  } // else  ksi < 1.0

  return finstate;

}// hop




int hop(int initstate, vector<double>& g, double ksi){
/** 
  \brief Attempts a stochastic hop from the initial state "initstate"
  \param[in] initstate The index of the state from which we try to hop out 
  \param[in] g The hopping probabilities from the current active state to all other states (type vector<double>)
  \param[in] ksi A random number that determines the outcome of the "hop" procedure

  Returned value: the index of the state to which we have hopped
*/
  int i;
  int nstates = g.size();
  double left, right; left = right = 0.0;
  int finstate = -1;
  

  // To avoid problems, lets renormalize the hopping probabilities
  double nrm = 0.0;
  if(ksi >= 1){ finstate = nstates - 1; } // hops to highest state if ksi = 1 to avoid rounding errors
  else{
    for(i=0;i<nstates;i++){  nrm += g[i];  }

    if(nrm>0.0){
      for(i=0;i<nstates;i++){    
        if(i==0){left = 0.0; right = g[i]/nrm; }
        else{  left = right; right = right + g[i]/nrm; }

        if((left<=ksi) && (ksi<=right)){  finstate = i;  }    
      } // for

    } // if nrm
    else{  finstate = initstate; }  // probability to hop to any other states is zero
                                    // so stay on the original state
                                  
    if(finstate==-1){
      std::cout<<"Something is wrong in the hop(...) function\n";
      throw std::runtime_error("libra C++ error - see the message printed above");
    }
  } // else  ksi < 1.0

  return finstate;

}// hop




vector<int> propose_hops(vector<MATRIX>& g, vector<int>& act_states, Random& rnd){

  //============== Compute the proposed hops =======================
  int ntraj = act_states.size();
  vector<int> fstates(ntraj,0); 

  for(int traj=0; traj<ntraj; traj++){

    double ksi = rnd.uniform(0.0,1.0);             /// generate random number 
    fstates[traj] = hop(act_states[traj], g[traj], ksi); /// Proposed hop

  }

  return fstates;

}


vector<int> propose_hops(vector< vector<double> >& g, vector<int>& act_states, Random& rnd){

  //============== Compute the proposed hops =======================
  int ntraj = act_states.size();
  vector<int> fstates(ntraj,0); 

  for(int traj=0; traj<ntraj; traj++){

    double ksi = rnd.uniform(0.0,1.0);             /// generate random number 
    fstates[traj] = hop(act_states[traj], g[traj], ksi); /// Proposed hop

  }

  return fstates;
}



}// namespace libdyn
}// liblibra

