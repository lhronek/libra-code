/*********************************************************************************
* Copyright (C) 2015-2017 Alexey V. Akimov
*
* This file is distributed under the terms of the GNU General Public License
* as published by the Free Software Foundation, either version 2 of
* the License, or (at your option) any later version.
* See the file LICENSE in the root directory of this distribution
* or <http://www.gnu.org/licenses/>.
*
*********************************************************************************/
/**
  \file dyn_decoherence.h
  \brief The file describes the functions for decoherence corrections
    
*/

#ifndef DYN_DECOHERENCE_H
#define DYN_DECOHERENCE_H

// External dependencies
#include "../math_linalg/liblinalg.h"
#include "../io/libio.h"
#include "../nhamiltonian/libnhamiltonian.h"
#include "dyn_control_params.h"
#include "dyn_variables.h"
#include "../Units.h"

/// liblibra namespace
namespace liblibra{

using namespace libio;
using namespace libnhamiltonian;

namespace bp = boost::python;

/// libdyn namespace
namespace libdyn{


///================  In dyn_decoherence_methods.cpp  ===================================

CMATRIX sdm(CMATRIX& Coeff, double dt, int act_st, MATRIX& decoh_rates, double tol);
CMATRIX sdm(CMATRIX& Coeff, double dt, int act_st, MATRIX& decoh_rates);
CMATRIX sdm(CMATRIX& Coeff, double dt, vector<int>& act_st, vector<MATRIX>& decoh_rates, double tol, int isNBRA);
CMATRIX sdm(CMATRIX& Coeff, double dt, vector<int>& act_st, vector<MATRIX>& decoh_rates, double tol);
CMATRIX sdm(CMATRIX& Coeff, double dt, vector<int>& act_st, vector<MATRIX>& decoh_rates);


void project_out(CMATRIX& Coeff, int traj, int i);
void collapse(CMATRIX& Coeff, int traj, int i, int collapse_option);
void collapse_dm(CMATRIX* dm, int i);
// the density-matrix forms of the collapse-type corrections - rep_tdse 3 acts on dyn_variables::dm_adi
void collapse_dm(CMATRIX& rho, int i);
void project_out_dm(CMATRIX& rho, int i);
void sdm_dm(CMATRIX& rho, double dt, int act_st, MATRIX& decoh_rates, double tol);
void sdm_dm(vector<CMATRIX*>& dm, double dt, vector<int>& act_st, vector<MATRIX>& decoh_rates, double tol, int isNBRA);
void instantaneous_decoherence_dm(vector<CMATRIX*>& dm, vector<int>& accepted_states, vector<int>& proposed_states,
   vector<int>& initial_states, int instantaneous_decoherence_variant, int collapse_option);
void instantaneous_decoherence_dia_dm(vector<CMATRIX*>& dm, nHamiltonian& ham, vector<int>& accepted_states,
   vector<int>& proposed_states, vector<int>& initial_states, int instantaneous_decoherence_variant, int collapse_option);
void bcsh_dm(vector<CMATRIX*>& dm, double dt, vector<int>& act_states, MATRIX& reversal_events);
void mfsd_dm(MATRIX& p, vector<CMATRIX*>& dm, MATRIX& invM, double dt, vector<int>& act_st, vector<MATRIX>& decoherence_rates,
   nHamiltonian& ham, Random& rnd, int isNBRA);
// the density-matrix forms of SHXF and MQCXF (rep_tdse 3)
void xf_destroy_AT_dm(dyn_variables& dyn_var, double threshold);
void xf_destroy_AT_dm(dyn_variables& dyn_var, nHamiltonian& ham, double threshold);
void shxf_dm(dyn_variables& dyn_var, nHamiltonian& ham, nHamiltonian& ham_prev, dyn_control_params& prms);
void mqcxf_dm(dyn_variables& dyn_var, nHamiltonian& ham, nHamiltonian& ham_prev, dyn_control_params& prms);
void update_ham_xf_dm(dyn_variables& dyn_var);
void update_forces_xf_dm(dyn_variables& dyn_var, nHamiltonian& ham, nHamiltonian& ham_prev);
void propagate_half_xf_dm(dyn_variables& dyn_var, nHamiltonian& Ham, dyn_control_params& prms);

void instantaneous_decoherence(CMATRIX& Coeff, 
   vector<int>& accepted_states, vector<int>& proposed_states, vector<int>& initial_states,
   int instantaneous_decoherence_variant, int collapse_option);

void instantaneous_decoherence_dia(CMATRIX& Coeff, nHamiltonian& ham,
   vector<int>& accepted_states, vector<int>& proposed_states, vector<int>& initial_states,
   int instantaneous_decoherence_variant, int collapse_option);
// the Lowdin frame of the diabats per trajectory - the diabatic corrections and the diabatic DISH act in it
nHamiltonian* lowdin_child(nHamiltonian& ham, int traj);
CMATRIX lowdin_ampl(CMATRIX& Coeff, nHamiltonian& ham, vector<CMATRIX>& X, vector<int>& lowdin);
void lowdin_ampl_back(CMATRIX& Coeff, CMATRIX& d, CMATRIX& before, vector<CMATRIX>& X, vector<int>& lowdin);
vector<CMATRIX> lowdin_dm(vector<CMATRIX*>& dm, nHamiltonian& ham, vector<CMATRIX>& X, vector<int>& lowdin);
void lowdin_dm_back(vector<CMATRIX*>& dm, vector<CMATRIX>& rho, vector<CMATRIX>& before, vector<CMATRIX>& X, vector<int>& lowdin);

CMATRIX afssh_dzdt(CMATRIX& dz, CMATRIX& Hvib, CMATRIX& F, CMATRIX& C, double mass, int act_state);
void integrate_afssh_moments(CMATRIX& dR, CMATRIX& dP, CMATRIX& Hvib, CMATRIX& F, CMATRIX& C, double mass, int act_state, double dt, int nsteps);


// For branching-corrected SH
//MATRIX wp_reversal_events(MATRIX& p, MATRIX& invM, vector<int>& act_states, 
//                          nHamiltonian& ham, vector<CMATRIX>& projectors, double dt);
void wp_reversal_events(dyn_variables& dyn_var, nHamiltonian& ham, double dt);
CMATRIX bcsh(CMATRIX& Coeff, double dt, vector<int>& act_states, MATRIX& reversal_events);


// For MF-SD of Schwartz
CMATRIX mfsd(MATRIX& p, CMATRIX& Coeff, MATRIX& invM, double dt, vector<int>& act_st, vector<MATRIX>& decoherence_rates, nHamiltonian& ham, Random& rnd, int isNBRA);
CMATRIX mfsd(MATRIX& p, CMATRIX& Coeff, MATRIX& invM, double dt, vector<int>& act_st, vector<MATRIX>& decoherence_rates, nHamiltonian& ham, Random& rnd);

// Independent-trajectory XF methods
void xf_hop_reset(dyn_variables& dyn_var, vector<int>& accepted_states, vector<int>& initial_states);
void update_ham_xf(dyn_variables& dyn_var);

void shxf(dyn_variables& dyn_var, nHamiltonian& ham, nHamiltonian& ham_prev, dyn_control_params& prms); // For SHXF
void mqcxf(dyn_variables& dyn_var, nHamiltonian& ham, nHamiltonian& ham_prev, dyn_control_params& prms); // For MQCXF

// XF propagation
void rotate_nab_phase(dyn_variables& dyn_var, nHamiltonian& ham, dyn_control_params& prms);
void update_forces_xf(dyn_variables& dyn_var, nHamiltonian& ham, nHamiltonian& ham_prev);
void propagate_half_xf(dyn_variables& dyn_var, nHamiltonian& ham, dyn_control_params& prms);
void XF_correction(CMATRIX& Ham, dyn_variables& dyn_var, CMATRIX& C, double wp_width, CMATRIX& T, int traj);

///================  In dyn_decoherence_time.cpp  ===================================

MATRIX edc_rates(CMATRIX& Hvib, double Ekin, double C_param, double eps_param, int isNBRA);
MATRIX edc_rates(CMATRIX& Hvib, double Ekin, double C_param, double eps_param);

vector<MATRIX> edc_rates(vector<CMATRIX>& Hvib, vector<double>& Ekin, double C_param, double eps_param, int isNBRA);
vector<MATRIX> edc_rates(vector<CMATRIX>& Hvib, vector<double>& Ekin, double C_param, double eps_param);


void dephasing_informed_correction(MATRIX& decoh_rates, CMATRIX& Hvib, MATRIX& ave_gaps, int isNBRA);
void dephasing_informed_correction(vector<MATRIX>& decoh_rates, vector<CMATRIX>& Hvib, MATRIX& ave_gaps, int isNBRA);

void dephasing_informed_correction(MATRIX& decoh_rates, CMATRIX& Hvib, MATRIX& ave_gaps);
void dephasing_informed_correction(vector<MATRIX>& decoh_rates, vector<CMATRIX>& Hvib, MATRIX& ave_gaps);



MATRIX coherence_intervals(CMATRIX& Coeff, MATRIX& rates);
MATRIX coherence_intervals(CMATRIX& Coeff, vector<MATRIX>& rates);
MATRIX coherence_intervals_dm(CMATRIX& rho, MATRIX& rates);
MATRIX coherence_intervals_dm(vector<CMATRIX*>& dm, vector<MATRIX>& rates);

// rep selects the basis of the forces and the state count: 1 adiabatic (the forms without rep), 0 diabatic (the diabatic DISH)
vector<MATRIX> schwartz_1(dyn_control_params& prms, CMATRIX& amplitudes, nHamiltonian& ham, MATRIX& inv_alp, int rep);
vector<MATRIX> schwartz_1(dyn_control_params& prms, CMATRIX& amplitudes, MATRIX& p, nHamiltonian& ham, MATRIX& w, int rep);
vector<MATRIX> schwartz_2(dyn_control_params& prms, nHamiltonian& ham, MATRIX& inv_alp, int rep);
vector<MATRIX> schwartz_1(dyn_control_params& prms, CMATRIX& amplitudes, nHamiltonian& ham, MATRIX& inv_alp);
vector<MATRIX> schwartz_1(dyn_control_params& prms, CMATRIX& amplitudes, MATRIX& p, nHamiltonian& ham, MATRIX& inv_alp);
vector<MATRIX> schwartz_2(dyn_control_params& prms, nHamiltonian& ham, MATRIX& inv_alp);


vector<MATRIX> Gu_Franco(dyn_control_params& prms, CMATRIX& amplitudes);
// the density-matrix forms of the amplitude-based rates - the density-matrix representations (rep_tdse 2 / 3)
// hand their per-trajectory densities (dyn_variables::dm_adi), the CMATRIXList forms serve python
vector<MATRIX> schwartz_1_dm(dyn_control_params& prms, vector<CMATRIX*>& dm, nHamiltonian& ham, MATRIX& inv_alp, int rep);
vector<MATRIX> schwartz_1_dm(dyn_control_params& prms, vector<CMATRIX*>& dm, MATRIX& p, nHamiltonian& ham, MATRIX& w, int rep);
vector<MATRIX> schwartz_1_dm(dyn_control_params& prms, vector<CMATRIX*>& dm_adi, nHamiltonian& ham, MATRIX& inv_alp);
vector<MATRIX> schwartz_1_dm(dyn_control_params& prms, vector<CMATRIX*>& dm_adi, MATRIX& p, nHamiltonian& ham, MATRIX& w);
vector<MATRIX> Gu_Franco_dm(dyn_control_params& prms, vector<CMATRIX*>& dm_adi);
vector<MATRIX> schwartz_1_dm(dyn_control_params& prms, vector<CMATRIX>& dm, nHamiltonian& ham, MATRIX& inv_alp, int rep);
vector<MATRIX> schwartz_1_dm(dyn_control_params& prms, vector<CMATRIX>& dm, MATRIX& p, nHamiltonian& ham, MATRIX& w, int rep);
vector<MATRIX> schwartz_1_dm(dyn_control_params& prms, vector<CMATRIX>& dm_adi, nHamiltonian& ham, MATRIX& inv_alp);
vector<MATRIX> schwartz_1_dm(dyn_control_params& prms, vector<CMATRIX>& dm_adi, MATRIX& p, nHamiltonian& ham, MATRIX& w);
vector<MATRIX> Gu_Franco_dm(dyn_control_params& prms, vector<CMATRIX>& dm_adi);


///================  In dyn_methods_dish.cpp  ===================================
/*
vector<int> dish(dyn_control_params& prms,
       MATRIX& q, MATRIX& p,  MATRIX& invM, CMATRIX& Coeff, 
       nHamiltonian& ham, vector<int>& act_states, MATRIX& coherence_time, 
       vector<MATRIX>& decoherence_rates, Random& rnd);
*/

vector<int> dish(dyn_variables& dyn_var, nHamiltonian& ham,
                 vector<MATRIX>& decoherence_rates, dyn_control_params& prms,Random& rnd);

void dish_rev2023(dyn_variables& dyn_var, nHamiltonian& ham,
                  vector<MATRIX>& decoherence_rates,
                  dyn_control_params& prms,Random& rnd);
// the density forms on dyn_var.dm_adi (rep_tdse 3)
vector<int> dish_dm(dyn_variables& dyn_var, nHamiltonian& ham, vector<MATRIX>& decoherence_rates, dyn_control_params& prms, Random& rnd);
// DISH in the diabatic basis (rep_sh 0) on the Lowdin diabatic amplitudes ampl_dia / densities dm_dia with the diabatic rates
vector<int> dish_dia(dyn_variables& dyn_var, nHamiltonian& ham, vector<MATRIX>& decoherence_rates, dyn_control_params& prms, Random& rnd);
vector<int> dish_dia_dm(dyn_variables& dyn_var, nHamiltonian& ham, vector<MATRIX>& decoherence_rates, dyn_control_params& prms, Random& rnd);
void dish_rev2023_dm(dyn_variables& dyn_var, nHamiltonian& ham, vector<MATRIX>& decoherence_rates, dyn_control_params& prms, Random& rnd);

/*
int ida(CMATRIX& Coeff, int old_st, int new_st, double E_old, double E_new, double T, double ksi);
int dish(Electronic& el, MATRIX& t_m, const MATRIX& tau_m, const CMATRIX& Hvib,
          int use_boltz_flag, double Ekin, double T, double ksi1, double ksi2);
int dish(Electronic& el, Nuclear& mol, Hamiltonian& ham, 
          MATRIX& t_m, const MATRIX& tau_m, int use_boltz_flag, double T, double ksi1, double ksi2);
*/




}// namespace libdyn
}// liblibra

#endif // DYN_DECOHERENCE_H

