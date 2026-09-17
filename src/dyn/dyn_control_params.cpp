/*********************************************************************************
* Copyright (C) 2019-2024 Alexey V. Akimov
*
* This file is distributed under the terms of the GNU General Public License
* as published by the Free Software Foundation, either version 3 of
* the License, or (at your option) any later version.
* See the file LICENSE in the root directory of this distribution
* or <http://www.gnu.org/licenses/>.
*
*********************************************************************************/
/**
  \file dyn_control_params.cpp
  \brief The file implements the methods to setup control parameters for dynamics
*/

#include "dyn_control_params.h"
#include <stdexcept>

/// liblibra namespace
namespace liblibra{

using namespace liblinalg;

/// libdyn namespace
namespace libdyn{

namespace bp = boost::python;

dyn_control_params::dyn_control_params(){

/**

  This function initializes the default values of control parameters

*/


  ///================= Computing Hamiltonian-related properties ====================
  rep_tdse = 1;
//  rep_ham = 0;
  ham_update_method = 1;
  ham_transform_method = 1;
  rep_sh = 1;
  rep_lz = 0;
  lz_target_space = 1;
  rep_force = 1;
  force_method = 1;
  ehrenfest_force_option = 0;
  ehrenfest_normalized = 1;
  sqc_gamma = 0.0;
  enforce_state_following = 0; 
  enforced_state_index = 0; 
  time_overlap_method = 0;
  nac_update_method = 1;
  nac_algo = -1;
  hvib_update_method = 1;
  do_ssy = 0;
  do_phase_correction = 1;
  do_nac_phase_correction = 0;
  phase_correction_tol = 1e-3; 
  state_tracking_algo = -1;
  MK_alpha = 0.0;
  MK_verbosity = 0;
  MK_scaling_function = 0;
  convergence = 0;
  max_number_attempts = 100;
  min_probability_reordering = 0.0;
  isNBRA = 0;
  compute_dm_dia = -1;
  omp_num_threads = 1;

  ///================= Surface hopping: proposal, acceptance =======================
  tsh_method = 0;
  hop_acceptance_algo = 0;
  momenta_rescaling_algo = 0;
  use_Jasper_Truhlar_criterion = 1;
  use_boltz_factor = 0;

  ///================= FSSH2 specific ====================
  fssh2_revision = 0;

  ///================= FSSH3 specific ====================
  fssh3_size_option = 1;
  fssh3_approach_option = 0;
  fssh3_decomp_option = 3;
  fssh3_dt = 0.001;
  fssh3_max_steps = 1000;
  fssh3_err_tol = 1e-7;
  
  ///================= QTSH specific ====================
  use_qtsh = 0;
  qtsh_force_option = 1;

  ///================= KCRPMD specific ====================
  use_kcrpmd = 0;
  kcrpmd_eta = 6.28;
  kcrpmd_a = 0.1;
  kcrpmd_b = 1000.;
  kcrpmd_c = 1.0;
  kcrpmd_gamma = 0.0;
  kcrpmd_gammaKP = 0.0;

  ///================= Decoherence options =========================================
  decoherence_algo = -1; 
  sdm_norm_tolerance = 0.0;
  dish_decoherence_event_option = 1;
  decoherence_times_type = -1;
  schwartz_decoherence_inv_alpha = NULL;
  schwartz_interaction_width = NULL;
  decoherence_C_param = 1.0;
  decoherence_eps_param = 0.1;
  dephasing_informed = 0;
  instantaneous_decoherence_variant = 1; 
  collapse_option = 0;
  decoherence_rates = NULL;
  ave_gaps = NULL;
  wp_width = NULL;
  wp_v = NULL;
  coherence_threshold = 0.01;
  e_mask = 0.0001;
  use_xf_force = 0;
  project_out_aux = 0;
  tp_algo = 1;
  use_td_width = 0;
  gap_correlation_time = 41.0;

  ///================= Entanglement of trajectories ================================
  entanglement_opt = 0;
  ETHD3_alpha = 1.0;
  ETHD3_beta = 1.0;

  ///============================ QTAG =============================================
  qtag_pot_approx_method = 0;
  ///================= Bath, Constraints, and Dynamical controls ===================

  Temperature = 300.0;
  ensemble = 0;
  thermostat_params = bp::dict();

  thermostat_dofs = vector<int>();
  quantum_dofs = vector<int>(1, 0);
  constrained_dofs = vector<int>();

  dt = 41.0;
  num_electronic_substeps = 1;
  electronic_integrator = 0;
  ampl_transformation_method = 1; 
  assume_always_consistent = 0;

  thermally_corrected_nbra = 0;
  total_energy = 0.01; // some reasonable value
  tcnbra_nu_therm = 0.001; 
  tcnbra_nhc_size = 1;
  tcnbra_do_nac_scaling = 1;

  properties_to_save = bp::list();

  reorg_energy = 0.0;
}


dyn_control_params::dyn_control_params(const dyn_control_params& x){ 
  //cout<<"dyn_control_params cctor\n";

  rep_tdse = x.rep_tdse;
//  rep_ham = x.rep_ham;
  ham_update_method = x.ham_update_method;
  ham_transform_method = x.ham_transform_method;
  rep_sh = x.rep_sh;
  rep_lz = x.rep_lz;
  lz_target_space = x.lz_target_space;
  rep_force = x.rep_force;
  force_method = x.force_method;
  ehrenfest_force_option = x.ehrenfest_force_option;
  ehrenfest_normalized = x.ehrenfest_normalized;
  sqc_gamma = x.sqc_gamma;
  enforce_state_following = x.enforce_state_following;
  enforced_state_index = x.enforced_state_index; 
  time_overlap_method = x.time_overlap_method;
  nac_update_method = x.nac_update_method;
  nac_algo = x.nac_algo;
  hvib_update_method = x.hvib_update_method;
  do_ssy = x.do_ssy;
  do_phase_correction = x.do_phase_correction;
  do_nac_phase_correction = x.do_nac_phase_correction;
  phase_correction_tol = x.phase_correction_tol; 
  state_tracking_algo = x.state_tracking_algo;
  MK_alpha = x.MK_alpha;
  MK_verbosity = x.MK_verbosity;
  MK_scaling_function = x.MK_scaling_function;
  convergence = x.convergence;
  max_number_attempts = x.max_number_attempts;
  min_probability_reordering = x.min_probability_reordering;
  isNBRA = x.isNBRA;
  compute_dm_dia = x.compute_dm_dia;
  omp_num_threads = x.omp_num_threads;

  ///================= Surface hopping: proposal, acceptance =======================
  tsh_method = x.tsh_method;
  hop_acceptance_algo = x.hop_acceptance_algo;
  momenta_rescaling_algo = x.momenta_rescaling_algo;
  use_Jasper_Truhlar_criterion = x.use_Jasper_Truhlar_criterion;
  use_boltz_factor = x.use_boltz_factor;

  ///================= FSSH2 specific ====================
  fssh2_revision = x.fssh2_revision;
 
  ///================= FSSH3 specific ====================
  fssh3_size_option = x.fssh3_size_option;
  fssh3_approach_option = x.fssh3_approach_option;
  fssh3_decomp_option = x.fssh3_decomp_option;
  fssh3_dt = x.fssh3_dt;
  fssh3_max_steps = x.fssh3_max_steps;
  fssh3_err_tol = x.fssh3_err_tol;
  
  ///================= QTSH specific ====================
  use_qtsh = x.use_qtsh;
  qtsh_force_option = x.qtsh_force_option;

  ///================= KC-RPMD specific ====================
  use_kcrpmd = x.use_kcrpmd;
  kcrpmd_eta = x.kcrpmd_eta;
  kcrpmd_a = x.kcrpmd_a;
  kcrpmd_b = x.kcrpmd_b;
  kcrpmd_c = x.kcrpmd_c;
  kcrpmd_gamma = x.kcrpmd_gamma;
  kcrpmd_gammaKP = x.kcrpmd_gammaKP;

  ///================= Decoherence options =========================================
  decoherence_algo = x.decoherence_algo; 
  sdm_norm_tolerance = x.sdm_norm_tolerance;
  dish_decoherence_event_option = x.dish_decoherence_event_option;
  decoherence_times_type = x.decoherence_times_type;
  decoherence_C_param = x.decoherence_C_param;
  decoherence_eps_param = x.decoherence_eps_param;
  dephasing_informed = x.dephasing_informed;
  instantaneous_decoherence_variant = x.instantaneous_decoherence_variant; 
  collapse_option = x.collapse_option;

  wp_width = new MATRIX( x.wp_width->n_rows, x.wp_width->n_cols );
  *wp_width = *x.wp_width;
  wp_v = new MATRIX( x.wp_v->n_rows, x.wp_v->n_cols );
  *wp_v = *x.wp_v;
  coherence_threshold = x.coherence_threshold;
  e_mask = x.e_mask;
  use_xf_force = x.use_xf_force;
  project_out_aux = x.project_out_aux;
  tp_algo = x.tp_algo;
  use_td_width = x.use_td_width;
  gap_correlation_time = x.gap_correlation_time;

  ///================= Entanglement of trajectories ================================
  entanglement_opt = x.entanglement_opt;
  ETHD3_alpha = x.ETHD3_alpha;
  ETHD3_beta = x.ETHD3_beta;

  ///============================ QTAG =============================================
  qtag_pot_approx_method = x.qtag_pot_approx_method;
  ///================= Bath, Constraints, and Dynamical controls ===================

  Temperature = x.Temperature;
  ensemble = x.ensemble;

  thermostat_params = bp::dict(x.thermostat_params);

  thermostat_dofs = x.thermostat_dofs;
  quantum_dofs = x.quantum_dofs;
  constrained_dofs = x.constrained_dofs;

  dt = x.dt;
  num_electronic_substeps = x.num_electronic_substeps;
  electronic_integrator = x.electronic_integrator;
  ampl_transformation_method = x.ampl_transformation_method;
  assume_always_consistent = x. assume_always_consistent;

  decoherence_rates = new MATRIX(x.decoherence_rates->n_rows, x.decoherence_rates->n_cols);  
  *decoherence_rates = *x.decoherence_rates;

  ave_gaps = new MATRIX( x.ave_gaps->n_rows, x.ave_gaps->n_cols );
  *ave_gaps = *x.ave_gaps;

  schwartz_decoherence_inv_alpha = new MATRIX( x.schwartz_decoherence_inv_alpha->n_rows, x.schwartz_decoherence_inv_alpha->n_cols );
  *schwartz_decoherence_inv_alpha = *x.schwartz_decoherence_inv_alpha;
  
  schwartz_interaction_width = new MATRIX( x.schwartz_interaction_width->n_rows, x.schwartz_interaction_width->n_cols );
  *schwartz_interaction_width = *x.schwartz_interaction_width;

  thermally_corrected_nbra = x.thermally_corrected_nbra;
  total_energy = x.total_energy;
  tcnbra_nu_therm = x.tcnbra_nu_therm;
  tcnbra_nhc_size = x.tcnbra_nhc_size;
  tcnbra_do_nac_scaling = x.tcnbra_do_nac_scaling;

  properties_to_save = bp::list(x.properties_to_save);
  
  reorg_energy = x.reorg_energy;
}

dyn_control_params::~dyn_control_params() {  

  //cout<<"dyn_control_params destructor\n";

  delete wp_width;
  delete wp_v;
  delete decoherence_rates;  
  delete ave_gaps;
  delete schwartz_decoherence_inv_alpha;
  delete schwartz_interaction_width;
}


void dyn_control_params::sanity_check(){

  /// hop acceptance/rescaling FAMILIES are representation-bound: the
  /// dia-energy options (acceptance 11, rescaling 110/111) index ham_dia
  /// with the DIABATIC hop states, every other option indexes adiabatic
  /// quantities - crossing them with the other rep_sh mis-gaps every hop
  /// (energy jumps by the dia-adi gap difference, silently non-conserving)
  if(tsh_method >= 0){
    int acc = hop_acceptance_algo;
    int res = momenta_rescaling_algo;
    int acc_dia = (acc==11);
    // the Boltzmann options 31-33 read the hopping representation's own energies and
    // pair with either rep_sh
    int acc_adi = (acc==10 || acc==20 || acc==21 || acc==40);
    int res_dia = (res==110 || res==111);
    int res_adi = (res==100 || res==101 || res==200 || res==201 ||
                   res==210 || res==211 || res==40);
    if(rep_sh==0 && (acc_adi || res_adi)){
      std::cout<<"Error in dyn_control_params::sanity_check: rep_sh = 0 hops between"
          <<" DIABATIC states, but hop_acceptance_algo = "<<acc
          <<" / momenta_rescaling_algo = "<<res<<" read ADIABATIC quantities -"
          <<" use the diabatic family (acceptance 0, 11 or the Boltzmann 31-33, rescaling 0, 110 or 111)\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }
    if(rep_sh==1 && (acc_dia || res_dia)){
      std::cout<<"Error in dyn_control_params::sanity_check: rep_sh = 1 hops between"
          <<" ADIABATIC states, but hop_acceptance_algo = "<<acc
          <<" / momenta_rescaling_algo = "<<res<<" read DIABATIC energies -"
          <<" use the adiabatic family (e.g. acceptance 10/20/21, rescaling 100/101/200/201/210/211)\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }
  }

  if(!(lz_target_space==0 || lz_target_space==1)){
    std::cout<<"Error in dyn_control_params::sanity_check: lz_target_space = "
        <<lz_target_space<<" is not allowed (0 - energy-adjacent only, 1 - all pairs)\n";
    std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
  }

  ///=================== Options for state tracking ======================
  if(state_tracking_algo==-1 ||
     state_tracking_algo==0 || state_tracking_algo==1 ||
     state_tracking_algo==2 || state_tracking_algo==21 || 
     state_tracking_algo==3 || state_tracking_algo==32 || state_tracking_algo==33 ||
     state_tracking_algo==4 || state_tracking_algo==5  || state_tracking_algo==6 ){ ; ; }
  else{
    std::cout<<"Error in dyn_control_params::sanity_check: state_tracking_algo = "
        <<state_tracking_algo<<" is not allowed\n";
    std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
  }

  /// the local-diabatization transform accumulates the per-step projector
  /// into the trajectory frame and rotates ham_adi - it needs an
  /// adiabatic-native source (the exclusive dispatch never re-diagonalizes
  /// a diabatic model under this option) and a square frame
  if(ham_transform_method==4 && ham_update_method!=2){
    std::cout<<"Error in dyn_control_params::sanity_check: ham_transform_method = 4"
        <<" (adiabatic->diabatic by local diabatization) requires an adiabatic-native"
        <<" model (ham_update_method = 2, got "<<ham_update_method<<") - with a"
        <<" diabatic model the loop never re-diagonalizes under this option and"
        <<" the adiabatic side stays frozen at its initial values\n";
    throw std::runtime_error("libra C++ error - see the message printed above");
  }

  /// the frame accumulates the per-step projector, which must come from a
  /// genuine raw-basis time-overlap - a basis-transform-derived St (method
  /// 1) reads the very matrix the frame overwrites
  if(ham_transform_method==4 && time_overlap_method!=0){
    std::cout<<"Error in dyn_control_params::sanity_check: ham_transform_method = 4"
        <<" requires supplied time-overlaps (time_overlap_method = 0, got "
        <<time_overlap_method<<") - the frame owns basis_transform, so the"
        <<" method-1 rebuild from basis transforms would read the frame itself\n";
    throw std::runtime_error("libra C++ error - see the message printed above");
  }

  /// the stored-U forward transform reads ham_dia, which only a diabatic
  /// model keeps current - and consumes the model-supplied transformation,
  /// which an adiabatic-native model applies through ham_update_method 2
  /// directly
  if(ham_transform_method==2 && ham_update_method!=1){
    std::cout<<"Error in dyn_control_params::sanity_check: ham_transform_method = 2"
        <<" (diabatic->adiabatic by the stored basis transform) requires a diabatic"
        <<" model (ham_update_method = 1, got "<<ham_update_method<<") - nothing"
        <<" else populates ham_dia for it to transform\n";
    throw std::runtime_error("libra C++ error - see the message printed above");
  }

  /// the back-transform reads ham_adi, which only an adiabatic-native
  /// model keeps current - under ham_update_method 1 the propagation loop
  /// never re-diagonalizes (the transform dispatch is exclusive), so the
  /// dynamics would silently run on the stale initial-step Hamiltonian
  if(ham_transform_method==3 && ham_update_method!=2){
    std::cout<<"Error in dyn_control_params::sanity_check: ham_transform_method = 3"
        <<" (adiabatic->diabatic back-transform) requires an adiabatic-native"
        <<" model (ham_update_method = 2, got "<<ham_update_method<<") - with a"
        <<" diabatic model the loop never re-diagonalizes under this option and"
        <<" the adiabatic side stays frozen at its initial values\n";
    throw std::runtime_error("libra C++ error - see the message printed above");
  }

  ///============== Integrator vs projector family pairing ==============
  /// rep_tdse == 1: the bare-energy local-diabatization integrators carry
  /// the whole electronic coupling through the rotation projector
  /// (state_tracking_algo -1/5/6 build T from St), while the NAC-carrying
  /// Hvib integrators need a relabel-only projector - crossing the
  /// families silently double-counts the basis rotation or freezes the
  /// electronic subsystem
  if(rep_tdse==1){
    int m = electronic_integrator % 100; // the 10x variants share the classification
    int rot_proj = (state_tracking_algo==-1 || state_tracking_algo==5 ||
                    state_tracking_algo==6);
    int ld_int = (m==0 || m==1 || m==2 || m==10 || m==11 || m==12);
    int hvib_int = (m==3 || m==4 || m==5 || m==7 || m==8 || m==13 ||
                    m==14 || m==15);
    if(ld_int && (!rot_proj || assume_always_consistent==1)){
      std::cout<<"Error in dyn_control_params::sanity_check: the bare-energy"
          <<" local-diabatization integrator (electronic_integrator = "
          <<electronic_integrator<<") carries the whole electronic coupling"
          <<" through the St-derived rotation projector - it requires"
          <<" state_tracking_algo -1, 5 or 6 and assume_always_consistent = 0"
          <<" (got state_tracking_algo = "<<state_tracking_algo
          <<", assume_always_consistent = "<<assume_always_consistent
          <<"); a relabel-only or identity projector freezes the electronic"
          <<" subsystem\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }
    // the NAC-side sign correction re-signs the couplings by continuity
    // (raw gauge); a relabel projector re-signs the amplitudes instead -
    // both at once double-correct every eigenvector branch switch
    if(do_nac_phase_correction==1 && !rot_proj && assume_always_consistent==0){
      std::cout<<"Error in dyn_control_params::sanity_check: do_nac_phase_correction = 1"
          <<" together with a relabel projector (state_tracking_algo = "<<state_tracking_algo
          <<", assume_always_consistent = 0) corrects each eigenvector sign flip twice -"
          <<" once in the NACs, once in the amplitudes; use it only with"
          <<" assume_always_consistent = 1 (no projector), or drop it\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }
    if(hvib_int && rot_proj && assume_always_consistent==0){
      std::cout<<"Error in dyn_control_params::sanity_check: the NAC-carrying"
          <<" Hvib integrator (electronic_integrator = "<<electronic_integrator
          <<") double-counts the basis rotation under the St-derived rotation"
          <<" projector (state_tracking_algo = "<<state_tracking_algo
          <<"); use a relabel projector (state_tracking_algo 0/1/2/21 with"
          <<" do_phase_correction = 1), or assume_always_consistent = 1 with"
          <<" do_nac_phase_correction = 1\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }
  }

  /// every representation implements a fixed set of electronic integrators
  /// in propagate_electronic - amplitudes (rep_tdse 0 and 1): -1, 0-8, 10-15;
  /// the diabatic density matrix (2): 0; the adiabatic density matrix (3):
  /// 0, 1, 10; each with the +100 variant. Any other value would leave the
  /// electronic variables untouched for the whole run - refuse it here (the
  /// dispatch throws as well)
  if(rep_tdse>=0 && rep_tdse<=3){
    int m = electronic_integrator % 100;
    int in_range = (electronic_integrator>=0 && electronic_integrator<200);
    int listed = 0;
    std::string implemented;
    if(rep_tdse==0 || rep_tdse==1){
      listed = (electronic_integrator==-1) || (in_range && ((m>=0 && m<=8) || (m>=10 && m<=15)));
      implemented = "-1, 0-8, 10-15 or their +100 variants";
    }
    else if(rep_tdse==2){ listed = (in_range && m==0); implemented = "0 or 100"; }
    else { listed = (in_range && (m==0 || m==1 || m==10)); implemented = "0, 1, 10 or their +100 variants"; }
    if(!listed){
      std::cout<<"Error in dyn_control_params::sanity_check: electronic_integrator = "
          <<electronic_integrator<<" is not implemented for rep_tdse = "<<rep_tdse
          <<" - propagate_electronic implements "<<implemented<<" there\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }
  }

  /// rep_tdse == 3 (adiabatic density matrix): methods 0/10 dress bare
  /// energies with T and frame-transform RHO, and the Zhu Liouvillian (1)
  /// generates nothing from a pure state without the rotation frame
  /// creating coherences - all three freeze under a relabel projector
  if(rep_tdse==3){
    int m = electronic_integrator % 100;
    int rot_proj = (state_tracking_algo==-1 || state_tracking_algo==5 ||
                    state_tracking_algo==6);
    if((m==0 || m==1 || m==10) && (!rot_proj || assume_always_consistent==1)){
      std::cout<<"Error in dyn_control_params::sanity_check: the adiabatic"
          <<" density-matrix integrator (electronic_integrator = "
          <<electronic_integrator<<") carries the electronic coupling"
          <<" through the St-derived rotation projector - it requires"
          <<" state_tracking_algo -1, 5 or 6 and assume_always_consistent = 0"
          <<" (got state_tracking_algo = "<<state_tracking_algo
          <<", assume_always_consistent = "<<assume_always_consistent<<")\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }
  }

  /// TC-NBRA scales per-trajectory couplings - a shared NBRA child
  /// would compound every trajectory's factor on the same matrices
  if(thermally_corrected_nbra==1 && isNBRA==1){
    std::cout<<"Error in dyn_control_params::sanity_check: thermally_corrected_nbra needs "
        <<"per-trajectory child Hamiltonians - run it with isNBRA = 0\n";
    std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
  }

  /// the Ehrenfest force unit function computes nothing for any other value
  if(ehrenfest_force_option!=0 && ehrenfest_force_option!=1){
    std::cout<<"Error in dyn_control_params::sanity_check: ehrenfest_force_option = "
        <<ehrenfest_force_option<<" is not allowed (0 - all terms, 1 - no derivative-coupling terms)\n";
    throw std::runtime_error("libra C++ error - see the message printed above");
  }

  /// the MMST representation is a stub: no propagation branch exists for it,
  /// the amplitudes would stay frozen through the whole run
  if(rep_tdse==4){
    std::cout<<"Error in dyn_control_params::sanity_check: rep_tdse = 4 (MMST adiabatic"
        <<" wfc) is not implemented - no electronic propagation exists for it;"
        <<" use rep_tdse = 1 (the MMST electronic EOM is the adiabatic TDSE) with"
        <<" force_method = 2 and sqc_gamma for the Meyer-Miller force\n";
    throw std::runtime_error("libra C++ error - see the message printed above");
  }

  /// the QTSH force unit functions compute nothing for any other value
  if(qtsh_force_option!=0 && qtsh_force_option!=1 && qtsh_force_option!=2){
    std::cout<<"Error in dyn_control_params::sanity_check: qtsh_force_option = "
        <<qtsh_force_option<<" is not allowed (0 - first-order off-diagonal force, 1 - with the"
        <<" second-order term, 2 - no derivative-coupling force, adiabatic only)\n";
    throw std::runtime_error("libra C++ error - see the message printed above");
  }

  if(ehrenfest_normalized!=0 && ehrenfest_normalized!=1){
    std::cout<<"Error in dyn_control_params::sanity_check: ehrenfest_normalized = "
        <<ehrenfest_normalized<<" is not allowed (1 - divide by <c|S|c>, 0 - the un-normalized Meyer-Miller form)\n";
    throw std::runtime_error("libra C++ error - see the message printed above");
  }

  /// Shall not use DISH with the decoherence correction
  /// (tsh_method 3 is Landau-Zener - DISH is 5)
  if(tsh_method==5 && decoherence_algo!=-1){
    cout<<"Error in dyn_control_params::sanity_check: Shall not use DISH (tsh_method == 5) with the \
           decoherence correction (any but decoherence_algo == -1)\n";
    std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
  }

  /// Check that we don't constrian the thermostatted DOFs and also quantum DOFs
  int sz1 = constrained_dofs.size();
  if(sz1!=0){

    int sz2 = thermostat_dofs.size();

    for(int i1=0; i1<sz1; i1++){
      for(int i2=0; i2<sz2; i2++){
        if( constrained_dofs[i1] == thermostat_dofs[i2] ){
           cout<<"Error in dyn_control_params::sanity_check: can not constrain the thermostatted DOF "<<thermostat_dofs[i2]<<endl;
           std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
        }
      }// for i2
    }// for i1

    sz2 = quantum_dofs.size();
    for(int i1=0; i1<sz1; i1++){
      for(int i2=0; i2<sz2; i2++){
        if( constrained_dofs[i1] == quantum_dofs[i2] ){
           cout<<"Error in dyn_control_params::sanity_check: can not constrain the quantum DOF "<<quantum_dofs[i2]<<endl;
           std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
        }
      }// for i2
    }// for i1

  }// sz1!=0


  if(num_electronic_substeps<=0){
      cout<<"Error in dyn_control_params::sanity_check: num_electronic_substeps = "<<num_electronic_substeps
          <<" should be a positive integer"<<endl;
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
  }

}



void dyn_control_params::set_parameters(bp::dict params){
/**
  Extract the parameters from the input dictionary
*/

  std::string key;
  for(int i=0;i<len(params.values());i++){
    key = bp::extract<std::string>(params.keys()[i]);

    ///================= Computing Hamiltonian-related properties ====================
    if(key=="rep_tdse") { rep_tdse = bp::extract<int>(params.values()[i]); }
//    else if(key=="rep_ham") { rep_ham = bp::extract<int>(params.values()[i]);   }
    else if(key=="ham_update_method") { ham_update_method = bp::extract<int>(params.values()[i]);   }
    else if(key=="ham_transform_method") { ham_transform_method = bp::extract<int>(params.values()[i]);   }
    else if(key=="rep_sh") { rep_sh = bp::extract<int>(params.values()[i]);  }
    else if(key=="rep_lz") { rep_lz = bp::extract<int>(params.values()[i]);  }
    else if(key=="lz_target_space") { lz_target_space = bp::extract<int>(params.values()[i]);  }
    else if(key=="rep_force") { rep_force = bp::extract<int>(params.values()[i]);  }
    else if(key=="force_method") { force_method = bp::extract<int>(params.values()[i]);  }
    else if(key=="ehrenfest_force_option") { ehrenfest_force_option = bp::extract<int>(params.values()[i]);  }
    else if(key=="ehrenfest_normalized") { ehrenfest_normalized = bp::extract<int>(params.values()[i]);  }
    else if(key=="sqc_gamma") { sqc_gamma = bp::extract<double>(params.values()[i]);  }
    else if(key=="enforce_state_following") { enforce_state_following = bp::extract<int>(params.values()[i]);  }
    else if(key=="enforced_state_index") { enforced_state_index = bp::extract<int>(params.values()[i]);  }
    else if(key=="time_overlap_method"){ time_overlap_method = bp::extract<double>(params.values()[i]); }
    else if(key=="nac_update_method") { nac_update_method = bp::extract<int>(params.values()[i]);  }
    else if(key=="nac_algo") { nac_algo = bp::extract<int>(params.values()[i]);  }
    else if(key=="hvib_update_method") { hvib_update_method = bp::extract<int>(params.values()[i]);   }
    else if(key=="do_ssy") { do_ssy = bp::extract<int>(params.values()[i]);   }
    else if(key=="do_phase_correction") { do_phase_correction = bp::extract<int>(params.values()[i]);  }
    else if(key=="phase_correction_tol") { phase_correction_tol = bp::extract<double>(params.values()[i]);  }
    else if(key=="do_nac_phase_correction"){ do_nac_phase_correction = bp::extract<int>(params.values()[i]); }
    else if(key=="state_tracking_algo"){  state_tracking_algo = bp::extract<int>(params.values()[i]);  }
    else if(key=="MK_alpha") { MK_alpha = bp::extract<double>(params.values()[i]);  }
    else if(key=="MK_verbosity") { MK_verbosity = bp::extract<int>(params.values()[i]);  }
    else if(key=="MK_scaling_function") { MK_scaling_function = bp::extract<int>(params.values()[i]);  }
    else if(key=="convergence") { convergence = bp::extract<int>(params.values()[i]);  }
    else if(key=="max_number_attempts") { max_number_attempts = bp::extract<int>(params.values()[i]);  }
    else if(key=="min_probability_reordering") { min_probability_reordering = bp::extract<double>(params.values()[i]);  }
    else if(key=="isNBRA") { isNBRA = bp::extract<int>(params.values()[i]);   }
    else if(key=="compute_dm_dia") { compute_dm_dia = bp::extract<int>(params.values()[i]);   }
    else if(key=="omp_num_threads") { omp_num_threads = bp::extract<int>(params.values()[i]);   }

    ///================= Surface hopping: proposal, acceptance =======================
    else if(key=="tsh_method") { tsh_method = bp::extract<int>(params.values()[i]);  }
    else if(key=="hop_acceptance_algo") { hop_acceptance_algo = bp::extract<int>(params.values()[i]);  }
    else if(key=="momenta_rescaling_algo"){ momenta_rescaling_algo = bp::extract<int>(params.values()[i]);  }
    else if(key=="use_Jasper_Truhlar_criterion"){ use_Jasper_Truhlar_criterion = bp::extract<int>(params.values()[i]); }
    else if(key=="use_boltz_factor"){ use_boltz_factor = bp::extract<int>(params.values()[i]);  }

    ///================= FSSH2 specific ====================
    else if(key=="fssh2_revision"){ fssh2_revision = bp::extract<int>(params.values()[i]);  }

    ///================= FSSH3 specific ====================
    else if(key=="fssh3_size_option"){ fssh3_size_option = bp::extract<int>(params.values()[i]);  }
    else if(key=="fssh3_approach_option"){ fssh3_approach_option = bp::extract<int>(params.values()[i]);  }
    else if(key=="fssh3_decomp_option"){ fssh3_decomp_option = bp::extract<int>(params.values()[i]);  }
    else if(key=="fssh3_dt"){ fssh3_dt = bp::extract<double>(params.values()[i]);  }
    else if(key=="fssh3_max_steps"){ fssh3_max_steps = bp::extract<int>(params.values()[i]);  }
    else if(key=="fssh3_err_tol"){ fssh3_err_tol = bp::extract<double>(params.values()[i]);  }
    
    ///================= QTSH specific ====================
    else if(key=="use_qtsh"){ use_qtsh = bp::extract<int>(params.values()[i]);  }
    else if(key=="qtsh_force_option"){ qtsh_force_option = bp::extract<int>(params.values()[i]);  }

    ///================= KC-RPMD specific ====================
    else if(key=="use_kcrpmd"){ use_kcrpmd = bp::extract<int>(params.values()[i]);  }
    else if(key=="kcrpmd_eta"){ kcrpmd_eta = bp::extract<double>(params.values()[i]);  }
    else if(key=="kcrpmd_a"){ kcrpmd_a = bp::extract<double>(params.values()[i]);  }
    else if(key=="kcrpmd_b"){ kcrpmd_b = bp::extract<double>(params.values()[i]);  }
    else if(key=="kcrpmd_c"){ kcrpmd_c = bp::extract<double>(params.values()[i]);  }
    else if(key=="kcrpmd_gamma"){ kcrpmd_gamma = bp::extract<double>(params.values()[i]);  }
    else if(key=="kcrpmd_gammaKP"){ kcrpmd_gammaKP = bp::extract<double>(params.values()[i]);  }

    ///================= Decoherence options =========================================
    else if(key=="decoherence_algo"){ decoherence_algo = bp::extract<int>(params.values()[i]); }
    else if(key=="sdm_norm_tolerance"){ sdm_norm_tolerance = bp::extract<double>(params.values()[i]); }
    else if(key=="dish_decoherence_event_option"){ dish_decoherence_event_option = bp::extract<int>(params.values()[i]); }
    else if(key=="decoherence_times_type"){ decoherence_times_type = bp::extract<int>(params.values()[i]); }
    else if(key=="schwartz_decoherence_inv_alpha"){ 
      MATRIX x( bp::extract<MATRIX>(params.values()[i]) );
      schwartz_decoherence_inv_alpha = new MATRIX(x.n_rows, x.n_cols);      
      for(int a=0;a<x.n_rows;a++){
        for(int b=0;b<x.n_cols;b++){ schwartz_decoherence_inv_alpha->set(a, b, x.get(a,b));   }
      } 
    }
    else if(key=="schwartz_interaction_width"){ 
      MATRIX x( bp::extract<MATRIX>(params.values()[i]) );
      schwartz_interaction_width = new MATRIX(x.n_rows, x.n_cols);      
      for(int a=0;a<x.n_rows;a++){
        for(int b=0;b<x.n_cols;b++){ schwartz_interaction_width->set(a, b, x.get(a,b));   }
      } 
    }
    else if(key=="decoherence_C_param"){ decoherence_C_param = bp::extract<double>(params.values()[i]); }
    else if(key=="decoherence_eps_param"){ decoherence_eps_param = bp::extract<double>(params.values()[i]); }
    else if(key=="dephasing_informed"){ dephasing_informed = bp::extract<int>(params.values()[i]); }
    else if(key=="instantaneous_decoherence_variant"){ instantaneous_decoherence_variant = bp::extract<int>(params.values()[i]); }
    else if(key=="collapse_option"){ collapse_option = bp::extract<int>(params.values()[i]); }
    else if(key=="decoherence_rates"){ 
      MATRIX x( bp::extract<MATRIX>(params.values()[i]) );
      decoherence_rates = new MATRIX(x.n_rows, x.n_cols);      
      for(int a=0;a<x.n_rows;a++){
        for(int b=0;b<x.n_cols;b++){ decoherence_rates->set(a, b, x.get(a,b));   }
      } 
    }
    else if(key=="ave_gaps"){ 
      MATRIX x( bp::extract<MATRIX>(params.values()[i]) );
      ave_gaps = new MATRIX(x.n_rows, x.n_cols);      
      for(int a=0;a<x.n_rows;a++){
        for(int b=0;b<x.n_cols;b++){ ave_gaps->set(a, b, x.get(a,b));   }
      } 
    }
    else if(key=="wp_width"){ 
      MATRIX x( bp::extract<MATRIX>(params.values()[i]) );
      wp_width = new MATRIX(x.n_rows, x.n_cols);      
      for(int a=0;a<x.n_rows;a++){
        for(int b=0;b<x.n_cols;b++){ wp_width->set(a, b, x.get(a,b));   }
      } 
    }
    else if(key=="wp_v"){ 
      MATRIX x( bp::extract<MATRIX>(params.values()[i]) );
      wp_v = new MATRIX(x.n_rows, x.n_cols);      
      for(int a=0;a<x.n_rows;a++){
        for(int b=0;b<x.n_cols;b++){ wp_v->set(a, b, x.get(a,b));   }
      } 
    }
    else if(key=="coherence_threshold"){ coherence_threshold = bp::extract<double>(params.values()[i]); }
    else if(key=="e_mask"){ e_mask = bp::extract<double>(params.values()[i]); }
    else if(key=="use_xf_force"){ use_xf_force = bp::extract<int>(params.values()[i]); }
    else if(key=="project_out_aux"){ project_out_aux = bp::extract<int>(params.values()[i]); }
    else if(key=="tp_algo"){ tp_algo = bp::extract<int>(params.values()[i]); }
    else if(key=="use_td_width"){ use_td_width = bp::extract<int>(params.values()[i]); }

    else if(key=="gap_correlation_time"){ gap_correlation_time = bp::extract<double>(params.values()[i]); }

    ///================= Entanglement of trajectories ================================
    else if(key=="entanglement_opt"){ entanglement_opt = bp::extract<int>(params.values()[i]); }
    else if(key=="ETHD3_alpha") { ETHD3_alpha = bp::extract<double>(params.values()[i]);   }
    else if(key=="ETHD3_beta") { ETHD3_beta = bp::extract<double>(params.values()[i]);   }

    ///================= Entanglement of trajectories ================================
    else if(key=="qtag_pot_approx_method"){ qtag_pot_approx_method = bp::extract<int>(params.values()[i]); }
    
    ///================= Bath, Constraints, and Dynamical controls ===================
    else if(key=="Temperature") { Temperature = bp::extract<double>(params.values()[i]);  }
    else if(key=="ensemble"){ ensemble = bp::extract<int>(params.values()[i]); }    
    else if(key=="thermostat_params"){ thermostat_params = bp::extract<bp::dict>(params.values()[i]); }    
    else if(key=="thermostat_dofs"){  
      thermostat_dofs.clear();
      boost::python::list tmp = extract<boost::python::list>(params.values()[i]);
      for(int j=0; j<len(tmp); j++){  thermostat_dofs.push_back( extract<double>(tmp[j]) );  }
    }
    else if(key=="quantum_dofs"){  
      quantum_dofs.clear();
      boost::python::list tmp = extract<boost::python::list>(params.values()[i]);
      for(int j=0; j<len(tmp); j++){  quantum_dofs.push_back( extract<double>(tmp[j]) );  }
    }
    else if(key=="constrained_dofs"){  
      constrained_dofs.clear();
      boost::python::list tmp = extract<boost::python::list>(params.values()[i]);
      for(int j=0; j<len(tmp); j++){  constrained_dofs.push_back( extract<double>(tmp[j]) );  }
    }
    else if(key=="dt") { dt = bp::extract<double>(params.values()[i]);  }
    else if(key=="num_electronic_substeps") { num_electronic_substeps = bp::extract<int>(params.values()[i]);  }
    else if(key=="electronic_integrator"){ electronic_integrator = bp::extract<int>(params.values()[i]); }
    else if(key=="ampl_transformation_method"){ ampl_transformation_method = bp::extract<int>(params.values()[i]); }
    else if(key=="assume_always_consistent"){  assume_always_consistent = bp::extract<int>(params.values()[i]); }

    else if(key=="thermally_corrected_nbra"){ thermally_corrected_nbra = bp::extract<int>(params.values()[i]); }
    else if(key=="total_energy") { total_energy = bp::extract<double>(params.values()[i]);  }
    else if(key=="tcnbra_nu_therm") { tcnbra_nu_therm =  bp::extract<double>(params.values()[i]); }
    else if(key=="tcnbra_nhc_size") { tcnbra_nhc_size =  bp::extract<int>(params.values()[i]); }
    else if(key=="tcnbra_do_nac_scaling") { tcnbra_do_nac_scaling =  bp::extract<int>(params.values()[i]); }
    
    else if(key=="properties_to_save"){ properties_to_save = bp::extract<bp::list>(params.values()[i]); }    

    else if(key=="reorg_energy") { reorg_energy =  bp::extract<double>(params.values()[i]); }

  }// for i

  sanity_check();

}


}// namespace libdyn
}// liblibra

