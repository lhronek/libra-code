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
  \file nHamiltonian.h
  \brief The file describes the generic Hamiltonian class
    
*/

#ifndef nHAMILTONIAN_H
#define nHAMILTONIAN_H

#if defined(USING_PCH)
#include "../pch.h"
#else
#include <complex>
#endif 

#include "../math_linalg/liblinalg.h"

/// liblibra namespace
namespace liblibra{


/// libnhamiltonian namespace
namespace libnhamiltonian{


using namespace liblinalg;

void check_mat_dimensions(CMATRIX* x, int nrows, int ncols);
void check_vector_dimensions(vector<CMATRIX*> ptx, vector<CMATRIX>& x_, vector<int>& x_mem_status, int nnucl);


void set_X1_by_ref(CMATRIX* ptx, CMATRIX& x_, int& x_mem_status, int nrows, int ncols);
void set_X1_by_val(CMATRIX* ptx, CMATRIX& x_, int& x_mem_status, int nrows, int ncols);
void init_X1(CMATRIX* ptx, int& x_mem_status, int nrows, int ncols);

void set_X2_by_ref(vector<CMATRIX*> ptx, vector<CMATRIX>& x_, vector<int>& x_mem_status, int nrows, int ncols, int nnucl);
void set_X2_by_val(vector<CMATRIX*> ptx, vector<CMATRIX>& x_, vector<int>& x_mem_status, int nrows, int ncols, int nnucl);
void init_X2(vector<CMATRIX*> ptx, vector<int>& x_mem_status, int nrows, int ncols, int nnucl);



// Forward declaration
//class nHamiltonian; 


class nHamiltonian{
/**
  Keep the pointers to the objects (memory) or allocate the memory on demand

  For each variable X (except for the very trivial cases), we'll have a variable "X_mem_status", which 
  which describes the status of the memory for that variable:

  0 - not allocated,
  1 - allocated internally
  2 - allocated externally

*/


  void check_cmatrix(bp::object obj, std::string matrix_name, int nrows, int ncols);
  void check_cmatrix_list(bp::object obj, std::string matrix_name, int nrows, int ncols, int length);

  void add_branches(int target_level, vector<nHamiltonian*>& res);

public:

  int level;                        ///< level in the tree hierarchy
  int id;                           ///< index of this Hamiltonian in this level of hierarchy
  nHamiltonian* parent;             ///< the Hamiltonian of a higher level
  vector<nHamiltonian*> children;   ///< the Hamiltonians of the lower level
  vector<int> children_owned;       ///< per child: 1 - allocated by add_new_children and deleted by this node's destructor, 0 - an
                                    ///< external child of add_child, the caller's (detached, not deleted)


  int nnucl;                 ///< number of nuclear degrees of freedom - expected

  /** Diabatic representation

  |Psi> = |psi_dia> C_dia

  |psi_dia> = (psi_dia_0, psi_dia_1, ... psi_dia_{ndia-1})

  ovlp_dia = <psi_dia|psi_dia>

  dc1_dia[n] = <psi_dia| d/dq_n |psi_dia>

  ham_dia = <psi_dia|H|psi_dia>     - non-diagonal matrix

  nac_dia = <psi_dia|d/dt|psi_dia>

  hvib_dia = ham_dia - ihbar * nac_dia

  d1ham_dia[n] = d/dq_n [<psi_dia|H|psi_dia>]

  d2ham_dia[n*nnucl+k] = d^2/(dq_n dq_k)[ <psi_dia|H|psi_dia>]

  density_matrix_operator = |psi_dia> den_mat_dia <psi_dia| 

  So: den_mat_dia - density matrix matrix in the diabatic basis

  */
  int ndia;                   ///< the number of electronic DOFs in the diabatic basis

//  CMATRIX* ampl_dia;          ///< Amplitudes of the diabatic states in the overal wavefunction
//  int ampl_dia_mem_status;

  CMATRIX* ovlp_dia;          ///< the overlap of the diabatic basis 
  int ovlp_dia_mem_status;       

  vector<CMATRIX*> dc1_dia;   ///< first-order derivative coupling matrices in the diabatic basis 
  vector<int> dc1_dia_mem_status;


  CMATRIX* ham_dia;           ///< Hamiltonian in diabatic representation
  int ham_dia_mem_status;

  CMATRIX* nac_dia;           ///< Nonadiabatic couplings (time-derivative couplings) in diabatic representation
  int nac_dia_mem_status;

  CMATRIX* hvib_dia;          ///< Vibronic Hamiltonian in diabatic representation
  int hvib_dia_mem_status;




  vector<CMATRIX*> d1ham_dia; ///< first order derivatives of the Hamiltonian matrix in the diabatic basis
  vector<int> d1ham_dia_mem_status;

  vector<CMATRIX*> d2ham_dia; ///< second order derivatives of the Hamiltonian in the diabatic basis
  vector<int> d2ham_dia_mem_status;

//  CMATRIX* den_mat_dia;       ///< Density matrix in the diabatic basis
//  int den_mat_dia_mem_status;


  /** Adiabatic representation: by definition, the states are orthonormal in this representation

  |Psi> = |psi_adi> C_adi

  |psi_adi> = (psi_adi_0, psi_adi_1, ... psi_adi_{nadi-1})

  dc1_adi[n] = <psi_adi| d/dq_n |psi_adi>

  ham_adi = <psi_adi|H|psi_adi>   - diagonal matrix

  nac_adi = <psi_adi|d/dt|psi_adi>

  hvib_adi = ham_adi - ihbar * nac_adi

  d1ham_adi[n] = d/dq_n [<psi_adi|H|psi_adi>]

  d2ham_adi[n*nnucl+k] =d^2/(dq_n dq_k)[ <psia_adi|H|psi_adi> ]

  density_matrix_operator = |psi_adi> den_mat_adi <psi_adi| 
  So: den_mat_adi - density matrix matrix in the adiabatic basis

  */
  int nadi;                   ///< the number of electronic DOFs in the adiabatic basis


  vector<CMATRIX*> dc1_adi;   ///< first-order derivative coupling matrices in the adiabatic basis 
  vector<int> dc1_adi_mem_status;

  CMATRIX* ham_adi;           ///< Hamiltonian in adiabatic representation (diagonal)
  int ham_adi_mem_status;


  CMATRIX* nac_adi;           ///< Nonadiabatic couplings (time-derivative couplings) in adiabatic representation
  int nac_adi_mem_status;

  CMATRIX* hvib_adi;          ///< Vibronic Hamiltonian in adiabatic representation
  int hvib_adi_mem_status;
  ///< whether the model object supplied hvib_adi at the last ingestion -
  ///< hvib_update_method 1 must not silently overwrite it
  int hvib_adi_supplied;
  int hvib_dia_supplied;
  int nac_adi_supplied;
  int nac_dia_supplied;
  ///< whether the model object carried the attribute at the last ingestion, zeros included - the
  ///< consume-side contract reads presence (an option that reads a model-supplied quantity refuses
  ///< a model that hands over none of it), the overwrite guards above read data
  int ham_dia_present;
  int ham_adi_present;
  int nac_adi_present;
  int hvib_adi_present;
  int hvib_dia_present;
  int time_overlap_adi_present;
  int d1ham_adi_present;
  int d1ham_dia_present;
  int dc1_adi_present;
  ///< whether the model object ever supplied ovlp_dia - a child never given a metric holds the
  ///< identity of its allocation, which the dynamical variables need not scan
  int ovlp_dia_supplied;
  ///< the Lowdin frame of ovlp_dia, memoized by the metric's elements (lowdin_cache_S): X = S^-1/2, Y = S^1/2 and,
  ///< for the frame's derivative, the eigenpairs S = V diag(s) V^H - filled on first use (lowdin_cache_ready), refilled
  ///< when the elements change, not copied with the content (a copy refills from its own metric). The state: -1
  ///< unknown, 0 identity metric, 1 frame set - lowdin_cache_eig: the eigenpairs set
  int lowdin_cache_state;
  int lowdin_cache_eig;
  CMATRIX lowdin_cache_S;
  CMATRIX lowdin_cache_X;
  CMATRIX lowdin_cache_Y;
  CMATRIX lowdin_cache_V;
  vector<double> lowdin_cache_s;
  ///< the model supplied its own dia->adi transformation - the
  ///< stored-U transforms (ham_transform_method 2) require it: the
  ///< square identity default must not silently alias dia == adi
  int basis_transform_supplied;
  ///< the tracking group of every diabatic state the model handed over with its basis_transform (the same label =
  ///< one group, a negative label a singleton) - the engine's dia_tracking_algo 5 aligns the groups, sized ndia
  vector<int> dia_groups;
  int dia_groups_supplied;
  ///< the model supplied obj.ham_dia (the diabatic-hop/LZ route) -
  ///< the adi->dia transforms would overwrite a DIFFERENT basis
  ///< (model diabats vs the transform frame): conflict, raise
  int ham_dia_supplied;


  vector<CMATRIX*> d1ham_adi; ///< first order derivatives of the Hamiltonian matrix in the adiabatic basis (diagonal)
  vector<int> d1ham_adi_mem_status;

  vector<CMATRIX*> d2ham_adi; ///< second order derivatives of the Hamiltonian matrix in the adiabatic basis
  vector<int> d2ham_adi_mem_status;



  /** Transformations
  
  |psi_adi> = |psi_dia> * U

  Stationary SE:  H |psi_adi> =  |psi_adi> * E

  project on <psi_adi|, keeping in mind that <psi_adi|psi_adi> = I

  <psi_dia| H |psi_adi> =  <psi_dia|psi_adi> E  

  <psi_dia| H |psi_dia> * U =  <psi_dia|psi_dia> * U * E  

  H_dia * U = ovlp_dia * U * H_adi

  Here, U = basis_transform

  Conversion of the density matrices:

  |psi_dia> * U * den_mat_adi * U.H() * <psi_dia| = |psi_dia> * den_mat_dia * <psi_dia|

  so:    U * den_mat_adi * U.H() = den_mat_dia
    
  */

  CMATRIX* basis_transform;  ///< These are the eigenvectors of the generalized eigenvalue problem for diabatic Hamiltonian
  int basis_transform_mem_status;

  CMATRIX* time_overlap_adi;  ///< These are <psi_i_adi (t) | psi_j_adi (t+dt)> matrices
  int time_overlap_adi_mem_status;
  ///< whether the model object supplied time_overlap_adi at the last
  ///< ingestion - time_overlap_method 1 must not silently overwrite it
  int time_overlap_adi_supplied;

  void require_children(int lvl, const char* fn) const; ///< fail-loud for deeper-level requests on childless nodes

  CMATRIX* time_overlap_dia;  ///< These are <psi_i_dia (t) | psi_j_dia (t+dt)> matrices
  int time_overlap_dia_mem_status;


  vector<int>* ordering_adi;  ///< the permutation describing how the indices of the internally-stored
                              /// properties like ham_adi or nac_adi relate to the indices of the 
                              /// actual "physical" states.

  CMATRIX* cum_phase_corr;    /// These are the cumulative phase corrections applied to basis_transform (this correction is indexed by
                              /// the inernal indexing of states, not physical)
                              /// Keep in mind that these phases have already been applied to basis_transform variable
                              /// (this is the convention)
  int cum_phase_corr_mem_status;


  /**     
     All the computation control parameters 
  **/
  int eigen_algo;  ///< eigensolver algorithm used to convert diabatic to adiabatic basis 
                   /// 0 - generalized eigensolver (may reorder states) [default]
                   /// 1 - eigensolver without reordering (assumes diabatic basis is orthonormaly)

  double phase_corr_ovlp_tol;  /// phase correction overlap tolerance, we only compute phase corrections 
                               /// if the time overlaps are above (in magnitude) this value, otherwise
                               /// we assume that uncommon state reordering (adiabatic state switching) 
                               /// has happened (which may occur in the stochastic state tracking method)
                               /// then, we don't apply the phase correction since it doesn't make sense



  double gs_kinetic_energy;
  /**
     
     All the basic methods: constructor, destructor, getters, setters, etc.

  **/
  ///< In nHamiltonian_basic.cpp

  void init_mem_status(int ndia_, int nadi_, int nnucl_);

  ///< Constructors
//  nHamiltonian(); 
  nHamiltonian(int ndia_, int nadi_, int nnucl_); 

  ///< Copy Constructor:
  nHamiltonian(const nHamiltonian&); 
  nHamiltonian(const nHamiltonian& src, int aux_mode);
 
  ///< Destructor
  ~nHamiltonian();    

  ///< Tree management utilities
  void set_levels(int lvl_);            ///< Updates node levels downstream
  void add_new_children(int _ndia, int _nadi, int _nnucl, int nchildren);
  nHamiltonian* child(int traj){
    ///< Per-trajectory child access with the NBRA clamp: a single
    ///< shared child (isNBRA==1) serves every trajectory, so an index
    ///< beyond the allocated children folds to 0 instead of reading
    ///< past the vector
    if(children.empty() || traj < 0){
      std::cout<<"Error in nHamiltonian::child(): trajectory "<<traj
          <<" requested on a node with "<<children.size()
          <<" children - the dynamics needs a parent hierarchy \n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }
    if(traj >= (int)children.size()){ traj = 0; }
    return children[traj];
  }
  void add_child(nHamiltonian& child);  ///< Associate an existing Hamiltonian with the present one
                                        ///< to become its child
  vector<int> get_full_id();            ///< Entire path of the note in the tree
  vector<nHamiltonian*> get_branches(int target_level);


  void copy_content(const nHamiltonian& src);
  void copy_content(nHamiltonian* src);
  void copy_level_content(nHamiltonian* src); ///< To copy content if all the memory is allocated

  void init_all(int der_lvl);
  void init_all(int der_lvl, int lvl);
  void init_mirror(const nHamiltonian& src);
  void init_adi_derivatives(int lvl);
  void init_aux(int lvl);

  void show_memory_status(vector<int>& id_);


  ///< Setters:
  // Diabatic
  void init_ovlp_dia();
  void set_ovlp_dia_by_ref(CMATRIX& ovlp_dia_);
  void set_ovlp_dia_by_val(CMATRIX& ovlp_dia_);

  void init_dc1_dia();
  void set_dc1_dia_by_ref(vector<CMATRIX>& dc1_dia_);
  void set_dc1_dia_by_val(vector<CMATRIX>& dc1_dia_);

  void init_ham_dia();
  void set_ham_dia_by_ref(CMATRIX& ham_dia_);
  void set_ham_dia_by_val(CMATRIX& ham_dia_);

  void init_nac_dia();
  void set_nac_dia_by_ref(CMATRIX& nac_dia_);
  void set_nac_dia_by_val(CMATRIX& nac_dia_);

  void init_hvib_dia();
  void set_hvib_dia_by_ref(CMATRIX& hvib_dia_);
  void set_hvib_dia_by_val(CMATRIX& hvib_dia_);

  void init_d1ham_dia();
  void set_d1ham_dia_by_ref(vector<CMATRIX>& d1ham_dia_);
  void set_d1ham_dia_by_val(vector<CMATRIX>& d1ham_dia_);

  void init_d2ham_dia();
  void set_d2ham_dia_by_ref(vector<CMATRIX>& d2ham_dia_);
  void set_d2ham_dia_by_val(vector<CMATRIX>& d2ham_dia_);




  // Adiabatic
  void init_dc1_adi();
  void set_dc1_adi_by_ref(vector<CMATRIX>& dc1_adi_);
  void set_dc1_adi_by_val(vector<CMATRIX>& dc1_adi_);

  void init_ham_adi();
  void set_ham_adi_by_ref(CMATRIX& ham_adi_);
  void set_ham_adi_by_val(CMATRIX& ham_adi_);

  void init_nac_adi();
  void set_nac_adi_by_ref(CMATRIX& nac_adi_);
  void set_nac_adi_by_val(CMATRIX& nac_adi_);

  void init_hvib_adi();
  void set_hvib_adi_by_ref(CMATRIX& hvib_adi_);
  void set_hvib_adi_by_val(CMATRIX& hvib_adi_);

  void init_d1ham_adi();
  void set_d1ham_adi_by_ref(vector<CMATRIX>& d1ham_adi_);
  void set_d1ham_adi_by_val(vector<CMATRIX>& d1ham_adi_);

  void init_d2ham_adi();
  void set_d2ham_adi_by_ref(vector<CMATRIX>& d2ham_adi_);
  void set_d2ham_adi_by_val(vector<CMATRIX>& d2ham_adi_);

  // Transforms
  void init_basis_transform();
  void set_basis_transform_by_ref(CMATRIX& basis_transform_);
  void set_basis_transform_by_val(CMATRIX& basis_transform_);

  void init_time_overlap_adi();
  void set_time_overlap_adi_by_ref(CMATRIX& time_overlap_adi_);
  void set_time_overlap_adi_by_val(CMATRIX& time_overlap_adi_);

  void init_time_overlap_dia();
  void set_time_overlap_dia_by_ref(CMATRIX& time_overlap_dia_);
  void set_time_overlap_dia_by_val(CMATRIX& time_overlap_dia_);


  void set_ordering_adi_by_ref(vector<int>& ordering_adi_);
  void set_ordering_adi_by_val(vector<int>& ordering_adi_);

  void init_cum_phase_corr();
  void set_cum_phase_corr_by_ref(CMATRIX& cum_phase_corr_);
  void set_cum_phase_corr_by_val(CMATRIX& cum_phase_corr_);


  /// Getters
  // Diabatic
  CMATRIX get_ovlp_dia();
  CMATRIX get_ovlp_dia(vector<int>& id_);
  CMATRIX get_dc1_dia(int i);
  CMATRIX get_dc1_dia(int i, vector<int>& id_);
  CMATRIX get_ham_dia();
  CMATRIX get_ham_dia(vector<int>& id_);
  CMATRIX get_nac_dia();
  CMATRIX get_nac_dia(vector<int>& id_);
  CMATRIX get_hvib_dia();
  CMATRIX get_hvib_dia(vector<int>& id_);
  CMATRIX get_d1ham_dia(int i);
  CMATRIX get_d1ham_dia(int i, vector<int>& id_);
  int lowdin_dia(CMATRIX& X, CMATRIX& Y);       ///< Lowdin frame of the diabats: X = S^-1/2, Y = S^1/2 of ovlp_dia. Returns 0 (X, Y untouched) for an absent or identity metric
  int lowdin_cache_ready();                    ///< the frame of the current ovlp_dia in the memo, refilled when the metric's elements changed - its state (0 absent or identity, 1 set)
  void lowdin_cache_eigenpairs();              ///< the memoized metric's eigenpairs for the frame's derivative, once per refill
  CMATRIX get_ham_dia_lowdin();                  ///< X ham_dia X - the diabatic Hamiltonian in the Lowdin frame (ham_dia itself at S = I)
  int lowdin_dia_derivative(int n, CMATRIX& X, CMATRIX& Y, CMATRIX& dX, CMATRIX& dY);  ///< the Lowdin frame and its derivative along nuclear DOF n from dS = dc1_dia[n] + dc1_dia[n]^H. Returns the lowdin_dia status, dX = dY = 0 for a constant metric
  CMATRIX get_hvib_dia_lowdin(MATRIX& p, const MATRIX& invM);   ///< X hvib_dia X + i (dY/dt) X - the vibronic generator of the Lowdin frame, p and invM (nnucl x 1)
  CMATRIX get_d1ham_dia_lowdin(int i);           ///< the total derivative of the Lowdin-frame Hamiltonian X ham_dia X along nuclear DOF i
  CMATRIX get_d2ham_dia(int i);
  CMATRIX get_d2ham_dia(int i, vector<int>& id_);
  CMATRIX get_d2ham_dia(int i,int j);
  CMATRIX get_d2ham_dia(int i, int j, vector<int>& id_);

  // Adiabatic
  CMATRIX get_dc1_adi(int i);
  CMATRIX get_dc1_adi(int i, vector<int>& id_);
  CMATRIX get_ham_adi();
  const CMATRIX& get_ham_adi_ref() const;   ///< zero-copy access
  const CMATRIX& get_hvib_adi_ref() const;  ///< zero-copy access
  CMATRIX get_ham_adi(vector<int>& id_);
  CMATRIX get_nac_adi();
  CMATRIX get_nac_adi(vector<int>& id_);
  CMATRIX get_hvib_adi();
  CMATRIX get_hvib_adi(vector<int>& id_);
  CMATRIX get_d1ham_adi(int i);
  CMATRIX get_d1ham_adi(int i, vector<int>& id_);
  CMATRIX get_d2ham_adi(int i);
  CMATRIX get_d2ham_adi(int i, vector<int>& id_);
  CMATRIX get_d2ham_adi(int i,int j);
  CMATRIX get_d2ham_adi(int i, int j, vector<int>& id_);

  // Transforms
  CMATRIX get_basis_transform();
  vector<int> get_dia_groups();
  CMATRIX get_basis_transform(vector<int>& id_);

  CMATRIX get_time_overlap_adi();
  CMATRIX get_time_overlap_adi(vector<int>& id_);

  CMATRIX get_time_overlap_dia();
  CMATRIX get_time_overlap_dia(vector<int>& id_);

  vector<int> get_ordering_adi(); 
  vector<int> get_ordering_adi(vector<int>& id_); 

  CMATRIX get_cum_phase_corr();
  CMATRIX get_cum_phase_corr(vector<int>& id_);


  void transform_all(CMATRIX* T, int option);
  void transform_all(vector<CMATRIX*>& T, int option);

  void transform_basis(CMATRIX* T, int option);
  void transform_basis(vector<CMATRIX*>& T, int option);

  ///< In nHamiltonian_compute_diabatic.cpp

  void compute_diabatic(int model, vector<double>& q, vector<double>& params, int lvl); // for internal model types
  void compute_diabatic(int model, vector<double>& q, vector<double>& params); // for internal model types

//  void compute_diabatic(bp::object py_funct, bp::object q, bp::object params, int lvl); // for models defined in Python
//  void compute_diabatic(bp::object py_funct, bp::object q, bp::object params); // for models defined in Python
  void compute_diabatic(bp::object py_funct, MATRIX& q, bp::object params, int lvl); // for models defined in Python
  void compute_diabatic(bp::object py_funct, MATRIX& q, bp::object params); // for models defined in Python


  ///< In nHamiltonian_compute_ETHD.cpp

  void add_ethd_dia(const MATRIX& q, const MATRIX& invM, int der_lvl);
  void add_ethd_adi(const MATRIX& q, const MATRIX& invM, int der_lvl);

  void add_ethd3_dia(const MATRIX& q, const MATRIX& invM, double alp, int der_lvl);
  void add_ethd3_adi(const MATRIX& q, const MATRIX& invM, double alp, int der_lvl);
  void add_ethd3_adi(const MATRIX& q, const MATRIX& p, const MATRIX& invM, double alp, double bet, int der_lvl);




  ///< In nHamiltonian_compute_adiabatic.cpp
  void update_ordering(vector<int>& perm_t, int lvl);
  void update_ordering(vector<int>& perm_t);

/*
  void apply_phase_corrections(CMATRIX* phase_corr, int lvl);
  void apply_phase_corrections(CMATRIX& phase_corr, int lvl);
  void apply_phase_corrections(CMATRIX* phase_corr);
  void apply_phase_corrections(CMATRIX& phase_corr);

  CMATRIX update_phases(CMATRIX& U_prev, int lvl);
  CMATRIX update_phases(CMATRIX& U_prev);
*/

  void compute_adiabatic(int der_lvl, int lvl);
  void transform_adi2dia(int lvl);
  void transform_dia2adi(int der_lvl, int lvl);
  void compute_adiabatic_derivatives();
  void compute_adiabatic(int der_lvl);
//  void compute_adiabatic(bp::object py_funct, bp::object q, bp::object params, int lvl); // for models defined in Python
//  void compute_adiabatic(bp::object py_funct, bp::object q, bp::object params); // for models defined in Python
  void compute_adiabatic(bp::object py_funct, MATRIX& q, bp::object params, int lvl); // for models defined in Python
  void compute_adiabatic(bp::object py_funct, MATRIX& q, bp::object params); // for models defined in Python



  ///< In nHamiltonian_compute_basis_transform.cpp

  void ampl_dia2adi(CMATRIX& ampl_dia, CMATRIX& ampl_adi);
  void ampl_dia2adi(CMATRIX& ampl_dia, CMATRIX& ampl_adi, vector<int>& id_);
  void ampl_dia2adi(CMATRIX& ampl_dia, CMATRIX& ampl_adi, int lvl, int split);
  void ampl_adi2dia(CMATRIX& ampl_dia, CMATRIX& ampl_adi);
  void ampl_adi2dia(CMATRIX& ampl_dia, CMATRIX& ampl_adi, vector<int>& id_);
  void ampl_adi2dia(CMATRIX& ampl_dia, CMATRIX& ampl_adi, int lvl, int split);

  void ampl_dia2adi(CMATRIX* ampl_dia, CMATRIX* ampl_adi);
  void ampl_dia2adi(CMATRIX* ampl_dia, CMATRIX* ampl_adi, vector<int>& id_);
  void ampl_dia2adi(CMATRIX* ampl_dia, CMATRIX* ampl_adi, int lvl, int split);
  void ampl_adi2dia(CMATRIX* ampl_dia, CMATRIX* ampl_adi);
  void ampl_adi2dia(CMATRIX* ampl_dia, CMATRIX* ampl_adi, vector<int>& id_);
  void ampl_adi2dia(CMATRIX* ampl_dia, CMATRIX* ampl_adi, int lvl, int split);




  ///< In nHamiltonian_compute_forces.cpp

  CMATRIX forces_adi(vector<int>& act_states);   // -dH_adi/dR in the adiabatic basis for several trajectories
  CMATRIX forces_dia(vector<int>& act_states);   // -dH_dia/dR in the diabatic basis for several trajectories

  CMATRIX all_forces_adi(vector<int>& id_);
  CMATRIX all_forces_dia(vector<int>& id_);

  
/*
  These variables are poorly-defined and unnecessary, so we are going to deprecate them, AVA 9/15/2022
  Instead - there will be functions to extract adiabatic and diabatic forces


  vector<CMATRIX> forces_tens_adi(CMATRIX& ampl_adi); // 
  vector<CMATRIX> forces_tens_adi(CMATRIX& ampl_adi, vector<int>& id_); // 
  vector<CMATRIX> forces_tens_dia(CMATRIX& ampl_dia); // 
  vector<CMATRIX> forces_tens_dia(CMATRIX& ampl_dia, vector<int>& id_); // 

  CMATRIX forces_adi(CMATRIX& ampl_adi);  // -dE/dR in the adiabatic basis, assuming Cadi = Cadi(t)
  CMATRIX forces_adi(CMATRIX& ampl_adi, vector<int>& id_);  // -dE/dR in the adiabatic basis, assuming Cadi = Cadi(t)
  CMATRIX forces_adi(vector<int>& act_states);   // -dE/dR in the adiabatic basis for several trajectories
  CMATRIX forces_adi(int act_state);   // -dE/dR in the adiabatic basis for several trajectories, all in the same state

  CMATRIX forces_dia(CMATRIX& ampl_dia);  // -dE/dR in the diabatic basis, assuming Cdia = Cdia(t)
  CMATRIX forces_dia(CMATRIX& ampl_dia, vector<int>& id_);  // -dE/dR in the diabatic basis, assuming Cdia = Cdia(t)
  CMATRIX forces_dia(vector<int>& act_states);   // -dE/dR in the diabatic basis for several trajectories
  CMATRIX forces_dia(int act_state);   // -dE/dR in the diabatic basis for several trajectories, all in the same state
*/


  ///< In nHamiltonian_compute_nac.cpp

  void compute_nac_dia(MATRIX& p, const MATRIX& invM);
  void compute_nac_dia(MATRIX& p, const MATRIX& invM, vector<int>& id_);
  void compute_nac_dia(MATRIX& p, const MATRIX& invM, int lvl, int split);
  void compute_nac_adi(double dt, int method);
  void compute_nac_adi(MATRIX& p, const MATRIX& invM);
  void compute_nac_adi(MATRIX& p, const MATRIX& invM, vector<int>& id_);
  void compute_nac_adi(MATRIX& p, const MATRIX& invM, int lvl, int split);

  void compute_hvib_dia();
  void compute_hvib_dia(vector<int>& id_);
  void compute_hvib_dia(int lvl);
  void compute_hvib_adi();
  void compute_hvib_adi(vector<int>& id_);
  void compute_hvib_adi(int lvl);




  ///< In nHamiltonian_compute_Ehrenfest.cpp and nHamiltonian_compute_Ehrenfest_forces.cpp
  complex<double> Ehrenfest_energy_dia(CMATRIX& ampl_dia);
  complex<double> Ehrenfest_energy_dia(CMATRIX& ampl_dia, vector<int>& id_);
  CMATRIX Ehrenfest_forces_dia_unit(CMATRIX& ampl_dia, int option, double gamma, int normalized);  ///< Ehrenfest forces in diabatic basis, gamma term, normalized 0 = un-normalized Meyer-Miller form
  CMATRIX Ehrenfest_forces_dia_unit(CMATRIX& ampl_dia, int option, double gamma);  ///< Ehrenfest forces in diabatic basis, with the Meyer-Miller gamma term
  CMATRIX Ehrenfest_forces_dia_unit(CMATRIX& ampl_dia, int option);      ///< Ehrenfest forces in diabatic basis
  CMATRIX Ehrenfest_forces_dia_unit(CMATRIX& ampl_dia);                  ///< Ehrenfest forces in diabatic basis
  CMATRIX Ehrenfest_forces_dia(CMATRIX& ampl_dia, int lvl, int option, double gamma, int normalized);  ///< Ehrenfest forces in diabatic basis, uniform gamma, normalization switch
  CMATRIX Ehrenfest_forces_dia(CMATRIX& ampl_dia, int lvl, int option, double gamma);  ///< Ehrenfest forces in diabatic basis, uniform Meyer-Miller gamma
  CMATRIX Ehrenfest_forces_dia(CMATRIX& ampl_dia, int lvl, int option);  ///< Ehrenfest forces in diabatic basis
  CMATRIX Ehrenfest_forces_dia(CMATRIX& ampl_dia, int lvl);              ///< Ehrenfest forces in diabatic basis


  ///< In nHamiltonian_compute_Ehrenfest.cpp and nHamiltonian_compute_Ehrenfest_forces.cpp
  ///< The adiabatic Ehrenfest and QTSH energies and forces take the amplitudes (densities) in the basis of the current
  ///< ham_adi: the step's projector moves them in the propagation (update_active_states) and no transform is applied here -
  ///< the former transform argument was the identity since upstream 85ff75d0 and is gone
  complex<double> Ehrenfest_energy_adi(CMATRIX& ampl_adi);
  complex<double> Ehrenfest_energy_adi(CMATRIX& ampl_adi, vector<int>& id_);
  CMATRIX Ehrenfest_forces_adi_unit(CMATRIX& ampl_adi, int option, double gamma, int normalized);  ///< Ehrenfest forces in adiabatic basis, normalized 0 = un-normalized Meyer-Miller form
  CMATRIX Ehrenfest_forces_adi_unit(CMATRIX& ampl_adi, int option, double gamma);  ///< Ehrenfest forces in adiabatic basis
  CMATRIX Ehrenfest_forces_adi_unit(CMATRIX& ampl_adi, int option);      ///< Ehrenfest forces in adiabatic basis
  CMATRIX Ehrenfest_forces_adi_unit(CMATRIX& ampl_adi);                  ///< Ehrenfest forces in adiabatic basis
  CMATRIX Ehrenfest_forces_adi(CMATRIX& ampl_adi, int lvl, int option, vector<double>& gammas, int normalized);  ///< Ehrenfest forces in adiabatic basis, per-trajectory gammas, normalization switch
  CMATRIX Ehrenfest_forces_adi(CMATRIX& ampl_adi, int lvl, int option, double gamma, int normalized);  ///< Ehrenfest forces in adiabatic basis, uniform gamma, normalization switch
  CMATRIX Ehrenfest_forces_adi(CMATRIX& ampl_adi, int lvl, int option, double gamma);  ///< Ehrenfest forces in adiabatic basis, uniform Meyer-Miller gamma
  CMATRIX Ehrenfest_forces_adi(CMATRIX& ampl_adi, int lvl, int option);  ///< Ehrenfest forces in adiabatic basis
  CMATRIX Ehrenfest_forces_adi(CMATRIX& ampl_adi, int lvl);              ///< Ehrenfest forces in adiabatic basis




  vector<CMATRIX> Ehrenfest_forces_tens_adi(CMATRIX& ampl_adi);  // Force tensor in adiabatic basis, assuming Cadi = Cadi(t)
  vector<CMATRIX> Ehrenfest_forces_tens_adi(CMATRIX& ampl_adi, vector<int>& id_);  // Force tensor in adiabatic basis, assuming Cadi = Cadi(t)
  vector<CMATRIX> Ehrenfest_forces_tens_dia(CMATRIX& ampl_dia);  // Force tensor in diabatic basis, assuming Cdia = Cdia(t)
  vector<CMATRIX> Ehrenfest_forces_tens_dia(CMATRIX& ampl_dia, vector<int>& id_);  // Force tensor in diabatic basis, assuming Cdia = Cdia(t)




  ///< In nHamiltonian_compute_QTSH.cpp and nHamiltonian_compute_QTSH_forces.cpp
  complex<double> QTSH_energy_dia(CMATRIX& ampl_dia);
  complex<double> QTSH_energy_dia(CMATRIX& ampl_dia, vector<int>& id_);
  complex<double> QTSH_energy_dia(CMATRIX& ampl_dia, MATRIX& p, const MATRIX& invM);  ///< the coherence energy in the Lowdin frame of the diabats (get_hvib_dia_lowdin), the frame of get_ham_dia_lowdin - p, invM (nnucl x 1)
  complex<double> QTSH_energy_dia(CMATRIX& ampl_dia, MATRIX& p, const MATRIX& invM, vector<int>& id_);
  CMATRIX QTSH_forces_dia_unit(CMATRIX& ampl_dia, int option);      ///< QTSH forces in diabatic basis
  CMATRIX QTSH_forces_dia_unit(CMATRIX& ampl_dia);                  ///< QTSH forces in diabatic basis
  CMATRIX QTSH_forces_dia(CMATRIX& ampl_dia, int lvl, int option);  ///< QTSH forces in diabatic basis
  CMATRIX QTSH_forces_dia(CMATRIX& ampl_dia, int lvl);              ///< QTSH forces in diabatic basis


  ///< In nHamiltonian_compute_QTSH.cpp and nHamiltonian_compute_QTSH_forces.cpp
  complex<double> QTSH_energy_adi(CMATRIX& ampl_adi);
  complex<double> QTSH_energy_adi(CMATRIX& ampl_adi, vector<int>& id_);
  CMATRIX QTSH_forces_adi_unit(CMATRIX& ampl_adi, int option);      ///< QTSH forces in adiabatic basis
  CMATRIX QTSH_forces_adi_unit(CMATRIX& ampl_adi);                  ///< QTSH forces in adiabatic basis
  CMATRIX QTSH_forces_adi(CMATRIX& ampl_adi, int lvl, int option);  ///< QTSH forces in adiabatic basis
  CMATRIX QTSH_forces_adi(CMATRIX& ampl_adi, int lvl);              ///< QTSH forces in adiabatic basis


///< The density-matrix forms of the Ehrenfest and QTSH energies and forces - the density-matrix representations
///< (rep_tdse 2 / 3) propagate the density matrix alone, their amplitudes keep the t0 values. Every amplitude
///< expression C^H A C / C^H S C becomes tr(A rho) / tr(S rho) for the plain-coefficient density rho = C C^H, so
///< the two forms agree for a pure state. In nHamiltonian_compute_Ehrenfest.cpp, _Ehrenfest_forces.cpp, _QTSH.cpp
///< and _QTSH_forces.cpp
  complex<double> Ehrenfest_energy_dia_dm(CMATRIX& dm_dia);
  complex<double> Ehrenfest_energy_dia_dm(CMATRIX& dm_dia, vector<int>& id_);
  complex<double> Ehrenfest_energy_adi_dm(CMATRIX& dm_adi);
  complex<double> Ehrenfest_energy_adi_dm(CMATRIX& dm_adi, vector<int>& id_);
  CMATRIX Ehrenfest_forces_dia_unit_dm(CMATRIX& dm_dia, int option, double gamma, int normalized);
  CMATRIX Ehrenfest_forces_dia_dm(vector<CMATRIX*>& dm_dia, int lvl, int option, double gamma, int normalized);
  CMATRIX Ehrenfest_forces_adi_unit_dm(CMATRIX& dm_adi, int option, double gamma, int normalized);
  CMATRIX Ehrenfest_forces_adi_dm(vector<CMATRIX*>& dm_adi, int lvl, int option, double gamma, int normalized);
  complex<double> QTSH_energy_dia_dm(CMATRIX& dm_dia);
  complex<double> QTSH_energy_dia_dm(CMATRIX& dm_dia, MATRIX& p, const MATRIX& invM);
  complex<double> QTSH_energy_dia_dm(CMATRIX& dm_dia, MATRIX& p, const MATRIX& invM, vector<int>& id_);
  CMATRIX QTSH_forces_dia_unit_dm(CMATRIX& dm_dia, int option);
  CMATRIX QTSH_forces_dia_dm(vector<CMATRIX*>& dm_dia, int lvl, int option);
  CMATRIX QTSH_forces_adi_unit_dm(CMATRIX& dm_adi, int option);
  CMATRIX QTSH_forces_adi_dm(vector<CMATRIX*>& dm_adi, int lvl, int option);


///< In nHamiltonian_compute_KCRPMD.cpp
  vector<MATRIX> generate_m_matrices(double beta);
  double kcrpmd_effective_potential(vector<double>& y_aux_var, const MATRIX& q, const MATRIX& invM, 
                                    double beta, double eta, double a, double b, double c);
  MATRIX kcrpmd_effective_force(vector<double>& y_aux_var, const MATRIX& q, const MATRIX& invM, 
                                    double beta, double eta, double a, double b, double c);
  vector<double> kcrpmd_effective_auxiliary_force(vector<double>& y_aux_var, 
                                    double beta, double eta, double a, double b, double c);






  friend bool operator == (const nHamiltonian& h1, const nHamiltonian& h2){
    return &h1 == &h2;
  }
  friend bool operator != (const nHamiltonian& h1, const nHamiltonian& h2){
    return !(h1 == h2);  // only compare addresses
  }



};

typedef std::vector<nHamiltonian> nHamiltonianList;  ///< data type for keeping a list of generic Hamiltonians of their derived classes


///< In nHamiltonian_compute adiabatic
/*

*/


///< In nHamiltonian_compute_ETHD.cpp
double ETHD_energy(const MATRIX& q, const MATRIX& invM);
MATRIX ETHD_forces(const MATRIX& q, const MATRIX& invM);


///< In nHamiltonian_compute_ETHD3.cpp
double ETHD3_energy(const MATRIX& q, const MATRIX& invM, double alp);
double ETHD3_energy(const MATRIX& q, const MATRIX& p, const MATRIX& invM, double alp, double bet);

MATRIX ETHD3_forces(const MATRIX& q, const MATRIX& invM, double alp);                                // -dH_{ETHD3}/dQ
MATRIX ETHD3_forces(const MATRIX& q, const MATRIX& p, const MATRIX& invM, double alp, double bet);   // -dH_{ETHD3}/dQ

MATRIX ETHD3_friction(const MATRIX& q, const MATRIX& p, const MATRIX& invM, double alp, double bet); // dH_{ETHD3}/dP


///< In nHamiltonian_compute_RPMD.cpp
double RPMD_internal_potential(const MATRIX& q, const MATRIX& invM, double beta);
MATRIX RPMD_internal_force(const MATRIX& q, const MATRIX& invM, double beta);




}// namespace libnhamiltonian
}// liblibra

#endif // nHAMILTONIAN_H
