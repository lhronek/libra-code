/*********************************************************************************
* Copyright (C) 2018-2022 Alexey V. Akimov
*
* This file is distributed under the terms of the GNU General Public License
* as published by the Free Software Foundation, either version 3 of
* the License, or (at your option) any later version.
* See the file LICENSE in the root directory of this distribution
* or <http://www.gnu.org/licenses/>.
*
*********************************************************************************/
/**
  \file nHamiltonian_compute_diabatic.cpp
  \brief The file implements calculations of the diabatic Hamiltonians
    
*/


#include <exception>
#if defined(USING_PCH)
#include "../pch.h"
#else
#include <stdlib.h>
#include <omp.h>
#endif 

#include "nHamiltonian.h"
//#include "../Hamiltonian_Model/libhamiltonian_model.h"
#include "../io/libio.h"

/// liblibra namespace
namespace liblibra{

/// libnhamiltonian namespace 
namespace libnhamiltonian{

///using namespace libhamiltonian_model;
using namespace libio;


namespace bp = boost::python;


bp::object import_py_funct(const std::string& module, const std::string& path, bp::object& globals)
{
    bp::dict locals;
    locals["module_name"] = module;
    locals["path"]        = path;

    bp::exec("import imp\n"
             "new_module = imp.load_module(module_name, open(path), path, ('py', 'U', imp.PY_SOURCE))\n",
             globals,
             locals);
    return locals["new_module"];
}


void nHamiltonian::compute_diabatic(int model, vector<double>& q, vector<double>& params){

  compute_diabatic(model, q, params, 0);

}

void nHamiltonian::compute_diabatic(int model, vector<double>& q, vector<double>& params, int lvl){

  if(level==lvl){
/* 
    if(model==100){   model_1S_1D_poly2(ham_dia, ovlp_dia, d1ham_dia, dc1_dia, q, params);  }
    if(model==100){   model_1S_1D_poly4(ham_dia, ovlp_dia, d1ham_dia, dc1_dia, q, params);  }
*/
  }
  else if(lvl>level){
    require_children(lvl, __func__);
  
    for(int i=0;i<children.size();i++){
      children[i]->compute_diabatic(model, q, params, lvl);
    }

  }

  else{
    cout<<"WARNING in nHamiltonian::compute_diabatic\n"; 
    cout<<"Can not run evaluation of function in the parent Hamiltonian from the\
     child node\n";    
  }



}


//void nHamiltonian::compute_diabatic(bp::object py_funct, bp::object q, bp::object params){
void nHamiltonian::compute_diabatic(bp::object py_funct, MATRIX& q, bp::object params){
/**
  Performs the diabatic properties calculation at the top-most level of the Hamiltonians 
  hierarchy. See the description of the more general function prototype for more info.
*/ 

  compute_diabatic(py_funct, q, params, 0);

}



//void nHamiltonian::compute_diabatic(bp::object py_funct, bp::object q, bp::object params, int lvl){
void nHamiltonian::compute_diabatic(bp::object py_funct, MATRIX& q, bp::object params, int lvl){
/**
  This function will call the <py_funct> function defined in Python and taking the signature:

  def py_funct(q, params, full_id)
      return obj

  The object <obj> returned by the function should contain variables named:
  "ham_dia" (CMATRIX), "ovld_dia" (CMATRIX), "d1ham_dia" (list of CMATRIX), "dc1_dia" (list of CMATRIX)
  of the specified types

  q - is the object containing coordinates of nuclei. Since almost any C++ type exposed to Python is seen 
  by Python as a Python object, you can use any convenient Libra type to store the coordinates, for instance
  a MATRIX object. 

  params - is an object containing any parameters needed by the <py_funct> Python function to perform
  the specified calculations. Again, there are many options to pass the parameters on the Python side, but 
  it may be convenient to use Python dictionary, since the parameters will be mostly used at the 
  Python level of calculations.

  full_id - is vector<int> object containing the "path" of the calling node in the hierarchy of Hamiltonians
  this information may be needed by the py_funct function in order to do the proper calculations

  lvl - is the level of the Hamiltonians in the hierarchy of Hamiltonians to be executed by this call 
  You can only request to perform computations at the same of higher (children) levels that the level of 
  the Hamiltonian from which you call this function. 

  Contentwise, this function computes the diabatic properties and populates the corresponding
  storage.

*/


  if(level==lvl){

    // Temporary storage

//    cout<<"nnucl = "<<nnucl<<endl;
//    q.show_matrix();
//exit(0);
    // Call the Python function with such arguments 
    bp::object obj = py_funct(q, params, get_full_id() );  
//exit(0);
    // Extract all the computed properties
    int has_attr=0;
    has_attr = (int)hasattr(obj,"ham_dia");        
    if(has_attr){  

      check_cmatrix(obj, "ham_dia", ndia, ndia);
      // a model may supply any family the hierarchy was not initialized
      // for - allocate on demand (the raw-pointer writes would otherwise
      // dereference NULL) - same policy for every ingested family below
      if(ham_dia_mem_status==0){ init_ham_dia(); }
      *ham_dia = extract<CMATRIX>(obj.attr("ham_dia")); 
    }
  
    has_attr = (int)hasattr(obj,"ovlp_dia");        
    if(has_attr){ 

      check_cmatrix(obj, "ovlp_dia", ndia, ndia);
      if(ovlp_dia_mem_status==0){ init_ovlp_dia(); }
      *ovlp_dia = extract<CMATRIX>(obj.attr("ovlp_dia"));   
      ovlp_dia_supplied = 1;
    }
//exit(0);

    has_attr = (int)hasattr(obj,"nac_dia");        
    nac_dia_supplied = 0;
    if(has_attr){    

      check_cmatrix(obj, "nac_dia", ndia, ndia);
      if(nac_dia_mem_status==0){ init_nac_dia(); }
      *nac_dia = extract<CMATRIX>(obj.attr("nac_dia"));    
      // an all-zero supply is the ecosystem's placeholder convention - not data
      nac_dia_supplied = (abs(nac_dia->max_elt()) != 0.0);
    }

    has_attr = (int)hasattr(obj,"hvib_dia");        
    hvib_dia_supplied = 0;
    if(has_attr){    

      check_cmatrix(obj, "hvib_dia", ndia, ndia);
      if(hvib_dia_mem_status==0){ init_hvib_dia(); }
      *hvib_dia = extract<CMATRIX>(obj.attr("hvib_dia"));    
      // an all-zero supply is the ecosystem's placeholder convention - not data
      hvib_dia_supplied = (abs(hvib_dia->max_elt()) != 0.0);
    }
    
    has_attr = (int)hasattr(obj,"dc1_dia");        
    if(has_attr){  

      check_cmatrix_list(obj, "dc1_dia", ndia, ndia, nnucl);
      if(dc1_dia_mem_status[0]==0){ init_dc1_dia(); }

      vector<CMATRIX> _dc1_dia(nnucl, CMATRIX(ndia,ndia));
      _dc1_dia = extract<CMATRIXList>(obj.attr("dc1_dia"));   

      for(int i=0;i<nnucl;i++){   *dc1_dia[i]   = _dc1_dia[i];    }
    }


    has_attr = (int)hasattr(obj,"d1ham_dia");        
    if(has_attr){  

      check_cmatrix_list(obj, "d1ham_dia", ndia, ndia, nnucl);
      if(d1ham_dia_mem_status[0]==0){ init_d1ham_dia(); }

      vector<CMATRIX> _d1ham_dia(nnucl, CMATRIX(ndia,ndia));
      _d1ham_dia = extract<CMATRIXList>(obj.attr("d1ham_dia"));   

      for(int i=0;i<nnucl;i++){    *d1ham_dia[i] = _d1ham_dia[i];  }
    }
//exit(0);

    has_attr = (int)hasattr(obj,"d2ham_dia");        
    if(has_attr){  

      check_cmatrix_list(obj, "d2ham_dia", ndia, ndia, nnucl*nnucl);

      // a model may supply second derivatives while the hierarchy was
      // initialized to der_lvl < 2 - allocate on demand (the raw-pointer
      // copy below would otherwise dereference NULL)
      if(d2ham_dia_mem_status[0]==0){ init_d2ham_dia(); }
      vector<CMATRIX> _d2ham_dia(nnucl*nnucl, CMATRIX(ndia,ndia));
      _d2ham_dia = extract<CMATRIXList>(obj.attr("d2ham_dia"));   

      for(int i=0;i<nnucl*nnucl;i++){    *d2ham_dia[i] = _d2ham_dia[i];  }
    }

    has_attr = (int)hasattr(obj,"basis_transform");
    basis_transform_supplied = 0;
    if(has_attr){

      // a diabatic model may supply its own SMOOTH dia->adi
      // transformation - the stored-U transform (ham_transform_method 2)
      // consumes it in place of the internal diagonalization, keeping
      // the model's gauge (no LAPACK branch switches)
      check_cmatrix(obj, "basis_transform", ndia, nadi);
      if(basis_transform_mem_status==0){ init_basis_transform(); }
      *basis_transform = extract<CMATRIX>(obj.attr("basis_transform"));
      // an all-zero supply is the ecosystem's placeholder convention - not data
      basis_transform_supplied = (abs(basis_transform->max_elt()) != 0.0);
    }

    has_attr = (int)hasattr(obj,"time_overlap_dia"); 
    if(has_attr){    

      check_cmatrix(obj, "time_overlap_dia", ndia, ndia);
      if(time_overlap_dia_mem_status==0){ init_time_overlap_dia(); }
      *time_overlap_dia = extract<CMATRIX>(obj.attr("time_overlap_dia"));    
    }

    // attributes of the OTHER representation are NOT read by this harvest -
    // silently dropping a supplied channel hides wrong models (basis_transform
    // is dual-harvested and exempt), so fail loud instead
    {
      const char* adi_names[] = {"ham_adi", "hvib_adi", "nac_adi",
                                 "time_overlap_adi",
                                 "dc1_adi", "d1ham_adi", "d2ham_adi",
                                 "gs_kinetic_energy"};
      for(int a=0;a<8;a++){
        if((int)hasattr(obj, adi_names[a])){
          cout<<"Error in compute_diabatic(): the model supplies obj."
              <<adi_names[a]<<", which the diabatic harvest"
              <<" (ham_update_method = 1) does not read - the supplied data"
              <<" would be silently dropped. Use ham_update_method = 2"
              <<" (adiabatic-native harvest) or drop the attribute\n";
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
            cout << "WARNING in compute_diabatic(): the model supplies obj."
                 << name << ", which matches NO ingested attribute name in"
                 << " either representation - it is silently dropped (check"
                 << " the spelling against the harvest lists in"
                 << " nHamiltonian_compute_adiabatic/diabatic.cpp)\n";
            warned_unknown = true;
          }
        }
      }
    }


  //exit(0); 
    
  }// if lvl == level

  else if(lvl>level){
    require_children(lvl, __func__);

#ifdef __OPENMP
    int nthreads = omp_get_num_threads();
#else
    int nthreads = 1;
#endif
    // exceptions (including python errors from the model callback) cannot
    // leave an OMP-outlined region - capture and rethrow outside
    std::exception_ptr omp_eptr = nullptr;
    #pragma omp parallel num_threads( nthreads )
    {
        #pragma omp for    
        for(int i=0;i<children.size();i++){
          try { children[i]->compute_diabatic(py_funct,q,params,lvl); }
          catch (...) {
            #pragma omp critical
            { if (!omp_eptr) { omp_eptr = std::current_exception(); } }
          }
        }
    }// pragma
    if (omp_eptr) { std::rethrow_exception(omp_eptr); }

  }

  else{
    cout<<"WARNING in nHamiltonian::compute_diabatic\n"; 
    cout<<"Can not run evaluation of function in the parent Hamiltonian from the\
     child node\n";    
  }



}




}// namespace libnhamiltonian
}// liblibra

