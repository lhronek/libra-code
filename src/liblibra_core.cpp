/*********************************************************************************
* Copyright (C) 2015-2021 Alexey V. Akimov
*
* This file is distributed under the terms of the GNU General Public License
* as published by the Free Software Foundation, either version 3 of
* the License, or (at your option) any later version.
* See the file LICENSE in the root directory of this distribution
* or <http://www.gnu.org/licenses/>.
*
*********************************************************************************/
/**
  \file liblibra_core.cpp
  \brief The file exports all core functionality of the Libra code
    
*/

#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include "liblibra_core.h"
#include <boost/version.hpp>
#include <Eigen/Core>
#include <dlfcn.h>
#include <unistd.h>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <sstream>
#include "libra_version.h"
#ifdef _OPENMP
#include <omp.h>
#endif

/// liblibra namespace
namespace liblibra{


using namespace boost::python;

using namespace libutil;
using namespace libio;
using namespace libcontext;
using namespace libcommon_types;

using namespace libann;
using namespace libdata;
using namespace libgraph;
using namespace liblinalg;
using namespace libmeigen;
using namespace liboperators;
using namespace librandom;
using namespace libspecialfunctions;
using namespace libsymmetry;

using namespace libmolint;
using namespace libqobjects;
using namespace libbasis;
using namespace libbasis_setups;
using namespace liblibint2_wrappers;

using namespace libcalculators;

using namespace librigidbody;

using namespace libchemobjects;
using namespace libcell;

using namespace libcontrol_parameters;
using namespace libnhamiltonian;

using namespace libmodels;

using namespace libfgr;
using namespace libivr;
using namespace libdyn;

using namespace libconverters;
//using namespace libscripts;
//using namespace libqchem_tools;
using namespace libsolvers;
using namespace libintegrators;

using namespace libmontecarlo;
using namespace libopt;


void export_libra_core_objects(){
/** 
  \brief Exporter of Timer class and other mathematical libraries and their components

*/

  export_util_objects();
  export_io_objects();
  export_context_objects();
  export_common_types_objects();

  export_NeuralNetwork_objects();
  export_Data_objects();
  export_GRAPH_objects();
  export_linalg_objects();
  export_mEigen_objects();
  export_Operators_objects();
  export_Random_objects();
  export_SpecialFunctions_objects();
  export_symmetry_objects();
  export_timer_objects();

  export_molint_objects();
  export_qobjects_objects();
  export_basis_objects();
  export_libint2_wrappers_objects();


  export_calculators_objects();

  export_models_objects();

  export_RigidBody_objects();
  export_chemobjects_objects();
  export_Cell_objects();


  export_Control_Parameters_objects();
  export_Model_Parameters_objects();
  export_basis_setups_objects();


  export_nhamiltonian_objects();

  export_fgr_objects();
  export_ivr_objects();

  export_Dyn_objects();

  export_converters_objects();
//  export_scripts_objects();
//  export_qchem_tools_objects();

  export_solvers_objects();

  export_integrators_objects();


  export_montecarlo_objects();
  export_opt_objects();


}// export_libra_core_objects()




namespace {  // import-banner helpers

std::string libra_kv_line(const std::string& label, const std::string& value){
  std::string s = " Libra| " + label;
  int pad = 80 - (int)s.size() - (int)value.size();
  if(pad < 1){ pad = 1; }
  return s + std::string((size_t)pad, ' ') + value + "\n";
}

std::string libra_openmp_version(){
#ifdef _OPENMP
  int c = _OPENMP;
  const char* v = "";
  if(c==200505){ v="2.5"; } else if(c==200805){ v="3.0"; }
  else if(c==201107){ v="3.1"; } else if(c==201307){ v="4.0"; }
  else if(c==201511){ v="4.5"; } else if(c==201811){ v="5.0"; }
  else if(c==202011){ v="5.1"; } else if(c==202111){ v="5.2"; }
  std::ostringstream o;
  if(v[0]){ o<<v<<" ("<<c<<")"; } else { o<<c; }
  return o.str();
#else
  return "off";
#endif
}

std::string libra_blas_runtime(){
  // if the loaded BLAS is OpenBLAS, it reports its own build string
  typedef const char* (*cfg_fn)();
  void* sym = dlsym(RTLD_DEFAULT, "openblas_get_config");
  if(!sym){ return "unknown"; }
  const char* cfg = reinterpret_cast<cfg_fn>(sym)();
  if(!cfg){ return "unknown"; }
  std::istringstream in(cfg);
  std::string tok, out, extra;
  for(int i=0; in>>tok; i++){
    if(i<2){ if(i){ out += " "; } out += tok; }
    else if(extra.empty() && tok.find('=')==std::string::npos){
      // the core name is the first mixed-case token after the version
      for(size_t k=0;k<tok.size();k++){
        if(islower((unsigned char)tok[k])){ extra = tok; break; }
      }
    }
  }
  if(!extra.empty()){ out += " " + extra; }
  return out.empty() ? "unknown" : out;
}

int libra_blas_threads(){
  typedef int (*nt_fn)();
  void* sym = dlsym(RTLD_DEFAULT, "openblas_get_num_threads");
  if(!sym){ return -1; }
  return reinterpret_cast<nt_fn>(sym)();
}

std::string libra_cpu_model(){
  std::ifstream f("/proc/cpuinfo");
  std::string ln;
  while(std::getline(f, ln)){
    if(ln.compare(0, 10, "model name")==0){
      size_t pp = ln.find(':');
      if(pp!=std::string::npos){
        pp = ln.find_first_not_of(" \t", pp+1);
        if(pp!=std::string::npos){ return ln.substr(pp); }
      }
    }
  }
  return "unknown";
}

std::string libra_build_info(){
  std::ostringstream o;
  o << libra_kv_line("version string:", std::string("Libra version ") + LIBRA_VERSION_STRING);
  o << libra_kv_line("source code revision number:", std::string("git:") + LIBRA_GIT_HASH);
  o << libra_kv_line("source tree state:", LIBRA_TREE_STATE);
  o << " Libra| libraflags: " << LIBRA_FLAGS << "\n";
  o << libra_kv_line("Python version", PY_VERSION);
  std::string bv = BOOST_LIB_VERSION;
  for(size_t k=0;k<bv.size();k++){ if(bv[k]=='_'){ bv[k]='.'; } }
  o << libra_kv_line("Boost version", bv);
  std::ostringstream ev;
  ev<<EIGEN_WORLD_VERSION<<"."<<EIGEN_MAJOR_VERSION<<"."<<EIGEN_MINOR_VERSION;
  o << libra_kv_line("Eigen version", ev.str());
  o << libra_kv_line("Libint version", LIBRA_LIBINT_VERSION);
  o << libra_kv_line("OpenMP version", libra_openmp_version());
  return o.str();
}

void libra_print_banner(){
  const char* q = std::getenv("LIBRA_BANNER");
  if(q && q[0]=='0'){ return; }

  static const char* scales[5] = {
    "   *********   ", "   *   *   *   ", "  ***  *  ***  ",
    "       *       ", "     *****     " };
  static const char* letters[5] = {
    "*   *** ***  ***   ** ",
    "*    *  *  * *  * *  *",
    "*    *  ***  ***  ****",
    "*    *  *  * * *  *  *",
    "*** *** ***  *  * *  *" };
  static const char* labels[5] = {
    "PROGRAM STARTED AT", "PROGRAM STARTED ON", "PROGRAM STARTED BY",
    "PROGRAM PROCESS ID", "PROGRAM STARTED IN" };

  char tbuf[64];
  std::time_t now = std::time(NULL);
  if(!std::strftime(tbuf, sizeof(tbuf), "%Y-%m-%d %H:%M:%S", std::localtime(&now))){
    std::snprintf(tbuf, sizeof(tbuf), "unknown");
  }
  char hbuf[256];
  if(gethostname(hbuf, sizeof(hbuf))){ std::snprintf(hbuf, sizeof(hbuf), "unknown"); }
  const char* usr = std::getenv("USER");
  if(!usr){ usr = "unknown"; }
  std::ostringstream pid;
  pid << getpid();
  char cbuf[4096];
  std::string cwd = getcwd(cbuf, sizeof(cbuf)) ? std::string(cbuf) : std::string("unknown");

  std::string values[5] = { tbuf, hbuf, usr, pid.str(), cwd };

  std::ostringstream o;
  for(int k=0;k<5;k++){
    std::string left = std::string(scales[k]) + "  " + letters[k] + "  " + labels[k];
    int field = 80 - (int)left.size();
    std::string v = values[k];
    if((int)v.size() > field){ v = "..." + v.substr(v.size() - (size_t)(field-3)); }
    int pad = field - (int)v.size();
    if(pad<0){ pad=0; }
    o << left << std::string((size_t)pad, ' ') << v << "\n";
  }
  o << libra_build_info();
  o << libra_kv_line("DGEMM library (runtime)", libra_blas_runtime());
  int omp_n = 1;
#ifdef _OPENMP
  omp_n = omp_get_max_threads();
#endif
  { std::ostringstream t; t<<omp_n;
    o << libra_kv_line("OMP: Max number of threads", t.str()); }
  { int bt = libra_blas_threads();
    std::ostringstream t;
    if(bt>0){ t<<bt; } else { t<<"unknown"; }
    o << libra_kv_line("BLAS: Number of threads", t.str()); }
  o << libra_kv_line("CPU model name", libra_cpu_model());
  // flush: python buffers fd 1 separately - without it the banner can
  // land after the script's own prints in pipes
  std::cout << o.str() << std::flush;
}

std::string libra_build_info_py(){ return libra_build_info(); }

} // anonymous namespace


#ifdef CYGWIN
BOOST_PYTHON_MODULE(cyglibra_core){
#else
BOOST_PYTHON_MODULE(liblibra_core){
#endif

  libra_print_banner();

  // Register converters:
  // See here: https://misspent.wordpress.com/2009/09/27/how-to-write-boost-python-converters/
  //to_python_converter<std::vector<DATA>, VecToList<DATA> >();
  export_libra_core_objects();

  boost::python::scope().attr("__version__") = std::string(LIBRA_VERSION_STRING);
  boost::python::def("build_info", &libra_build_info_py);

}

}// liblibra
