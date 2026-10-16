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
  \file libopt.cpp
  \brief The file implements Python export function
    
*/


#if defined(USING_PCH)
#include "../pch.h"
#else
#include <stdlib.h>
#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#endif 

#include "libopt.h"


/// liblibra namespace
namespace liblibra{

using namespace liblinalg;

/// libopt namespace
namespace libopt{

using namespace boost::python;
namespace bp = boost::python;




// the forms callable from Python: boost cannot bind a Python number to a double& output, so the outputs are returned in
// a list after the function's value
boost::python::list expt_hyperbolic(double x, double alpha){
  double val = 0.0, deriv = 0.0;
  hyperbolic(x, alpha, val, deriv);
  boost::python::list res;
  res.append(val);
  res.append(deriv);
  return res;
}

boost::python::list expt_derivs1(MATRIX& x, MATRIX& y, MATRIX& A, MATRIX& gradA, double L, int opt, int opt2){
  double gradL = 0.0;
  double Lagrangian = derivs1(x, y, A, gradA, L, gradL, opt, opt2);
  boost::python::list res;
  res.append(Lagrangian);
  res.append(gradL);
  return res;
}

boost::python::list expt_derivs1_new(MATRIX& x, MATRIX& y, MATRIX& A, MATRIX& gradA, double L, MATRIX& num_coeff, MATRIX& denom_coeff, int opt){
  double gradL = 0.0;
  double Lagrangian = derivs1_new(x, y, A, gradA, L, gradL, num_coeff, denom_coeff, opt);
  boost::python::list res;
  res.append(Lagrangian);
  res.append(gradL);
  return res;
}

boost::python::list expt_derivs3(MATRIX& x, MATRIX& y, MATRIX& A, MATRIX& gradA, double L, int opt, int opt2, MATRIX& s){
  double gradL = 0.0;
  double Lagrangian = derivs3(x, y, A, gradA, L, gradL, opt, opt2, s);
  boost::python::list res;
  res.append(Lagrangian);
  res.append(gradL);
  return res;
}

boost::python::list expt_grad_step(MATRIX& A, MATRIX& x, MATRIX& x_new, double dt, double L, MATRIX& mu, int opt, double dT, int approach_option){
  double Lag = 0.0, L0 = 0.0, L1 = 0.0, L2 = 0.0, L3 = 0.0;
  grad_step(A, x, x_new, dt, L, mu, opt, Lag, L0, L1, L2, L3, dT, approach_option);
  boost::python::list res;
  res.append(L);
  res.append(Lag);
  res.append(L0);
  res.append(L1);
  res.append(L2);
  res.append(L3);
  return res;
}

void export_opt_objects(){
/** 
  \brief Exporter of libopt classes and functions

*/

  // Defined in matrix_gradients.cpp

  void (*expt_normalize_transition_matrix_v1)
  (MATRIX& flux, MATRIX& num_coeff, MATRIX& denom_coeff) = &normalize_transition_matrix; 
  def("normalize_transition_matrix", expt_normalize_transition_matrix_v1);

  void (*expt_hyperbolic_v1)(double x, double alpha, double& val, double& deriv) = &hyperbolic;
  def("hyperbolic", expt_hyperbolic_v1);
  def("hyperbolic", expt_hyperbolic);

  MATRIX (*expt_transform_to_hyperbolic_v1)(MATRIX& A, double alp) = &transform_to_hyperbolic;
  def("transform_to_hyperbolic", expt_transform_to_hyperbolic_v1);

  double (*expt_derivs0_v1)(MATRIX& x, MATRIX& y, MATRIX& A, MATRIX& gradA) = &derivs0;
  def("derivs0", expt_derivs0_v1);

  double (*expt_derivs1_v1)
  (MATRIX& x, MATRIX& y, MATRIX& A, MATRIX& gradA, double& L, double& gradL, int opt, int opt2) = &derivs1;
  def("derivs1", expt_derivs1_v1);
  def("derivs1", expt_derivs1);

  double (*expt_derivs1_new_v1)
  (MATRIX& x, MATRIX& y, MATRIX& A, MATRIX& gradA, double& L, 
  double& gradL, MATRIX& num_coeff, MATRIX& denom_coeff, int opt) = &derivs1_new;
  def("derivs1_new", expt_derivs1_new_v1);
  def("derivs1_new", expt_derivs1_new);

  double (*expt_derivs2_v1)
  (MATRIX& x, MATRIX& y, MATRIX& A, MATRIX& gradA, MATRIX& mu, 
  MATRIX& gradmu, int opt, int opt2) = &derivs2;
  def("derivs2", expt_derivs2_v1);

  double (*expt_derivs3_v1)
  (MATRIX& x, MATRIX& y, MATRIX& A, MATRIX& gradA, double& L, 
  double& gradL, int opt, int opt2, MATRIX& s) = &derivs3;
  def("derivs3", expt_derivs3_v1);
  def("derivs3", expt_derivs3);

  double (*expt_derivs4_v1)
  (MATRIX& x, MATRIX& y, MATRIX& A, MATRIX& gradA, MATRIX& mu, 
  MATRIX& gradmu, int opt) = &derivs4;
  def("derivs4", expt_derivs4_v1);


  // Defined in grad_descent.cpp

  MATRIX (*expt_grad_descent_v1)
  (bp::object grad_function, MATRIX& dof, bp::object funct_params, 
   double grad_tol, double step_size, int max_steps) = &grad_descent;

  def("grad_descent",expt_grad_descent_v1);


  void (*expt_grad_step_v1)(MATRIX& A, MATRIX& x, MATRIX& x_new, double dt, 
  double& L, MATRIX& mu, int opt, double& Lag, double& L0, double& L1, 
  double& L2, double& L3, double dT, int approach_option) = &grad_step;

  def("grad_step", expt_grad_step_v1);
  def("grad_step", expt_grad_step);

  MATRIX (*expt_run_opt_v1)
  (MATRIX& x, MATRIX& x_new, double dt, double nsteps, double err_tol, int opt, 
  vector<double>& err, double dT, int approach_option) = &run_opt;

  def("run_opt", expt_run_opt_v1);

}// export_opt_objects()




#ifdef CYGWIN
BOOST_PYTHON_MODULE(cygopt){
#else
BOOST_PYTHON_MODULE(libopt){
#endif

  // Register converters:
  // See here: https://misspent.wordpress.com/2009/09/27/how-to-write-boost-python-converters/
  //to_python_converter<std::vector<DATA>, VecToList<DATA> >();

  export_opt_objects();

}


}// libopt
}// liblibra

