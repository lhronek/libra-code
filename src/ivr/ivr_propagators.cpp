/*********************************************************************************
* Copyright (C) 2018 Alexey V. Akimov
*
* This file is distributed under the terms of the GNU General Public License
* as published by the Free Software Foundation, either version 2 of
* the License, or (at your option) any later version.
* See the file LICENSE in the root directory of this distribution
* or <http://www.gnu.org/licenses/>.
*
*********************************************************************************/
/**
  \file ivr_propagators.cpp
  \brief These are the C++ re-implementation of the Fortran codes from the Ananth group.
         The original codes can be found here:
         https://github.com/AnanthGroup/SC-IVR-Code-Package    

  According to original documentation:

! This file contains subroutines that 
! propagate trajectories and the monodromy
! matrix for any mD SC-IVR

! Trajectory flags are different for 
! Forward-Backward and Forward-Forward
! implementations so they use separate
! subroutines



*/

#include "ivr.h"


/// liblibra namespace
namespace liblibra{

/// libivr namespace
namespace libivr{



MATRIX divm(MATRIX& M, MATRIX& mass){

  int n = M.n_rows;

  MATRIX res(n,n);

  for(int i=0;i<n;i++){
    for(int j=0;j<n;j++){
      res.set(i,j,  M.get(i,j)/mass.get(i,i) );
    }
  }

  return res;
}



void ivr_potential(boost::python::object py_funct, MATRIX& q, boost::python::object params, double& v, MATRIX& dv, MATRIX& d2v){
/**
  The potential, its gradient and its Hessian of the single surface at q from a model function with libra's hand-over:
  ham_dia (1 x 1), d1ham_dia (Ndof entries of 1 x 1) and d2ham_dia (Ndof x Ndof entries of 1 x 1, the row-major Hessian)
*/
  int Ndof = q.n_rows;
  boost::python::list full_id;
  full_id.append(0);
  boost::python::object obj = py_funct(q, params, full_id);
  for(const char* name : {"ham_dia", "d1ham_dia", "d2ham_dia"}){
    if(!PyObject_HasAttrString(obj.ptr(), name)){
      cout<<"Error in Integrator(): the model hands over no "<<name<<" - the potential, its gradient and its Hessian are needed\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }
  }
  CMATRIX h = boost::python::extract<CMATRIX>(obj.attr("ham_dia"));
  if(h.n_rows!=1 || h.n_cols!=1){
    cout<<"Error in Integrator(): ham_dia is "<<h.n_rows<<" x "<<h.n_cols<<" - one surface (1 x 1) is propagated\n";
    std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
  }
  v = h.get(0,0).real();
  CMATRIXList d1 = boost::python::extract<CMATRIXList>(obj.attr("d1ham_dia"));
  CMATRIXList d2 = boost::python::extract<CMATRIXList>(obj.attr("d2ham_dia"));
  if((int)d1.size()!=Ndof || (int)d2.size()!=Ndof*Ndof){
    cout<<"Error in Integrator(): d1ham_dia has "<<d1.size()<<" entries and d2ham_dia "<<d2.size()<<" - "<<Ndof<<" and "<<Ndof*Ndof<<" are needed\n";
    std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
  }
  for(int i=0;i<Ndof;i++){
    dv.set(i,0, d1[i].get(0,0).real());
    for(int k=0;k<Ndof;k++){ d2v.set(i,k, d2[i*Ndof+k].get(0,0).real()); }
  }
}

void Integrator(MATRIX& q, MATRIX& p, vector<MATRIX>& M, double& action, MATRIX& mass, double dt,
                boost::python::object py_funct, boost::python::object params){
/**
  \brief Symplectic integrator

  \param[in/out] q - coordinates in the phase space (Ndof x 1 matrix)
  \param[in/out] p - momenta in the phase space (Ndof x 1 matrix)
  \param[in/out] M - Monodromy matrix array defined as M = (M11,M12,M21,M22)
             each element of the array is a Ndof x Ndof matrix
  \param[in/out] action - action 
  \param[in] mass - massed associated with all DOFs, a Ndof x Ndof matrix
  \param[in] dt - integration timestep
  \param[in] py_funct, params - the model function and its parameters: py_funct(q, params, full_id) hands over the
             single surface's ham_dia, d1ham_dia and d2ham_dia (the potential, its gradient and its Hessian)
*/
  int Ndof = q.n_rows;

  action = 0.0;
  double v = 0.0;        // potential
  MATRIX dv(Ndof,1);     // derivatives
  MATRIX d2v(Ndof,Ndof); // 2-nd order derivatives

  const double const1 = 1.0/sqrt(3.0);
  vector<double> a(4,0.0), b(4,0.0);
  a[0]  = 0.5 * (1.0 - const1) * dt;
  a[1]  = const1 * dt;
  a[2]  =-const1 * dt;
  a[3]  = 0.5 * (1.0 + const1) * dt;

  b[0]  = 0.0;
  b[1]  = 0.5 * (0.5 + const1) * dt;
  b[2]  = 0.5 * dt;
  b[3]  = 0.5 * (0.5 - const1) * dt;

  // Points
  for(int j=0;j<4;j++){ 

   if(j>0){
      ivr_potential(py_funct, q, params, v, dv, d2v);

      action   = action - b[j] * v;
      p   = p - b[j] * dv;
      M[3] = M[3] - b[j] * d2v * M[1];
      M[2] = M[2] - b[j] * d2v * M[0];

   }// if j>0

   double ke = 0.0; // kinetic energy

   for(int i=0;i<Ndof;i++){
      ke = ke + 0.5*p.M[i]*p.M[i]/mass.get(i,i);
      q.M[i] = q.M[i] + a[j] * p.M[i]/mass.get(i,i);
   }
   action = action + a[j] * ke;

   M[1] = M[1] + a[j] * divm(M[3], mass);
   M[0] = M[0] + a[j] * divm(M[2], mass);


  }// for j

}

double Integrator(MATRIX& q, MATRIX& p, vector<MATRIX>& M, MATRIX& mass, double dt,
                  boost::python::object py_funct, boost::python::object params){
/**
  The form callable from Python - the step's action returned (boost cannot bind a Python number to the double&)
*/
  double action = 0.0;
  Integrator(q, p, M, action, mass, dt, py_funct, params);
  return action;
}

}/// namespace libivr
}/// liblibra
