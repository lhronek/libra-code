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

#include <Eigen/LU>
#include <stdexcept>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/Core>
#include "mEigen.h"
#include <cmath>

/// liblibra namespace
namespace liblibra{


using namespace Eigen;
using namespace std;
using namespace liblinalg;

/// libmeigen namespace
namespace libmeigen{





void sqrt_matrix(CMATRIX& S, CMATRIX& S_half, CMATRIX& S_i_half, double thresh, int do_phase_correction){
/**
  This function computes S^{1/2} and S^{-1/2} for given matrix S
  \param[in] S Input matrix
  \param[out] S_half Computed S^{1/2} matrix
  \param[out] S_i_half Computed S^{-1/2} matrix
  \param[in] threshold - if an absolute value of any eigenvalue of S is below this level, we stop,
   throwing an error message


*/

  if(S.n_cols != S.n_rows){
    cout<<"Error in libmeigen::sqrt_matrix : the input matrix is not square\n"; std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above"); 
  }
  if(S_half.n_cols != S_half.n_rows){
    cout<<"Error in libmeigen::sqrt_matrix : the output S^{1/2} matrix is not square\n"; std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above"); 
  }
  if(S_i_half.n_cols != S_i_half.n_rows){
    cout<<"Error in libmeigen::sqrt_matrix : the output S^{-1/2} matrix is not square\n"; std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above"); 
  }
  if(S.n_cols != S_half.n_cols){
    cout<<"Error in libmeigen::sqrt_matrix : size of matrix S is not the same as that of matrix S^{1/2}\n"; std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above"); 
  }
  if(S.n_cols != S_i_half.n_cols){
    cout<<"Error in libmeigen::sqrt_matrix : size of matrix S is not the same as that of matrix S^{-1/2}\n"; std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above"); 
  }


  int i;
 
  // Let us first diagonalize the overlap matrix S
  int sz = S.n_cols;  
  CMATRIX* C; C = new CMATRIX(sz, sz);  *C = complex<double>(0.0, 0.0);
  CMATRIX* Seig; Seig = new CMATRIX(sz, sz);  *Seig = complex<double>(0.0,0.0);


  // Find the eigenvalues of the the S matrix
  solve_eigen(S, *Seig, *C, 0);  // S * C = C * Seig  ==>  S = C * Seig * C.H()

  //if(do_phase_correction){   correct_phase(C);  }

  // Diagonal form of the S^{-1/2} and S^{1/2} matrices
  S_i_half = complex<double>(0.0,0.0);  // S^{-1/2}
  S_half = complex<double>(0.0,0.0);    // S^{1/2}

  for(i=0;i<sz;i++){
    complex<double> val = std::sqrt(Seig->get(i,i));

    double nrm  = std::abs(val);
    if(nrm<thresh){  
      std::cout<<"\n Error in sqrt_matrix: One of the eigenvalues of the matrix S is "<< val
               <<"\n this is below the used threshold of "<<thresh
               <<"\n So... the matrix is likely singular or your threshold is too large"
               <<"\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }
    else{
      S_i_half.M[i*sz+i] = 1.0/val;
      S_half.M[i*sz+i] = val;
    }
  }

  // Convert to the original basis
  S_i_half = (*C) * S_i_half * ((*C).H());
  S_half = (*C) * S_half * ((*C).H());

  delete C;
  delete Seig;


}// sqrt_matrix


void sqrt_matrix(CMATRIX& S, CMATRIX& S_half, CMATRIX& S_i_half, double thresh){

  sqrt_matrix(S, S_half, S_i_half, thresh, 0);

}

void sqrt_matrix(CMATRIX& S, CMATRIX& S_half, CMATRIX& S_i_half){

  sqrt_matrix(S, S_half, S_i_half, -1.0, 0);

}

int lowdin_frame(CMATRIX& S, CMATRIX& X, CMATRIX& Y){
/**
  The Lowdin (symmetric) orthonormalization frame of a metric S: X = S^-1/2 and Y = S^1/2.
  Returns 1 with X and Y filled, or 0 for an identity metric (max |S - I| <= 1e-12) with X and
  Y untouched - callers keep their S = I code path bit-identical
*/
  int n = S.n_cols;
  if(S.n_rows != n){
    cout<<"Error in libmeigen::lowdin_frame : the metric is not square\n"; std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
  }
  double dev = 0.0;
  for(int i=0; i<n; i++){
    for(int j=0; j<n; j++){
      double d = std::abs(S.get(i,j) - ((i==j) ? 1.0 : 0.0));
      if(d > dev){ dev = d; }
    }
  }
  if(dev <= 1e-12){ return 0; }
  if(X.n_rows != n || X.n_cols != n || Y.n_rows != n || Y.n_cols != n){
    cout<<"Error in libmeigen::lowdin_frame : X and Y must be "<<n<<" x "<<n<<" matrices\n"; std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
  }
  sqrt_matrix(S, Y, X);
  return 1;
}

void sqrt_matrix_derivative(CMATRIX& S, CMATRIX& dS, CMATRIX& dY, CMATRIX& dX){
/**
  Derivatives of Y = S^1/2 and X = S^-1/2 along a Hermitian variation dS of a positive-definite S.
  With S = V s V^H (Daleckii-Krein): (V^H dF V)_ab = (V^H dS V)_ab (f(s_a) - f(s_b))/(s_a - s_b), the
  divided differences written in their cancellation-free forms 1/(sqrt(s_a) + sqrt(s_b)) and
  -1/(sqrt(s_a) sqrt(s_b) (sqrt(s_a) + sqrt(s_b))), which are the derivatives f' on the diagonal
*/
  int n = S.n_cols;
  if(S.n_rows != n || dS.n_rows != n || dS.n_cols != n || dY.n_rows != n || dY.n_cols != n || dX.n_rows != n || dX.n_cols != n){
    cout<<"Error in libmeigen::sqrt_matrix_derivative : S, dS, dY and dX must all be "<<n<<" x "<<n<<" matrices\n"; std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
  }
  CMATRIX s(n, n);
  CMATRIX V(n, n);
  solve_eigen(S, s, V, 0);
  CMATRIX W(n, n);
  W = V.H() * dS * V;
  CMATRIX GY(n, n);
  CMATRIX GX(n, n);
  for(int a=0; a<n; a++){
    for(int b=0; b<n; b++){
      double sa = s.get(a,a).real();
      double sb = s.get(b,b).real();
      if(sa <= 0.0 || sb <= 0.0){
        cout<<"Error in libmeigen::sqrt_matrix_derivative : the metric is not positive definite (eigenvalue "<<std::min(sa, sb)<<")\n"; std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
      }
      double ra = std::sqrt(sa);
      double rb = std::sqrt(sb);
      GY.set(a, b, W.get(a,b) / (ra + rb));
      GX.set(a, b, -W.get(a,b) / (ra * rb * (ra + rb)));
    }
  }
  dY = V * GY * V.H();
  dX = V * GX * V.H();
}



void exp_matrix(CMATRIX& res, CMATRIX& S, complex<double> dt, int do_phase_correction){
/**
  This function computes exp(S*dt) for a given matrix S
  \param[in] S input matrix
  \param[in] dt scaling factor

*/

  if(S.n_cols != S.n_rows){
    cout<<"Error in libmeigen::exp_matrix : the input matrix is not square\n"; std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above"); 
  }


  int i;
 
  // Let us first diagonalize the input matrix x
  int sz = S.n_cols;  
  CMATRIX* C; C = new CMATRIX(sz, sz);  *C = complex<double>(0.0, 0.0);
  CMATRIX* Seig; Seig = new CMATRIX(sz, sz);  *Seig = complex<double>(0.0,0.0);


  // Find the eigenvalues of the the S matrix
  solve_eigen(S, *Seig, *C, 0);  // S * C = C * Seig  ==>  S = C * Seig * C.H()

  //if(do_phase_correction){   correct_phase(C);  }

  
  for(i=0;i<sz;i++){ res.M[i*sz+i]= std::exp(dt * Seig->get(i,i)); }

  // Convert to the original basis
  res = (*C) * res * ((*C).H());

  delete C;
  delete Seig;

}// exp_matrix


void exp_matrix(CMATRIX& res, CMATRIX& S, complex<double> dt){
  exp_matrix(res,S, dt, 0);
}



}// namespace libmeigen
}// namespace liblibra
