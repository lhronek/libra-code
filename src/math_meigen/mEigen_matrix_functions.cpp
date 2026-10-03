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
#include "../math_linalg/blas_interface.h"
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
  This function computes S^{1/2} and S^{-1/2} for given matrix S - the eigenpairs from zheevd,
  the LAPACK kernel at every matrix size when the build linked it (the eigen size threshold of solve_eigen does not apply), the Eigen solver otherwise
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
  // S * C = C * Seig  ==>  S = C * Seig * C.H() - the LAPACK zheevd when the build linked it,
  // the Eigen solver otherwise (both refuse a non-Hermitian S). The result is a function of S,
  // so the eigenvector gauge of either solver cancels in it
  if(have_lapack()){ lapack_heevd(S, *Seig, *C); }
  else{ solve_eigen(S, *Seig, *C, 0); }

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

  // the default singularity guard: a root below 1e-8 (an eigenvalue at the double-precision floor,
  // 1e-16) is a singular metric - S^-1/2 would carry 1e8-sized garbage, inf or NaN into everything
  // built from it - every caller takes this default
  sqrt_matrix(S, S_half, S_i_half, 1e-8, 0);

}

void sqrt_matrix(MATRIX& S, MATRIX& S_half, MATRIX& S_i_half, double thresh, int do_phase_correction){
/**
  S^{1/2} and S^{-1/2} of a real symmetric positive-definite S - the real counterpart of the
  CMATRIX version: the eigenpairs from LAPACK dsyevd when the build linked it, at every matrix
  size (the eigen size threshold of solve_eigen does not apply), from the Eigen solver otherwise,
  the back-transforms as dgemm-backed MATRIX products. A negative eigenvalue has no
  real root and is refused, a root below thresh stops with the singularity message (thresh <= 0
  disables that check)
  \param[in] S Input matrix
  \param[out] S_half Computed S^{1/2} matrix
  \param[out] S_i_half Computed S^{-1/2} matrix
  \param[in] thresh - the singularity threshold on the roots of the eigenvalues
  \param[in] do_phase_correction - signature parity with the CMATRIX version, unused
*/
  int sz = S.n_cols;
  if(S.n_rows != sz || S_half.n_rows != sz || S_half.n_cols != sz || S_i_half.n_rows != sz || S_i_half.n_cols != sz){
    cout<<"Error in libmeigen::sqrt_matrix : S, S^{1/2} and S^{-1/2} must all be "<<sz<<" x "<<sz<<" matrices\n"; std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
  }
  MATRIX C(sz, sz);
  MATRIX Seig(sz, sz);
  if(have_lapack()){ lapack_syevd(S, Seig, C); }
  else{ solve_eigen(S, Seig, C, 0); }
  MATRIX D_half(sz, sz);
  MATRIX D_i_half(sz, sz);
  for(int i=0;i<sz;i++){
    double e = Seig.get(i,i);
    if(e < 0.0){
      std::cout<<"\n Error in sqrt_matrix: the matrix S has a negative eigenvalue "<<e
               <<"\n a real S^{1/2} needs a positive semi-definite S - the CMATRIX version returns the complex root\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }
    double val = std::sqrt(e);
    if(val < thresh){
      std::cout<<"\n Error in sqrt_matrix: One of the eigenvalues of the matrix S is "<< e <<" (root "<<val<<")"
               <<"\n this is below the used threshold of "<<thresh
               <<"\n So... the matrix is likely singular or your threshold is too large"
               <<"\n";
      std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
    }
    D_half.set(i, i, val);
    D_i_half.set(i, i, 1.0/val);
  }
  S_half = C * D_half * C.T();
  S_i_half = C * D_i_half * C.T();

}// sqrt_matrix (MATRIX)


void sqrt_matrix(MATRIX& S, MATRIX& S_half, MATRIX& S_i_half, double thresh){

  sqrt_matrix(S, S_half, S_i_half, thresh, 0);

}

void sqrt_matrix(MATRIX& S, MATRIX& S_half, MATRIX& S_i_half){

  // the default singularity guard: a root below 1e-8 (an eigenvalue at the double-precision floor,
  // 1e-16) is a singular metric - S^-1/2 would carry 1e8-sized garbage, inf or NaN into everything
  // built from it - every caller takes this default
  sqrt_matrix(S, S_half, S_i_half, 1e-8, 0);

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

void metric_eigenpairs(CMATRIX& S, CMATRIX& V, vector<double>& s){
/**
  The eigenpairs S = V diag(s) V^H of a Hermitian metric, for the frame functions below: zheevd, the LAPACK kernel
  at every matrix size when the build linked it (the eigen size threshold of solve_eigen does not apply), the Eigen
  solver otherwise. The divided differences of the derivatives are gauge-invariant (V G V^H with G built from
  V^H dS V), so either solver's eigenvector phases serve
*/
  int n = S.n_cols;
  if(S.n_rows != n || V.n_rows != n || V.n_cols != n){
    cout<<"Error in libmeigen::metric_eigenpairs : S and V must be square matrices of one size\n"; std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
  }
  CMATRIX e(n, n);
  if(have_lapack()){ lapack_heevd(S, e, V); }
  else{ solve_eigen(S, e, V, 0); }
  s.assign(n, 0.0);
  for(int a=0; a<n; a++){ s[a] = e.get(a,a).real(); }
}

void sqrt_matrix_derivative(CMATRIX& V, vector<double>& s, CMATRIX& dS, CMATRIX& dY, CMATRIX& dX){
/**
  Derivatives of Y = S^1/2 and X = S^-1/2 along a Hermitian variation dS of a positive-definite S, from the
  eigenpairs S = V diag(s) V^H (metric_eigenpairs) - the form for a caller that holds the eigenpairs of an unchanged
  metric across variations. Daleckii-Krein: (V^H dF V)_ab = (V^H dS V)_ab (f(s_a) - f(s_b))/(s_a - s_b), the divided
  differences written in their cancellation-free forms 1/(sqrt(s_a) + sqrt(s_b)) and
  -1/(sqrt(s_a) sqrt(s_b) (sqrt(s_a) + sqrt(s_b))), which are the derivatives f' on the diagonal
*/
  int n = V.n_cols;
  if(V.n_rows != n || (int)s.size() != n || dS.n_rows != n || dS.n_cols != n || dY.n_rows != n || dY.n_cols != n || dX.n_rows != n || dX.n_cols != n){
    cout<<"Error in libmeigen::sqrt_matrix_derivative : V, dS, dY and dX must all be "<<n<<" x "<<n<<" matrices and s must hold "<<n<<" eigenvalues\n"; std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
  }
  CMATRIX W(n, n);
  W = V.H() * dS * V;
  CMATRIX GY(n, n);
  CMATRIX GX(n, n);
  for(int a=0; a<n; a++){
    for(int b=0; b<n; b++){
      double sa = s[a];
      double sb = s[b];
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

void sqrt_matrix_derivative(CMATRIX& S, CMATRIX& dS, CMATRIX& dY, CMATRIX& dX){
/**
  Derivatives of Y = S^1/2 and X = S^-1/2 along a Hermitian variation dS of a positive-definite S - the eigenpairs
  of S computed here (metric_eigenpairs) and the derivatives from them (the overload above)
*/
  int n = S.n_cols;
  if(S.n_rows != n || dS.n_rows != n || dS.n_cols != n || dY.n_rows != n || dY.n_cols != n || dX.n_rows != n || dX.n_cols != n){
    cout<<"Error in libmeigen::sqrt_matrix_derivative : S, dS, dY and dX must all be "<<n<<" x "<<n<<" matrices\n"; std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
  }
  CMATRIX V(n, n);
  vector<double> s;
  metric_eigenpairs(S, V, s);
  sqrt_matrix_derivative(V, s, dS, dY, dX);
}



void exp_matrix(CMATRIX& res, CMATRIX& S, complex<double> dt, int do_phase_correction){
/**
  This function computes exp(S*dt) for a given matrix S - the eigenpairs from zheevd,
  the LAPACK kernel at every matrix size when the build linked it (the eigen size threshold of solve_eigen does not apply), the Eigen solver otherwise
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
  // S * C = C * Seig  ==>  S = C * Seig * C.H() - the LAPACK zheevd when the build linked it,
  // the Eigen solver otherwise (both refuse a non-Hermitian S). The result is a function of S,
  // so the eigenvector gauge of either solver cancels in it
  if(have_lapack()){ lapack_heevd(S, *Seig, *C); }
  else{ solve_eigen(S, *Seig, *C, 0); }

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
