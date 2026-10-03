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

#if defined(USING_PCH)
#include "../pch.h"
#else
#include <stdexcept>
#include <Eigen/LU>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/Core>
#include <Eigen/SVD>
#endif 

#include "mEigen.h"
#include "../math_linalg/blas_interface.h"

/// liblibra namespace
namespace liblibra{

using namespace Eigen;
using namespace std;
using namespace liblinalg;

/// libmeigen namespace
namespace libmeigen{



void FullPivLU_decomposition(MATRIX& A, MATRIX& P, MATRIX& L, MATRIX& U, MATRIX& Q){
/** A wrapper of Eigen::FullPivLU<MatrixXd>.matrixLU() 
   A - the source matrix
   P - the permutation matrix
   L - lower triangular
   U - upper triangular
   Q - another permutation matrix

  Really, what happens is:

  A = P * L * U * Q,  
  P, Q - are the permutation matrices

*/

  int N = A.n_cols;
  int i,j;

  if(have_lapack() && N >= get_lapack_fullpiv_min_n()){
    // LAPACK complete-pivoting LU (blas_interface.h): P A Q = L U in the same convention
    lapack_getc2(A, P, L, U, Q);
    return;
  }
  MatrixXd a(N,N);
  for(i=0;i<N;i++){
    for(j=0;j<N;j++){
      a(i,j) = A.M[i*N+j];
    }// for j
  }// for i

  Eigen::FullPivLU<MatrixXd> lu(a);


  MatrixXd l(N,N); 
  l = lu.matrixLU().triangularView<StrictlyLower>();

  MatrixXd u(N,N);
  u = lu.matrixLU().triangularView<Upper>();

  MatrixXd p(N,N);
  p = lu.permutationP(); 

  MatrixXd q(N,N);
  q = lu.permutationQ(); 


  for(i=0;i<N;i++){
    for(j=0;j<N;j++){
      P.M[i*N+j] = p(i,j);

      L.M[i*N+j] = l(i,j);
      if(i==j){   L.M[i*N+j] = 1.0;  }

      U.M[i*N+j] = u(i,j);

      Q.M[i*N+j] = q(i,j);

    }// for j
  }// for i
  
}


void FullPivLU_decomposition(CMATRIX& A, CMATRIX& P, CMATRIX& L, CMATRIX& U, CMATRIX& Q){
/** A wrapper of Eigen::FullPivLU<MatrixXcd>.matrixLU() 
   A - the source matrix
   P - the permutation matrix
   L - lower triangular
   U - upper triangular
   Q - another permutation matrix

  Really, what happens is:

  A = P * L * U * Q,  
  P, Q - are the permutation matrices

*/


  int N = A.n_cols;
  int i,j;

  if(have_lapack() && N >= get_lapack_fullpiv_min_n()){
    // LAPACK complete-pivoting LU (blas_interface.h): P A Q = L U in the same convention
    lapack_getc2(A, P, L, U, Q);
    return;
  }
  MatrixXcd a(N,N);
  for(i=0;i<N;i++){
    for(j=0;j<N;j++){
      a(i,j) = A.M[i*N+j];
    }// for j
  }// for i

  Eigen::FullPivLU<MatrixXcd> lu(a);


  MatrixXcd l(N,N); 
  l = lu.matrixLU().triangularView<StrictlyLower>();

  MatrixXcd u(N,N);
  u = lu.matrixLU().triangularView<Upper>();

  MatrixXcd p(N,N);
  p = lu.permutationP(); 

  MatrixXcd q(N,N);
  q = lu.permutationQ(); 


  for(i=0;i<N;i++){
    for(j=0;j<N;j++){
      P.M[i*N+j] = p(i,j);

      L.M[i*N+j] = l(i,j);
      if(i==j){   L.M[i*N+j] = 1.0;  }

      U.M[i*N+j] = u(i,j);

      Q.M[i*N+j] = q(i,j);

    }// for j
  }// for i
  
}




// the factor shapes of A (m x n) = U (m x m) S (m x n) V^H (n x n) - those of lapack_gesdd, demanded on the
// Eigen path too
static void require_svd_shapes(const char* fn, CMATRIX& A, CMATRIX& U, CMATRIX& S, CMATRIX& V){
  int m = A.n_rows;
  int n = A.n_cols;
  if(m < 1 || n < 1 || U.n_rows != m || U.n_cols != m || S.n_rows != m || S.n_cols != n || V.n_rows != n || V.n_cols != n){
    std::cout<<"Error in libmeigen::"<<fn<<" : for A "<<m<<" x "<<n<<" the factors must be U "<<m<<" x "<<m
             <<", S "<<m<<" x "<<n<<", V "<<n<<" x "<<n<<", got U "<<U.n_rows<<" x "<<U.n_cols<<", S "<<S.n_rows
             <<" x "<<S.n_cols<<", V "<<V.n_rows<<" x "<<V.n_cols<<"\n";
    std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
  }
}


void JacobiSVD_decomposition(CMATRIX& A, CMATRIX& U, CMATRIX& S, CMATRIX& V){
/** 

   A wrapper of Eigen::JacobiSVD<MatrixXd> class

   A - the source matrix:  A = U * S * V^{*}
   U - left matrix
   S - singular values matrix
   V - right matrix

*/

  int m = A.n_rows;
  int n = A.n_cols;
  int i,j;
  require_svd_shapes("JacobiSVD_decomposition", A, U, S, V);

  if(have_lapack() && std::max(m, n) >= get_lapack_svd_min_n()){
    // LAPACK divide-and-conquer SVD (blas_interface.h): A = U S V^H in the same convention
    lapack_gesdd(A, U, S, V);
    return;
  }
  MatrixXcd a(m,n);
  for(i=0;i<m;i++){
    for(j=0;j<n;j++){ a(i,j) = A.M[i*n+j]; }
  }

  Eigen::JacobiSVD<Eigen::MatrixXcd> svd(a, Eigen::ComputeFullU | Eigen::ComputeFullV);
  const MatrixXcd& u = svd.matrixU();   // m x m
  const MatrixXcd& v = svd.matrixV();   // n x n
  int mn = std::min(m, n);              // the singular values, descending

  for(i=0;i<m;i++){
    for(j=0;j<m;j++){ U.M[i*m+j] = u(i,j); }
  }
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){ V.M[i*n+j] = v(i,j); }
  }
  S = complex<double>(0.0, 0.0);
  for(i=0;i<mn;i++){ S.M[i*n+i] = complex<double>(svd.singularValues()(i), 0.0); }

}


void BDCSVD_decomposition(CMATRIX& A, CMATRIX& U, CMATRIX& S, CMATRIX& V){
/** 

   A wrapper of Eigen::BDCSVD<MatrixXd> class

   A - the source matrix:  A = U * S * V^{*}
   U - left matrix
   S - singular values matrix
   V - right matrix

*/

  int m = A.n_rows;
  int n = A.n_cols;
  int i,j;
  require_svd_shapes("BDCSVD_decomposition", A, U, S, V);

  if(have_lapack() && std::max(m, n) >= get_lapack_svd_min_n()){
    // LAPACK divide-and-conquer SVD (blas_interface.h): A = U S V^H in the same convention
    lapack_gesdd(A, U, S, V);
    return;
  }
  MatrixXcd a(m,n);
  for(i=0;i<m;i++){
    for(j=0;j<n;j++){ a(i,j) = A.M[i*n+j]; }
  }

  Eigen::BDCSVD<Eigen::MatrixXcd> svd(a, Eigen::ComputeFullU | Eigen::ComputeFullV);
  const MatrixXcd& u = svd.matrixU();   // m x m
  const MatrixXcd& v = svd.matrixV();   // n x n
  int mn = std::min(m, n);              // the singular values, descending

  for(i=0;i<m;i++){
    for(j=0;j<m;j++){ U.M[i*m+j] = u(i,j); }
  }
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){ V.M[i*n+j] = v(i,j); }
  }
  S = complex<double>(0.0, 0.0);
  for(i=0;i<mn;i++){ S.M[i*n+i] = complex<double>(svd.singularValues()(i), 0.0); }

}



void LLT_decomposition(MATRIX& A, MATRIX& L){
/** A wrapper of Eigen::LLT<MatrixXd>.matrixL()
   A - the source matrix
   L - lower triangular

  Really, what happens is:

  A = L * L.T()

*/

  int N = A.n_cols;
  int i,j;

  if(have_lapack() && N >= get_lapack_llt_min_n()){
    // LAPACK Cholesky factor (blas_interface.h): a non-positive-definite A throws here
    lapack_potrf(A, L);
    return;
  }
  MatrixXd a(N,N);
  for(i=0;i<N;i++){
    for(j=0;j<N;j++){
      a(i,j) = A.M[i*N+j];
    }// for j
  }// for i

  // Eigen's LLT reads one triangle and reports a non-positive-definite input only through info():
  // both silent outcomes (a factor of the mirrored lower triangle, a NaN or partial factor) are refused
  double amax_ = 0.0, sdev_ = 0.0;
  for(i=0;i<N;i++){
    for(j=0;j<N;j++){
      amax_ = std::max(amax_, std::abs(A.M[i*N+j]));
      sdev_ = std::max(sdev_, std::abs(A.M[i*N+j] - A.M[j*N+i]));
    }// for j
  }// for i
  if(sdev_ > 1e-8 * std::max(amax_, 1.0)){
    cout<<"Error in libmeigen::LLT_decomposition : the matrix is not symmetric (max |A - A^T| = "<<sdev_<<")\n"; std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
  }
  Eigen::LLT<MatrixXd> llt(a);
  if(llt.info() != Eigen::Success){
    cout<<"Error in libmeigen::LLT_decomposition : the matrix is not positive definite\n"; std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
  }
  Eigen::MatrixXd l( llt.matrixL() );

  for(i=0;i<N;i++){
    for(j=0;j<N;j++){
      L.M[i*N+j] = l(i,j);
    }// for j
  }// for i

}




}// namespace libmeigen
}// namespace liblibra
