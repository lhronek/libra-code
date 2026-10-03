/*********************************************************************************
* Copyright (C) 2015-2019 Alexey V. Akimov
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
#include <Eigen/LU>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/Core>
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




void solve_eigen(CMATRIX* H, CMATRIX* E, CMATRIX* C, int symm){
/** Solve H * C = C * E
 i-th column of C contains i-th MO (coefficients of expansion in terms of AOs)
 C[i][j] - is the weight of j-th AO in i-th MO

 ACHTUNG:    EigenSolver<MatrixXd> solution(A); - messes up the ordering of eigenvalues
 GeneralizedSelfAdjointEigenSolver<MatrixXd> solution(A,B); does not!, so 
 here we actually specialize the generalized eigensolver.
*/

  int N_bas = H->n_cols;
  if(have_lapack() && N_bas >= get_lapack_eigen_min_n()){
    // the standard problem goes straight to zheevd (blas_interface.h) - no identity metric,
    // symm = 1 hands it the Hermitized matrix, symm = 0 keeps the kernel's fail-loud guard
    if(symm==1){
      CMATRIX Hs(N_bas, N_bas);
      for(int i=0;i<N_bas;i++){
        for(int j=0;j<N_bas;j++){ Hs.M[i*N_bas+j] = 0.5*(H->M[i*N_bas+j] + std::conj(H->M[j*N_bas+i])); }
      }
      lapack_heevd(Hs, *E, *C);
    }
    else{ lapack_heevd(*H, *E, *C); }
    return;
  }
  CMATRIX* S; S = new CMATRIX(N_bas,N_bas); S->diag(N_bas, 1.0);

  solve_eigen(H, S, E, C, symm);  

  delete S;
}


void solve_eigen(CMATRIX& H, CMATRIX& E, CMATRIX& C, int symm){
/** Solve H * C = C * E */
  solve_eigen(&H, &E, &C, symm);  
}





void solve_eigen(MATRIX* H, CMATRIX* E, CMATRIX* C, int symm){
/** Solve H * C = C * E
 i-th column of C contains i-th MO (coefficients of expansion in terms of AOs)
 C[i][j] - is the weight of j-th AO in i-th MO

 ACHTUNG:    EigenSolver<MatrixXd> solution(A); - messes up the ordering of eigenvalues
 GeneralizedSelfAdjointEigenSolver<MatrixXd> solution(A,B); does not!, so 
 here we actually specialize the generalized eigensolver.
*/

  if(H->n_rows!=H->n_cols){
    std::cout<<"Error in solve_eigen: The H matrix must be square\n";
    std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
  }
  if(H->n_cols!=C->n_rows){  // N_bas
    std::cout<<"Error in solve_eigen: The # of cols of H and the # of rows of C matrices must be equal\n";
    std::cout<<"Make sure C is allocated \n";
    std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
  }
  if(C->n_cols!=E->n_rows){  // N_mo
    std::cout<<"Error in solve_eigen: The # of cols of C and the # of rows of E matrices must be equal\n";
    std::cout<<"Make sure C and E are allocated \n";
    std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
  }
  if(C->n_cols!=E->n_cols){  // N_mo
    std::cout<<"Error in solve_eigen: The # of cols of C and the # of cols of E matrices must be equal\n";
    std::cout<<"Make sure C and E are allocated \n";
    std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
  }
  if(C->n_cols<1 || C->n_cols>H->n_cols){  // the eigenpairs requested
    std::cout<<"Error in solve_eigen: The # of cols of C (the eigenpairs requested) must be between 1 and the size of H\n";
    std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
  }
  int N_bas = H->n_cols;
  if(have_lapack() && N_bas >= get_lapack_eigen_min_n()){
    // the real standard problem goes straight to dsyevd (blas_interface.h), the real eigenpairs
    // copied into the complex outputs with zero imaginary parts
    int N_mo = C->n_cols;
    MATRIX Er(N_mo, N_mo);
    MATRIX Cr(N_bas, N_mo);
    if(symm==1){
      MATRIX Hs(N_bas, N_bas);
      for(int i=0;i<N_bas;i++){
        for(int j=0;j<N_bas;j++){ Hs.M[i*N_bas+j] = 0.5*(H->M[i*N_bas+j] + H->M[j*N_bas+i]); }
      }
      lapack_syevd(Hs, Er, Cr);
    }
    else{ lapack_syevd(*H, Er, Cr); }
    *E = complex<double>(0.0, 0.0);
    for(int i=0;i<N_mo;i++){
      E->M[i*N_mo+i] = complex<double>(Er.M[i*N_mo+i], 0.0);
      for(int j=0;j<N_bas;j++){  C->M[j*N_mo+i] = complex<double>(Cr.M[j*N_mo+i], 0.0);  }
    }
    return;
  }
  MATRIX* S; S = new MATRIX(N_bas,N_bas); S->diag(N_bas, 1.0);

  solve_eigen(H, S, E, C, symm);  

  delete S;
}


void solve_eigen(MATRIX& H, CMATRIX& E, CMATRIX& C, int symm){
/** Solve H * C = C * E */
  solve_eigen(&H, &E, &C, symm);  
}





void solve_eigen(MATRIX* H, MATRIX* E, MATRIX* C, int symm){
/** Solve H * C = C * E
 i-th column of C contains i-th MO (coefficients of expansion in terms of AOs)
 C[i][j] - is the weight of j-th AO in i-th MO

 ACHTUNG:    EigenSolver<MatrixXd> solution(A); - messes up the ordering of eigenvalues
 GeneralizedSelfAdjointEigenSolver<MatrixXd> solution(A,B); does not!, so 
 here we actually specialize the generalized eigensolver.
*/

  int N_bas = H->n_cols;
  if(have_lapack() && N_bas >= get_lapack_eigen_min_n()){
    // the standard problem goes straight to dsyevd (blas_interface.h) - no identity metric,
    // symm = 1 hands it the symmetrized matrix, symm = 0 keeps the kernel's fail-loud guard
    if(symm==1){
      MATRIX Hs(N_bas, N_bas);
      for(int i=0;i<N_bas;i++){
        for(int j=0;j<N_bas;j++){ Hs.M[i*N_bas+j] = 0.5*(H->M[i*N_bas+j] + H->M[j*N_bas+i]); }
      }
      lapack_syevd(Hs, *E, *C);
    }
    else{ lapack_syevd(*H, *E, *C); }
    return;
  }
  MATRIX* S; S = new MATRIX(N_bas,N_bas); S->diag(N_bas, 1.0);

  solve_eigen(H, S, E, C, symm);  

  delete S;
}


void solve_eigen(MATRIX& H, MATRIX& E, MATRIX& C, int symm){
/** Solve H * C = C * E */
  solve_eigen(&H, &E, &C, symm);  
}




}// namespace libmeigen
}// namespace liblibra
