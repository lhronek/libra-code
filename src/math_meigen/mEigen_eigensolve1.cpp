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
#include <stdexcept>
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
#include <cstdlib>

/// liblibra namespace
namespace liblibra{

using namespace Eigen;
using namespace std;
using namespace liblinalg;

/// libmeigen namespace
namespace libmeigen{

// The generalized self-adjoint solver reads ONE triangle of its inputs:
// with symm = 0 (use as-is) a non-Hermitian H or S would be solved silently
// as its mirror. Fail loud instead - the general (non-Hermitian) problems
// have their own entry, solve_eigen_nosort.
template <typename T1>
static void require_hermitian_pair(const T1* H, const T1* S, int N, const char* fn){
  double amax = 0.0, dev = 0.0;
  for(int i = 0; i < N; i++){
    for(int j = 0; j < N; j++){
      amax = std::max(amax, std::abs(H[i*N+j]));
      amax = std::max(amax, std::abs(S[i*N+j]));
      dev = std::max(dev, std::abs(H[i*N+j] - std::conj(H[j*N+i])));
      dev = std::max(dev, std::abs(S[i*N+j] - std::conj(S[j*N+i])));
    }
  }
  if(dev > 1e-8 * std::max(amax, 1.0)){
    std::cout<<"Error in "<<fn<<"(): symm = 0 with a non-Hermitian H or S (max |A - A^H| = "
        <<dev<<") - the generalized self-adjoint solver would read one triangle only;"
        <<" pass symm = 1 to Hermitize, or use solve_eigen_nosort for a general matrix\n";
    std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
  }
}



// The matrix size from which solve_eigen takes the LAPACK kernels (blas_interface.h). Measured
// crossovers against the Eigen solver (one OpenBLAS thread, 4-core desktop, 2026-09-04): real
// problems ~128-256, complex standard ~32-64, complex generalized ~64-96, then 2-6x LAPACK gains
// at n = 600. Below the threshold the Eigen solver is the faster one for the small model
// Hamiltonians and its eigenvector signs are the ones the small-model certifiers pin. Default
// LAPACK_EIGEN_MIN_N_DEFAULT, overridden once by the LIBRA_LAPACK_EIGEN_MIN_N environment
// variable and at any time by set_lapack_eigen_min_n
static const int LAPACK_EIGEN_MIN_N_DEFAULT = 96;
static int lapack_eigen_min_n = -1;
// solve_eigen_nosort: the Eigen general solvers fall behind dgeev/zgeev already at a few states
// (real 8: 30 vs 36 us, 32: 2.9 ms vs 0.36 ms, 600: 748 s vs 0.9 s - measured 2026-09-04), so the
// general kernels take over almost at once
static const int LAPACK_NOSORT_MIN_N_DEFAULT = 8;
static int lapack_nosort_min_n = -1;

static int read_min_n(int& value, int dflt, const char* env_name, const char* fn){
  if(value < 1){
    const char* env = std::getenv(env_name);
    value = dflt;
    if(env != NULL && env[0] != 0){
      int v = std::atoi(env);
      if(v < 1){
        std::cout<<"Error in libmeigen::"<<fn<<" : "<<env_name<<" must be a positive integer, got '"<<env<<"'\n";
        std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
      }
      value = v;
    }
  }
  return value;
}

static void write_min_n(int& value, int n, const char* fn){
  if(n < 1){
    std::cout<<"Error in libmeigen::"<<fn<<" : the size must be a positive integer, got "<<n<<"\n";
    std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
  }
  value = n;
}

int get_lapack_eigen_min_n(){ return read_min_n(lapack_eigen_min_n, LAPACK_EIGEN_MIN_N_DEFAULT, "LIBRA_LAPACK_EIGEN_MIN_N", "get_lapack_eigen_min_n"); }
void set_lapack_eigen_min_n(int n){ write_min_n(lapack_eigen_min_n, n, "set_lapack_eigen_min_n"); }
int get_lapack_nosort_min_n(){ return read_min_n(lapack_nosort_min_n, LAPACK_NOSORT_MIN_N_DEFAULT, "LIBRA_LAPACK_NOSORT_MIN_N", "get_lapack_nosort_min_n"); }
void set_lapack_nosort_min_n(int n){ write_min_n(lapack_nosort_min_n, n, "set_lapack_nosort_min_n"); }


void solve_eigen(CMATRIX* H, CMATRIX* S, CMATRIX* E, CMATRIX* C, int symm){
/** Solve the generalized eigenvalue problem: H * C = S * C * E
  i-th column of C contains i-th MO (coefficients of expansion in terms of AOs)
  C[i][j] - is the weight of j-th AO in i-th MO
  More, generally C.col(i) is the eigenvector corresponding the eigenstate E_i

  int symm - is the flag for H and S matrices symmetrization:
   symm = 0 - don't symmetrize (use the input matrices as they are)  
   symm = 1 - do symmetrize them

  H * C = (N_bas x N_bas)   x  ( N_bas x N_mo ) = N_mo x N_mo

  S * C * E = (N_bas x N_bas) x (N_bas x N_mo ) x (N_mo x N_mo) = N_bas x N_mo
 

*/


  int i,j;

  if(H->n_rows!=S->n_rows){  // N_bas
    std::cout<<"Error in solve_eigen: The # of rows of the H and S matrices must be equal\n";
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
  // The above two conditions imply E is square matrix
  int N_bas = H->n_cols;
  int N_mo  = C->n_cols;

  if(have_lapack() && N_bas >= get_lapack_eigen_min_n()){
    // LAPACK divide-and-conquer path (blas_interface.h): the kernel reads one triangle and
    // refuses a non-Hermitian input, so symm = 0 keeps the fail-loud guard and symm = 1 hands
    // it the Hermitized pair. The lowest N_mo eigenpairs land in E and C as in the Eigen path
    if(symm==1){
      CMATRIX Hs(N_bas, N_bas);
      CMATRIX Ss(N_bas, N_bas);
      for(i=0;i<N_bas;i++){
        for(j=0;j<N_bas;j++){
          Hs.M[i*N_bas+j] = 0.5*(H->M[i*N_bas+j] + std::conj(H->M[j*N_bas+i]));
          Ss.M[i*N_bas+j] = 0.5*(S->M[i*N_bas+j] + std::conj(S->M[j*N_bas+i]));
        }// for j
      }// for i
      lapack_hegvd(Hs, Ss, *E, *C);
    }
    else{ lapack_hegvd(*H, *S, *E, *C); }
    return;
  }


  // Wrapper matrices for Eigen3
  MatrixXcd A(N_bas,N_bas), B(N_bas,N_bas);

  if(symm==0){
    require_hermitian_pair(H->M, S->M, N_bas, "solve_eigen");
    for(i=0;i<N_bas;i++){
      for(j=0;j<N_bas;j++){
        A(i,j) = H->M[i*N_bas+j];     
        B(i,j) = S->M[i*N_bas+j];     
      }// for j
    }// for i
  }// no symmetrization

  else if(symm==1){
    for(i=0;i<N_bas;i++){
      for(j=0;j<N_bas;j++){
        // Symmetrize, to reduce numerical errors:
        A(i,j) = 0.5*(H->M[i*N_bas+j] + std::conj(H->M[j*N_bas+i]));     
        B(i,j) = 0.5*(S->M[i*N_bas+j] + std::conj(S->M[j*N_bas+i]));     
      }// for j
    }// for i
  } // symmetrize


  // Solve eigenvalue problem
  GeneralizedSelfAdjointEigenSolver<MatrixXcd> solution(A,B);
  if(solution.info()!=Success){ cout<<"Eigen fails\n";   std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above"); }


  // Copy results into output matrices
  for(i=0;i<N_mo;i++){

     E->M[i*N_mo+i] = solution.eigenvalues()[i]; 

     for(j=0;j<N_bas;j++){  C->M[j*N_mo+i] = solution.eigenvectors().col(i)[j];    }// for j

  }// for i

  if(0){
    // Checking solution - for debug purposes, inactive by default
    cout<<"in solve_eigen (with MATRIX):\n";
    cout<<"A = "<<A<<endl;
    cout<<"B = "<<B<<endl;
    cout<<"E = "<<(*E)<<endl;
    cout<<"C = "<<(*C)<<endl;
    cout<<"S = "<<(*S)<<endl;
    cout<<"H = "<<(*H)<<endl;
    cout<<"A*C = "<<(*H) * (*C)<<endl;
    cout<<"S*C*E = "<<(*S) * (*C) * (*E)<<endl;
    cout<<"A*C - S*C*E = "<<*H * (*C) - (*S) * (*C) * (*E)<<endl;
    cout<<"C.H()*S*C= "<<((*C).H()) * (*S) * (*C)<<endl; // This gives unity, indeed.
  }


}//void solve_eigen(CMATRIX* H, CMATRIX* S, CMATRIX* E, CMATRIX* C, int symm)


void solve_eigen(CMATRIX& H, CMATRIX& S, CMATRIX& E, CMATRIX& C, int symm){
/** Solve the generalized eigenvalue problem: H * C = S * C * E
  i-th column of C contains i-th MO (coefficients of expansion in terms of AOs)
  C[i][j] - is the weight of j-th AO in i-th MO
  More, generally C.col(i) is the eigenvector corresponding the eigenstate E_i

  int symm - is the flag for H and S matrices symmetrization:
   symm = 0 - don't symmetrize (use the input matrices as they are)  
   symm = 1 - do symmetrize them

  H * C = (N_bas x N_bas)   x  ( N_bas x N_mo ) = N_mo x N_mo

  S * C * E = (N_bas x N_bas) x (N_bas x N_mo ) x (N_mo x N_mo) = N_bas x N_mo
*/ 

  solve_eigen(&H, &S, &E, &C, symm);

}



void solve_eigen(MATRIX* H, MATRIX* S, CMATRIX* E, CMATRIX* C, int symm){
/** Solve the generalized eigenvalue problem: H * C = S * C * E
  i-th column of C contains i-th MO (coefficients of expansion in terms of AOs)
  C[i][j] - is the weight of j-th AO in i-th MO
  More, generally C.col(i) is the eigenvector corresponding the eigenstate E_i

  int symm - is the flag for H and S matrices symmetrization:
   symm = 0 - don't symmetrize (use the input matrices as they are)  
   symm = 1 - do symmetrize them

  H * C = (N_bas x N_bas)   x  ( N_bas x N_mo ) = N_mo x N_mo

  S * C * E = (N_bas x N_bas) x (N_bas x N_mo ) x (N_mo x N_mo) = N_bas x N_mo
 

*/


  int i,j;

  if(H->n_rows!=S->n_rows){  // N_bas
    std::cout<<"Error in solve_eigen: The # of rows of the H and S matrices must be equal\n";
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
  // The above two conditions imply E is square matrix
  int N_bas = H->n_cols;
  int N_mo  = C->n_cols;

  if(have_lapack() && N_bas >= get_lapack_eigen_min_n()){
    // LAPACK path through the real kernel (blas_interface.h) - the real symmetric pair has real
    // eigenpairs, copied into the complex outputs with zero imaginary parts. symm = 0 keeps the
    // kernel's fail-loud Hermiticity guard, symm = 1 hands it the symmetrized pair
    MATRIX Er(N_mo, N_mo);
    MATRIX Cr(N_bas, N_mo);
    if(symm==1){
      MATRIX Hs(N_bas, N_bas);
      MATRIX Ss(N_bas, N_bas);
      for(i=0;i<N_bas;i++){
        for(j=0;j<N_bas;j++){
          Hs.M[i*N_bas+j] = 0.5*(H->M[i*N_bas+j] + H->M[j*N_bas+i]);
          Ss.M[i*N_bas+j] = 0.5*(S->M[i*N_bas+j] + S->M[j*N_bas+i]);
        }// for j
      }// for i
      lapack_sygvd(Hs, Ss, Er, Cr);
    }
    else{ lapack_sygvd(*H, *S, Er, Cr); }
    *E = complex<double>(0.0, 0.0);
    for(i=0;i<N_mo;i++){
      E->M[i*N_mo+i] = complex<double>(Er.M[i*N_mo+i], 0.0);
      for(j=0;j<N_bas;j++){  C->M[j*N_mo+i] = complex<double>(Cr.M[j*N_mo+i], 0.0);  }// for j
    }// for i
    return;
  }


  *E = 0.0;
  *C = 0.0;

  // Wrapper matrices for Eigen3
  MatrixXcd A(N_bas,N_bas), B(N_bas,N_bas);

  if(symm==0){
    require_hermitian_pair(H->M, S->M, N_bas, "solve_eigen");
    for(i=0;i<N_bas;i++){
      for(j=0;j<N_bas;j++){
        A(i,j) = complex<double>(H->M[i*N_bas+j], 0.0);     
        B(i,j) = complex<double>(S->M[i*N_bas+j], 0.0);     
      }// for j
    }// for i
  }// no symmetrization

  else if(symm==1){
    for(i=0;i<N_bas;i++){
      for(j=0;j<N_bas;j++){
        // Symmetrize, to reduce numerical errors:
        A(i,j) = complex<double>( 0.5*(H->M[i*N_bas+j] + H->M[j*N_bas+i]), 0.0);     
        B(i,j) = complex<double>( 0.5*(S->M[i*N_bas+j] + S->M[j*N_bas+i]), 0.0);     
      }// for j
    }// for i
  } // symmetrize


  // Solve eigenvalue problem
  GeneralizedSelfAdjointEigenSolver<MatrixXcd> solution(A,B);
  if(solution.info()!=Success){ cout<<"Eigen fails\n";   std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above"); }


  // Copy results into output matrices
  for(i=0;i<N_mo;i++){

     E->M[i*N_mo+i] = solution.eigenvalues()[i]; 

     for(j=0;j<N_bas;j++){  C->M[j*N_mo+i] = solution.eigenvectors().col(i)[j];    }// for j

  }// for i

  if(0){
    // Checking solution - for debug purposes, inactive by default
    cout<<"in solve_eigen (with MATRIX):\n";
    cout<<"A = "<<A<<endl;
    cout<<"B = "<<B<<endl;
    cout<<"E = "<<(*E)<<endl;
    cout<<"C = "<<(*C)<<endl;
    cout<<"S = "<<(*S)<<endl;
    cout<<"H = "<<(*H)<<endl;
//    cout<<"A*C = "<<(*H) * (*C)<<endl;
//    cout<<"S*C*E = "<<(*S) * (*C) * (*E)<<endl;
//    cout<<"A*C - S*C*E = "<<*H * (*C) - (*S) * (*C) * (*E)<<endl;
//    cout<<"C.H()*S*C= "<<((*C).H()) * (*S) * (*C)<<endl; // This gives unity, indeed.

  }


}//void solve_eigen(MATRIX* H, MATRIX* S, CMATRIX* E, CMATRIX* C, int symm)



void solve_eigen(MATRIX& H, MATRIX& S, CMATRIX& E, CMATRIX& C, int symm){
/** Solve the generalized eigenvalue problem: H * C = S * C * E
  i-th column of C contains i-th MO (coefficients of expansion in terms of AOs)
  C[i][j] - is the weight of j-th AO in i-th MO
  More, generally C.col(i) is the eigenvector corresponding the eigenstate E_i

  int symm - is the flag for H and S matrices symmetrization:
   symm = 0 - don't symmetrize (use the input matrices as they are)  
   symm = 1 - do symmetrize them

  H * C = (N_bas x N_bas)   x  ( N_bas x N_mo ) = N_mo x N_mo

  S * C * E = (N_bas x N_bas) x (N_bas x N_mo ) x (N_mo x N_mo) = N_bas x N_mo
 

*/

  solve_eigen(&H, &S, &E, &C, symm);

}



void solve_eigen(MATRIX* H, MATRIX* S, MATRIX* E, MATRIX* C, int symm){
/** Solve the generalized eigenvalue problem: H * C = S * C * E
  i-th column of C contains i-th MO (coefficients of expansion in terms of AOs)
  C[i][j] - is the weight of j-th AO in i-th MO
  More, generally C.col(i) is the eigenvector corresponding the eigenstate E_i

  int symm - is the flag for H and S matrices symmetrization:
   symm = 0 - don't symmetrize (use the input matrices as they are)  
   symm = 1 - do symmetrize them

  H * C = (N_bas x N_bas)   x  ( N_bas x N_mo ) = N_mo x N_mo

  S * C * E = (N_bas x N_bas) x (N_bas x N_mo ) x (N_mo x N_mo) = N_bas x N_mo
 

*/


  int i,j;

  if(H->n_rows!=S->n_rows){  // N_bas
    std::cout<<"Error in solve_eigen: The # of rows of the H and S matrices must be equal\n";
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
  // The above two conditions imply E is square matrix
  int N_bas = H->n_cols;
  int N_mo  = C->n_cols;

  if(have_lapack() && N_bas >= get_lapack_eigen_min_n()){
    // LAPACK divide-and-conquer path (blas_interface.h): symm = 0 keeps the kernel's fail-loud
    // Hermiticity guard, symm = 1 hands it the symmetrized pair
    if(symm==1){
      MATRIX Hs(N_bas, N_bas);
      MATRIX Ss(N_bas, N_bas);
      for(i=0;i<N_bas;i++){
        for(j=0;j<N_bas;j++){
          Hs.M[i*N_bas+j] = 0.5*(H->M[i*N_bas+j] + H->M[j*N_bas+i]);
          Ss.M[i*N_bas+j] = 0.5*(S->M[i*N_bas+j] + S->M[j*N_bas+i]);
        }// for j
      }// for i
      lapack_sygvd(Hs, Ss, *E, *C);
    }
    else{ lapack_sygvd(*H, *S, *E, *C); }
    return;
  }


  *E = 0.0;
  *C = 0.0;

  // Wrapper matrices for Eigen3
  MatrixXd A(N_bas,N_bas), B(N_bas,N_bas);

  if(symm==0){
    require_hermitian_pair(H->M, S->M, N_bas, "solve_eigen");
    for(i=0;i<N_bas;i++){
      for(j=0;j<N_bas;j++){
        A(i,j) = H->M[i*N_bas+j];     
        B(i,j) = S->M[i*N_bas+j];     
      }// for j
    }// for i
  }// no symmetrization

  else if(symm==1){
    for(i=0;i<N_bas;i++){
      for(j=0;j<N_bas;j++){
        // Symmetrize, to reduce numerical errors:
        A(i,j) = 0.5*(H->M[i*N_bas+j] + H->M[j*N_bas+i]);     
        B(i,j) = 0.5*(S->M[i*N_bas+j] + S->M[j*N_bas+i]);     
      }// for j
    }// for i
  } // symmetrize


  // Solve eigenvalue problem
  GeneralizedSelfAdjointEigenSolver<MatrixXd> solution(A,B);
  if(solution.info()!=Success){ cout<<"Eigen fails\n";   std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above"); }


  // Copy results into output matrices
  for(i=0;i<N_mo;i++){

     E->M[i*N_mo+i] = solution.eigenvalues()[i]; 

     for(j=0;j<N_bas;j++){  C->M[j*N_mo+i] = solution.eigenvectors().col(i)[j];    }// for j

  }// for i

  if(0){
    // Checking solution - for debug purposes, inactive by default
    cout<<"in solve_eigen (with MATRIX):\n";
    cout<<"A = "<<A<<endl;
    cout<<"B = "<<B<<endl;
    cout<<"E = "<<(*E)<<endl;
    cout<<"C = "<<(*C)<<endl;
    cout<<"S = "<<(*S)<<endl;
    cout<<"H = "<<(*H)<<endl;
    cout<<"A*C = "<<(*H) * (*C)<<endl;
    cout<<"S*C*E = "<<(*S) * (*C) * (*E)<<endl;
    cout<<"A*C - S*C*E = "<<*H * (*C) - (*S) * (*C) * (*E)<<endl;
    cout<<"C.T()*S*C= "<<((*C).T()) * (*S) * (*C)<<endl; // This gives unity, indeed.
  }


}//void solve_eigen(MATRIX* H, MATRIX* S, MATRIX* E, MATRIX* C, int symm)



void solve_eigen(MATRIX& H, MATRIX& S, MATRIX& E, MATRIX& C, int symm){
/** Solve the generalized eigenvalue problem: H * C = S * C * E
  i-th column of C contains i-th MO (coefficients of expansion in terms of AOs)
  C[i][j] - is the weight of j-th AO in i-th MO
  More, generally C.col(i) is the eigenvector corresponding the eigenstate E_i

  int symm - is the flag for H and S matrices symmetrization:
   symm = 0 - don't symmetrize (use the input matrices as they are)  
   symm = 1 - do symmetrize them

  H * C = (N_bas x N_bas)   x  ( N_bas x N_mo ) = N_mo x N_mo

  S * C * E = (N_bas x N_bas) x (N_bas x N_mo ) x (N_mo x N_mo) = N_bas x N_mo
 

*/

  solve_eigen(&H, &S, &E, &C, symm);

}







}// namespace libmeigen
}// namespace liblibra
