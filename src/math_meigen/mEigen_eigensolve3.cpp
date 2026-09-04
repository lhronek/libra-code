/*********************************************************************************
* Copyright (C) 2019 Alexey V. Akimov
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


/// liblibra namespace
namespace liblibra{

using namespace Eigen;
using namespace std;
using namespace liblinalg;

/// libmeigen namespace
namespace libmeigen{


void solve_eigen_nosort(CMATRIX* H, CMATRIX* E, CMATRIX* C, int symm){
/** Solve the eigenvalue problem: H * C = C * E
  i-th column of C contains i-th MO (coefficients of expansion in terms of AOs)
  C[i][j] - is the weight of j-th AO in i-th MO
  More, generally C.col(i) is the eigenvector corresponding the eigenstate E_i

  int symm - is the flag for H and S matrices symmetrization:
   symm = 0 - don't symmetrize (use the input matrices as they are)  
   symm = 1 - do symmetrize them

  H * C = (N_bas x N_bas) x ( N_bas x N_mo ) = N_mo x N_mo

  C * E = (N_bas x N_mo ) x (N_mo x N_mo) = N_bas x N_mo
 

*/


  int i,j;

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

  if(have_lapack() && N_bas >= get_lapack_nosort_min_n()){
    // LAPACK zgeev (blas_interface.h) on the full matrix, then the order of the Eigen
    // ComplexEigenSolver - ascending modulus, selection swaps - so both paths return the
    // eigenpairs in the same order before the first N_mo are copied out
    CMATRIX Ef(N_bas, N_bas);
    CMATRIX Cf(N_bas, N_bas);
    if(symm==1){
      CMATRIX Hs(N_bas, N_bas);
      for(i=0;i<N_bas;i++){
        for(j=0;j<N_bas;j++){ Hs.M[i*N_bas+j] = 0.5*(H->M[i*N_bas+j] + std::conj(H->M[j*N_bas+i])); }
      }
      lapack_geev(Hs, Ef, Cf);
    }
    else{ lapack_geev(*H, Ef, Cf); }
    for(i=0;i<N_bas;i++){
      int k = i;
      double amin = std::abs(Ef.get(i,i));
      for(j=i+1;j<N_bas;j++){
        double a = std::abs(Ef.get(j,j));
        if(a < amin){ amin = a; k = j; }
      }
      if(k != i){
        complex<double> tmp = Ef.get(i,i);
        Ef.set(i, i, Ef.get(k,k));
        Ef.set(k, k, tmp);
        Cf.swap_cols(i, k);
      }
    }
    for(i=0;i<N_mo;i++){
      E->M[i*N_mo+i] = Ef.get(i,i);
      for(j=0;j<N_bas;j++){  C->M[j*N_mo+i] = Cf.get(j,i);  }// for j
    }// for i
    return;
  }


  // Wrapper matrices for Eigen3
  MatrixXcd A(N_bas,N_bas);

  if(symm==0){
    for(i=0;i<N_bas;i++){
      for(j=0;j<N_bas;j++){
        // Symmetrize, to reduce numerical errors:
        A(i,j) = H->M[i*N_bas+j];     
      }// for j
    }// for i
  }// no symmetrization

  else if(symm==1){
    for(i=0;i<N_bas;i++){
      for(j=0;j<N_bas;j++){
        // Symmetrize, to reduce numerical errors:
        A(i,j) = 0.5*(H->M[i*N_bas+j] + std::conj(H->M[j*N_bas+i]));     
      }// for j
    }// for i
  } // symmetrize


  // Solve eigenvalue problem
  ComplexEigenSolver<MatrixXcd> solution(A);
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
    cout<<"E = "<<(*E)<<endl;
    cout<<"C = "<<(*C)<<endl;
    cout<<"H = "<<(*H)<<endl;
    cout<<"A*C = "<<(*H) * (*C)<<endl;
    cout<<"C*E = "<<(*C) * (*E)<<endl;
    cout<<"A*C - C*E = "<<*H * (*C) - (*C) * (*E)<<endl;
    cout<<"C.H()*C= "<<((*C).H()) * (*C)<<endl; // This gives unity, indeed.
  }


}//void solve_eigen_nosort(CMATRIX* H, CMATRIX* E, CMATRIX* C, int symm)


void solve_eigen_nosort(CMATRIX& H, CMATRIX& E, CMATRIX& C, int symm){
/** Solve the eigenvalue problem: H * C = C * E
  i-th column of C contains i-th MO (coefficients of expansion in terms of AOs)
  C[i][j] - is the weight of j-th AO in i-th MO
  More, generally C.col(i) is the eigenvector corresponding the eigenstate E_i

  int symm - is the flag for H and S matrices symmetrization:
   symm = 0 - don't symmetrize (use the input matrices as they are)  
   symm = 1 - do symmetrize them

  H * C = (N_bas x N_bas) x (N_bas x N_mo) = N_mo x N_mo

  C * E = (N_bas x N_bas) x (N_bas x N_mo ) x (N_mo x N_mo) = N_bas x N_mo
*/ 

  solve_eigen_nosort(&H, &E, &C, symm);

}



void solve_eigen_nosort(MATRIX* H, CMATRIX* E, CMATRIX* C, int symm){
/** Solve the eigenvalue problem: H * C = C * E
  i-th column of C contains i-th MO (coefficients of expansion in terms of AOs)
  C[i][j] - is the weight of j-th AO in i-th MO
  More, generally C.col(i) is the eigenvector corresponding the eigenstate E_i

  int symm - is the flag for H and S matrices symmetrization:
   symm = 0 - don't symmetrize (use the input matrices as they are)  
   symm = 1 - do symmetrize them

  H * C = (N_bas x N_bas) x (N_bas x N_mo) = N_mo x N_mo

  C * E = (N_bas x N_mo ) x (N_mo x N_mo) = N_bas x N_mo
 

*/


  int i,j;

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

  if(have_lapack() && N_bas >= get_lapack_nosort_min_n()){
    // LAPACK dgeev (blas_interface.h): the eigenvalues in Schur order with the conjugate pairs
    // adjacent, the positive imaginary part first - the convention of the Eigen EigenSolver too
    CMATRIX Ef(N_bas, N_bas);
    CMATRIX Cf(N_bas, N_bas);
    if(symm==1){
      MATRIX Hs(N_bas, N_bas);
      for(i=0;i<N_bas;i++){
        for(j=0;j<N_bas;j++){ Hs.M[i*N_bas+j] = 0.5*(H->M[i*N_bas+j] + H->M[j*N_bas+i]); }
      }
      lapack_geev(Hs, Ef, Cf);
    }
    else{ lapack_geev(*H, Ef, Cf); }
    *E = complex<double>(0.0, 0.0);
    for(i=0;i<N_mo;i++){
      E->M[i*N_mo+i] = Ef.get(i,i);
      for(j=0;j<N_bas;j++){  C->M[j*N_mo+i] = Cf.get(j,i);  }// for j
    }// for i
    return;
  }


  *E = 0.0;
  *C = 0.0;

  // Wrapper matrices for Eigen3
  MatrixXd A(N_bas,N_bas);

  if(symm==0){
    for(i=0;i<N_bas;i++){
      for(j=0;j<N_bas;j++){
        // Symmetrize, to reduce numerical errors:
        A(i,j) = H->M[i*N_bas+j];     
      }// for j
    }// for i
  }// no symmetrization

  else if(symm==1){
    for(i=0;i<N_bas;i++){
      for(j=0;j<N_bas;j++){
        // Symmetrize, to reduce numerical errors:
        A(i,j) = 0.5*(H->M[i*N_bas+j] + H->M[j*N_bas+i]);     
      }// for j
    }// for i
  } // symmetrize


  // Solve eigenvalue problem
  EigenSolver<MatrixXd> solution(A);
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
    cout<<"E = "<<(*E)<<endl;
    cout<<"C = "<<(*C)<<endl;
    cout<<"H = "<<(*H)<<endl;
//    cout<<"A*C = "<<(*H) * (*C)<<endl;
//    cout<<"C*E = "<<(*C) * (*E)<<endl;
//    cout<<"A*C - C*E = "<<*H * (*C) - (*C) * (*E)<<endl;
//    cout<<"C.H()*C= "<<((*C).H()) * (*C)<<endl; // This gives unity, indeed.

  }

}//void solve_eigen_nosort(MATRIX* H, CMATRIX* E, CMATRIX* C, int symm)




void solve_eigen_nosort(MATRIX& H, CMATRIX& E, CMATRIX& C, int symm){
/** Solve the eigenvalue problem: H * C = C * E
  i-th column of C contains i-th MO (coefficients of expansion in terms of AOs)
  C[i][j] - is the weight of j-th AO in i-th MO
  More, generally C.col(i) is the eigenvector corresponding the eigenstate E_i

  int symm - is the flag for H and S matrices symmetrization:
   symm = 0 - don't symmetrize (use the input matrices as they are)  
   symm = 1 - do symmetrize them

  H * C = (N_bas x N_bas) x (N_bas x N_mo) = N_mo x N_mo

  C * E = (N_bas x N_mo) x (N_mo x N_mo) = N_bas x N_mo
 

*/

  solve_eigen_nosort(&H, &E, &C, symm);

}



void solve_eigen_nosort(MATRIX* H, MATRIX* E, MATRIX* C, int symm){
/** Solve the eigenvalue problem: H * C = C * E
  i-th column of C contains i-th MO (coefficients of expansion in terms of AOs)
  C[i][j] - is the weight of j-th AO in i-th MO
  More, generally C.col(i) is the eigenvector corresponding the eigenstate E_i

  int symm - is the flag for H and S matrices symmetrization:
   symm = 0 - don't symmetrize (use the input matrices as they are)  
   symm = 1 - do symmetrize them

  H * C = (N_bas x N_bas) x (N_bas x N_mo) = N_mo x N_mo

  C * E = (N_bas x N_mo ) x (N_mo x N_mo) = N_bas x N_mo
 

*/


  int i,j;

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

  if(have_lapack() && N_bas >= get_lapack_nosort_min_n()){
    // LAPACK dgeev (blas_interface.h) with the real parts of the eigenpairs kept, as the Eigen path does
    CMATRIX Ef(N_bas, N_bas);
    CMATRIX Cf(N_bas, N_bas);
    if(symm==1){
      MATRIX Hs(N_bas, N_bas);
      for(i=0;i<N_bas;i++){
        for(j=0;j<N_bas;j++){ Hs.M[i*N_bas+j] = 0.5*(H->M[i*N_bas+j] + H->M[j*N_bas+i]); }
      }
      lapack_geev(Hs, Ef, Cf);
    }
    else{ lapack_geev(*H, Ef, Cf); }
    *E = 0.0;
    for(i=0;i<N_mo;i++){
      E->M[i*N_mo+i] = Ef.get(i,i).real();
      for(j=0;j<N_bas;j++){  C->M[j*N_mo+i] = Cf.get(j,i).real();  }// for j
    }// for i
    return;
  }


  *E = 0.0;
  *C = 0.0;

  // Wrapper matrices for Eigen3
  MatrixXd A(N_bas,N_bas);

  if(symm==0){
    for(i=0;i<N_bas;i++){
      for(j=0;j<N_bas;j++){
        // Symmetrize, to reduce numerical errors:
        A(i,j) = H->M[i*N_bas+j];     
      }// for j
    }// for i
  }// no symmetrization

  else if(symm==1){
    for(i=0;i<N_bas;i++){
      for(j=0;j<N_bas;j++){
        // Symmetrize, to reduce numerical errors:
        A(i,j) = 0.5*(H->M[i*N_bas+j] + H->M[j*N_bas+i]);     
      }// for j
    }// for i
  } // symmetrize


  // Solve eigenvalue problem
  EigenSolver<MatrixXd> solution(A);
  if(solution.info()!=Success){ cout<<"Eigen fails\n";   std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above"); }


  // Copy results into output matrices
  for(i=0;i<N_mo;i++){

     E->M[i*N_mo+i] = solution.eigenvalues()[i].real(); 
     for(j=0;j<N_bas;j++){  C->M[j*N_mo+i] = solution.eigenvectors().col(i)[j].real();    }// for j

  }// for i

  if(0){
    // Checking solution - for debug purposes, inactive by default
    cout<<"in solve_eigen (with MATRIX):\n";
    cout<<"A = "<<A<<endl;
    cout<<"E = "<<(*E)<<endl;
    cout<<"C = "<<(*C)<<endl;
    cout<<"H = "<<(*H)<<endl;
    cout<<"A*C = "<<(*H) * (*C)<<endl;
    cout<<"C*E = "<<(*C) * (*E)<<endl;
    cout<<"A*C - C*E = "<<*H * (*C) - (*C) * (*E)<<endl;
    cout<<"C.T()*C= "<<((*C).T()) * (*C)<<endl; // This gives unity, indeed.
  }


}//void solve_eigen_nosort(MATRIX* H, MATRIX* E, MATRIX* C, int symm)



void solve_eigen_nosort(MATRIX& H, MATRIX& E, MATRIX& C, int symm){
/** Solve the eigenvalue problem: H * C = C * E
  i-th column of C contains i-th MO (coefficients of expansion in terms of AOs)
  C[i][j] - is the weight of j-th AO in i-th MO
  More, generally C.col(i) is the eigenvector corresponding the eigenstate E_i

  int symm - is the flag for H and S matrices symmetrization:
   symm = 0 - don't symmetrize (use the input matrices as they are)  
   symm = 1 - do symmetrize them

  H * C = (N_bas x N_bas) x (N_bas x N_mo) = N_mo x N_mo

  C * E = (N_bas x N_mo ) x (N_mo x N_mo) = N_bas x N_mo
 
*/

  solve_eigen_nosort(&H, &E, &C, symm);

}



}// namespace libmeigen
}// namespace liblibra
