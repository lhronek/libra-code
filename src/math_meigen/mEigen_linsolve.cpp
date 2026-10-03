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
#include "../math_linalg/blas_interface.h"
#include <limits>


/// liblibra namespace
namespace liblibra{

using namespace Eigen;
using namespace std;
using namespace liblinalg;

/// libmeigen namespace
namespace libmeigen{

void least_squares_solve(MATRIX& A, MATRIX& X, MATRIX& B, int option){
/**
   Solving Ax = b for x using the BDCSVD decomposition

   A = N x M
   x = M x K
   b = N x K

*/

  int N = A.n_rows;
  int M = A.n_cols;
  int K = B.n_cols;
  int i,j;

  if(option < 0 || option > 3){
    std::cout<<"Error in least_squares_solve(): option = "<<option<<" has no branch here (implemented: 0, 1, 2, 3)\n";
    std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
  }

  if(have_lapack() && std::max(N, M) >= get_lapack_lstsq_min_n()){
    // LAPACK (blas_interface.h) with Eigen's rank threshold eps min(N, M): the SVD driver for option
    // 0 (Eigen's BDCSVD), the LU solve for option 1 on a square system (Eigen's FullPivLU), the
    // complete orthogonal factorization for the rest (Eigen's FullPivHouseholderQR and
    // CompleteOrthogonalDecomposition solves) - on a full-rank system all of them return the same
    // solution, on a rank-deficient one the minimum-norm solution where Eigen's QR solve returns a
    // basic one
    double rcond = std::numeric_limits<double>::epsilon() * std::min(N, M);
    if(option == 0){ lapack_gelsd(A, X, B, rcond); return; }
    // option 1 on a square system is the LU solve as in Eigen, for a full-rank system - the rank
    // (the complete-pivoting LU with Eigen's threshold) decides first, since gesv solves a numerically
    // singular system without a word and returns a 1e16-sized vector - a rank-deficient one takes the
    // minimum-norm least squares
    if(option == 1 && N == M && lapack_getc2_rank(A, rcond) == N && lapack_gesv(A, X, B) == 0){ return; }
    lapack_gelsy(A, X, B, rcond);
    return;
  }
  Eigen::MatrixXd a(N,M);
  Eigen::MatrixXd x(M,K);
  Eigen::MatrixXd b(N,K);

  for(i=0;i<N;i++){  for(j=0;j<M;j++){ a(i,j) = A.M[i*M+j]; }  }
  for(i=0;i<N;i++){  for(j=0;j<K;j++){ b(i,j) = B.M[i*K+j]; }  }

  if(option==0){
  //  x = a.template bdcSvd<Eigen::ComputeThinU | Eigen::ComputeThinV>().solve(b);
  // x = BDCSVD(a).solve(b);
   x = a.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
  }
  else if(option==1){
   // FullPivLU::solve is an exact-system solver: on a rectangular system it returned a vector that
   // is not the least-squares solution (residual 13.0 against 8.5 on a 40 x 12 test) - the complete
   // orthogonal decomposition serves that case, the LU the square one
   // and the LU solve answers a full-rank square system alone - a rank-deficient one (FullPivLU's rank,
   // the threshold eps n) takes the minimum-norm least squares of the complete orthogonal decomposition
   if(N == M){
     Eigen::FullPivLU<Eigen::MatrixXd> lu(a);
     if(lu.rank() == N){ x = lu.solve(b); }
     else{ x = a.completeOrthogonalDecomposition().solve(b); }
   }
   else{ x = a.completeOrthogonalDecomposition().solve(b); }
  }
  else if(option==2){
   x = a.fullPivHouseholderQr().solve(b);
  }
  else if(option==3){
   x = a.completeOrthogonalDecomposition().solve(b);
  }

  for(i=0;i<M;i++){  for(j=0;j<K;j++){ X.M[i*K+j] = x(i,j); }  }

}

bool linsys_solver(const MATRIX& A, MATRIX& X, const MATRIX& B, const double NormThreshold){
/**
  Adapted from the OpenBabel QEq method: solve A x = b for the column vector x - the LU solution
  first, and when it is not finite or its residual |A x - b| exceeds NormThreshold, the minimum-norm
  SVD solution in its place (least_squares_solve options 1 and 0). Returns true with the solution in
  X - a non-finite SVD residual returns false, a residual above NormThreshold is an error (the charge
  equations have no solution to that accuracy)
*/
    if(A.n_rows!=A.n_cols){
        std::cout<<"Error: The matrix A in equation A * x = b must be square\n"; std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");  // N
    }
    if(A.n_cols!=X.n_rows){
        std::cout<<"Error: The number of cols of matrix A and the number of rows in matrix x in equation A * x = b must be equal\n"; std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");  // N
    }
    if( (X.n_cols!=B.n_cols) || (X.n_rows!=B.n_rows)){
        std::cout<<"Error: The dimensions of matrices x and b in equation A * x = b must be equal\n"; std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above"); 
    }
    if(X.n_cols!=1){
        std::cout<<"Error: The matrices x and b in equation A * x = b should be column-vectors (num_of_cols ==1) \n"; std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above"); 
    }
    // least_squares_solve takes non-const references and leaves its inputs untouched
    MATRIX a(A);
    MATRIX b(B);
    least_squares_solve(a, X, b, 1);
    MATRIX resid(B.n_rows, 1);  resid = a * X - b;
    double resnorm = 0.0;
    for(int i = 0; i < resid.n_rows; i++){ resnorm += resid.M[i] * resid.M[i]; }
    resnorm = std::sqrt(resnorm);
    if(!std::isfinite(resnorm) || resnorm > NormThreshold){
        std::cout << "Warning, LU solver failed: the residual has norm " << resnorm
                  << " which exceeds the recommended threshold of " << NormThreshold
                  << ". Proceeding with singular value decomposition.\n";
        least_squares_solve(a, X, b, 0);
        resid = a * X - b;
        resnorm = 0.0;
        for(int i = 0; i < resid.n_rows; i++){ resnorm += resid.M[i] * resid.M[i]; }
        resnorm = std::sqrt(resnorm);
        if(!std::isfinite(resnorm)){
            std::cout << "SVD solver returned a non-finite residual. Solution may be not reliable!\n";
            return false;
        }
        if(resnorm > NormThreshold){
            std::cout << "Error in linsys_solver(): the residual of the SVD solution has norm " << resnorm
                      << ", above the threshold " << NormThreshold << " - the charge equations have no solution to"
                      << " that accuracy\n";
            std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above");
        }
    }
    return true;
}


void solve_linsys(MATRIX& C,MATRIX& D, MATRIX& X,double eps,int maxiter){
/**
  Solve C X = D: the exact LU solution for a square nonsingular C, the minimum-norm least-squares
  solution otherwise (least_squares_solve, option 1) - the solution the Gauss-Seidel iteration on the
  normal equations C^T C X = C^T D (S. Simons, Math. Gazette 80, 567 (1996)) converged to when it
  converged. That iteration squared the condition number and stopped on the size of its step, which
  accepted a wrong X at a loose eps (the QTAG momentum fit: 420 % off for an off-center cloud) and
  never finished at a tight one.
  Inputs: C (n x m), D (n x p)
          eps, maxiter - kept for the signature, without effect on the direct solution
  Output: X (m x p)
*/
    if(C.n_rows!=D.n_rows)
        {std::cout<<"Error: The number of rows of matrices C and D in equation CX = D must be equal\n"; std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above"); } // n
    if(C.n_cols!=X.n_rows)
        {std::cout<<"Error: The number of cols of matrix C and num of rows in matrix D in equation CX = D must be equal\n"; std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above"); } // m
    if(X.n_cols!=D.n_cols)
        {std::cout<<"Error: The number of cols of matrices X and D in equation CX = D must be equal\n"; std::cout << std::flush; throw std::runtime_error("libra C++ error - see the message printed above"); } // p
    (void)eps; (void)maxiter;
    least_squares_solve(C, X, D, 1);
}





}// namespace libmeigen
}// namespace liblibra
