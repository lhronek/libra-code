/*********************************************************************************
* Copyright (C) 2015 Alexey V. Akimov
*
* This file is distributed under the terms of the GNU General Public License
* as published by the Free Software Foundation, either version 2 of
* the License, or (at your option) any later version.
* See the file LICENSE in the root directory of this distribution
* or <http://www.gnu.org/licenses/>.
*
*********************************************************************************/

#ifndef MMATH_EIGEN_H
#define MMATH_EIGEN_H

#if defined(USING_PCH)
#include "../pch.h"
#else
#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#endif

#include "../math_linalg/liblinalg.h"

/// liblibra namespace
namespace liblibra{

using namespace boost::python;
using namespace liblinalg; 

/// libmeigen namespace
namespace libmeigen{


///=========== Look in: mEigen_determinant.cpp =================
///< Several versions of the determinants
///< LAPACK getrf (the pivot product with the permutation sign) from get_lapack_det_min_n states on - det and FullPivLU_det coincide there - Eigen's determinant / FullPivLU below it
double det(MATRIX&);
double FullPivLU_det(MATRIX&);
complex<double> det(CMATRIX&);
complex<double> FullPivLU_det(CMATRIX& A);


///=========== Look in: mEigen_eigensolve1.cpp ==================
///< Solving the generalized eigenvalue problem: H * C = S * C * E
///< using GeneralizedSelfAdjointEigenSolver<MatrixXd> solution(A,B)- reordering may happen
///< All-MATRIX versions
void solve_eigen(MATRIX* H, MATRIX* S, MATRIX* E, MATRIX* C, int symm);      ///< pointers
void solve_eigen(MATRIX& H, MATRIX& S, MATRIX& E, MATRIX& C, int symm);      ///< references
///< All-CMATRIX versions
void solve_eigen(CMATRIX* H, CMATRIX* S, CMATRIX* E, CMATRIX* C, int symm);  ///< pointers
void solve_eigen(CMATRIX& H, CMATRIX& S, CMATRIX& E, CMATRIX& C, int symm);  ///< references
///< Mixed MATRIX-CMATRIX versions
void solve_eigen(MATRIX* H, MATRIX* S, CMATRIX* E, CMATRIX* C, int symm);    ///< pointers
void solve_eigen(MATRIX& H, MATRIX& S, CMATRIX& E, CMATRIX& C, int symm);    ///< references


///=========== Look in: mEigen_eigensolve2.cpp ==================
///< Solving the eigenvalue problem: H * C = C * E  
///< These functions essentially wrap around the functions that call
///< GeneralizedSelfAdjointEigenSolver<MatrixXd> solution(A,B), so the reordering may happen
///< All-CMATRIX versions
void solve_eigen(CMATRIX* H, CMATRIX* E, CMATRIX* C, int symm);              ///< pointers  
void solve_eigen(CMATRIX& H, CMATRIX& E, CMATRIX& C, int symm);              ///< references
///< Mixed MATRIX-CMATRIX versions
void solve_eigen(MATRIX* H, CMATRIX* E, CMATRIX* C, int symm);               ///< pointers  
void solve_eigen(MATRIX& H, CMATRIX& E, CMATRIX& C, int symm);               ///< references
///< All-MATRIX versions
void solve_eigen(MATRIX* H, MATRIX* E, MATRIX* C, int symm);                 ///< pointers
void solve_eigen(MATRIX& H, MATRIX& E, MATRIX& C, int symm);                 ///< references


///=========== Look in: mEigen_eigensolve3.cpp ==================
///< Solving the eigenvalue problem: H * C = C * E 
///< These functions call the EigenSolver<MatrixXd> solution(A) directly from Eigen3
///< so, not reordering is made
///< All-MATRIX versions
void solve_eigen_nosort(MATRIX* H, MATRIX* E, MATRIX* C, int symm);          ///< pointers
void solve_eigen_nosort(MATRIX& H, MATRIX& E, MATRIX& C, int symm);          ///< references
///< All-CMATRIX versions
void solve_eigen_nosort(CMATRIX* H, CMATRIX* E, CMATRIX* C, int symm);       ///< pointers  
void solve_eigen_nosort(CMATRIX& H, CMATRIX& E, CMATRIX& C, int symm);       ///< references
///< Mixed MATRIX-CMATRIX versions
void solve_eigen_nosort(MATRIX* H, CMATRIX* E, CMATRIX* C, int symm);        ///< pointers  
void solve_eigen_nosort(MATRIX& H, CMATRIX& E, CMATRIX& C, int symm);        ///< references
int get_lapack_eigen_min_n();        ///< solve_eigen takes the LAPACK kernels of blas_interface.h from this matrix size on and the Eigen solver below it (whose eigenvector signs the small-model certifiers pin) - the LIBRA_LAPACK_EIGEN_MIN_N environment variable overrides the built-in default at first use
void set_lapack_eigen_min_n(int n);  ///< 1 = every size through LAPACK, a huge value = Eigen everywhere
int get_lapack_nosort_min_n();       ///< the same switch for solve_eigen_nosort (the general dgeev/zgeev kernels) - its own, much lower default, LIBRA_LAPACK_NOSORT_MIN_N overrides it
void set_lapack_nosort_min_n(int n);
int get_lapack_inverse_min_n();      ///< the switch for FullPivLU_inverse and the threshold-free inv_matrix (the LU kernels dgetri/zgetri) - default 1, LIBRA_LAPACK_INVERSE_MIN_N overrides it
void set_lapack_inverse_min_n(int n);
int get_lapack_det_min_n();          ///< the switch for det and FullPivLU_det (the LU determinant through dgetrf/zgetrf) - default 1 (the kernel wins at every complex size, Eigen's real LU determinant only between ~8 and ~64 states), LIBRA_LAPACK_DET_MIN_N overrides it
void set_lapack_det_min_n(int n);
int get_lapack_fullpiv_min_n();      ///< the switch for FullPivLU_rank_invertible and FullPivLU_decomposition (the complete-pivoting LU dgetc2/zgetc2) - default 1 (LAPACK_FULLPIV_MIN_N_DEFAULT), LIBRA_LAPACK_FULLPIV_MIN_N overrides it
void set_lapack_fullpiv_min_n(int n);
int get_lapack_svd_min_n();          ///< the switch for JacobiSVD_decomposition and BDCSVD_decomposition (the divide-and-conquer SVD zgesdd, both coincide there) - default LAPACK_SVD_MIN_N_DEFAULT, LIBRA_LAPACK_SVD_MIN_N overrides it
void set_lapack_svd_min_n(int n);
int get_lapack_llt_min_n();          ///< the switch for LLT_decomposition (the Cholesky factor dpotrf) - default LAPACK_LLT_MIN_N_DEFAULT, LIBRA_LAPACK_LLT_MIN_N overrides it
void set_lapack_llt_min_n(int n);
int get_lapack_lstsq_min_n();        ///< the switch for least_squares_solve (dgelsd for option 0, dgesv for option 1 on a square system, dgelsy otherwise) by the larger dimension of A - default LAPACK_LSTSQ_MIN_N_DEFAULT, LIBRA_LAPACK_LSTSQ_MIN_N overrides it
void set_lapack_lstsq_min_n(int n);




///=========== Look in: mEigen_inverse.cpp ==================
///< Invertability check
///< rank = pivots of the complete-pivoting LU above eps n times the largest one (Eigen's rule), invertible = full rank: LAPACK getc2 from get_lapack_fullpiv_min_n states on, Eigen's FullPivLU below
void FullPivLU_rank_invertible(MATRIX& A, int& rank, int& is_inver);
void FullPivLU_rank_invertible(CMATRIX& A, int& rank, int& is_inver);
boost::python::list FullPivLU_rank_invertible(MATRIX& A);
boost::python::list FullPivLU_rank_invertible(CMATRIX& A);

///< Matrix inversion via an LU decomposition
///< LU inverse: LAPACK dgetri/zgetri when linked (from get_lapack_inverse_min_n states on, an exactly singular matrix throws), Eigen's FullPivLU otherwise
void FullPivLU_inverse(MATRIX& A, MATRIX& invA);
void FullPivLU_inverse(CMATRIX& A, CMATRIX& invA);

///< Matrix inversion via solving the eigenvalue problem
///< S^-1: thresh <= 0 takes the LU kernel on the matrix as given (a non-Hermitian S included), thresh > 0 the eigendecomposition route with the |eigenvalue| guard (Hermitian S)
void inv_matrix(MATRIX& S, MATRIX& S_inv, double thresh, int do_phase_correction);
void inv_matrix(MATRIX& S, MATRIX& S_inv, double thresh);
void inv_matrix(MATRIX& S, MATRIX& S_inv);

void inv_matrix(CMATRIX& S, CMATRIX& S_inv, double thresh, int do_phase_correction);
void inv_matrix(CMATRIX& S, CMATRIX& S_inv, double thresh);
void inv_matrix(CMATRIX& S, CMATRIX& S_inv);


///=========== Look in: mEigen_matrix_functions.cpp ==================
///< Square root and inverse of a matrix
void sqrt_matrix(CMATRIX& S, CMATRIX& S_half, CMATRIX& S_i_half, double thresh, int do_phase_correction);
void sqrt_matrix(CMATRIX& S, CMATRIX& S_half, CMATRIX& S_i_half, double thresh);
void sqrt_matrix(CMATRIX& S, CMATRIX& S_half, CMATRIX& S_i_half);
void sqrt_matrix(MATRIX& S, MATRIX& S_half, MATRIX& S_i_half, double thresh, int do_phase_correction);  ///< real symmetric positive-definite S: LAPACK dsyevd when the build linked it, the Eigen solver otherwise
void sqrt_matrix(MATRIX& S, MATRIX& S_half, MATRIX& S_i_half, double thresh);
void sqrt_matrix(MATRIX& S, MATRIX& S_half, MATRIX& S_i_half);
int lowdin_frame(CMATRIX& S, CMATRIX& X, CMATRIX& Y);  ///< Lowdin frame of a metric: X = S^-1/2, Y = S^1/2. Returns 0 (X, Y untouched) for an identity S (max |S - I| <= 1e-12)
void sqrt_matrix_derivative(CMATRIX& S, CMATRIX& dS, CMATRIX& dY, CMATRIX& dX);  ///< d(S^1/2) and d(S^-1/2) along a Hermitian variation dS of a positive-definite S (Daleckii-Krein divided differences of one eigendecomposition)
void exp_matrix(CMATRIX& res, CMATRIX& S, complex<double> dt, int do_phase_correction);
void exp_matrix(CMATRIX& res, CMATRIX& S, complex<double> dt);


///=========== Look in: mEigen_decompositions.cpp ==================
///< LU decomposition
///< P A Q = L U (L unit lower, U upper, P and Q permutation matrices): LAPACK getc2 from get_lapack_fullpiv_min_n states on, Eigen's FullPivLU below
void FullPivLU_decomposition(MATRIX& A, MATRIX& P, MATRIX& L, MATRIX& U, MATRIX& Q);
void FullPivLU_decomposition(CMATRIX& A, CMATRIX& P, CMATRIX& L, CMATRIX& U, CMATRIX& Q);

///< A = U S V^H (S the diagonal of descending singular values, V not V^H): LAPACK zgesdd from get_lapack_svd_min_n states on (both functions coincide there), Eigen's JacobiSVD / BDCSVD below
void JacobiSVD_decomposition(CMATRIX& A, CMATRIX& U, CMATRIX& S, CMATRIX& V);
void BDCSVD_decomposition(CMATRIX& A, CMATRIX& U, CMATRIX& S, CMATRIX& V);

///< A = L L^T, L lower: LAPACK dpotrf from get_lapack_llt_min_n states on (a non-positive-definite or non-symmetric A throws there), Eigen's LLT below
void LLT_decomposition(MATRIX& A, MATRIX& L);

///=========== Look in: mEigen_linsolve.cpp ==================
///< Solver for a system of linear equations (iterative schemes)
///< min |A X - B|: options 0 SVD, 1 LU (a square system, the complete orthogonal decomposition for a rectangular one), 2 column-pivoted QR, 3 complete orthogonal decomposition - LAPACK dgelsd / dgesv / dgelsy with Eigen's rank threshold eps min(N, M) from get_lapack_lstsq_min_n on, Eigen's solvers below
void least_squares_solve(MATRIX& A, MATRIX& X, MATRIX& B, int option);
bool linsys_solver(const MATRIX& A, MATRIX& X, const MATRIX& B, const double NormThreshold);
void solve_linsys(MATRIX&,MATRIX&, MATRIX&,double,int);

}// namespace libmeigen
}// namespace liblibra


#endif
