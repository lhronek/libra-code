#ifndef BLAS_INTERFACE_H
#define BLAS_INTERFACE_H
/**
  The BLAS/LAPACK interface of liblibra: the external entry points and the
  LAPACK kernels on libra matrices built on them (blas_interface.cpp).

  "BLAS" here names the linked library (OpenBLAS, MKL, ...), which bundles
  both the BLAS and the LAPACK routine families - the extern "C" block
  declares the entry points liblibra uses from either, without cblas.h or
  lapacke.h, whose availability and layout differ across distributions. The
  symbols resolve from whatever library the build probes linked
  (WITH_BLAS_GEMM, WITH_LAPACK_ZHEEVD, WITH_LAPACK); translation units
  reference them only inside the corresponding HAVE_* guards.

  The kernels (liblinalg namespace): the divide-and-conquer eigensolvers
  (dsyevd, zheevd, dsygvd, zhegvd), the general eigensolvers (dgeev, zgeev),
  the LU inverse (dgetrf + dgetri), the LU determinant (dgetrf: the
  product of the pivots with the permutation sign, exactly 0 for an exactly
  singular matrix), the complete-pivoting LU (dgetc2, zgetc2: the rank
  with a relative pivot threshold, and P A Q = L U with explicit permutation
  matrices), the divide-and-conquer SVD (dgesdd, zgesdd: A = U S V^H with
  full U and V, S the rectangular diagonal of descending singular values)
  the Cholesky factor (dpotrf, zpotrf: A = L L^H of a Hermitian
  positive-definite A, a non-definite A throws), the minimum-norm least
  squares solvers (dgelsd through the SVD, dgelsy through the complete
  orthogonal factorization, both with a relative rank threshold) and the
  square LU solve (dgesv, partial pivoting, an exactly singular system
  reported rather than thrown) in libra's conventions - row-major MATRIX/CMATRIX in and out, E a diagonal
  matrix of ascending eigenvalues, C with the eigenvectors in its columns
  (C.col(k) belongs to E(k,k)), the generalized kernels S-orthonormal
  (C^H S C = 1). C may be n x m with 1 <= m <= n and E m x m: the lowest m
  eigenpairs are returned (the partial output solve_eigen admits). The
  eigensolvers read one triangle, so they refuse a non-Hermitian input
  (max |A - A^H| > 1e-8 max(|A|, 1)) instead of solving its mirror. The
  inverse has no conditioning guard - inv_matrix of libmeigen carries the
  eigenvalue threshold. The general kernels return the eigenvalues in the
  order of the Schur form (no sorting, complex conjugate pairs of a real
  matrix adjacent with the positive imaginary part first) and unit-norm right
  eigenvectors with the largest component real. Workspaces are thread-local
  and grow on demand (the kernels sit in the per-trajectory, per-step path of
  the dynamics).
  Without the backend every kernel throws - have_lapack() tells the callers,
  and the Eigen kernels of libmeigen stay available as the fallback.
*/
#include <complex>
#include <cstddef>
#include "MATRIX.h"
#include "CMATRIX.h"

extern "C" {

// CBLAS
void cblas_zgemm(int layout, int transa, int transb, int m, int n, int k,
                 const void* alpha, const void* a, int lda, const void* b,
                 int ldb, const void* beta, void* c, int ldc);
void cblas_dgemm(int layout, int transa, int transb, int m, int n, int k,
                 double alpha, const double* a, int lda, const double* b,
                 int ldb, double beta, double* c, int ldc);

// LAPACK (Fortran ABI; the trailing size_t arguments are the hidden
// character-length slots of the modern Fortran calling convention)
void zheevd_(const char* jobz, const char* uplo, const int* n,
             std::complex<double>* a, const int* lda, double* w,
             std::complex<double>* work, const int* lwork, double* rwork,
             const int* lrwork, int* iwork, const int* liwork, int* info,
             std::size_t jobz_len, std::size_t uplo_len);
void dsyevd_(const char* jobz, const char* uplo, const int* n, double* a,
             const int* lda, double* w, double* work, const int* lwork,
             int* iwork, const int* liwork, int* info,
             std::size_t jobz_len, std::size_t uplo_len);
void dsygvd_(const int* itype, const char* jobz, const char* uplo, const int* n,
             double* a, const int* lda, double* b, const int* ldb, double* w,
             double* work, const int* lwork, int* iwork, const int* liwork,
             int* info, std::size_t jobz_len, std::size_t uplo_len);
void zhegvd_(const int* itype, const char* jobz, const char* uplo, const int* n,
             std::complex<double>* a, const int* lda, std::complex<double>* b,
             const int* ldb, double* w, std::complex<double>* work,
             const int* lwork, double* rwork, const int* lrwork, int* iwork,
             const int* liwork, int* info, std::size_t jobz_len,
             std::size_t uplo_len);
void dgetrf_(const int* m, const int* n, double* a, const int* lda, int* ipiv,
             int* info);
void dgetri_(const int* n, double* a, const int* lda, const int* ipiv,
             double* work, const int* lwork, int* info);
void zgetrf_(const int* m, const int* n, std::complex<double>* a,
             const int* lda, int* ipiv, int* info);
void zgetri_(const int* n, std::complex<double>* a, const int* lda,
             const int* ipiv, std::complex<double>* work, const int* lwork,
             int* info);
void dgetc2_(const int* n, double* a, const int* lda, int* ipiv, int* jpiv,
             int* info);
void zgetc2_(const int* n, std::complex<double>* a, const int* lda, int* ipiv,
             int* jpiv, int* info);
void dgesdd_(const char* jobz, const int* m, const int* n, double* a,
             const int* lda, double* s, double* u, const int* ldu, double* vt,
             const int* ldvt, double* work, const int* lwork, int* iwork,
             int* info, std::size_t jobz_len);
void zgesdd_(const char* jobz, const int* m, const int* n,
             std::complex<double>* a, const int* lda, double* s,
             std::complex<double>* u, const int* ldu, std::complex<double>* vt,
             const int* ldvt, std::complex<double>* work, const int* lwork,
             double* rwork, int* iwork, int* info, std::size_t jobz_len);
void dpotrf_(const char* uplo, const int* n, double* a, const int* lda,
             int* info, std::size_t uplo_len);
void zpotrf_(const char* uplo, const int* n, std::complex<double>* a,
             const int* lda, int* info, std::size_t uplo_len);
void dgelsd_(const int* m, const int* n, const int* nrhs, double* a,
             const int* lda, double* b, const int* ldb, double* s,
             const double* rcond, int* rank, double* work, const int* lwork,
             int* iwork, int* info);
void dgelsy_(const int* m, const int* n, const int* nrhs, double* a,
             const int* lda, double* b, const int* ldb, int* jpvt,
             const double* rcond, int* rank, double* work, const int* lwork,
             int* info);
void dgesv_(const int* n, const int* nrhs, double* a, const int* lda,
            int* ipiv, double* b, const int* ldb, int* info);
void dgeev_(const char* jobvl, const char* jobvr, const int* n, double* a,
            const int* lda, double* wr, double* wi, double* vl,
            const int* ldvl, double* vr, const int* ldvr, double* work,
            const int* lwork, int* info, std::size_t jobvl_len,
            std::size_t jobvr_len);
void zgeev_(const char* jobvl, const char* jobvr, const int* n,
            std::complex<double>* a, const int* lda, std::complex<double>* w,
            std::complex<double>* vl, const int* ldvl,
            std::complex<double>* vr, const int* ldvr,
            std::complex<double>* work, const int* lwork, double* rwork,
            int* info, std::size_t jobvl_len, std::size_t jobvr_len);

}// extern "C"

namespace liblibra{
namespace liblinalg{

// CBLAS layout/transpose constants (values fixed by the CBLAS standard)
const int CBLAS_ROW_MAJOR  = 101;
const int CBLAS_NO_TRANS   = 111;
const int CBLAS_TRANS      = 112;
const int CBLAS_CONJ_TRANS = 113;

// LAPACK kernels on libra matrices (blas_interface.cpp)
bool have_lapack();  ///< the LAPACK backend was probed and linked into this build

void lapack_syevd(const MATRIX& A, MATRIX& E, MATRIX& C);                        ///< A C = C E, A real symmetric
void lapack_heevd(const CMATRIX& A, CMATRIX& E, CMATRIX& C);                     ///< A C = C E, A Hermitian
void lapack_sygvd(const MATRIX& H, const MATRIX& S, MATRIX& E, MATRIX& C);       ///< H C = S C E, C^T S C = 1, S positive definite
void lapack_hegvd(const CMATRIX& H, const CMATRIX& S, CMATRIX& E, CMATRIX& C);   ///< H C = S C E, C^H S C = 1, S positive definite
void lapack_geev(const MATRIX& A, CMATRIX& E, CMATRIX& C);                       ///< A C = C E for a general real matrix - complex eigenvalues and eigenvectors
void lapack_geev(const CMATRIX& A, CMATRIX& E, CMATRIX& C);                      ///< A C = C E for a general complex matrix
void lapack_getri(const MATRIX& A, MATRIX& Ainv);                                ///< inverse of a general real matrix (LU, partial pivoting)
double lapack_det(const MATRIX& A);                                              ///< determinant of a general real matrix (LU)
int lapack_getc2_rank(const MATRIX& A, double rel_threshold);                    ///< rank = the complete-pivoting LU pivots above rel_threshold times the largest one
int lapack_getc2_rank(const CMATRIX& A, double rel_threshold);
void lapack_getc2(const MATRIX& A, MATRIX& P, MATRIX& L, MATRIX& U, MATRIX& Q);   ///< complete-pivoting LU: P A Q = L U, L unit lower, U upper, P and Q permutation matrices
void lapack_getc2(const CMATRIX& A, CMATRIX& P, CMATRIX& L, CMATRIX& U, CMATRIX& Q);
void lapack_gesdd(const MATRIX& A, MATRIX& U, MATRIX& S, MATRIX& V);              ///< A (m x n) = U (m x m) S (m x n) V^T (n x n), singular values descending
void lapack_gesdd(const CMATRIX& A, CMATRIX& U, CMATRIX& S, CMATRIX& V);          ///< A = U S V^H, the column phases of U and V paired
void lapack_potrf(const MATRIX& A, MATRIX& L);                                   ///< A = L L^T of a symmetric positive-definite A, L lower
void lapack_potrf(const CMATRIX& A, CMATRIX& L);                                 ///< A = L L^H of a Hermitian positive-definite A, L lower
int lapack_gelsd(const MATRIX& A, MATRIX& X, const MATRIX& B, double rcond);      ///< min |A X - B| with minimum-norm X (m x n, n x k, m x k) through the SVD, singular values below rcond times the largest dropped, returns the rank
int lapack_gelsy(const MATRIX& A, MATRIX& X, const MATRIX& B, double rcond);      ///< the same through the complete orthogonal factorization (column-pivoted QR), returns the rank
int lapack_gesv(const MATRIX& A, MATRIX& X, const MATRIX& B);                     ///< A X = B for a square A by LU with partial pivoting - returns 0 with X filled, or the index of the exactly zero pivot with X untouched
std::complex<double> lapack_det(const CMATRIX& A);                               ///< determinant of a general complex matrix (LU)
void lapack_getri(const CMATRIX& A, CMATRIX& Ainv);                              ///< inverse of a general complex matrix

}// namespace liblinalg
}// namespace liblibra

#endif
