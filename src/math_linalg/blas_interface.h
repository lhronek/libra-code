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
  (dsyevd, zheevd, dsygvd, zhegvd) and the LU inverse (dgetrf + dgetri) in
  libra's conventions - row-major MATRIX/CMATRIX in and out, E a diagonal
  matrix of ascending eigenvalues, C with the eigenvectors in its columns
  (C.col(k) belongs to E(k,k)), the generalized kernels S-orthonormal
  (C^H S C = 1). The eigensolvers read one triangle, so they refuse a
  non-Hermitian input (max |A - A^H| > 1e-8 max(|A|, 1)) instead of solving
  its mirror. The inverse has no conditioning guard - inv_matrix of
  libmeigen carries the eigenvalue threshold. Without the backend every
  kernel throws - have_lapack() tells the callers, and the Eigen kernels of
  libmeigen (solve_eigen, FullPivLU_inverse) stay available.
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
void lapack_getri(const MATRIX& A, MATRIX& Ainv);                                ///< inverse of a general real matrix (LU, partial pivoting)
void lapack_getri(const CMATRIX& A, CMATRIX& Ainv);                              ///< inverse of a general complex matrix

}// namespace liblinalg
}// namespace liblibra

#endif
