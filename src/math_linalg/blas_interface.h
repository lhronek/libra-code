#ifndef BLAS_INTERFACE_H
#define BLAS_INTERFACE_H
/**
  Declarations of the external BLAS/LAPACK entry points used by liblibra.

  "BLAS" here names the linked library (OpenBLAS, MKL, ...), which bundles
  both the BLAS and the LAPACK routine families - this header declares the
  entry points liblibra uses from either. It is self-contained on purpose:
  no dependency on cblas.h/lapacke.h, whose availability and layout differ
  across distributions. The symbols resolve from whatever library the build
  probes linked (WITH_BLAS_GEMM, WITH_LAPACK_ZHEEVD); translation units
  reference them only inside the corresponding HAVE_* guards.
*/
#include <complex>
#include <cstddef>

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

}// extern "C"

namespace liblibra{
namespace liblinalg{

// CBLAS layout/transpose constants (values fixed by the CBLAS standard)
const int CBLAS_ROW_MAJOR  = 101;
const int CBLAS_NO_TRANS   = 111;
const int CBLAS_TRANS      = 112;
const int CBLAS_CONJ_TRANS = 113;

}// namespace liblinalg
}// namespace liblibra

#endif
