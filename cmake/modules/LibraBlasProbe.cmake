# Runtime probes for the CBLAS/LAPACK backends: a symbol-existence check
# links against an ILP64 build as readily as an LP64 one, and the ILP64
# callee then reads 32-bit dimension arguments as 64-bit garbage. These
# probes RUN a 2x2 zgemm / zheevd through the exact prototypes of
# src/math_linalg/blas_interface.h and compare the results.
INCLUDE(CheckCXXSourceRuns)

# libra_probe_zgemm(<libs> <result-var>): TRUE when cblas_zgemm with the
# int-argument CBLAS ABI multiplies two 2x2 complex matrices correctly
FUNCTION(libra_probe_zgemm libs result)
  SET(CMAKE_REQUIRED_LIBRARIES ${libs})
  CHECK_CXX_SOURCE_RUNS("
#include <complex>
#include <cmath>
extern \"C\" void cblas_zgemm(int, int, int, int, int, int, const void*,
                            const void*, int, const void*, int,
                            const void*, void*, int);
int main() {
  typedef std::complex<double> Z;
  const Z a[4] = {Z(1, 1), Z(2, 0), Z(0, 3), Z(4, -1)};
  const Z b[4] = {Z(0, 1), Z(1, 0), Z(2, 2), Z(-1, 1)};
  Z c[4] = {Z(0, 0), Z(0, 0), Z(0, 0), Z(0, 0)};
  const Z one(1, 0), zero(0, 0);
  // 101 = CblasRowMajor, 111 = CblasNoTrans
  cblas_zgemm(101, 111, 111, 2, 2, 2, &one, a, 2, b, 2, &zero, c, 2);
  Z ref[4];
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++) {
      ref[i * 2 + j] = Z(0, 0);
      for (int k = 0; k < 2; k++) ref[i * 2 + j] += a[i * 2 + k] * b[k * 2 + j];
    }
  for (int i = 0; i < 4; i++)
    if (std::abs(c[i] - ref[i]) > 1e-12) return 1;
  return 0;
}" ${result})
  UNSET(CMAKE_REQUIRED_LIBRARIES)
  SET(${result} ${${result}} PARENT_SCOPE)
ENDFUNCTION()

# libra_probe_zheevd(<libs> <result-var>): TRUE when zheevd_ with the
# int-argument Fortran ABI diagonalizes a 2x2 Hermitian matrix correctly
FUNCTION(libra_probe_zheevd libs result)
  SET(CMAKE_REQUIRED_LIBRARIES ${libs})
  CHECK_CXX_SOURCE_RUNS("
#include <complex>
#include <cmath>
#include <cstddef>
extern \"C\" void zheevd_(const char*, const char*, const int*,
                         std::complex<double>*, const int*, double*,
                         std::complex<double>*, const int*, double*,
                         const int*, int*, const int*, int*,
                         std::size_t, std::size_t);
int main() {
  typedef std::complex<double> Z;
  // column-major Hermitian [[2, 1-i], [1+i, 3]]: eigenvalues (5 -+ sqrt(9))/2 = 1, 4
  Z a[4] = {Z(2, 0), Z(1, 1), Z(1, -1), Z(3, 0)};
  double w[2];
  int n = 2, lda = 2, info = 0;
  Z work[64]; double rwork[64]; int iwork[64];
  int lwork = 64, lrwork = 64, liwork = 64;
  zheevd_(\"V\", \"L\", &n, a, &lda, w, work, &lwork, rwork, &lrwork,
          iwork, &liwork, &info, 1, 1);
  if (info != 0) return 1;
  if (std::abs(w[0] - 1.0) > 1e-12 || std::abs(w[1] - 4.0) > 1e-12) return 2;
  return 0;
}" ${result})
  UNSET(CMAKE_REQUIRED_LIBRARIES)
  SET(${result} ${${result}} PARENT_SCOPE)
ENDFUNCTION()
