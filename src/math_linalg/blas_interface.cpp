/**
  The LAPACK kernels of blas_interface.h on libra matrices.

  Storage: libra matrices are row-major and LAPACK reads column-major, so a
  buffer handed over as-is is the TRANSPOSE. The symmetric kernels see the
  matrix itself, the Hermitian ones its conjugate (same eigenvalues, conjugated
  eigenvectors), and the LU inverse returns inv(A^T) = inv(A)^T, which read
  back row-major IS inv(A) - the only copies are the O(n^2) eigenvector
  transposes into libra's column convention.
*/
#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include "blas_interface.h"

/// liblibra namespace
namespace liblibra{

/// liblinalg namespace
namespace liblinalg{


bool have_lapack(){
#ifdef HAVE_LAPACK
  return true;
#else
  return false;
#endif
}


namespace {

void fail(const char* fn, const std::string& what){
  std::cout<<"Error in liblinalg::"<<fn<<" : "<<what<<"\n"; std::cout << std::flush;
  throw std::runtime_error("libra C++ error - see the message printed above");
}

#ifdef HAVE_LAPACK

template <typename M>
void require_n_by_n(const char* fn, const M& A, const char* name, int n){
  if(A.n_rows != n || A.n_cols != n){
    std::ostringstream o;
    o<<name<<" must be "<<n<<" x "<<n<<", got "<<A.n_rows<<" x "<<A.n_cols;
    fail(fn, o.str());
  }
}

// the eigensolvers read one triangle - a non-Hermitian input would be solved as
// its mirror silently (the guard of libmeigen's require_hermitian_pair)
template <typename T>
void require_hermitian(const char* fn, const T* M, int n, const char* name){
  double amax = 0.0, dev = 0.0;
  for(int i = 0; i < n; i++){
    for(int j = 0; j < n; j++){
      amax = std::max(amax, std::abs(M[i*n+j]));
      dev = std::max(dev, std::abs(M[i*n+j] - std::conj(M[j*n+i])));
    }
  }
  if(dev > 1e-8 * std::max(amax, 1.0)){
    std::ostringstream o;
    o<<name<<" is not Hermitian (max |A - A^H| = "<<dev<<") - the kernel reads one triangle, symmetrize the input first";
    fail(fn, o.str());
  }
}

std::string info_text(const char* routine, int info){
  std::ostringstream o;
  o<<routine<<" returned info = "<<info;
  return o.str();
}

// eigenvector k is column k of the column-major LAPACK buffer - libra keeps it in column k of C
void columns_to_C(const double* a, int n, MATRIX& C){
  for(int k = 0; k < n; k++){
    for(int i = 0; i < n; i++){ C.M[i*n+k] = a[(std::size_t)k*n + i]; }
  }
}

// the complex buffers solved conj(A): the eigenvectors of A are their conjugates
void columns_to_C(const std::complex<double>* a, int n, CMATRIX& C){
  for(int k = 0; k < n; k++){
    for(int i = 0; i < n; i++){ C.M[i*n+k] = std::conj(a[(std::size_t)k*n + i]); }
  }
}

#else

const char* MISSING = "libra was built without the LAPACK backend (WITH_LAPACK probe) - use libmeigen's solve_eigen / FullPivLU_inverse";

#endif

}// anonymous namespace


void lapack_syevd(const MATRIX& A, MATRIX& E, MATRIX& C){
  const char* fn = "lapack_syevd";
#ifdef HAVE_LAPACK
  int n = A.n_rows;
  if(n < 1){ fail(fn, "A is empty"); }
  require_n_by_n(fn, A, "A", n);
  require_n_by_n(fn, E, "E", n);
  require_n_by_n(fn, C, "C", n);
  require_hermitian(fn, A.M, n, "A");
  std::vector<double> a(A.M, A.M + (std::size_t)n*n);
  std::vector<double> w(n);
  int n_ = n, lda = n, info = 0, lwork = -1, liwork = -1;
  double wq = 0.0;
  int iwq = 0;
  dsyevd_("V", "L", &n_, a.data(), &lda, w.data(), &wq, &lwork, &iwq, &liwork, &info, 1, 1);
  if(info != 0){ fail(fn, info_text("dsyevd (workspace query)", info)); }
  lwork = (int)wq;
  liwork = iwq;
  std::vector<double> work(std::max(lwork, 1));
  std::vector<int> iwork(std::max(liwork, 1));
  dsyevd_("V", "L", &n_, a.data(), &lda, w.data(), work.data(), &lwork, iwork.data(), &liwork, &info, 1, 1);
  if(info != 0){ fail(fn, info_text("dsyevd", info) + " (the algorithm failed to converge)"); }
  E = 0.0;
  for(int k = 0; k < n; k++){ E.M[k*n+k] = w[k]; }
  columns_to_C(a.data(), n, C);
#else
  (void)A; (void)E; (void)C;
  fail(fn, MISSING);
#endif
}


void lapack_heevd(const CMATRIX& A, CMATRIX& E, CMATRIX& C){
  const char* fn = "lapack_heevd";
#ifdef HAVE_LAPACK
  int n = A.n_rows;
  if(n < 1){ fail(fn, "A is empty"); }
  require_n_by_n(fn, A, "A", n);
  require_n_by_n(fn, E, "E", n);
  require_n_by_n(fn, C, "C", n);
  require_hermitian(fn, A.M, n, "A");
  std::vector< std::complex<double> > a(A.M, A.M + (std::size_t)n*n);
  std::vector<double> w(n);
  int n_ = n, lda = n, info = 0, lwork = -1, lrwork = -1, liwork = -1;
  std::complex<double> wq(0.0, 0.0);
  double rwq = 0.0;
  int iwq = 0;
  zheevd_("V", "L", &n_, a.data(), &lda, w.data(), &wq, &lwork, &rwq, &lrwork, &iwq, &liwork, &info, 1, 1);
  if(info != 0){ fail(fn, info_text("zheevd (workspace query)", info)); }
  lwork = (int)wq.real();
  lrwork = (int)rwq;
  liwork = iwq;
  std::vector< std::complex<double> > work(std::max(lwork, 1));
  std::vector<double> rwork(std::max(lrwork, 1));
  std::vector<int> iwork(std::max(liwork, 1));
  zheevd_("V", "L", &n_, a.data(), &lda, w.data(), work.data(), &lwork, rwork.data(), &lrwork, iwork.data(), &liwork, &info, 1, 1);
  if(info != 0){ fail(fn, info_text("zheevd", info) + " (the algorithm failed to converge)"); }
  E = std::complex<double>(0.0, 0.0);
  for(int k = 0; k < n; k++){ E.M[k*n+k] = std::complex<double>(w[k], 0.0); }
  columns_to_C(a.data(), n, C);
#else
  (void)A; (void)E; (void)C;
  fail(fn, MISSING);
#endif
}


void lapack_sygvd(const MATRIX& H, const MATRIX& S, MATRIX& E, MATRIX& C){
  const char* fn = "lapack_sygvd";
#ifdef HAVE_LAPACK
  int n = H.n_rows;
  if(n < 1){ fail(fn, "H is empty"); }
  require_n_by_n(fn, H, "H", n);
  require_n_by_n(fn, S, "S", n);
  require_n_by_n(fn, E, "E", n);
  require_n_by_n(fn, C, "C", n);
  require_hermitian(fn, H.M, n, "H");
  require_hermitian(fn, S.M, n, "S");
  std::vector<double> a(H.M, H.M + (std::size_t)n*n);
  std::vector<double> b(S.M, S.M + (std::size_t)n*n);
  std::vector<double> w(n);
  int itype = 1, n_ = n, lda = n, ldb = n, info = 0, lwork = -1, liwork = -1;
  double wq = 0.0;
  int iwq = 0;
  dsygvd_(&itype, "V", "L", &n_, a.data(), &lda, b.data(), &ldb, w.data(), &wq, &lwork, &iwq, &liwork, &info, 1, 1);
  if(info != 0){ fail(fn, info_text("dsygvd (workspace query)", info)); }
  lwork = (int)wq;
  liwork = iwq;
  std::vector<double> work(std::max(lwork, 1));
  std::vector<int> iwork(std::max(liwork, 1));
  dsygvd_(&itype, "V", "L", &n_, a.data(), &lda, b.data(), &ldb, w.data(), work.data(), &lwork, iwork.data(), &liwork, &info, 1, 1);
  if(info > n){
    std::ostringstream o;
    o<<"S is not positive definite (its leading minor of order "<<(info - n)<<" is not)";
    fail(fn, o.str());
  }
  if(info != 0){ fail(fn, info_text("dsygvd", info) + " (the algorithm failed to converge)"); }
  E = 0.0;
  for(int k = 0; k < n; k++){ E.M[k*n+k] = w[k]; }
  columns_to_C(a.data(), n, C);
#else
  (void)H; (void)S; (void)E; (void)C;
  fail(fn, MISSING);
#endif
}


void lapack_hegvd(const CMATRIX& H, const CMATRIX& S, CMATRIX& E, CMATRIX& C){
  const char* fn = "lapack_hegvd";
#ifdef HAVE_LAPACK
  int n = H.n_rows;
  if(n < 1){ fail(fn, "H is empty"); }
  require_n_by_n(fn, H, "H", n);
  require_n_by_n(fn, S, "S", n);
  require_n_by_n(fn, E, "E", n);
  require_n_by_n(fn, C, "C", n);
  require_hermitian(fn, H.M, n, "H");
  require_hermitian(fn, S.M, n, "S");
  std::vector< std::complex<double> > a(H.M, H.M + (std::size_t)n*n);
  std::vector< std::complex<double> > b(S.M, S.M + (std::size_t)n*n);
  std::vector<double> w(n);
  int itype = 1, n_ = n, lda = n, ldb = n, info = 0, lwork = -1, lrwork = -1, liwork = -1;
  std::complex<double> wq(0.0, 0.0);
  double rwq = 0.0;
  int iwq = 0;
  zhegvd_(&itype, "V", "L", &n_, a.data(), &lda, b.data(), &ldb, w.data(), &wq, &lwork, &rwq, &lrwork, &iwq, &liwork, &info, 1, 1);
  if(info != 0){ fail(fn, info_text("zhegvd (workspace query)", info)); }
  lwork = (int)wq.real();
  lrwork = (int)rwq;
  liwork = iwq;
  std::vector< std::complex<double> > work(std::max(lwork, 1));
  std::vector<double> rwork(std::max(lrwork, 1));
  std::vector<int> iwork(std::max(liwork, 1));
  zhegvd_(&itype, "V", "L", &n_, a.data(), &lda, b.data(), &ldb, w.data(), work.data(), &lwork, rwork.data(), &lrwork, iwork.data(), &liwork, &info, 1, 1);
  if(info > n){
    std::ostringstream o;
    o<<"S is not positive definite (its leading minor of order "<<(info - n)<<" is not)";
    fail(fn, o.str());
  }
  if(info != 0){ fail(fn, info_text("zhegvd", info) + " (the algorithm failed to converge)"); }
  E = std::complex<double>(0.0, 0.0);
  for(int k = 0; k < n; k++){ E.M[k*n+k] = std::complex<double>(w[k], 0.0); }
  columns_to_C(a.data(), n, C);
#else
  (void)H; (void)S; (void)E; (void)C;
  fail(fn, MISSING);
#endif
}


void lapack_getri(const MATRIX& A, MATRIX& Ainv){
  const char* fn = "lapack_getri";
#ifdef HAVE_LAPACK
  int n = A.n_rows;
  if(n < 1){ fail(fn, "A is empty"); }
  require_n_by_n(fn, A, "A", n);
  require_n_by_n(fn, Ainv, "Ainv", n);
  std::vector<double> a(A.M, A.M + (std::size_t)n*n);
  std::vector<int> ipiv(n);
  int n_ = n, lda = n, info = 0, lwork = -1;
  dgetrf_(&n_, &n_, a.data(), &lda, ipiv.data(), &info);
  if(info > 0){
    std::ostringstream o;
    o<<"A is singular (U("<<info<<","<<info<<") of its LU factorization is exactly zero)";
    fail(fn, o.str());
  }
  if(info != 0){ fail(fn, info_text("dgetrf", info)); }
  double wq = 0.0;
  dgetri_(&n_, a.data(), &lda, ipiv.data(), &wq, &lwork, &info);
  if(info != 0){ fail(fn, info_text("dgetri (workspace query)", info)); }
  lwork = (int)wq;
  std::vector<double> work(std::max(lwork, 1));
  dgetri_(&n_, a.data(), &lda, ipiv.data(), work.data(), &lwork, &info);
  if(info != 0){ fail(fn, info_text("dgetri", info)); }
  std::copy(a.begin(), a.end(), Ainv.M);
#else
  (void)A; (void)Ainv;
  fail(fn, MISSING);
#endif
}


void lapack_getri(const CMATRIX& A, CMATRIX& Ainv){
  const char* fn = "lapack_getri";
#ifdef HAVE_LAPACK
  int n = A.n_rows;
  if(n < 1){ fail(fn, "A is empty"); }
  require_n_by_n(fn, A, "A", n);
  require_n_by_n(fn, Ainv, "Ainv", n);
  std::vector< std::complex<double> > a(A.M, A.M + (std::size_t)n*n);
  std::vector<int> ipiv(n);
  int n_ = n, lda = n, info = 0, lwork = -1;
  zgetrf_(&n_, &n_, a.data(), &lda, ipiv.data(), &info);
  if(info > 0){
    std::ostringstream o;
    o<<"A is singular (U("<<info<<","<<info<<") of its LU factorization is exactly zero)";
    fail(fn, o.str());
  }
  if(info != 0){ fail(fn, info_text("zgetrf", info)); }
  std::complex<double> wq(0.0, 0.0);
  zgetri_(&n_, a.data(), &lda, ipiv.data(), &wq, &lwork, &info);
  if(info != 0){ fail(fn, info_text("zgetri (workspace query)", info)); }
  lwork = (int)wq.real();
  std::vector< std::complex<double> > work(std::max(lwork, 1));
  zgetri_(&n_, a.data(), &lda, ipiv.data(), work.data(), &lwork, &info);
  if(info != 0){ fail(fn, info_text("zgetri", info)); }
  std::copy(a.begin(), a.end(), Ainv.M);
#else
  (void)A; (void)Ainv;
  fail(fn, MISSING);
#endif
}


}// namespace liblinalg
}// namespace liblibra
