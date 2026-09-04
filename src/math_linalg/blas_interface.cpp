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

// the outputs of an eigensolver: C (n x m) receives the lowest m eigenvectors in its columns,
// E (m x m) their eigenvalues on the diagonal, 1 <= m <= n
template <typename M>
int require_outputs(const char* fn, const M& E, const M& C, int n){
  int m = C.n_cols;
  if(C.n_rows != n || m < 1 || m > n){
    std::ostringstream o;
    o<<"C must be "<<n<<" x m with 1 <= m <= "<<n<<", got "<<C.n_rows<<" x "<<C.n_cols;
    fail(fn, o.str());
  }
  if(E.n_rows != m || E.n_cols != m){
    std::ostringstream o;
    o<<"E must be "<<m<<" x "<<m<<" (the column count of C), got "<<E.n_rows<<" x "<<E.n_cols;
    fail(fn, o.str());
  }
  return m;
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

void not_positive_definite(const char* fn, int info, int n){
  std::ostringstream o;
  o<<"S is not positive definite (its leading minor of order "<<(info - n)<<" is not)";
  fail(fn, o.str());
}

// thread-local workspaces of one kernel family, grown on demand
struct RealWork{
  std::vector<double> a, b, w, work;
  std::vector<int> iwork;
};
struct ComplexWork{
  std::vector< std::complex<double> > a, b, work;
  std::vector<double> w, rwork;
  std::vector<int> iwork;
};

template <typename V>
void grow(V& v, std::size_t n){ if(v.size() < n){ v.resize(n); } }

// eigenvector k is column k of the column-major LAPACK buffer - libra keeps it in column k of C
void columns_to_C(const double* a, int n, int m, double* w, MATRIX& E, MATRIX& C){
  E = 0.0;
  for(int k = 0; k < m; k++){
    E.M[k*m+k] = w[k];
    for(int i = 0; i < n; i++){ C.M[i*m+k] = a[(std::size_t)k*n + i]; }
  }
}

// the complex buffers solved conj(A): the eigenvectors of A are their conjugates
void columns_to_C(const std::complex<double>* a, int n, int m, double* w, CMATRIX& E, CMATRIX& C){
  E = std::complex<double>(0.0, 0.0);
  for(int k = 0; k < m; k++){
    E.M[k*m+k] = std::complex<double>(w[k], 0.0);
    for(int i = 0; i < n; i++){ C.M[i*m+k] = std::conj(a[(std::size_t)k*n + i]); }
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
  int m = require_outputs(fn, E, C, n);
  require_hermitian(fn, A.M, n, "A");
  thread_local RealWork ws;
  grow(ws.a, (std::size_t)n*n);
  grow(ws.w, n);
  std::copy(A.M, A.M + (std::size_t)n*n, ws.a.begin());
  int n_ = n, lda = n, info = 0, lwork = -1, liwork = -1;
  double wq = 0.0;
  int iwq = 0;
  dsyevd_("V", "L", &n_, ws.a.data(), &lda, ws.w.data(), &wq, &lwork, &iwq, &liwork, &info, 1, 1);
  if(info != 0){ fail(fn, info_text("dsyevd (workspace query)", info)); }
  lwork = (int)wq;
  liwork = iwq;
  grow(ws.work, std::max(lwork, 1));
  grow(ws.iwork, std::max(liwork, 1));
  dsyevd_("V", "L", &n_, ws.a.data(), &lda, ws.w.data(), ws.work.data(), &lwork, ws.iwork.data(), &liwork, &info, 1, 1);
  if(info != 0){ fail(fn, info_text("dsyevd", info) + " (the algorithm failed to converge)"); }
  columns_to_C(ws.a.data(), n, m, ws.w.data(), E, C);
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
  int m = require_outputs(fn, E, C, n);
  require_hermitian(fn, A.M, n, "A");
  thread_local ComplexWork ws;
  grow(ws.a, (std::size_t)n*n);
  grow(ws.w, n);
  std::copy(A.M, A.M + (std::size_t)n*n, ws.a.begin());
  int n_ = n, lda = n, info = 0, lwork = -1, lrwork = -1, liwork = -1;
  std::complex<double> wq(0.0, 0.0);
  double rwq = 0.0;
  int iwq = 0;
  zheevd_("V", "L", &n_, ws.a.data(), &lda, ws.w.data(), &wq, &lwork, &rwq, &lrwork, &iwq, &liwork, &info, 1, 1);
  if(info != 0){ fail(fn, info_text("zheevd (workspace query)", info)); }
  lwork = (int)wq.real();
  lrwork = (int)rwq;
  liwork = iwq;
  grow(ws.work, std::max(lwork, 1));
  grow(ws.rwork, std::max(lrwork, 1));
  grow(ws.iwork, std::max(liwork, 1));
  zheevd_("V", "L", &n_, ws.a.data(), &lda, ws.w.data(), ws.work.data(), &lwork, ws.rwork.data(), &lrwork, ws.iwork.data(), &liwork, &info, 1, 1);
  if(info != 0){ fail(fn, info_text("zheevd", info) + " (the algorithm failed to converge)"); }
  columns_to_C(ws.a.data(), n, m, ws.w.data(), E, C);
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
  int m = require_outputs(fn, E, C, n);
  require_hermitian(fn, H.M, n, "H");
  require_hermitian(fn, S.M, n, "S");
  thread_local RealWork ws;
  grow(ws.a, (std::size_t)n*n);
  grow(ws.b, (std::size_t)n*n);
  grow(ws.w, n);
  std::copy(H.M, H.M + (std::size_t)n*n, ws.a.begin());
  std::copy(S.M, S.M + (std::size_t)n*n, ws.b.begin());
  int itype = 1, n_ = n, lda = n, ldb = n, info = 0, lwork = -1, liwork = -1;
  double wq = 0.0;
  int iwq = 0;
  dsygvd_(&itype, "V", "L", &n_, ws.a.data(), &lda, ws.b.data(), &ldb, ws.w.data(), &wq, &lwork, &iwq, &liwork, &info, 1, 1);
  if(info != 0){ fail(fn, info_text("dsygvd (workspace query)", info)); }
  lwork = (int)wq;
  liwork = iwq;
  grow(ws.work, std::max(lwork, 1));
  grow(ws.iwork, std::max(liwork, 1));
  dsygvd_(&itype, "V", "L", &n_, ws.a.data(), &lda, ws.b.data(), &ldb, ws.w.data(), ws.work.data(), &lwork, ws.iwork.data(), &liwork, &info, 1, 1);
  if(info > n){ not_positive_definite(fn, info, n); }
  if(info != 0){ fail(fn, info_text("dsygvd", info) + " (the algorithm failed to converge)"); }
  columns_to_C(ws.a.data(), n, m, ws.w.data(), E, C);
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
  int m = require_outputs(fn, E, C, n);
  require_hermitian(fn, H.M, n, "H");
  require_hermitian(fn, S.M, n, "S");
  thread_local ComplexWork ws;
  grow(ws.a, (std::size_t)n*n);
  grow(ws.b, (std::size_t)n*n);
  grow(ws.w, n);
  std::copy(H.M, H.M + (std::size_t)n*n, ws.a.begin());
  std::copy(S.M, S.M + (std::size_t)n*n, ws.b.begin());
  int itype = 1, n_ = n, lda = n, ldb = n, info = 0, lwork = -1, lrwork = -1, liwork = -1;
  std::complex<double> wq(0.0, 0.0);
  double rwq = 0.0;
  int iwq = 0;
  zhegvd_(&itype, "V", "L", &n_, ws.a.data(), &lda, ws.b.data(), &ldb, ws.w.data(), &wq, &lwork, &rwq, &lrwork, &iwq, &liwork, &info, 1, 1);
  if(info != 0){ fail(fn, info_text("zhegvd (workspace query)", info)); }
  lwork = (int)wq.real();
  lrwork = (int)rwq;
  liwork = iwq;
  grow(ws.work, std::max(lwork, 1));
  grow(ws.rwork, std::max(lrwork, 1));
  grow(ws.iwork, std::max(liwork, 1));
  zhegvd_(&itype, "V", "L", &n_, ws.a.data(), &lda, ws.b.data(), &ldb, ws.w.data(), ws.work.data(), &lwork, ws.rwork.data(), &lrwork, ws.iwork.data(), &liwork, &info, 1, 1);
  if(info > n){ not_positive_definite(fn, info, n); }
  if(info != 0){ fail(fn, info_text("zhegvd", info) + " (the algorithm failed to converge)"); }
  columns_to_C(ws.a.data(), n, m, ws.w.data(), E, C);
#else
  (void)H; (void)S; (void)E; (void)C;
  fail(fn, MISSING);
#endif
}


void lapack_geev(const MATRIX& A, CMATRIX& E, CMATRIX& C){
  const char* fn = "lapack_geev";
#ifdef HAVE_LAPACK
  int n = A.n_rows;
  if(n < 1){ fail(fn, "A is empty"); }
  require_n_by_n(fn, A, "A", n);
  int m = require_outputs(fn, E, C, n);
  thread_local RealWork ws;
  thread_local std::vector<double> wi, vr;
  grow(ws.a, (std::size_t)n*n);
  grow(ws.w, n);
  grow(wi, n);
  grow(vr, (std::size_t)n*n);
  // a general matrix has no symmetry to lean on: the column-major copy is the transpose
  for(int i = 0; i < n; i++){
    for(int j = 0; j < n; j++){ ws.a[(std::size_t)j*n + i] = A.M[i*n+j]; }
  }
  int n_ = n, lda = n, ldvl = 1, ldvr = n, info = 0, lwork = -1;
  double vl_dummy = 0.0;
  double wq = 0.0;
  dgeev_("N", "V", &n_, ws.a.data(), &lda, ws.w.data(), wi.data(), &vl_dummy, &ldvl, vr.data(), &ldvr, &wq, &lwork, &info, 1, 1);
  if(info != 0){ fail(fn, info_text("dgeev (workspace query)", info)); }
  lwork = (int)wq;
  grow(ws.work, std::max(lwork, 1));
  dgeev_("N", "V", &n_, ws.a.data(), &lda, ws.w.data(), wi.data(), &vl_dummy, &ldvl, vr.data(), &ldvr, ws.work.data(), &lwork, &info, 1, 1);
  if(info != 0){ fail(fn, info_text("dgeev", info) + " (the QR algorithm failed to compute all eigenvalues)"); }
  // a complex conjugate pair (wi > 0 first) is stored as the real and imaginary columns of the first
  // member - the second member is its conjugate
  E = std::complex<double>(0.0, 0.0);
  for(int k = 0; k < m; k++){ E.M[k*m+k] = std::complex<double>(ws.w[k], wi[k]); }
  for(int k = 0; k < m; k++){
    if(wi[k] == 0.0){
      for(int i = 0; i < n; i++){ C.M[i*m+k] = std::complex<double>(vr[(std::size_t)k*n + i], 0.0); }
    }
    else if(wi[k] > 0.0){
      for(int i = 0; i < n; i++){ C.M[i*m+k] = std::complex<double>(vr[(std::size_t)k*n + i], vr[(std::size_t)(k+1)*n + i]); }
    }
    else{
      for(int i = 0; i < n; i++){ C.M[i*m+k] = std::complex<double>(vr[(std::size_t)(k-1)*n + i], -vr[(std::size_t)k*n + i]); }
    }
  }
#else
  (void)A; (void)E; (void)C;
  fail(fn, MISSING);
#endif
}


void lapack_geev(const CMATRIX& A, CMATRIX& E, CMATRIX& C){
  const char* fn = "lapack_geev";
#ifdef HAVE_LAPACK
  int n = A.n_rows;
  if(n < 1){ fail(fn, "A is empty"); }
  require_n_by_n(fn, A, "A", n);
  int m = require_outputs(fn, E, C, n);
  thread_local ComplexWork ws;
  thread_local std::vector< std::complex<double> > wz, vr;
  grow(ws.a, (std::size_t)n*n);
  grow(wz, n);
  grow(vr, (std::size_t)n*n);
  grow(ws.rwork, 2*n);
  for(int i = 0; i < n; i++){
    for(int j = 0; j < n; j++){ ws.a[(std::size_t)j*n + i] = A.M[i*n+j]; }
  }
  int n_ = n, lda = n, ldvl = 1, ldvr = n, info = 0, lwork = -1;
  std::complex<double> vl_dummy(0.0, 0.0);
  std::complex<double> wq(0.0, 0.0);
  zgeev_("N", "V", &n_, ws.a.data(), &lda, wz.data(), &vl_dummy, &ldvl, vr.data(), &ldvr, &wq, &lwork, ws.rwork.data(), &info, 1, 1);
  if(info != 0){ fail(fn, info_text("zgeev (workspace query)", info)); }
  lwork = (int)wq.real();
  grow(ws.work, std::max(lwork, 1));
  zgeev_("N", "V", &n_, ws.a.data(), &lda, wz.data(), &vl_dummy, &ldvl, vr.data(), &ldvr, ws.work.data(), &lwork, ws.rwork.data(), &info, 1, 1);
  if(info != 0){ fail(fn, info_text("zgeev", info) + " (the QR algorithm failed to compute all eigenvalues)"); }
  E = std::complex<double>(0.0, 0.0);
  for(int k = 0; k < m; k++){
    E.M[k*m+k] = wz[k];
    for(int i = 0; i < n; i++){ C.M[i*m+k] = vr[(std::size_t)k*n + i]; }
  }
#else
  (void)A; (void)E; (void)C;
  fail(fn, MISSING);
#endif
}


double lapack_det(const MATRIX& A){
  const char* fn = "lapack_det";
#ifdef HAVE_LAPACK
  int n = A.n_rows;
  if(n < 1){ fail(fn, "A is empty"); }
  require_n_by_n(fn, A, "A", n);
  thread_local RealWork ws;
  grow(ws.a, (std::size_t)n*n);
  grow(ws.iwork, n);
  // det(A^T) = det(A): the row-major buffer read column-major serves as it is
  std::copy(A.M, A.M + (std::size_t)n*n, ws.a.begin());
  int n_ = n, lda = n, info = 0;
  dgetrf_(&n_, &n_, ws.a.data(), &lda, ws.iwork.data(), &info);
  if(info < 0){ fail(fn, info_text("dgetrf", info)); }
  if(info > 0){ return 0.0; }
  double d = 1.0;
  for(int i = 0; i < n; i++){
    d *= ws.a[(std::size_t)i*n + i];
    if(ws.iwork[i] != i + 1){ d = -d; }
  }
  return d;
#else
  (void)A;
  fail(fn, MISSING);
  return 0.0;
#endif
}


std::complex<double> lapack_det(const CMATRIX& A){
  const char* fn = "lapack_det";
#ifdef HAVE_LAPACK
  int n = A.n_rows;
  if(n < 1){ fail(fn, "A is empty"); }
  require_n_by_n(fn, A, "A", n);
  thread_local ComplexWork ws;
  grow(ws.a, (std::size_t)n*n);
  grow(ws.iwork, n);
  std::copy(A.M, A.M + (std::size_t)n*n, ws.a.begin());
  int n_ = n, lda = n, info = 0;
  zgetrf_(&n_, &n_, ws.a.data(), &lda, ws.iwork.data(), &info);
  if(info < 0){ fail(fn, info_text("zgetrf", info)); }
  if(info > 0){ return std::complex<double>(0.0, 0.0); }
  std::complex<double> d(1.0, 0.0);
  for(int i = 0; i < n; i++){
    d *= ws.a[(std::size_t)i*n + i];
    if(ws.iwork[i] != i + 1){ d = -d; }
  }
  return d;
#else
  (void)A;
  fail(fn, MISSING);
  return std::complex<double>(0.0, 0.0);
#endif
}


namespace {

#ifdef HAVE_LAPACK

// the rank from a completed complete-pivoting factorization: the pivots above rel_threshold
// times the largest one (Eigen's FullPivLU rule with its default threshold eps n)
template <typename T>
int pivots_above(const T* a, int n, double rel_threshold){
  double pmax = 0.0;
  for(int i = 0; i < n; i++){ pmax = std::max(pmax, std::abs(a[(std::size_t)i*n + i])); }
  int rank = 0;
  for(int i = 0; i < n; i++){ if(std::abs(a[(std::size_t)i*n + i]) > rel_threshold * pmax){ rank++; } }
  return rank;
}

// the factors of P A Q = L U from the packed column-major LU and the sequential interchanges
// ipiv (rows) and jpiv (columns): P = S_n ... S_1, Q = T_1 ... T_n applied to the identity
template <typename T, typename M>
void unpack_getc2(const T* a, const int* ipiv, const int* jpiv, int n, M& P, M& L, M& U, M& Q){
  P = 0.0; L = 0.0; U = 0.0; Q = 0.0;
  for(int i = 0; i < n; i++){ P.M[i*n+i] = 1.0; Q.M[i*n+i] = 1.0; }
  for(int i = 0; i < n; i++){
    if(ipiv[i] != i + 1){ P.swap_rows(i, ipiv[i] - 1); }
    if(jpiv[i] != i + 1){ Q.swap_cols(i, jpiv[i] - 1); }
  }
  for(int i = 0; i < n; i++){
    L.M[i*n+i] = 1.0;
    for(int j = 0; j < i; j++){ L.M[i*n+j] = a[(std::size_t)j*n + i]; }
    for(int j = i; j < n; j++){ U.M[i*n+j] = a[(std::size_t)j*n + i]; }
  }
}

#endif

}// anonymous namespace


int lapack_getc2_rank(const MATRIX& A, double rel_threshold){
  const char* fn = "lapack_getc2_rank";
#ifdef HAVE_LAPACK
  int n = A.n_rows;
  if(n < 1){ fail(fn, "A is empty"); }
  require_n_by_n(fn, A, "A", n);
  thread_local RealWork ws;
  thread_local std::vector<int> jpiv;
  grow(ws.a, (std::size_t)n*n);
  grow(ws.iwork, n);
  grow(jpiv, n);
  // rank(A^T) = rank(A): the row-major buffer read column-major serves as it is
  std::copy(A.M, A.M + (std::size_t)n*n, ws.a.begin());
  int n_ = n, lda = n, info = 0;
  dgetc2_(&n_, ws.a.data(), &lda, ws.iwork.data(), jpiv.data(), &info);
  if(info < 0){ fail(fn, info_text("dgetc2", info)); }
  return pivots_above(ws.a.data(), n, rel_threshold);
#else
  (void)A; (void)rel_threshold;
  fail(fn, MISSING);
  return -1;
#endif
}


int lapack_getc2_rank(const CMATRIX& A, double rel_threshold){
  const char* fn = "lapack_getc2_rank";
#ifdef HAVE_LAPACK
  int n = A.n_rows;
  if(n < 1){ fail(fn, "A is empty"); }
  require_n_by_n(fn, A, "A", n);
  thread_local ComplexWork ws;
  thread_local std::vector<int> jpiv;
  grow(ws.a, (std::size_t)n*n);
  grow(ws.iwork, n);
  grow(jpiv, n);
  std::copy(A.M, A.M + (std::size_t)n*n, ws.a.begin());
  int n_ = n, lda = n, info = 0;
  zgetc2_(&n_, ws.a.data(), &lda, ws.iwork.data(), jpiv.data(), &info);
  if(info < 0){ fail(fn, info_text("zgetc2", info)); }
  return pivots_above(ws.a.data(), n, rel_threshold);
#else
  (void)A; (void)rel_threshold;
  fail(fn, MISSING);
  return -1;
#endif
}


void lapack_getc2(const MATRIX& A, MATRIX& P, MATRIX& L, MATRIX& U, MATRIX& Q){
  const char* fn = "lapack_getc2";
#ifdef HAVE_LAPACK
  int n = A.n_rows;
  if(n < 1){ fail(fn, "A is empty"); }
  require_n_by_n(fn, A, "A", n);
  require_n_by_n(fn, P, "P", n);
  require_n_by_n(fn, L, "L", n);
  require_n_by_n(fn, U, "U", n);
  require_n_by_n(fn, Q, "Q", n);
  thread_local RealWork ws;
  thread_local std::vector<int> jpiv;
  grow(ws.a, (std::size_t)n*n);
  grow(ws.iwork, n);
  grow(jpiv, n);
  // the factors of A itself: transpose-copy into the column-major buffer
  for(int i = 0; i < n; i++){
    for(int j = 0; j < n; j++){ ws.a[(std::size_t)j*n + i] = A.M[i*n+j]; }
  }
  int n_ = n, lda = n, info = 0;
  dgetc2_(&n_, ws.a.data(), &lda, ws.iwork.data(), jpiv.data(), &info);
  if(info < 0){ fail(fn, info_text("dgetc2", info)); }
  unpack_getc2(ws.a.data(), ws.iwork.data(), jpiv.data(), n, P, L, U, Q);
#else
  (void)A; (void)P; (void)L; (void)U; (void)Q;
  fail(fn, MISSING);
#endif
}


void lapack_getc2(const CMATRIX& A, CMATRIX& P, CMATRIX& L, CMATRIX& U, CMATRIX& Q){
  const char* fn = "lapack_getc2";
#ifdef HAVE_LAPACK
  int n = A.n_rows;
  if(n < 1){ fail(fn, "A is empty"); }
  require_n_by_n(fn, A, "A", n);
  require_n_by_n(fn, P, "P", n);
  require_n_by_n(fn, L, "L", n);
  require_n_by_n(fn, U, "U", n);
  require_n_by_n(fn, Q, "Q", n);
  thread_local ComplexWork ws;
  thread_local std::vector<int> jpiv;
  grow(ws.a, (std::size_t)n*n);
  grow(ws.iwork, n);
  grow(jpiv, n);
  for(int i = 0; i < n; i++){
    for(int j = 0; j < n; j++){ ws.a[(std::size_t)j*n + i] = A.M[i*n+j]; }
  }
  int n_ = n, lda = n, info = 0;
  zgetc2_(&n_, ws.a.data(), &lda, ws.iwork.data(), jpiv.data(), &info);
  if(info < 0){ fail(fn, info_text("zgetc2", info)); }
  unpack_getc2(ws.a.data(), ws.iwork.data(), jpiv.data(), n, P, L, U, Q);
#else
  (void)A; (void)P; (void)L; (void)U; (void)Q;
  fail(fn, MISSING);
#endif
}


namespace {

#ifdef HAVE_LAPACK

template <typename M>
void require_svd_shapes(const char* fn, const M& A, const M& U, const M& S, const M& V, int m, int n){
  if(m < 1 || n < 1){ fail(fn, "A is empty"); }
  if(U.n_rows != m || U.n_cols != m || S.n_rows != m || S.n_cols != n || V.n_rows != n || V.n_cols != n){
    std::ostringstream o;
    o<<"for A "<<m<<" x "<<n<<" the factors must be U "<<m<<" x "<<m<<", S "<<m<<" x "<<n<<", V "<<n<<" x "<<n
     <<", got U "<<U.n_rows<<" x "<<U.n_cols<<", S "<<S.n_rows<<" x "<<S.n_cols<<", V "<<V.n_rows<<" x "<<V.n_cols;
    fail(fn, o.str());
  }
}

#endif

}// anonymous namespace


void lapack_gesdd(const MATRIX& A, MATRIX& U, MATRIX& S, MATRIX& V){
  const char* fn = "lapack_gesdd";
#ifdef HAVE_LAPACK
  int m = A.n_rows, n = A.n_cols;
  require_svd_shapes(fn, A, U, S, V, m, n);
  int mn = std::min(m, n), mx = std::max(m, n);
  thread_local RealWork ws;
  thread_local std::vector<double> u, vt;
  grow(ws.a, (std::size_t)m*n);
  grow(ws.w, mn);
  grow(u, (std::size_t)m*m);
  grow(vt, (std::size_t)n*n);
  grow(ws.iwork, 8*mn);
  // the factors of A itself: transpose-copy into the column-major buffer (m x n, lda = m)
  for(int i = 0; i < m; i++){
    for(int j = 0; j < n; j++){ ws.a[(std::size_t)j*m + i] = A.M[i*n+j]; }
  }
  int m_ = m, n_ = n, lda = m, ldu = m, ldvt = n, info = 0, lwork = -1;
  double wq = 0.0;
  dgesdd_("A", &m_, &n_, ws.a.data(), &lda, ws.w.data(), u.data(), &ldu, vt.data(), &ldvt, &wq, &lwork, ws.iwork.data(), &info, 1);
  if(info != 0){ fail(fn, info_text("dgesdd (workspace query)", info)); }
  lwork = (int)wq;
  grow(ws.work, std::max(lwork, 1));
  dgesdd_("A", &m_, &n_, ws.a.data(), &lda, ws.w.data(), u.data(), &ldu, vt.data(), &ldvt, ws.work.data(), &lwork, ws.iwork.data(), &info, 1);
  if(info != 0){ fail(fn, info_text("dgesdd", info) + " (the algorithm failed to converge)"); }
  // U column-major -> libra rows, S the rectangular diagonal, V = VT^T (the column-major VT read row-major)
  for(int i = 0; i < m; i++){
    for(int j = 0; j < m; j++){ U.M[i*m+j] = u[(std::size_t)j*m + i]; }
  }
  S = 0.0;
  for(int k = 0; k < mn; k++){ S.M[k*n+k] = ws.w[k]; }
  std::copy(vt.begin(), vt.begin() + (std::size_t)n*n, V.M);
  (void)mx;
#else
  (void)A; (void)U; (void)S; (void)V;
  fail(fn, MISSING);
#endif
}


void lapack_gesdd(const CMATRIX& A, CMATRIX& U, CMATRIX& S, CMATRIX& V){
  const char* fn = "lapack_gesdd";
#ifdef HAVE_LAPACK
  int m = A.n_rows, n = A.n_cols;
  require_svd_shapes(fn, A, U, S, V, m, n);
  int mn = std::min(m, n), mx = std::max(m, n);
  thread_local ComplexWork ws;
  thread_local std::vector< std::complex<double> > u, vt;
  grow(ws.a, (std::size_t)m*n);
  grow(ws.w, mn);
  grow(u, (std::size_t)m*m);
  grow(vt, (std::size_t)n*n);
  grow(ws.iwork, 8*mn);
  grow(ws.rwork, (std::size_t)5*mn*mn + 7*mn + (std::size_t)2*mx*mn);
  for(int i = 0; i < m; i++){
    for(int j = 0; j < n; j++){ ws.a[(std::size_t)j*m + i] = A.M[i*n+j]; }
  }
  int m_ = m, n_ = n, lda = m, ldu = m, ldvt = n, info = 0, lwork = -1;
  std::complex<double> wq(0.0, 0.0);
  zgesdd_("A", &m_, &n_, ws.a.data(), &lda, ws.w.data(), u.data(), &ldu, vt.data(), &ldvt, &wq, &lwork, ws.rwork.data(), ws.iwork.data(), &info, 1);
  if(info != 0){ fail(fn, info_text("zgesdd (workspace query)", info)); }
  lwork = (int)wq.real();
  grow(ws.work, std::max(lwork, 1));
  zgesdd_("A", &m_, &n_, ws.a.data(), &lda, ws.w.data(), u.data(), &ldu, vt.data(), &ldvt, ws.work.data(), &lwork, ws.rwork.data(), ws.iwork.data(), &info, 1);
  if(info != 0){ fail(fn, info_text("zgesdd", info) + " (the algorithm failed to converge)"); }
  // U column-major -> libra rows, S the rectangular diagonal, V = VT^H (conjugate of the column-major VT read row-major)
  for(int i = 0; i < m; i++){
    for(int j = 0; j < m; j++){ U.M[i*m+j] = u[(std::size_t)j*m + i]; }
  }
  S = std::complex<double>(0.0, 0.0);
  for(int k = 0; k < mn; k++){ S.M[k*n+k] = std::complex<double>(ws.w[k], 0.0); }
  for(std::size_t idx = 0; idx < (std::size_t)n*n; idx++){ V.M[idx] = std::conj(vt[idx]); }
#else
  (void)A; (void)U; (void)S; (void)V;
  fail(fn, MISSING);
#endif
}


void lapack_potrf(const MATRIX& A, MATRIX& L){
  const char* fn = "lapack_potrf";
#ifdef HAVE_LAPACK
  int n = A.n_rows;
  if(n < 1){ fail(fn, "A is empty"); }
  require_n_by_n(fn, A, "A", n);
  require_n_by_n(fn, L, "L", n);
  require_hermitian(fn, A.M, n, "A");
  thread_local RealWork ws;
  grow(ws.a, (std::size_t)n*n);
  // A symmetric: the row-major buffer read column-major is A itself
  std::copy(A.M, A.M + (std::size_t)n*n, ws.a.begin());
  int n_ = n, lda = n, info = 0;
  dpotrf_("L", &n_, ws.a.data(), &lda, &info, 1);
  if(info > 0){
    std::ostringstream o;
    o<<"A is not positive definite (its leading minor of order "<<info<<" is not)";
    fail(fn, o.str());
  }
  if(info != 0){ fail(fn, info_text("dpotrf", info)); }
  L = 0.0;
  for(int i = 0; i < n; i++){
    for(int j = 0; j <= i; j++){ L.M[i*n+j] = ws.a[(std::size_t)j*n + i]; }
  }
#else
  (void)A; (void)L;
  fail(fn, MISSING);
#endif
}


void lapack_potrf(const CMATRIX& A, CMATRIX& L){
  const char* fn = "lapack_potrf";
#ifdef HAVE_LAPACK
  int n = A.n_rows;
  if(n < 1){ fail(fn, "A is empty"); }
  require_n_by_n(fn, A, "A", n);
  require_n_by_n(fn, L, "L", n);
  require_hermitian(fn, A.M, n, "A");
  thread_local ComplexWork ws;
  grow(ws.a, (std::size_t)n*n);
  // A Hermitian: the row-major buffer read column-major is conj(A) = A^T, whose factor is conj(L)
  std::copy(A.M, A.M + (std::size_t)n*n, ws.a.begin());
  int n_ = n, lda = n, info = 0;
  zpotrf_("L", &n_, ws.a.data(), &lda, &info, 1);
  if(info > 0){
    std::ostringstream o;
    o<<"A is not positive definite (its leading minor of order "<<info<<" is not)";
    fail(fn, o.str());
  }
  if(info != 0){ fail(fn, info_text("zpotrf", info)); }
  L = std::complex<double>(0.0, 0.0);
  for(int i = 0; i < n; i++){
    for(int j = 0; j <= i; j++){ L.M[i*n+j] = std::conj(ws.a[(std::size_t)j*n + i]); }
  }
#else
  (void)A; (void)L;
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
  thread_local RealWork ws;
  grow(ws.a, (std::size_t)n*n);
  grow(ws.iwork, n);
  std::copy(A.M, A.M + (std::size_t)n*n, ws.a.begin());
  int n_ = n, lda = n, info = 0, lwork = -1;
  dgetrf_(&n_, &n_, ws.a.data(), &lda, ws.iwork.data(), &info);
  if(info > 0){
    std::ostringstream o;
    o<<"A is singular (U("<<info<<","<<info<<") of its LU factorization is exactly zero)";
    fail(fn, o.str());
  }
  if(info != 0){ fail(fn, info_text("dgetrf", info)); }
  double wq = 0.0;
  dgetri_(&n_, ws.a.data(), &lda, ws.iwork.data(), &wq, &lwork, &info);
  if(info != 0){ fail(fn, info_text("dgetri (workspace query)", info)); }
  lwork = (int)wq;
  grow(ws.work, std::max(lwork, 1));
  dgetri_(&n_, ws.a.data(), &lda, ws.iwork.data(), ws.work.data(), &lwork, &info);
  if(info != 0){ fail(fn, info_text("dgetri", info)); }
  std::copy(ws.a.begin(), ws.a.begin() + (std::size_t)n*n, Ainv.M);
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
  thread_local ComplexWork ws;
  grow(ws.a, (std::size_t)n*n);
  grow(ws.iwork, n);
  std::copy(A.M, A.M + (std::size_t)n*n, ws.a.begin());
  int n_ = n, lda = n, info = 0, lwork = -1;
  zgetrf_(&n_, &n_, ws.a.data(), &lda, ws.iwork.data(), &info);
  if(info > 0){
    std::ostringstream o;
    o<<"A is singular (U("<<info<<","<<info<<") of its LU factorization is exactly zero)";
    fail(fn, o.str());
  }
  if(info != 0){ fail(fn, info_text("zgetrf", info)); }
  std::complex<double> wq(0.0, 0.0);
  zgetri_(&n_, ws.a.data(), &lda, ws.iwork.data(), &wq, &lwork, &info);
  if(info != 0){ fail(fn, info_text("zgetri (workspace query)", info)); }
  lwork = (int)wq.real();
  grow(ws.work, std::max(lwork, 1));
  zgetri_(&n_, ws.a.data(), &lda, ws.iwork.data(), ws.work.data(), &lwork, &info);
  if(info != 0){ fail(fn, info_text("zgetri", info)); }
  std::copy(ws.a.begin(), ws.a.begin() + (std::size_t)n*n, Ainv.M);
#else
  (void)A; (void)Ainv;
  fail(fn, MISSING);
#endif
}


}// namespace liblinalg
}// namespace liblibra
