/*********************************************************************************
 * Copyright (C) 2021 Mohammad Shakiba and Alexey V. Akimov
 *
 * This file is distributed under the terms of the GNU General Public License
 * as published by the Free Software Foundation, either version 3 of
 * the License, or (at your option) any later version.
 * See the file LICENSE in the root directory of this distribution
 * or <http://www.gnu.org/licenses/>.
 *
 * This file uses Libint library (https://github.com/evaleev/libint) for
 * computing the atomic orbitals overlaps for two different geometries. This can
 * be used to compute the molecular orbital overlaps for non-adiabatic molecular
 * dynamics. The code uses Pybid11 to interface the code with Python.
 *
 * This code maily uses the test files in
 * (https://github.com/evaleev/libint/tree/master/tests/hartree-fock) with
 * modifications to be used for computing the atomic orbitals overlap.
 *********************************************************************************/

/**
  \file libint2_wrappers.cpp
  \brief The file describes the classes and functions that interface libint2
  functions and types with those of Libra and expose them to Python

*/

#include "libint2_wrappers.h"
#include <cmath>
#include <atomic>

/// liblibra namespace
namespace liblibra {

using namespace liblinalg;
using namespace libspecialfunctions;

namespace liblibint2_wrappers {

using libint2::Engine;
using libint2::Operator;
using libint2::Shell;
using std::cout;
using std::endl;

// We first need to initailize a libint2::Shell. Here is how we do it.
std::vector<libint2::Shell>
initialize_shell(int l_val, bool is_spherical,
                 const std::vector<double> &exponents,
                 const std::vector<double> &coeff, VECTOR &coords) {

  std::vector<Shell> shells;

  // The same as above explanation for exponents and contraction coefficients
  libint2::svector<double> q;
  libint2::svector<double> p;
  for (auto i = 0u; i < exponents.size(); i++) {
    q.push_back(exponents[i]);
    p.push_back(coeff[i]);
  }

  // Append to shell
  shells.push_back({
      q, // exponents of primitive Gaussians
      {  // contraction 0: s shell (l=0), spherical=false, contraction
       // coefficients
       {l_val, is_spherical, p}},
      {{coords.x, coords.y, coords.z}} // origin coordinates
  });

  return shells;
}

// Create a unit shell with exponents of 0 and coefficients of 1 from
// libint2::Shell::unit().
libint2::Shell unit_shell() {

  const auto &unitshell = libint2::Shell::unit();

  return unitshell;
}

// ---------------------------------------------------------------------
// Solid-harmonic component ordering of the linked libint runtime.
//
// libint emits pure-shell components in one of two compile-time orders:
// gaussian (m = 0, +1, -1, +2, -2, ...) or standard (m = -l..+l). The
// molden/molog readers and the step-2 writers all assume gaussian, and
// the LIBINT_SHGSHELL_ORDERING header macro cannot be trusted (packaged
// headers are known to disagree with the shipped binary). Probe the
// runtime once: the overlap of an s shell at the origin with a pure p
// shell displaced along z is nonzero only in the p_z (m = 0) component,
// whose position identifies the ordering.
static bool shg_ordering_is_standard() {
  static const bool is_standard = [] {
    if (!libint2::initialized()) { libint2::initialize(); }
    libint2::svector<double> e;
    e.push_back(1.0);
    libint2::svector<double> c;
    c.push_back(1.0);
    libint2::Shell s_shell{e, {{0, false, c}}, {{0.0, 0.0, 0.0}}};
    libint2::Shell p_shell{e, {{1, true, c}}, {{0.0, 0.0, 1.0}}};
    libint2::Engine engine(libint2::Operator::overlap, 1, 1, 0);
    const auto &buf = engine.results();
    engine.compute(s_shell, p_shell);
    const double a0 = std::fabs(buf[0][0]);
    const double a1 = std::fabs(buf[0][1]);
    const double a2 = std::fabs(buf[0][2]);
    const double big = std::max(a0, std::max(a1, a2));
    const double tol = 1e-8 * (big > 0.0 ? big : 1.0);
    const bool gauss = (a0 > tol) && (a1 <= tol) && (a2 <= tol);
    const bool standard = (a1 > tol) && (a0 <= tol) && (a2 <= tol);
    if (!gauss && !standard) {
      std::cout << "Error in libint2_wrappers: the solid-harmonic ordering "
                   "probe (s at origin vs pure p at z = 1) has no single "
                   "dominant component (" << a0 << ", " << a1 << ", " << a2
                << ") - expected p_z first (gaussian) or second (standard)\n";
      throw std::runtime_error(
          "libra C++ error - see the message printed above");
    }
    std::cout << "libint solid-harmonic ordering: "
              << (standard ? "standard (m = -l..+l) - wrapper outputs are "
                             "permuted to the gaussian convention"
                           : "gaussian (m = 0, +1, -1, ...)")
              << "\n";
    return standard;
  }();
  return is_standard;
}

// gaussian position of the f-th standard-ordered component of a pure
// shell with angular momentum l: m = f - l; gaussian packs m = 0 first,
// then +1, -1, +2, -2, ...
static inline std::size_t shg_gaussian_pos(int l, std::size_t f) {
  const int m = static_cast<int>(f) - l;
  return static_cast<std::size_t>(m == 0 ? 0 : (m > 0 ? 2 * m - 1 : -2 * m));
}

// Component index map of one shell: identity for cartesian shells and on
// gaussian-ordered runtimes.
static void shg_index_map(const libint2::Shell &sh, std::vector<std::size_t> &map) {
  const std::size_t n = sh.size();
  map.resize(n);
  const int l = sh.contr[0].l;
  const bool pure = sh.contr[0].pure;
  for (std::size_t f = 0; f < n; f++) {
    map[f] = (pure && l > 0) ? shg_gaussian_pos(l, f) : f;
  }
}

// This function adds the basis sets for each atomic type to a libint2::Shell
// variable
void add_to_shell(std::vector<libint2::Shell> &shells, int l_val,
                  bool is_spherical, const std::vector<double> &exponents,
                  const std::vector<double> &coeff, VECTOR &coords) {

  // We need libint2::svector (take a look at shell.h file in libint library
  // folder) q for exponents
  libint2::svector<double> q;
  // p for contraction coefficients (libint will automatically converts them to
  // refer to normalization free primitives before computing integrals)
  // (Shell::renorm in shell.h)
  libint2::svector<double> p;

  // Appending the exponents and contraction coefficients from Python to q and p
  // respectively.
  for (auto i = 0u; i < exponents.size(); i++) {
    q.push_back(exponents[i]);
    p.push_back(coeff[i]);
  }

  // The spherical/cartesian flag for shell
  shells.push_back({
      q, // exponents of primitive Gaussians
      {  // contraction 0: s shell (l=0), spherical=false, contraction
       // coefficients
       {l_val, is_spherical, p}},
      {{coords.x, coords.y, coords.z}} // origin coordinates
  });
}

// This function prints the shells. It is useful to check the results (this is
// also interfaced with Python using Pybind11)
void print_shells(std::vector<libint2::Shell> &shells) {

  std::cout << "\n\tShells are:\n";
  for (auto s = 0u; s < shells.size(); s++) {
    std::cout << shells[s] << std::endl;
    // The shell size
    std::cout << "\n The shell size is:\n" << shells.size() << std::endl;
  }
}

// This function prints the shells. It is useful to check the results (this is
// also interfaced with Python using Pybind11)
void print_shells_2(std::vector<libint2::Shell> &shells) {

  std::cout << "\n\tShells are:\n";
  for (auto s = 0u; s < shells.size(); s++) {
    std::cout << shells[s] << std::endl;
    // The shell size
    std::cout << "\n The shell size is:\n" << shells.size() << std::endl;
  }
}

// Below are the functions that we use for starting the engine for computing the
// overlap integrals.
size_t nbasis(const std::vector<libint2::Shell> &shells) {
  size_t n = 0;
  for (auto i = 0u; i < shells.size(); i++) {
    n += shells[i].size();
  }

  return n;
}

size_t max_nprim(const std::vector<libint2::Shell> &shells) {
  size_t n = 0;
  for (auto i = 0u; i < shells.size(); i++) {
    n = std::max(shells[i].nprim(), n);
  }

  return n;
}

int max_l(const std::vector<libint2::Shell> &shells) {
  int l = 0;
  for (auto i = 0u; i < shells.size(); i++) {
    for (auto j = 0u; j < shells[i].contr.size(); j++) {
      l = std::max(shells[i].contr[j].l, l);
    }
  }

  return l;
}

std::vector<size_t>
map_shell_to_basis_function(const std::vector<libint2::Shell> &shells) {
  std::vector<size_t> result;
  result.reserve(shells.size());

  size_t n = 0;
  for (auto i = 0u; i < shells.size(); i++) {
    result.push_back(n);
    n += shells[i].size();
  }

  return result;
}

/*

// Computing the integrals between two shells serial
MATRIX compute_1body_ints(const std::vector<libint2::Shell>& shells_1, const
std::vector<libint2::Shell>& shells_2,libint2::Operator obtype)
{

  const auto n_1 = nbasis(shells_1);
  const auto n_2 = nbasis(shells_2);
  //std::cout << "nbasis (shells_1)" << n_1 << "\n";
  //std::cout << "nbasis (shells_2)" << n_2 << "\n";
  Matrix result(n_1,n_2);
  int max_n;
  int max_lval;
  if (max_nprim(shells_1)>max_nprim(shells_2))
  {
    max_n = max_nprim(shells_1);
  }
  else
  {
    max_n = max_nprim(shells_2);
  }
  if (max_l(shells_1)>max_l(shells_2))
  {
    max_lval = max_l(shells_1);
  }
  else
  {
    max_lval = max_l(shells_2);
  }

  Engine engine(obtype, max_n, max_lval, 0);

  auto shell2bf_1 = map_shell_to_basis_function(shells_1);
  auto shell2bf_2 = map_shell_to_basis_function(shells_2);

  // buf[0] points to the target shell set after every call  to engine.compute()
  const auto& buf = engine.results();

  // loop over unique shell pairs, {s1,s2} such that s1 >= s2
  // this is due to the permutational symmetry of the real integrals over
Hermitian operators: (1|2) = (2|1)
  //std::cout << shells_1.size();
  for(auto s1=0; s1!=shells_1.size(); ++s1) {

    auto bf1 = shell2bf_1[s1]; // first basis function in this shell
    //std::cout << "Flag bf1: " << bf1 << "\n";
    auto n1 = shells_1[s1].size();

    //std::cout << "shells_1[s1]" << shells_1[s1].size() << "\n";

    //for(auto s2=0; s2<=s1; ++s2) {
    for(auto s2=0; s2!=shells_2.size(); ++s2) {

      auto bf2 = shell2bf_2[s2];
      //std::cout << "Flag bf2: " << bf2 << "\n";
      auto n2 = shells_2[s2].size();

      //std::cout << "shells_2[s2]" << shells_2[s2].size() << "\n";

      // compute shell pair
      engine.compute(shells_1[s1], shells_2[s2]);
      //std::cout << "Flag after engine.compute" << "\n";
      // "map" buffer to a const Eigen Matrix, and copy it to the corresponding
blocks of the result Eigen::Map<const Matrix> buf_mat(buf[0], n1, n2);
      //std::cout << "Flag after buf_mat" << "\n" << "buf_mat.size: \n" <<
buf_mat.size() << "\n";
      //std::cout << "Flag after buf_mat" << "\n" << "buf_mat  \n" << buf_mat <<
"\n";
      //std::cout << "Flag after buf_mat" << "\n" << "bf1.size" << bf1.size() <<
"\n";
      //std::cout << "Flag after buf_mat" << "\n" << "bf2.size" << bf2.size() <<
"\n"; result.block(bf1, bf2, n1, n2) = buf_mat;
      //std::cout << "Flag after result.block" << "\n";
      if (s1 != s2) // if s1 >= s2, copy {s1,s2} to the corresponding {s2,s1}
block, note the transpose! result.block(bf2, bf1, n2, n1) = buf_mat.transpose();
      //std::cout << "Flag after buf_mat.transpose" << "\n";
    }
  }
  //std::cout << "Flag before MATRIX res" << "\n";
  MATRIX res(n_1, n_2);
  for(int i=0;i<n_1;i++){
    for(int j=0; j<n_2;j++){
      res.set(i,j, result(i,j));
    }
  }


  return res;
}

*/

// Computing the integrals between two shells in parallel using OpenMP (This is
// also adopted from libint test files with some modifications) In the main file
// we can also use other operators such as kinetic, nuclear, etc but since we
// only need overlap operator here we omit those parts.
MATRIX compute_1body_ints_parallel(const std::vector<libint2::Shell> &shells_1,
                                   const std::vector<libint2::Shell> &shells_2,
                                   libint2::Operator obtype, int nthreads) {
  // using libint2::nthreads;

  const auto n_1 = nbasis(shells_1);
  const auto n_2 = nbasis(shells_2);

  // the runtime's solid-harmonic ordering, probed once: standard-ordered
  // blocks are permuted below so the output is always gaussian-ordered
  const bool shg_std = shg_ordering_is_standard();

  // The AO overlap matrix n x n
  Matrix result(n_1, n_2);
  int max_n;
  int max_lval;
  if (max_nprim(shells_1) > max_nprim(shells_2)) {
    max_n = max_nprim(shells_1);
  } else {
    max_n = max_nprim(shells_2);
  }
  if (max_l(shells_1) > max_l(shells_2)) {
    max_lval = max_l(shells_1);
  } else {
    max_lval = max_l(shells_2);
  }

  // Initializing different engines based on the nthreads
  std::vector<libint2::Engine> engines(nthreads);
  // Initializing the first engine. Others will be the same as this one (as is
  // shown below)
  engines[0] = libint2::Engine(obtype, max_n, max_lval, 0);
  // This part is for other operators, I'll keep it but not necessary.
  //  engines[0].set_params(oparams);
  // Other engines
  for (auto i = 1; i != nthreads; ++i) {
    engines[i] = engines[0];
  }

  // Mapping the shells into basis sets
  auto shell2bf1 = map_shell_to_basis_function(shells_1);
  auto shell2bf2 = map_shell_to_basis_function(shells_2);
  // first shell pair that produced a null batch (packed s1*1e6 + s2)
  std::atomic<long long> null_pair(-1);
  // Compute for each thread_id
  auto compute = [&](int thread_id) {
    const auto &buf = engines[thread_id].results();

    for (auto s1 = 0ul; s1 != shells_1.size(); ++s1) {
      //    for (auto s1 = 0; s1 < shells_1.size(); ++s1) {
      auto bf1 = shell2bf1[s1]; // first basis function in this shell
      auto n1 = shells_1[s1].size();
      // std::cout << "shells_1[" << s1 << "]" << shells_1[s1].size() << "\n";
      auto s1_offset = s1 * (s1 + 1) / 2;
      // for (auto s2 = 0; s2 < shells_2.size(); ++s2) {
      for (auto s2 = 0u; s2 != shells_2.size(); ++s2) {
        // std::cout << "shells_2[" << s2 << "]" << shells_2[s2].size() << "\n";
        auto s12 = s1_offset + s2;
        if (static_cast<int>(s12 % nthreads) != thread_id)
          continue;
        auto bf2 = shell2bf2[s2]; // first basis function in this shell
        auto n2 = shells_2[s2].size();

        // Make compute for each engine
        engines[thread_id].compute(shells_1[s1], shells_2[s2]);
        // a 1-body engine must always produce a batch (screening-to-null
        // is a 2-body behavior) - a null here means a corrupted engine or
        // shell setup; record it and throw after the join (throwing from
        // a worker thread would std::terminate)
        if (buf[0] == nullptr) {
          long long expected = -1;
          null_pair.compare_exchange_strong(
              expected, static_cast<long long>(s1) * 1000000LL +
                            static_cast<long long>(s2));
          return;
        }
        // The results block
        Eigen::Map<const Matrix> buf_mat(buf[0], n1, n2);
        if (!shg_std) {
          result.block(bf1, bf2, n1, n2) = buf_mat;
        } else {
          std::vector<std::size_t> map1, map2;
          shg_index_map(shells_1[s1], map1);
          shg_index_map(shells_2[s2], map2);
          for (std::size_t f1 = 0; f1 < n1; f1++) {
            for (std::size_t f2 = 0; f2 < n2; f2++) {
              result(bf1 + map1[f1], bf2 + map2[f2]) = buf_mat(f1, f2);
            }
          }
        }
        // if (s1 != s2)  // if s1 >= s2, copy {s1,s2} to the corresponding
        //                // {s2,s1} block, note the transpose!
        // result.block(bf2, bf1, n2, n1) = buf_mat.transpose();
      }
    }
  };
  // std::cout << "Done";
  MATRIX res(n_1, n_2);
  // Now for compute, do parallel
  parallel_do(compute, nthreads);
  if (null_pair.load() >= 0) {
    std::cout << "Error in compute_1body_ints_parallel: libint returned a "
                 "null result batch for shell pair ("
              << null_pair.load() / 1000000LL << ", "
              << null_pair.load() % 1000000LL
              << ") - 1-body engines must always produce a batch; the "
                 "engine or shell setup is corrupted\n";
    throw std::runtime_error(
        "libra C++ error - see the message printed above");
  }
  for (auto i = 0u; i < n_1; ++i) {
    for (auto j = 0u; j < n_2; ++j) {
      res.set(i, j, result(i, j));
    }
  }

  // Return the AO overlap values
  return res;
}

// The main function for computing the AO overlaps. We will use this in python.
MATRIX compute_overlaps(const std::vector<libint2::Shell> &shells_1,
                        const std::vector<libint2::Shell> &shells_2,
                        int number_of_threads) {

  // int nthreads;

  //  using libint2::Operator;
  //  using libint2::nthreads;

  auto nthreads_cstr = getenv("LIBINT_NUM_THREADS");
  int nthreads = number_of_threads;
  if (nthreads_cstr && strcmp(nthreads_cstr, "")) {
    std::istringstream iss(nthreads_cstr);
    iss >> nthreads;
    if (nthreads > 1 << 16 || nthreads <= 0)
      nthreads = 1;
  }

#if defined(_OPENMP)
  omp_set_num_threads(nthreads);
#endif
  //     ;; //std::cout << "Will scale over " << nthreads
  // #if defined(_OPENMP)
  //     ;;//          << " OpenMP"
  // #else
  //     ;;//          << " C++11"
  // #endif
  //     ;;//          << " threads" << std::endl;
  // Initialize Libint
  libint2::initialize();
  // Compute the AO overlap matrix
  auto S = compute_1body_ints_parallel(shells_1, shells_2, Operator::overlap,
                                       nthreads);
  // std::cout << "\n\tFinished computing overlap integral\n";
  //  End of AO matrix calculation
  libint2::finalize(); // done with libint

  return S;
}

double compute_4center_eri(const std::vector<libint2::Shell> &shells_1,
                           const std::vector<libint2::Shell> &shells_2,
                           const std::vector<libint2::Shell> &shells_3,
                           const std::vector<libint2::Shell> &shells_4,
                           const int deriv_order) {
  // int number_of_threads) {

  int max_n = std::max(std::max(max_nprim(shells_1), max_nprim(shells_2)),
                       std::max(max_nprim(shells_3), max_nprim(shells_4)));
  int max_lval = std::max(std::max(max_l(shells_1), max_l(shells_2)),
                          std::max(max_l(shells_3), max_l(shells_4)));

  // Initialize Libint
  libint2::initialize();

  using libint2::BraKet;
  using libint2::Engine;
  using libint2::Operator;
  Engine engine(Operator::coulomb, max_n, max_lval, deriv_order);
  const auto &buf = engine.results();
  engine.compute(shells_1[0], shells_2[0], shells_3[0], shells_4[0]);
  const auto *integral_val = buf[0];
  libint2::finalize(); // done with libint
  // 2-body engines screen negligible quartets to a null batch - the
  // integral is genuinely zero within the engine precision
  return integral_val == nullptr ? 0.0 : integral_val[0];
}

/*
MATRIX compute_overlaps_serial(const std::vector<libint2::Shell>& shells_1,
const std::vector<libint2::Shell>& shells_2) {
    // Initialize Libint
    libint2::initialize();
        // Compute the AO overlap matrix
    auto S = compute_1body_ints(shells_1, shells_2, Operator::overlap);
    //std::cout << "\n\tFinished computing overlap integral\n";
    libint2::finalize(); // done with libint
    return S;
}

*/

std::vector<MATRIX> compute_1body_ints_parallel_emultipole3(
    const std::vector<libint2::Shell> &shells_1,
    const std::vector<libint2::Shell> &shells_2, int nthreads) {
  // using libint2::nthreads;

  const auto n_1 = nbasis(shells_1);
  const auto n_2 = nbasis(shells_2);

  // the runtime's solid-harmonic ordering, probed once (see the overlap
  // kernel note)
  const bool shg_std = shg_ordering_is_standard();

  std::vector<MATRIX> res;
  for (int i = 0; i < 20; i++) {
    MATRIX m2(n_1, n_2);
    res.push_back(std::move(m2));
  }

  std::vector<Matrix> results(20);
  for (Matrix &m : results) {
    m = Matrix(n_1, n_2);
  }
  int max_n;
  int max_lval;
  if (max_nprim(shells_1) > max_nprim(shells_2)) {
    max_n = max_nprim(shells_1);
  } else {
    max_n = max_nprim(shells_2);
  }
  if (max_l(shells_1) > max_l(shells_2)) {
    max_lval = max_l(shells_1);
  } else {
    max_lval = max_l(shells_2);
  }

  // Initializing different engines based on the nthreads
  std::vector<libint2::Engine> engines(nthreads);
  // Initializing the first engine. Others will be the same as this one (as is
  // shown below)
  engines[0] =
      libint2::Engine(libint2::Operator::emultipole3, max_n, max_lval, 0);
  // This part is for other operators, I'll keep it but not necessary.
  //  engines[0].set_params(oparams);
  // Other engines
  for (auto i = 1; i != nthreads; ++i) {
    engines[i] = engines[0];
  }

  // Mapping the shells into basis sets
  auto shell2bf1 = map_shell_to_basis_function(shells_1);
  auto shell2bf2 = map_shell_to_basis_function(shells_2);
  // first shell pair that produced a null batch (packed s1*1e6 + s2)
  std::atomic<long long> null_pair(-1);
  // Compute for each thread_id
  auto compute = [&](int thread_id) {
    const auto &buf = engines[thread_id].results();
    for (auto s1 = 0ul; s1 != shells_1.size(); ++s1) {
      //    for (auto s1 = 0; s1 < shells_1.size(); ++s1) {
      auto bf1 = shell2bf1[s1]; // first basis function in this shell
      auto n1 = shells_1[s1].size();
      // std::cout << "shells_1[" << s1 << "]" << shells_1[s1].size() << "\n";
      auto s1_offset = s1 * (s1 + 1) / 2;
      // for (auto s2 = 0; s2 < shells_2.size(); ++s2) {
      for (auto s2 = 0u; s2 != shells_2.size(); ++s2) {
        // std::cout << "shells_2[" << s2 << "]" << shells_2[s2].size() << "\n";
        auto s12 = s1_offset + s2;
        if (s12 % nthreads != static_cast<std::size_t>(thread_id))
          continue;
        auto bf2 = shell2bf2[s2]; // first basis function in this shell
        auto n2 = shells_2[s2].size();

        // Make compute for each engine
        engines[thread_id].compute(shells_1[s1], shells_2[s2]);
        std::vector<std::size_t> map1, map2;
        if (shg_std) {
          shg_index_map(shells_1[s1], map1);
          shg_index_map(shells_2[s2], map2);
        }
        for (int index = 0; index < 20; index++) {
          // 1-body engines must always produce a batch (see the overlap
          // kernel) - record and throw after the join
          if (buf[index] == nullptr) {
            long long expected = -1;
            null_pair.compare_exchange_strong(
                expected, static_cast<long long>(s1) * 1000000LL +
                              static_cast<long long>(s2));
            return;
          }
          // The results block
          Eigen::Map<const Matrix> buf_mat(buf[index], n1, n2);
          if (!shg_std) {
            results[index].block(bf1, bf2, n1, n2) = buf_mat;
          } else {
            for (std::size_t f1 = 0; f1 < n1; f1++) {
              for (std::size_t f2 = 0; f2 < n2; f2++) {
                results[index](bf1 + map1[f1], bf2 + map2[f2]) =
                    buf_mat(f1, f2);
              }
            }
          }
          // NO transpose-mirror into the {s2,s1} block here: the loop runs over
          // ALL shell pairs (both triangles are computed directly), and the
          // mirror is only valid when shells_1 == shells_2 - for two different
          // shell sets (time-overlap pairs) it overwrote the upper triangle
          // with wrong values (and raced between threads). Same convention as
          // the overlap kernel above, where the mirror is disabled too.
        }
      }
    }
  };
  // std::cout << "Done";
  //  Now for compute, do parallel
  parallel_do(compute, nthreads);
  if (null_pair.load() >= 0) {
    std::cout << "Error in compute_1body_ints_parallel_emultipole3: libint "
                 "returned a null result batch for shell pair ("
              << null_pair.load() / 1000000LL << ", "
              << null_pair.load() % 1000000LL
              << ") - 1-body engines must always produce a batch; the "
                 "engine or shell setup is corrupted\n";
    throw std::runtime_error(
        "libra C++ error - see the message printed above");
  }
  for (int index = 0; index < 20; index++) {
    for (auto i = 0u; i < n_1; ++i) {
      for (auto j = 0u; j < n_2; ++j) {
        res[index].set(i, j, results[index](i, j));
      }
    }
  }
  // Return the AO matrices as a vector of MATRIX
  return res;
}

std::vector<MATRIX>
compute_emultipole3(const std::vector<libint2::Shell> &shells_1,
                    const std::vector<libint2::Shell> &shells_2,
                    int number_of_threads) {

  // int nthreads;

  //  using libint2::Operator;
  //  using libint2::nthreads;

  auto nthreads_cstr = getenv("LIBINT_NUM_THREADS");
  int nthreads = number_of_threads;
  if (nthreads_cstr && strcmp(nthreads_cstr, "")) {
    std::istringstream iss(nthreads_cstr);
    iss >> nthreads;
    if (nthreads > 1 << 16 || nthreads <= 0)
      nthreads = 1;
  }

#if defined(_OPENMP)
  omp_set_num_threads(nthreads);
#endif
  //     ;; //std::cout << "Will scale over " << nthreads
  // #if defined(_OPENMP)
  //     ;;//          << " OpenMP"
  // #else
  //     ;;//          << " C++11"
  // #endif
  //     ;;//          << " threads" << std::endl;
  // Initialize Libint
  libint2::initialize();
  // Compute the AO overlap matrix
  std::vector<MATRIX> S =
      compute_1body_ints_parallel_emultipole3(shells_1, shells_2, nthreads);
  // auto S = compute_1body_ints_parallel_emultipole3(shells_1, shells_2,
  // nthreads); std::cout << "\n\tFinished computing overlap integral\n";
  //  End of AO matrix calculation
  libint2::finalize(); // done with libint

  return S;
}

} // namespace liblibint2_wrappers
} // namespace liblibra
