# Set the compiler as described here: https://cmake.org/Wiki/CMake_FAQ#How_do_I_use_a_different_compiler.3F
#
#  Verify that Cmake is up to date
#
CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
set (CMAKE_CXX_STANDARD 14)
#SET(CMAKE_LEGACY_CYGWIN_WIN32 0)


#
#  Project name
#


project("libra")


#
#  User-defined Find modules
#
LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)


#
# Try using the PCH - it will only work if you have the right version of cmake
#
option(USING_PCH "Build using pre-compiled header support" ON)
option(WITH_OPENMP "Use OpenMP for parallelization" ON)


#
#  OpenMP
#
if(WITH_OPENMP)
  find_package(OpenMP REQUIRED)
endif()


#
#  Python library
#
MESSAGE(STATUS "Looking for Python libraries...")
FIND_PACKAGE(Python3 3.6 REQUIRED COMPONENTS Interpreter Development)
#FIND_PACKAGE(Python3 3.6 REQUIRED COMPONENTS Development)

#IF(Python3_FOUND)
#  MESSAGE("Success!")
#  INCLUDE_DIRECTORIES("${PYTHON_INCLUDE_DIRS}")
IF(NOT Python3_FOUND)
  MESSAGE(STATUS "Unable to find correct Python version. If your Python include/libraries are installed in a non-standard location")
  MESSAGE(STATUS "Try using -DPython3_ROOT_DIR=\${path_to_the_python_root_dir}")
  MESSAGE(FATAL_ERROR)
ENDIF()

#if(NOT Python3_EXECUTABLE MATCHES "conda")
#  message(FATAL_ERROR "System Python detected. Use Conda Python.")
#endif()

message(STATUS "Python executable: ${Python3_EXECUTABLE}")

#if(Python3_EXECUTABLE MATCHES "^/usr/")
#    message(FATAL_ERROR
#        "System Python detected (${Python3_EXECUTABLE}). "
#        "Please activate a Conda environment.")
#endif()

if(DEFINED ENV{CONDA_PREFIX})
    file(REAL_PATH "$ENV{CONDA_PREFIX}" CONDA_PREFIX_REAL)
    file(REAL_PATH "${Python3_EXECUTABLE}" PYTHON_REAL)

    string(FIND "${PYTHON_REAL}" "${CONDA_PREFIX_REAL}" _pos)

    if(_pos EQUAL -1)
        message(FATAL_ERROR
            "Python executable (${PYTHON_REAL}) is not inside the active "
            "Conda environment (${CONDA_PREFIX_REAL}).")
    endif()
endif()



MESSAGE(STATUS "Found Python libraries directory: ${Python3_LIBRARIES}")
MESSAGE(STATUS "Found Python include directory:   ${Python3_INCLUDE_DIRS}")
SET(LIBRA_Python3_VERSION "${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR}")




#
#  Boost library
#

MESSAGE(STATUS "Looking for Boost libraries...")
SET(Boost_NO_BOOST_CMAKE TRUE) # to fix the problem of finding python37 libs
SET(Boost_USE_STATIC_LIBS=OFF)
SET(Boost_USE_MULTITHREADED=OFF)
SET(Boost_USE_STATIC_RUNTIME=OFF)

FIND_PACKAGE(Boost 1.82.0 REQUIRED COMPONENTS regex python${LIBRA_Python3_VERSION})
IF(Boost_FOUND)
  INCLUDE_DIRECTORIES("${Boost_INCLUDE_DIRS}")
ENDIF()

MESSAGE(STATUS "Found Boost libraries directory: ${Boost_LIBRARIES}")
MESSAGE(STATUS "Found Boost include directory:   ${Boost_INCLUDE_DIRS}")


MESSAGE(STATUS "Looking for Eigen3 library...")
FIND_PACKAGE(Eigen3)
IF(EIGEN3_FOUND)
  INCLUDE_DIRECTORIES("${EIGEN3_INCLUDE_DIR}")
  SET(EIGEN3_INCLUDE_DIRS "${EIGEN3_INCLUDE_DIR}")
  MESSAGE(STATUS "Found Eigen3 include directory: ${EIGEN3_INCLUDE_DIR}")
ELSEIF(NOT EIGEN3_FOUND)
  MESSAGE(FATAL_ERROR "Unable to find Eigen3")
ENDIF()


MESSAGE(STATUS "Looking for libint2 libraries...")
FIND_PACKAGE(Libint2 CONFIG 2.7.1 REQUIRED COMPONENTS shared gss impure_sh onebody_d0_l6 g12_d0_l4 g12_d1_l4 eri_c4_d0_l5 eri_c4_d1_l4) #gss e5 g5)
if(Libint2_FOUND)
  MESSAGE(STATUS "Found libint2 include directory: ${LIBINT2_INCLUDE_DIRS}")
  INCLUDE_DIRECTORIES(Libint2::int2 Libint2::cxx Libint2::int2-cxx Libint2::impure_sh Libint2::gss Libint2::onebody_d0_l6 Libint::g12_d0_l4 Libint2::g12_d1_l4 Libint2::shared Libint2::cxx_ho Libint2::c  Libint2::eri_c4_d0_l5  Libint2::eri_c4_d1_l4)
endif()

message(STATUS "Eigen3_DIR = ${Eigen3_DIR}")
message(STATUS "EIGEN3_INCLUDE_DIR = ${EIGEN3_INCLUDE_DIR}")
message(STATUS "EIGEN3_INCLUDE_DIRS = ${EIGEN3_INCLUDE_DIRS}")
message(STATUS "Eigen3_VERSION = ${Eigen3_VERSION}")


#
# GNU compiler definitions
#
IF(CMAKE_COMPILER_IS_GNUCXX)
# For Linux
  ADD_DEFINITIONS("-Wall -Wl,-z,defs")
# For Cygwin
#  ADD_DEFINITIONS("-Wall -DCYGWIN")
ELSE()
  MESSAGE(FATAL_ERROR "CMakeLists.txt has not been tested/written for your compiler.")
ENDIF()



#
#  Cmake configuration
#
MESSAGE(STATUS "Setting up the CMake configuration...")
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE "RELEASE")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
ENDIF()
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

# Optional host-specific vectorization. PROJECT-WIDE by construction: the
# Eigen templates instantiated in math_meigen, libint2_wrappers and solvers
# are folded into one liblibra_core.so, and Eigen's alignment/kernel choice
# depends on __AVX__ (EIGEN_IDEAL_MAX_ALIGN_BYTES 16 -> 32) - every TU must
# agree. Opt-in: the objects require AVX2+FMA hardware, and FMA contraction
# shifts eigensolver results at machine epsilon.
OPTION(WITH_AVX2 "Compile the whole project with -mavx2 -mfma (host must support AVX2/FMA)" OFF)
# WITH_AVX2_MEIGEN was the earlier per-target spelling - honor it as an alias
IF(DEFINED WITH_AVX2_MEIGEN AND WITH_AVX2_MEIGEN)
  SET(WITH_AVX2 ON CACHE BOOL "Compile the whole project with -mavx2 -mfma (host must support AVX2/FMA)" FORCE)
ENDIF()
SET(LIBRA_HAVE_AVX2 "" CACHE INTERNAL "project compiled with -mavx2 -mfma")
IF(WITH_AVX2)
  INCLUDE(CheckCXXCompilerFlag)
  CHECK_CXX_COMPILER_FLAG("-mavx2" LIBRA_CXX_HAS_AVX2)
  CHECK_CXX_COMPILER_FLAG("-mfma" LIBRA_CXX_HAS_FMA)
  IF(LIBRA_CXX_HAS_AVX2 AND LIBRA_CXX_HAS_FMA)
    MESSAGE(STATUS "compiling the project with -mavx2 -mfma")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma")
    SET(LIBRA_HAVE_AVX2 1 CACHE INTERNAL "project compiled with -mavx2 -mfma")
  ELSE()
    MESSAGE(STATUS "compiler lacks -mavx2/-mfma - WITH_AVX2 ignored")
  ENDIF()
ENDIF()



#
# Set the libraries
#
if(NOT WITH_OPENMP)
  SET(ext_libs Boost::regex Boost::python${LIBRA_Python3_VERSION} Python3::Python)
else()
  SET(ext_libs Boost::regex Boost::python${LIBRA_Python3_VERSION} Python3::Python OpenMP::OpenMP_C OpenMP::OpenMP_CXX)
endif()



#
# Now building the project
#
MESSAGE("Going into subdirectory src...")
#
#  Build-time version stamping for the import banner
#
set(LIBRA_VERSION_STRING "" CACHE STRING "Override the composed Libra version string verbatim")
# the development branch carries no release number - releases stamp one
# through LIBRA_VERSION_STRING, or set a fixed base here:
#set(LIBRA_VERSION_BASE "5.13.0")
set(LIBRA_VERSION_BASE "devel")
# the flag string is read at BUILD time through the target property
# LIBRA_FLAGS_STR, composed after ADD_SUBDIRECTORY(src) from the backend
# probe results - the banner records what was linked, not what was asked
add_custom_target(libra_version_h ALL
  COMMAND ${CMAKE_COMMAND}
    -DSRC_DIR=${CMAKE_SOURCE_DIR}
    -DOUT_FILE=${CMAKE_BINARY_DIR}/generated/libra_version.h
    -DBASE_VERSION=${LIBRA_VERSION_BASE}
    "-DOVERRIDE=${LIBRA_VERSION_STRING}"
    "-DLIBINT_VER=${Libint2_VERSION}"
    "-DFLAGS_STR=$<TARGET_PROPERTY:libra_version_h,LIBRA_FLAGS_STR>"
    -P ${CMAKE_SOURCE_DIR}/cmake/gen_version.cmake
  COMMENT "Stamping libra_version.h"
  BYPRODUCTS ${CMAKE_BINARY_DIR}/generated/libra_version.h)
INCLUDE_DIRECTORIES(${CMAKE_BINARY_DIR}/generated)

ADD_SUBDIRECTORY("src")

# provenance flags of the LINKED build: each numerics-changing backend
# appears iff its subdirectory probe passed (LIBRA_HAVE_* cache flags)
set(LIBRA_FLAGS "libint boost_python")
if(WITH_OPENMP)
  set(LIBRA_FLAGS "omp ${LIBRA_FLAGS}")
endif()
if(LIBRA_HAVE_GEMM_BACKEND)
  set(LIBRA_FLAGS "${LIBRA_FLAGS} blas_gemm")
endif()
if(LIBRA_HAVE_ZHEEVD_BACKEND)
  set(LIBRA_FLAGS "${LIBRA_FLAGS} lapack_zheevd")
endif()
if(LIBRA_HAVE_AVX2)
  set(LIBRA_FLAGS "${LIBRA_FLAGS} avx2")
endif()
if(USING_PCH)
  set(LIBRA_FLAGS "${LIBRA_FLAGS} pch")
endif()
set_property(TARGET libra_version_h PROPERTY LIBRA_FLAGS_STR "${LIBRA_FLAGS}")


