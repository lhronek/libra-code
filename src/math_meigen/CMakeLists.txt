#
#
#
#
#MESSAGE("Going into subdirectory Eigen/eigen-eigen-b9cd8366d4e8...")
#ADD_SUBDIRECTORY("Eigen/eigen-eigen-b9cd8366d4e8")

#MESSAGE("Going into subdirectory Eigen/eigen-eigen-b9cd8366d4e8...")
#ADD_SUBDIRECTORY("Eigen")


# Make the execution of this script conditional, depending on the Eigen library
# availablility
# In general, we need to call the find_Eigen module
# but for now, I just set the include dir manually
#
#set (EIGEN_INCLUDE /Eigen/eigen-eigen-b9cd8366d4e8/ )
#set (EIGEN_INCLUDE ./ )
#set(EIGEN_INCLUDE ${EIGEN3_INCLUDE_DIR} )

#
#  Source files and headers in this directory
#
file(GLOB EIG_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
file(GLOB EIG_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp ${EIG_HEADERS}) 


INCLUDE_DIRECTORIES(${EIGEN_INCLUDE_DIR})


#
#  Create both static and dynamic libraries
#
ADD_LIBRARY(meigen SHARED ${EIG_SRC})
ADD_LIBRARY(meigen_stat STATIC ${EIG_SRC})


if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.16 AND USING_PCH)  # Support for PCHs in CMake was added in 3.16
   message(STATUS "Compiling using pre-compiled header support: in math_meigen")
   target_precompile_headers(meigen REUSE_FROM pch) # With PUBLIC they will be used by targets using this target
   target_precompile_headers(meigen_stat REUSE_FROM pch_stat) # With PUBLIC they will be used by targets using this target
endif()




#
#  Link to external libraries
#
TARGET_LINK_LIBRARIES(meigen      linalg ${ext_libs})
TARGET_LINK_LIBRARIES(meigen_stat linalg_stat ${ext_libs})

#
#  Optional host-specific vectorization of the Eigen-based numerics: the
#  project-wide -O2 carries no SIMD flags, leaving Eigen's kernels at the
#  SSE2 baseline. Opt-in: the produced objects require AVX2+FMA hardware,
#  and FMA contraction shifts eigensolver results at machine epsilon.
#
OPTION(WITH_AVX2_MEIGEN "Compile math_meigen with -mavx2 -mfma (host must support AVX2/FMA)" OFF)
IF(WITH_AVX2_MEIGEN)
  INCLUDE(CheckCXXCompilerFlag)
  CHECK_CXX_COMPILER_FLAG("-mavx2" MEIGEN_CXX_HAS_AVX2)
  CHECK_CXX_COMPILER_FLAG("-mfma" MEIGEN_CXX_HAS_FMA)
  IF(MEIGEN_CXX_HAS_AVX2 AND MEIGEN_CXX_HAS_FMA)
    MESSAGE(STATUS "math_meigen: compiling with -mavx2 -mfma")
    TARGET_COMPILE_OPTIONS(meigen      PRIVATE -mavx2 -mfma)
    TARGET_COMPILE_OPTIONS(meigen_stat PRIVATE -mavx2 -mfma)
  ELSE()
    MESSAGE(STATUS "math_meigen: compiler lacks -mavx2/-mfma - option ignored")
  ENDIF()
ENDIF()


