#
#  Source files and headers in this directory
#
file(GLOB SPFUNC_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
file(GLOB SPFUNC_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp ${SPFUNC_HEADERS}) 


#
#  Build both static and dynamic libraries
#
ADD_LIBRARY(specialfunctions      SHARED ${SPFUNC_SRC})
ADD_LIBRARY(specialfunctions_stat STATIC ${SPFUNC_SRC})


if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.16 AND USING_PCH)  # Support for PCHs in CMake was added in 3.16
   message(STATUS "Compiling using pre-compiled header support: in math_specialfunctions")
   target_precompile_headers(specialfunctions REUSE_FROM pch) # With PUBLIC they will be used by targets using this target
   target_precompile_headers(specialfunctions_stat REUSE_FROM pch_stat) # With PUBLIC they will be used by targets using this target
endif()



#
#  Link to external libraries
#
TARGET_LINK_LIBRARIES(specialfunctions      linalg meigen random ${ext_libs})
TARGET_LINK_LIBRARIES(specialfunctions_stat linalg_stat meigen_stat random_stat ${ext_libs})

#
#  Optional LAPACK divide-and-conquer eigensolver for the matrix exponential
#
OPTION(WITH_LAPACK_ZHEEVD "Use LAPACK zheevd for the matrix exponential eigensolve" ON)
IF(WITH_LAPACK_ZHEEVD)
  INCLUDE(CheckFunctionExists)
  SET(LIBRA_ZHEEVD_LIBS "")
  FIND_PACKAGE(BLAS QUIET)   # OpenBLAS/MKL bundle LAPACK and CBLAS
  IF(BLAS_FOUND)
    SET(CMAKE_REQUIRED_LIBRARIES ${BLAS_LIBRARIES})
    CHECK_FUNCTION_EXISTS(zheevd_ LIBRA_HAVE_ZHEEVD)
    CHECK_FUNCTION_EXISTS(cblas_zgemm LIBRA_HAVE_ZGEMM_FOR_EXP)
    UNSET(CMAKE_REQUIRED_LIBRARIES)
    IF(LIBRA_HAVE_ZHEEVD AND LIBRA_HAVE_ZGEMM_FOR_EXP)
      SET(LIBRA_ZHEEVD_LIBS ${BLAS_LIBRARIES})
    ENDIF()
  ENDIF()
  # some distributions resolve FIND_PACKAGE(BLAS) to a reference
  # library without CBLAS/LAPACK and ignore BLA_VENDOR (Gentoo's
  # pkg-config-based FindBLAS) - probe libopenblas directly
  IF(NOT LIBRA_ZHEEVD_LIBS)
    FIND_LIBRARY(LIBRA_OPENBLAS_LIB NAMES openblas)
    IF(LIBRA_OPENBLAS_LIB)
      SET(CMAKE_REQUIRED_LIBRARIES ${LIBRA_OPENBLAS_LIB})
      CHECK_FUNCTION_EXISTS(zheevd_ LIBRA_HAVE_ZHEEVD_OPENBLAS)
      CHECK_FUNCTION_EXISTS(cblas_zgemm LIBRA_HAVE_ZGEMM_FOR_EXP_OPENBLAS)
      UNSET(CMAKE_REQUIRED_LIBRARIES)
      IF(LIBRA_HAVE_ZHEEVD_OPENBLAS AND LIBRA_HAVE_ZGEMM_FOR_EXP_OPENBLAS)
        SET(LIBRA_ZHEEVD_LIBS ${LIBRA_OPENBLAS_LIB})
      ENDIF()
    ENDIF()
  ENDIF()
  IF(LIBRA_ZHEEVD_LIBS)
    MESSAGE(STATUS "math_specialfunctions: LAPACK zheevd exponential enabled: ${LIBRA_ZHEEVD_LIBS}")
    SET_SOURCE_FILES_PROPERTIES(${CMAKE_CURRENT_SOURCE_DIR}/SpecialFunctions.cpp
      PROPERTIES COMPILE_DEFINITIONS HAVE_LAPACK_ZHEEVD)
    TARGET_LINK_LIBRARIES(specialfunctions      ${LIBRA_ZHEEVD_LIBS})
    TARGET_LINK_LIBRARIES(specialfunctions_stat ${LIBRA_ZHEEVD_LIBS})
  ELSE()
    MESSAGE(STATUS "math_specialfunctions: zheevd/cblas_zgemm not found in BLAS or libopenblas - Eigen path kept")
  ENDIF()
ENDIF()

                                                 
