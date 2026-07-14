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
  FIND_PACKAGE(BLAS QUIET)   # OpenBLAS/MKL bundle LAPACK
  IF(BLAS_FOUND)
    INCLUDE(CheckFunctionExists)
    SET(CMAKE_REQUIRED_LIBRARIES ${BLAS_LIBRARIES})
    CHECK_FUNCTION_EXISTS(zheevd_ LIBRA_HAVE_ZHEEVD)
    CHECK_FUNCTION_EXISTS(cblas_zgemm LIBRA_HAVE_ZGEMM_FOR_EXP)
    UNSET(CMAKE_REQUIRED_LIBRARIES)
    IF(LIBRA_HAVE_ZHEEVD AND LIBRA_HAVE_ZGEMM_FOR_EXP)
      MESSAGE(STATUS "math_specialfunctions: LAPACK zheevd exponential enabled")
      SET_SOURCE_FILES_PROPERTIES(${CMAKE_CURRENT_SOURCE_DIR}/SpecialFunctions.cpp
        PROPERTIES COMPILE_DEFINITIONS HAVE_LAPACK_ZHEEVD)
      TARGET_LINK_LIBRARIES(specialfunctions      ${BLAS_LIBRARIES})
      TARGET_LINK_LIBRARIES(specialfunctions_stat ${BLAS_LIBRARIES})
    ELSE()
      MESSAGE(STATUS "math_specialfunctions: zheevd/cblas_zgemm not found - Eigen path kept")
    ENDIF()
  ENDIF()
ENDIF()

                                                 
