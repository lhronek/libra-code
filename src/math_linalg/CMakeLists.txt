#
#  Source files and headers in this directory
#
file(GLOB LINALG_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
file(GLOB LINALG_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp ${LINALG_HEADERS}) 


#
#  Build both static and dynamic libraries
#
ADD_LIBRARY(linalg      SHARED ${LINALG_SRC})
ADD_LIBRARY(linalg_stat STATIC ${LINALG_SRC})


if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.16 AND USING_PCH)  # Support for PCHs in CMake was added in 3.16
   message(STATUS "Compiling using pre-compiled header support: in math_linalg")
   target_precompile_headers(linalg REUSE_FROM pch) # With PUBLIC they will be used by targets using this target
   target_precompile_headers(linalg_stat REUSE_FROM pch_stat) # With PUBLIC they will be used by targets using this target
endif()

#
#  Link to external libraries
#
TARGET_LINK_LIBRARIES(linalg      io ${ext_libs})
TARGET_LINK_LIBRARIES(linalg_stat io_stat ${ext_libs})

#
#  Optional CBLAS backend for large MATRIX/CMATRIX products
#
OPTION(WITH_BLAS_GEMM "Route large MATRIX/CMATRIX products through CBLAS gemm" ON)
IF(WITH_BLAS_GEMM)
  FIND_PACKAGE(BLAS QUIET)
  IF(BLAS_FOUND)
    INCLUDE(CheckFunctionExists)
    SET(CMAKE_REQUIRED_LIBRARIES ${BLAS_LIBRARIES})
    CHECK_FUNCTION_EXISTS(cblas_zgemm LIBRA_HAVE_CBLAS_GEMM)
    UNSET(CMAKE_REQUIRED_LIBRARIES)
    IF(LIBRA_HAVE_CBLAS_GEMM)
      MESSAGE(STATUS "math_linalg: CBLAS gemm backend enabled: ${BLAS_LIBRARIES}")
      SET_SOURCE_FILES_PROPERTIES(${CMAKE_CURRENT_SOURCE_DIR}/CMATRIX.cpp ${CMAKE_CURRENT_SOURCE_DIR}/MATRIX.cpp
        PROPERTIES COMPILE_DEFINITIONS HAVE_CBLAS_GEMM)
      TARGET_LINK_LIBRARIES(linalg      ${BLAS_LIBRARIES})
      TARGET_LINK_LIBRARIES(linalg_stat ${BLAS_LIBRARIES})
    ELSE()
      MESSAGE(STATUS "math_linalg: BLAS found but cblas_zgemm missing - gemm backend disabled")
    ENDIF()
  ELSE()
    MESSAGE(STATUS "math_linalg: no BLAS found - gemm backend disabled")
  ENDIF()
ENDIF()

                                                 
