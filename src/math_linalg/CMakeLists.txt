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
  # the probes RUN a 2x2 gemm through the int-argument prototype: a
  # symbol check alone accepts an ILP64 library, whose callee reads the
  # 32-bit dimensions as garbage (cmake/modules/LibraBlasProbe.cmake)
  INCLUDE(LibraBlasProbe)
  SET(LIBRA_GEMM_LIBS "")
  FIND_PACKAGE(BLAS QUIET)
  IF(BLAS_FOUND)
    libra_probe_zgemm("${BLAS_LIBRARIES}" LIBRA_HAVE_CBLAS_GEMM)
    IF(LIBRA_HAVE_CBLAS_GEMM)
      SET(LIBRA_GEMM_LIBS ${BLAS_LIBRARIES})
    ENDIF()
  ENDIF()
  # some distributions resolve FIND_PACKAGE(BLAS) to a CBLAS-less
  # reference library and ignore BLA_VENDOR (Gentoo's pkg-config-based
  # FindBLAS) - probe libopenblas directly before giving up
  IF(NOT LIBRA_GEMM_LIBS)
    FIND_LIBRARY(LIBRA_OPENBLAS_LIB NAMES openblas)
    IF(LIBRA_OPENBLAS_LIB)
      libra_probe_zgemm("${LIBRA_OPENBLAS_LIB}" LIBRA_HAVE_CBLAS_GEMM_OPENBLAS)
      IF(LIBRA_HAVE_CBLAS_GEMM_OPENBLAS)
        SET(LIBRA_GEMM_LIBS ${LIBRA_OPENBLAS_LIB})
      ENDIF()
    ENDIF()
  ENDIF()
  IF(LIBRA_GEMM_LIBS)
    MESSAGE(STATUS "math_linalg: CBLAS gemm backend enabled: ${LIBRA_GEMM_LIBS}")
    SET_SOURCE_FILES_PROPERTIES(${CMAKE_CURRENT_SOURCE_DIR}/CMATRIX.cpp ${CMAKE_CURRENT_SOURCE_DIR}/MATRIX.cpp
      PROPERTIES COMPILE_DEFINITIONS HAVE_CBLAS_GEMM)
    TARGET_LINK_LIBRARIES(linalg      ${LIBRA_GEMM_LIBS})
    TARGET_LINK_LIBRARIES(linalg_stat ${LIBRA_GEMM_LIBS})
  ELSE()
    MESSAGE(STATUS "math_linalg: no working int-ABI cblas_zgemm in BLAS or libopenblas - gemm backend disabled")
  ENDIF()
ENDIF()

                                                 
