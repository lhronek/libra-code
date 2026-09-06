#ifndef BUFFER_IO_H
#define BUFFER_IO_H
/**
  Bulk transfer between libra matrices and Python buffer objects through the
  buffer protocol: a C-contiguous numpy array of the matrix's element type
  (float64 for MATRIX, complex128 for CMATRIX, int32 for IMATRIX) holding
  exactly n_rows x n_cols elements is copied in one memcpy, in either direction.
  The item size, the format and the total length are checked, so a wrong dtype
  or shape throws instead of reading garbage. libra_py.data_conv builds its
  numpy conversions on these - the element-wise set/get loops took 0.8 s per
  1000 x 1000 matrix each way.
*/
#if defined(USING_PCH)
#include "../pch.h"
#else
#include <boost/python.hpp>
#endif
#include "IMATRIX.h"
#include "MATRIX.h"
#include "CMATRIX.h"

namespace liblibra{
namespace liblinalg{

void set_from_buffer(MATRIX& m, boost::python::object buf);     ///< m <- a C-contiguous float64 buffer of n_rows x n_cols elements
void set_from_buffer(CMATRIX& m, boost::python::object buf);    ///< complex128
void set_from_buffer(IMATRIX& m, boost::python::object buf);    ///< int32
void copy_to_buffer(MATRIX& m, boost::python::object buf);      ///< a writable C-contiguous float64 buffer of n_rows x n_cols elements <- m
void copy_to_buffer(CMATRIX& m, boost::python::object buf);     ///< complex128
void copy_to_buffer(IMATRIX& m, boost::python::object buf);     ///< int32

}// namespace liblinalg
}// namespace liblibra

#endif
