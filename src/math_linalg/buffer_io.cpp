/**
  Bulk transfer between libra matrices and Python buffer objects (buffer_io.h).
*/
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <string>
#include <complex>
#include "buffer_io.h"

/// liblibra namespace
namespace liblibra{

/// liblinalg namespace
namespace liblinalg{

namespace {

void fail(const char* fn, const std::string& what){
  std::cout<<"Error in liblinalg::"<<fn<<" : "<<what<<"\n"; std::cout << std::flush;
  throw std::runtime_error("libra C++ error - see the message printed above");
}

template <typename T> const char* expected_format();
template <> const char* expected_format<double>(){ return "d"; }
template <> const char* expected_format<std::complex<double> >(){ return "Zd"; }
template <> const char* expected_format<int>(){ return "i"; }

// the struct-module format of the buffer, with the native byte-order prefixes accepted (the
// hosts libra runs on are little-endian, so "<" is native too) - a buffer without a format
// (NULL) is judged by its item size alone
bool format_matches(const char* fmt, const char* expected){
  if(fmt == NULL){ return true; }
  if(*fmt == '@' || *fmt == '=' || *fmt == '<'){ fmt++; }
  return std::strcmp(fmt, expected) == 0;
}

template <typename T, typename M>
void checked_view(const char* fn, const M& mat, boost::python::object obj, int flags, Py_buffer& view){
  if(PyObject_GetBuffer(obj.ptr(), &view, flags | PyBUF_FORMAT) != 0){
    PyErr_Clear();
    fail(fn, std::string("the object exposes no ") + ((flags & PyBUF_WRITABLE) ? "writable " : "") +
             "C-contiguous buffer - pass a C-contiguous numpy array of the matrix's element type");
  }
  std::size_t need = sizeof(T) * (std::size_t)mat.n_rows * (std::size_t)mat.n_cols;
  // the shape as well as the byte count: a same-size array of another shape (a transpose, a flat vector) would
  // be copied in the wrong layout, which symmetric operands hide
  bool shaped = (view.ndim == 2) && (view.shape != NULL) &&
                (view.shape[0] == (Py_ssize_t)mat.n_rows) && (view.shape[1] == (Py_ssize_t)mat.n_cols);
  bool ok = (view.itemsize == (Py_ssize_t)sizeof(T)) && format_matches(view.format, expected_format<T>()) &&
            ((std::size_t)view.len == need) && shaped;
  if(!ok){
    std::ostringstream o;
    o<<"the buffer does not match the "<<mat.n_rows<<" x "<<mat.n_cols<<" matrix of "<<sizeof(T)<<"-byte '"<<expected_format<T>()
     <<"' elements ("<<need<<" bytes): got itemsize "<<view.itemsize<<", format '"<<(view.format ? view.format : "")<<"', "<<view.len
     <<" bytes, shape (";
    for(int d = 0; d < view.ndim; d++){ o<<(d ? ", " : "")<<(view.shape ? view.shape[d] : -1); }
    o<<")";
    PyBuffer_Release(&view);
    fail(fn, o.str());
  }
}

template <typename T, typename M>
void set_from_buffer_impl(M& mat, boost::python::object obj){
  Py_buffer view;
  checked_view<T, M>("set_from_buffer", mat, obj, PyBUF_CONTIG_RO, view);
  std::memcpy(mat.M, view.buf, (std::size_t)view.len);
  PyBuffer_Release(&view);
}

template <typename T, typename M>
void copy_to_buffer_impl(M& mat, boost::python::object obj){
  Py_buffer view;
  checked_view<T, M>("copy_to_buffer", mat, obj, PyBUF_CONTIG, view);
  std::memcpy(view.buf, mat.M, (std::size_t)view.len);
  PyBuffer_Release(&view);
}

}// anonymous namespace


void set_from_buffer(MATRIX& m, boost::python::object buf){ set_from_buffer_impl<double, MATRIX>(m, buf); }
void set_from_buffer(CMATRIX& m, boost::python::object buf){ set_from_buffer_impl<std::complex<double>, CMATRIX>(m, buf); }
void set_from_buffer(IMATRIX& m, boost::python::object buf){ set_from_buffer_impl<int, IMATRIX>(m, buf); }
void copy_to_buffer(MATRIX& m, boost::python::object buf){ copy_to_buffer_impl<double, MATRIX>(m, buf); }
void copy_to_buffer(CMATRIX& m, boost::python::object buf){ copy_to_buffer_impl<std::complex<double>, CMATRIX>(m, buf); }
void copy_to_buffer(IMATRIX& m, boost::python::object buf){ copy_to_buffer_impl<int, IMATRIX>(m, buf); }


}// namespace liblinalg
}// namespace liblibra
