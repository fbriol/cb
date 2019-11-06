#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <sstream>
#include "sparse.hpp"

namespace py = pybind11;

template <typename Array>
void check_array_ndim(const std::string& name, const int64_t ndim,
                      const Array& a) {
  if (a.ndim() != ndim) {
    throw std::invalid_argument(name + " must be a " + std::to_string(ndim) +
                                "-dimensional array");
  }
}

template <typename Array, typename... Args>
void check_array_ndim(const std::string& name, const int64_t ndim,
                      const Array& a, Args... args) {
  static_assert(sizeof...(Args) % 3 == 0,
                "number of parameters is expected to be a multiple of 3");
  check_array_ndim(name, ndim, a);
  check_array_ndim(args...);
}

template <typename Array>
auto ndarray_shape(const Array& array) -> std::string {
  std::stringstream ss;
  ss << "(";
  for (auto ix = 0; ix < array.ndim(); ++ix) {
    ss << array.shape(ix) << ", ";
  }
  ss << ")";
  return ss.str();
}

template <typename Array1, typename Array2>
void check_ndarray_shape(const std::string& name1, const Array1& a1,
                         const std::string& name2, const Array2& a2) {
  auto match = a1.ndim() == a2.ndim();
  if (match) {
    for (auto ix = 0; ix < a1.ndim(); ++ix) {
      if (a1.shape(ix) != a2.shape(ix)) {
        match = false;
        break;
      }
    }
  }
  if (!match) {
    throw std::invalid_argument(name1 + ", " + name2 +
                                " could not be broadcast together with shape " +
                                ndarray_shape(a1) + "  " + ndarray_shape(a2));
  }
}

template <typename Array1, typename Array2, typename... Args>
void check_ndarray_shape(const std::string& name1, const Array1& a1,
                         const std::string& name2, const Array2& a2,
                         Args... args) {
  static_assert(sizeof...(Args) % 2 == 0,
                "an even number of parameters is expected");
  check_ndarray_shape(name1, a1, name2, a2);
  check_ndarray_shape(name1, a1, args...);
}

PYBIND11_MODULE(core, m) {
  py::class_<Matrix>(m, "Matrix")
      .def(py::init<>())
      .def("shape", &Matrix::shape)
      .def("get", &Matrix::get, py::arg("key"))
      .def(
          "set",
          [](Matrix& self, py::array_t<uint32_t> i, py::array_t<uint32_t> j,
             py::array_t<double> x) {
            check_array_ndim("i", 1, i, "j", 1, j, "x", 1, x);
            check_ndarray_shape("i", i, "j", j, "x", x);
            auto _i = i.unchecked<1>();
            auto _j = j.unchecked<1>();
            auto _x = x.unchecked<1>();

            for (auto ix = 0; ix < x.size(); ++ix) {
              std::cout << _x[ix] << std::endl;
              self.set({_i[ix], _j[ix]}, _x[ix]);
            }
          },
          py::arg("i"), py::arg("j"), py::arg("x"))
      .def("__setitem__",
           [](Matrix& self, const py::tuple& slices,
              py::array_t<double>& x) -> void {
             if (slices.size() != 2) {
               throw std::invalid_argument(
                   "number of indices must be equal to 2");
             }
             size_t i_start, i_stop, i_step, i_slicelength;
             size_t j_start, j_stop, j_step, j_slicelength;
             auto shape = self.shape();

             if (!slices[0].cast<py::slice>().compute(
                     std::get<0>(shape), &i_start, &i_stop, &i_step,
                     &i_slicelength)) {
               throw py::error_already_set();
             }

             if (!slices[1].cast<py::slice>().compute(
                     std::get<1>(shape), &j_start, &j_stop, &j_step,
                     &j_slicelength)) {
               throw py::error_already_set();
             }

             if (x.ndim() != 2 ||
                 static_cast<size_t>(x.shape(0)) != i_slicelength ||
                 static_cast<size_t>(x.shape(1)) != j_slicelength) {
               throw std::runtime_error(
                   "could not broadcast input array from shape " +
                   ndarray_shape(x) + " into shape (" +
                   std::to_string(i_slicelength) + ", " +
                   std::to_string(j_slicelength) + ")");
             }

             auto _x = x.unchecked<2>();

             for (size_t ix = 0; ix < i_slicelength; ++ix) {
               auto start = j_start;
               for (size_t jx = 0; jx < j_slicelength; ++jx) {
                 self.set({i_start, j_start}, _x(ix, jx));
                 start += j_step;
               }
               i_start += i_step;
             }
           })
      .def("__getitem__",
           [](const Matrix& self,
              const py::tuple& slices) -> py::array_t<double> {
             if (slices.size() != 2) {
               throw std::invalid_argument(
                   "number of indices must be equal to 2");
             }
             size_t i_start, i_stop, i_step, i_slicelength;
             size_t j_start, j_stop, j_step, j_slicelength;
             auto shape = self.shape();

             if (!slices[0].cast<py::slice>().compute(
                     std::get<0>(shape), &i_start, &i_stop, &i_step,
                     &i_slicelength)) {
               throw py::error_already_set();
             }

             if (!slices[1].cast<py::slice>().compute(
                     std::get<1>(shape), &j_start, &j_stop, &j_step,
                     &j_slicelength)) {
               throw py::error_already_set();
             }
             auto x = py::array_t<double>({i_slicelength, j_slicelength});
             auto _x = x.mutable_unchecked<2>();

             for (size_t ix = 0; ix < i_slicelength; ++ix) {
               auto start = j_start;
               for (size_t jx = 0; jx < j_slicelength; ++jx) {
                 _x(ix, jx) = self.get({i_start, start});
                 start += j_step;
               }
               i_start += i_step;
             }
             return x;
           });
}