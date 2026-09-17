#ifndef RNN_CPP_ACTIVATIONS_HPP
#define RNN_CPP_ACTIVATIONS_HPP

#include "matrix.hpp"

namespace activations {

double tanh_scalar(double value);
double tanh_derivative_from_output(double tanh_output);
Matrix tanh(const Matrix& values);
Matrix tanh_derivative_from_output(const Matrix& tanh_outputs);

} // namespace activations

#endif
