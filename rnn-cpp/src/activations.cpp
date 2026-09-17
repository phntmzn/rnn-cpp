#include "activations.hpp"

#include <cmath>

namespace activations {

double tanh_scalar(double value) {
    return std::tanh(value);
}

double tanh_derivative_from_output(double tanh_output) {
    return 1.0 - tanh_output * tanh_output;
}

Matrix tanh(const Matrix& values) {
    return values.apply(tanh_scalar);
}

Matrix tanh_derivative_from_output(const Matrix& tanh_outputs) {
    return tanh_outputs.apply([](double value) {
        return tanh_derivative_from_output(value);
    });
}

} // namespace activations
