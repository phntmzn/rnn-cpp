#include "matrix.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <stdexcept>

Matrix::Matrix() : rows_(0), cols_(0) {}

Matrix::Matrix(std::size_t rows, std::size_t cols, double value)
    : rows_(rows), cols_(cols), data_(rows * cols, value) {}

Matrix Matrix::zeros(std::size_t rows, std::size_t cols) {
    return Matrix(rows, cols, 0.0);
}

Matrix Matrix::random(std::size_t rows, std::size_t cols, double min, double max, std::mt19937& rng) {
    Matrix result(rows, cols);
    std::uniform_real_distribution<double> dist(min, max);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            result(row, col) = dist(rng);
        }
    }
    return result;
}

std::size_t Matrix::rows() const {
    return rows_;
}

std::size_t Matrix::cols() const {
    return cols_;
}

double& Matrix::operator()(std::size_t row, std::size_t col) {
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("matrix index out of range");
    }
    return data_[row * cols_ + col];
}

double Matrix::operator()(std::size_t row, std::size_t col) const {
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("matrix index out of range");
    }
    return data_[row * cols_ + col];
}

Matrix Matrix::transpose() const {
    Matrix result(cols_, rows_);
    for (std::size_t row = 0; row < rows_; ++row) {
        for (std::size_t col = 0; col < cols_; ++col) {
            result(col, row) = (*this)(row, col);
        }
    }
    return result;
}

Matrix Matrix::apply(const std::function<double(double)>& fn) const {
    Matrix result(rows_, cols_);
    for (std::size_t row = 0; row < rows_; ++row) {
        for (std::size_t col = 0; col < cols_; ++col) {
            result(row, col) = fn((*this)(row, col));
        }
    }
    return result;
}

double Matrix::sum() const {
    double total = 0.0;
    for (double value : data_) {
        total += value;
    }
    return total;
}

void Matrix::fill(double value) {
    std::fill(data_.begin(), data_.end(), value);
}

void Matrix::clip(double min_value, double max_value) {
    for (double& value : data_) {
        value = std::max(min_value, std::min(max_value, value));
    }
}

Matrix& Matrix::operator+=(const Matrix& other) {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("matrix dimensions must match for addition");
    }
    for (std::size_t index = 0; index < data_.size(); ++index) {
        data_[index] += other.data_[index];
    }
    return *this;
}

Matrix& Matrix::operator-=(const Matrix& other) {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("matrix dimensions must match for subtraction");
    }
    for (std::size_t index = 0; index < data_.size(); ++index) {
        data_[index] -= other.data_[index];
    }
    return *this;
}

Matrix& Matrix::operator*=(double scalar) {
    for (double& value : data_) {
        value *= scalar;
    }
    return *this;
}

Matrix operator+(Matrix lhs, const Matrix& rhs) {
    lhs += rhs;
    return lhs;
}

Matrix operator-(Matrix lhs, const Matrix& rhs) {
    lhs -= rhs;
    return lhs;
}

Matrix operator*(Matrix lhs, double scalar) {
    lhs *= scalar;
    return lhs;
}

Matrix operator*(double scalar, Matrix rhs) {
    rhs *= scalar;
    return rhs;
}

Matrix matmul(const Matrix& lhs, const Matrix& rhs) {
    if (lhs.cols() != rhs.rows()) {
        throw std::invalid_argument("inner matrix dimensions must match");
    }
    Matrix result(lhs.rows(), rhs.cols(), 0.0);
    for (std::size_t row = 0; row < lhs.rows(); ++row) {
        for (std::size_t col = 0; col < rhs.cols(); ++col) {
            double total = 0.0;
            for (std::size_t k = 0; k < lhs.cols(); ++k) {
                total += lhs(row, k) * rhs(k, col);
            }
            result(row, col) = total;
        }
    }
    return result;
}

Matrix hadamard(const Matrix& lhs, const Matrix& rhs) {
    if (lhs.rows() != rhs.rows() || lhs.cols() != rhs.cols()) {
        throw std::invalid_argument("matrix dimensions must match for hadamard product");
    }
    Matrix result(lhs.rows(), lhs.cols());
    for (std::size_t row = 0; row < lhs.rows(); ++row) {
        for (std::size_t col = 0; col < lhs.cols(); ++col) {
            result(row, col) = lhs(row, col) * rhs(row, col);
        }
    }
    return result;
}

std::ostream& operator<<(std::ostream& os, const Matrix& matrix) {
    for (std::size_t row = 0; row < matrix.rows(); ++row) {
        for (std::size_t col = 0; col < matrix.cols(); ++col) {
            os << std::setw(10) << matrix(row, col);
        }
        os << '\n';
    }
    return os;
}
