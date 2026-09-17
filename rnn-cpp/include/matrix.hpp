#ifndef RNN_CPP_MATRIX_HPP
#define RNN_CPP_MATRIX_HPP

#include <cstddef>
#include <functional>
#include <iosfwd>
#include <random>
#include <vector>

class Matrix {
public:
    Matrix();
    Matrix(std::size_t rows, std::size_t cols, double value = 0.0);

    static Matrix zeros(std::size_t rows, std::size_t cols);
    static Matrix random(std::size_t rows, std::size_t cols, double min, double max, std::mt19937& rng);

    std::size_t rows() const;
    std::size_t cols() const;

    double& operator()(std::size_t row, std::size_t col);
    double operator()(std::size_t row, std::size_t col) const;

    Matrix transpose() const;
    Matrix apply(const std::function<double(double)>& fn) const;
    double sum() const;
    void fill(double value);
    void clip(double min_value, double max_value);

    Matrix& operator+=(const Matrix& other);
    Matrix& operator-=(const Matrix& other);
    Matrix& operator*=(double scalar);

private:
    std::size_t rows_;
    std::size_t cols_;
    std::vector<double> data_;
};

Matrix operator+(Matrix lhs, const Matrix& rhs);
Matrix operator-(Matrix lhs, const Matrix& rhs);
Matrix operator*(Matrix lhs, double scalar);
Matrix operator*(double scalar, Matrix rhs);
Matrix matmul(const Matrix& lhs, const Matrix& rhs);
Matrix hadamard(const Matrix& lhs, const Matrix& rhs);
std::ostream& operator<<(std::ostream& os, const Matrix& matrix);

#endif
