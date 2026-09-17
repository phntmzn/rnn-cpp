#ifndef RNN_CPP_RNN_HPP
#define RNN_CPP_RNN_HPP

#include "matrix.hpp"

#include <random>
#include <string>
#include <vector>

struct TrainingExample {
    std::vector<double> inputs;
    std::vector<double> targets;
};

struct PredictionResult {
    std::vector<double> predictions;
    double loss;
};

class RNN {
public:
    RNN(std::size_t input_size, std::size_t hidden_size, std::size_t output_size, unsigned int seed = 7);

    PredictionResult predict(const std::vector<double>& inputs) const;
    double train_one(const TrainingExample& example, double learning_rate);
    double train(const std::vector<TrainingExample>& examples, int epochs, double learning_rate);

private:
    std::size_t input_size_;
    std::size_t hidden_size_;
    std::size_t output_size_;

    Matrix wxh_;
    Matrix whh_;
    Matrix why_;
    Matrix bh_;
    Matrix by_;
};

std::vector<TrainingExample> load_sequences(const std::string& path);

#endif
