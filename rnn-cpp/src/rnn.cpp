#include "rnn.hpp"

#include "activations.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace {

Matrix scalar_input(double value) {
    Matrix input(1, 1);
    input(0, 0) = value;
    return input;
}

double squared_error(double predicted, double target) {
    const double error = predicted - target;
    return 0.5 * error * error;
}

void clip_all(std::vector<Matrix*>& matrices, double min_value, double max_value) {
    for (Matrix* matrix : matrices) {
        matrix->clip(min_value, max_value);
    }
}

} // namespace

RNN::RNN(std::size_t input_size, std::size_t hidden_size, std::size_t output_size, unsigned int seed)
    : input_size_(input_size),
      hidden_size_(hidden_size),
      output_size_(output_size),
      wxh_(hidden_size, input_size),
      whh_(hidden_size, hidden_size),
      why_(output_size, hidden_size),
      bh_(hidden_size, 1, 0.0),
      by_(output_size, 1, 0.0) {
    std::mt19937 rng(seed);
    wxh_ = Matrix::random(hidden_size_, input_size_, -0.1, 0.1, rng);
    whh_ = Matrix::random(hidden_size_, hidden_size_, -0.1, 0.1, rng);
    why_ = Matrix::random(output_size_, hidden_size_, -0.1, 0.1, rng);
}

PredictionResult RNN::predict(const std::vector<double>& inputs) const {
    Matrix hidden = Matrix::zeros(hidden_size_, 1);
    PredictionResult result;
    result.loss = 0.0;

    for (double value : inputs) {
        Matrix x = scalar_input(value);
        hidden = activations::tanh(matmul(wxh_, x) + matmul(whh_, hidden) + bh_);
        Matrix output = matmul(why_, hidden) + by_;
        result.predictions.push_back(output(0, 0));
    }

    return result;
}

double RNN::train_one(const TrainingExample& example, double learning_rate) {
    if (example.inputs.size() != example.targets.size()) {
        throw std::invalid_argument("training example inputs and targets must have equal length");
    }

    std::vector<Matrix> xs;
    std::vector<Matrix> hs;
    std::vector<Matrix> ys;
    xs.reserve(example.inputs.size());
    hs.reserve(example.inputs.size() + 1);
    ys.reserve(example.inputs.size());

    hs.push_back(Matrix::zeros(hidden_size_, 1));
    double loss = 0.0;

    for (std::size_t t = 0; t < example.inputs.size(); ++t) {
        xs.push_back(scalar_input(example.inputs[t]));
        hs.push_back(activations::tanh(matmul(wxh_, xs.back()) + matmul(whh_, hs.back()) + bh_));
        ys.push_back(matmul(why_, hs.back()) + by_);
        loss += squared_error(ys.back()(0, 0), example.targets[t]);
    }

    Matrix dwxh = Matrix::zeros(hidden_size_, input_size_);
    Matrix dwhh = Matrix::zeros(hidden_size_, hidden_size_);
    Matrix dwhy = Matrix::zeros(output_size_, hidden_size_);
    Matrix dbh = Matrix::zeros(hidden_size_, 1);
    Matrix dby = Matrix::zeros(output_size_, 1);
    Matrix dh_next = Matrix::zeros(hidden_size_, 1);

    for (std::size_t step = example.inputs.size(); step-- > 0;) {
        Matrix dy(output_size_, 1);
        dy(0, 0) = ys[step](0, 0) - example.targets[step];

        dwhy += matmul(dy, hs[step + 1].transpose());
        dby += dy;

        Matrix dh = matmul(why_.transpose(), dy) + dh_next;
        Matrix dh_raw = hadamard(dh, activations::tanh_derivative_from_output(hs[step + 1]));

        dbh += dh_raw;
        dwxh += matmul(dh_raw, xs[step].transpose());
        dwhh += matmul(dh_raw, hs[step].transpose());
        dh_next = matmul(whh_.transpose(), dh_raw);
    }

    std::vector<Matrix*> gradients = {&dwxh, &dwhh, &dwhy, &dbh, &dby};
    clip_all(gradients, -5.0, 5.0);

    wxh_ -= dwxh * learning_rate;
    whh_ -= dwhh * learning_rate;
    why_ -= dwhy * learning_rate;
    bh_ -= dbh * learning_rate;
    by_ -= dby * learning_rate;

    return loss / static_cast<double>(example.inputs.size());
}

double RNN::train(const std::vector<TrainingExample>& examples, int epochs, double learning_rate) {
    if (examples.empty()) {
        throw std::invalid_argument("training data is empty");
    }

    double average_loss = 0.0;
    for (int epoch = 1; epoch <= epochs; ++epoch) {
        double total_loss = 0.0;
        for (const TrainingExample& example : examples) {
            total_loss += train_one(example, learning_rate);
        }
        average_loss = total_loss / static_cast<double>(examples.size());

        if (epoch == 1 || epoch % 50 == 0 || epoch == epochs) {
            std::cout << "epoch " << epoch << " loss " << average_loss << '\n';
        }
    }
    return average_loss;
}

std::vector<TrainingExample> load_sequences(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("could not open data file: " + path);
    }

    std::vector<TrainingExample> examples;
    std::string line;

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::istringstream stream(line);
        std::vector<double> sequence;
        double value = 0.0;
        while (stream >> value) {
            sequence.push_back(value / 10.0);
        }

        if (sequence.size() < 2) {
            continue;
        }

        TrainingExample example;
        example.inputs.assign(sequence.begin(), sequence.end() - 1);
        example.targets.assign(sequence.begin() + 1, sequence.end());
        examples.push_back(example);
    }

    return examples;
}
