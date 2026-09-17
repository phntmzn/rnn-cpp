#include "rnn.hpp"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    const std::string data_path = argc > 1 ? argv[1] : "data/sequences.txt";
    const int epochs = argc > 2 ? std::atoi(argv[2]) : 500;
    const double learning_rate = argc > 3 ? std::atof(argv[3]) : 0.01;

    try {
        std::vector<TrainingExample> examples = load_sequences(data_path);
        RNN model(1, 12, 1);

        std::cout << "loaded " << examples.size() << " sequences\n";
        std::cout << "training for " << epochs << " epochs at learning rate " << learning_rate << "\n\n";

        model.train(examples, epochs, learning_rate);

        std::vector<double> demo = {0.0, 0.1, 0.2, 0.3, 0.4};
        PredictionResult result = model.predict(demo);

        std::cout << "\npredictions for normalized sequence 0 1 2 3 4:\n";
        for (std::size_t i = 0; i < result.predictions.size(); ++i) {
            std::cout << "input " << std::fixed << std::setprecision(1) << demo[i] * 10.0
                      << " -> predicted next " << std::setprecision(3) << result.predictions[i] * 10.0
                      << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
