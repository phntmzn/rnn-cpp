# rnn-cpp

`rnn-cpp` is a compact recurrent neural network written from scratch in modern C++.
It is meant for learning: no external ML framework, no hidden tensor library, just
matrix operations, activation functions, and a vanilla recurrent cell.

The demo trains on short numeric sequences in `data/sequences.txt`. Each line is a
sequence of numbers. The model learns to predict the next value at each time step.

## Project Layout

```text
rnn-cpp/
├── CMakeLists.txt
├── README.md
├── include/
│   ├── rnn.hpp
│   ├── matrix.hpp
│   └── activations.hpp
├── src/
│   ├── main.cpp
│   ├── rnn.cpp
│   ├── matrix.cpp
│   └── activations.cpp
├── data/
│   └── sequences.txt
├── models/
├── outputs/
└── build/
```

## Build

```sh
cmake -S . -B build
cmake --build build
```

## Run

```sh
./build/rnn_demo
```

Optional arguments:

```sh
./build/rnn_demo data/sequences.txt 500 0.01
```

Arguments are:

1. training data path
2. number of epochs
3. learning rate

## Data Format

Each line in `data/sequences.txt` is one numeric sequence:

```text
0 1 2 3 4 5
2 4 6 8 10 12
```

For a sequence `0 1 2 3`, the network receives `0`, `1`, and `2`, then learns to
predict `1`, `2`, and `3`.

## Notes

This is a plain educational RNN. It uses backpropagation through time over each
full sequence, mean squared error, tanh hidden states, and a linear output layer.
It is intentionally small enough to read in one sitting.
