cmake_minimum_required(VERSION 3.16)

project(rnn_cpp
    VERSION 1.0.0
    DESCRIPTION "A small recurrent neural network from scratch in C++"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(rnn_core
    src/activations.cpp
    src/matrix.cpp
    src/rnn.cpp
)

target_include_directories(rnn_core PUBLIC include)

if(MSVC)
    target_compile_options(rnn_core PRIVATE /W4)
else()
    target_compile_options(rnn_core PRIVATE -Wall -Wextra -pedantic)
endif()

add_executable(rnn_demo src/main.cpp)
target_link_libraries(rnn_demo PRIVATE rnn_core)
