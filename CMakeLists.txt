cmake_minimum_required(VERSION 3.0)

project(cpp-neural-network)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories("eigen")
include_directories("EigenRand")

add_executable(Neural_Network activation_function.cpp layer.cpp load_data.cpp loss.cpp main.cpp network.cpp)
