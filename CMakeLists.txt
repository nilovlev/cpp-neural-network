cmake_minimum_required(VERSION 3.0)

project(cpp-neural-network)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories("eigen")
include_directories("EigenRand")

file(GLOB SOURCES *.cpp)

add_executable(Neural_Network ${SOURCES})
