cmake_minimum_required(VERSION 3.0)

project(cpp-neural-network)

set(EXEC hello)

set(CMAKE_CXX_STANDARD 17)

include_directories("eigen")
include_directories("mnist")

file(GLOB SOURCES *.cpp)

add_executable(Neural_Network ${SOURCES})
