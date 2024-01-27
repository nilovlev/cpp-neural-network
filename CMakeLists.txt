cmake_minimum_required(VERSION 3.0)

project(cpp-neural-network)

set(CMAKE_CXX_STANDARD 17)

include_directories("eigen")

file(GLOB SOURCES *.cpp)

add_executable(Neural_Network ${SOURCES})
