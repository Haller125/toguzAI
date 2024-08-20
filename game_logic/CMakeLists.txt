cmake_minimum_required(VERSION 3.28)
project(game_logic VERSION 0.1.0 DESCRIPTION "Game logic for Toguz Korgool")

set(CMAKE_CXX_STANDARD 14)

# Find the Python libraries
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Add the path to the Python headers
include_directories(${Python_INCLUDE_DIRS})
include_directories(./pybind11/include)
include_directories(.)


add_library(toguz SHARED
        toguz.h
        toguz_board_wrapper.cpp)

# Link the Python libraries
target_link_libraries(game_logic ${Python_LIBRARIES})
