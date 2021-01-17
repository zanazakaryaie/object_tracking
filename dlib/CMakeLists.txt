cmake_minimum_required(VERSION 2.8)

project(dlib_tracker)

SET(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -O3 -std=c++11")

find_package(OpenCV REQUIRED)

find_package(dlib REQUIRED)

add_executable(dlib_tracker main.cpp)
target_link_libraries(dlib_tracker ${OpenCV_LIBS} dlib::dlib)
