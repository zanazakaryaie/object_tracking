cmake_minimum_required(VERSION 2.8)

project(klt_tracker)

SET(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -O3 -std=c++11")

find_package(OpenCV REQUIRED)

add_executable(klt_tracker main.cpp FastKltTracker.cpp)
target_link_libraries(klt_tracker ${OpenCV_LIBS})
