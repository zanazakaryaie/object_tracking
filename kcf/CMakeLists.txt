cmake_minimum_required(VERSION 2.8)

project(kcf_tracker)

SET(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -O3 -std=c++11")

find_package(OpenCV REQUIRED)

include_directories(KCFcpp/src/) 
file( GLOB KCF_SOURCES "KCFcpp/src/*.cpp" )
add_library( KCF ${KCF_SOURCES})

add_executable(kcf_tracker main.cpp)
target_link_libraries(kcf_tracker ${OpenCV_LIBS} KCF)
