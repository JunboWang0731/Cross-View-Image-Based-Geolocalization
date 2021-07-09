cmake_minimum_required(VERSION 2.6)
add_definitions(-std=c++11)

project(geolocalization)
set(CMAKE_BUILD_TYPE "Debug")

find_package(OpenCV 3.0 QUIET)
find_package(Eigen3 REQUIRED)
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

find_package( Ceres REQUIRED )
include_directories( ${CERES_INCLUDE_DIRS} )

include_directories(${EIGEN3_INCLUDE_DIRS})

add_executable(Test src/Test.cpp)
target_link_libraries(Test ${OpenCV_LIBS} ${CERES_LIBRARIES})
