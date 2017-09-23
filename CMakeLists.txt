project(ExtendedKF)

cmake_minimum_required (VERSION 3.5)

add_definitions(-std=c++11)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS, "${CXX_FLAGS}")

set(sources src/tools.cpp src/FusionEKF.cpp src/kalman_filter.cpp include/tools.h include/FusionEKF.h include/kalman_filter.h)

include_directories(include/)

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin") 

include_directories(/usr/local/include)
include_directories(/usr/local/opt/openssl/include)
link_directories(/usr/local/lib)
link_directories(/usr/local/opt/openssl/lib)
link_directories(/usr/local/Cellar/libuv/1.11.0/lib)

endif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin") 

add_library(ExtendedKFLib ${sources})
add_executable(ExtendedKF src/main.cpp)
target_link_libraries(ExtendedKF ExtendedKFLib z ssl uv uWS)

enable_testing()
set(test_sources test/Catch/catch.hpp test/test_main.cpp test/test_tools.cpp)
add_executable(tests ${test_sources})
target_link_libraries(tests ExtendedKFLib)