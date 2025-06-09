cmake_minimum_required(VERSION 3.10)
project(SOD)

set(CMAKE_CXX_STANDARD 17)

# find Eigen
find_package(Eigen3 REQUIRED)

# 
add_executable(main 
    src/main_omp.cpp
    src/split.cpp
    src/derivatives.cpp
    src/nov_5.cpp
)

# 
target_include_directories(main PRIVATE
    ${EIGEN3_INCLUDE_DIR}
    include/
)