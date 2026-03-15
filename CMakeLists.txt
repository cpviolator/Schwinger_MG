cmake_minimum_required(VERSION 3.16)
project(schwinger_mg LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenMP — on macOS with Homebrew libomp, help CMake find it
if(APPLE)
    execute_process(COMMAND brew --prefix libomp OUTPUT_VARIABLE LIBOMP_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(LIBOMP_PREFIX)
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib")
    endif()
endif()
find_package(OpenMP REQUIRED)

# Source files
set(SOURCES
    src/linalg.cpp
    src/smoother.cpp
    src/eigensolver.cpp
    src/solvers.cpp
    src/multigrid.cpp
    src/hmc.cpp
    src/main.cpp
)

add_executable(schwinger ${SOURCES})

target_include_directories(schwinger PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(schwinger PRIVATE OpenMP::OpenMP_CXX m)

# Release by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
