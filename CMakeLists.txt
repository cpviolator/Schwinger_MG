cmake_minimum_required(VERSION 3.16)
project(schwinger_mg LANGUAGES CXX C Fortran)

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

# ======================= FEAST Eigensolver =======================
include(FetchContent)
FetchContent_Declare(feast
    GIT_REPOSITORY https://github.com/certik/feast.git
    GIT_TAG master
)
FetchContent_MakeAvailable(feast)

# Build FEAST as a static library from fetched sources
# Only core + RCI files needed (skip sparse/banded drivers with gfortran issues)
add_library(feast_lib STATIC
    ${feast_SOURCE_DIR}/src/feast.f90
    ${feast_SOURCE_DIR}/src/feast_dense.f90
    ${feast_SOURCE_DIR}/src/f90_functions_wrapper.f90
    ${feast_SOURCE_DIR}/src/point_gauss_legendre.f90
    ${feast_SOURCE_DIR}/src/lbprim.f90
    ${feast_SOURCE_DIR}/src/lsprim.f90
    ${feast_SOURCE_DIR}/src/c_functions_wrapper.c
)
target_include_directories(feast_lib PUBLIC ${feast_SOURCE_DIR}/include)
# Fortran preprocessing: -cpp enables #ifdef, -UMPI disables MPI
target_compile_options(feast_lib PRIVATE
    $<$<COMPILE_LANGUAGE:Fortran>:-cpp -UMPI -w -std=legacy>
)

# LAPACK/BLAS — macOS Accelerate or system LAPACK
if(APPLE)
    find_library(ACCELERATE_LIB Accelerate)
    if(ACCELERATE_LIB)
        target_link_libraries(feast_lib PUBLIC ${ACCELERATE_LIB})
    endif()
else()
    find_package(LAPACK REQUIRED)
    target_link_libraries(feast_lib PUBLIC ${LAPACK_LIBRARIES})
endif()

# Source files
set(SOURCES
    src/linalg.cpp
    src/smoother.cpp
    src/eigensolver_core.cpp
    src/eigen_evolve.cpp
    src/eigen_forecast.cpp
    src/solvers.cpp
    src/coarse_op.cpp
    src/multigrid.cpp
    src/hmc_forces.cpp
    src/hmc_trajectory.cpp
    src/hmc_multiscale.cpp
    src/feast_solver.cpp
    src/mg_builder.cpp
    src/hmc_utils.cpp
    src/mode_hmc.cpp
    src/mode_mg_hmc.cpp
    src/mode_forecast_study.cpp
    src/mode_studies.cpp
    src/mode_hmc_benchmark.cpp
    src/main.cpp
)

add_executable(schwinger ${SOURCES})

target_include_directories(schwinger PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(schwinger PRIVATE OpenMP::OpenMP_CXX feast_lib m)

# Release by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

# ======================= CTest Suite =======================
enable_testing()

# All tests use -t 1 (single thread) for reproducibility and speed

# --- Basic MG Solver ---
add_test(NAME mg_2level
    COMMAND schwinger -L 16 --mg-levels 2 -t 1)
add_test(NAME mg_2level_clover
    COMMAND schwinger -L 32 --mg-levels 2 --csw 1.0 -t 1)
add_test(NAME mg_3level
    COMMAND schwinger -L 32 --mg-levels 3 -k 8 -t 1)

# --- Standard HMC ---
add_test(NAME hmc_wilson
    COMMAND schwinger --hmc -L 16 --hmc-beta 2.0 --hmc-traj 10 --hmc-therm 10 -t 1)
add_test(NAME hmc_clover
    COMMAND schwinger --hmc --csw 1.0 -L 16 --hmc-beta 2.0 --hmc-traj 10 --hmc-therm 10 -t 1)
add_test(NAME hmc_wilson_eo
    COMMAND schwinger --hmc --even-odd -L 16 --hmc-beta 2.0 --hmc-traj 10 --hmc-therm 10 -t 1)
add_test(NAME hmc_clover_eo
    COMMAND schwinger --hmc --even-odd --csw 1.0 -L 16 --hmc-beta 2.0 --hmc-traj 10 --hmc-therm 10 -t 1)

# --- MG Multi-Timescale HMC (Leapfrog, n_outer=20 n_inner=1 = baseline) ---
add_test(NAME mg_multiscale_leapfrog
    COMMAND schwinger -L 16 --hmc-mg-multiscale --mg-levels 2 -b 4 -k 4
        --hmc-n-outer 20 --hmc-n-inner 1 --hmc-n-defl 8 --hmc-traj 10 --hmc-therm 10 -t 1)

# --- MG Multi-Timescale HMC (Omelyan, n_outer=20 for std acceptance) ---
add_test(NAME mg_multiscale_omelyan
    COMMAND schwinger -L 16 --hmc-mg-multiscale --hmc-omelyan --mg-levels 2 -b 4 -k 4
        --hmc-n-outer 20 --hmc-n-inner 1 --hmc-n-defl 8 --hmc-traj 10 --hmc-therm 10 -t 1)

# --- Nested FGI (n_outer=20 so std side also accepts) ---
add_test(NAME mg_multiscale_fgi
    COMMAND schwinger -L 16 --hmc-mg-multiscale --hmc-force-gradient --mg-levels 2 -b 4 -k 4
        --hmc-n-outer 20 --hmc-n-inner 1 --hmc-n-defl 8 --hmc-traj 10 --hmc-therm 10 -t 1)

# --- FGI with Clover ---
add_test(NAME mg_fgi_clover
    COMMAND schwinger -L 16 --hmc-mg-multiscale --hmc-force-gradient --csw 1.0
        --mg-levels 2 -b 4 -k 4 --hmc-n-outer 20 --hmc-n-inner 1 --hmc-n-defl 8 --hmc-traj 10 --hmc-therm 10 -t 1)

# --- FGI with Deflation Refresh (true multi-timescale: n_inner=5) ---
add_test(NAME mg_fgi_defl_refresh
    COMMAND schwinger -L 16 --hmc-mg-multiscale --hmc-force-gradient --mg-levels 2 -b 4 -k 4
        --hmc-n-outer 5 --hmc-n-inner 5 --hmc-n-defl 8 --hmc-defl-refresh 3 --hmc-traj 10 --hmc-therm 10 -t 1)

# --- MG Multi-Timescale with Even-Odd ---
# NOTE: e/o is disabled internally for multi-timescale (force splitting requires
# consistent full-lattice action). The --even-odd flag is accepted but ignored.
# These tests verify the flag doesn't cause errors.
add_test(NAME mg_multiscale_eo
    COMMAND schwinger -L 16 --hmc-mg-multiscale --even-odd --mg-levels 2 -b 4 -k 4
        --hmc-n-outer 20 --hmc-n-inner 1 --hmc-n-defl 8 --hmc-traj 10 --hmc-therm 10 -t 1)
add_test(NAME mg_fgi_eo
    COMMAND schwinger -L 16 --hmc-mg-multiscale --hmc-force-gradient --even-odd
        --mg-levels 2 -b 4 -k 4 --hmc-n-outer 20 --hmc-n-inner 1 --hmc-n-defl 8 --hmc-traj 10 --hmc-therm 10 -t 1)

# --- Reversibility Tests ---
add_test(NAME revtest_wilson
    COMMAND schwinger -L 16 --hmc-mg-multiscale --hmc-revtest --mg-levels 2 -b 4 -k 4
        --hmc-n-outer 5 --hmc-n-inner 3 --hmc-n-defl 8 -t 1)
add_test(NAME revtest_clover
    COMMAND schwinger -L 16 --hmc-mg-multiscale --hmc-revtest --csw 1.0 --mg-levels 2 -b 4 -k 4
        --hmc-n-outer 5 --hmc-n-inner 3 --hmc-n-defl 8 -t 1)

# --- Force Verification (numerical derivative checks) ---
add_test(NAME force_wilson
    COMMAND schwinger -L 8 --verify-forces --hmc-beta 2.0 -t 1)
add_test(NAME force_clover
    COMMAND schwinger -L 8 --verify-forces --csw 1.0 --hmc-beta 2.0 -t 1)
add_test(NAME force_wilson_eo
    COMMAND schwinger -L 8 --verify-forces --even-odd --hmc-beta 2.0 -t 1)
add_test(NAME force_clover_eo
    COMMAND schwinger -L 8 --verify-forces --even-odd --csw 1.0 --hmc-beta 2.0 -t 1)

# --- Force Verification (twisted mass / twisted clover) ---
add_test(NAME force_tm
    COMMAND schwinger -L 8 --verify-forces --mu-t 0.1 --hmc-beta 2.0 -t 1)
add_test(NAME force_tm_eo
    COMMAND schwinger -L 8 --verify-forces --mu-t 0.1 --even-odd --hmc-beta 2.0 -t 1)
add_test(NAME force_tc
    COMMAND schwinger -L 8 --verify-forces --mu-t 0.1 --csw 1.0 --hmc-beta 2.0 -t 1)
add_test(NAME force_tc_eo
    COMMAND schwinger -L 8 --verify-forces --mu-t 0.1 --csw 1.0 --even-odd --hmc-beta 2.0 -t 1)

# --- Twisted Mass / Twisted Clover HMC ---
add_test(NAME hmc_tm
    COMMAND schwinger --hmc --mu-t 0.1 -L 16 --hmc-beta 2.0 --hmc-traj 10 --hmc-therm 10 -t 1)
add_test(NAME hmc_tm_eo
    COMMAND schwinger --hmc --mu-t 0.1 --even-odd -L 16 --hmc-beta 2.0 --hmc-traj 10 --hmc-therm 10 -t 1)
add_test(NAME hmc_tc
    COMMAND schwinger --hmc --mu-t 0.1 --csw 1.0 -L 16 --hmc-beta 2.0 --hmc-traj 10 --hmc-therm 10 -t 1)
add_test(NAME hmc_tc_eo
    COMMAND schwinger --hmc --mu-t 0.1 --csw 1.0 --even-odd -L 16 --hmc-beta 2.0 --hmc-traj 10 --hmc-therm 10 -t 1)

# --- FEAST Eigensolver ---
add_test(NAME feast_mg_solver
    COMMAND schwinger -L 16 --mg-levels 2 -b 4 -k 4 --eigensolver feast --feast-emax 0.5 -t 1 -n 5)
add_test(NAME feast_null_space
    COMMAND schwinger -L 16 --mg-levels 2 -b 4 -k 4 --eigensolver feast --feast-emax 0.3 -t 1 -n 5)
add_test(NAME feast_coarse_defl
    COMMAND schwinger -L 16 --hmc-mg-multiscale --mg-levels 2 -b 4 -k 4
        --eigensolver feast --feast-emax 0.5
        --hmc-n-outer 5 --hmc-n-inner 1 --hmc-n-defl 8 --hmc-traj 3 --hmc-therm 3 -t 1)

# --- Sparse Coarse Eigenvector Evolution ---
add_test(NAME sparse_coarse_evolution
    COMMAND schwinger -L 32 --test-sparse-coarse --mg-levels 2 -b 4 -k 4
        --n-defl 8 -n 5 -e 0.02 -t 1)

# Set timeout for all tests (120 seconds)
set_tests_properties(
    mg_2level mg_2level_clover mg_3level
    hmc_wilson hmc_clover hmc_wilson_eo hmc_clover_eo
    mg_multiscale_leapfrog mg_multiscale_omelyan
    mg_multiscale_fgi mg_fgi_clover mg_fgi_defl_refresh
    mg_multiscale_eo mg_fgi_eo
    revtest_wilson revtest_clover
    force_wilson force_clover force_wilson_eo force_clover_eo
    force_tm force_tm_eo force_tc force_tc_eo
    hmc_tm hmc_tm_eo hmc_tc hmc_tc_eo
    feast_mg_solver feast_null_space feast_coarse_defl
    sparse_coarse_evolution
    PROPERTIES TIMEOUT 300
)
