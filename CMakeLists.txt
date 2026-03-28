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
    src/coarse_op.cpp
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
    sparse_coarse_evolution
    PROPERTIES TIMEOUT 300
)
