#pragma once
#include <complex>
#include <vector>
#include <functional>
#include <iostream>

using cx = std::complex<double>;
using Vec = std::vector<cx>;
using RVec = std::vector<double>;
using OpApply = std::function<void(const Vec& src, Vec& dst)>;

constexpr int OMP_MIN_SIZE = 4096;

// Global fine-grid matvec counter (for benchmarking eigensolver cost)
extern long long g_matvec_count;

// Verbosity levels
enum Verbosity { V_SILENT = 0, V_SUMMARY = 1, V_VERBOSE = 2, V_DEBUG = 3 };

// Global verbosity (set once from CLI in main.cpp, default = V_VERBOSE)
extern int g_verbosity;

// Convenience: VOUT(level) gates output on verbosity level
// Usage: VOUT(V_VERBOSE) << "MG level: " << dim << "\n";
#define VOUT(level) if (g_verbosity >= (level)) std::cout
