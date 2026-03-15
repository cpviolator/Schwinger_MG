#pragma once
#include <complex>
#include <vector>
#include <functional>

using cx = std::complex<double>;
using Vec = std::vector<cx>;
using RVec = std::vector<double>;
using OpApply = std::function<void(const Vec& src, Vec& dst)>;

constexpr int OMP_MIN_SIZE = 4096;

// Global fine-grid matvec counter (for benchmarking eigensolver cost)
extern long long g_matvec_count;
