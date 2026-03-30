#pragma once
#include "config.h"
#include "multigrid.h"
#include <random>

// Build MG hierarchy + optionally sparse coarse operator + deflation in one call.
// Wraps: build_mg_hierarchy (with FEAST or inverse iteration null space)
//        + setup_sparse_coarse (if n_defl > 0).
MGHierarchy build_full_mg(
    const DiracOp& D, const MGConfig& mcfg, const SolverConfig& scfg,
    std::mt19937& rng, int n_defl = 0,
    bool verbose = true,
    const std::vector<Vec>* warm_start = nullptr);
