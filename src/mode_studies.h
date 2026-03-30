#pragma once
#include "config.h"
#include "types.h"
#include "lattice.h"
#include "gauge.h"
#include <random>

// Sparse coarse eigenvector evolution study (lines 1296-1597 of main.cpp)
int run_sparse_coarse_study(GaugeField& gauge, const Lattice& lat,
                            const LatticeConfig& lcfg, const MGConfig& mcfg,
                            const SolverConfig& scfg, const HMCConfig& hcfg,
                            const StudyConfig& study, std::mt19937& rng);

// Test deflation mode (lines 1602-2801 of main.cpp)
int run_deflation_study(GaugeField& gauge, const Lattice& lat,
                        const LatticeConfig& lcfg, const MGConfig& mcfg,
                        const SolverConfig& scfg, const StudyConfig& study,
                        std::mt19937& rng);

// No-MG baseline mode (lines 2806-2869 of main.cpp)
int run_no_mg_baseline(GaugeField& gauge, const Lattice& lat,
                       const LatticeConfig& lcfg, const SolverConfig& scfg,
                       const StudyConfig& study, std::mt19937& rng);

// Default MG solver study (lines 2874-3262 of main.cpp)
int run_mg_solver_study(GaugeField& gauge, const Lattice& lat,
                        const LatticeConfig& lcfg, const MGConfig& mcfg,
                        const SolverConfig& scfg, const StudyConfig& study,
                        std::mt19937& rng);
