#pragma once
#include "config.h"
#include "types.h"
#include "lattice.h"
#include "gauge.h"
#include <random>

// MG multi-timescale HMC with coarse-grid deflation (lines 678-1010 of main.cpp)
// Includes reversibility test when revtest=true
int run_mg_hmc(GaugeField& gauge, const Lattice& lat,
               const LatticeConfig& lcfg, const MGConfig& mcfg,
               const SolverConfig& scfg, const HMCConfig& hcfg,
               std::mt19937& rng, bool revtest);
