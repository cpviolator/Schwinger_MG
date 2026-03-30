#pragma once
#include "config.h"
#include "types.h"
#include "lattice.h"
#include "gauge.h"
#include <random>

// Force verification mode (lines 369-377 of main.cpp)
int run_verify_forces(GaugeField& gauge, const LatticeConfig& lcfg,
                      const SolverConfig& scfg, const HMCConfig& hcfg);

// Standard HMC mode (lines 382-467 of main.cpp)
int run_hmc_mode(GaugeField& gauge, const Lattice& lat,
                 const LatticeConfig& lcfg, const SolverConfig& scfg,
                 const HMCConfig& hcfg, std::mt19937& rng);

// Fine-grid multi-timescale HMC (lines 472-673 of main.cpp)
int run_multiscale_hmc(GaugeField& gauge, const Lattice& lat,
                       const LatticeConfig& lcfg, const MGConfig& mcfg,
                       const SolverConfig& scfg, const HMCConfig& hcfg,
                       std::mt19937& rng);
