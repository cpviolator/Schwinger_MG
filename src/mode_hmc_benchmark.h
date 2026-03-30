#pragma once
#include "config.h"
#include "types.h"
#include "lattice.h"
#include "gauge.h"
#include <random>

// HMC physics benchmark: thermalise + measure + force verification + reversibility
// Prints structured key=value output for automated parsing.
int run_hmc_benchmark(GaugeField& gauge, const Lattice& lat,
                      const LatticeConfig& lcfg, const SolverConfig& scfg,
                      const HMCConfig& hcfg, std::mt19937& rng);
