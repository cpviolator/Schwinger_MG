#pragma once
#include "config.h"
#include "types.h"
#include "lattice.h"
#include "gauge.h"
#include <random>

// HMC physics benchmark: thermalise + measure + force verification + reversibility
// Prints structured key=value output for automated parsing.
// When mcfg.mg_levels >= 2, builds MG hierarchy and uses preconditioned CG.
// When hcfg.use_eo is true, uses even-odd Schur complement.
// Otherwise, plain CG on full lattice (gold standard).
int run_hmc_benchmark(GaugeField& gauge, const Lattice& lat,
                      const LatticeConfig& lcfg, const MGConfig& mcfg,
                      const SolverConfig& scfg, const HMCConfig& hcfg,
                      std::mt19937& rng);

// FEAST vs TRLM null-space benchmark: compare cold/warm start
// for both TRLM (inverse iteration) and FEAST on evolved gauge configs.
int run_feast_benchmark(GaugeField& gauge, const Lattice& lat,
                        const LatticeConfig& lcfg, const MGConfig& mcfg,
                        const SolverConfig& scfg, const HMCConfig& hcfg,
                        std::mt19937& rng);
