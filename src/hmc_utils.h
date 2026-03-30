#pragma once
#include "config.h"
#include "types.h"
#include "lattice.h"
#include "gauge.h"
#include "dirac.h"
#include "multigrid.h"
#include "hmc.h"
#include <random>
#include <string>

// Generate the standard thermalisation config file path
std::string therm_cfg_path(int L, double beta, int n_therm);

// Try to load a thermalised gauge config, or thermalise from scratch.
// Returns true if loaded, false if thermalised.
// Always saves the config after thermalisation.
bool load_or_thermalise(GaugeField& gauge, const Lattice& lat,
                        const LatticeConfig& lcfg, const MGConfig& mcfg,
                        const SolverConfig& scfg, const HMCConfig& hcfg,
                        std::mt19937& rng, int n_defl);
