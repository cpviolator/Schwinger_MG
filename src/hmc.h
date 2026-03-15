#pragma once
#include "types.h"
#include "linalg.h"
#include "lattice.h"
#include "gauge.h"
#include "dirac.h"
#include "solvers.h"
#include <array>
#include <random>
#include <functional>

// Perturb gauge field by small random phases (simulates one MD step)
void perturb_gauge(GaugeField& g, std::mt19937& rng, double epsilon);

// Gauge field I/O
struct GaugeHeader {
    uint32_t magic;   // 0x53574E47 = "SWNG"
    uint32_t L;
    double beta;
    double mass;
    double avg_plaq;
};

bool save_gauge(const GaugeField& g, double beta, double mass,
                const std::string& filename);
bool load_gauge(GaugeField& g, GaugeHeader& hdr, const std::string& filename);

// Momentum field
struct MomentumField {
    const Lattice& lat;
    std::array<RVec, 2> pi;

    MomentumField(const Lattice& lat_) : lat(lat_) {
        pi[0].resize(lat.V, 0.0);
        pi[1].resize(lat.V, 0.0);
    }

    void randomise(std::mt19937& rng);
    double kinetic_energy() const;
};

// Wilson gauge action
double gauge_action(const GaugeField& g, double beta);

// Gauge force
void gauge_force(const GaugeField& g, double beta,
                 std::array<RVec, 2>& force);

// Fermion action
struct FermionActionResult {
    double action;
    Vec X;
    int cg_iters;
};

FermionActionResult fermion_action(const DiracOp& D, const Vec& phi,
                                    int max_iter, double tol);
FermionActionResult fermion_action_precond(
    const DiracOp& D, const Vec& phi,
    const std::function<Vec(const Vec&)>& precond,
    int max_iter, double tol);

// Fermion force
void fermion_force(const DiracOp& D, const Vec& X,
                   std::array<RVec, 2>& force);

// Force verification
void verify_forces(const GaugeField& g, double beta, double mass, double wilson_r,
                   int max_iter, double tol);

// Pseudofermion generation
void generate_pseudofermion(const DiracOp& D, std::mt19937& rng, Vec& phi);

// HMC parameters and results
struct HMCParams {
    double beta;
    double tau;
    int n_steps;
    int cg_maxiter;
    double cg_tol;
    bool use_mg;
};

struct HMCResult {
    bool accepted;
    double dH;
    double accept_rate;
    int total_cg_iters;
};

HMCResult hmc_trajectory(
    GaugeField& gauge, const Lattice& lat, double mass, double wilson_r,
    const HMCParams& params, std::mt19937& rng,
    const std::function<Vec(const Vec&)>* precond = nullptr);
