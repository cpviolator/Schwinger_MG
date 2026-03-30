#include "hmc_utils.h"
#include "mg_builder.h"
#include <iostream>
#include <iomanip>
#include <chrono>

std::string therm_cfg_path(int L, double beta, int n_therm) {
    return "gauge_L" + std::to_string(L) + "_b"
        + std::to_string(beta).substr(0,4) + "_t"
        + std::to_string(n_therm) + ".bin";
}

bool load_or_thermalise(GaugeField& gauge, const Lattice& lat,
                        const LatticeConfig& lcfg, const MGConfig& mcfg,
                        const SolverConfig& scfg, const HMCConfig& hcfg,
                        std::mt19937& rng, int n_defl)
{
    std::string cfg_path = therm_cfg_path(lcfg.L, hcfg.beta, hcfg.n_therm);

    if (gauge.load(cfg_path)) {
        VOUT(V_SUMMARY) << "--- Loaded thermalised config from " << cfg_path
                  << "  <plaq>=" << std::fixed << std::setprecision(4)
                  << gauge.avg_plaq() << " ---\n\n";
        return true;
    }

    VOUT(V_SUMMARY) << "--- Thermalisation: " << hcfg.n_therm
              << " standard HMC trajectories ---\n";

    DiracOp D_th(lat, gauge, lcfg.mass, lcfg.wilson_r, lcfg.c_sw, lcfg.mu_t);
    std::mt19937 rng_th(lcfg.seed);
    auto mg_th = build_full_mg(D_th, mcfg, scfg, rng_th, n_defl, true);

    std::function<Vec(const Vec&)> pc_th = [&mg_th](const Vec& b) -> Vec {
        return mg_th.precondition(b);
    };

    HMCParams hp;
    hp.beta = hcfg.beta;
    hp.tau = hcfg.tau;
    hp.n_steps = hcfg.n_steps;
    hp.cg_maxiter = scfg.max_iter;
    hp.cg_tol = scfg.tol;
    hp.c_sw = lcfg.c_sw;
    hp.mu_t = lcfg.mu_t;

    int accepted = 0;
    for (int t = 0; t < hcfg.n_therm; t++) {
        if (t > 0 && t % 5 == 0) {
            DiracOp Dr(lat, gauge, lcfg.mass, lcfg.wilson_r, lcfg.c_sw, lcfg.mu_t);
            std::mt19937 rng_rb(lcfg.seed + t);
            mg_th = build_full_mg(Dr, mcfg, scfg, rng_rb, n_defl, false);
        }
        auto res = hmc_trajectory(gauge, lat, lcfg.mass, lcfg.wilson_r, hp, rng, &pc_th);
        if (res.accepted) accepted++;
        if ((t+1) % 10 == 0 || t == hcfg.n_therm - 1)
            VOUT(V_SUMMARY) << "  traj " << t+1 << "/" << hcfg.n_therm
                      << "  accept=" << accepted << "/" << (t+1)
                      << "  <plaq>=" << std::fixed << std::setprecision(4)
                      << gauge.avg_plaq() << "\n";
    }
    if (gauge.save(cfg_path))
        VOUT(V_VERBOSE) << "  Saved thermalised config to " << cfg_path << "\n";
    VOUT(V_SUMMARY) << "\n";

    return false;
}
