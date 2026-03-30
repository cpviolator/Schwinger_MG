#include "mode_hmc_benchmark.h"
#include "dirac.h"
#include "hmc.h"
#include "linalg.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <vector>
#include <numeric>

using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double>;

int run_hmc_benchmark(GaugeField& gauge, const Lattice& lat,
                      const LatticeConfig& lcfg, const SolverConfig& scfg,
                      const HMCConfig& hcfg, std::mt19937& rng)
{
    int L = lcfg.L;
    double mass = lcfg.mass;
    double wilson_r = lcfg.wilson_r;
    double beta = hcfg.beta;
    int n_therm = hcfg.n_therm;
    int n_traj = hcfg.n_traj;
    int n_steps = hcfg.n_steps;
    double tau = hcfg.tau;
    double dt = tau / n_steps;

    std::cout << "=== HMC Physics Benchmark ===\n";
    std::cout << "L=" << L << "  DOF=" << lat.ndof
              << "  mass=" << mass << "  beta=" << beta << "\n";
    std::cout << "n_steps=" << n_steps << "  dt=" << std::fixed
              << std::setprecision(4) << dt
              << "  tau=" << tau << "\n";
    std::cout << "therm=" << n_therm << "  traj=" << n_traj << "\n\n";

    HMCParams params;
    params.beta = beta;
    params.tau = tau;
    params.n_steps = n_steps;
    params.cg_maxiter = scfg.max_iter;
    params.cg_tol = scfg.tol;
    params.use_mg = false;
    params.c_sw = lcfg.c_sw;
    params.mu_t = lcfg.mu_t;
    params.use_eo = false;

    // === 1. Force verification ===
    std::cout << "--- Force Verification (L=" << L << ") ---\n";
    bool force_pass = verify_forces(gauge, beta, mass, wilson_r,
                                    scfg.max_iter, scfg.tol, lcfg.c_sw,
                                    false, 1e-4, lcfg.mu_t);
    std::cout << "\n";

    // === 2. Thermalisation ===
    std::cout << "--- Thermalisation (" << n_therm << " trajectories) ---\n";
    for (int t = 0; t < n_therm; t++) {
        auto res = hmc_trajectory(gauge, lat, mass, wilson_r, params, rng);
        if ((t+1) % 10 == 0 || t == n_therm - 1)
            std::cout << "  traj " << t+1 << "/" << n_therm
                      << "  <plaq>=" << std::fixed << std::setprecision(4)
                      << gauge.avg_plaq()
                      << "  dH=" << std::scientific << std::setprecision(3) << res.dH
                      << "  " << (res.accepted ? "Y" : "N") << "\n";
    }
    std::cout << "  Thermalised <plaq>=" << std::fixed << std::setprecision(6)
              << gauge.avg_plaq() << "\n\n";

    // === 3. Production measurement ===
    std::cout << "--- Production (" << n_traj << " trajectories) ---\n";
    std::cout << std::setw(5) << "traj"
              << std::setw(8) << "Plaq"
              << std::setw(12) << "dH"
              << std::setw(4) << "A"
              << std::setw(6) << "CG"
              << std::setw(8) << "Time"
              << "\n";
    std::cout << std::string(43, '-') << "\n";

    std::vector<double> dH_vals, abs_dH_vals, exp_neg_dH_vals;
    std::vector<double> plaq_vals, time_vals;
    std::vector<int> cg_vals;
    int n_accept = 0;

    for (int t = 0; t < n_traj; t++) {
        auto t0 = Clock::now();
        auto res = hmc_trajectory(gauge, lat, mass, wilson_r, params, rng);
        double wall = Duration(Clock::now() - t0).count();

        if (res.accepted) n_accept++;
        dH_vals.push_back(res.dH);
        abs_dH_vals.push_back(std::abs(res.dH));
        exp_neg_dH_vals.push_back(std::exp(-res.dH));
        plaq_vals.push_back(gauge.avg_plaq());
        cg_vals.push_back(res.total_cg_iters);
        time_vals.push_back(wall);

        std::cout << std::setw(5) << t
                  << std::setw(8) << std::fixed << std::setprecision(4) << gauge.avg_plaq()
                  << std::setw(12) << std::scientific << std::setprecision(4) << res.dH
                  << std::setw(4) << (res.accepted ? "Y" : "N")
                  << std::setw(6) << res.total_cg_iters
                  << std::setw(8) << std::fixed << std::setprecision(2) << wall
                  << "\n";
    }

    // === 4. Reversibility test ===
    std::cout << "\n--- Reversibility Test ---\n";
    auto rev = reversibility_test_plain(gauge, lat, mass, wilson_r, params, rng);
    std::cout << "  ||dU||/||U|| = " << std::scientific << std::setprecision(3)
              << rev.gauge_delta << "\n";
    std::cout << "  ||dp||/||p|| = " << rev.mom_delta << "\n";
    std::cout << "  dH_fwd       = " << std::setprecision(6) << rev.dH_forward << "\n";
    std::cout << "  dH_bwd       = " << rev.dH_backward << "\n";
    std::cout << "  dH_fwd+bwd   = " << std::setprecision(3)
              << rev.dH_forward + rev.dH_backward << "\n";
    std::cout << "  CG iters     = " << rev.total_cg << "\n\n";

    // === 5. Summary statistics ===
    auto mean = [](const std::vector<double>& v) {
        return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    };
    auto variance = [&mean](const std::vector<double>& v) {
        double m = mean(v);
        double s = 0;
        for (auto x : v) s += (x - m) * (x - m);
        return s / v.size();
    };

    double accept_rate = (double)n_accept / n_traj;
    double mean_dH = mean(dH_vals);
    double mean_abs_dH = mean(abs_dH_vals);
    double creutz = mean(exp_neg_dH_vals);
    double var_dH = variance(dH_vals);
    double mean_plaq = mean(plaq_vals);
    double mean_cg = std::accumulate(cg_vals.begin(), cg_vals.end(), 0.0) / n_traj;
    double mean_time = mean(time_vals);

    // Count positive/negative dH
    int n_pos = 0, n_neg = 0;
    for (auto d : dH_vals) { if (d > 0) n_pos++; else n_neg++; }

    std::cout << "=== BENCHMARK RESULTS L=" << L << " ===\n";
    std::cout << "L                   = " << L << "\n";
    std::cout << "beta                = " << std::fixed << std::setprecision(6) << beta << "\n";
    std::cout << "mass                = " << mass << "\n";
    std::cout << "n_therm             = " << n_therm << "\n";
    std::cout << "n_traj              = " << n_traj << "\n";
    std::cout << "n_steps             = " << n_steps << "\n";
    std::cout << "dt                  = " << std::setprecision(6) << dt << "\n";
    std::cout << "acceptance_rate     = " << std::setprecision(6) << accept_rate << "\n";
    std::cout << "mean_dH             = " << std::scientific << std::setprecision(6) << mean_dH << "\n";
    std::cout << "mean_abs_dH         = " << mean_abs_dH << "\n";
    std::cout << "creutz_exp_neg_dH   = " << std::fixed << std::setprecision(6) << creutz << "\n";
    std::cout << "var_dH              = " << std::scientific << std::setprecision(6) << var_dH << "\n";
    std::cout << "n_positive_dH       = " << n_pos << "\n";
    std::cout << "n_negative_dH       = " << n_neg << "\n";
    std::cout << "mean_plaq           = " << std::fixed << std::setprecision(6) << mean_plaq << "\n";
    std::cout << "mean_cg_iters       = " << std::setprecision(1) << mean_cg << "\n";
    std::cout << "mean_time           = " << std::setprecision(4) << mean_time << "\n";
    std::cout << "force_check         = " << (force_pass ? "PASS" : "FAIL") << "\n";
    std::cout << "rev_gauge_delta     = " << std::scientific << std::setprecision(3) << rev.gauge_delta << "\n";
    std::cout << "rev_mom_delta       = " << rev.mom_delta << "\n";
    std::cout << "rev_dH_fwd          = " << std::setprecision(6) << rev.dH_forward << "\n";
    std::cout << "rev_dH_bwd          = " << rev.dH_backward << "\n";

    // Return non-zero if force check failed or reversibility is bad
    if (!force_pass) { std::cerr << "FAIL: force verification\n"; return 1; }
    if (rev.gauge_delta > 1e-10) { std::cerr << "FAIL: reversibility\n"; return 1; }
    return 0;
}
