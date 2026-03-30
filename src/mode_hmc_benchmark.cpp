#include "mode_hmc_benchmark.h"
#include "dirac.h"
#include "hmc.h"
#include "multigrid.h"
#include "mg_builder.h"
#include "prolongator.h"
#include "coarse_op.h"
#include "eigensolver.h"
#include "linalg.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <vector>
#include <numeric>
#include <memory>

using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double>;

int run_hmc_benchmark(GaugeField& gauge, const Lattice& lat,
                      const LatticeConfig& lcfg, const MGConfig& mcfg,
                      const SolverConfig& scfg, const HMCConfig& hcfg,
                      std::mt19937& rng)
{
    int L = lcfg.L;
    double mass = lcfg.mass;
    double wilson_r = lcfg.wilson_r;
    double beta = hcfg.beta;
    int n_therm = hcfg.n_therm;
    int n_traj = hcfg.n_traj;
    double tau = hcfg.tau;
    bool use_mg = (mcfg.mg_levels >= 2);
    bool use_eo = hcfg.use_eo;
    bool use_multiscale = hcfg.force_gradient || hcfg.omelyan;
    // Multi-timescale requires MG for coarse deflation
    bool use_mg_multiscale = use_multiscale && use_mg;

    // For single-timescale: n_steps is the leapfrog/Omelyan step count
    // For multi-timescale: n_outer × n_inner total MD steps
    int n_steps = hcfg.n_steps;
    double dt = tau / n_steps;

    // Describe configuration
    std::string solver_desc = "plain CG";
    if (use_eo) solver_desc = "even-odd CG";
    if (use_mg && !use_eo) solver_desc = "MG-preconditioned CG";
    if (use_mg && use_eo) solver_desc = "MG-preconditioned even-odd CG";

    std::string integrator_desc;
    if (use_mg_multiscale && hcfg.force_gradient)
        integrator_desc = "Multi-timescale FGI (n_outer=" + std::to_string(hcfg.n_outer)
            + " n_inner=" + std::to_string(hcfg.n_inner) + ")";
    else if (use_mg_multiscale && hcfg.omelyan)
        integrator_desc = "Multi-timescale Omelyan (n_outer=" + std::to_string(hcfg.n_outer)
            + " n_inner=" + std::to_string(hcfg.n_inner) + ")";
    else if (hcfg.omelyan)
        integrator_desc = "Omelyan (2MN, lambda=0.1932)";
    else
        integrator_desc = "Leapfrog";

    std::cout << "=== HMC Physics Benchmark ===\n";
    std::cout << "L=" << L << "  DOF=" << lat.ndof
              << "  mass=" << mass << "  beta=" << beta << "\n";
    if (use_mg_multiscale)
        std::cout << "n_outer=" << hcfg.n_outer << "  n_inner=" << hcfg.n_inner
                  << "  total_steps=" << hcfg.n_outer * hcfg.n_inner
                  << "  tau=" << std::fixed << std::setprecision(4) << tau << "\n";
    else
        std::cout << "n_steps=" << n_steps << "  dt=" << std::fixed
                  << std::setprecision(4) << dt << "  tau=" << tau << "\n";
    std::cout << "solver=" << solver_desc << "\n";
    std::cout << "integrator=" << integrator_desc << "\n";
    std::cout << "therm=" << n_therm << "  traj=" << n_traj << "\n\n";

    // === Setup single-timescale HMC params ===
    HMCParams params;
    params.beta = beta;
    params.tau = tau;
    params.n_steps = n_steps;
    params.cg_maxiter = scfg.max_iter;
    params.cg_tol = scfg.tol;
    params.use_mg = false;
    params.c_sw = lcfg.c_sw;
    params.mu_t = lcfg.mu_t;
    params.use_eo = use_eo;
    params.omelyan = hcfg.omelyan && !use_mg_multiscale;

    // === Setup multi-timescale params ===
    MGMultiScaleParams ms_params;
    ms_params.beta = beta;
    ms_params.tau = tau;
    ms_params.n_outer = hcfg.n_outer;
    ms_params.n_inner = hcfg.n_inner;
    ms_params.cg_maxiter = scfg.max_iter;
    ms_params.cg_tol = scfg.tol;
    ms_params.c_sw = lcfg.c_sw;
    ms_params.mu_t = lcfg.mu_t;
    ms_params.use_eo = use_eo;
    ms_params.defl_refresh = hcfg.defl_refresh;
    if (hcfg.omelyan) ms_params.outer_type = OuterIntegrator::Omelyan;
    if (hcfg.force_gradient) ms_params.outer_type = OuterIntegrator::FGI;

    // === Build MG hierarchy ===
    std::unique_ptr<MGHierarchy> mg;
    std::function<Vec(const Vec&)> mg_precond_fn;
    const std::function<Vec(const Vec&)>* precond_ptr = nullptr;
    CoarseDeflState cdefl;
    auto D_mg = std::make_unique<DiracOp>(lat, gauge, mass, wilson_r, lcfg.c_sw, lcfg.mu_t);

    if (use_mg) {
        std::cout << "--- Building MG hierarchy ---\n";
        std::mt19937 rng_mg(lcfg.seed + 111);
        int n_defl = hcfg.n_defl > 0 ? hcfg.n_defl : 8;
        mg = std::make_unique<MGHierarchy>(
            build_full_mg(*D_mg, mcfg, scfg, rng_mg, use_mg_multiscale ? n_defl : 0, true));
        if (mcfg.symmetric_mg) mg->set_symmetric(0.8);
        mg_precond_fn = [&mg](const Vec& b) -> Vec { return mg->precondition(b); };
        precond_ptr = &mg_precond_fn;

        if (use_mg_multiscale) {
            cdefl.eigvecs = mg->sparse_Ac.defl_vecs;
            cdefl.eigvals = mg->sparse_Ac.defl_vals;
            std::cout << "  Coarse deflation: " << cdefl.eigvecs.size() << " vectors\n";
        }
        std::cout << "\n";
    }

    // === 1. Force verification ===
    std::cout << "--- Force Verification (L=" << L << ") ---\n";
    bool force_pass = verify_forces(gauge, beta, mass, wilson_r,
                                    scfg.max_iter, scfg.tol, lcfg.c_sw,
                                    use_eo, 1e-4, lcfg.mu_t);
    std::cout << "\n";

    // Rebind MG operator after force verification
    if (use_mg) {
        D_mg = std::make_unique<DiracOp>(lat, gauge, mass, wilson_r, lcfg.c_sw, lcfg.mu_t);
        mg->levels[0].op = D_mg->as_DdagD_op();
        mg->levels[0].Ac.build(*D_mg, mg->geo_prolongators[0]);
        mg->rebuild_deeper_levels();
    }

    // === 2. Thermalisation (plain CG, no MG — fast at heavy mass, avoids sparse coarse issues) ===
    std::cout << "--- Thermalisation (" << n_therm << " trajectories, plain CG) ---\n";
    HMCParams therm_params = params;
    therm_params.omelyan = false;
    for (int t = 0; t < n_therm; t++) {
        auto res = hmc_trajectory(gauge, lat, mass, wilson_r, therm_params, rng);
        if ((t+1) % 10 == 0 || t == n_therm - 1)
            std::cout << "  traj " << t+1 << "/" << n_therm
                      << "  <plaq>=" << std::fixed << std::setprecision(4)
                      << gauge.avg_plaq()
                      << "  dH=" << std::scientific << std::setprecision(3) << res.dH
                      << "  " << (res.accepted ? "Y" : "N") << "\n";
    }
    std::cout << "  Thermalised <plaq>=" << std::fixed << std::setprecision(6)
              << gauge.avg_plaq() << "\n\n";

    // Rebuild MG on thermalised config
    if (use_mg) {
        D_mg = std::make_unique<DiracOp>(lat, gauge, mass, wilson_r, lcfg.c_sw, lcfg.mu_t);
        std::mt19937 rng_mg2(lcfg.seed + 222);
        int n_defl = hcfg.n_defl > 0 ? hcfg.n_defl : 8;
        *mg = build_full_mg(*D_mg, mcfg, scfg, rng_mg2, use_mg_multiscale ? n_defl : 0, false);
        if (mcfg.symmetric_mg) mg->set_symmetric(0.8);
        if (use_mg_multiscale) {
            cdefl.eigvecs = mg->sparse_Ac.defl_vecs;
            cdefl.eigvals = mg->sparse_Ac.defl_vals;
        }
    }

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
        // Rebuild MG periodically based on rebuild_freq
        int rb_freq = hcfg.rebuild_freq;  // 0=never, >0=every N traj
        if (use_mg && rb_freq > 0 && t > 0 && t % rb_freq == 0) {
            D_mg = std::make_unique<DiracOp>(lat, gauge, mass, wilson_r, lcfg.c_sw, lcfg.mu_t);

            if (hcfg.eigen_forecast) {
                // Warm rebuild: reuse previous null vectors with 5 inverse iterations
                auto warm_vecs = mg->null_vecs_l0;
                std::mt19937 rng_rb(lcfg.seed + 5000 + t);
                *mg = build_mg_hierarchy_warm(*D_mg, mcfg.mg_levels, mcfg.block_size,
                    mcfg.k_null, mcfg.resolved_coarse_block(), 5, rng_rb,
                    warm_vecs, mcfg.w_cycle, mcfg.pre_smooth, mcfg.post_smooth);
                if (mcfg.symmetric_mg) mg->set_symmetric(0.8);
            } else {
                // Galerkin rebuild: just re-project P†AP (no new null vectors)
                mg->levels[0].op = D_mg->as_DdagD_op();
                mg->levels[0].Ac.build(*D_mg, mg->geo_prolongators[0]);
                mg->rebuild_deeper_levels();
            }

            if (use_mg_multiscale && !cdefl.eigvecs.empty()) {
                OpApply A_new = D_mg->as_DdagD_op();
                mg->sparse_Ac.build(mg->geo_prolongators[0], A_new, lat.ndof);
                evolve_coarse_deflation(cdefl, mg->sparse_Ac);
            }
        }

        auto t0 = Clock::now();
        double dH;
        int cg_iters;
        bool accepted;

        if (use_mg_multiscale) {
            auto res = hmc_trajectory_mg_multiscale(
                gauge, lat, mass, wilson_r, ms_params,
                cdefl, mg->geo_prolongators[0], mg_precond_fn, rng);
            dH = res.dH;
            cg_iters = res.highmode_cg_iters;
            accepted = res.accepted;
        } else {
            auto res = hmc_trajectory(gauge, lat, mass, wilson_r, params, rng, precond_ptr);
            dH = res.dH;
            cg_iters = res.total_cg_iters;
            accepted = res.accepted;
        }

        double wall = Duration(Clock::now() - t0).count();
        if (accepted) n_accept++;
        dH_vals.push_back(dH);
        abs_dH_vals.push_back(std::abs(dH));
        exp_neg_dH_vals.push_back(std::exp(-dH));
        plaq_vals.push_back(gauge.avg_plaq());
        cg_vals.push_back(cg_iters);
        time_vals.push_back(wall);

        std::cout << std::setw(5) << t
                  << std::setw(8) << std::fixed << std::setprecision(4) << gauge.avg_plaq()
                  << std::setw(12) << std::scientific << std::setprecision(4) << dH
                  << std::setw(4) << (accepted ? "Y" : "N")
                  << std::setw(6) << cg_iters
                  << std::setw(8) << std::fixed << std::setprecision(2) << wall
                  << "\n";
    }

    // === 4. Reversibility test ===
    std::cout << "\n--- Reversibility Test ---\n";
    if (use_mg_multiscale) {
        auto rev = reversibility_test_mg_multiscale(
            gauge, lat, mass, wilson_r, ms_params,
            cdefl, mg->geo_prolongators[0], mg_precond_fn, rng);
        std::cout << "  ||dU||/||U|| = " << std::scientific << std::setprecision(3)
                  << rev.gauge_delta << "\n";
        std::cout << "  ||dp||/||p|| = " << rev.mom_delta << "\n";
        std::cout << "  dH_fwd       = " << std::setprecision(6) << rev.dH_forward << "\n";
        std::cout << "  dH_bwd       = " << rev.dH_backward << "\n";
        std::cout << "  dH_fwd+bwd   = " << std::setprecision(3)
                  << rev.dH_forward + rev.dH_backward << "\n\n";
    } else {
        auto rev = reversibility_test_plain(gauge, lat, mass, wilson_r, params, rng, precond_ptr);
        std::cout << "  ||dU||/||U|| = " << std::scientific << std::setprecision(3)
                  << rev.gauge_delta << "\n";
        std::cout << "  ||dp||/||p|| = " << rev.mom_delta << "\n";
        std::cout << "  dH_fwd       = " << std::setprecision(6) << rev.dH_forward << "\n";
        std::cout << "  dH_bwd       = " << rev.dH_backward << "\n";
        std::cout << "  dH_fwd+bwd   = " << std::setprecision(3)
                  << rev.dH_forward + rev.dH_backward << "\n\n";
    }

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

    int n_pos = 0, n_neg = 0;
    for (auto d : dH_vals) { if (d > 0) n_pos++; else n_neg++; }

    std::cout << "=== BENCHMARK RESULTS L=" << L << " ===\n";
    std::cout << "L                   = " << L << "\n";
    std::cout << "beta                = " << std::fixed << std::setprecision(6) << beta << "\n";
    std::cout << "mass                = " << mass << "\n";
    std::cout << "solver              = " << solver_desc << "\n";
    std::cout << "integrator          = " << integrator_desc << "\n";
    std::cout << "n_therm             = " << n_therm << "\n";
    std::cout << "n_traj              = " << n_traj << "\n";
    if (use_mg_multiscale) {
        std::cout << "n_outer             = " << hcfg.n_outer << "\n";
        std::cout << "n_inner             = " << hcfg.n_inner << "\n";
    } else {
        std::cout << "n_steps             = " << n_steps << "\n";
        std::cout << "dt                  = " << std::setprecision(6) << dt << "\n";
    }
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

    if (!force_pass) { std::cerr << "FAIL: force verification\n"; return 1; }
    return 0;
}
