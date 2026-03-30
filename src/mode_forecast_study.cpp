#include "mode_forecast_study.h"
#include "config.h"
#include "hmc_utils.h"
#include "mg_builder.h"
#include "dirac.h"
#include "hmc.h"
#include "multigrid.h"
#include "eigensolver.h"
#include "linalg.h"
#include "feast_solver.h"
#include "coarse_op.h"
#include "prolongator.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <set>
#include <sstream>

int run_forecast_study(GaugeField& gauge, const Lattice& lat,
                       const LatticeConfig& lcfg, const MGConfig& mcfg,
                       const SolverConfig& scfg, const HMCConfig& hcfg,
                       std::mt19937& rng) {
    if (mcfg.mg_levels < 2) {
        std::cerr << "--forecast-study requires --mg-levels >= 2\n";
        return 1;
    }

    using Clock = std::chrono::high_resolution_clock;
    using Dur = std::chrono::duration<double>;

    // All parameters come from CLI — no hardcoded overrides
    int n_traj = hcfg.n_traj;
    int n_defl = scfg.n_defl > 0 ? scfg.n_defl : hcfg.n_defl;
    int ndof = lat.ndof;
    int total_steps = hcfg.n_outer * hcfg.n_inner;

    std::cout << "=== Eigenspace Forecasting Comparative Study ===\n\n";
    std::cout << "L=" << lcfg.L << "  DOF=" << ndof << "  mass=" << lcfg.mass
              << "  beta=" << hcfg.beta << "  c_sw=" << lcfg.c_sw << "\n";
    std::cout << "MG levels=" << mcfg.mg_levels << "  block=" << mcfg.block_size
              << "  k_null=" << mcfg.k_null << "  n_defl=" << n_defl << "\n";
    std::cout << "Integrator: Nested FGI  n_outer=" << hcfg.n_outer
              << "  n_inner=" << hcfg.n_inner
              << "  total_steps=" << total_steps << "\n";
    std::cout << "defl_refresh=" << hcfg.defl_refresh
              << "  fresh_period=0 (disabled for study)\n";
    std::cout << "Thermalisation: " << hcfg.n_therm
              << " trajectories  Measurement: " << n_traj << " per arm\n";
    std::cout << "Monomials: KE=kinetic  SG=gauge  SF=fermion  dH=total\n\n";

    // --- Phase 1: Thermalise ---
    load_or_thermalise(gauge, lat, lcfg, mcfg, scfg, hcfg, rng, n_defl);

    // --- Phase 2: Build MG hierarchy on thermalised config ---
    DiracOp D_mg(lat, gauge, lcfg.mass, lcfg.wilson_r, lcfg.c_sw, lcfg.mu_t);
    auto mg = build_full_mg(D_mg, mcfg, scfg, rng, n_defl, true);

    auto& P = mg.geo_prolongators[0];
    std::function<Vec(const Vec&)> mg_precond = [&mg](const Vec& b) -> Vec {
        return mg.precondition(b);
    };

    // Set up MS params (FGI)
    MGMultiScaleParams ms_params;
    ms_params.beta = hcfg.beta;
    ms_params.tau = hcfg.tau;
    ms_params.n_outer = hcfg.n_outer;
    ms_params.n_inner = hcfg.n_inner;
    ms_params.cg_maxiter = scfg.max_iter;
    ms_params.cg_tol = scfg.tol;
    ms_params.c_sw = lcfg.c_sw;
    ms_params.mu_t = lcfg.mu_t;
    ms_params.outer_type = OuterIntegrator::FGI;
    ms_params.defl_refresh = hcfg.defl_refresh;
    ms_params.mg_perturb_freq = hcfg.mg_perturb_freq;
    ms_params.use_eo = hcfg.use_eo;

    // --- Phase 3: Hybrid prolongator refresh study ---
    // Compare: stale P, RR-only, warm rebuild, RR + periodic warm rebuild
    struct ArmStats {
        int accept = 0;
        double dH_sum = 0, cg_sum = 0, time_sum = 0;
    };

    struct ArmConfig {
        std::string label;
        int arm_rebuild_freq;
        bool feast_coarse_evolve;
        bool feast_fine_refresh;    // FEAST warm-start for fine null space (MG-preconditioned)
    };
    std::string rf = std::to_string(hcfg.rebuild_freq);
    std::vector<ArmConfig> arms = {
        {"Stale",                     0, false, false},
        {"Rebuild/" + rf,             hcfg.rebuild_freq, false, false},
        {"FEAST-MG/traj",             0, false, true},
        {"FEAST-MG+Rb/" + rf,         hcfg.rebuild_freq, false, true},
    };
    int n_arms = (int)arms.size();

    // Parse --only-arms "2,3,5" to run subset of arms
    std::set<int> run_arms;
    if (!hcfg.only_arms.empty()) {
        std::istringstream ss(hcfg.only_arms);
        std::string tok;
        while (std::getline(ss, tok, ','))
            run_arms.insert(std::stoi(tok));
    }

    std::cout << "=== Prolongator Refresh Study ===\n";
    for (int i = 0; i < n_arms; i++) {
        bool skip = (!run_arms.empty() && run_arms.find(i) == run_arms.end());
        std::cout << "  [" << i << "] " << std::setw(22) << std::left << (arms[i].label + ":")
                  << (arms[i].feast_fine_refresh ? "FEAST-MG fine+coarse  " : "")
                  << (arms[i].feast_coarse_evolve ? "FEAST coarse  " : "")
                  << "rebuild=" << (arms[i].arm_rebuild_freq > 0
                     ? "every " + std::to_string(arms[i].arm_rebuild_freq) + " traj" : "never")
                  << (skip ? "  [SKIP]" : "") << "\n";
    }
    std::cout << std::right << "\n";

    std::vector<ArmStats> stats(n_arms);

    for (int ai = 0; ai < n_arms; ai++) {
        if (!run_arms.empty() && run_arms.find(ai) == run_arms.end()) {
            std::cout << "--- " << arms[ai].label << " --- [SKIPPED]\n\n";
            continue;
        }
        auto& ac = arms[ai];
        std::cout << "--- " << ac.label << " ---\n";
        std::cout << std::setw(5) << "traj"
                  << std::setw(8) << "Plaq"
                  << std::setw(10) << "dH"
                  << std::setw(4) << "A"
                  << std::setw(6) << "CG"
                  << std::setw(8) << "Time"
                  << "\n";
        std::cout << std::string(41, '-') << "\n";

        GaugeField gauge_arm = gauge;
        std::mt19937 rng_arm(lcfg.seed + 3000);
        std::mt19937 rng_mg_arm(lcfg.seed + 111);

        auto mg_arm = mg;
        mg_arm.rebind_prolongator_lambdas();  // fix stale captures from copy

        DiracOp D_arm(lat, gauge_arm, lcfg.mass, lcfg.wilson_r, lcfg.c_sw, lcfg.mu_t);
        mg_arm.levels[0].op = [&D_arm](const Vec& s, Vec& d) {
            D_arm.apply_DdagD(s, d);
        };
        // keep use_sparse_coarse = true (stencil rebuilt cheaply, no TRLM)
        mg_arm.init_Dv_cache(D_arm);

        std::function<Vec(const Vec&)> pc_arm = [&mg_arm](const Vec& b) -> Vec {
            return mg_arm.precondition(b);
        };

        // Inner force: empty cdefl — identical physics across arms
        CoarseDeflState cdefl_arm;

        for (int t = 0; t < n_traj; t++) {
            if (lcfg.c_sw != 0.0) D_arm.compute_clover_field();

            auto t0 = Clock::now();
            auto res = hmc_trajectory_mg_multiscale(
                gauge_arm, lat, lcfg.mass, lcfg.wilson_r, ms_params,
                cdefl_arm, P, pc_arm, rng_arm);
            double dt = Dur(Clock::now() - t0).count();

            if (res.accepted) stats[ai].accept++;
            stats[ai].dH_sum += std::abs(res.dH);
            stats[ai].cg_sum += res.highmode_cg_iters;
            stats[ai].time_sum += dt;

            if (res.accepted && lcfg.c_sw != 0.0) D_arm.compute_clover_field();

            // === MG preconditioner refresh (only — inner force unchanged) ===
            bool do_rebuild = (ac.arm_rebuild_freq > 0 && (t+1) % ac.arm_rebuild_freq == 0);
            if (do_rebuild) {
                // Warm rebuild: TRLM inverse iteration for fine null space
                auto warm_vecs = mg_arm.null_vecs_l0;
                mg_arm = build_mg_hierarchy_warm(D_arm, mcfg.mg_levels, mcfg.block_size,
                    mcfg.k_null, mcfg.resolved_coarse_block(), 5, rng_mg_arm,
                    warm_vecs, mcfg.w_cycle, 3, 3);
                mg_arm.levels[0].op = [&D_arm](const Vec& s, Vec& d) {
                    D_arm.apply_DdagD(s, d);
                };
                // TRLM for coarse deflation (initial setup after rebuild)
                mg_arm.setup_sparse_coarse(
                    [&D_arm](const Vec& s, Vec& d){ D_arm.apply_DdagD(s, d); },
                    ndof, n_defl);
            } else if (res.accepted && ac.feast_fine_refresh) {
                // FEAST warm-start: use stale MG to precondition shifted solves,
                // recompute fine eigenvectors, rebuild P + Galerkin + coarse defl
                mg_arm.refresh_prolongator_feast(D_arm, scfg.feast_emax);
                // Also FEAST warm-start the coarse deflation
                mg_arm.sparse_Ac.setup_deflation(n_defl, 0, 100, 1e-10, "feast", scfg.feast_emax);
            } else if (res.accepted && ac.feast_coarse_evolve) {
                mg_arm.levels[0].Ac.build(D_arm, mg_arm.geo_prolongators[0]);
                mg_arm.rebuild_deeper_levels();
                mg_arm.sparse_Ac.setup_deflation(n_defl, 0, 100, 1e-10, "feast", scfg.feast_emax);
            } else if (res.accepted) {
                // Stale P — only rebuild Galerkin coarse ops
                mg_arm.levels[0].Ac.build(D_arm, mg_arm.geo_prolongators[0]);
                mg_arm.rebuild_deeper_levels();
            }

            std::cout << std::fixed
                      << std::setw(5) << t
                      << std::setw(8) << std::setprecision(4) << gauge_arm.avg_plaq()
                      << std::setw(10) << std::setprecision(4) << res.dH
                      << std::setw(4) << (res.accepted ? "Y" : "N")
                      << std::setw(6) << res.highmode_cg_iters
                      << std::setw(8) << std::setprecision(2) << dt
                      << "\n";
        }
        std::cout << "\n";
    }

    // === Summary ===
    std::cout << "=== Prolongator Study Summary (" << n_traj << " trajectories) ===\n";
    std::cout << std::setw(16) << "Arm"
              << std::setw(8) << "Acc"
              << std::setw(10) << "Avg|dH|"
              << std::setw(8) << "AvgCG"
              << std::setw(10) << "AvgTime"
              << std::setw(10) << "CG_ratio"
              << "\n";
    std::cout << std::string(62, '-') << "\n";
    for (int ai = 0; ai < n_arms; ai++) {
        std::cout << std::setw(16) << arms[ai].label
                  << std::setw(7) << std::fixed << std::setprecision(0)
                  << 100.0 * stats[ai].accept / n_traj << "%"
                  << std::setw(10) << std::scientific << std::setprecision(2)
                  << stats[ai].dH_sum / n_traj
                  << std::setw(8) << (int)(stats[ai].cg_sum / n_traj)
                  << std::setw(10) << std::fixed << std::setprecision(2)
                  << stats[ai].time_sum / n_traj << "s"
                  << std::setw(10) << std::setprecision(3)
                  << (stats[0].cg_sum > 0 ? stats[ai].cg_sum / stats[0].cg_sum : 0.0)
                  << "\n";
    }

    return 0;
}
