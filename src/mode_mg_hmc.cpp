#include "mode_mg_hmc.h"
#include "config.h"
#include "dirac.h"
#include "hmc.h"
#include "multigrid.h"
#include "eigensolver.h"
#include "linalg.h"
#include "mg_builder.h"
#include "feast_solver.h"
#include "coarse_op.h"
#include "prolongator.h"
#include "solvers.h"
#include <iostream>
#include <iomanip>
#include <chrono>

int run_mg_hmc(GaugeField& gauge, const Lattice& lat,
               const LatticeConfig& lcfg, const MGConfig& mcfg,
               const SolverConfig& scfg, const HMCConfig& hcfg,
               std::mt19937& rng, bool revtest)
{
    if (mcfg.mg_levels < 2) {
        std::cerr << "--hmc-mg-multiscale requires --mg-levels >= 2\n";
        return 1;
    }

    using Clock = std::chrono::high_resolution_clock;
    using Dur = std::chrono::duration<double>;

    int n_traj = hcfg.n_traj > 0 ? hcfg.n_traj : 20;
    int n_defl = scfg.n_defl > 0 ? scfg.n_defl : hcfg.n_defl;
    int total_steps = hcfg.n_outer * hcfg.n_inner;

    std::cout << "=== MG Multi-Timescale HMC (Coarse-Grid Deflation) ===\n\n";
    std::cout << "L=" << lcfg.L << "  DOF=" << lat.ndof << "  mass=" << lcfg.mass
              << "  beta=" << hcfg.beta << "\n";
    std::cout << "MG levels=" << mcfg.mg_levels << "  block=" << mcfg.block_size
              << "  k_null=" << mcfg.k_null << "\n";
    std::cout << "n_outer=" << hcfg.n_outer << "  n_inner=" << hcfg.n_inner
              << "  total_steps=" << total_steps
              << "  tau=" << hcfg.tau << "\n";
    std::cout << "n_defl=" << n_defl << "  traj=" << n_traj << "\n";
    std::cout << "Monomials: KE=kinetic  SG=gauge  SF=fermion  dH=total\n\n";

    // --- Load thermalised config if available ---
    {
        std::string cfg = "gauge_L" + std::to_string(lcfg.L) + "_b"
            + std::to_string(hcfg.beta).substr(0,4) + "_t50.bin";
        if (gauge.load(cfg)) {
            std::cout << "--- Loaded thermalised config from " << cfg
                      << "  <plaq>=" << std::fixed << std::setprecision(4)
                      << gauge.avg_plaq() << " ---\n";
        } else {
            // Try t20 variant
            cfg = "gauge_L" + std::to_string(lcfg.L) + "_b"
                + std::to_string(hcfg.beta).substr(0,4) + "_t20.bin";
            if (gauge.load(cfg))
                std::cout << "--- Loaded config from " << cfg
                          << "  <plaq>=" << std::fixed << std::setprecision(4)
                          << gauge.avg_plaq() << " ---\n";
            else
                std::cout << "--- Using cold start (no saved config found) ---\n";
        }
    }

    // --- Build MG hierarchy ---
    std::cout << "--- Building MG hierarchy ---\n";
    DiracOp D_mg(lat, gauge, lcfg.mass, lcfg.wilson_r, lcfg.c_sw, lcfg.mu_t);
    std::mt19937 rng_mg(lcfg.seed + 111);
    auto mg = build_full_mg(D_mg, mcfg, scfg, rng_mg, n_defl, true);

    int cdim = mg.sparse_Ac.dim;
    std::cout << "  Coarse dim: " << cdim << "  n_defl: " << n_defl << "\n";

    // Extract coarse deflation state
    CoarseDeflState cdefl;
    cdefl.eigvecs = mg.sparse_Ac.defl_vecs;
    cdefl.eigvals = mg.sparse_Ac.defl_vals;

    std::cout << "  Coarse eigenvalues: ";
    for (int i = 0; i < std::min(n_defl, 8); i++)
        std::cout << std::scientific << std::setprecision(4)
                  << cdefl.eigvals[i] << " ";
    if (n_defl > 8) std::cout << "...";
    std::cout << "\n\n";

    // MG preconditioner
    std::function<Vec(const Vec&)> mg_precond = [&mg](const Vec& b) -> Vec {
        return mg.precondition(b);
    };

    // Prolongator reference
    auto& P = mg.geo_prolongators[0];

    // --- Run two HMC streams ---
    GaugeField gauge_std = gauge;
    GaugeField gauge_ms = gauge;
    std::mt19937 rng_std(lcfg.seed + 2000);
    std::mt19937 rng_ms(lcfg.seed + 2000);

    // Standard: MG-preconditioned CG at every step
    HMCParams std_params;
    std_params.beta = hcfg.beta;
    std_params.tau = hcfg.tau;
    std_params.n_steps = total_steps;
    std_params.cg_maxiter = scfg.max_iter;
    std_params.cg_tol = scfg.tol;
    std_params.use_mg = false;
    std_params.c_sw = lcfg.c_sw;
    std_params.mu_t = lcfg.mu_t;
    std_params.use_eo = hcfg.use_eo;

    // Multi-timescale
    MGMultiScaleParams ms_params;
    ms_params.beta = hcfg.beta;
    ms_params.tau = hcfg.tau;
    ms_params.n_outer = hcfg.n_outer;
    ms_params.n_inner = hcfg.n_inner;
    ms_params.cg_maxiter = scfg.max_iter;
    ms_params.cg_tol = scfg.tol;
    ms_params.c_sw = lcfg.c_sw;
    ms_params.mu_t = lcfg.mu_t;
    ms_params.use_eo = hcfg.use_eo;
    if (hcfg.omelyan) ms_params.outer_type = OuterIntegrator::Omelyan;
    if (hcfg.force_gradient) ms_params.outer_type = OuterIntegrator::FGI;
    ms_params.defl_refresh = hcfg.defl_refresh;

    std::string outer_name = "Leapfrog";
    if (hcfg.omelyan) outer_name = "Omelyan (2MN)";
    if (hcfg.force_gradient) outer_name = "FGI (MILC PQPQP, 4th order)";
    std::cout << "Outer: " << outer_name << "  Inner: Leapfrog (gauge+lowmode)\n\n";

    // --- Reversibility test ---
    if (revtest) {
        std::cout << "--- Reversibility Test ---\n";
        std::cout << "  Running forward -> negate pi -> backward for each integrator...\n\n";

        // Test all configured integrator types
        struct IntCfg { OuterIntegrator t; const char* name; };
        std::vector<IntCfg> configs = {
            {OuterIntegrator::Leapfrog, "Leapfrog"},
            {OuterIntegrator::Omelyan, "Omelyan"},
            {OuterIntegrator::FGI, "FGI (MILC)"},
        };

        std::cout << std::setw(15) << "Integrator"
                  << std::setw(14) << "||dU||/||U||"
                  << std::setw(14) << "||dp||/||p||"
                  << std::setw(12) << "dH_fwd"
                  << std::setw(12) << "dH_bwd"
                  << std::setw(14) << "dH_fwd+bwd"
                  << std::setw(8) << "CG"
                  << "\n";

        for (auto& cfg : configs) {
            MGMultiScaleParams rp = ms_params;
            rp.outer_type = cfg.t;
            std::mt19937 rng_rev(lcfg.seed + 9999);
            auto res = reversibility_test_mg_multiscale(
                gauge, lat, lcfg.mass, lcfg.wilson_r, rp, cdefl, P, mg_precond, rng_rev);

            std::cout << std::setw(15) << cfg.name
                      << std::setw(14) << std::scientific << std::setprecision(3) << res.gauge_delta
                      << std::setw(14) << res.mom_delta
                      << std::setw(12) << std::setprecision(4) << res.dH_forward
                      << std::setw(12) << res.dH_backward
                      << std::setw(14) << std::setprecision(3) << res.dH_forward + res.dH_backward
                      << std::setw(8) << res.total_cg
                      << "\n";
        }

        std::cout << "\n  A time-reversible integrator should give:\n"
                  << "    ||dU||/||U|| ~ 1e-15 (machine epsilon)\n"
                  << "    ||dp||/||p|| ~ 1e-15 (machine epsilon)\n"
                  << "    dH_fwd + dH_bwd ~ 0 (exact cancellation)\n\n";
        return 0;
    }

    // Standard also uses MG preconditioner for fair comparison
    std::function<Vec(const Vec&)> std_precond = mg_precond;

    int std_accept = 0, ms_accept = 0;
    double std_dH_sum = 0, ms_dH_sum = 0;
    int std_cg_sum = 0, ms_cg_sum = 0;
    double std_time_sum = 0, ms_time_sum = 0;
    double ms_low_time_sum = 0, ms_high_time_sum = 0;
    int ms_low_evals_sum = 0;
    EigenForecastState eigen_forecast;

    // --- Standard HMC header ---
    std::cout << "--- Standard HMC (reference) ---\n";
    std::cout << std::setw(5) << "traj"
              << std::setw(8) << "Plaq"
              << std::setw(9) << "dKE"
              << std::setw(9) << "dSG"
              << std::setw(9) << "dSF"
              << std::setw(10) << "dH"
              << std::setw(4) << "A"
              << std::setw(6) << "CG"
              << std::setw(8) << "Time"
              << "\n";
    std::cout << std::string(68, '-') << "\n";

    // --- MS HMC header (printed after std block) ---

    // Storage for MS results to print after std
    struct MSRow {
        MGMultiScaleResult res;
        double t_ms, plaq;
    };
    std::vector<MSRow> ms_rows;

    for (int t = 0; t < n_traj; t++) {
        // --- Standard HMC (MG-preconditioned) ---
        auto t0_std = Clock::now();
        auto res_std = hmc_trajectory(gauge_std, lat, lcfg.mass, lcfg.wilson_r,
                                       std_params, rng_std, &std_precond);
        double t_std = Dur(Clock::now() - t0_std).count();
        if (res_std.accepted) std_accept++;
        std_dH_sum += std::abs(res_std.dH);
        std_cg_sum += res_std.total_cg_iters;
        std_time_sum += t_std;

        // --- MG Multi-timescale HMC ---
        auto t0_ms = Clock::now();
        auto res_ms = hmc_trajectory_mg_multiscale(gauge_ms, lat, lcfg.mass, lcfg.wilson_r,
                                                    ms_params, cdefl, P,
                                                    mg_precond, rng_ms,
                                                    hcfg.eigen_forecast ? &eigen_forecast : nullptr);
        double t_ms = Dur(Clock::now() - t0_ms).count();
        if (res_ms.accepted) ms_accept++;
        ms_dH_sum += std::abs(res_ms.dH);
        ms_cg_sum += res_ms.highmode_cg_iters;
        ms_time_sum += t_ms;
        ms_low_time_sum += res_ms.lowmode_time;
        ms_high_time_sum += res_ms.highmode_time;
        ms_low_evals_sum += res_ms.lowmode_force_evals;

        ms_rows.push_back({res_ms, t_ms, gauge_ms.avg_plaq()});

        // Evolve coarse deflation after MS trajectory
        // Rebuild sparse Ac for new gauge + RR evolve
        if (res_ms.accepted) {
            DiracOp D_new(lat, gauge_ms, lcfg.mass, lcfg.wilson_r, lcfg.c_sw, lcfg.mu_t);
            OpApply A_new = [&D_new](const Vec& s, Vec& d) { D_new.apply_DdagD(s, d); };
            mg.sparse_Ac.build(P, A_new, D_new.lat.ndof);
            evolve_coarse_deflation(cdefl, mg.sparse_Ac,
                                   hcfg.eigen_forecast ? &eigen_forecast : nullptr);
        }

        // Periodically do fresh TRLM
        if (hcfg.fresh_period > 0 && (t+1) % hcfg.fresh_period == 0) {
            mg.sparse_Ac.setup_deflation(n_defl);
            cdefl.eigvecs = mg.sparse_Ac.defl_vecs;
            cdefl.eigvals = mg.sparse_Ac.defl_vals;
            if (hcfg.eigen_forecast) eigen_forecast.reset(); // fresh TRLM = discontinuity
        }

        // Print standard HMC row
        std::cout << std::fixed;
        std::cout << std::setw(5) << t
                  << std::setw(8) << std::setprecision(4) << gauge_std.avg_plaq()
                  << std::setw(9) << std::setprecision(3) << res_std.dKE
                  << std::setw(9) << std::setprecision(3) << res_std.dSG
                  << std::setw(9) << std::setprecision(3) << res_std.dSF
                  << std::setw(10) << std::setprecision(4) << res_std.dH
                  << std::setw(4) << (res_std.accepted ? "Y" : "N")
                  << std::setw(6) << res_std.total_cg_iters
                  << std::setw(8) << std::setprecision(2) << t_std
                  << "\n";
    }

    // --- Print MS block ---
    std::cout << "\n--- MG Multi-timescale ---\n";
    std::cout << std::setw(5) << "traj"
              << std::setw(8) << "Plaq"
              << std::setw(9) << "dKE"
              << std::setw(9) << "dSG"
              << std::setw(9) << "dSF"
              << std::setw(10) << "dH"
              << std::setw(4) << "A"
              << std::setw(6) << "CG"
              << std::setw(6) << "LEv"
              << std::setw(8) << "Time"
              << "\n";
    std::cout << std::string(74, '-') << "\n";
    for (int t = 0; t < n_traj; t++) {
        auto& r = ms_rows[t];
        std::cout << std::fixed;
        std::cout << std::setw(5) << t
                  << std::setw(8) << std::setprecision(4) << r.plaq
                  << std::setw(9) << std::setprecision(3) << r.res.dKE
                  << std::setw(9) << std::setprecision(3) << r.res.dSG
                  << std::setw(9) << std::setprecision(3) << r.res.dSF
                  << std::setw(10) << std::setprecision(4) << r.res.dH
                  << std::setw(4) << (r.res.accepted ? "Y" : "N")
                  << std::setw(6) << r.res.highmode_cg_iters
                  << std::setw(6) << r.res.lowmode_force_evals
                  << std::setw(8) << std::setprecision(2) << r.t_ms
                  << "\n";
    }

    std::cout << "\n=== Summary over " << n_traj << " trajectories ===\n";
    std::cout << "  Standard MG-HMC (n_steps=" << total_steps << "):\n";
    std::cout << "    Accept rate:  " << std::fixed << std::setprecision(1)
              << 100.0 * std_accept / n_traj << "%\n";
    std::cout << "    Avg |dH|:     " << std::scientific << std::setprecision(3)
              << std_dH_sum / n_traj << "\n";
    std::cout << "    Avg CG iters: " << std_cg_sum / n_traj << "\n";
    std::cout << "    Avg wall time:" << std::fixed << std::setprecision(3)
              << std_time_sum / n_traj << "s\n";

    std::cout << "\n  MG Multi-timescale (n_outer=" << hcfg.n_outer
              << " n_inner=" << hcfg.n_inner << " n_defl=" << n_defl << "):\n";
    std::cout << "    Accept rate:  " << std::fixed << std::setprecision(1)
              << 100.0 * ms_accept / n_traj << "%\n";
    std::cout << "    Avg |dH|:     " << std::scientific << std::setprecision(3)
              << ms_dH_sum / n_traj << "\n";
    std::cout << "    Avg CG iters: " << ms_cg_sum / n_traj
              << " (" << hcfg.n_outer + 1 << " outer + 2 H evals)\n";
    std::cout << "    Low-mode evals:" << ms_low_evals_sum / n_traj << " avg/traj\n";
    std::cout << "    Avg wall time:" << std::fixed << std::setprecision(3)
              << ms_time_sum / n_traj << "s"
              << " (low=" << ms_low_time_sum / n_traj
              << "s high=" << ms_high_time_sum / n_traj << "s)\n";

    if (std_time_sum > 0) {
        std::cout << "\n  CG ratio (MS/Std): " << std::setprecision(2)
                  << (double)ms_cg_sum / std_cg_sum << "x\n";
        std::cout << "  Wall speedup: " << std::setprecision(2)
                  << std_time_sum / ms_time_sum << "x\n";
    }

    return 0;
}
