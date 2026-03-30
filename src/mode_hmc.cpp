#include "mode_hmc.h"
#include "dirac.h"
#include "hmc.h"
#include "eigensolver.h"
#include "solvers.h"
#include "linalg.h"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>
#include <vector>
#include <cmath>

using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double>;

// -----------------------------------------------------------------
//  Force verification mode
// -----------------------------------------------------------------
int run_verify_forces(GaugeField& gauge, const LatticeConfig& lcfg,
                      const SolverConfig& scfg, const HMCConfig& hcfg)
{
    std::cout << "=== Force Verification Mode ===\n";
    std::cout << "c_sw=" << lcfg.c_sw << "  mu_t=" << lcfg.mu_t
              << "  even-odd=" << (hcfg.use_eo ? "yes" : "no")
              << "  beta=" << hcfg.beta << "\n\n";
    bool pass = verify_forces(gauge, hcfg.beta, lcfg.mass, lcfg.wilson_r,
                              scfg.max_iter, scfg.tol, lcfg.c_sw, hcfg.use_eo, 1e-4, lcfg.mu_t);
    return pass ? 0 : 1;
}

// -----------------------------------------------------------------
//  Standard HMC mode
// -----------------------------------------------------------------
int run_hmc_mode(GaugeField& gauge, const Lattice& lat,
                 const LatticeConfig& lcfg, const SolverConfig& scfg,
                 const HMCConfig& hcfg, std::mt19937& rng)
{
    std::cout << "=== HMC Mode ===\n";
    std::cout << "beta=" << hcfg.beta << "  tau=" << hcfg.tau
              << "  steps=" << hcfg.n_steps << "  dt=" << hcfg.tau/hcfg.n_steps << "\n";
    std::cout << "trajectories=" << hcfg.n_traj << "  therm=" << hcfg.n_therm
              << "  save_every=" << hcfg.save_every << "\n";
    bool has_ld = (hcfg.use_eo && lcfg.c_sw != 0.0);
    std::cout << "Monomials: KE=kinetic  SG=gauge  SF=fermion";
    if (has_ld) std::cout << "  LD=log-det";
    std::cout << "  dH=total\n\n";

    HMCParams params;
    params.beta = hcfg.beta;
    params.tau = hcfg.tau;
    params.n_steps = hcfg.n_steps;
    params.cg_maxiter = scfg.max_iter;
    params.cg_tol = scfg.tol;
    params.use_mg = false;
    params.c_sw = lcfg.c_sw;
    params.mu_t = lcfg.mu_t;
    params.use_eo = hcfg.use_eo;

    // Eigenspace tracking (chronological x0 + Ritz harvesting)
    TrackingState tracking_state;
    TrackingState* tracking = nullptr;
    if (hcfg.enable_tracking) {
        tracking_state.n_ritz = hcfg.tracking_n_ritz;
        tracking_state.pool_capacity = hcfg.tracking_pool_cap;
        tracking_state.n_ev = hcfg.tracking_n_ev;
        tracking = &tracking_state;
        std::cout << "Tracking: chrono-x0 + " << hcfg.tracking_n_ritz << " Ritz/solve\n";
    }

    int n_accept = 0;
    int total_traj = hcfg.n_therm + hcfg.n_traj;

    std::cout << std::setw(5) << "Traj"
              << std::setw(8) << "Plaq"
              << std::setw(9) << "dKE"
              << std::setw(9) << "dSG"
              << std::setw(9) << "dSF";
    if (has_ld) std::cout << std::setw(9) << "dLD";
    std::cout << std::setw(10) << "dH"
              << std::setw(4) << "A"
              << std::setw(7) << "Rate"
              << std::setw(6) << "CG"
              << std::setw(8) << "Time"
              << "\n";
    int hdr_width = 61 + (has_ld ? 9 : 0);
    std::cout << std::string(hdr_width, '-') << "\n";

    int saved_count = 0;
    for (int traj = 0; traj < total_traj; traj++) {
        auto t0 = Clock::now();
        auto result = hmc_trajectory(gauge, lat, lcfg.mass, lcfg.wilson_r, params, rng,
                                      nullptr, tracking);
        double dt_traj = Duration(Clock::now() - t0).count();

        if (traj >= hcfg.n_therm) n_accept += result.accepted;
        int measurement_traj = (traj >= hcfg.n_therm) ? (traj - hcfg.n_therm + 1) : 0;
        double rate = (measurement_traj > 0) ? (double)n_accept / measurement_traj : 0.0;

        std::cout << std::fixed;
        std::cout << std::setw(5) << traj
                  << std::setw(8) << std::setprecision(4) << gauge.avg_plaq()
                  << std::setw(9) << std::setprecision(3) << result.dKE
                  << std::setw(9) << std::setprecision(3) << result.dSG
                  << std::setw(9) << std::setprecision(3) << result.dSF;
        if (has_ld) std::cout << std::setw(9) << std::setprecision(3) << result.dLD;
        std::cout << std::setw(10) << std::setprecision(4) << result.dH
                  << std::setw(4) << (result.accepted ? "Y" : "N")
                  << std::setw(6) << std::setprecision(0)
                  << (traj >= hcfg.n_therm ? 100.0*rate : 0.0) << "%"
                  << std::setw(6) << result.total_cg_iters
                  << std::setw(8) << std::setprecision(2) << dt_traj;
        if (traj < hcfg.n_therm) std::cout << " [therm]";
        std::cout << "\n";

        if (traj >= hcfg.n_therm && hcfg.save_every > 0 &&
            (traj - hcfg.n_therm) % hcfg.save_every == 0) {
            std::string fname = hcfg.save_prefix + "_L" + std::to_string(lcfg.L)
                + "_b" + std::to_string((int)(hcfg.beta*100))
                + "_m" + std::to_string((int)(lcfg.mass*10000))
                + "_" + std::to_string(saved_count) + ".bin";
            save_gauge(gauge, hcfg.beta, lcfg.mass, fname);
            std::cout << "  -> saved " << fname << "\n";
            saved_count++;
        }
    }

    std::cout << "\n=== HMC Summary ===\n";
    std::cout << "Final <plaq> = " << std::fixed << std::setprecision(6) << gauge.avg_plaq() << "\n";
    std::cout << "Acceptance rate (post-therm): " << std::fixed << std::setprecision(1)
              << 100.0 * n_accept / hcfg.n_traj << "%\n";
    std::cout << "Configs saved: " << saved_count << "\n";

    return 0;
}

// -----------------------------------------------------------------
//  Fine-grid multi-timescale HMC
// -----------------------------------------------------------------
int run_multiscale_hmc(GaugeField& gauge, const Lattice& lat,
                       const LatticeConfig& lcfg, const MGConfig& mcfg,
                       const SolverConfig& scfg, const HMCConfig& hcfg,
                       std::mt19937& rng)
{
    using Clock = std::chrono::high_resolution_clock;
    using Dur = std::chrono::duration<double>;

    int n_traj = hcfg.n_traj > 0 ? hcfg.n_traj : 20;
    int total_steps = hcfg.n_outer * hcfg.n_inner;

    std::cout << "=== Multi-Timescale HMC Test ===\n\n";
    std::cout << "L=" << lcfg.L << "  DOF=" << lat.ndof << "  mass=" << lcfg.mass
              << "  beta=" << hcfg.beta << "\n";
    std::cout << "n_outer=" << hcfg.n_outer << "  n_inner=" << hcfg.n_inner
              << "  total_steps=" << total_steps
              << "  tau=" << hcfg.tau << "\n";
    std::cout << "n_defl=" << hcfg.n_defl
              << "  fresh_period=" << hcfg.fresh_period
              << "  traj=" << n_traj << "\n";
    std::cout << "Monomials: KE=kinetic  SG=gauge  SF=fermion  dH=total\n\n";

    // --- Compute initial eigenvectors ---
    std::cout << "--- Computing " << hcfg.n_defl << " eigenvectors of D†D ---\n";
    DiracOp D_init(lat, gauge, lcfg.mass, lcfg.wilson_r, lcfg.c_sw, lcfg.mu_t);
    OpApply A_init = [&D_init](const Vec& s, Vec& d) { D_init.apply_DdagD(s, d); };
    auto trlm = trlm_eigensolver(A_init, lat.ndof, hcfg.n_defl,
                                  std::min(2*hcfg.n_defl + 10, lat.ndof),
                                  100, 1e-10);
    DeflationState defl;
    defl.eigvecs = std::move(trlm.eigvecs);
    defl.eigvals = std::move(trlm.eigvals);
    defl.valid = true;
    defl.update_cache(D_init);

    std::cout << "  Eigenvalues: ";
    for (int i = 0; i < std::min(hcfg.n_defl, 8); i++)
        std::cout << std::scientific << std::setprecision(4) << defl.eigvals[i] << " ";
    if (hcfg.n_defl > 8) std::cout << "...";
    std::cout << "\n\n";

    // --- Run two HMC streams: standard vs multi-timescale ---
    GaugeField gauge_std = gauge;
    GaugeField gauge_ms = gauge;
    std::mt19937 rng_std(lcfg.seed + 1000);
    std::mt19937 rng_ms(lcfg.seed + 1000);  // same seed for fair comparison

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

    MultiScaleParams ms_params;
    ms_params.beta = hcfg.beta;
    ms_params.tau = hcfg.tau;
    ms_params.n_outer = hcfg.n_outer;
    ms_params.n_inner = hcfg.n_inner;
    ms_params.cg_maxiter = scfg.max_iter;
    ms_params.cg_tol = scfg.tol;
    ms_params.c_sw = lcfg.c_sw;
    ms_params.mu_t = lcfg.mu_t;
    ms_params.use_eo = hcfg.use_eo;

    int std_accept = 0, ms_accept = 0;
    double std_dH_sum = 0, ms_dH_sum = 0;
    int std_cg_sum = 0, ms_cg_sum = 0;
    double std_time_sum = 0, ms_time_sum = 0;
    double ms_low_time_sum = 0, ms_high_time_sum = 0;
    int ms_low_evals_sum = 0;

    // --- Standard HMC block ---
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

    struct MSRowFG {
        MultiScaleResult res;
        double t_ms, plaq;
    };
    std::vector<MSRowFG> ms_rows;

    for (int t = 0; t < n_traj; t++) {
        // --- Standard HMC ---
        auto t0_std = Clock::now();
        auto res_std = hmc_trajectory(gauge_std, lat, lcfg.mass, lcfg.wilson_r,
                                       std_params, rng_std);
        double t_std = Dur(Clock::now() - t0_std).count();
        if (res_std.accepted) std_accept++;
        std_dH_sum += std::abs(res_std.dH);
        std_cg_sum += res_std.total_cg_iters;
        std_time_sum += t_std;

        // --- Multi-timescale HMC ---
        auto t0_ms = Clock::now();
        auto res_ms = hmc_trajectory_multiscale(gauge_ms, lat, lcfg.mass, lcfg.wilson_r,
                                                 ms_params, defl, rng_ms);
        double t_ms = Dur(Clock::now() - t0_ms).count();
        if (res_ms.accepted) ms_accept++;
        ms_dH_sum += std::abs(res_ms.dH);
        ms_cg_sum += res_ms.highmode_cg_iters;
        ms_time_sum += t_ms;
        ms_low_time_sum += res_ms.lowmode_time;
        ms_high_time_sum += res_ms.highmode_time;
        ms_low_evals_sum += res_ms.lowmode_force_evals;

        ms_rows.push_back({res_ms, t_ms, gauge_ms.avg_plaq()});

        // Evolve deflation state after multi-timescale trajectory
        DiracOp D_new(lat, gauge_ms, lcfg.mass, lcfg.wilson_r, lcfg.c_sw, lcfg.mu_t);
        bool do_fresh = (hcfg.fresh_period > 0) && ((t+1) % hcfg.fresh_period == 0);
        evolve_deflation_state(defl, D_new, do_fresh);

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
    std::cout << "\n--- Multi-timescale (fine-grid deflation) ---\n";
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
    std::cout << "  Standard HMC (n_steps=" << total_steps << "):\n";
    std::cout << "    Accept rate:  " << std::fixed << std::setprecision(1)
              << 100.0 * std_accept / n_traj << "%\n";
    std::cout << "    Avg |dH|:     " << std::scientific << std::setprecision(3)
              << std_dH_sum / n_traj << "\n";
    std::cout << "    Avg CG iters: " << std_cg_sum / n_traj << "\n";
    std::cout << "    Avg wall time:" << std::fixed << std::setprecision(3)
              << std_time_sum / n_traj << "s\n";

    std::cout << "\n  Multi-timescale HMC (n_outer=" << hcfg.n_outer
              << " n_inner=" << hcfg.n_inner << " n_defl=" << hcfg.n_defl << "):\n";
    std::cout << "    Accept rate:  " << std::fixed << std::setprecision(1)
              << 100.0 * ms_accept / n_traj << "%\n";
    std::cout << "    Avg |dH|:     " << std::scientific << std::setprecision(3)
              << ms_dH_sum / n_traj << "\n";
    std::cout << "    Avg CG iters: " << ms_cg_sum / n_traj
              << " (high-mode only, " << hcfg.n_outer + 1 << " solves + 2 H evals)\n";
    std::cout << "    Low-mode evals:" << ms_low_evals_sum / n_traj << " avg/traj\n";
    std::cout << "    Avg wall time:" << std::fixed << std::setprecision(3)
              << ms_time_sum / n_traj << "s"
              << " (low=" << ms_low_time_sum / n_traj
              << "s high=" << ms_high_time_sum / n_traj << "s)\n";

    double cg_ratio = (double)ms_cg_sum / std_cg_sum;
    double time_ratio = ms_time_sum / std_time_sum;
    std::cout << "\n  CG iter ratio (MS/Std): " << std::fixed << std::setprecision(2)
              << cg_ratio << "x\n";
    std::cout << "  Wall time ratio (MS/Std): " << std::setprecision(2)
              << time_ratio << "x\n";
    std::cout << "  Wall time speedup: " << std::setprecision(2)
              << 1.0 / time_ratio << "x\n";

    return 0;
}
