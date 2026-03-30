#include "mode_hmc_benchmark.h"
#include "dirac.h"
#include "hmc.h"
#include "multigrid.h"
#include "mg_builder.h"
#include "prolongator.h"
#include "coarse_op.h"
#include "eigensolver.h"
#include "feast_solver.h"
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

    VOUT(V_SUMMARY) << "=== HMC Physics Benchmark ===\n";
    VOUT(V_SUMMARY) << "L=" << L << "  DOF=" << lat.ndof
              << "  mass=" << mass << "  beta=" << beta << "\n";
    if (use_mg_multiscale) {
        VOUT(V_SUMMARY) << "n_outer=" << hcfg.n_outer << "  n_inner=" << hcfg.n_inner
                  << "  total_steps=" << hcfg.n_outer * hcfg.n_inner
                  << "  tau=" << std::fixed << std::setprecision(4) << tau << "\n";
    } else {
        VOUT(V_SUMMARY) << "n_steps=" << n_steps << "  dt=" << std::fixed
                  << std::setprecision(4) << dt << "  tau=" << tau << "\n";
    }
    VOUT(V_SUMMARY) << "solver=" << solver_desc << "\n";
    VOUT(V_SUMMARY) << "integrator=" << integrator_desc << "\n";
    VOUT(V_SUMMARY) << "therm=" << n_therm << "  traj=" << n_traj << "\n\n";

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
        VOUT(V_VERBOSE) << "--- Building MG hierarchy ---\n";
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
            VOUT(V_VERBOSE) << "  Coarse deflation: " << cdefl.eigvecs.size() << " vectors\n";
        }
        VOUT(V_VERBOSE) << "\n";
    }

    // === 1. Force verification ===
    VOUT(V_VERBOSE) << "--- Force Verification (L=" << L << ") ---\n";
    bool force_pass = verify_forces(gauge, beta, mass, wilson_r,
                                    scfg.max_iter, scfg.tol, lcfg.c_sw,
                                    use_eo, 1e-4, lcfg.mu_t);
    VOUT(V_VERBOSE) << "\n";

    // Rebind MG operator after force verification
    if (use_mg) {
        D_mg = std::make_unique<DiracOp>(lat, gauge, mass, wilson_r, lcfg.c_sw, lcfg.mu_t);
        mg->levels[0].op = D_mg->as_DdagD_op();
        mg->levels[0].Ac.build(*D_mg, mg->geo_prolongators[0]);
        mg->rebuild_deeper_levels();
    }

    // === 2. Thermalisation (plain CG, no MG — fast at heavy mass, avoids sparse coarse issues) ===
    VOUT(V_SUMMARY) << "--- Thermalisation (" << n_therm << " trajectories, plain CG) ---\n";
    HMCParams therm_params = params;
    therm_params.omelyan = false;
    for (int t = 0; t < n_therm; t++) {
        auto res = hmc_trajectory(gauge, lat, mass, wilson_r, therm_params, rng);
        if ((t+1) % 10 == 0 || t == n_therm - 1)
            VOUT(V_SUMMARY) << "  traj " << t+1 << "/" << n_therm
                      << "  <plaq>=" << std::fixed << std::setprecision(4)
                      << gauge.avg_plaq()
                      << "  dH=" << std::scientific << std::setprecision(3) << res.dH
                      << "  " << (res.accepted ? "Y" : "N") << "\n";
    }
    VOUT(V_SUMMARY) << "  Thermalised <plaq>=" << std::fixed << std::setprecision(6)
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
    VOUT(V_SUMMARY) << "--- Production (" << n_traj << " trajectories) ---\n";
    VOUT(V_SUMMARY) << std::setw(5) << "traj"
              << std::setw(8) << "Plaq"
              << std::setw(12) << "dH"
              << std::setw(4) << "A"
              << std::setw(6) << "CG"
              << std::setw(8) << "Time"
              << "\n";
    VOUT(V_SUMMARY) << std::string(43, '-') << "\n";

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

        VOUT(V_SUMMARY) << std::setw(5) << t
                  << std::setw(8) << std::fixed << std::setprecision(4) << gauge.avg_plaq()
                  << std::setw(12) << std::scientific << std::setprecision(4) << dH
                  << std::setw(4) << (accepted ? "Y" : "N")
                  << std::setw(6) << cg_iters
                  << std::setw(8) << std::fixed << std::setprecision(2) << wall
                  << "\n";
    }

    // === 4. Reversibility test ===
    VOUT(V_VERBOSE) << "\n--- Reversibility Test ---\n";
    if (use_mg_multiscale) {
        auto rev = reversibility_test_mg_multiscale(
            gauge, lat, mass, wilson_r, ms_params,
            cdefl, mg->geo_prolongators[0], mg_precond_fn, rng);
        VOUT(V_VERBOSE) << "  ||dU||/||U|| = " << std::scientific << std::setprecision(3)
                  << rev.gauge_delta << "\n";
        VOUT(V_VERBOSE) << "  ||dp||/||p|| = " << rev.mom_delta << "\n";
        VOUT(V_VERBOSE) << "  dH_fwd       = " << std::setprecision(6) << rev.dH_forward << "\n";
        VOUT(V_VERBOSE) << "  dH_bwd       = " << rev.dH_backward << "\n";
        VOUT(V_VERBOSE) << "  dH_fwd+bwd   = " << std::setprecision(3)
                  << rev.dH_forward + rev.dH_backward << "\n\n";
    } else {
        auto rev = reversibility_test_plain(gauge, lat, mass, wilson_r, params, rng, precond_ptr);
        VOUT(V_VERBOSE) << "  ||dU||/||U|| = " << std::scientific << std::setprecision(3)
                  << rev.gauge_delta << "\n";
        VOUT(V_VERBOSE) << "  ||dp||/||p|| = " << rev.mom_delta << "\n";
        VOUT(V_VERBOSE) << "  dH_fwd       = " << std::setprecision(6) << rev.dH_forward << "\n";
        VOUT(V_VERBOSE) << "  dH_bwd       = " << rev.dH_backward << "\n";
        VOUT(V_VERBOSE) << "  dH_fwd+bwd   = " << std::setprecision(3)
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

    VOUT(V_SUMMARY) << "=== BENCHMARK RESULTS L=" << L << " ===\n";
    VOUT(V_SUMMARY) << "L                   = " << L << "\n";
    VOUT(V_SUMMARY) << "beta                = " << std::fixed << std::setprecision(6) << beta << "\n";
    VOUT(V_SUMMARY) << "mass                = " << mass << "\n";
    VOUT(V_SUMMARY) << "solver              = " << solver_desc << "\n";
    VOUT(V_SUMMARY) << "integrator          = " << integrator_desc << "\n";
    VOUT(V_SUMMARY) << "n_therm             = " << n_therm << "\n";
    VOUT(V_SUMMARY) << "n_traj              = " << n_traj << "\n";
    if (use_mg_multiscale) {
        VOUT(V_SUMMARY) << "n_outer             = " << hcfg.n_outer << "\n";
        VOUT(V_SUMMARY) << "n_inner             = " << hcfg.n_inner << "\n";
    } else {
        VOUT(V_SUMMARY) << "n_steps             = " << n_steps << "\n";
        VOUT(V_SUMMARY) << "dt                  = " << std::setprecision(6) << dt << "\n";
    }
    VOUT(V_SUMMARY) << "acceptance_rate     = " << std::setprecision(6) << accept_rate << "\n";
    VOUT(V_SUMMARY) << "mean_dH             = " << std::scientific << std::setprecision(6) << mean_dH << "\n";
    VOUT(V_SUMMARY) << "mean_abs_dH         = " << mean_abs_dH << "\n";
    VOUT(V_SUMMARY) << "creutz_exp_neg_dH   = " << std::fixed << std::setprecision(6) << creutz << "\n";
    VOUT(V_SUMMARY) << "var_dH              = " << std::scientific << std::setprecision(6) << var_dH << "\n";
    VOUT(V_SUMMARY) << "n_positive_dH       = " << n_pos << "\n";
    VOUT(V_SUMMARY) << "n_negative_dH       = " << n_neg << "\n";
    VOUT(V_SUMMARY) << "mean_plaq           = " << std::fixed << std::setprecision(6) << mean_plaq << "\n";
    VOUT(V_SUMMARY) << "mean_cg_iters       = " << std::setprecision(1) << mean_cg << "\n";
    VOUT(V_SUMMARY) << "mean_time           = " << std::setprecision(4) << mean_time << "\n";
    VOUT(V_SUMMARY) << "force_check         = " << (force_pass ? "PASS" : "FAIL") << "\n";

    if (!force_pass) { std::cerr << "FAIL: force verification\n"; return 1; }
    return 0;
}

// =====================================================================
//  FEAST vs TRLM Eigenspace Construction Benchmark
//  On a sequence of evolved gauge configs, compare:
//    1. TRLM cold (20 inverse iterations from random) — baseline
//    2. FEAST warm (seeded from previous eigenspace) — test case
//  Measures: build time, FEAST convergence (solves/iters), eigenvalue
//  residuals, and MG preconditioner quality (CG iterations).
// =====================================================================
int run_feast_benchmark(GaugeField& gauge, const Lattice& lat,
                        const LatticeConfig& lcfg, const MGConfig& mcfg,
                        const SolverConfig& scfg, const HMCConfig& hcfg,
                        std::mt19937& rng)
{
    int L = lcfg.L;
    double mass = lcfg.mass;
    double wilson_r = lcfg.wilson_r;
    int k_null = mcfg.k_null;
    int block_size = mcfg.block_size;
    int cb = mcfg.resolved_coarse_block();
    int n_traj = hcfg.n_traj > 0 ? hcfg.n_traj : 10;
    double feast_emax = scfg.feast_emax > 0 ? scfg.feast_emax : 0.0; // 0 = auto

    VOUT(V_SUMMARY) << "=== FEAST vs TRLM Eigenspace Construction Benchmark ===\n";
    VOUT(V_SUMMARY) << "L=" << L << "  DOF=" << lat.ndof
              << "  mass=" << mass << "  beta=" << hcfg.beta << "\n";
    VOUT(V_SUMMARY) << "block=" << block_size << "  k_null=" << k_null
              << "  feast_emax=" << (feast_emax > 0 ? std::to_string(feast_emax) : "auto")
              << "\n";
    VOUT(V_SUMMARY) << "trajectories=" << n_traj << "  therm=" << hcfg.n_therm << "\n\n";

    HMCParams hmc_params;
    hmc_params.beta = hcfg.beta;
    hmc_params.tau = hcfg.tau;
    hmc_params.n_steps = hcfg.n_steps;
    hmc_params.cg_maxiter = scfg.max_iter;
    hmc_params.cg_tol = scfg.tol;
    hmc_params.c_sw = lcfg.c_sw;
    hmc_params.mu_t = lcfg.mu_t;

    // Thermalise
    if (hcfg.n_therm > 0) {
        VOUT(V_SUMMARY) << "--- Thermalisation (" << hcfg.n_therm << " trajectories) ---\n";
        for (int t = 0; t < hcfg.n_therm; t++) {
            hmc_trajectory(gauge, lat, mass, wilson_r, hmc_params, rng);
            if ((t+1) % 10 == 0 || t == hcfg.n_therm - 1)
                VOUT(V_SUMMARY) << "  traj " << t+1 << "  <plaq>=" << std::fixed
                          << std::setprecision(4) << gauge.avg_plaq() << "\n";
        }
        VOUT(V_SUMMARY) << "\n";
    }

    // Helper: compute max eigenvalue residual ||Av - λv|| / ||Av|| for null vecs
    auto max_residual = [&](const std::vector<Vec>& vecs, const DiracOp& D) -> double {
        OpApply A = [&D](const Vec& s, Vec& d) { D.apply_DdagD(s, d); };
        double max_res = 0;
        for (int i = 0; i < (int)vecs.size(); i++) {
            Vec Av(lat.ndof);
            A(vecs[i], Av);
            double rq = std::real(dot(vecs[i], Av));
            Vec r = Av;
            axpy(cx(-rq), vecs[i], r);
            double res = norm(r) / std::max(norm(Av), 1e-30);
            max_res = std::max(max_res, res);
        }
        return max_res;
    };

    // Helper: measure CG quality with a given set of null vectors
    std::mt19937 rng_mg(lcfg.seed + 111);
    auto measure_cg = [&](const std::vector<Vec>& null_vecs, const DiracOp& D) -> int {
        auto mg_test = build_mg_hierarchy(D, mcfg.mg_levels, block_size, k_null,
                                           cb, 0, rng_mg, mcfg.w_cycle, 3, 3, false,
                                           &null_vecs);
        mg_test.set_symmetric(0.8);
        auto precond = [&mg_test](const Vec& b) -> Vec { return mg_test.precondition(b); };
        OpApply A = [&D](const Vec& s, Vec& d) { D.apply_DdagD(s, d); };
        std::mt19937 rng_rhs(lcfg.seed + 777);
        Vec rhs = random_vec(lat.ndof, rng_rhs);
        auto res = cg_solve_precond(A, lat.ndof, rhs, precond, scfg.max_iter, scfg.tol);
        return res.iterations;
    };

    // =====================================================================
    //  Eigenvector tracking through MD trajectory
    //  Compare: stale, RR-every-step, forecast+RR
    //  This is a theoretical best-case study — how well CAN we track?
    // =====================================================================

    // Build initial eigenvectors with TRLM
    DiracOp D0(lat, gauge, mass, wilson_r, lcfg.c_sw, lcfg.mu_t);
    std::mt19937 rng_init(lcfg.seed + 111);

    // Use TRLM to get k_null true eigenpairs of D†D
    OpApply A0 = D0.as_DdagD_op();
    int n_ev = k_null;
    auto trlm_init = trlm_eigensolver(A0, lat.ndof, n_ev, 0, 200, 1e-10);
    if (!trlm_init.converged) {
        std::cerr << "Initial TRLM failed\n"; return 1;
    }

    VOUT(V_SUMMARY) << "Initial TRLM: " << n_ev << " eigenpairs, residual="
              << std::scientific << std::setprecision(2) << max_residual(trlm_init.eigvecs, D0)
              << "\n";
    VOUT(V_SUMMARY) << "  Eigenvalues:";
    for (int i = 0; i < n_ev; i++)
        VOUT(V_SUMMARY) << " " << std::scientific << std::setprecision(4) << trlm_init.eigvals[i];
    VOUT(V_SUMMARY) << "\n\n";

    // Generate momenta for the MD trajectory
    MomentumField mom(lat);
    mom.randomise(rng);
    int n_md_steps = hcfg.n_steps > 0 ? hcfg.n_steps : 20;
    double dt = hcfg.tau / n_md_steps;

    VOUT(V_SUMMARY) << "MD trajectory: " << n_md_steps << " steps, dt=" << std::fixed
              << std::setprecision(4) << dt << ", tau=" << hcfg.tau << "\n\n";

    // Three copies of eigenvectors for tracking
    auto evecs_stale = trlm_init.eigvecs;     // never updated
    auto evals_stale = trlm_init.eigvals;
    auto evecs_rr = trlm_init.eigvecs;        // RR at every step
    auto evals_rr = trlm_init.eigvals;
    auto evecs_fc = trlm_init.eigvecs;        // forecast + RR
    auto evals_fc = trlm_init.eigvals;

    // Forecast state for generator extrapolation
    EigenForecastState forecast;

    // Header
    VOUT(V_SUMMARY) << std::setw(5) << "step" << std::setw(8) << "plaq"
              << "  |  Stale              |  RR-every-step       |  Forecast+RR\n";
    VOUT(V_SUMMARY) << std::setw(13) << ""
              << "  |  max_res   ev0      |  max_res   ev0      |  max_res   ev0\n";
    VOUT(V_SUMMARY) << std::string(83, '-') << "\n";

    // Initial state
    VOUT(V_SUMMARY) << std::setw(5) << 0
              << std::setw(8) << std::fixed << std::setprecision(4) << gauge.avg_plaq()
              << "  | " << std::scientific << std::setprecision(2) << max_residual(evecs_stale, D0)
              << "  " << std::setprecision(4) << evals_stale[0]
              << "  | " << std::setprecision(2) << max_residual(evecs_rr, D0)
              << "  " << std::setprecision(4) << evals_rr[0]
              << "  | " << std::setprecision(2) << max_residual(evecs_fc, D0)
              << "  " << std::setprecision(4) << evals_fc[0]
              << "\n";

    for (int step = 0; step < n_md_steps; step++) {
        // Gauge update: U → exp(i dt π) U
        #pragma omp parallel for collapse(2) schedule(static)
        for (int mu = 0; mu < 2; mu++)
            for (int s = 0; s < lat.V; s++)
                gauge.U[mu][s] *= std::exp(cx(0, dt * mom.pi[mu][s]));

        DiracOp D(lat, gauge, mass, wilson_r, lcfg.c_sw, lcfg.mu_t);
        OpApply A = D.as_DdagD_op();

        // Update momentum with gauge force (leapfrog-like)
        std::array<RVec, 2> gf;
        gauge_force(gauge, hcfg.beta, gf);
        for (int mu = 0; mu < 2; mu++)
            for (int s = 0; s < lat.V; s++)
                mom.pi[mu][s] += dt * gf[mu][s];

        // --- Stale: measure quality only ---
        double res_stale = max_residual(evecs_stale, D);

        // --- RR every step ---
        auto rr_res = rr_evolve(A, evecs_rr, lat.ndof);
        evecs_rr = std::move(rr_res.eigvecs);
        evals_rr = std::move(rr_res.eigvals);
        double res_rr = rr_res.max_residual;

        // --- Forecast + RR ---
        // Step 1: If we have history, forecast the rotation and pre-rotate
        if (forecast.history_len > 0) {
            auto R_pred = forecast_rotation(forecast);
            if (!R_pred.empty())
                apply_rotation(evecs_fc, R_pred, lat.ndof);
        }
        // Step 2: RR to correct any prediction error
        auto rr_fc = rr_evolve(A, evecs_fc, lat.ndof);
        // Step 3: Extract generator and store in forecast state
        if (!rr_fc.rotation.empty()) {
            std::vector<Vec> H_cols;
            extract_generator(rr_fc.rotation, n_ev, H_cols);
            forecast.k = n_ev;
            forecast.H_history.push_back(H_cols);
            forecast.history_len++;
            if (forecast.history_len > EigenForecastState::max_history) {
                forecast.H_history.erase(forecast.H_history.begin());
                forecast.history_len = EigenForecastState::max_history;
            }
        }
        evecs_fc = std::move(rr_fc.eigvecs);
        evals_fc = std::move(rr_fc.eigvals);
        double res_fc = rr_fc.max_residual;

        VOUT(V_SUMMARY) << std::setw(5) << step + 1
                  << std::setw(8) << std::fixed << std::setprecision(4) << gauge.avg_plaq()
                  << "  | " << std::scientific << std::setprecision(2) << res_stale
                  << "  " << std::setprecision(4) << evals_stale[0]
                  << "  | " << std::setprecision(2) << res_rr
                  << "  " << std::setprecision(4) << evals_rr[0]
                  << "  | " << std::setprecision(2) << res_fc
                  << "  " << std::setprecision(4) << evals_fc[0]
                  << "\n";
    }

    // Final comparison: compute true eigenvalues with fresh TRLM
    DiracOp D_final(lat, gauge, mass, wilson_r, lcfg.c_sw, lcfg.mu_t);
    OpApply A_final = D_final.as_DdagD_op();
    auto trlm_final = trlm_eigensolver(A_final, lat.ndof, n_ev, 0, 200, 1e-10);

    VOUT(V_SUMMARY) << std::string(83, '-') << "\n";
    VOUT(V_SUMMARY) << "True eigenvalues (fresh TRLM):";
    for (int i = 0; i < n_ev; i++)
        VOUT(V_SUMMARY) << " " << std::scientific << std::setprecision(4) << trlm_final.eigvals[i];
    VOUT(V_SUMMARY) << "\n";

    VOUT(V_SUMMARY) << "\n=== TRACKING QUALITY SUMMARY ===\n";
    VOUT(V_SUMMARY) << "  Stale:         max_res=" << std::scientific << std::setprecision(2)
              << max_residual(evecs_stale, D_final)
              << "  ev0_err=" << std::abs(evals_stale[0] - trlm_final.eigvals[0]) << "\n";
    VOUT(V_SUMMARY) << "  RR/step:       max_res=" << max_residual(evecs_rr, D_final)
              << "  ev0_err=" << std::abs(evals_rr[0] - trlm_final.eigvals[0]) << "\n";
    VOUT(V_SUMMARY) << "  Forecast+RR:   max_res=" << max_residual(evecs_fc, D_final)
              << "  ev0_err=" << std::abs(evals_fc[0] - trlm_final.eigvals[0]) << "\n";

    // Also measure MG quality with each
    VOUT(V_SUMMARY) << "\n  MG quality (CG iters with each eigenspace):\n";
    VOUT(V_SUMMARY) << "    Stale:       CG=" << measure_cg(evecs_stale, D_final) << "\n";
    VOUT(V_SUMMARY) << "    RR/step:     CG=" << measure_cg(evecs_rr, D_final) << "\n";
    VOUT(V_SUMMARY) << "    Forecast+RR: CG=" << measure_cg(evecs_fc, D_final) << "\n";
    VOUT(V_SUMMARY) << "    Fresh TRLM:  CG=" << measure_cg(trlm_final.eigvecs, D_final) << "\n";

    return 0;
}
