#include "mode_studies.h"
#include "config.h"
#include "hmc_utils.h"
#include "mg_builder.h"
#include "dirac.h"
#include "hmc.h"
#include "multigrid.h"
#include "eigensolver.h"
#include "linalg.h"
#include "solvers.h"
#include "prolongator.h"
#include "coarse_op.h"
#include "feast_solver.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <memory>

using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double>;

// =====================================================================
//  Sparse coarse eigenvector evolution study
// =====================================================================
int run_sparse_coarse_study(GaugeField& gauge, const Lattice& lat,
                            const LatticeConfig& lcfg, const MGConfig& mcfg,
                            const SolverConfig& scfg, const HMCConfig& hcfg,
                            const StudyConfig& study, std::mt19937& rng) {
    if (mcfg.mg_levels < 2) {
        std::cerr << "--test-sparse-coarse requires --mg-levels >= 2\n";
        return 1;
    }

    int n_defl = scfg.n_defl > 0 ? scfg.n_defl : 16;
    int ndof = lat.ndof;
    int coarse_dim_est = (lcfg.L/mcfg.block_size) * (lcfg.L/mcfg.block_size) * mcfg.k_null;

    std::cout << "=== Sparse Coarse Eigenvector Evolution Study ===\n\n";
    std::cout << "L=" << lcfg.L << "  DOF=" << ndof
              << "  coarse_dim=" << coarse_dim_est
              << "  n_defl=" << n_defl << "\n";

    // --- Thermalise if requested (save/load gauge configs) ---
    if (hcfg.n_therm > 0)
        load_or_thermalise(gauge, lat, lcfg, mcfg, scfg, hcfg, rng, n_defl);

    // --- Sweep over coarse problem difficulty ---
    // Smaller blocks + more null vectors = bigger coarse dim = harder coarse solve
    struct CoarseConfig {
        int blk;    // block size
        int knull;  // null vectors per block
        const char* label;
    };
    std::vector<CoarseConfig> configs = {
        {4, 4, "b4k4"},   // coarse_dim = (L/4)^2 * 4  (baseline)
        {4, 8, "b4k8"},   // coarse_dim = (L/4)^2 * 8  (2x bigger)
        {2, 4, "b2k4"},   // coarse_dim = (L/2)^2 * 4  (16x bigger)
        {2, 8, "b2k8"},   // coarse_dim = (L/2)^2 * 8  (32x bigger)
    };

    int galerkin_period = 5;
    int n_rhs = 5;

    // CG solve returning {iters, wall_time}
    struct SolveResult { int iters; double time; };
    auto solve_timed = [](const SparseCoarseOp& op, const Vec& b,
                           bool use_defl) -> SolveResult {
        auto t0 = Clock::now();
        int dim = op.dim;
        double bnorm = norm(b);
        if (bnorm < 1e-30) return {0, 0.0};
        Vec x = zeros(dim);
        if (use_defl) {
            for (int i = 0; i < (int)op.defl_vecs.size(); i++) {
                if (op.defl_vals[i] > 1e-14) {
                    cx coeff = dot(op.defl_vecs[i], b) / op.defl_vals[i];
                    axpy(coeff, op.defl_vecs[i], x);
                }
            }
        }
        Vec Ax(dim);
        op.apply_to(x, Ax);
        Vec r(dim);
        for (int i = 0; i < dim; i++) r[i] = b[i] - Ax[i];
        Vec p = r;
        cx rr_val = dot(r, r);
        int iter = 0;
        while (iter < 2000) {
            Vec Ap(dim);
            op.apply_to(p, Ap);
            cx pAp = dot(p, Ap);
            if (std::abs(pAp) < 1e-30) break;
            cx alpha = rr_val / pAp;
            for (int i = 0; i < dim; i++) {
                x[i] += alpha * p[i];
                r[i] -= alpha * Ap[i];
            }
            iter++;
            if (norm(r) / bnorm < 1e-12) break;
            cx rr_new = dot(r, r);
            cx beta = rr_new / rr_val;
            rr_val = rr_new;
            for (int i = 0; i < dim; i++)
                p[i] = r[i] + beta * p[i];
        }
        double elapsed = Duration(Clock::now() - t0).count();
        return {iter, elapsed};
    };

    for (auto& cfg : configs) {
        // Check block size divides L
        if (lcfg.L % cfg.blk != 0) {
            std::cout << "\n--- Skipping " << cfg.label
                      << ": block " << cfg.blk << " does not divide L=" << lcfg.L << " ---\n";
            continue;
        }
        int nb = lcfg.L / cfg.blk;
        int cdim_est = nb * nb * cfg.knull;

        std::cout << "\n==========================================================\n";
        std::cout << "  Config: " << cfg.label
                  << "  block=" << cfg.blk << "x" << cfg.blk
                  << "  k_null=" << cfg.knull
                  << "  coarse_dim=" << cdim_est << "\n";
        std::cout << "  Galerkin every " << galerkin_period << " steps, RR every step"
                  << ", " << study.n_steps << " configs, " << n_rhs << " RHS/step"
                  << ", n_defl=" << n_defl << "\n";
        std::cout << "==========================================================\n";

        // Build MG hierarchy for this config
        DiracOp D(lat, gauge, lcfg.mass, lcfg.wilson_r, lcfg.c_sw, lcfg.mu_t);
        OpApply A = [&D](const Vec& s, Vec& d){ D.apply_DdagD(s, d); };
        std::mt19937 rng_mg(lcfg.seed + 111);
        auto mg = build_mg_hierarchy(D, mcfg.mg_levels, cfg.blk, cfg.knull,
                                      mcfg.resolved_coarse_block(), 20, rng_mg, mcfg.w_cycle,
                                      3, 3, true);
        mg.setup_sparse_coarse(A, ndof, n_defl);
        int cdim = mg.sparse_Ac.dim;
        std::cout << "  Actual coarse dim: " << cdim << "\n";

        auto& P_ev = mg.geo_prolongators[0];

        // Evolution study
        GaugeField gauge_ev = gauge;
        auto D_ev = std::make_unique<DiracOp>(lat, gauge_ev, lcfg.mass, lcfg.wilson_r, lcfg.c_sw, lcfg.mu_t);
        OpApply A_ev = [&D_ev](const Vec& s, Vec& d){ D_ev->apply_DdagD(s, d); };
        std::mt19937 rng_ev(lcfg.seed + 777);

        SparseCoarseOp sac;
        sac.build(P_ev, A_ev, D_ev->lat.ndof);
        sac.setup_deflation(n_defl);

        std::cout << "\n" << std::setw(5) << "step"
                  << std::setw(10) << "Defl_it"
                  << std::setw(10) << "NoDfl_it"
                  << std::setw(10) << "Saved%"
                  << std::setw(12) << "Defl_wall"
                  << std::setw(12) << "NoDfl_wall"
                  << std::setw(10) << "Speedup"
                  << std::setw(10) << "Build_t"
                  << std::setw(10) << "RR_t"
                  << std::setw(6) << "Gal?"
                  << "\n";

        int total_defl_it = 0, total_nodfl_it = 0;
        double total_defl_wall = 0, total_nodfl_wall = 0;
        double total_build_t = 0, total_rr_t = 0;

        for (int step = 0; step < study.n_steps; step++) {
            MomentumField mom(lat);
            mom.randomise(rng_ev);

            // Apply gauge update
            for (int mu = 0; mu < 2; mu++)
                for (int s = 0; s < lat.V; s++)
                    gauge_ev.U[mu][s] *= std::exp(cx(0, study.eps * mom.pi[mu][s]));
            D_ev = std::make_unique<DiracOp>(lat, gauge_ev, lcfg.mass, lcfg.wilson_r, lcfg.c_sw, lcfg.mu_t);
            A_ev = [&D_ev](const Vec& s, Vec& d){ D_ev->apply_DdagD(s, d); };

            // Galerkin rebuild only every galerkin_period steps
            double t_build = 0;
            bool did_galerkin = (step % galerkin_period == 0);
            if (did_galerkin) {
                auto t0 = Clock::now();
                sac.build(P_ev, A_ev, D_ev->lat.ndof);
                t_build = Duration(Clock::now() - t0).count();
                total_build_t += t_build;
            }

            // RR evolve every step
            auto t_rr_start = Clock::now();
            {
                OpApply current_op = sac.as_op();
                auto rr_res = rr_evolve(current_op, sac.defl_vecs, sac.dim);
                sac.defl_vecs = std::move(rr_res.eigvecs);
                sac.defl_vals = std::move(rr_res.eigvals);
            }
            double t_rr = Duration(Clock::now() - t_rr_start).count();
            total_rr_t += t_rr;

            // Solve with the TRUE operator
            SparseCoarseOp sac_true;
            sac_true.build(P_ev, A_ev, D_ev->lat.ndof);

            SparseCoarseOp sac_defl = sac_true;
            sac_defl.defl_vecs = sac.defl_vecs;
            sac_defl.defl_vals = sac.defl_vals;

            int step_defl_it = 0, step_nodfl_it = 0;
            double step_defl_wall = 0, step_nodfl_wall = 0;
            for (int r = 0; r < n_rhs; r++) {
                Vec crhs = random_vec(sac_true.dim, rng_ev);
                auto res_d = solve_timed(sac_defl, crhs, true);
                auto res_n = solve_timed(sac_true, crhs, false);
                step_defl_it += res_d.iters;
                step_nodfl_it += res_n.iters;
                step_defl_wall += res_d.time;
                step_nodfl_wall += res_n.time;
            }

            total_defl_it += step_defl_it;
            total_nodfl_it += step_nodfl_it;
            total_defl_wall += step_defl_wall;
            total_nodfl_wall += step_nodfl_wall;

            double saved_pct = 100.0 * (1.0 - (double)step_defl_it / step_nodfl_it);
            double speedup = step_nodfl_wall / step_defl_wall;

            std::cout << std::fixed << std::setprecision(4);
            std::cout << std::setw(5) << step
                      << std::setw(10) << step_defl_it / n_rhs
                      << std::setw(10) << step_nodfl_it / n_rhs
                      << std::setw(9) << std::setprecision(1) << saved_pct << "%"
                      << std::setw(12) << std::setprecision(4) << step_defl_wall
                      << std::setw(12) << step_nodfl_wall
                      << std::setw(9) << std::setprecision(2) << speedup << "x"
                      << std::setw(10) << std::setprecision(4) << t_build
                      << std::setw(10) << t_rr
                      << std::setw(6) << (did_galerkin ? "Y" : "")
                      << "\n";
        }

        double avg_defl_it = (double)total_defl_it / (study.n_steps * n_rhs);
        double avg_nodfl_it = (double)total_nodfl_it / (study.n_steps * n_rhs);
        double overall_saved = 100.0 * (1.0 - avg_defl_it / avg_nodfl_it);
        double overall_speedup = total_nodfl_wall / total_defl_wall;
        int n_rebuilds = (study.n_steps + galerkin_period - 1) / galerkin_period;

        std::cout << "\n  === " << cfg.label << " Summary over " << study.n_steps << " steps ===\n";
        std::cout << "    Coarse dim:      " << cdim << "\n";
        std::cout << "    Deflated CG:     " << std::fixed << std::setprecision(1)
                  << avg_defl_it << " avg iters/RHS, "
                  << std::setprecision(3) << total_defl_wall << "s total solve\n";
        std::cout << "    Undeflated CG:   " << std::setprecision(1)
                  << avg_nodfl_it << " avg iters/RHS, "
                  << std::setprecision(3) << total_nodfl_wall << "s total solve\n";
        std::cout << "    Iter savings:    " << std::setprecision(1) << overall_saved << "%\n";
        std::cout << "    CG wall speedup: " << std::setprecision(2) << overall_speedup << "x\n";
        std::cout << "    Overhead:        " << std::setprecision(3)
                  << total_build_t << "s Galerkin (" << n_rebuilds
                  << " rebuilds) + " << total_rr_t << "s RR (" << study.n_steps << " steps)\n";
        std::cout << "    Net wall time:   Defl = " << std::setprecision(3)
                  << total_defl_wall + total_build_t + total_rr_t << "s"
                  << "  vs  NoDfl = " << total_nodfl_wall << "s\n";
    }

    return 0;
}

// =====================================================================
//  Test deflation mode
// =====================================================================
int run_deflation_study(GaugeField& gauge, const Lattice& lat,
                        const LatticeConfig& lcfg, const MGConfig& mcfg,
                        const SolverConfig& scfg, const StudyConfig& study,
                        std::mt19937& rng) {
    if (mcfg.mg_levels < 2) {
        std::cerr << "--test-deflation requires --mg-levels >= 2\n";
        return 1;
    }
    int ndefl = scfg.n_defl > 0 ? scfg.n_defl : mcfg.k_null * 4;
    std::cout << "=== Test: Coarse Eigenvector Prolongation ===\n\n";

    DiracOp D(lat, gauge, lcfg.mass, lcfg.wilson_r, lcfg.c_sw, lcfg.mu_t);
    OpApply A = [&D](const Vec& s, Vec& d){ D.apply_DdagD(s, d); };
    std::mt19937 rng_mg(lcfg.seed);
    auto mg = build_mg_hierarchy(D, mcfg.mg_levels, mcfg.block_size, mcfg.k_null,
                                  mcfg.resolved_coarse_block(), 20, rng_mg, mcfg.w_cycle,
                                  3, 3, true);
    mg.set_symmetric(0.8);

    int last = (int)mg.levels.size() - 1;
    int cdim = mg.levels[last].Ac.dim;
    ndefl = std::min(ndefl, cdim / 2);
    std::cout << "Coarsest level dim: " << cdim
              << ", deflation vectors: " << ndefl << "\n\n";

    if (!study.cheb_only) {
    auto t0 = Clock::now();
    auto& Ac = mg.levels[last].Ac;
    OpApply coarse_op = [&Ac](const Vec& s, Vec& d) { d = Ac.apply(s); };
    auto identity = [](const Vec& v) { return v; };

    std::mt19937 drng(42);
    std::vector<Vec> X0(ndefl);
    for (int i = 0; i < ndefl; i++) X0[i] = random_vec(cdim, drng);

    auto coarse_res = lobpcg_update(coarse_op, cdim, ndefl, X0, identity, 50, 1e-12);
    double t_coarse_eig = Duration(Clock::now() - t0).count();

    std::cout << "--- Coarse eigenvalues (LOBPCG on Ac, dim=" << cdim << ") ---\n";
    std::cout << "  Time: " << std::fixed << std::setprecision(4)
              << t_coarse_eig << "s, " << coarse_res.iterations << " iters\n";
    std::cout << std::setw(5) << "i" << std::setw(15) << "lambda_c"
              << std::setw(15) << "coarse_res" << "\n";
    for (int i = 0; i < ndefl; i++) {
        Vec Av = Ac.apply(coarse_res.eigvecs[i]);
        Vec resid = Av;
        axpy(cx(-coarse_res.eigvals[i]), coarse_res.eigvecs[i], resid);
        double rnorm = norm(resid) / std::max(norm(Av), 1e-30);
        std::cout << std::setw(5) << i
                  << std::setw(15) << std::scientific << std::setprecision(6)
                  << coarse_res.eigvals[i]
                  << std::setw(15) << rnorm << "\n";
    }

    // Prolong to fine grid
    t0 = Clock::now();
    std::vector<Vec> fine_vecs(ndefl);
    for (int i = 0; i < ndefl; i++)
        fine_vecs[i] = mg.prolong_to_fine(coarse_res.eigvecs[i]);

    for (int i = 0; i < ndefl; i++) {
        for (int j = 0; j < i; j++) {
            cx proj = dot(fine_vecs[j], fine_vecs[i]);
            axpy(-proj, fine_vecs[j], fine_vecs[i]);
        }
        double nv = norm(fine_vecs[i]);
        if (nv > 1e-14) scale(fine_vecs[i], cx(1.0/nv));
    }
    double t_prolong = Duration(Clock::now() - t0).count();

    std::cout << "\n--- Prolonged to fine grid (dim=" << lat.ndof << ") ---\n";
    std::cout << "  Prolongation + orthonorm time: " << std::fixed
              << std::setprecision(4) << t_prolong << "s\n";
    std::cout << std::setw(5) << "i" << std::setw(15) << "lambda_c"
              << std::setw(15) << "RQ_fine"
              << std::setw(15) << "||Av-lv||/||Av||" << "\n";

    std::vector<double> fine_rq(ndefl);
    for (int i = 0; i < ndefl; i++) {
        Vec Av(lat.ndof);
        A(fine_vecs[i], Av);
        fine_rq[i] = std::real(dot(fine_vecs[i], Av));
        Vec resid = Av;
        axpy(cx(-fine_rq[i]), fine_vecs[i], resid);
        double rnorm = norm(resid) / std::max(norm(Av), 1e-30);
        std::cout << std::setw(5) << i
                  << std::setw(15) << std::scientific << std::setprecision(6)
                  << coarse_res.eigvals[i]
                  << std::setw(15) << fine_rq[i]
                  << std::setw(15) << rnorm << "\n";
    }

    // Inverse iteration refinement
    auto mg_precond = [&mg](const Vec& v) { return mg.precondition(v); };
    std::vector<Vec> refined_vecs = fine_vecs;

    auto print_eigquality = [&](const std::vector<Vec>& vecs) {
        for (int i = 0; i < ndefl; i++) {
            Vec Av(lat.ndof);
            A(vecs[i], Av);
            double rq = std::real(dot(vecs[i], Av));
            Vec resid = Av;
            axpy(cx(-rq), vecs[i], resid);
            double rnorm = norm(resid) / std::max(norm(Av), 1e-30);
            std::cout << std::setw(5) << i
                      << std::setw(15) << std::scientific << std::setprecision(6)
                      << rq
                      << std::setw(15) << rnorm << "\n";
        }
    };

    for (int n_refine : {1, 2, 3, 5, 10, 20}) {
        std::vector<Vec> vecs = fine_vecs;

        t0 = Clock::now();
        for (int iter = 0; iter < n_refine; iter++) {
            for (int i = 0; i < ndefl; i++)
                vecs[i] = mg.precondition(vecs[i]);
            for (int pass = 0; pass < 2; pass++) {
                for (int i = 0; i < ndefl; i++) {
                    for (int j = 0; j < i; j++) {
                        cx proj = dot(vecs[j], vecs[i]);
                        axpy(-proj, vecs[j], vecs[i]);
                    }
                    double nv = norm(vecs[i]);
                    if (nv > 1e-14) scale(vecs[i], cx(1.0/nv));
                }
            }
        }
        double t_refine = Duration(Clock::now() - t0).count();

        std::cout << "\n--- Inverse iteration: " << n_refine << " step(s) ---\n";
        std::cout << "  Time: " << std::fixed << std::setprecision(4)
                  << t_refine << "s (total: "
                  << t_coarse_eig + t_prolong + t_refine << "s)\n";
        std::cout << std::setw(5) << "i" << std::setw(15) << "RQ"
                  << std::setw(15) << "||Av-lv||/||Av||" << "\n";
        print_eigquality(vecs);

        if (n_refine == 5) refined_vecs = vecs;
    }

    // LOBPCG comparison
    for (int n_refine : {5, 10}) {
        t0 = Clock::now();
        auto refine_res = lobpcg_update(A, lat.ndof, ndefl, fine_vecs,
                                         mg_precond, n_refine, 1e-12);
        double t_refine = Duration(Clock::now() - t0).count();

        std::cout << "\n--- LOBPCG: " << n_refine << " iter(s) (warm start) ---\n";
        std::cout << "  Time: " << std::fixed << std::setprecision(4)
                  << t_refine << "s (total: "
                  << t_coarse_eig + t_prolong + t_refine << "s)\n";
        std::cout << std::setw(5) << "i" << std::setw(15) << "RQ"
                  << std::setw(15) << "||Av-lv||/||Av||" << "\n";
        print_eigquality(refine_res.eigvecs);
    }

    // Chebyshev comparison
    for (int n_cheb_iter : {3, 5, 10}) {
        t0 = Clock::now();
        auto cheb_res = chebyshev_subspace_iteration(
            A, lat.ndof, ndefl, fine_vecs,
            /*poly_deg=*/20, /*max_iter=*/n_cheb_iter, /*tol=*/1e-10);
        double t_cheb = Duration(Clock::now() - t0).count();

        std::cout << "\n--- Chebyshev: " << n_cheb_iter
                  << " iter(s), deg=" << 20
                  << " (lambda_max=" << std::scientific << std::setprecision(2)
                  << cheb_res.lambda_max_used << ") ---\n";
        std::cout << "  Time: " << std::fixed << std::setprecision(4)
                  << t_cheb << "s (total: "
                  << t_coarse_eig + t_prolong + t_cheb << "s), "
                  << cheb_res.iterations << " outer iters\n";
        std::cout << std::setw(5) << "i" << std::setw(15) << "RQ"
                  << std::setw(15) << "||Av-lv||/||Av||" << "\n";
        print_eigquality(cheb_res.eigvecs);
    }

    // Direct Lanczos on Ac
    {
        t0 = Clock::now();
        auto& Ac_ref = mg.levels[last].Ac;
        auto coarse_evecs = Ac_ref.smallest_eigenvectors(ndefl);
        double t_direct = Duration(Clock::now() - t0).count();

        std::cout << "\n--- Direct Lanczos on Ac (dim=" << cdim << ") ---\n";
        std::cout << "  Time: " << std::fixed << std::setprecision(4)
                  << t_direct << "s\n";
        std::cout << std::setw(5) << "i" << std::setw(15) << "lambda_c"
                  << std::setw(15) << "coarse_res" << "\n";
        for (int i = 0; i < ndefl; i++) {
            Vec Av_c = Ac_ref.apply(coarse_evecs[i]);
            double rq_c = std::real(dot(coarse_evecs[i], Av_c));
            Vec resid_c = Av_c;
            axpy(cx(-rq_c), coarse_evecs[i], resid_c);
            double rnorm_c = norm(resid_c) / std::max(norm(Av_c), 1e-30);
            std::cout << std::setw(5) << i
                      << std::setw(15) << std::scientific << std::setprecision(6)
                      << rq_c
                      << std::setw(15) << rnorm_c << "\n";
        }

        std::vector<Vec> fine_lanczos(ndefl);
        for (int i = 0; i < ndefl; i++)
            fine_lanczos[i] = mg.prolong_to_fine(coarse_evecs[i]);
        for (int i = 0; i < ndefl; i++) {
            for (int pass = 0; pass < 2; pass++)
                for (int j = 0; j < i; j++) {
                    cx proj = dot(fine_lanczos[j], fine_lanczos[i]);
                    axpy(-proj, fine_lanczos[j], fine_lanczos[i]);
                }
            double nv = norm(fine_lanczos[i]);
            if (nv > 1e-14) scale(fine_lanczos[i], cx(1.0/nv));
        }
        std::cout << "  Prolonged fine-grid quality:\n";
        std::cout << std::setw(5) << "i" << std::setw(15) << "RQ"
                  << std::setw(15) << "||Av-lv||/||Av||" << "\n";
        print_eigquality(fine_lanczos);
    }

    // Solve comparison
    std::cout << "\n--- Solve comparison ---\n";
    Vec rhs = random_vec(lat.ndof, rng_mg);

    t0 = Clock::now();
    auto precond = [&mg](const Vec& v) { return mg.precondition(v); };
    auto res_cg = cg_solve_precond(A, lat.ndof, rhs, precond, scfg.max_iter, scfg.tol);
    double t_cg = Duration(Clock::now() - t0).count();

    t0 = Clock::now();
    auto res_defl_c = cg_solve_deflated(A, lat.ndof, rhs, precond,
                                         fine_vecs, fine_rq, scfg.max_iter, scfg.tol);
    double t_defl_c = Duration(Clock::now() - t0).count();

    std::vector<double> refined_rq(ndefl);
    for (int i = 0; i < ndefl; i++) {
        Vec Av(lat.ndof); A(refined_vecs[i], Av);
        refined_rq[i] = std::real(dot(refined_vecs[i], Av));
    }
    t0 = Clock::now();
    auto res_defl_r = cg_solve_deflated(A, lat.ndof, rhs, precond,
                                         refined_vecs, refined_rq,
                                         scfg.max_iter, scfg.tol);
    double t_defl_r = Duration(Clock::now() - t0).count();

    auto& Ac_solve = mg.levels[last].Ac;
    auto coarse_evecs_solve = Ac_solve.smallest_eigenvectors(ndefl);
    std::vector<Vec> fine_cheb_init(ndefl);
    for (int i = 0; i < ndefl; i++)
        fine_cheb_init[i] = mg.prolong_to_fine(coarse_evecs_solve[i]);
    for (int i = 0; i < ndefl; i++) {
        for (int pass = 0; pass < 2; pass++)
            for (int j = 0; j < i; j++) {
                cx proj = dot(fine_cheb_init[j], fine_cheb_init[i]);
                axpy(-proj, fine_cheb_init[j], fine_cheb_init[i]);
            }
        double nv = norm(fine_cheb_init[i]);
        if (nv > 1e-14) scale(fine_cheb_init[i], cx(1.0/nv));
    }
    auto cheb_defl = chebyshev_subspace_iteration(
        A, lat.ndof, ndefl, fine_cheb_init,
        /*poly_deg=*/20, /*max_iter=*/5, /*tol=*/1e-8);
    std::vector<double> cheb_rq(ndefl);
    for (int i = 0; i < ndefl; i++) cheb_rq[i] = cheb_defl.eigvals[i];

    t0 = Clock::now();
    auto res_defl_cheb = cg_solve_deflated(A, lat.ndof, rhs, precond,
                                            cheb_defl.eigvecs, cheb_rq,
                                            scfg.max_iter, scfg.tol);
    double t_defl_cheb = Duration(Clock::now() - t0).count();

    t0 = Clock::now();
    auto res_bare = cg_solve(A, lat.ndof, rhs, scfg.max_iter, scfg.tol);
    double t_bare = Duration(Clock::now() - t0).count();

    std::cout << "  Bare CG:                " << std::setw(5) << res_bare.iterations
              << " iters  " << std::fixed << std::setprecision(4) << t_bare << "s\n";
    std::cout << "  CG + MG:                " << std::setw(5) << res_cg.iterations
              << " iters  " << t_cg << "s\n";
    std::cout << "  CG + MG + coarseDefl:   " << std::setw(5) << res_defl_c.iterations
              << " iters  " << t_defl_c << "s"
              << "  (prolonged, no refine)\n";
    std::cout << "  CG + MG + invIterDefl:  " << std::setw(5) << res_defl_r.iterations
              << " iters  " << t_defl_r << "s"
              << "  (coarse->prolong->5 inv. iter.)\n";
    std::cout << "  CG + MG + chebDefl:     " << std::setw(5) << res_defl_cheb.iterations
              << " iters  " << t_defl_cheb << "s"
              << "  (Lanczos->prolong->Cheb5)\n";

    std::cout << "\n--- Cost summary ---\n";
    std::cout << "  Coarse eigensolve:       " << std::setprecision(4) << t_coarse_eig << "s\n";
    std::cout << "  Prolongation + orthonorm: " << t_prolong << "s\n";

    // Eigenvector Recovery Comparison
    double eig_tol = 1e-10;
    std::cout << "\n======================================================\n";
    std::cout << "  Eigenvector Recovery Comparison (tol=" << std::scientific
              << eig_tol << ")\n";
    std::cout << "  k=" << ndefl << " smallest eigenpairs of D+D, dim="
              << lat.ndof << "\n";
    std::cout << "======================================================\n\n";

    OpApply A_count = [&](const Vec& s, Vec& d) {
        g_matvec_count++;
        A(s, d);
    };

    auto run_cheb_sweep = [&](const char* label, const std::vector<Vec>& X0_cheb,
                               int poly_deg, const std::vector<int>& iters,
                               long long base_matvecs) {
        for (int n_outer : iters) {
            g_matvec_count = 0;
            t0 = Clock::now();
            auto cheb = chebyshev_subspace_iteration(
                A_count, lat.ndof, ndefl, X0_cheb,
                poly_deg, n_outer, eig_tol);
            double t_cheb_s = Duration(Clock::now() - t0).count();
            long long mv = g_matvec_count + base_matvecs;

            double max_res = 0;
            for (int i = 0; i < ndefl; i++) {
                Vec Av(lat.ndof);
                A(cheb.eigvecs[i], Av);
                double rq = std::real(dot(cheb.eigvecs[i], Av));
                Vec resid2 = Av;
                axpy(cx(-rq), cheb.eigvecs[i], resid2);
                double rnorm = norm(resid2) / std::max(norm(Av), 1e-30);
                max_res = std::max(max_res, rnorm);
            }
            bool conv = max_res < eig_tol;
            std::cout << "  Cheb(" << std::setw(2) << n_outer
                      << " outer, deg=" << std::setw(2) << poly_deg << "): "
                      << std::fixed << std::setprecision(4) << t_cheb_s << "s  "
                      << "matvecs=" << std::setw(6) << mv << "  "
                      << "max_res=" << std::scientific << std::setprecision(2)
                      << max_res
                      << (conv ? "  *** CONVERGED ***" : "") << "\n";
            if (conv) break;
        }
    };

    // Method 1: Coarse Lanczos -> prolong -> Chebyshev (deg=20)
    std::cout << "--- Method 1: Coarse Lanczos -> Prolong -> Chebyshev (deg=20) ---\n";
    {
        t0 = Clock::now();
        auto coarse_evecs_m1 = mg.levels[last].Ac.smallest_eigenvectors(ndefl);
        double t_coarse_m1 = Duration(Clock::now() - t0).count();

        t0 = Clock::now();
        std::vector<Vec> fine_m1(ndefl);
        for (int i = 0; i < ndefl; i++)
            fine_m1[i] = mg.prolong_to_fine(coarse_evecs_m1[i]);
        for (int i = 0; i < ndefl; i++) {
            for (int pass = 0; pass < 2; pass++)
                for (int j = 0; j < i; j++) {
                    cx proj = dot(fine_m1[j], fine_m1[i]);
                    axpy(-proj, fine_m1[j], fine_m1[i]);
                }
            double nv = norm(fine_m1[i]);
            if (nv > 1e-14) scale(fine_m1[i], cx(1.0/nv));
        }
        double t_prol = Duration(Clock::now() - t0).count();

        std::cout << "  Coarse Lanczos:  " << std::fixed << std::setprecision(4)
                  << t_coarse_m1 << "s  (0 fine matvecs)\n";
        std::cout << "  Prolongation:    " << t_prol << "s  (0 fine matvecs)\n";

        run_cheb_sweep("M1", fine_m1, 20,
                       {1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 80}, 0);
    }

    // Method 2: Chebyshev from random start (deg=20)
    std::cout << "\n--- Method 2: Chebyshev from Random Start (deg=20) ---\n";
    {
        std::mt19937 rng_rand(123);
        std::vector<Vec> X_rand(ndefl);
        for (int i = 0; i < ndefl; i++) X_rand[i] = random_vec(lat.ndof, rng_rand);

        run_cheb_sweep("M2", X_rand, 20,
                       {1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 80, 120}, 0);
    }

    // Method 3: Chebyshev from random start (deg=40)
    std::cout << "\n--- Method 3: Chebyshev from Random Start (deg=40) ---\n";
    {
        std::mt19937 rng_rand3(123);
        std::vector<Vec> X_rand3(ndefl);
        for (int i = 0; i < ndefl; i++) X_rand3[i] = random_vec(lat.ndof, rng_rand3);

        run_cheb_sweep("M3", X_rand3, 40,
                       {1, 2, 3, 5, 8, 10, 15, 20, 30, 50}, 0);
    }

    // Method 4: Coarse -> Prolong -> Chebyshev (deg=40)
    std::cout << "\n--- Method 4: Coarse -> Prolong -> Chebyshev (deg=40) ---\n";
    {
        t0 = Clock::now();
        auto coarse_evecs_m4 = mg.levels[last].Ac.smallest_eigenvectors(ndefl);
        std::vector<Vec> fine_m4(ndefl);
        for (int i = 0; i < ndefl; i++)
            fine_m4[i] = mg.prolong_to_fine(coarse_evecs_m4[i]);
        for (int i = 0; i < ndefl; i++) {
            for (int pass = 0; pass < 2; pass++)
                for (int j = 0; j < i; j++) {
                    cx proj = dot(fine_m4[j], fine_m4[i]);
                    axpy(-proj, fine_m4[j], fine_m4[i]);
                }
            double nv = norm(fine_m4[i]);
            if (nv > 1e-14) scale(fine_m4[i], cx(1.0/nv));
        }
        double t_setup_m4 = Duration(Clock::now() - t0).count();
        std::cout << "  Coarse Lanczos + Prolong: " << std::fixed
                  << std::setprecision(4) << t_setup_m4 << "s  (0 fine matvecs)\n";

        run_cheb_sweep("M4", fine_m4, 40,
                       {1, 2, 3, 5, 8, 10, 15, 20, 30, 50}, 0);
    }

    // Method 5: LOBPCG + MG preconditioner from random start
    std::cout << "\n--- Method 5: LOBPCG + MG Preconditioner from Random Start ---\n";
    std::cout << "  (matvec count includes MG smoother applications)\n";
    {
        std::mt19937 rng_rand2(123);
        std::vector<Vec> X_rand2(ndefl);
        for (int i = 0; i < ndefl; i++) X_rand2[i] = random_vec(lat.ndof, rng_rand2);

        for (int n_iter : {5, 10, 20, 50, 100}) {
            g_matvec_count = 0;
            t0 = Clock::now();
            auto lobpcg_res = lobpcg_update(A_count, lat.ndof, ndefl, X_rand2,
                                             mg_precond, n_iter, eig_tol);
            double t_lobpcg = Duration(Clock::now() - t0).count();
            long long mv = g_matvec_count;

            double max_res = 0;
            for (int i = 0; i < ndefl; i++) {
                Vec Av(lat.ndof);
                A(lobpcg_res.eigvecs[i], Av);
                double rq = std::real(dot(lobpcg_res.eigvecs[i], Av));
                Vec resid2 = Av;
                axpy(cx(-rq), lobpcg_res.eigvecs[i], resid2);
                double rnorm = norm(resid2) / std::max(norm(Av), 1e-30);
                max_res = std::max(max_res, rnorm);
            }
            bool conv = max_res < eig_tol;
            std::cout << "  LOBPCG(" << std::setw(3) << n_iter << " iter): "
                      << std::fixed << std::setprecision(4) << t_lobpcg << "s  "
                      << "matvecs=" << std::setw(6) << mv << "  "
                      << "(" << lobpcg_res.iterations << " actual)  "
                      << "max_res=" << std::scientific << std::setprecision(2)
                      << max_res
                      << (conv ? "  *** CONVERGED ***" : "") << "\n";
            if (conv) break;
        }
    }

    // Method 6: TRLM from random start
    std::cout << "\n--- Method 6: TRLM (Thick Restart Lanczos) ---\n";
    {
        g_matvec_count = 0;
        t0 = Clock::now();
        auto trlm_res = trlm_eigensolver(A_count, lat.ndof, ndefl,
                                          /*n_kr=*/0, /*max_restarts=*/200,
                                          eig_tol);
        double t_trlm = Duration(Clock::now() - t0).count();
        long long mv_trlm = g_matvec_count;

        double max_res = 0;
        for (int i = 0; i < ndefl; i++) {
            Vec Av(lat.ndof);
            A(trlm_res.eigvecs[i], Av);
            double rq = std::real(dot(trlm_res.eigvecs[i], Av));
            Vec resid2 = Av;
            axpy(cx(-rq), trlm_res.eigvecs[i], resid2);
            double rnorm = norm(resid2) / std::max(norm(Av), 1e-30);
            max_res = std::max(max_res, rnorm);
        }
        std::cout << "  TRLM(no cheby): "
                  << std::fixed << std::setprecision(4) << t_trlm << "s  "
                  << "matvecs=" << std::setw(6) << mv_trlm << "  "
                  << "restarts=" << trlm_res.num_restarts << "  "
                  << "max_res=" << std::scientific << std::setprecision(2)
                  << max_res
                  << (trlm_res.converged ? "  *** CONVERGED ***" : "") << "\n";
    }
    } // end if (!cheb_only)

    // Method 7: TRLM + Chebyshev parameter scan
    double eig_tol_m7 = 1e-10;
    OpApply A_count_m7 = [&](const Vec& s, Vec& d) {
        g_matvec_count++;
        A(s, d);
    };
    std::mt19937 rng_m7(lcfg.seed + 7777);
    // Use raw n_defl (not coarse-clamped) for the eigensolver study
    int n_ev = scfg.n_defl > 0 ? scfg.n_defl : ndefl;
    n_ev = std::min(n_ev, lat.ndof / 2);  // don't exceed half the matrix dim
    std::cout << "\n=== TRLM Chebyshev Study: L=" << lcfg.L
              << ", n_ev=" << n_ev << ", dim=" << lat.ndof << " ===\n";
    {
        // 1) Bare TRLM to discover spectrum
        g_matvec_count = 0;
        auto t0 = Clock::now();
        auto bare_res = trlm_eigensolver(A_count_m7, lat.ndof,
                                          std::min(n_ev + 4, lat.ndof),
                                          /*n_kr=*/0, /*max_restarts=*/500, eig_tol_m7);
        double t_bare = Duration(Clock::now() - t0).count();
        long long mv_bare = g_matvec_count;

        // Power iteration for lambda_max
        Vec v = random_vec(lat.ndof, rng_m7);
        double nv2 = norm(v); scale(v, cx(1.0/nv2));
        Vec Av2(lat.ndof);
        double lmax = 0;
        for (int i = 0; i < 100; i++) {
            A(v, Av2);
            double rq = std::real(dot(v, Av2));
            lmax = std::max(lmax, rq);
            double nrm = norm(Av2);
            if (nrm > 1e-14) { v = Av2; scale(v, cx(1.0/nrm)); }
        }
        double lambda_max = lmax * 1.1;

        double ev_min = bare_res.eigvals[0];
        double ev_max = bare_res.eigvals[n_ev - 1];
        double ev_next = (int(bare_res.eigvals.size()) > n_ev)
                         ? bare_res.eigvals[n_ev] : ev_max * 2.0;
        double gap_ratio = ev_next / ev_max;

        std::cout << "  Bare TRLM: " << mv_bare << " matvecs, "
                  << bare_res.num_restarts << " restarts, "
                  << std::fixed << std::setprecision(2) << t_bare << "s"
                  << (bare_res.converged ? "  CONVERGED" : "  FAILED") << "\n";
        std::cout << "  ev[0]=" << std::scientific << std::setprecision(4) << ev_min
                  << "  ev[" << n_ev-1 << "]=" << ev_max
                  << "  ev[" << n_ev << "]=" << ev_next
                  << "  gap=" << std::fixed << std::setprecision(3) << gap_ratio
                  << "  lambda_max=" << std::scientific << std::setprecision(3) << lambda_max
                  << "\n\n";

        // Print a few eigenvalues to show spectrum shape
        std::cout << "  Spectrum sample: ";
        for (int i : {0, 1, 2, n_ev/4, n_ev/2, 3*n_ev/4, n_ev-2, n_ev-1, n_ev}) {
            if (i >= 0 && i < (int)bare_res.eigvals.size()) {
                std::cout << "[" << i << "]=" << std::scientific
                          << std::setprecision(3) << bare_res.eigvals[i] << " ";
            }
        }
        std::cout << "\n\n";

        // Helper
        auto run_trlm_cheb = [&](const char* label, double amin, double amax,
                                  int deg, int n_kr_in = 0) {
            g_matvec_count = 0;
            auto res = trlm_eigensolver(A_count_m7, lat.ndof, n_ev,
                                         n_kr_in, /*max_restarts=*/200,
                                         eig_tol_m7, deg, amin, amax);
            long long mv = g_matvec_count;

            // Compute actual residual for verification
            double max_res = 0;
            int n_check = std::min(n_ev, (int)res.eigvecs.size());
            for (int i = 0; i < n_check; i++) {
                Vec Av3(lat.ndof);
                A(res.eigvecs[i], Av3);
                double rq = std::real(dot(res.eigvecs[i], Av3));
                Vec r2 = Av3;
                axpy(cx(-rq), res.eigvecs[i], r2);
                double rnorm = norm(r2) / std::max(norm(Av3), 1e-30);
                max_res = std::max(max_res, rnorm);
            }

            // Compute default n_kr for display
            int n_kr_display = n_kr_in;
            if (n_kr_display == 0) {
                if (deg > 0)
                    n_kr_display = std::min(n_ev + 24, lat.ndof);
                else
                    n_kr_display = std::min(std::max(2*n_ev+6, n_ev+32), lat.ndof);
            }

            std::cout << std::setw(10) << label
                      << std::setw(6) << deg
                      << std::setw(6) << n_kr_display
                      << std::setw(10) << mv
                      << std::setw(8) << res.num_restarts
                      << std::scientific << std::setprecision(2)
                      << std::setw(11) << max_res
                      << (res.converged ? "  CONV" : "  FAIL") << "\n";
        };

        double cheb_amax = lambda_max;

        // Table header
        auto print_header = [&]() {
            std::cout << std::setw(10) << "label"
                      << std::setw(6) << "deg"
                      << std::setw(6) << "n_kr"
                      << std::setw(10) << "matvecs"
                      << std::setw(8) << "restart"
                      << std::setw(11) << "max_res"
                      << "  status\n";
            std::cout << std::string(60, '-') << "\n";
        };

        // 2) Bare TRLM baselines with various n_kr
        std::cout << "--- Bare TRLM baselines ---\n";
        print_header();
        for (int extra : {16, 24, 32, 48, 64, 96, 128}) {
            int nkr = n_ev + extra;
            if (nkr > lat.ndof) continue;
            char lbl[32];
            snprintf(lbl, sizeof(lbl), "nkr=%d", nkr);
            run_trlm_cheb(lbl, 0.0, 0.0, 0, nkr);
        }
        std::cout << "\n";

        // 3) a_min scan with fixed deg=16, default n_kr
        //    Only test values above the gap — below-gap always fails
        std::cout << "--- a_min scan (deg=16, a_max=" << std::scientific
                  << std::setprecision(2) << cheb_amax << ") ---\n";
        print_header();
        for (double amin : {ev_next * 1.1, ev_next * 2.0, ev_next * 5.0,
                            1.0, 2.0, 4.0}) {
            if (amin <= ev_max || amin >= cheb_amax) continue;
            char lbl[32];
            snprintf(lbl, sizeof(lbl), "%.4f", amin);
            run_trlm_cheb(lbl, amin, cheb_amax, 16);
        }
        std::cout << "\n";

        // 4) deg x n_kr scan at best a_min = ev_next * 1.1
        //    (just above the first unwanted eigenvalue)
        double best_amin = ev_next * 1.1;
        std::cout << "--- deg x n_kr scan (a_min=" << std::scientific
                  << std::setprecision(3) << best_amin
                  << ", a_max=" << cheb_amax << ") ---\n";
        print_header();
        for (int deg : {4, 8, 16, 32}) {
            for (int extra : {16, 24, 32, 48, 64, 96}) {
                int nkr = n_ev + extra;
                if (nkr > lat.ndof) continue;
                char lbl[32];
                snprintf(lbl, sizeof(lbl), "d%d/k%d", deg, nkr);
                run_trlm_cheb(lbl, best_amin, cheb_amax, deg, nkr);
            }
            std::cout << std::string(60, '-') << "\n";
        }
    }

    // =============================================================
    // Method 8: Force-based vs RR eigenvector evolution
    // =============================================================
    // Compare two approaches for evolving eigenvectors across MD steps:
    // 1) Force-based: update arrow matrix using dD from momentum (0 matvecs)
    // 2) RR projection: apply A_new and re-project (k matvecs)
    std::cout << "\n=== Method 8: Force-Based vs RR Eigenvector Evolution ===\n";
    std::cout << "  Physical MD evolution (momentum-driven gauge updates)\n\n";
    {
        int k_want = scfg.n_defl > 0 ? scfg.n_defl : ndefl;
        k_want = std::min(k_want, lat.ndof / 4);
        int n_md_steps = std::min(study.n_steps, 50);
        double hmc_beta_m8 = 2.0;

        for (double dt : {0.01, 0.02, 0.05}) {
          for (int k_total : {k_want, k_want * 4, k_want * 8}) {
            if (k_total > lat.ndof / 2) continue;
            std::cout << "\n  ====== dt=" << std::fixed << std::setprecision(4) << dt
                      << ", k_want=" << k_want << ", k_total=" << k_total << " ======\n";

            // Run both methods on identical gauge evolution
            for (int method : {0, 1}) {
                const char* name = method == 0
                    ? "Force-based (0 matvecs/step)"
                    : "RR projection (k matvecs/step)";
                std::cout << "\n  --- " << name << " ---\n";
                std::cout << std::setw(5) << "Step"
                          << std::setw(8) << "Plaq"
                          << std::setw(14) << "ev[0]"
                          << std::setw(14) << "ev[0]_true"
                          << std::setw(10) << "rel_err"
                          << std::setw(12) << "true_res"
                          << "\n";

                GaugeField gauge_m8 = gauge;
                MomentumField mom_m8(lat);
                std::mt19937 rng_m8(lcfg.seed + 8888);
                mom_m8.randomise(rng_m8);

                auto D_m8 = std::make_unique<DiracOp>(lat, gauge_m8, lcfg.mass, lcfg.wilson_r, lcfg.c_sw, lcfg.mu_t);
                OpApply A_m8 = [&D_m8](const Vec& s, Vec& d){ D_m8->apply_DdagD(s, d); };

                // Initial eigenvectors from TRLM
                auto res0 = trlm_eigensolver(A_m8, lat.ndof, k_total,
                                              /*n_kr=*/0, /*max_restarts=*/200, eig_tol_m7);
                if (!res0.converged) {
                    std::cout << "  TRLM failed\n";
                    continue;
                }
                int init_mv = res0.iterations;

                std::vector<Vec> evecs = std::move(res0.eigvecs);
                std::vector<double> evals = std::move(res0.eigvals);

                // For force-based: store w_i = D v_i
                std::vector<Vec> Dv_store;
                if (method == 0) {
                    Dv_store.resize(k_total);
                    for (int i = 0; i < k_total; i++) {
                        Dv_store[i].resize(lat.ndof);
                        D_m8->apply(evecs[i], Dv_store[i]);
                    }
                }

                int total_matvecs = 0;
                int n_refreshes = 0;
                int total_refresh_mv = 0;

                for (int step = 0; step < n_md_steps; step++) {
                    if (method == 0) {
                        // Force-based: compute dD using momentum BEFORE gauge update
                        auto dD = [&D_m8, &mom_m8, dt](const Vec& src, Vec& dst) {
                            D_m8->apply_delta_D(src, dst, mom_m8.pi, dt);
                        };
                        auto dD_dag = [&D_m8, &mom_m8, dt, ndof=lat.ndof](const Vec& src, Vec& dst) {
                            int V = ndof / 2;
                            Vec g5src(ndof);
                            for (int sv = 0; sv < V; sv++) {
                                g5src[2*sv]   = src[2*sv];
                                g5src[2*sv+1] = -src[2*sv+1];
                            }
                            Vec g5dst(ndof);
                            D_m8->apply_delta_D(g5src, g5dst, mom_m8.pi, dt);
                            for (int sv = 0; sv < V; sv++) {
                                dst[2*sv]   = g5dst[2*sv];
                                dst[2*sv+1] = -g5dst[2*sv+1];
                            }
                        };

                        auto fe = force_evolve(evecs, evals, Dv_store,
                                               dD, dD_dag, lat.ndof);
                        evecs = std::move(fe.eigvecs);
                        evals = std::move(fe.eigvals);
                        Dv_store = std::move(fe.Dv);
                    }

                    // Physical MD gauge update: U -> exp(i dt pi) U
                    for (int mu = 0; mu < 2; mu++)
                        for (int sv = 0; sv < lat.V; sv++)
                            gauge_m8.U[mu][sv] *= std::exp(cx(0, dt * mom_m8.pi[mu][sv]));

                    D_m8 = std::make_unique<DiracOp>(lat, gauge_m8, lcfg.mass, lcfg.wilson_r, lcfg.c_sw, lcfg.mu_t);

                    if (method == 1) {
                        // RR projection
                        auto rr = rr_evolve(A_m8, evecs, lat.ndof);
                        evecs = std::move(rr.eigvecs);
                        evals = std::move(rr.eigvals);
                        total_matvecs += rr.matvecs;
                    }

                    // Update momentum with gauge force (leapfrog)
                    std::array<RVec, 2> gf;
                    gauge_force(gauge_m8, hmc_beta_m8, gf);
                    for (int mu = 0; mu < 2; mu++)
                        for (int sv = 0; sv < lat.V; sv++)
                            mom_m8.pi[mu][sv] += dt * gf[mu][sv];

                    // Compute true residual of k_want eigenvectors periodically
                    if (step % 5 == 0 || step == n_md_steps - 1) {
                        // Max residual over wanted eigenvectors
                        double max_true_res = 0;
                        for (int iv = 0; iv < std::min(k_want, k_total); iv++) {
                            Vec Av(lat.ndof);
                            A_m8(evecs[iv], Av);
                            double av_norm = norm(Av);
                            Vec r_vec = Av;
                            axpy(cx(-evals[iv]), evecs[iv], r_vec);
                            double res_iv = norm(r_vec) / std::max(av_norm, 1e-30);
                            max_true_res = std::max(max_true_res, res_iv);
                        }
                        total_matvecs += std::min(k_want, k_total);

                        auto true_eig = trlm_eigensolver(A_m8, lat.ndof, 1,
                                              /*n_kr=*/0, /*max_restarts=*/200, 1e-10);
                        double true_ev0 = true_eig.converged ? true_eig.eigvals[0] : -1;
                        double ev_err = true_ev0 > 0 ?
                            std::abs(evals[0] - true_ev0) / std::max(std::abs(true_ev0), 1e-30) : -1;

                        std::cout << std::setw(5) << step
                                  << std::setw(8) << std::fixed << std::setprecision(4) << gauge_m8.avg_plaq()
                                  << std::setw(14) << std::scientific << std::setprecision(6) << evals[0]
                                  << std::setw(14) << std::scientific << std::setprecision(6) << true_ev0
                                  << std::setw(10) << std::scientific << std::setprecision(2) << ev_err
                                  << std::setw(12) << std::scientific << std::setprecision(2) << max_true_res;

                        // Adaptive refresh if residual too large
                        if (max_true_res > 0.5) {
                            auto refresh = trlm_eigensolver(A_m8, lat.ndof, k_total,
                                                  /*n_kr=*/0, /*max_restarts=*/200, eig_tol_m7);
                            if (refresh.converged) {
                                evecs = std::move(refresh.eigvecs);
                                evals = std::move(refresh.eigvals);
                                total_refresh_mv += refresh.iterations;
                                n_refreshes++;
                                std::cout << "  *REF";
                                if (method == 0) {
                                    Dv_store.resize(k_total);
                                    for (int i = 0; i < k_total; i++) {
                                        Dv_store[i].resize(lat.ndof);
                                        D_m8->apply(evecs[i], Dv_store[i]);
                                    }
                                }
                            }
                        }
                        std::cout << "\n";
                    }
                }
                std::cout << "  Tracking matvecs: " << total_matvecs
                          << ", refresh: " << total_refresh_mv << " (" << n_refreshes << "x)"
                          << ", init: " << init_mv
                          << ", TOTAL: " << (total_matvecs + total_refresh_mv + init_mv) << "\n";
            }
          }
        }
    }

    // =============================================================
    // Method 9: Hybrid tracker -- force + Lanczos extension
    // =============================================================
    // Track n_kr Krylov subspace with force-based rotation,
    // periodically extend with Lanczos to bring in fresh directions.
    std::cout << "\n=== Method 9: Hybrid Force + Lanczos Extension ===\n";
    std::cout << "  n_kr Krylov vectors tracked, periodic Lanczos refresh\n\n";
    {
        int n_ev_h = scfg.n_defl > 0 ? scfg.n_defl : ndefl;
        n_ev_h = std::min(n_ev_h, lat.ndof / 4);
        int n_md_steps_h = std::min(study.n_steps, 50);
        double hmc_beta_h = 2.0;

        for (double dt : {0.01, 0.05}) {
          // Try different Lanczos extension intervals
          for (int ext_interval : {1, 2, 5}) {
            std::cout << "  ====== dt=" << std::fixed << std::setprecision(3) << dt
                      << ", n_ev=" << n_ev_h
                      << ", Lanczos every " << ext_interval << " steps ======\n";

            GaugeField gauge_h = gauge;
            MomentumField mom_h(lat);
            std::mt19937 rng_h(lcfg.seed + 8888);
            mom_h.randomise(rng_h);

            auto D_h = std::make_unique<DiracOp>(lat, gauge_h, lcfg.mass, lcfg.wilson_r, lcfg.c_sw, lcfg.mu_t);
            OpApply A_h = [&D_h](const Vec& s, Vec& d){ D_h->apply_DdagD(s, d); };
            auto applyD = [&D_h](const Vec& s, Vec& d){ D_h->apply(s, d); };

            // Initial TRLM -- use larger Krylov space for better tracking
            int n_kr_h = std::max(4 * n_ev_h, 2 * n_ev_h + 16);
            n_kr_h = std::min(n_kr_h, lat.ndof / 2);
            auto res0 = trlm_eigensolver(A_h, lat.ndof, n_ev_h,
                                          n_kr_h, /*max_restarts=*/200, eig_tol_m7);
            if (!res0.converged) {
                std::cout << "  Initial TRLM failed\n\n";
                continue;
            }
            int init_mv = res0.iterations;

            // Initialise hybrid tracker
            auto tracker = hybrid_tracker_init(A_h, applyD, res0,
                                                lat.ndof, n_ev_h, n_kr_h);

            std::cout << "  n_kr=" << n_kr_h << ", init_mv=" << init_mv << "\n";
            std::cout << std::setw(5) << "Step"
                      << std::setw(8) << "Plaq"
                      << std::setw(14) << "ev[0]"
                      << std::setw(14) << "ev[0]_true"
                      << std::setw(10) << "rel_err"
                      << std::setw(10) << "max_res"
                      << std::setw(8) << "mv"
                      << std::setw(6) << "type"
                      << "\n";

            int total_track_mv = 0;

            for (int step = 0; step < n_md_steps_h; step++) {
                bool do_lanczos = (step % ext_interval == 0);

                if (!do_lanczos) {
                    // Force-based step: compute dD before gauge update
                    auto dD = [&D_h, &mom_h, dt](const Vec& src, Vec& dst) {
                        D_h->apply_delta_D(src, dst, mom_h.pi, dt);
                    };

                    auto hr = hybrid_force_step(tracker, dD);
                    total_track_mv += hr.matvecs;
                }

                // Physical MD gauge update
                for (int mu = 0; mu < 2; mu++)
                    for (int sv = 0; sv < lat.V; sv++)
                        gauge_h.U[mu][sv] *= std::exp(cx(0, dt * mom_h.pi[mu][sv]));

                D_h = std::make_unique<DiracOp>(lat, gauge_h, lcfg.mass, lcfg.wilson_r, lcfg.c_sw, lcfg.mu_t);

                if (do_lanczos) {
                    // Lanczos extension step: brings new directions
                    int n_ext = n_kr_h - n_ev_h;
                    auto hr = hybrid_lanczos_step(tracker, A_h, applyD, n_ext);
                    total_track_mv += hr.matvecs;

                    // Get true eigenvalue for comparison
                    auto true_eig = trlm_eigensolver(A_h, lat.ndof, 1,
                                          /*n_kr=*/0, /*max_restarts=*/200, 1e-10);
                    double true_ev0 = true_eig.converged ? true_eig.eigvals[0] : -1;
                    double ev_err = true_ev0 > 0 ?
                        std::abs(hr.eigvals[0] - true_ev0) / std::max(std::abs(true_ev0), 1e-30) : -1;

                    std::cout << std::setw(5) << step
                              << std::setw(8) << std::fixed << std::setprecision(4) << gauge_h.avg_plaq()
                              << std::setw(14) << std::scientific << std::setprecision(6) << hr.eigvals[0]
                              << std::setw(14) << std::scientific << std::setprecision(6) << true_ev0
                              << std::setw(10) << std::scientific << std::setprecision(2) << ev_err
                              << std::setw(10) << std::scientific << std::setprecision(2) << hr.max_residual
                              << std::setw(8) << hr.matvecs
                              << std::setw(6) << "Lcz"
                              << "\n";
                }

                // Update momentum (leapfrog)
                std::array<RVec, 2> gf;
                gauge_force(gauge_h, hmc_beta_h, gf);
                for (int mu = 0; mu < 2; mu++)
                    for (int sv = 0; sv < lat.V; sv++)
                        mom_h.pi[mu][sv] += dt * gf[mu][sv];
            }

            // Compare with fresh TRLM cost
            int fresh_every = init_mv * n_md_steps_h;
            std::cout << "  Total tracking matvecs: " << total_track_mv
                      << " + init: " << init_mv
                      << " = " << (total_track_mv + init_mv)
                      << " (vs fresh every step: " << fresh_every << ")\n\n";
          }
        }
    }

    // =============================================================
    // Method 10: Multi-source EigenTracker
    // =============================================================
    // Combines four information sources:
    //   1. Force-based evolution (0 D+D matvecs)
    //   2. CG Ritz harvesting (0 extra matvecs -- from fermion solve)
    //   3. Chebyshev-filtered probes (periodic, ~20 matvecs each)
    //   4. Coarse-grid spectral proxy (nearly free)
    //
    // Runs a physical MD trajectory with leapfrog momentum updates.
    // At each step, the tracker maintains eigenvectors using cheap
    // sources and is compared against fresh TRLM ground truth.
    std::cout << "\n=== Method 10: Multi-Source EigenTracker ===\n";
    std::cout << "  Pool-based tracker: force + solver harvest + Chebyshev probe\n\n";
    {
        int n_ev_t = scfg.n_defl > 0 ? scfg.n_defl : ndefl;
        n_ev_t = std::min(n_ev_t, lat.ndof / 4);
        int pool_cap = std::max(3 * n_ev_t, 2 * n_ev_t + 12);
        pool_cap = std::min(pool_cap, lat.ndof / 2);
        int n_md_steps_t = std::min(study.n_steps, 50);
        double hmc_beta_t = 2.0;
        int cheb_degree = 20;
        int cheb_interval = 5;  // Chebyshev probe every N steps

        // Power iteration to estimate lambda_max once
        GaugeField gauge_lmax = gauge;
        DiracOp D_lmax(lat, gauge_lmax, lcfg.mass, lcfg.wilson_r, lcfg.c_sw, lcfg.mu_t);
        OpApply A_lmax = [&D_lmax](const Vec& s, Vec& d){ D_lmax.apply_DdagD(s, d); };
        double lambda_max_est = 0;
        {
            std::mt19937 rng_pi(77777);
            Vec v = random_vec(lat.ndof, rng_pi);
            for (int it = 0; it < 30; it++) {
                Vec Av(lat.ndof);
                A_lmax(v, Av);
                double nAv = norm(Av);
                lambda_max_est = nAv / norm(v);
                v = Av;
                scale(v, cx(1.0 / nAv));
            }
            lambda_max_est *= 1.2; // safety margin
        }

        for (double dt : {0.01, 0.05}) {
          // Sweep: force-only, force+harvest, force+harvest+cheb
          struct Strategy {
              const char* name;
              bool use_harvest;
              bool use_cheb;
              int  perturb_order;  // 0=none, 1/2/3=perturbation order
          };
          Strategy strategies[] = {
              {"Force only    ", false, false, 0},
              {"Force+Ptb(1)  ", false, false, 1},
              {"Force+Ptb(2)  ", false, false, 2},
              {"Force+Ptb(3)  ", false, false, 3},
              {"F+Ptb(2)+Harv ", true,  false, 2},
              {"F+Ptb(2)+H+Ch ", true,  true,  2},
          };

          for (auto& strat : strategies) {
            std::cout << "  ====== dt=" << std::fixed << std::setprecision(3) << dt
                      << ", pool=" << pool_cap
                      << ", " << strat.name << " ======\n";

            GaugeField gauge_t = gauge;
            MomentumField mom_t(lat);
            std::mt19937 rng_t(lcfg.seed + 9999);
            mom_t.randomise(rng_t);

            auto D_t = std::make_unique<DiracOp>(lat, gauge_t, lcfg.mass, lcfg.wilson_r, lcfg.c_sw, lcfg.mu_t);
            OpApply A_t = [&D_t](const Vec& s, Vec& d){ D_t->apply_DdagD(s, d); };
            auto applyD_t = [&D_t](const Vec& s, Vec& d){ D_t->apply(s, d); };

            // Initial TRLM
            int n_kr_t = std::max(4 * n_ev_t, 2 * n_ev_t + 16);
            n_kr_t = std::min(n_kr_t, lat.ndof / 2);
            auto res0 = trlm_eigensolver(A_t, lat.ndof, n_ev_t,
                                          n_kr_t, /*max_restarts=*/200, eig_tol_m7);
            if (!res0.converged) {
                std::cout << "  Initial TRLM failed\n\n";
                continue;
            }
            int init_mv = res0.iterations;

            // Initialise tracker
            EigenTracker tracker;
            tracker.init(res0, applyD_t, lat.ndof, n_ev_t, pool_cap);

            int total_tracker_mv = 0;  // D+D matvecs used by tracker
            int total_solver_mv = 0;   // CG iterations (needed anyway)
            int total_true_mv = 0;     // TRLM for ground truth

            std::cout << "  pool_cap=" << pool_cap << ", init_mv=" << init_mv << "\n";
            std::cout << std::setw(5) << "Step"
                      << std::setw(8) << "Plaq"
                      << std::setw(14) << "ev[0]"
                      << std::setw(14) << "ev[0]_true"
                      << std::setw(10) << "rel_err"
                      << std::setw(10) << "max_res"
                      << std::setw(8) << "trk_mv"
                      << std::setw(7) << "cg_it"
                      << std::setw(6) << "absorb"
                      << std::setw(6) << "pool"
                      << "\n";

            for (int step = 0; step < n_md_steps_t; step++) {
                int step_tracker_mv = 0;

                // 1. Perturbation-directed extension BEFORE gauge update
                //    Each order builds a deeper perturbation Krylov subspace:
                //      Order 1: span{v, dA v}
                //      Order 2: span{v, dA v, (dA)^2 v}
                //      Order p: span{v, dA v, ..., (dA)^p v}
                //    After each call, compress() rotates to optimal Ritz vectors,
                //    so the next call naturally captures the next order correction.
                if (strat.perturb_order > 0) {
                    auto dD = [&D_t, &mom_t, dt](const Vec& src, Vec& dst) {
                        D_t->apply_delta_D(src, dst, mom_t.pi, dt);
                    };
                    // dD+ v = g5 dD(g5 v) for Wilson-Dirac
                    auto dD_dag = [&D_t, &mom_t, dt](const Vec& src, Vec& dst) {
                        int V = D_t->lat.V;
                        Vec g5src(src.size());
                        for (int s = 0; s < V; s++) {
                            g5src[2*s]   =  src[2*s];
                            g5src[2*s+1] = -src[2*s+1];
                        }
                        Vec g5dst(src.size());
                        D_t->apply_delta_D(g5src, g5dst, mom_t.pi, dt);
                        dst.resize(src.size());
                        for (int s = 0; s < V; s++) {
                            dst[2*s]   =  g5dst[2*s];
                            dst[2*s+1] = -g5dst[2*s+1];
                        }
                    };
                    auto D_dag = [&D_t](const Vec& src, Vec& dst) {
                        D_t->apply_dag(src, dst);
                    };
                    auto D_fwd = [&D_t](const Vec& src, Vec& dst) {
                        D_t->apply(src, dst);
                    };
                    for (int ord = 0; ord < strat.perturb_order; ord++) {
                        tracker.perturbation_extend(dD, dD_dag, D_dag, D_fwd);
                    }
                    // Cost: order x n_ev x ~1.25 D+D
                    step_tracker_mv += strat.perturb_order * n_ev_t * 5 / 4;
                }

                // 2. Force-evolve tracker BEFORE gauge update
                //    (uses current D and momentum to compute dD)
                {
                    auto dD = [&D_t, &mom_t, dt](const Vec& src, Vec& dst) {
                        D_t->apply_delta_D(src, dst, mom_t.pi, dt);
                    };
                    tracker.force_update(dD);
                    // force_update: 0 D+D matvecs
                }

                // 3. Physical MD gauge update: U -> exp(i eps pi) U
                for (int mu = 0; mu < 2; mu++)
                    for (int sv = 0; sv < lat.V; sv++)
                        gauge_t.U[mu][sv] *= std::exp(cx(0, dt * mom_t.pi[mu][sv]));

                // Recreate Dirac operator for new gauge
                D_t = std::make_unique<DiracOp>(lat, gauge_t, lcfg.mass, lcfg.wilson_r, lcfg.c_sw, lcfg.mu_t);

                // 4. CG solve (simulating fermion force computation)
                //    Harvest Ritz pairs for free
                int cg_iters = 0;
                int n_absorbed = 0;
                if (strat.use_harvest) {
                    Vec rhs_t = random_vec(lat.ndof, rng_t);
                    std::vector<RitzPair> cg_ritz;
                    auto cg_res = cg_solve_ritz(A_t, lat.ndof, rhs_t,
                                                 scfg.max_iter, scfg.tol,
                                                 n_ev_t, cg_ritz);
                    cg_iters = cg_res.iterations;
                    total_solver_mv += cg_iters;

                    // Absorb Ritz vectors into tracker
                    if (!cg_ritz.empty()) {
                        std::vector<Vec> ritz_vecs;
                        for (auto& rp : cg_ritz)
                            ritz_vecs.push_back(std::move(rp.vector));
                        n_absorbed = tracker.absorb(ritz_vecs, applyD_t);
                        step_tracker_mv += n_absorbed; // D applications
                    }
                }

                // 5. Chebyshev probe (periodic)
                if (strat.use_cheb && step % cheb_interval == 0) {
                    tracker.chebyshev_probe(A_t, applyD_t, rng_t,
                                            lambda_max_est, cheb_degree);
                    step_tracker_mv += cheb_degree + 1; // D+D + 1 D application
                }

                total_tracker_mv += step_tracker_mv;

                // 6. Ground truth: fresh TRLM (for comparison only)
                auto true_eig = trlm_eigensolver(A_t, lat.ndof, 1,
                                      /*n_kr=*/0, /*max_restarts=*/200, 1e-10);
                total_true_mv += true_eig.iterations;
                double true_ev0 = true_eig.converged ? true_eig.eigvals[0] : -1;

                double ev_err = (true_ev0 > 0 && !tracker.eigvals.empty()) ?
                    std::abs(tracker.eigvals[0] - true_ev0) /
                    std::max(std::abs(true_ev0), 1e-30) : -1;

                // Compute residual every few steps (costs n_ev matvecs)
                double max_res = -1;
                if (step % 5 == 0 || step == n_md_steps_t - 1) {
                    max_res = tracker.max_residual(A_t);
                    total_tracker_mv += n_ev_t;
                }

                std::cout << std::setw(5) << step
                          << std::setw(8) << std::fixed << std::setprecision(4)
                          << gauge_t.avg_plaq()
                          << std::setw(14) << std::scientific << std::setprecision(6)
                          << (tracker.eigvals.empty() ? -1.0 : tracker.eigvals[0])
                          << std::setw(14) << std::scientific << std::setprecision(6)
                          << true_ev0
                          << std::setw(10) << std::scientific << std::setprecision(2)
                          << ev_err
                          << std::setw(10) << std::scientific << std::setprecision(2)
                          << max_res
                          << std::setw(8) << step_tracker_mv
                          << std::setw(7) << cg_iters
                          << std::setw(6) << n_absorbed
                          << std::setw(6) << tracker.pool_used()
                          << "\n";

                // Update momentum (leapfrog)
                std::array<RVec, 2> gf;
                gauge_force(gauge_t, hmc_beta_t, gf);
                for (int mu = 0; mu < 2; mu++)
                    for (int sv = 0; sv < lat.V; sv++)
                        mom_t.pi[mu][sv] += dt * gf[mu][sv];
            }

            // Summary
            int fresh_cost = total_true_mv;  // what it costs to do TRLM every step
            std::cout << "  --- Summary ---\n";
            std::cout << "  Tracker overhead: " << total_tracker_mv << " D+D matvecs\n";
            std::cout << "  CG solver (needed anyway): " << total_solver_mv << " iters\n";
            std::cout << "  Fresh TRLM (ground truth): " << fresh_cost << " total matvecs\n";
            std::cout << "  Tracker/Fresh ratio: " << std::fixed << std::setprecision(1)
                      << 100.0 * total_tracker_mv / std::max(fresh_cost, 1)
                      << "%\n\n";
          }
        }
    }

    return 0;
}

// =====================================================================
//  No-MG baseline mode
// =====================================================================
int run_no_mg_baseline(GaugeField& gauge, const Lattice& lat,
                       const LatticeConfig& lcfg, const SolverConfig& scfg,
                       const StudyConfig& study, std::mt19937& rng) {
    std::cout << "--- No-MG Baseline Mode ---\n";
    std::cout << "Running unpreconditioned CG and FGMRES on D+D\n\n";

    std::cout << std::setw(5)  << "Step"
              << std::setw(8)  << "Plaq"
              << std::setw(8) << "CG_it"
              << std::setw(10) << "CG_time"
              << std::setw(8) << "FGM_it"
              << std::setw(10) << "FGM_time"
              << "\n";
    std::cout << std::string(49, '-') << "\n";

    double total_t_cg = 0, total_t_fgmres = 0;
    int    total_it_cg = 0, total_it_fgmres = 0;

    for (int step = 0; step < study.n_steps; step++) {
        perturb_gauge(gauge, rng, study.eps);
        DiracOp D(lat, gauge, lcfg.mass, lcfg.wilson_r, lcfg.c_sw, lcfg.mu_t);
        Vec rhs = random_vec(lat.ndof, rng);
        OpApply A = [&D](const Vec& s, Vec& d){ D.apply_DdagD(s, d); };

        auto t0 = Clock::now();
        auto res_cg = cg_solve(A, lat.ndof, rhs, scfg.max_iter, scfg.tol);
        double dt_cg = Duration(Clock::now() - t0).count();

        t0 = Clock::now();
        auto identity_precond = [](const Vec& v) -> Vec { return v; };
        auto res_fgmres = fgmres_solve_generic(A, lat.ndof, rhs,
                                                 identity_precond,
                                                 scfg.krylov, scfg.max_iter, scfg.tol, 0);
        double dt_fgmres = Duration(Clock::now() - t0).count();

        total_t_cg += dt_cg;         total_it_cg += res_cg.iterations;
        total_t_fgmres += dt_fgmres;  total_it_fgmres += res_fgmres.iterations;

        std::cout << std::setw(5) << step
                  << std::setw(8) << std::fixed << std::setprecision(4) << gauge.avg_plaq()
                  << std::setw(8) << res_cg.iterations
                  << std::setw(10) << std::fixed << std::setprecision(4) << dt_cg
                  << std::setw(8) << res_fgmres.iterations
                  << std::setw(10) << std::fixed << std::setprecision(4) << dt_fgmres
                  << "\n";
    }

    std::cout << std::string(49, '-') << "\n";
    std::cout << std::setw(13) << "Total"
              << std::setw(8) << total_it_cg
              << std::setw(10) << std::fixed << std::setprecision(4) << total_t_cg
              << std::setw(8) << total_it_fgmres
              << std::setw(10) << std::fixed << std::setprecision(4) << total_t_fgmres
              << "\n";
    std::cout << std::setw(13) << "Avg/step"
              << std::setw(8) << std::setprecision(1) << (double)total_it_cg / study.n_steps
              << std::setw(10) << std::setprecision(4) << total_t_cg / study.n_steps
              << std::setw(8) << std::setprecision(1) << (double)total_it_fgmres / study.n_steps
              << std::setw(10) << std::setprecision(4) << total_t_fgmres / study.n_steps
              << "\n\n";

    std::cout << "CG speedup over FGMRES: " << std::fixed << std::setprecision(2)
              << total_t_fgmres / std::max(total_t_cg, 1e-30) << "x\n\n";

    return 0;
}

// =====================================================================
//  Default MG solver study
// =====================================================================
int run_mg_solver_study(GaugeField& gauge, const Lattice& lat,
                        const LatticeConfig& lcfg, const MGConfig& mcfg,
                        const SolverConfig& scfg, const StudyConfig& study,
                        std::mt19937& rng) {
    auto t_setup_start = Clock::now();

    DiracOp D0(lat, gauge, lcfg.mass, lcfg.wilson_r, lcfg.c_sw, lcfg.mu_t);

    std::unique_ptr<MGHierarchy> mg_stale;

    std::vector<Vec> null_vecs;
    Prolongator P_stale(lat, mcfg.block_size, mcfg.block_size, mcfg.k_null);
    CoarseOp Ac_stale;
    Prolongator P_ritz(lat, mcfg.block_size, mcfg.block_size, mcfg.k_null);
    CoarseOp Ac_ritz;

    std::cout << "--- Initial MG Setup ---\n";

    if (mcfg.mg_levels > 1) {
        std::cout << "Building hierarchy:\n";
        mg_stale = std::make_unique<MGHierarchy>(
            build_mg_hierarchy(D0, mcfg.mg_levels, mcfg.block_size, mcfg.k_null, mcfg.resolved_coarse_block(),
                               20, rng, mcfg.w_cycle, 3, 3));
    } else {
        null_vecs = compute_near_null_space(D0, mcfg.k_null, 20, rng);
        P_stale.build_from_vectors(null_vecs);
        Ac_stale.build(D0, P_stale);
        P_ritz.build_from_vectors(null_vecs);
        Ac_ritz.build(D0, P_ritz);
        std::cout << "Coarse dim: " << P_ritz.coarse_dim << "\n";
    }

    double t_setup = Duration(Clock::now() - t_setup_start).count();
    std::cout << "Setup time: " << std::fixed << std::setprecision(3)
              << t_setup << " s  (amortised over all solves)\n\n";

    // -----------------------------------------------------------------
    //  Evolve gauge field and compare strategies
    // -----------------------------------------------------------------
    if (mcfg.mg_levels > 1) {
        int R = study.refresh_interval;
        int n_coarse_defl = scfg.n_defl > 0 ? scfg.n_defl : mcfg.k_null * 4;

        std::mt19937 rng_wper(lcfg.seed);
        auto mg_fgm_warm = std::make_unique<MGHierarchy>(
            build_mg_hierarchy(D0, mcfg.mg_levels, mcfg.block_size, mcfg.k_null, mcfg.resolved_coarse_block(),
                               20, rng_wper, mcfg.w_cycle, 3, 3, false));

        std::mt19937 rng_sym_wper(lcfg.seed);
        auto mg_cgs_warm = std::make_unique<MGHierarchy>(
            build_mg_hierarchy(D0, mcfg.mg_levels, mcfg.block_size, mcfg.k_null, mcfg.resolved_coarse_block(),
                               20, rng_sym_wper, mcfg.w_cycle, 3, 3, false));
        mg_cgs_warm->set_symmetric(0.8);

        std::mt19937 rng_sym_defl(lcfg.seed);
        auto mg_cgs_defl = std::make_unique<MGHierarchy>(
            build_mg_hierarchy(D0, mcfg.mg_levels, mcfg.block_size, mcfg.k_null, mcfg.resolved_coarse_block(),
                               20, rng_sym_defl, mcfg.w_cycle, 3, 3, false));
        mg_cgs_defl->set_symmetric(0.8);
        {
            int last_l = (int)mg_cgs_defl->levels.size() - 1;
            int cdim = mg_cgs_defl->levels[last_l].Ac.dim;
            n_coarse_defl = std::min(n_coarse_defl, cdim / 2);
            n_coarse_defl = std::max(n_coarse_defl, 1);
        }
        OpApply A0 = [&D0](const Vec& s, Vec& d){ D0.apply_DdagD(s, d); };
        std::vector<Vec> cached_defl_warm_c;
        std::vector<Vec> cached_defl_warm_f;
        auto [cached_defl_vecs, cached_defl_vals] =
            mg_cgs_defl->build_fine_deflation(n_coarse_defl, A0, lat.ndof, 3,
                                               &cached_defl_warm_c, &cached_defl_warm_f);

        std::cout << "Initial coarse deflation eigenvalues (" << n_coarse_defl << " vectors):";
        for (int i = 0; i < (int)cached_defl_vals.size(); i++)
            std::cout << " " << std::scientific << std::setprecision(3) << cached_defl_vals[i];
        std::cout << std::fixed << "\n\n";

        std::cout << "--- Gauge Evolution: " << mcfg.mg_levels << "-Level "
                  << (mcfg.w_cycle ? "W" : "V") << "-Cycle MG ---\n";
        std::cout << "  CG=bare, CGsW=CG+symMG+WarmPer, CGsD=+deflation,\n"
                  << "  FGMw=FGMRES+MG+WarmPer, CGsF=CG+symMG+Fresh\n";
        std::cout << "  Warm refresh every " << R << " steps, "
                  << n_coarse_defl << " coarse deflation vectors\n\n";
        std::cout << std::setw(5)  << "Step"
                  << std::setw(8)  << "Plaq"
                  << std::setw(6)  << "CG"
                  << std::setw(7)  << "CGsW"
                  << std::setw(7)  << "CGsD"
                  << std::setw(7)  << "FGMw"
                  << std::setw(7)  << "CGsF"
                  << "\n";
        std::cout << std::string(47, '-') << "\n";

        double total_t_cg = 0, total_t_cgs_warm = 0, total_t_cgs_defl = 0;
        double total_t_fgm_warm = 0, total_t_cgs_fresh = 0;
        double total_t_fresh_setup = 0;
        double total_t_defl_update = 0;
        int    total_it_cg = 0, total_it_cgs_warm = 0, total_it_cgs_defl = 0;
        int    total_it_fgm_warm = 0, total_it_cgs_fresh = 0;

        for (int step = 0; step < study.n_steps; step++) {
            perturb_gauge(gauge, rng, study.eps);
            DiracOp D(lat, gauge, lcfg.mass, lcfg.wilson_r, lcfg.c_sw, lcfg.mu_t);
            Vec rhs = random_vec(lat.ndof, rng);
            OpApply A = [&D](const Vec& s, Vec& d){ D.apply_DdagD(s, d); };

            auto t0 = Clock::now();
            auto res_cg = cg_solve(A, lat.ndof, rhs, scfg.max_iter, scfg.tol);
            double dt_cg = Duration(Clock::now() - t0).count();
            total_t_cg += dt_cg;  total_it_cg += res_cg.iterations;

            // CG + symmetric MG WarmPer
            t0 = Clock::now();
            if (step % R == 0) {
                std::mt19937 rng_ref(lcfg.seed + step + 6000);
                auto warm_vecs = mg_cgs_warm->null_vecs_l0;
                *mg_cgs_warm = build_mg_hierarchy(D, mcfg.mg_levels, mcfg.block_size,
                    mcfg.k_null, mcfg.resolved_coarse_block(), 5, rng_ref, mcfg.w_cycle, 3, 3, false,
                    &warm_vecs);
            } else {
                mg_cgs_warm->levels[0].op = A;
                mg_cgs_warm->levels[0].Ac.build(D, mg_cgs_warm->geo_prolongators[0]);
                mg_cgs_warm->rebuild_deeper_levels();
            }
            mg_cgs_warm->set_symmetric(0.8);
            int it_cgs_warm;
            {
                auto precond_w = [&mg_cgs_warm](const Vec& v) { return mg_cgs_warm->precondition(v); };
                auto res = cg_solve_precond(A, lat.ndof, rhs, precond_w, scfg.max_iter, scfg.tol);
                total_t_cgs_warm += Duration(Clock::now() - t0).count();
                total_it_cgs_warm += res.iterations;
                it_cgs_warm = res.iterations;
            }

            // CG + symmetric MG WarmPer + coarse-grid deflation
            t0 = Clock::now();
            if (step % R == 0) {
                std::mt19937 rng_ref(lcfg.seed + step + 7000);
                auto warm_vecs = mg_cgs_defl->null_vecs_l0;
                *mg_cgs_defl = build_mg_hierarchy(D, mcfg.mg_levels, mcfg.block_size,
                    mcfg.k_null, mcfg.resolved_coarse_block(), 5, rng_ref, mcfg.w_cycle, 3, 3, false,
                    &warm_vecs);
            } else {
                mg_cgs_defl->levels[0].op = A;
                mg_cgs_defl->levels[0].Ac.build(D, mg_cgs_defl->geo_prolongators[0]);
                mg_cgs_defl->rebuild_deeper_levels();
            }
            mg_cgs_defl->set_symmetric(0.8);
            {
                auto t_defl = Clock::now();
                auto [dv, dl] = mg_cgs_defl->build_fine_deflation(
                    n_coarse_defl, A, lat.ndof, 3,
                    &cached_defl_warm_c, &cached_defl_warm_f);
                double dt_defl = Duration(Clock::now() - t_defl).count();
                total_t_defl_update += dt_defl;
                cached_defl_vecs = std::move(dv);
                cached_defl_vals = std::move(dl);
            }
            int it_cgs_defl;
            {
                auto precond_d = [&mg_cgs_defl](const Vec& v) { return mg_cgs_defl->precondition(v); };
                auto res = cg_solve_deflated(A, lat.ndof, rhs, precond_d,
                                              cached_defl_vecs, cached_defl_vals,
                                              scfg.max_iter, scfg.tol);
                total_t_cgs_defl += Duration(Clock::now() - t0).count();
                total_it_cgs_defl += res.iterations;
                it_cgs_defl = res.iterations;
            }

            // FGMRES + MG WarmPer
            t0 = Clock::now();
            if (step % R == 0) {
                std::mt19937 rng_ref(lcfg.seed + step + 4000);
                auto warm_vecs = mg_fgm_warm->null_vecs_l0;
                *mg_fgm_warm = build_mg_hierarchy(D, mcfg.mg_levels, mcfg.block_size,
                    mcfg.k_null, mcfg.resolved_coarse_block(), 5, rng_ref, mcfg.w_cycle, 3, 3, false,
                    &warm_vecs);
            } else {
                mg_fgm_warm->levels[0].op = A;
                mg_fgm_warm->levels[0].Ac.build(D, mg_fgm_warm->geo_prolongators[0]);
                mg_fgm_warm->rebuild_deeper_levels();
            }
            int it_fgm_warm;
            {
                auto res = fgmres_solve_mg(A, lat.ndof, rhs, *mg_fgm_warm,
                                            scfg.krylov, scfg.max_iter, scfg.tol, 0);
                total_t_fgm_warm += Duration(Clock::now() - t0).count();
                total_it_fgm_warm += res.iterations;
                it_fgm_warm = res.iterations;
            }

            // CG + symmetric MG Fresh
            t0 = Clock::now();
            std::mt19937 rng_fresh(lcfg.seed + step + 1000);
            auto mg_fresh = build_mg_hierarchy(D, mcfg.mg_levels, mcfg.block_size,
                mcfg.k_null, mcfg.resolved_coarse_block(), 20, rng_fresh, mcfg.w_cycle, 3, 3, false);
            mg_fresh.set_symmetric(0.8);
            double dt_fresh_setup = Duration(Clock::now() - t0).count();
            total_t_fresh_setup += dt_fresh_setup;
            int it_cgs_fresh;
            {
                t0 = Clock::now();
                auto precond_f = [&mg_fresh](const Vec& v) { return mg_fresh.precondition(v); };
                auto res = cg_solve_precond(A, lat.ndof, rhs, precond_f, scfg.max_iter, scfg.tol);
                total_t_cgs_fresh += dt_fresh_setup + Duration(Clock::now() - t0).count();
                total_it_cgs_fresh += res.iterations;
                it_cgs_fresh = res.iterations;
            }

            std::cout << std::setw(5) << step
                      << std::setw(8) << std::fixed << std::setprecision(4) << gauge.avg_plaq()
                      << std::setw(6) << res_cg.iterations
                      << std::setw(7) << it_cgs_warm
                      << std::setw(7) << it_cgs_defl
                      << std::setw(7) << it_fgm_warm
                      << std::setw(7) << it_cgs_fresh;
            if (step % R == 0) std::cout << " W";
            if (!cached_defl_vals.empty())
                std::cout << "  emin=" << std::scientific << std::setprecision(2)
                          << cached_defl_vals[0] << std::fixed;
            std::cout << "\n";
        }

        std::cout << std::string(47, '-') << "\n";
        std::cout << std::fixed;

        std::cout << "\n--- Timing Summary (incl. " << std::setprecision(2)
                  << t_setup << "s initial setup, amortised) ---\n";

        double setup_amort = t_setup;
        auto pct = [](double v, double ref) {
            return 100.0 * v / std::max(ref, 1e-30);
        };

        double eff_cg        = total_t_cg;
        double eff_cgs_warm  = total_t_cgs_warm + setup_amort;
        double eff_cgs_defl  = total_t_cgs_defl + setup_amort;
        double eff_fgm_warm  = total_t_fgm_warm + setup_amort;
        double eff_cgs_fresh = total_t_cgs_fresh;

        std::cout << "  Strategy        Time(s)   vs CG    Iters   Iters/solve\n";
        std::cout << "  CG (bare)   " << std::setw(10) << std::setprecision(4) << eff_cg
                  << std::setw(8) << std::setprecision(1) << 100.0 << "%"
                  << std::setw(8) << total_it_cg
                  << std::setw(10) << std::setprecision(1) << (double)total_it_cg / study.n_steps << "\n";
        std::cout << "  CG+sMG Wm   " << std::setw(10) << std::setprecision(4) << eff_cgs_warm
                  << std::setw(8) << std::setprecision(1) << pct(eff_cgs_warm, eff_cg) << "%"
                  << std::setw(8) << total_it_cgs_warm
                  << std::setw(10) << std::setprecision(1) << (double)total_it_cgs_warm / study.n_steps << "\n";
        std::cout << "  CG+sMG WmDf " << std::setw(10) << std::setprecision(4) << eff_cgs_defl
                  << std::setw(8) << std::setprecision(1) << pct(eff_cgs_defl, eff_cg) << "%"
                  << std::setw(8) << total_it_cgs_defl
                  << std::setw(10) << std::setprecision(1) << (double)total_it_cgs_defl / study.n_steps << "\n";
        std::cout << "  FGM+MG Wm   " << std::setw(10) << std::setprecision(4) << eff_fgm_warm
                  << std::setw(8) << std::setprecision(1) << pct(eff_fgm_warm, eff_cg) << "%"
                  << std::setw(8) << total_it_fgm_warm
                  << std::setw(10) << std::setprecision(1) << (double)total_it_fgm_warm / study.n_steps << "\n";
        std::cout << "  CG+sMG Fr   " << std::setw(10) << std::setprecision(4) << eff_cgs_fresh
                  << std::setw(8) << std::setprecision(1) << pct(eff_cgs_fresh, eff_cg) << "%"
                  << std::setw(8) << total_it_cgs_fresh
                  << std::setw(10) << std::setprecision(1) << (double)total_it_cgs_fresh / study.n_steps << "\n";
        std::cout << "  (W = warm refresh step, " << n_coarse_defl << " coarse deflation vecs)\n";
        std::cout << "  (Coarse deflation update: " << std::setprecision(4)
                  << total_t_defl_update << "s total, "
                  << std::setprecision(4) << total_t_defl_update / study.n_steps << "s/step)\n";
        std::cout << "  (Fresh setup: " << std::setprecision(4) << total_t_fresh_setup << "s total)\n\n";

    } else {
        // Single-level mode: 3-strategy comparison
        std::cout << "--- Gauge Evolution: Comparing Deflation Strategies ---\n";
        std::cout << std::setw(5)  << "Step"
                  << std::setw(8)  << "Plaq"
                  << std::setw(8)  << "Stale"
                  << std::setw(10) << "t_stale"
                  << std::setw(8)  << "Ritz"
                  << std::setw(10) << "t_ritz"
                  << std::setw(8)  << "Fresh"
                  << std::setw(10) << "t_fresh"
                  << std::setw(12) << "MinRitz"
                  << "\n";
        std::cout << std::string(79, '-') << "\n";

        double total_t_stale = 0, total_t_ritz = 0, total_t_fresh = 0;
        double total_t_fresh_setup = 0, total_t_fresh_solve = 0;
        int    total_it_stale = 0, total_it_ritz = 0, total_it_fresh = 0;

        for (int step = 0; step < study.n_steps; step++) {
            perturb_gauge(gauge, rng, study.eps);
            DiracOp D(lat, gauge, lcfg.mass, lcfg.wilson_r, lcfg.c_sw, lcfg.mu_t);
            Vec rhs = random_vec(lat.ndof, rng);

            auto t0 = Clock::now();
            Ac_stale.build(D, P_stale);
            auto res_stale = fgmres_solve(D, rhs, P_stale, Ac_stale,
                                           scfg.krylov, scfg.max_iter, scfg.tol, 3, 3, 0);
            double dt_stale = Duration(Clock::now() - t0).count();

            t0 = Clock::now();
            Ac_ritz.build(D, P_ritz);
            auto res_ritz = fgmres_solve(D, rhs, P_ritz, Ac_ritz,
                                          scfg.krylov, scfg.max_iter, scfg.tol, 3, 3, 4);
            double dt_ritz = Duration(Clock::now() - t0).count();

            t0 = Clock::now();
            auto fresh_null = compute_near_null_space(D, mcfg.k_null, 20, rng);
            Prolongator P_fresh(lat, mcfg.block_size, mcfg.block_size, mcfg.k_null);
            P_fresh.build_from_vectors(fresh_null);
            CoarseOp Ac_fresh;
            Ac_fresh.build(D, P_fresh);
            double dt_fresh_setup = Duration(Clock::now() - t0).count();

            t0 = Clock::now();
            auto res_fresh = fgmres_solve(D, rhs, P_fresh, Ac_fresh,
                                           scfg.krylov, scfg.max_iter, scfg.tol, 3, 3, 0);
            double dt_fresh_solve = Duration(Clock::now() - t0).count();
            double dt_fresh = dt_fresh_setup + dt_fresh_solve;
            total_t_fresh_setup += dt_fresh_setup;
            total_t_fresh_solve += dt_fresh_solve;

            total_t_stale += dt_stale;  total_it_stale += res_stale.iterations;
            total_t_ritz  += dt_ritz;   total_it_ritz  += res_ritz.iterations;
            total_t_fresh += dt_fresh;  total_it_fresh += res_fresh.iterations;

            std::cout << std::setw(5)  << step
                      << std::setw(8)  << std::fixed << std::setprecision(4) << gauge.avg_plaq()
                      << std::setw(8)  << res_stale.iterations
                      << std::setw(10) << std::fixed << std::setprecision(4) << dt_stale
                      << std::setw(8)  << res_ritz.iterations
                      << std::setw(10) << std::fixed << std::setprecision(4) << dt_ritz
                      << std::setw(8)  << res_fresh.iterations
                      << std::setw(10) << std::fixed << std::setprecision(4) << dt_fresh;
            if (!res_ritz.ritz_pairs.empty())
                std::cout << std::setw(12) << std::scientific
                          << std::setprecision(2) << res_ritz.ritz_pairs[0].value;
            std::cout << "\n";
        }

        std::cout << std::string(79, '-') << "\n";
        std::cout << std::fixed << std::setprecision(4);
        std::cout << std::setw(13) << "Total"
                  << std::setw(8)  << total_it_stale
                  << std::setw(10) << total_t_stale
                  << std::setw(8)  << total_it_ritz
                  << std::setw(10) << total_t_ritz
                  << std::setw(8)  << total_it_fresh
                  << std::setw(10) << total_t_fresh
                  << "\n";
        std::cout << std::setw(13) << "Avg/step"
                  << std::setw(8)  << std::setprecision(1) << (double)total_it_stale / study.n_steps
                  << std::setw(10) << std::setprecision(4) << total_t_stale / study.n_steps
                  << std::setw(8)  << std::setprecision(1) << (double)total_it_ritz / study.n_steps
                  << std::setw(10) << std::setprecision(4) << total_t_ritz / study.n_steps
                  << std::setw(8)  << std::setprecision(1) << (double)total_it_fresh / study.n_steps
                  << std::setw(10) << std::setprecision(4) << total_t_fresh / study.n_steps
                  << "\n\n";

        double speedup_vs_fresh = total_t_fresh / std::max(total_t_ritz, 1e-30);
        std::cout << "Ritz is " << std::fixed << std::setprecision(2)
                  << speedup_vs_fresh << "x faster than Fresh (total wall time)\n";
        std::cout << "Ritz uses " << std::fixed << std::setprecision(1)
                  << 100.0 * total_t_ritz / std::max(total_t_fresh, 1e-30)
                  << "% of Fresh wall time for "
                  << std::setprecision(1)
                  << 100.0 * total_it_ritz / std::max((double)total_it_fresh, 1e-30)
                  << "% of Fresh iterations\n\n";

        std::cout << "--- Timing Breakdown (total seconds) ---\n";
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "  Ritz:   total=" << total_t_ritz
                  << " (intra-solve deflation, no cross-step overhead)\n";
        std::cout << "  Fresh:  setup=" << total_t_fresh_setup
                  << "  solve=" << total_t_fresh_solve
                  << "  total=" << total_t_fresh << "\n";
        std::cout << "  Ritz overhead vs Stale solve: "
                  << std::setprecision(1)
                  << 100.0 * (total_t_ritz - total_t_stale) / std::max(total_t_stale, 1e-30)
                  << "%  (Ritz extraction + update cost)\n";
        std::cout << "  Fresh setup is "
                  << std::setprecision(1)
                  << 100.0 * total_t_fresh_setup / std::max(total_t_fresh, 1e-30)
                  << "% of Fresh total time\n\n";
    }

    std::cout << "================================================================\n"
              << "  'CG'        = unpreconditioned CG (baseline)\n"
              << "  'CG+sMG Wm' = CG + symmetric MG + WarmPer refresh\n"
              << "  'CG+sMG WmDf' = + LOBPCG coarse-grid deflation\n"
              << "  'FGM+MG Wm' = FGMRES + MG (MR smoother) + WarmPer\n"
              << "  'CG+sMG Fr' = CG + symmetric MG, fresh setup (gold)\n"
              << "================================================================\n";

    return 0;
}
