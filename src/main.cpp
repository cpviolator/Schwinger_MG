// main.cpp — CLI and benchmark driver for 2D Schwinger MG solver
#include "types.h"
#include "linalg.h"
#include "lattice.h"
#include "gauge.h"
#include "dirac.h"
#include "prolongator.h"
#include "coarse_op.h"
#include "eigensolver.h"
#include "smoother.h"
#include "multigrid.h"
#include "solvers.h"
#include "hmc.h"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstring>
#include <string>
#include <memory>
#include <omp.h>

using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double>;

void print_usage(const char* prog) {
    std::cout
        << "Usage: " << prog << " [options]\n"
        << "\n"
        << "2D Schwinger model MG solver with Ritz-maintained subspace.\n"
        << "Compares three deflation strategies as the gauge field evolves.\n"
        << "\n"
        << "Options:\n"
        << "  -L <int>       Lattice size (LxL)             [default: 16]\n"
        << "  -m <float>     Fermion mass                   [default: 0.05]\n"
        << "  -e <float>     Gauge perturbation per step    [default: 0.12]\n"
        << "  -b <int>       Block size (bxb)               [default: 4]\n"
        << "  -k <int>       Null vectors per block          [default: 4]\n"
        << "  -n <int>       Number of MD steps             [default: 30]\n"
        << "  -s <int>       RNG seed                       [default: 42]\n"
        << "  -w <float>     Hot-start width (initial U)    [default: 0.4]\n"
        << "  -r <float>     Wilson parameter               [default: 1.0]\n"
        << "  -t <int>       Number of OpenMP threads        [default: all cores]\n"
        << "  --tol <float>  FGMRES tolerance               [default: 1e-10]\n"
        << "  --krylov <int> Krylov subspace dimension      [default: 30]\n"
        << "  --maxiter <int> Max FGMRES iterations         [default: 300]\n"
        << "  --mg-levels <int> Number of MG levels          [default: 1]\n"
        << "  --w-cycle      Use W-cycle (default for levels>1)\n"
        << "  --v-cycle      Use V-cycle\n"
        << "  --coarse-block <int> Coarse-level block size   [default: k*4]\n"
        << "  --refresh <int> Periodic refresh interval    [default: 3]\n"
        << "  --adaptive-threshold <float> Iter ratio trigger [default: 1.3]\n"
        << "  --help, -h     Show this help message\n"
        << "\n"
        << "HMC options:\n"
        << "  --hmc          Run HMC instead of MG benchmark\n"
        << "  --hmc-traj <int>   Number of trajectories          [default: 100]\n"
        << "  --hmc-tau <float>  Trajectory length                [default: 1.0]\n"
        << "  --hmc-steps <int>  Leapfrog steps per trajectory    [default: 20]\n"
        << "  --hmc-beta <float> Gauge coupling beta              [default: 2.0]\n"
        << "  --hmc-therm <int>  Thermalisation trajectories       [default: 20]\n"
        << "  --hmc-save-every <int>  Save every N trajectories   [default: 10]\n"
        << "  --hmc-save-prefix <str> Gauge file prefix           [default: gauge]\n"
        << "  --hmc-load <file>  Load initial gauge configuration\n"
        << "\n"
        << "Examples:\n"
        << "  " << prog << " -L 32 -m 0.01 -n 50       # larger lattice, lighter mass\n"
        << "  " << prog << " -L 8 -b 2 -k 8            # small lattice, more null vecs\n"
        << "  " << prog << " -e 0.05 -n 100             # slow drift, many steps\n"
        << "  " << prog << " -L 32 --mg-levels 2        # 2-level W-cycle MG\n"
        << "  " << prog << " -L 64 --mg-levels 3 -k 8   # 3-level MG\n"
        << "  " << prog << " --hmc -L 16 -m 0.01 --hmc-beta 4.0 --hmc-traj 200\n";
}

int main(int argc, char** argv) {
    // --- defaults ---
    int    L         = 16;
    double mass      = 0.05;
    double eps       = 0.12;
    int    block_size= 4;
    int    k_null    = 4;
    int    n_steps   = 30;
    int    seed      = 42;
    double hot_width = 0.4;
    double wilson_r  = 1.0;
    double tol       = 1e-10;
    int    krylov    = 30;
    int    max_iter  = 300;
    int    n_threads = 0;
    int    mg_levels = 1;
    int    coarse_block = 0;
    int    cycle_type = -1;
    int    refresh_interval = 3;
    double adaptive_threshold = 1.3;
    bool   no_mg     = false;
    bool   symmetric_mg = false;
    bool   test_deflation = false;
    bool   cheb_only = false;
    int    n_defl_vecs = 0;
    bool   run_hmc   = false;
    int    hmc_traj  = 100;
    double hmc_tau   = 1.0;
    int    hmc_steps = 20;
    double hmc_beta  = 2.0;
    int    hmc_therm = 20;
    int    hmc_save_every = 10;
    std::string hmc_save_prefix = "gauge";
    std::string hmc_load_file;

    // --- parse args ---
    for (int i = 1; i < argc; i++) {
        auto match = [&](const char* flag) { return std::strcmp(argv[i], flag) == 0; };
        auto next_int = [&]() { return std::atoi(argv[++i]); };
        auto next_dbl = [&]() { return std::atof(argv[++i]); };

        if (match("-h") || match("--help")) { print_usage(argv[0]); return 0; }
        else if (match("-L"))        L         = next_int();
        else if (match("-m"))        mass      = next_dbl();
        else if (match("-e"))        eps       = next_dbl();
        else if (match("-b"))        block_size= next_int();
        else if (match("-k"))        k_null    = next_int();
        else if (match("-n"))        n_steps   = next_int();
        else if (match("-s"))        seed      = next_int();
        else if (match("-w"))        hot_width = next_dbl();
        else if (match("-r"))        wilson_r  = next_dbl();
        else if (match("--tol"))     tol       = next_dbl();
        else if (match("--krylov"))  krylov    = next_int();
        else if (match("-t"))        n_threads = next_int();
        else if (match("--maxiter")) max_iter  = next_int();
        else if (match("--mg-levels")) mg_levels = next_int();
        else if (match("--coarse-block")) coarse_block = next_int();
        else if (match("--refresh")) refresh_interval = next_int();
        else if (match("--adaptive-threshold")) adaptive_threshold = next_dbl();
        else if (match("--w-cycle")) cycle_type = 1;
        else if (match("--v-cycle")) cycle_type = 0;
        else if (match("--no-mg"))   no_mg = true;
        else if (match("--symmetric-mg")) symmetric_mg = true;
        else if (match("--hmc"))     run_hmc = true;
        else if (match("--hmc-traj"))  hmc_traj  = next_int();
        else if (match("--hmc-tau"))   hmc_tau   = next_dbl();
        else if (match("--hmc-steps")) hmc_steps = next_int();
        else if (match("--hmc-beta"))  hmc_beta  = next_dbl();
        else if (match("--hmc-therm")) hmc_therm = next_int();
        else if (match("--hmc-save-every")) hmc_save_every = next_int();
        else if (match("--hmc-save-prefix")) hmc_save_prefix = argv[++i];
        else if (match("--hmc-load")) hmc_load_file = argv[++i];
        else if (match("--test-deflation")) test_deflation = true;
        else if (match("--cheb-only")) { test_deflation = true; cheb_only = true; }
        else if (match("--n-defl")) n_defl_vecs = next_int();
        else {
            std::cerr << "Unknown option: " << argv[i] << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    if (cycle_type < 0) cycle_type = (mg_levels > 1) ? 1 : 0;
    bool w_cycle = (cycle_type == 1);
    if (coarse_block == 0) coarse_block = k_null * 4;

    if (L % block_size != 0) {
        std::cerr << "Error: L (" << L << ") must be divisible by block size ("
                  << block_size << ")\n";
        return 1;
    }
    if (L < 4 || block_size < 1 || k_null < 1 || n_steps < 1) {
        std::cerr << "Error: invalid parameter (L >= 4, b >= 1, k >= 1, n >= 1)\n";
        return 1;
    }

    if (n_threads > 0) omp_set_num_threads(n_threads);

    std::cout << "================================================================\n"
              << "  2D Schwinger Model: MG with EigCG-Inspired Ritz Harvesting\n"
              << "================================================================\n\n";

    std::mt19937 rng(seed);
    Lattice lat(L);

    GaugeField gauge(lat);
    gauge.randomise(rng, hot_width);

    std::cout << "Lattice: " << L << "x" << L
              << "  DOF: " << lat.ndof
              << "  mass: " << mass
              << "  r: " << wilson_r << "\n";
    std::cout << "Block: " << block_size << "x" << block_size
              << "  Null vectors/block: " << k_null << "\n";
    std::cout << "MG levels: " << mg_levels
              << "  cycle: " << (w_cycle ? "W" : "V")
              << (mg_levels > 1 ? "  coarse_block: " + std::to_string(coarse_block)
                                 + "  refresh: " + std::to_string(refresh_interval)
                                : "")
              << "\n";
    std::cout << "Krylov: " << krylov
              << "  maxiter: " << max_iter
              << "  tol: " << std::scientific << std::setprecision(1) << tol << "\n";
    std::cout << "OpenMP threads: " << omp_get_max_threads() << "\n";
    std::cout << "Gauge perturbation per step: epsilon = " << std::fixed
              << std::setprecision(4) << eps
              << "  steps: " << n_steps
              << "  seed: " << seed << "\n";
    if (!hmc_load_file.empty()) {
        GaugeHeader hdr;
        if (!load_gauge(gauge, hdr, hmc_load_file)) return 1;
        std::cout << "Loaded gauge from " << hmc_load_file
                  << " (beta=" << hdr.beta << " mass=" << hdr.mass
                  << " plaq=" << hdr.avg_plaq << ")\n";
    }

    std::cout << "Initial <plaq> = " << std::fixed << std::setprecision(4)
              << gauge.avg_plaq() << "\n\n";

    // -----------------------------------------------------------------
    //  HMC mode
    // -----------------------------------------------------------------
    if (run_hmc) {
        std::cout << "=== HMC Mode ===\n";
        std::cout << "beta=" << hmc_beta << "  tau=" << hmc_tau
                  << "  steps=" << hmc_steps << "  dt=" << hmc_tau/hmc_steps << "\n";
        std::cout << "trajectories=" << hmc_traj << "  therm=" << hmc_therm
                  << "  save_every=" << hmc_save_every << "\n\n";

        HMCParams params;
        params.beta = hmc_beta;
        params.tau = hmc_tau;
        params.n_steps = hmc_steps;
        params.cg_maxiter = max_iter;
        params.cg_tol = tol;
        params.use_mg = false;

        int n_accept = 0;
        int total_traj = hmc_therm + hmc_traj;

        std::cout << std::setw(6) << "Traj"
                  << std::setw(8) << "Plaq"
                  << std::setw(10) << "dH"
                  << std::setw(6) << "Acc"
                  << std::setw(8) << "Rate"
                  << std::setw(8) << "CG_it"
                  << std::setw(10) << "Time(s)"
                  << "\n";
        std::cout << std::string(56, '-') << "\n";

        int saved_count = 0;
        for (int traj = 0; traj < total_traj; traj++) {
            auto t0 = Clock::now();
            auto result = hmc_trajectory(gauge, lat, mass, wilson_r, params, rng);
            double dt_traj = Duration(Clock::now() - t0).count();

            if (traj >= hmc_therm) n_accept += result.accepted;
            int measurement_traj = (traj >= hmc_therm) ? (traj - hmc_therm + 1) : 0;
            double rate = (measurement_traj > 0) ? (double)n_accept / measurement_traj : 0.0;

            std::cout << std::setw(6) << traj
                      << std::setw(8) << std::fixed << std::setprecision(4) << gauge.avg_plaq()
                      << std::setw(10) << std::fixed << std::setprecision(4) << result.dH
                      << std::setw(6) << (result.accepted ? "Y" : "N")
                      << std::setw(7) << std::fixed << std::setprecision(2)
                      << (traj >= hmc_therm ? 100.0*rate : 0.0) << "%"
                      << std::setw(8) << result.total_cg_iters
                      << std::setw(10) << std::fixed << std::setprecision(2) << dt_traj;
            if (traj < hmc_therm) std::cout << " [therm]";
            std::cout << "\n";

            if (traj >= hmc_therm && hmc_save_every > 0 &&
                (traj - hmc_therm) % hmc_save_every == 0) {
                std::string fname = hmc_save_prefix + "_L" + std::to_string(L)
                    + "_b" + std::to_string((int)(hmc_beta*100))
                    + "_m" + std::to_string((int)(mass*10000))
                    + "_" + std::to_string(saved_count) + ".bin";
                save_gauge(gauge, hmc_beta, mass, fname);
                std::cout << "  -> saved " << fname << "\n";
                saved_count++;
            }
        }

        std::cout << "\n=== HMC Summary ===\n";
        std::cout << "Final <plaq> = " << std::fixed << std::setprecision(6) << gauge.avg_plaq() << "\n";
        std::cout << "Acceptance rate (post-therm): " << std::fixed << std::setprecision(1)
                  << 100.0 * n_accept / hmc_traj << "%\n";
        std::cout << "Configs saved: " << saved_count << "\n";

        return 0;
    }

    // -----------------------------------------------------------------
    //  Test deflation mode
    // -----------------------------------------------------------------
    if (test_deflation) {
        if (mg_levels < 2) {
            std::cerr << "--test-deflation requires --mg-levels >= 2\n";
            return 1;
        }
        int ndefl = n_defl_vecs > 0 ? n_defl_vecs : k_null * 4;
        std::cout << "=== Test: Coarse Eigenvector Prolongation ===\n\n";

        DiracOp D(lat, gauge, mass, wilson_r);
        OpApply A = [&D](const Vec& s, Vec& d){ D.apply_DdagD(s, d); };
        std::mt19937 rng_mg(seed);
        auto mg = build_mg_hierarchy(D, mg_levels, block_size, k_null,
                                      coarse_block, 20, rng_mg, w_cycle,
                                      3, 3, true);
        mg.set_symmetric(0.8);

        int last = (int)mg.levels.size() - 1;
        int cdim = mg.levels[last].Ac.dim;
        ndefl = std::min(ndefl, cdim / 2);
        std::cout << "Coarsest level dim: " << cdim
                  << ", deflation vectors: " << ndefl << "\n\n";

        if (!cheb_only) {
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
        auto res_cg = cg_solve_precond(A, lat.ndof, rhs, precond, max_iter, tol);
        double t_cg = Duration(Clock::now() - t0).count();

        t0 = Clock::now();
        auto res_defl_c = cg_solve_deflated(A, lat.ndof, rhs, precond,
                                             fine_vecs, fine_rq, max_iter, tol);
        double t_defl_c = Duration(Clock::now() - t0).count();

        std::vector<double> refined_rq(ndefl);
        for (int i = 0; i < ndefl; i++) {
            Vec Av(lat.ndof); A(refined_vecs[i], Av);
            refined_rq[i] = std::real(dot(refined_vecs[i], Av));
        }
        t0 = Clock::now();
        auto res_defl_r = cg_solve_deflated(A, lat.ndof, rhs, precond,
                                             refined_vecs, refined_rq,
                                             max_iter, tol);
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
                                                max_iter, tol);
        double t_defl_cheb = Duration(Clock::now() - t0).count();

        t0 = Clock::now();
        auto res_bare = cg_solve(A, lat.ndof, rhs, max_iter, tol);
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
        std::mt19937 rng_m7(seed + 7777);
        // Use raw n_defl_vecs (not coarse-clamped) for the eigensolver study
        int n_ev = n_defl_vecs > 0 ? n_defl_vecs : ndefl;
        n_ev = std::min(n_ev, lat.ndof / 2);  // don't exceed half the matrix dim
        std::cout << "\n=== TRLM Chebyshev Study: L=" << L
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

            // 4) deg × n_kr scan at best a_min = ev_next * 1.1
            //    (just above the first unwanted eigenvalue)
            double best_amin = ev_next * 1.1;
            std::cout << "--- deg x n_kr scan (a_min=" << std::scientific
                      << std::setprecision(3) << best_amin
                      << ", a_max=" << cheb_amax << ") ---\n";
            print_header();
            for (int deg : {4, 8, 16, 32, 64}) {
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

        return 0;
    }

    // -----------------------------------------------------------------
    //  No-MG baseline mode
    // -----------------------------------------------------------------
    if (no_mg) {
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

        for (int step = 0; step < n_steps; step++) {
            perturb_gauge(gauge, rng, eps);
            DiracOp D(lat, gauge, mass, wilson_r);
            Vec rhs = random_vec(lat.ndof, rng);
            OpApply A = [&D](const Vec& s, Vec& d){ D.apply_DdagD(s, d); };

            auto t0 = Clock::now();
            auto res_cg = cg_solve(A, lat.ndof, rhs, max_iter, tol);
            double dt_cg = Duration(Clock::now() - t0).count();

            t0 = Clock::now();
            auto identity_precond = [](const Vec& v) -> Vec { return v; };
            auto res_fgmres = fgmres_solve_generic(A, lat.ndof, rhs,
                                                     identity_precond,
                                                     krylov, max_iter, tol, 0);
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
                  << std::setw(8) << std::setprecision(1) << (double)total_it_cg / n_steps
                  << std::setw(10) << std::setprecision(4) << total_t_cg / n_steps
                  << std::setw(8) << std::setprecision(1) << (double)total_it_fgmres / n_steps
                  << std::setw(10) << std::setprecision(4) << total_t_fgmres / n_steps
                  << "\n\n";

        std::cout << "CG speedup over FGMRES: " << std::fixed << std::setprecision(2)
                  << total_t_fgmres / std::max(total_t_cg, 1e-30) << "x\n\n";

        return 0;
    }

    // -----------------------------------------------------------------
    //  Initial MG setup
    // -----------------------------------------------------------------
    auto t_setup_start = Clock::now();

    DiracOp D0(lat, gauge, mass, wilson_r);

    std::unique_ptr<MGHierarchy> mg_stale;

    std::vector<Vec> null_vecs;
    Prolongator P_stale(lat, block_size, block_size, k_null);
    CoarseOp Ac_stale;
    Prolongator P_ritz(lat, block_size, block_size, k_null);
    CoarseOp Ac_ritz;

    std::cout << "--- Initial MG Setup ---\n";

    if (mg_levels > 1) {
        std::cout << "Building hierarchy:\n";
        mg_stale = std::make_unique<MGHierarchy>(
            build_mg_hierarchy(D0, mg_levels, block_size, k_null, coarse_block,
                               20, rng, w_cycle, 3, 3));
    } else {
        null_vecs = compute_near_null_space(D0, k_null, 20, rng);
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
    if (mg_levels > 1) {
        int R = refresh_interval;
        int n_coarse_defl = n_defl_vecs > 0 ? n_defl_vecs : k_null * 4;

        std::mt19937 rng_wper(seed);
        auto mg_fgm_warm = std::make_unique<MGHierarchy>(
            build_mg_hierarchy(D0, mg_levels, block_size, k_null, coarse_block,
                               20, rng_wper, w_cycle, 3, 3, false));

        std::mt19937 rng_sym_wper(seed);
        auto mg_cgs_warm = std::make_unique<MGHierarchy>(
            build_mg_hierarchy(D0, mg_levels, block_size, k_null, coarse_block,
                               20, rng_sym_wper, w_cycle, 3, 3, false));
        mg_cgs_warm->set_symmetric(0.8);

        std::mt19937 rng_sym_defl(seed);
        auto mg_cgs_defl = std::make_unique<MGHierarchy>(
            build_mg_hierarchy(D0, mg_levels, block_size, k_null, coarse_block,
                               20, rng_sym_defl, w_cycle, 3, 3, false));
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

        std::cout << "--- Gauge Evolution: " << mg_levels << "-Level "
                  << (w_cycle ? "W" : "V") << "-Cycle MG ---\n";
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

        for (int step = 0; step < n_steps; step++) {
            perturb_gauge(gauge, rng, eps);
            DiracOp D(lat, gauge, mass, wilson_r);
            Vec rhs = random_vec(lat.ndof, rng);
            OpApply A = [&D](const Vec& s, Vec& d){ D.apply_DdagD(s, d); };

            auto t0 = Clock::now();
            auto res_cg = cg_solve(A, lat.ndof, rhs, max_iter, tol);
            double dt_cg = Duration(Clock::now() - t0).count();
            total_t_cg += dt_cg;  total_it_cg += res_cg.iterations;

            // CG + symmetric MG WarmPer
            t0 = Clock::now();
            if (step % R == 0) {
                std::mt19937 rng_ref(seed + step + 6000);
                auto warm_vecs = mg_cgs_warm->null_vecs_l0;
                *mg_cgs_warm = build_mg_hierarchy(D, mg_levels, block_size,
                    k_null, coarse_block, 5, rng_ref, w_cycle, 3, 3, false,
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
                auto res = cg_solve_precond(A, lat.ndof, rhs, precond_w, max_iter, tol);
                total_t_cgs_warm += Duration(Clock::now() - t0).count();
                total_it_cgs_warm += res.iterations;
                it_cgs_warm = res.iterations;
            }

            // CG + symmetric MG WarmPer + coarse-grid deflation
            t0 = Clock::now();
            if (step % R == 0) {
                std::mt19937 rng_ref(seed + step + 7000);
                auto warm_vecs = mg_cgs_defl->null_vecs_l0;
                *mg_cgs_defl = build_mg_hierarchy(D, mg_levels, block_size,
                    k_null, coarse_block, 5, rng_ref, w_cycle, 3, 3, false,
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
                                              max_iter, tol);
                total_t_cgs_defl += Duration(Clock::now() - t0).count();
                total_it_cgs_defl += res.iterations;
                it_cgs_defl = res.iterations;
            }

            // FGMRES + MG WarmPer
            t0 = Clock::now();
            if (step % R == 0) {
                std::mt19937 rng_ref(seed + step + 4000);
                auto warm_vecs = mg_fgm_warm->null_vecs_l0;
                *mg_fgm_warm = build_mg_hierarchy(D, mg_levels, block_size,
                    k_null, coarse_block, 5, rng_ref, w_cycle, 3, 3, false,
                    &warm_vecs);
            } else {
                mg_fgm_warm->levels[0].op = A;
                mg_fgm_warm->levels[0].Ac.build(D, mg_fgm_warm->geo_prolongators[0]);
                mg_fgm_warm->rebuild_deeper_levels();
            }
            int it_fgm_warm;
            {
                auto res = fgmres_solve_mg(A, lat.ndof, rhs, *mg_fgm_warm,
                                            krylov, max_iter, tol, 0);
                total_t_fgm_warm += Duration(Clock::now() - t0).count();
                total_it_fgm_warm += res.iterations;
                it_fgm_warm = res.iterations;
            }

            // CG + symmetric MG Fresh
            t0 = Clock::now();
            std::mt19937 rng_fresh(seed + step + 1000);
            auto mg_fresh = build_mg_hierarchy(D, mg_levels, block_size,
                k_null, coarse_block, 20, rng_fresh, w_cycle, 3, 3, false);
            mg_fresh.set_symmetric(0.8);
            double dt_fresh_setup = Duration(Clock::now() - t0).count();
            total_t_fresh_setup += dt_fresh_setup;
            int it_cgs_fresh;
            {
                t0 = Clock::now();
                auto precond_f = [&mg_fresh](const Vec& v) { return mg_fresh.precondition(v); };
                auto res = cg_solve_precond(A, lat.ndof, rhs, precond_f, max_iter, tol);
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
                  << std::setw(10) << std::setprecision(1) << (double)total_it_cg / n_steps << "\n";
        std::cout << "  CG+sMG Wm   " << std::setw(10) << std::setprecision(4) << eff_cgs_warm
                  << std::setw(8) << std::setprecision(1) << pct(eff_cgs_warm, eff_cg) << "%"
                  << std::setw(8) << total_it_cgs_warm
                  << std::setw(10) << std::setprecision(1) << (double)total_it_cgs_warm / n_steps << "\n";
        std::cout << "  CG+sMG WmDf " << std::setw(10) << std::setprecision(4) << eff_cgs_defl
                  << std::setw(8) << std::setprecision(1) << pct(eff_cgs_defl, eff_cg) << "%"
                  << std::setw(8) << total_it_cgs_defl
                  << std::setw(10) << std::setprecision(1) << (double)total_it_cgs_defl / n_steps << "\n";
        std::cout << "  FGM+MG Wm   " << std::setw(10) << std::setprecision(4) << eff_fgm_warm
                  << std::setw(8) << std::setprecision(1) << pct(eff_fgm_warm, eff_cg) << "%"
                  << std::setw(8) << total_it_fgm_warm
                  << std::setw(10) << std::setprecision(1) << (double)total_it_fgm_warm / n_steps << "\n";
        std::cout << "  CG+sMG Fr   " << std::setw(10) << std::setprecision(4) << eff_cgs_fresh
                  << std::setw(8) << std::setprecision(1) << pct(eff_cgs_fresh, eff_cg) << "%"
                  << std::setw(8) << total_it_cgs_fresh
                  << std::setw(10) << std::setprecision(1) << (double)total_it_cgs_fresh / n_steps << "\n";
        std::cout << "  (W = warm refresh step, " << n_coarse_defl << " coarse deflation vecs)\n";
        std::cout << "  (Coarse deflation update: " << std::setprecision(4)
                  << total_t_defl_update << "s total, "
                  << std::setprecision(4) << total_t_defl_update / n_steps << "s/step)\n";
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

        for (int step = 0; step < n_steps; step++) {
            perturb_gauge(gauge, rng, eps);
            DiracOp D(lat, gauge, mass, wilson_r);
            Vec rhs = random_vec(lat.ndof, rng);

            auto t0 = Clock::now();
            Ac_stale.build(D, P_stale);
            auto res_stale = fgmres_solve(D, rhs, P_stale, Ac_stale,
                                           krylov, max_iter, tol, 3, 3, 0);
            double dt_stale = Duration(Clock::now() - t0).count();

            t0 = Clock::now();
            Ac_ritz.build(D, P_ritz);
            auto res_ritz = fgmres_solve(D, rhs, P_ritz, Ac_ritz,
                                          krylov, max_iter, tol, 3, 3, 4);
            double dt_ritz = Duration(Clock::now() - t0).count();

            t0 = Clock::now();
            auto fresh_null = compute_near_null_space(D, k_null, 20, rng);
            Prolongator P_fresh(lat, block_size, block_size, k_null);
            P_fresh.build_from_vectors(fresh_null);
            CoarseOp Ac_fresh;
            Ac_fresh.build(D, P_fresh);
            double dt_fresh_setup = Duration(Clock::now() - t0).count();

            t0 = Clock::now();
            auto res_fresh = fgmres_solve(D, rhs, P_fresh, Ac_fresh,
                                           krylov, max_iter, tol, 3, 3, 0);
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
                  << std::setw(8)  << std::setprecision(1) << (double)total_it_stale / n_steps
                  << std::setw(10) << std::setprecision(4) << total_t_stale / n_steps
                  << std::setw(8)  << std::setprecision(1) << (double)total_it_ritz / n_steps
                  << std::setw(10) << std::setprecision(4) << total_t_ritz / n_steps
                  << std::setw(8)  << std::setprecision(1) << (double)total_it_fresh / n_steps
                  << std::setw(10) << std::setprecision(4) << total_t_fresh / n_steps
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
