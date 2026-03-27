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
        << "2D Schwinger model (U(1) lattice gauge theory) with Wilson-clover fermions,\n"
        << "multigrid solver, and multi-timescale HMC with nested force-gradient integration.\n"
        << "\n"
        << "=== Lattice & Fermion Options ===\n"
        << "  -L <int>              Lattice size (LxL)                [16]\n"
        << "  -m <float>            Fermion mass                      [0.05]\n"
        << "  -r <float>            Wilson parameter                  [1.0]\n"
        << "  --csw <float>         Clover coefficient (0=Wilson)     [0.0]\n"
        << "  -w <float>            Hot-start width (initial gauge)   [0.4]\n"
        << "  -s <int>              RNG seed                          [42]\n"
        << "  -t <int>              OpenMP threads                    [all cores]\n"
        << "\n"
        << "=== Multigrid Options ===\n"
        << "  --mg-levels <int>     Number of MG levels               [1]\n"
        << "  -b <int>              Block size (bxb)                  [4]\n"
        << "  -k <int>              Null vectors per block            [4]\n"
        << "  --coarse-block <int>  Coarse-level block size           [k*4]\n"
        << "  --w-cycle             Use W-cycle (default for levels>1)\n"
        << "  --v-cycle             Use V-cycle\n"
        << "  --no-mg               Disable multigrid\n"
        << "  --symmetric-mg        Use Richardson smoothing (for CG)\n"
        << "\n"
        << "=== Solver Options ===\n"
        << "  --tol <float>         CG/FGMRES tolerance              [1e-10]\n"
        << "  --krylov <int>        Krylov subspace dimension         [30]\n"
        << "  --maxiter <int>       Max solver iterations             [300]\n"
        << "\n"
        << "=== Standard HMC ===\n"
        << "  --hmc                 Run standard HMC\n"
        << "  --hmc-traj <int>      Number of trajectories            [100]\n"
        << "  --hmc-tau <float>     Trajectory length                 [1.0]\n"
        << "  --hmc-steps <int>     Leapfrog steps per trajectory     [20]\n"
        << "  --hmc-beta <float>    Gauge coupling beta               [2.0]\n"
        << "  --hmc-therm <int>     Thermalisation trajectories       [20]\n"
        << "  --hmc-save-every <N>  Save gauge every N trajectories   [10]\n"
        << "  --hmc-save-prefix <s> Gauge file prefix                 [gauge]\n"
        << "  --hmc-load <file>     Load initial gauge configuration\n"
        << "\n"
        << "=== Multi-Timescale HMC (Fine-Grid Deflation) ===\n"
        << "  --hmc-multiscale      Enable fine-grid deflation multi-timescale HMC\n"
        << "  --hmc-n-outer <int>   Outer leapfrog steps              [10]\n"
        << "  --hmc-n-inner <int>   Inner steps per outer             [5]\n"
        << "  --hmc-n-defl <int>    Deflation eigenvectors            [8]\n"
        << "  --hmc-fresh-period <N> Fresh TRLM every N trajectories  [10]\n"
        << "\n"
        << "=== MG Multi-Timescale HMC (Coarse-Grid Deflation) ===\n"
        << "  --hmc-mg-multiscale   Enable MG coarse-grid deflation HMC\n"
        << "  --hmc-n-outer <int>   Outer integrator steps            [10]\n"
        << "  --hmc-n-inner <int>   Inner leapfrog steps per outer    [5]\n"
        << "  --hmc-n-defl <int>    Coarse deflation eigenvectors     [8]\n"
        << "  --hmc-omelyan         Use Omelyan (2MN) outer integrator\n"
        << "  --hmc-force-gradient  Use nested FGI (MILC PQPQP) outer integrator\n"
        << "  --hmc-defl-refresh <N> Refresh coarse deflation every N inner steps [0=off]\n"
        << "  --hmc-revtest         Run reversibility test (forward+backward)\n"
        << "  --hmc-fresh-period <N> Fresh TRLM every N trajectories  [10]\n"
        << "\n"
        << "=== Sparse Coarse Operator Study ===\n"
        << "  --test-sparse-coarse  Run coarse eigenvector evolution study\n"
        << "  --n-defl <int>        Number of deflation vectors       [16]\n"
        << "  -n <int>              Number of gauge evolution steps    [30]\n"
        << "  -e <float>            Gauge perturbation per step       [0.12]\n"
        << "\n"
        << "=== Deflation Test ===\n"
        << "  --test-deflation      Run coarse eigenvector prolongation test\n"
        << "  --cheb-only           Only test Chebyshev-filtered eigenvectors\n"
        << "\n"
        << "  --help, -h            Show this help\n"
        << "\n"
        << "======================= EXAMPLES =======================\n"
        << "\n"
        << "--- Basic MG Solver ---\n"
        << "  " << prog << " -L 32 --mg-levels 2                    # 2-level W-cycle\n"
        << "  " << prog << " -L 64 --mg-levels 3 -k 8 -t 8          # 3-level MG, 8 threads\n"
        << "  " << prog << " -L 32 --mg-levels 2 --csw 1.0           # Wilson-clover MG\n"
        << "\n"
        << "--- Standard HMC ---\n"
        << "  " << prog << " --hmc -L 16 --hmc-beta 2.0 --hmc-traj 100\n"
        << "  " << prog << " --hmc -L 32 --csw 1.0 --hmc-beta 2.0    # clover HMC\n"
        << "  " << prog << " --hmc -L 32 --hmc-therm 50 --hmc-save-every 5\n"
        << "\n"
        << "--- MG Multi-Timescale HMC (Leapfrog outer) ---\n"
        << "  " << prog << " -L 32 --hmc-mg-multiscale --mg-levels 2 -b 4 -k 4 \\\n"
        << "         --hmc-n-outer 20 --hmc-n-inner 2 --hmc-n-defl 16 --hmc-traj 20\n"
        << "\n"
        << "--- MG Multi-Timescale HMC (Omelyan outer) ---\n"
        << "  " << prog << " -L 32 --hmc-mg-multiscale --hmc-omelyan --mg-levels 2 \\\n"
        << "         -b 4 -k 4 --hmc-n-outer 10 --hmc-n-inner 2 --hmc-n-defl 16\n"
        << "\n"
        << "--- Nested Force-Gradient Integrator (MILC-style) ---\n"
        << "  " << prog << " -L 32 --hmc-mg-multiscale --hmc-force-gradient --mg-levels 2 \\\n"
        << "         -b 4 -k 4 --hmc-n-outer 5 --hmc-n-inner 3 --hmc-n-defl 16 --hmc-traj 20\n"
        << "\n"
        << "--- Nested FGI with Wilson-Clover ---\n"
        << "  " << prog << " -L 32 --hmc-mg-multiscale --hmc-force-gradient --csw 1.0 \\\n"
        << "         --mg-levels 2 -b 4 -k 4 --hmc-n-outer 7 --hmc-n-inner 3 \\\n"
        << "         --hmc-n-defl 16 --hmc-traj 20 --hmc-beta 2.0\n"
        << "\n"
        << "--- Nested FGI with Periodic Deflation Refresh ---\n"
        << "  " << prog << " -L 32 --hmc-mg-multiscale --hmc-force-gradient --mg-levels 2 \\\n"
        << "         -b 4 -k 4 --hmc-n-outer 5 --hmc-n-inner 5 --hmc-n-defl 16 \\\n"
        << "         --hmc-defl-refresh 3 --hmc-traj 20\n"
        << "\n"
        << "--- Reversibility Test (all integrators) ---\n"
        << "  " << prog << " -L 32 --hmc-mg-multiscale --hmc-revtest --mg-levels 2 \\\n"
        << "         -b 4 -k 4 --hmc-n-outer 5 --hmc-n-inner 3 --hmc-n-defl 16\n"
        << "  " << prog << " -L 32 --hmc-mg-multiscale --hmc-revtest --csw 1.0 \\\n"
        << "         --mg-levels 2 -b 4 -k 4 --hmc-n-outer 5 --hmc-n-inner 3\n"
        << "\n"
        << "--- Sparse Coarse Eigenvector Evolution Study ---\n"
        << "  " << prog << " -L 128 --test-sparse-coarse --mg-levels 2 -b 4 -k 4 \\\n"
        << "         --n-defl 16 -n 50 -e 0.02 --hmc-therm 20 --hmc-beta 2.0\n"
        << "\n"
        << "--- Thermalise and Save Gauge Configuration ---\n"
        << "  " << prog << " -L 32 --test-sparse-coarse --mg-levels 2 -b 4 -k 4 \\\n"
        << "         --hmc-therm 50 --hmc-beta 2.0 --n-defl 16 -n 1 -e 0.02\n"
        << "  (saves to gauge_L32_b2.00_t50.bin, auto-loaded by subsequent runs)\n";
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);  // line-buffered for crash diagnostics
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
    double c_sw      = 0.0;  // clover coefficient (0 = pure Wilson)
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
    bool   test_sparse_coarse = false;
    bool   cheb_only = false;
    int    n_defl_vecs = 0;
    bool   run_hmc   = false;
    bool   hmc_multiscale = false;
    bool   hmc_mg_multiscale = false;
    bool   hmc_force_gradient = false;
    bool   hmc_omelyan = false;
    bool   hmc_revtest = false;
    int    hmc_defl_refresh = 0;
    int    hmc_n_outer = 10;
    int    hmc_n_inner = 5;
    int    hmc_n_defl  = 8;
    int    hmc_fresh_period = 10;
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
        else if (match("--csw"))     c_sw      = next_dbl();
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
        else if (match("--hmc-multiscale")) hmc_multiscale = true;
        else if (match("--hmc-mg-multiscale")) hmc_mg_multiscale = true;
        else if (match("--hmc-force-gradient")) hmc_force_gradient = true;
        else if (match("--hmc-omelyan")) hmc_omelyan = true;
        else if (match("--hmc-revtest")) hmc_revtest = true;
        else if (match("--hmc-defl-refresh")) hmc_defl_refresh = std::atoi(argv[++i]);
        else if (match("--hmc-n-outer")) hmc_n_outer = next_int();
        else if (match("--hmc-n-inner")) hmc_n_inner = next_int();
        else if (match("--hmc-n-defl")) hmc_n_defl = next_int();
        else if (match("--hmc-fresh-period")) hmc_fresh_period = next_int();
        else if (match("--test-deflation")) test_deflation = true;
        else if (match("--test-sparse-coarse")) test_sparse_coarse = true;
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
    //  Multi-timescale HMC test
    // -----------------------------------------------------------------
    if (hmc_multiscale) {
        using Clock = std::chrono::high_resolution_clock;
        using Dur = std::chrono::duration<double>;

        int n_traj = hmc_traj > 0 ? hmc_traj : 20;
        int total_steps = hmc_n_outer * hmc_n_inner;

        std::cout << "=== Multi-Timescale HMC Test ===\n\n";
        std::cout << "L=" << L << "  DOF=" << lat.ndof << "  mass=" << mass
                  << "  beta=" << hmc_beta << "\n";
        std::cout << "n_outer=" << hmc_n_outer << "  n_inner=" << hmc_n_inner
                  << "  total_steps=" << total_steps
                  << "  tau=" << hmc_tau << "\n";
        std::cout << "n_defl=" << hmc_n_defl
                  << "  fresh_period=" << hmc_fresh_period
                  << "  traj=" << n_traj << "\n\n";

        // --- Compute initial eigenvectors ---
        std::cout << "--- Computing " << hmc_n_defl << " eigenvectors of D†D ---\n";
        DiracOp D_init(lat, gauge, mass, wilson_r, c_sw);
        OpApply A_init = [&D_init](const Vec& s, Vec& d) { D_init.apply_DdagD(s, d); };
        auto trlm = trlm_eigensolver(A_init, lat.ndof, hmc_n_defl,
                                      std::min(2*hmc_n_defl + 10, lat.ndof),
                                      100, 1e-10);
        DeflationState defl;
        defl.eigvecs = std::move(trlm.eigvecs);
        defl.eigvals = std::move(trlm.eigvals);
        defl.valid = true;
        defl.update_cache(D_init);

        std::cout << "  Eigenvalues: ";
        for (int i = 0; i < std::min(hmc_n_defl, 8); i++)
            std::cout << std::scientific << std::setprecision(4) << defl.eigvals[i] << " ";
        if (hmc_n_defl > 8) std::cout << "...";
        std::cout << "\n\n";

        // --- Run two HMC streams: standard vs multi-timescale ---
        GaugeField gauge_std = gauge;
        GaugeField gauge_ms = gauge;
        std::mt19937 rng_std(seed + 1000);
        std::mt19937 rng_ms(seed + 1000);  // same seed for fair comparison

        HMCParams std_params;
        std_params.beta = hmc_beta;
        std_params.tau = hmc_tau;
        std_params.n_steps = total_steps;
        std_params.cg_maxiter = max_iter;
        std_params.cg_tol = tol;
        std_params.use_mg = false;
        std_params.c_sw = c_sw;

        MultiScaleParams ms_params;
        ms_params.beta = hmc_beta;
        ms_params.tau = hmc_tau;
        ms_params.n_outer = hmc_n_outer;
        ms_params.n_inner = hmc_n_inner;
        ms_params.cg_maxiter = max_iter;
        ms_params.cg_tol = tol;
        ms_params.c_sw = c_sw;

        int std_accept = 0, ms_accept = 0;
        double std_dH_sum = 0, ms_dH_sum = 0;
        int std_cg_sum = 0, ms_cg_sum = 0;
        double std_time_sum = 0, ms_time_sum = 0;
        double ms_low_time_sum = 0, ms_high_time_sum = 0;
        int ms_low_evals_sum = 0;

        std::cout << std::setw(5) << "traj"
                  << " |" << std::setw(7) << "Std_CG"
                  << std::setw(9) << "Std_dH"
                  << std::setw(7) << "Std_A"
                  << std::setw(9) << "Std_t"
                  << " |" << std::setw(7) << "MS_CG"
                  << std::setw(9) << "MS_dH"
                  << std::setw(7) << "MS_A"
                  << std::setw(9) << "MS_t"
                  << std::setw(9) << "Low_t"
                  << std::setw(9) << "High_t"
                  << std::setw(8) << "LowEv"
                  << " |" << std::setw(8) << "<plaq>"
                  << "\n";

        for (int t = 0; t < n_traj; t++) {
            // --- Standard HMC ---
            auto t0_std = Clock::now();
            auto res_std = hmc_trajectory(gauge_std, lat, mass, wilson_r,
                                           std_params, rng_std);
            double t_std = Dur(Clock::now() - t0_std).count();
            if (res_std.accepted) std_accept++;
            std_dH_sum += std::abs(res_std.dH);
            std_cg_sum += res_std.total_cg_iters;
            std_time_sum += t_std;

            // --- Multi-timescale HMC ---
            auto t0_ms = Clock::now();
            auto res_ms = hmc_trajectory_multiscale(gauge_ms, lat, mass, wilson_r,
                                                     ms_params, defl, rng_ms);
            double t_ms = Dur(Clock::now() - t0_ms).count();
            if (res_ms.accepted) ms_accept++;
            ms_dH_sum += std::abs(res_ms.dH);
            ms_cg_sum += res_ms.highmode_cg_iters;
            ms_time_sum += t_ms;
            ms_low_time_sum += res_ms.lowmode_time;
            ms_high_time_sum += res_ms.highmode_time;
            ms_low_evals_sum += res_ms.lowmode_force_evals;

            // Evolve deflation state after multi-timescale trajectory
            DiracOp D_new(lat, gauge_ms, mass, wilson_r, c_sw);
            bool do_fresh = (hmc_fresh_period > 0) && ((t+1) % hmc_fresh_period == 0);
            evolve_deflation_state(defl, D_new, do_fresh);

            // Print per-trajectory stats
            std::cout << std::fixed;
            std::cout << std::setw(5) << t
                      << " |" << std::setw(7) << res_std.total_cg_iters
                      << std::setw(9) << std::setprecision(3) << res_std.dH
                      << std::setw(7) << (res_std.accepted ? "Y" : "N")
                      << std::setw(9) << std::setprecision(3) << t_std
                      << " |" << std::setw(7) << res_ms.highmode_cg_iters
                      << std::setw(9) << std::setprecision(3) << res_ms.dH
                      << std::setw(7) << (res_ms.accepted ? "Y" : "N")
                      << std::setw(9) << std::setprecision(3) << t_ms
                      << std::setw(9) << std::setprecision(3) << res_ms.lowmode_time
                      << std::setw(9) << std::setprecision(3) << res_ms.highmode_time
                      << std::setw(8) << res_ms.lowmode_force_evals
                      << " |" << std::setw(8) << std::setprecision(4)
                      << gauge_ms.avg_plaq()
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

        std::cout << "\n  Multi-timescale HMC (n_outer=" << hmc_n_outer
                  << " n_inner=" << hmc_n_inner << " n_defl=" << hmc_n_defl << "):\n";
        std::cout << "    Accept rate:  " << std::fixed << std::setprecision(1)
                  << 100.0 * ms_accept / n_traj << "%\n";
        std::cout << "    Avg |dH|:     " << std::scientific << std::setprecision(3)
                  << ms_dH_sum / n_traj << "\n";
        std::cout << "    Avg CG iters: " << ms_cg_sum / n_traj
                  << " (high-mode only, " << hmc_n_outer + 1 << " solves + 2 H evals)\n";
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

    // -----------------------------------------------------------------
    //  MG Multi-timescale HMC (coarse-grid deflation)
    // -----------------------------------------------------------------
    if (hmc_mg_multiscale) {
        if (mg_levels < 2) {
            std::cerr << "--hmc-mg-multiscale requires --mg-levels >= 2\n";
            return 1;
        }

        using Clock = std::chrono::high_resolution_clock;
        using Dur = std::chrono::duration<double>;

        int n_traj = hmc_traj > 0 ? hmc_traj : 20;
        int n_defl = n_defl_vecs > 0 ? n_defl_vecs : hmc_n_defl;
        int total_steps = hmc_n_outer * hmc_n_inner;

        std::cout << "=== MG Multi-Timescale HMC (Coarse-Grid Deflation) ===\n\n";
        std::cout << "L=" << L << "  DOF=" << lat.ndof << "  mass=" << mass
                  << "  beta=" << hmc_beta << "\n";
        std::cout << "MG levels=" << mg_levels << "  block=" << block_size
                  << "  k_null=" << k_null << "\n";
        std::cout << "n_outer=" << hmc_n_outer << "  n_inner=" << hmc_n_inner
                  << "  total_steps=" << total_steps
                  << "  tau=" << hmc_tau << "\n";
        std::cout << "n_defl=" << n_defl << "  traj=" << n_traj << "\n\n";

        // --- Load thermalised config if available ---
        {
            std::string cfg = "gauge_L" + std::to_string(L) + "_b"
                + std::to_string(hmc_beta).substr(0,4) + "_t50.bin";
            if (gauge.load(cfg)) {
                std::cout << "--- Loaded thermalised config from " << cfg
                          << "  <plaq>=" << std::fixed << std::setprecision(4)
                          << gauge.avg_plaq() << " ---\n";
            } else {
                // Try t20 variant
                cfg = "gauge_L" + std::to_string(L) + "_b"
                    + std::to_string(hmc_beta).substr(0,4) + "_t20.bin";
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
        DiracOp D_mg(lat, gauge, mass, wilson_r, c_sw);
        OpApply A_mg = [&D_mg](const Vec& s, Vec& d) { D_mg.apply_DdagD(s, d); };
        std::mt19937 rng_mg(seed + 111);
        auto mg = build_mg_hierarchy(D_mg, mg_levels, block_size, k_null,
                                      coarse_block, 20, rng_mg, w_cycle,
                                      3, 3, true);

        // Setup sparse coarse operator + TRLM deflation
        mg.setup_sparse_coarse(A_mg, lat.ndof, n_defl);

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
        std::mt19937 rng_std(seed + 2000);
        std::mt19937 rng_ms(seed + 2000);

        // Standard: MG-preconditioned CG at every step
        HMCParams std_params;
        std_params.beta = hmc_beta;
        std_params.tau = hmc_tau;
        std_params.n_steps = total_steps;
        std_params.cg_maxiter = max_iter;
        std_params.cg_tol = tol;
        std_params.use_mg = false;
        std_params.c_sw = c_sw;

        // Multi-timescale
        MGMultiScaleParams ms_params;
        ms_params.beta = hmc_beta;
        ms_params.tau = hmc_tau;
        ms_params.n_outer = hmc_n_outer;
        ms_params.n_inner = hmc_n_inner;
        ms_params.cg_maxiter = max_iter;
        ms_params.cg_tol = tol;
        ms_params.c_sw = c_sw;
        if (hmc_omelyan) ms_params.outer_type = OuterIntegrator::Omelyan;
        if (hmc_force_gradient) ms_params.outer_type = OuterIntegrator::FGI;
        ms_params.defl_refresh = hmc_defl_refresh;

        std::string outer_name = "Leapfrog";
        if (hmc_omelyan) outer_name = "Omelyan (2MN)";
        if (hmc_force_gradient) outer_name = "FGI (MILC PQPQP, 4th order)";
        std::cout << "Outer: " << outer_name << "  Inner: Leapfrog (gauge+lowmode)\n\n";

        // --- Reversibility test ---
        if (hmc_revtest) {
            std::cout << "--- Reversibility Test ---\n";
            std::cout << "  Running forward → negate π → backward for each integrator...\n\n";

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
                std::mt19937 rng_rev(seed + 9999);
                auto res = reversibility_test_mg_multiscale(
                    gauge, lat, mass, wilson_r, rp, cdefl, P, mg_precond, rng_rev);

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

        std::cout << std::setw(5) << "traj"
                  << " |" << std::setw(7) << "Std_CG"
                  << std::setw(9) << "Std_dH"
                  << std::setw(5) << "A"
                  << std::setw(9) << "Std_t"
                  << " |" << std::setw(7) << "MS_CG"
                  << std::setw(9) << "MS_dH"
                  << std::setw(5) << "A"
                  << std::setw(9) << "MS_t"
                  << std::setw(8) << "Low_t"
                  << std::setw(8) << "Hi_t"
                  << std::setw(6) << "LEv"
                  << " |" << std::setw(8) << "<plaq>"
                  << "\n";

        for (int t = 0; t < n_traj; t++) {
            // --- Standard HMC (MG-preconditioned) ---
            auto t0_std = Clock::now();
            auto res_std = hmc_trajectory(gauge_std, lat, mass, wilson_r,
                                           std_params, rng_std, &std_precond);
            double t_std = Dur(Clock::now() - t0_std).count();
            if (res_std.accepted) std_accept++;
            std_dH_sum += std::abs(res_std.dH);
            std_cg_sum += res_std.total_cg_iters;
            std_time_sum += t_std;

            // --- MG Multi-timescale HMC ---
            auto t0_ms = Clock::now();
            auto res_ms = hmc_trajectory_mg_multiscale(gauge_ms, lat, mass, wilson_r,
                                                        ms_params, cdefl, P,
                                                        mg_precond, rng_ms);
            double t_ms = Dur(Clock::now() - t0_ms).count();
            if (res_ms.accepted) ms_accept++;
            ms_dH_sum += std::abs(res_ms.dH);
            ms_cg_sum += res_ms.highmode_cg_iters;
            ms_time_sum += t_ms;
            ms_low_time_sum += res_ms.lowmode_time;
            ms_high_time_sum += res_ms.highmode_time;
            ms_low_evals_sum += res_ms.lowmode_force_evals;

            // Evolve coarse deflation after MS trajectory
            // Rebuild sparse Ac for new gauge + RR evolve
            if (res_ms.accepted) {
                DiracOp D_new(lat, gauge_ms, mass, wilson_r, c_sw);
                OpApply A_new = [&D_new](const Vec& s, Vec& d) { D_new.apply_DdagD(s, d); };
                mg.sparse_Ac.build(P, A_new, D_new.lat.ndof);
                evolve_coarse_deflation(cdefl, mg.sparse_Ac);
            }

            // Periodically do fresh TRLM
            if (hmc_fresh_period > 0 && (t+1) % hmc_fresh_period == 0) {
                mg.sparse_Ac.setup_deflation(n_defl);
                cdefl.eigvecs = mg.sparse_Ac.defl_vecs;
                cdefl.eigvals = mg.sparse_Ac.defl_vals;
            }

            std::cout << std::fixed;
            std::cout << std::setw(5) << t
                      << " |" << std::setw(7) << res_std.total_cg_iters
                      << std::setw(9) << std::setprecision(3) << res_std.dH
                      << std::setw(5) << (res_std.accepted ? "Y" : "N")
                      << std::setw(9) << std::setprecision(3) << t_std
                      << " |" << std::setw(7) << res_ms.highmode_cg_iters
                      << std::setw(9) << std::setprecision(3) << res_ms.dH
                      << std::setw(5) << (res_ms.accepted ? "Y" : "N")
                      << std::setw(9) << std::setprecision(3) << t_ms
                      << std::setw(8) << std::setprecision(3) << res_ms.lowmode_time
                      << std::setw(8) << std::setprecision(3) << res_ms.highmode_time
                      << std::setw(6) << res_ms.lowmode_force_evals
                      << " |" << std::setw(8) << std::setprecision(4)
                      << gauge_ms.avg_plaq()
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

        std::cout << "\n  MG Multi-timescale (n_outer=" << hmc_n_outer
                  << " n_inner=" << hmc_n_inner << " n_defl=" << n_defl << "):\n";
        std::cout << "    Accept rate:  " << std::fixed << std::setprecision(1)
                  << 100.0 * ms_accept / n_traj << "%\n";
        std::cout << "    Avg |dH|:     " << std::scientific << std::setprecision(3)
                  << ms_dH_sum / n_traj << "\n";
        std::cout << "    Avg CG iters: " << ms_cg_sum / n_traj
                  << " (" << hmc_n_outer + 1 << " outer + 2 H evals)\n";
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

    // -----------------------------------------------------------------
    //  Test sparse coarse operator
    // -----------------------------------------------------------------
    if (test_sparse_coarse) {
        if (mg_levels < 2) {
            std::cerr << "--test-sparse-coarse requires --mg-levels >= 2\n";
            return 1;
        }

        int n_defl = n_defl_vecs > 0 ? n_defl_vecs : 16;
        int ndof = lat.ndof;
        int coarse_dim_est = (L/block_size) * (L/block_size) * k_null;

        std::cout << "=== Sparse Coarse Eigenvector Evolution Study ===\n\n";
        std::cout << "L=" << L << "  DOF=" << ndof
                  << "  coarse_dim=" << coarse_dim_est
                  << "  n_defl=" << n_defl << "\n";

        // --- Thermalise if requested (save/load gauge configs) ---
        if (hmc_therm > 0) {
            std::string cfg_path = "gauge_L" + std::to_string(L)
                + "_b" + std::to_string(hmc_beta).substr(0,4)
                + "_t" + std::to_string(hmc_therm) + ".bin";

            if (gauge.load(cfg_path)) {
                std::cout << "\n--- Loaded thermalised config from " << cfg_path
                          << "  <plaq>=" << std::fixed << std::setprecision(4)
                          << gauge.avg_plaq() << " ---\n";
            } else {
                std::cout << "\n--- Thermalisation: " << hmc_therm
                          << " HMC trajectories (beta=" << hmc_beta
                          << " tau=" << hmc_tau << " steps=" << hmc_steps << ") ---\n";

                DiracOp D_therm(lat, gauge, mass, wilson_r, c_sw);
                OpApply A_therm = [&D_therm](const Vec& s, Vec& d){ D_therm.apply_DdagD(s, d); };
                std::mt19937 rng_therm(seed);
                auto mg_therm = build_mg_hierarchy(D_therm, mg_levels, block_size, k_null,
                                                    coarse_block, 20, rng_therm, w_cycle,
                                                    3, 3, true);
                mg_therm.setup_sparse_coarse(A_therm, ndof, n_defl);

                std::function<Vec(const Vec&)> precond_fn = [&mg_therm](const Vec& b) -> Vec {
                    return mg_therm.precondition(b);
                };

                HMCParams hparams;
                hparams.beta = hmc_beta;
                hparams.tau = hmc_tau;
                hparams.n_steps = hmc_steps;
                hparams.cg_maxiter = max_iter;
                hparams.cg_tol = tol;
                hparams.use_mg = false;

                int accepted = 0;
                for (int t = 0; t < hmc_therm; t++) {
                    if (t > 0 && t % 5 == 0) {
                        DiracOp D_rebuild(lat, gauge, mass, wilson_r, c_sw);
                        OpApply A_rebuild = [&D_rebuild](const Vec& s, Vec& d){ D_rebuild.apply_DdagD(s, d); };
                        mg_therm = build_mg_hierarchy(D_rebuild, mg_levels, block_size, k_null,
                                                       coarse_block, 20, rng_therm, w_cycle,
                                                       3, 3, false);
                        mg_therm.setup_sparse_coarse(A_rebuild, ndof, n_defl);
                    }

                    auto res = hmc_trajectory(gauge, lat, mass, wilson_r,
                                               hparams, rng, &precond_fn);
                    if (res.accepted) accepted++;
                    if ((t+1) % 5 == 0 || t == hmc_therm - 1) {
                        std::cout << "  traj " << t+1 << "/" << hmc_therm
                                  << "  accept=" << accepted << "/" << (t+1)
                                  << "  <plaq>=" << std::fixed << std::setprecision(4)
                                  << gauge.avg_plaq() << "  dH=" << std::scientific
                                  << res.dH << "\n";
                    }
                }
                std::cout << "  Final <plaq> = " << std::fixed << std::setprecision(4)
                          << gauge.avg_plaq() << "\n";

                if (gauge.save(cfg_path))
                    std::cout << "  Saved config to " << cfg_path << "\n";
            }
        }

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
            if (L % cfg.blk != 0) {
                std::cout << "\n--- Skipping " << cfg.label
                          << ": block " << cfg.blk << " does not divide L=" << L << " ---\n";
                continue;
            }
            int nb = L / cfg.blk;
            int cdim_est = nb * nb * cfg.knull;

            std::cout << "\n==========================================================\n";
            std::cout << "  Config: " << cfg.label
                      << "  block=" << cfg.blk << "x" << cfg.blk
                      << "  k_null=" << cfg.knull
                      << "  coarse_dim=" << cdim_est << "\n";
            std::cout << "  Galerkin every " << galerkin_period << " steps, RR every step"
                      << ", " << n_steps << " configs, " << n_rhs << " RHS/step"
                      << ", n_defl=" << n_defl << "\n";
            std::cout << "==========================================================\n";

            // Build MG hierarchy for this config
            DiracOp D(lat, gauge, mass, wilson_r, c_sw);
            OpApply A = [&D](const Vec& s, Vec& d){ D.apply_DdagD(s, d); };
            std::mt19937 rng_mg(seed + 111);
            auto mg = build_mg_hierarchy(D, mg_levels, cfg.blk, cfg.knull,
                                          coarse_block, 20, rng_mg, w_cycle,
                                          3, 3, true);
            mg.setup_sparse_coarse(A, ndof, n_defl);
            int cdim = mg.sparse_Ac.dim;
            std::cout << "  Actual coarse dim: " << cdim << "\n";

            auto& P_ev = mg.geo_prolongators[0];

            // Evolution study
            GaugeField gauge_ev = gauge;
            auto D_ev = std::make_unique<DiracOp>(lat, gauge_ev, mass, wilson_r, c_sw);
            OpApply A_ev = [&D_ev](const Vec& s, Vec& d){ D_ev->apply_DdagD(s, d); };
            std::mt19937 rng_ev(seed + 777);

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

            for (int step = 0; step < n_steps; step++) {
                MomentumField mom(lat);
                mom.randomise(rng_ev);

                // Apply gauge update
                for (int mu = 0; mu < 2; mu++)
                    for (int s = 0; s < lat.V; s++)
                        gauge_ev.U[mu][s] *= std::exp(cx(0, eps * mom.pi[mu][s]));
                D_ev = std::make_unique<DiracOp>(lat, gauge_ev, mass, wilson_r, c_sw);
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

            double avg_defl_it = (double)total_defl_it / (n_steps * n_rhs);
            double avg_nodfl_it = (double)total_nodfl_it / (n_steps * n_rhs);
            double overall_saved = 100.0 * (1.0 - avg_defl_it / avg_nodfl_it);
            double overall_speedup = total_nodfl_wall / total_defl_wall;
            int n_rebuilds = (n_steps + galerkin_period - 1) / galerkin_period;

            std::cout << "\n  === " << cfg.label << " Summary over " << n_steps << " steps ===\n";
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
                      << " rebuilds) + " << total_rr_t << "s RR (" << n_steps << " steps)\n";
            std::cout << "    Net wall time:   Defl = " << std::setprecision(3)
                      << total_defl_wall + total_build_t + total_rr_t << "s"
                      << "  vs  NoDfl = " << total_nodfl_wall << "s\n";
        }

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

        DiracOp D(lat, gauge, mass, wilson_r, c_sw);
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
        // 1) Force-based: update arrow matrix using δD from momentum (0 matvecs)
        // 2) RR projection: apply A_new and re-project (k matvecs)
        std::cout << "\n=== Method 8: Force-Based vs RR Eigenvector Evolution ===\n";
        std::cout << "  Physical MD evolution (momentum-driven gauge updates)\n\n";
        {
            int k_want = n_defl_vecs > 0 ? n_defl_vecs : ndefl;
            k_want = std::min(k_want, lat.ndof / 4);
            int n_md_steps = std::min(n_steps, 50);
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
                    std::mt19937 rng_m8(seed + 8888);
                    mom_m8.randomise(rng_m8);

                    auto D_m8 = std::make_unique<DiracOp>(lat, gauge_m8, mass, wilson_r, c_sw);
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
                            // Force-based: compute δD using momentum BEFORE gauge update
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

                        D_m8 = std::make_unique<DiracOp>(lat, gauge_m8, mass, wilson_r, c_sw);

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
        // Method 9: Hybrid tracker — force + Lanczos extension
        // =============================================================
        // Track n_kr Krylov subspace with force-based rotation,
        // periodically extend with Lanczos to bring in fresh directions.
        std::cout << "\n=== Method 9: Hybrid Force + Lanczos Extension ===\n";
        std::cout << "  n_kr Krylov vectors tracked, periodic Lanczos refresh\n\n";
        {
            int n_ev_h = n_defl_vecs > 0 ? n_defl_vecs : ndefl;
            n_ev_h = std::min(n_ev_h, lat.ndof / 4);
            int n_md_steps_h = std::min(n_steps, 50);
            double hmc_beta_h = 2.0;

            for (double dt : {0.01, 0.05}) {
              // Try different Lanczos extension intervals
              for (int ext_interval : {1, 2, 5}) {
                std::cout << "  ====== dt=" << std::fixed << std::setprecision(3) << dt
                          << ", n_ev=" << n_ev_h
                          << ", Lanczos every " << ext_interval << " steps ======\n";

                GaugeField gauge_h = gauge;
                MomentumField mom_h(lat);
                std::mt19937 rng_h(seed + 8888);
                mom_h.randomise(rng_h);

                auto D_h = std::make_unique<DiracOp>(lat, gauge_h, mass, wilson_r, c_sw);
                OpApply A_h = [&D_h](const Vec& s, Vec& d){ D_h->apply_DdagD(s, d); };
                auto applyD = [&D_h](const Vec& s, Vec& d){ D_h->apply(s, d); };

                // Initial TRLM — use larger Krylov space for better tracking
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
                        // Force-based step: compute δD before gauge update
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

                    D_h = std::make_unique<DiracOp>(lat, gauge_h, mass, wilson_r, c_sw);

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
        //   1. Force-based evolution (0 D†D matvecs)
        //   2. CG Ritz harvesting (0 extra matvecs — from fermion solve)
        //   3. Chebyshev-filtered probes (periodic, ~20 matvecs each)
        //   4. Coarse-grid spectral proxy (nearly free)
        //
        // Runs a physical MD trajectory with leapfrog momentum updates.
        // At each step, the tracker maintains eigenvectors using cheap
        // sources and is compared against fresh TRLM ground truth.
        std::cout << "\n=== Method 10: Multi-Source EigenTracker ===\n";
        std::cout << "  Pool-based tracker: force + solver harvest + Chebyshev probe\n\n";
        {
            int n_ev_t = n_defl_vecs > 0 ? n_defl_vecs : ndefl;
            n_ev_t = std::min(n_ev_t, lat.ndof / 4);
            int pool_cap = std::max(3 * n_ev_t, 2 * n_ev_t + 12);
            pool_cap = std::min(pool_cap, lat.ndof / 2);
            int n_md_steps_t = std::min(n_steps, 50);
            double hmc_beta_t = 2.0;
            int cheb_degree = 20;
            int cheb_interval = 5;  // Chebyshev probe every N steps

            // Power iteration to estimate λ_max once
            GaugeField gauge_lmax = gauge;
            DiracOp D_lmax(lat, gauge_lmax, mass, wilson_r, c_sw);
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
                std::mt19937 rng_t(seed + 9999);
                mom_t.randomise(rng_t);

                auto D_t = std::make_unique<DiracOp>(lat, gauge_t, mass, wilson_r, c_sw);
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

                int total_tracker_mv = 0;  // D†D matvecs used by tracker
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
                    //      Order 1: span{v, δA v}
                    //      Order 2: span{v, δA v, (δA)² v}
                    //      Order p: span{v, δA v, ..., (δA)^p v}
                    //    After each call, compress() rotates to optimal Ritz vectors,
                    //    so the next call naturally captures the next order correction.
                    if (strat.perturb_order > 0) {
                        auto dD = [&D_t, &mom_t, dt](const Vec& src, Vec& dst) {
                            D_t->apply_delta_D(src, dst, mom_t.pi, dt);
                        };
                        // δD† v = γ₅ δD(γ₅ v) for Wilson-Dirac
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
                        // Cost: order × n_ev × ~1.25 D†D
                        step_tracker_mv += strat.perturb_order * n_ev_t * 5 / 4;
                    }

                    // 2. Force-evolve tracker BEFORE gauge update
                    //    (uses current D and momentum to compute δD)
                    {
                        auto dD = [&D_t, &mom_t, dt](const Vec& src, Vec& dst) {
                            D_t->apply_delta_D(src, dst, mom_t.pi, dt);
                        };
                        tracker.force_update(dD);
                        // force_update: 0 D†D matvecs
                    }

                    // 3. Physical MD gauge update: U → exp(iεπ) U
                    for (int mu = 0; mu < 2; mu++)
                        for (int sv = 0; sv < lat.V; sv++)
                            gauge_t.U[mu][sv] *= std::exp(cx(0, dt * mom_t.pi[mu][sv]));

                    // Recreate Dirac operator for new gauge
                    D_t = std::make_unique<DiracOp>(lat, gauge_t, mass, wilson_r, c_sw);

                    // 4. CG solve (simulating fermion force computation)
                    //    Harvest Ritz pairs for free
                    int cg_iters = 0;
                    int n_absorbed = 0;
                    if (strat.use_harvest) {
                        Vec rhs_t = random_vec(lat.ndof, rng_t);
                        std::vector<RitzPair> cg_ritz;
                        auto cg_res = cg_solve_ritz(A_t, lat.ndof, rhs_t,
                                                     max_iter, tol,
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
                        step_tracker_mv += cheb_degree + 1; // D†D + 1 D application
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
                std::cout << "  Tracker overhead: " << total_tracker_mv << " D†D matvecs\n";
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
            DiracOp D(lat, gauge, mass, wilson_r, c_sw);
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

    DiracOp D0(lat, gauge, mass, wilson_r, c_sw);

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
            DiracOp D(lat, gauge, mass, wilson_r, c_sw);
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
            DiracOp D(lat, gauge, mass, wilson_r, c_sw);
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
