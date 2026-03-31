// main.cpp — CLI parser and mode dispatcher for 2D Schwinger MG solver
#include "config.h"
#include "types.h"
#include "lattice.h"
#include "gauge.h"
#include "hmc.h"
#include "mode_hmc.h"
#include "mode_mg_hmc.h"
#include "mode_forecast_study.h"
#include "mode_studies.h"
#include "mode_hmc_benchmark.h"

#include <iostream>
#include <iomanip>
#include <cstring>
#include <string>
#include <omp.h>

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
        << "  --mu-t <float>        Twisted mass parameter (0=untwisted) [0.0]\n"
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
        << "  --even-odd            Use even-odd (Schur complement) preconditioning\n"
        << "  --hmc-omelyan         Use Omelyan (2MN) integrator instead of leapfrog\n"
        << "  --hmc-traj <int>      Number of measurement trajectories           [100]\n"
        << "  --hmc-tau <float>     Trajectory length (MD time units)             [1.0]\n"
        << "  --hmc-steps <int>     Leapfrog/Omelyan steps per trajectory         [20]\n"
        << "  --hmc-beta <float>    Gauge coupling beta                           [2.0]\n"
        << "  --hmc-therm <int>     Thermalisation trajectories (always accept)   [20]\n"
        << "  --hmc-save-every <N>  Save gauge every N measurement trajectories   [10]\n"
        << "  --hmc-save-prefix <s> Gauge file prefix                             [gauge]\n"
        << "  --hmc-load <file>     Load initial gauge configuration from file\n"
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
        << "  --hmc-eigen-forecast  Enable chronological eigenspace forecasting\n"
        << "  --forecast-study      Run forecasting vs baseline comparative study\n"
        << "  --rebuild-freq <int>  Warm MG rebuild period for study        [5]\n"
        << "  --hmc-revtest         Run reversibility test (forward+backward)\n"
        << "  --eigensolver <s>     Eigensolver: trlm or feast           [trlm]\n"
        << "  --feast-emax <float>  FEAST spectral window upper bound     [auto]\n"
        << "  --verify-forces       Numerical derivative force verification\n"
        << "  --hmc-fresh-period <N> Fresh TRLM every N trajectories  [10]\n"
        << "\n"
        << "=== Sparse Coarse Operator Study ===\n"
        << "  --test-sparse-coarse  Run coarse eigenvector evolution study\n"
        << "  --n-defl <int>        Number of deflation vectors       [16]\n"
        << "  -n <int>              Number of gauge evolution steps    [30]\n"
        << "  -e <float>            Gauge perturbation per step       [0.12]\n"
        << "\n"
        << "=== Eigenspace Tracking ===\n"
        << "  --hmc-tracking        Enable eigenspace tracking during HMC\n"
        << "                        Harvests CG solutions + Ritz pairs to maintain\n"
        << "                        MG quality. Starts after thermalisation.\n"
        << "  --tracking-n-ritz <N> Ritz pairs extracted per CG solve     [4]\n"
        << "                        Zero extra matvecs (CG-Lanczos equivalence).\n"
        << "  --tracking-pool <N>   Max eigenvectors in tracking pool     [16]\n"
        << "                        Pool auto-compresses via RR when full.\n"
        << "  --tracking-n-ev <N>   Wanted eigenvectors for MG prolongator [4]\n"
        << "  --tracking-history <N> CG solutions to keep for extrapolation [1]\n"
        << "                        1=constant, 2=linear, 3=quadratic, etc.\n"
        << "  --rebuild-freq <N>    Rebuild MG from pool every N traj     [5]\n"
        << "                        0=Galerkin only (no pool refresh).\n"
        << "\n"
        << "=== Benchmarks ===\n"
        << "  --hmc-benchmark       Run HMC physics benchmark (force/rev/stats)\n"
        << "  --feast-benchmark     Run FEAST vs TRLM eigenspace comparison\n"
        << "\n"
        << "=== Deflation Test ===\n"
        << "  --test-deflation      Run coarse eigenvector prolongation test\n"
        << "  --cheb-only           Only test Chebyshev-filtered eigenvectors\n"
        << "\n"
        << "=== Output Control ===\n"
        << "  --verbosity <0-3>     0=silent 1=summary 2=verbose 3=debug  [2]\n"
        << "  --quiet               Alias for --verbosity 0\n"
        << "\n"
        << "  --help, -h            Show this help\n"
        << "\n"
        << "======================= EXAMPLES =======================\n"
        << "\n"
        << "--- Basic MG Solver ---\n"
        << "  " << prog << " -L 16 --mg-levels 2\n"
        << "  " << prog << " -L 32 --mg-levels 2 --csw 1.0\n"
        << "  " << prog << " -L 32 --mg-levels 3 -k 8\n"
        << "\n"
        << "--- Standard HMC (Wilson) ---\n"
        << "  " << prog << " --hmc -L 16 --hmc-beta 2.0 --hmc-traj 10 --hmc-therm 10\n"
        << "\n"
        << "--- Standard HMC (Wilson-Clover) ---\n"
        << "  " << prog << " --hmc --csw 1.0 -L 16 --hmc-beta 2.0 --hmc-traj 10 --hmc-therm 10\n"
        << "\n"
        << "--- Standard HMC with Even-Odd ---\n"
        << "  " << prog << " --hmc --even-odd -L 16 --hmc-beta 2.0 --hmc-traj 10 --hmc-therm 10\n"
        << "\n"
        << "--- Standard HMC with Clover + Even-Odd ---\n"
        << "  " << prog << " --hmc --even-odd --csw 1.0 -L 16 --hmc-beta 2.0 --hmc-traj 10 --hmc-therm 10\n"
        << "\n"
        << "--- MG Multi-Timescale HMC (Leapfrog) ---\n"
        << "  " << prog << " -L 16 --hmc-mg-multiscale --mg-levels 2 -b 4 -k 4"
        << " --hmc-n-outer 10 --hmc-n-inner 2 --hmc-n-defl 8 --hmc-traj 10 --hmc-therm 10\n"
        << "\n"
        << "--- MG Multi-Timescale HMC (Omelyan) ---\n"
        << "  " << prog << " -L 16 --hmc-mg-multiscale --hmc-omelyan --mg-levels 2 -b 4 -k 4"
        << " --hmc-n-outer 5 --hmc-n-inner 2 --hmc-n-defl 8 --hmc-traj 10 --hmc-therm 10\n"
        << "\n"
        << "--- Nested Force-Gradient Integrator (MILC-style) ---\n"
        << "  " << prog << " -L 16 --hmc-mg-multiscale --hmc-force-gradient --mg-levels 2 -b 4 -k 4"
        << " --hmc-n-outer 5 --hmc-n-inner 3 --hmc-n-defl 8 --hmc-traj 10 --hmc-therm 10\n"
        << "\n"
        << "--- Nested FGI with Wilson-Clover ---\n"
        << "  " << prog << " -L 16 --hmc-mg-multiscale --hmc-force-gradient --csw 1.0"
        << " --mg-levels 2 -b 4 -k 4 --hmc-n-outer 5 --hmc-n-inner 3 --hmc-n-defl 8 --hmc-traj 10 --hmc-therm 10\n"
        << "\n"
        << "--- Nested FGI with Periodic Deflation Refresh ---\n"
        << "  " << prog << " -L 16 --hmc-mg-multiscale --hmc-force-gradient --mg-levels 2 -b 4 -k 4"
        << " --hmc-n-outer 5 --hmc-n-inner 5 --hmc-n-defl 8 --hmc-defl-refresh 3 --hmc-traj 10 --hmc-therm 10\n"
        << "\n"
        << "--- MG Multi-Timescale with Even-Odd ---\n"
        << "  " << prog << " -L 16 --hmc-mg-multiscale --even-odd --mg-levels 2 -b 4 -k 4"
        << " --hmc-n-outer 10 --hmc-n-inner 2 --hmc-n-defl 8 --hmc-traj 10 --hmc-therm 10\n"
        << "\n"
        << "--- MG FGI with Even-Odd ---\n"
        << "  " << prog << " -L 16 --hmc-mg-multiscale --hmc-force-gradient --even-odd"
        << " --mg-levels 2 -b 4 -k 4 --hmc-n-outer 5 --hmc-n-inner 3 --hmc-n-defl 8 --hmc-traj 10 --hmc-therm 10\n"
        << "\n"
        << "--- Reversibility Test (Wilson) ---\n"
        << "  " << prog << " -L 16 --hmc-mg-multiscale --hmc-revtest --mg-levels 2 -b 4 -k 4"
        << " --hmc-n-outer 5 --hmc-n-inner 3 --hmc-n-defl 8\n"
        << "\n"
        << "--- Reversibility Test (Wilson-Clover) ---\n"
        << "  " << prog << " -L 16 --hmc-mg-multiscale --hmc-revtest --csw 1.0 --mg-levels 2 -b 4 -k 4"
        << " --hmc-n-outer 5 --hmc-n-inner 3 --hmc-n-defl 8\n"
        << "\n"
        << "--- Force Verification (Wilson) ---\n"
        << "  " << prog << " -L 8 --verify-forces --hmc-beta 2.0\n"
        << "\n"
        << "--- Force Verification (Clover) ---\n"
        << "  " << prog << " -L 8 --verify-forces --csw 1.0 --hmc-beta 2.0\n"
        << "\n"
        << "--- Force Verification (Wilson E/O) ---\n"
        << "  " << prog << " -L 8 --verify-forces --even-odd --hmc-beta 2.0\n"
        << "\n"
        << "--- Force Verification (Clover E/O) ---\n"
        << "  " << prog << " -L 8 --verify-forces --even-odd --csw 1.0 --hmc-beta 2.0\n"
        << "\n"
        << "--- Sparse Coarse Eigenvector Evolution Study ---\n"
        << "  " << prog << " -L 32 --test-sparse-coarse --mg-levels 2 -b 4 -k 4"
        << " --n-defl 8 -n 5 -e 0.02\n";
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);  // line-buffered for crash diagnostics

    // --- Config structs with defaults ---
    LatticeConfig lcfg;
    MGConfig mcfg;
    SolverConfig scfg;
    HMCConfig hcfg;
    StudyConfig study;

    // --- Mode flags ---
    bool run_hmc = false;
    bool hmc_multiscale = false;
    bool hmc_mg_multiscale = false;
    bool hmc_revtest = false;
    bool verify_forces_flag = false;
    bool forecast_study = false;
    bool test_deflation = false;
    bool test_sparse_coarse = false;
    bool hmc_benchmark = false;
    bool feast_benchmark = false;
    int cycle_type = -1;

    // --- Parse args ---
    for (int i = 1; i < argc; i++) {
        auto match = [&](const char* flag) { return std::strcmp(argv[i], flag) == 0; };
        auto next_int = [&]() { return std::atoi(argv[++i]); };
        auto next_dbl = [&]() { return std::atof(argv[++i]); };

        if (match("-h") || match("--help")) { print_usage(argv[0]); return 0; }
        else if (match("-L"))        lcfg.L           = next_int();
        else if (match("-m"))        lcfg.mass         = next_dbl();
        else if (match("-e"))        study.eps          = next_dbl();
        else if (match("-b"))        mcfg.block_size    = next_int();
        else if (match("-k"))        mcfg.k_null        = next_int();
        else if (match("-n"))        study.n_steps      = next_int();
        else if (match("-s"))        lcfg.seed          = next_int();
        else if (match("-w"))        lcfg.hot_width     = next_dbl();
        else if (match("-r"))        lcfg.wilson_r      = next_dbl();
        else if (match("--csw"))     lcfg.c_sw          = next_dbl();
        else if (match("--mu-t"))    lcfg.mu_t          = next_dbl();
        else if (match("--tol"))     scfg.tol           = next_dbl();
        else if (match("--krylov"))  scfg.krylov        = next_int();
        else if (match("-t"))        lcfg.n_threads     = next_int();
        else if (match("--maxiter")) scfg.max_iter      = next_int();
        else if (match("--mg-levels")) mcfg.mg_levels   = next_int();
        else if (match("--coarse-block")) mcfg.coarse_block = next_int();
        else if (match("--refresh")) study.refresh_interval = next_int();
        else if (match("--adaptive-threshold")) study.adaptive_threshold = next_dbl();
        else if (match("--w-cycle")) cycle_type = 1;
        else if (match("--v-cycle")) cycle_type = 0;
        else if (match("--no-mg"))   mcfg.no_mg = true;
        else if (match("--symmetric-mg")) mcfg.symmetric_mg = true;
        else if (match("--hmc"))     run_hmc = true;
        else if (match("--hmc-traj"))  hcfg.n_traj       = next_int();
        else if (match("--hmc-tau"))   hcfg.tau           = next_dbl();
        else if (match("--hmc-steps")) hcfg.n_steps       = next_int();
        else if (match("--hmc-beta"))  hcfg.beta          = next_dbl();
        else if (match("--hmc-therm")) hcfg.n_therm       = next_int();
        else if (match("--hmc-save-every")) hcfg.save_every = next_int();
        else if (match("--hmc-save-prefix")) hcfg.save_prefix = argv[++i];
        else if (match("--hmc-load")) hcfg.load_file     = argv[++i];
        else if (match("--hmc-multiscale")) hmc_multiscale = true;
        else if (match("--hmc-mg-multiscale")) hmc_mg_multiscale = true;
        else if (match("--hmc-force-gradient")) hcfg.force_gradient = 1;
        else if (match("--hmc-fgi-qpqpq")) hcfg.force_gradient = 2;
        else if (match("--hmc-omelyan")) hcfg.omelyan     = true;
        else if (match("--hmc-revtest")) hmc_revtest       = true;
        else if (match("--verify-forces")) verify_forces_flag = true;
        else if (match("--hmc-defl-refresh")) hcfg.defl_refresh = next_int();
        else if (match("--hmc-eigen-forecast")) hcfg.eigen_forecast = true;
        else if (match("--forecast-study")) forecast_study = true;
        else if (match("--hmc-benchmark")) hmc_benchmark = true;
        else if (match("--hmc-tracking")) hcfg.enable_tracking = true;
        else if (match("--verbosity")) lcfg.verbosity = next_int();
        else if (match("--quiet"))     lcfg.verbosity = 0;
        else if (match("--tracking-n-ritz")) hcfg.tracking_n_ritz = next_int();
        else if (match("--tracking-pool")) hcfg.tracking_pool_cap = next_int();
        else if (match("--tracking-n-ev")) hcfg.tracking_n_ev = next_int();
        else if (match("--tracking-history")) hcfg.tracking_history = next_int();
        else if (match("--inner-tracking")) hcfg.inner_tracking = next_int();
        else if (match("--feast-benchmark")) feast_benchmark = true;
        else if (match("--rebuild-freq")) hcfg.rebuild_freq = next_int();
        else if (match("--mg-perturb-freq")) hcfg.mg_perturb_freq = next_int();
        else if (match("--eigensolver")) scfg.eigensolver  = argv[++i];
        else if (match("--feast-emax")) scfg.feast_emax   = next_dbl();
        else if (match("--only-arms")) hcfg.only_arms     = argv[++i];
        else if (match("--even-odd")) hcfg.use_eo         = true;
        else if (match("--hmc-n-outer")) hcfg.n_outer     = next_int();
        else if (match("--hmc-n-inner")) hcfg.n_inner     = next_int();
        else if (match("--hmc-n-defl")) hcfg.n_defl       = next_int();
        else if (match("--hmc-fresh-period")) hcfg.fresh_period = next_int();
        else if (match("--test-deflation")) test_deflation = true;
        else if (match("--test-sparse-coarse")) test_sparse_coarse = true;
        else if (match("--cheb-only")) { test_deflation = true; study.cheb_only = true; }
        else if (match("--n-defl")) scfg.n_defl = next_int();
        else {
            std::cerr << "Unknown option: " << argv[i] << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    // --- Resolve derived settings ---
    if (cycle_type < 0) cycle_type = (mcfg.mg_levels > 1) ? 1 : 0;
    mcfg.w_cycle = (cycle_type == 1);

    if (lcfg.L % mcfg.block_size != 0) {
        std::cerr << "Error: L (" << lcfg.L << ") must be divisible by block size ("
                  << mcfg.block_size << ")\n";
        return 1;
    }
    if (lcfg.L < 4 || mcfg.block_size < 1 || mcfg.k_null < 1 || study.n_steps < 1) {
        std::cerr << "Error: invalid parameter (L >= 4, b >= 1, k >= 1, n >= 1)\n";
        return 1;
    }

    if (lcfg.n_threads > 0) omp_set_num_threads(lcfg.n_threads);
    g_verbosity = lcfg.verbosity;

    // --- Common setup ---
    VOUT(V_SUMMARY) << "================================================================\n"
                     << "  2D Schwinger Model: MG with EigCG-Inspired Ritz Harvesting\n"
                     << "================================================================\n\n";

    std::mt19937 rng(lcfg.seed);
    Lattice lat(lcfg.L);
    GaugeField gauge(lat);
    gauge.randomise(rng, lcfg.hot_width);

    VOUT(V_SUMMARY) << "Lattice: " << lcfg.L << "x" << lcfg.L
              << "  DOF: " << lat.ndof
              << "  mass: " << lcfg.mass
              << "  r: " << lcfg.wilson_r << "\n";
    VOUT(V_VERBOSE) << "Block: " << mcfg.block_size << "x" << mcfg.block_size
              << "  Null vectors/block: " << mcfg.k_null << "\n";
    VOUT(V_VERBOSE) << "MG levels: " << mcfg.mg_levels
              << "  cycle: " << (mcfg.w_cycle ? "W" : "V")
              << (mcfg.mg_levels > 1
                  ? "  coarse_block: " + std::to_string(mcfg.resolved_coarse_block())
                    + "  refresh: " + std::to_string(study.refresh_interval)
                  : "")
              << "\n";
    VOUT(V_VERBOSE) << "Krylov: " << scfg.krylov
              << "  maxiter: " << scfg.max_iter
              << "  tol: " << std::scientific << std::setprecision(1) << scfg.tol << "\n";
    VOUT(V_VERBOSE) << "OpenMP threads: " << omp_get_max_threads() << "\n";
    VOUT(V_VERBOSE) << "Gauge perturbation per step: epsilon = " << std::fixed
              << std::setprecision(4) << study.eps
              << "  steps: " << study.n_steps
              << "  seed: " << lcfg.seed << "\n";

    if (!hcfg.load_file.empty()) {
        GaugeHeader hdr;
        if (!load_gauge(gauge, hdr, hcfg.load_file)) return 1;
        VOUT(V_SUMMARY) << "Loaded gauge from " << hcfg.load_file
                  << " (beta=" << hdr.beta << " mass=" << hdr.mass
                  << " plaq=" << hdr.avg_plaq << ")\n";
    }

    VOUT(V_SUMMARY) << "Initial <plaq> = " << std::fixed << std::setprecision(4)
              << gauge.avg_plaq() << "\n\n";

    // --- Mode dispatch ---
    if (verify_forces_flag)
        return run_verify_forces(gauge, lcfg, scfg, hcfg);

    if (feast_benchmark)
        return run_feast_benchmark(gauge, lat, lcfg, mcfg, scfg, hcfg, rng);

    if (hmc_benchmark)
        return run_hmc_benchmark(gauge, lat, lcfg, mcfg, scfg, hcfg, rng);

    if (run_hmc)
        return run_hmc_mode(gauge, lat, lcfg, mcfg, scfg, hcfg, rng);

    if (hmc_multiscale)
        return run_multiscale_hmc(gauge, lat, lcfg, mcfg, scfg, hcfg, rng);

    if (hmc_mg_multiscale)
        return run_mg_hmc(gauge, lat, lcfg, mcfg, scfg, hcfg, rng, hmc_revtest);

    if (forecast_study)
        return run_forecast_study(gauge, lat, lcfg, mcfg, scfg, hcfg, rng);

    if (test_sparse_coarse)
        return run_sparse_coarse_study(gauge, lat, lcfg, mcfg, scfg, hcfg, study, rng);

    if (test_deflation)
        return run_deflation_study(gauge, lat, lcfg, mcfg, scfg, study, rng);

    if (mcfg.no_mg)
        return run_no_mg_baseline(gauge, lat, lcfg, scfg, study, rng);

    // Default: MG solver comparison study
    return run_mg_solver_study(gauge, lat, lcfg, mcfg, scfg, study, rng);
}
