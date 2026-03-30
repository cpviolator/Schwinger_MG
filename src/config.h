#pragma once
#include <string>

struct LatticeConfig {
    int L = 16;
    double mass = 0.05;
    double wilson_r = 1.0;
    double c_sw = 0.0;
    double mu_t = 0.0;
    int seed = 42;
    int n_threads = 0;
    double hot_width = 0.4;
};

struct MGConfig {
    int mg_levels = 1;
    int block_size = 4;
    int k_null = 4;
    int coarse_block = 0;   // 0 = k_null * 4
    bool w_cycle = true;
    int pre_smooth = 3;
    int post_smooth = 3;
    bool no_mg = false;
    bool symmetric_mg = false;

    int resolved_coarse_block() const {
        return coarse_block > 0 ? coarse_block : k_null * 4;
    }
};

struct SolverConfig {
    double tol = 1e-10;
    int krylov = 30;
    int max_iter = 300;
    std::string eigensolver = "trlm";
    double feast_emax = 0.0;
    int n_defl = 0;   // from --n-defl (0 = use mode-specific default)
};

struct HMCConfig {
    double beta = 2.0;
    double tau = 1.0;
    int n_steps = 20;
    int n_traj = 100;
    int n_therm = 20;
    int n_outer = 10;
    int n_inner = 5;
    int n_defl = 8;
    int save_every = 10;
    int fresh_period = 10;
    int defl_refresh = 0;
    bool use_eo = false;
    bool force_gradient = false;
    bool omelyan = false;
    bool eigen_forecast = false;
    std::string save_prefix = "gauge";
    std::string load_file;
    // Study options
    int rebuild_freq = 5;
    int mg_perturb_freq = 0;
    std::string only_arms;
    // Eigenspace tracking
    bool enable_tracking = false;
    int tracking_n_ritz = 4;
    int tracking_pool_cap = 16;
    int tracking_n_ev = 4;
};

struct StudyConfig {
    double eps = 0.12;
    int n_steps = 30;
    bool cheb_only = false;
    int refresh_interval = 3;
    double adaptive_threshold = 1.3;
};
