#pragma once
#include "types.h"
#include "linalg.h"
#include "lattice.h"
#include "gauge.h"
#include "dirac.h"
#include "solvers.h"
#include "prolongator.h"
#include "coarse_op.h"
#include <array>
#include <random>
#include <functional>
#include <memory>

// Perturb gauge field by small random phases (simulates one MD step)
void perturb_gauge(GaugeField& g, std::mt19937& rng, double epsilon);

// Gauge field I/O
struct GaugeHeader {
    uint32_t magic;   // 0x53574E47 = "SWNG"
    uint32_t L;
    double beta;
    double mass;
    double avg_plaq;
};

bool save_gauge(const GaugeField& g, double beta, double mass,
                const std::string& filename);
bool load_gauge(GaugeField& g, GaugeHeader& hdr, const std::string& filename);

// Momentum field
struct MomentumField {
    const Lattice& lat;
    std::array<RVec, 2> pi;

    MomentumField(const Lattice& lat_) : lat(lat_) {
        pi[0].resize(lat.V, 0.0);
        pi[1].resize(lat.V, 0.0);
    }

    void randomise(std::mt19937& rng);
    double kinetic_energy() const;
};

// Wilson gauge action
double gauge_action(const GaugeField& g, double beta);

// Gauge force
void gauge_force(const GaugeField& g, double beta,
                 std::array<RVec, 2>& force);

// Fermion action
struct FermionActionResult {
    double action;
    Vec X;
    int cg_iters;
};

FermionActionResult fermion_action(const DiracOp& D, const Vec& phi,
                                    int max_iter, double tol);
FermionActionResult fermion_action_precond(
    const DiracOp& D, const Vec& phi,
    const std::function<Vec(const Vec&)>& precond,
    int max_iter, double tol);

// Fermion force: F = Re(chi† dD/dA X)
// Standard version: chi = D*X (computed internally)
void fermion_force(const DiracOp& D, const Vec& X,
                   std::array<RVec, 2>& force);
// Bilinear version: chi provided explicitly (for even-odd force)
void fermion_force_bilinear(const DiracOp& D, const Vec& chi, const Vec& X,
                            std::array<RVec, 2>& force);
// Clover force contribution (called by fermion_force_bilinear when c_sw != 0)
void fermion_force_clover(const DiracOp& D, const Vec& chi, const Vec& X,
                          std::array<RVec, 2>& force);

// Log-det force from det(D_ee) for clover even-odd HMC
// F = +2 d(log det(D_ee))/dA (only for c_sw != 0, only even sites contribute)
void logdet_ee_force(const DiracOp& D,
                     std::array<RVec, 2>& force);

// Clover plaquette derivative insertion: accumulates weight × dF_01/dA into force.
// Used by fermion_force_clover, logdet_ee_force, and schur_deriv.
void clover_deriv_insert(const DiracOp& D, const RVec& w, double factor,
                         std::array<RVec, 2>& force);

// Hopping-only bilinear force: Re(chi† dD_hop/dA X) without clover contribution
void hopping_force_bilinear(const DiracOp& D, const Vec& chi, const Vec& X,
                            std::array<RVec, 2>& force);

// Even-odd Schur complement force: F = -dS_eo/dA where S_eo = φ_o†(M†M)⁻¹φ_o
// x_o = CG solution, y_o = M x_o
void eo_fermion_force(const DiracOp& D, const EvenOddDiracOp& eoD,
                      const Vec& x_o, const Vec& y_o,
                      std::array<RVec, 2>& force);

// Force verification — returns true if relative error < threshold
bool verify_forces(const GaugeField& g, double beta, double mass, double wilson_r,
                   int max_iter, double tol, double c_sw = 0.0,
                   bool use_eo = false, double err_threshold = 1e-4,
                   double mu_t = 0.0);

// Pseudofermion generation
void generate_pseudofermion(const DiracOp& D, std::mt19937& rng, Vec& phi);

// HMC parameters and results
struct HMCParams {
    double beta;
    double tau;
    int n_steps;
    int cg_maxiter;
    double cg_tol;
    bool use_mg;
    double c_sw = 0.0;
    double mu_t = 0.0;   // twisted mass parameter
    bool use_eo = false;  // even-odd preconditioning
    bool omelyan = false;       // use Omelyan (2MN) integrator instead of leapfrog
    bool force_accept = false;  // always accept (for thermalisation)
};

struct HMCResult {
    bool accepted;
    double dH;         // total Hamiltonian violation
    double dKE;        // kinetic energy change
    double dSG;        // gauge action change
    double dSF;        // fermion action change
    double dLD;        // log-det change (clover e/o only, else 0)
    int total_cg_iters;
};

// --- Eigenspace tracking during HMC ---

struct TrackingState {
    // Chronological initial guess (Brower et al.)
    // Stores up to history_depth previous solutions for extrapolation.
    // With depth=1: x0 = X_{s-1} (constant extrapolation)
    // With depth=2: x0 = 2*X_{s-1} - X_{s-2} (linear)
    // With depth=3: x0 = 3*X_{s-1} - 3*X_{s-2} + X_{s-3} (quadratic)
    std::vector<Vec> solution_history;  // most recent first
    bool has_prev_solution = false;

    // EigenTracker pool — opaque pointer, initialized in hmc_trajectory.cpp
    void* tracker_ptr = nullptr;  // actually EigenTracker*, cast in .cpp
    bool tracker_initialized = false;

    // Configuration
    int n_ritz = 4;
    int pool_capacity = 16;
    int n_ev = 4;
    int history_depth = 1;  // chronological solutions to keep (1-10)

    // Statistics
    int force_eval_count = 0;
    int total_ritz_absorbed = 0;
    int total_solutions_absorbed = 0;

    int total_force_evals = 0;  // accumulates across trajectories

    void reset_trajectory() {
        // Keep solution history across trajectories (gauge is continuous)
        total_force_evals += force_eval_count;
        force_eval_count = 0;
    }

    // Store a new solution and maintain history depth
    void push_solution(Vec&& sol) {
        solution_history.insert(solution_history.begin(), std::move(sol));
        if ((int)solution_history.size() > history_depth)
            solution_history.resize(history_depth);
        has_prev_solution = true;
    }

    // Get extrapolated initial guess from solution history
    // depth=1: x0 = X_{s-1}
    // depth=2: x0 = 2*X_{s-1} - X_{s-2}
    // depth=3: x0 = 3*X_{s-1} - 3*X_{s-2} + X_{s-3}
    Vec extrapolated_x0() const {
        int h = (int)solution_history.size();
        if (h == 0) return {};
        if (h == 1) return solution_history[0];
        int n = (int)solution_history[0].size();
        Vec x0(n, 0.0);
        // Binomial extrapolation coefficients: (-1)^(k+1) * C(h, k)
        // h=2: [2, -1], h=3: [3, -3, 1], h=4: [4, -6, 4, -1], etc.
        for (int k = 0; k < h; k++) {
            // C(h, k+1) * (-1)^k
            double coeff = 1.0;
            for (int j = 0; j < k+1; j++)
                coeff *= (double)(h - j) / (j + 1);
            if (k % 2 == 1) coeff = -coeff;
            for (int i = 0; i < n; i++)
                x0[i] += coeff * solution_history[k][i];
        }
        return x0;
    }

    // Extract the best n_ev vectors from the tracker pool for MG prolongator.
    // Returns empty if tracker not initialized.
    std::vector<Vec> get_null_vectors() const;

    // Clean up tracker allocation
    void destroy_tracker();

    ~TrackingState() { destroy_tracker(); }
    TrackingState() = default;
    TrackingState(const TrackingState&) = delete;
    TrackingState& operator=(const TrackingState&) = delete;
    TrackingState(TrackingState&& o) noexcept
        : solution_history(std::move(o.solution_history)),
          has_prev_solution(o.has_prev_solution),
          tracker_ptr(o.tracker_ptr), tracker_initialized(o.tracker_initialized),
          n_ritz(o.n_ritz), pool_capacity(o.pool_capacity), n_ev(o.n_ev),
          force_eval_count(o.force_eval_count),
          total_ritz_absorbed(o.total_ritz_absorbed),
          total_solutions_absorbed(o.total_solutions_absorbed)
    { o.tracker_ptr = nullptr; o.tracker_initialized = false; }
    TrackingState& operator=(TrackingState&& o) noexcept {
        if (this != &o) {
            destroy_tracker();
            solution_history = std::move(o.solution_history);
            has_prev_solution = o.has_prev_solution;
            tracker_ptr = o.tracker_ptr; tracker_initialized = o.tracker_initialized;
            n_ritz = o.n_ritz; pool_capacity = o.pool_capacity; n_ev = o.n_ev;
            force_eval_count = o.force_eval_count;
            total_ritz_absorbed = o.total_ritz_absorbed;
            total_solutions_absorbed = o.total_solutions_absorbed;
            o.tracker_ptr = nullptr; o.tracker_initialized = false;
        }
        return *this;
    }
};

HMCResult hmc_trajectory(
    GaugeField& gauge, const Lattice& lat, double mass, double wilson_r,
    const HMCParams& params, std::mt19937& rng,
    const std::function<Vec(const Vec&)>* precond = nullptr,
    TrackingState* tracking = nullptr);

// --- Multi-timescale HMC with exact low-mode treatment ---

struct DeflationState {
    std::vector<Vec> eigvecs;    // fine-grid eigenvectors of D†D
    std::vector<double> eigvals; // corresponding eigenvalues
    std::vector<Vec> Dv;         // cached D*eigvec for each vector
    bool valid = false;

    void update_cache(const DiracOp& D) {
        int nd = (int)eigvecs.size();
        Dv.resize(nd);
        for (int i = 0; i < nd; i++) {
            Dv[i].resize(D.lat.ndof);
            D.apply(eigvecs[i], Dv[i]);
        }
    }

    Vec deflated_initial_guess(const Vec& rhs) const {
        int n = (int)rhs.size();
        Vec x0 = zeros(n);
        for (int i = 0; i < (int)eigvecs.size(); i++) {
            if (eigvals[i] > 1e-14) {
                cx coeff = dot(eigvecs[i], rhs) / eigvals[i];
                axpy(coeff, eigvecs[i], x0);
            }
        }
        return x0;
    }
};

struct MultiScaleParams {
    double beta;
    double tau = 1.0;
    int n_outer = 10;
    int n_inner = 5;
    int cg_maxiter = 500;
    double cg_tol = 1e-10;
    double c_sw = 0.0;
    double mu_t = 0.0;
    bool use_eo = false;
};

struct MultiScaleResult {
    bool accepted;
    double dH;
    double dKE, dSG, dSF;
    int highmode_cg_iters;
    int lowmode_force_evals;
    double highmode_time;
    double lowmode_time;
};

void lowmode_fermion_force(const DiracOp& D,
    const DeflationState& defl,
    const Vec& phi,
    std::array<RVec, 2>& force);

void evolve_deflation_state(DeflationState& defl,
    const DiracOp& D_new, bool fresh_trlm = false);

MultiScaleResult hmc_trajectory_multiscale(
    GaugeField& gauge, const Lattice& lat, double mass, double wilson_r,
    const MultiScaleParams& params, DeflationState& defl,
    std::mt19937& rng,
    const std::function<Vec(const Vec&)>* precond = nullptr);

// --- MG-based multi-timescale HMC (coarse-grid deflation) ---

struct CoarseDeflState {
    std::vector<Vec> eigvecs;    // coarse-grid eigenvectors of A_c
    std::vector<double> eigvals; // corresponding eigenvalues

    // Project pseudofermion through restrict→deflation→prolong
    Vec lowmode_solution(const Vec& phi, const Prolongator& P) const {
        Vec phi_c = P.restrict_vec(phi);
        int cdim = (int)phi_c.size();
        Vec x_c = zeros(cdim);
        for (int i = 0; i < (int)eigvecs.size(); i++) {
            if (eigvals[i] > 1e-14) {
                cx coeff = dot(eigvecs[i], phi_c) / eigvals[i];
                axpy(coeff, eigvecs[i], x_c);
            }
        }
        return P.prolong(x_c);
    }
};

// Outer integrator type for the expensive (fermion) force
enum class OuterIntegrator {
    Leapfrog,   // 2nd order, 1 force eval per step
    Omelyan,    // 2nd order optimised (2MN), 2 force evals per step
    FGI         // 4th order Hessian-free force gradient (MILC PQPQP_FGI)
                // P(λh) inner(h/2) FG((1-2λ)h) inner(h/2) P(λh)
                // λ=1/6, ξ=1/72. FG step = 2 force evals via gauge displacement.
};

struct MGMultiScaleParams {
    double beta;
    double tau = 1.0;
    int n_outer = 4;        // outer steps (expensive force: MG-preconditioned solve)
    int n_inner = 5;        // inner steps per outer (cheap force: gauge + low-mode)
    int cg_maxiter = 500;
    double cg_tol = 1e-10;
    int inner_smooth = 3;   // MR smoothing iters in inner force (0=raw projection)
    OuterIntegrator outer_type = OuterIntegrator::Leapfrog;
    int defl_refresh = 0;   // refresh coarse deflation every N inner steps (0=never)
    int mg_perturb_freq = 0; // perturbation-refresh MG prolongator every N inner steps (0=never)
    double c_sw = 0.0;     // clover coefficient
    double mu_t = 0.0;    // twisted mass parameter
    bool use_eo = false;   // even-odd preconditioning
};

struct MGMultiScaleResult {
    bool accepted;
    double dH;
    double dKE, dSG, dSF;
    int highmode_cg_iters;
    int lowmode_force_evals;
    double highmode_time;
    double lowmode_time;
};

// Compute low-mode force via restrict-project-prolong pipeline
// If smooth_iters > 0, applies pre/post smoothing (cheap MG cycle)
void coarse_lowmode_force(const DiracOp& D,
    const CoarseDeflState& cdefl,
    const Prolongator& P,
    const Vec& phi,
    std::array<RVec, 2>& force,
    int smooth_iters = 0);

// Evolve coarse deflation state (RR on coarse operator)
// If forecast is provided, pre-rotates eigenvectors using predicted rotation
// and stores the generator for future predictions.
struct EigenForecastState;  // forward declaration
void evolve_coarse_deflation(CoarseDeflState& cdefl,
    const SparseCoarseOp& Ac_new,
    EigenForecastState* forecast = nullptr);

// MG multi-timescale trajectory
struct MGHierarchy;  // forward declaration
MGMultiScaleResult hmc_trajectory_mg_multiscale(
    GaugeField& gauge, const Lattice& lat, double mass, double wilson_r,
    const MGMultiScaleParams& params,
    CoarseDeflState& cdefl,
    Prolongator& P,
    std::function<Vec(const Vec&)>& mg_precond,
    std::mt19937& rng,
    EigenForecastState* forecast = nullptr,
    const std::function<void()>* pre_solve = nullptr,
    MGHierarchy* mg_hierarchy = nullptr);  // called before each CG

// Reversibility test: forward trajectory → negate momenta → backward trajectory
// Returns ||U_final - U_initial|| / ||U_initial|| (should be ~machine epsilon)
struct ReversibilityResult {
    double gauge_delta;     // relative gauge field difference
    double mom_delta;       // relative momentum difference
    double dH_forward;      // dH of forward trajectory
    double dH_backward;     // dH of backward trajectory (should equal -dH_forward)
    int total_cg;
};

ReversibilityResult reversibility_test_mg_multiscale(
    GaugeField& gauge, const Lattice& lat, double mass, double wilson_r,
    const MGMultiScaleParams& params,
    CoarseDeflState& cdefl,
    Prolongator& P,
    std::function<Vec(const Vec&)>& mg_precond,
    std::mt19937& rng);

// Plain leapfrog MD evolution (factored out for reversibility testing).
// Evolves gauge and mom in-place. Returns total CG iterations.
// tracking: optional eigenspace tracking state (nullptr = disabled)
int plain_leapfrog_evolve(
    GaugeField& gauge, MomentumField& mom,
    const Lattice& lat, double mass, double wilson_r,
    const Vec& phi, const HMCParams& params,
    const std::function<Vec(const Vec&)>* precond = nullptr,
    TrackingState* tracking = nullptr);

// Plain-HMC reversibility test: forward → negate π → backward
ReversibilityResult reversibility_test_plain(
    GaugeField& gauge, const Lattice& lat, double mass, double wilson_r,
    const HMCParams& params, std::mt19937& rng,
    const std::function<Vec(const Vec&)>* precond = nullptr);
