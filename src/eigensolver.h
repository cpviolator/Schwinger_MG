#pragma once
#include "types.h"
#include <vector>
#include <random>

struct LOBPCGResult {
    std::vector<Vec> eigvecs;
    std::vector<double> eigvals;
    int iterations;
};

struct ChebSubspaceResult {
    std::vector<Vec> eigvecs;
    std::vector<double> eigvals;
    int iterations;
    double lambda_max_used;
};

struct TRLMResult {
    std::vector<Vec> eigvecs;
    std::vector<double> eigvals;
    int iterations;
    int num_restarts;
    bool converged;
};

void jacobi_eigen(std::vector<Vec>& A_cols, int n,
                  RVec& evals, std::vector<Vec>& evecs);

void lanczos_eigen(std::vector<Vec>& A_cols, int n,
                   RVec& evals, std::vector<Vec>& evecs);

ChebSubspaceResult chebyshev_subspace_iteration(
    const OpApply& A, int n, int k,
    const std::vector<Vec>& X0,
    int poly_deg = 20, int max_iter = 10, double tol = 1e-8,
    double lambda_max = 0.0);

LOBPCGResult lobpcg_update(
    const OpApply& A, int n, int k,
    const std::vector<Vec>& X0,
    const std::function<Vec(const Vec&)>& precond,
    int max_iter = 5, double tol = 1e-8);

// Saved Krylov subspace state from a converged TRLM run.
// Stores n_kr orthonormal vectors spanning the Krylov space,
// allowing cheap re-projection when the operator changes.
struct TRLMState {
    std::vector<Vec> kSpace;  // n_kr orthonormal Krylov vectors
    int n_ev;                 // number of wanted eigenpairs
    int n_kr;                 // Krylov space size
    int n;                    // vector dimension
    bool valid = false;
};

// Thick Restart Lanczos Method with optional Chebyshev acceleration.
// Computes n_ev smallest eigenvalues of Hermitian operator A.
//   A: operator to eigensolve
//   n: vector dimension
//   n_ev: number of eigenvalues wanted
//   n_kr: Krylov space size (must be >= n_ev + 6)
//   max_restarts: maximum restart iterations
//   tol: convergence tolerance
//   poly_deg: Chebyshev polynomial degree (0 = no Chebyshev)
//   a_min: Chebyshev lower bound (0 = auto)
//   a_max: Chebyshev upper bound (0 = auto-estimate)
//   state_out: if non-null, save Krylov subspace for later re-projection
TRLMResult trlm_eigensolver(
    const OpApply& A, int n, int n_ev,
    int n_kr = 0, int max_restarts = 100, double tol = 1e-10,
    int poly_deg = 0, double a_min = 0.0, double a_max = 0.0,
    TRLMState* state_out = nullptr,
    const Vec* start_vec = nullptr);

// Re-project a saved Krylov subspace onto a new operator.
// Uses the previous eigenvectors as starting vector for fresh TRLM.
TRLMResult trlm_update(
    const OpApply& A_new, const TRLMState& state,
    int max_restarts = 20, double tol = 1e-10,
    TRLMState* state_out = nullptr);

// Rayleigh-Ritz eigenvector evolution: project A_new onto the subspace
// of k eigenvectors, diagonalise the k×k projected matrix, return
// updated eigenvectors and eigenvalues.
// Cost: k matvecs + k² dots.
// The returned eigenvectors are exact within span{v_1,...,v_k} but
// may have leaked out of the subspace if the operator changed significantly.
struct RREvolveResult {
    std::vector<Vec> eigvecs;
    std::vector<double> eigvals;
    double max_residual;  // max ||A v_i - λ_i v_i|| / ||A v_i||
    int matvecs;
    std::vector<Vec> rotation;  // k×k rotation matrix U (eigenvectors of M)
};

RREvolveResult rr_evolve(
    const OpApply& A_new,
    const std::vector<Vec>& old_eigvecs,
    int n);

// Force-based eigenvector evolution (zero matvecs).
// Uses perturbation theory: δ(D†D) ≈ δD† D + D† δD
// where δD is determined by the momentum field π and step size dt.
//
// Inputs:
//   eigvecs: current eigenvectors of D†D
//   eigvals: current eigenvalues
//   Dv: D applied to each eigenvector (stored auxiliary vectors)
//   apply_deltaD: function computing (D_new - D_old) v
//   apply_deltaD_dag: function computing (D_new - D_old)† v
//   n: vector dimension
//
// Outputs:
//   Updated eigvecs, eigvals, Dv (all rotated in-place)
//   Returns max residual norm and number of effective matvecs (0)
struct ForceEvolveResult {
    std::vector<Vec> eigvecs;
    std::vector<double> eigvals;
    std::vector<Vec> Dv;       // updated D v_i
    double max_residual;
    int matvecs;               // always 0 for force-based
};

ForceEvolveResult force_evolve(
    const std::vector<Vec>& eigvecs,
    const std::vector<double>& eigvals,
    const std::vector<Vec>& Dv,
    const std::function<void(const Vec&, Vec&)>& apply_deltaD,
    const std::function<void(const Vec&, Vec&)>& apply_deltaD_dag,
    int n);

// Variant with pre-computed delta_D * v_i vectors (avoids functor overhead)
ForceEvolveResult force_evolve_precomputed(
    const std::vector<Vec>& eigvecs,
    const std::vector<double>& eigvals,
    const std::vector<Vec>& Dv,
    const std::vector<Vec>& dDv,   // pre-computed delta_D * v_i
    int n);

// --- Chronological generator forecasting for eigenspace evolution ---
// Tracks the Hermitian generator H of the RR rotation U = exp(iH) across
// trajectories. Extrapolates H to predict the next rotation, pre-rotates
// eigenvectors before RR to improve accuracy or enable RR skipping.
struct EigenForecastState {
    std::vector<std::vector<Vec>> H_history; // circular buffer of generators (k×k)
    int k = 0;                               // eigenvector count
    int history_len = 0;
    static constexpr int max_history = 3;    // for quadratic extrapolation

    void reset() { history_len = 0; H_history.clear(); k = 0; }
};

// Extract Hermitian generator H ≈ -i(U - I) from rotation matrix U
void extract_generator(const std::vector<Vec>& U_evecs, int k,
                       std::vector<Vec>& H_cols);

// Extrapolate generator history and construct predicted rotation R = exp(iH_pred)
// Returns k×k unitary matrix as cols[col][row]
std::vector<Vec> forecast_rotation(const EigenForecastState& state);

// Apply k×k rotation matrix R to eigenvectors: new_v_i = Σ_j R[j][i] * old_v_j
void apply_rotation(std::vector<Vec>& eigvecs,
                    const std::vector<Vec>& R, int n);

// Multiply two k×k matrices: C = A × B (stored as cols[col][row])
void mat_mul_kk(const std::vector<Vec>& A, const std::vector<Vec>& B,
                std::vector<Vec>& C, int k);

// Frobenius norm of k×k matrix
double frobenius_norm(const std::vector<Vec>& M, int k);

// Hybrid eigenvector tracker: maintains n_kr Krylov vectors with force-based
// rotation, periodically extends with Lanczos steps using the real operator.
//
// The full cycle:
//  1. Force-based update: rotate n_kr vectors using δD (0 matvecs)
//  2. Lanczos extension: run n_ext Lanczos steps from residual direction
//     using A_new, orthogonalising against existing subspace (n_ext matvecs)
//  3. Thick restart: project A_new onto the (n_kr + n_ext) subspace,
//     diagonalise, keep best n_kr Ritz vectors
//
// This brings fresh directions into the subspace at low cost.
struct HybridTrackerState {
    std::vector<Vec> kSpace;    // n_kr orthonormal Krylov vectors
    std::vector<Vec> Dv;        // D v_i for force-based updates
    std::vector<double> eigvals; // n_ev Ritz values (from last projection)
    int n_ev;                    // number of wanted eigenpairs
    int n_kr;                    // Krylov space size
    int n;                       // vector dimension
    bool valid = false;
};

struct HybridTrackResult {
    std::vector<double> eigvals;  // n_ev smallest Ritz values
    double max_residual;          // max residual of n_ev eigenvectors
    int matvecs;                  // matvecs used this step
};

// Initialise tracker from a converged TRLM result.
// Stores all n_kr Krylov vectors + their D applications.
HybridTrackerState hybrid_tracker_init(
    const OpApply& A,
    const std::function<void(const Vec&, Vec&)>& apply_D,
    const TRLMResult& trlm_result,
    int n, int n_ev, int n_kr);

// Force-based step: rotate kSpace using δD (0 full matvecs).
HybridTrackResult hybrid_force_step(
    HybridTrackerState& state,
    const std::function<void(const Vec&, Vec&)>& apply_deltaD);

// Lanczos extension step: extend subspace with n_ext Lanczos vectors,
// then thick-restart back to n_kr. Returns updated Ritz values.
HybridTrackResult hybrid_lanczos_step(
    HybridTrackerState& state,
    const OpApply& A,
    const std::function<void(const Vec&, Vec&)>& apply_D,
    int n_ext = 0);

// =====================================================================
//  EigenTracker — pool-based eigenvector maintenance
//
//  Maintains a pool of orthonormal vectors spanning an approximate
//  low-eigenvalue subspace of D†D.  For each pool vector v_i, caches
//  Dv_i = D v_i so that the projected matrix v_i† D†D v_j = Dv_i† Dv_j
//  can be computed without any D†D matvecs.
//
//  Information sources (in order of cost):
//    1. Force evolution  — 0 D†D matvecs (sparse δD only)
//    2. Solver harvest   — 0 extra matvecs (Ritz pairs from CG)
//    3. Chebyshev probe  — ~degree D†D matvecs
//    4. Fresh TRLM       — ~300 matvecs (expensive, last resort)
//
//  Usage pattern in HMC:
//    tracker.init(trlm_result, apply_D, n, n_ev, pool_cap);
//    for each MD step:
//      tracker.force_update(apply_deltaD);   // rotate subspace
//      // ... gauge update ...
//      auto [cg_res, ritz] = cg_solve_ritz(A, n, rhs, ...);
//      tracker.absorb(ritz_vecs, apply_D);   // harvest solver info
//      if (step % K == 0)
//        tracker.chebyshev_probe(A, rng, ...); // occasional probe
//      tracker.compress();                     // re-diag, keep best
// =====================================================================
struct EigenTracker {
    std::vector<Vec> pool;        // pool vectors (orthonormal, Ritz-ordered)
    std::vector<Vec> Dpool;       // D applied to each pool vector
    std::vector<double> eigvals;  // n_ev eigenvalue estimates
    int n_ev;                     // number of wanted eigenpairs
    int pool_capacity;            // max pool size
    int n;                        // vector dimension
    bool valid = false;

    // Initialise from a converged TRLM result.
    // Stores trlm eigenvectors + fills remaining slots with random vectors.
    // Computes D v_i for every pool vector (pool_capacity D applications).
    void init(const TRLMResult& trlm,
              const std::function<void(const Vec&, Vec&)>& apply_D,
              int n_, int n_ev_, int pool_capacity_);

    // Force-evolve: update Dpool via δD, re-project, re-diag, rotate.
    // Cost: pool_size sparse δD applications + O(pool²) dots.
    // Zero full D†D matvecs.
    void force_update(
        const std::function<void(const Vec&, Vec&)>& apply_deltaD);

    // Absorb new vectors (e.g. from CG Ritz extraction).
    // Orthogonalises against existing pool, adds vectors with significant
    // norm, computes D v_new for each.  Returns number actually absorbed.
    // If pool exceeds capacity, calls compress() automatically.
    // Cost: n_absorbed D applications.
    int absorb(const std::vector<Vec>& new_vecs,
               const std::function<void(const Vec&, Vec&)>& apply_D);

    // Chebyshev-filtered random probe: apply T_p(D†D) to a random vector,
    // orthogonalise against pool, absorb.
    // Cost: degree D†D applications + 1 D application.
    void chebyshev_probe(const OpApply& A,
                         const std::function<void(const Vec&, Vec&)>& apply_D,
                         std::mt19937& rng,
                         double lambda_max, int degree);

    // Compress: RR-project D†D onto pool using cached Dpool, diagonalise,
    // keep best pool_capacity vectors.  Zero D†D matvecs.
    void compress();

    // Perturbation-directed extension: compute the out-of-subspace
    // component of δ(D†D) v_i for each wanted eigenvector and add it
    // to the pool.  These directions point exactly where each eigenvector
    // is moving under the gauge perturbation.
    //
    // Cost: n_ev × (~1.25 D†D) — much cheaper than Chebyshev probes
    // and precisely targeted rather than random.
    //
    // Call BEFORE gauge update + force_update, so that δD reflects
    // the upcoming perturbation.
    //
    // apply_deltaD:     v ↦ δD v   (sparse, ~1/4 D cost)
    // apply_deltaD_dag: v ↦ δD† v  (sparse, ~1/4 D cost)
    // apply_D_dag:      v ↦ D† v   (full D cost)
    // apply_D:          v ↦ D v    (full D cost, for Dpool of new vectors)
    void perturbation_extend(
        const std::function<void(const Vec&, Vec&)>& apply_deltaD,
        const std::function<void(const Vec&, Vec&)>& apply_deltaD_dag,
        const std::function<void(const Vec&, Vec&)>& apply_D_dag,
        const std::function<void(const Vec&, Vec&)>& apply_D);

    // Compute max residual ||Av - λv|| / ||Av|| for first n_ev vectors.
    // Cost: n_ev D†D applications.
    double max_residual(const OpApply& A) const;

    int pool_used() const { return (int)pool.size(); }
};
