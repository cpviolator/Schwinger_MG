#pragma once
#include "types.h"
#include "linalg.h"
#include "lattice.h"
#include "dirac.h"
#include "gauge.h"
#include "prolongator.h"
#include "coarse_op.h"
#include "eigensolver.h"
#include "smoother.h"
#include <vector>
#include <functional>
#include <random>
#include <iostream>
#include <omp.h>

struct MGLevel {
    OpApply op;                                   // operator at this level
    int dim;                                      // vector dimension
    std::function<Vec(const Vec&)> restrict_fn;   // fine -> coarse
    std::function<Vec(const Vec&)> prolong_fn;    // coarse -> fine
    CoarseOp Ac;                                  // coarse operator (Galerkin)
    int pre_smooth;
    int post_smooth;
    double richardson_omega = 0.0;  // 0 = use MR (default), >0 = Richardson
    // Coarsest-level deflation: eigenvectors and eigenvalues of Ac
    std::vector<Vec> defl_vecs;
    std::vector<double> defl_vals;
    // Coarse solve function: defaults to dense LU, can be overridden
    // with sparse deflated CG when SparseCoarseOp is available.
    std::function<Vec(const Vec&)> coarse_solve;
};

Vec mg_vcycle(const DiracOp& D, Prolongator& P, CoarseOp& Ac,
              const Vec& b, int pre_smooth, int post_smooth);

Vec mg_cycle(std::vector<MGLevel>& levels, int lev, const Vec& b,
             bool w_cycle);

struct MGHierarchy {
    std::vector<MGLevel> levels;
    // Storage for prolongators (must outlive the MGLevel references)
    std::vector<Prolongator> geo_prolongators;          // level 0
    std::vector<CoarseProlongator> coarse_prolongators;  // levels 1+
    // Storage for coarse operators at intermediate levels
    std::vector<CoarseOp> intermediate_Ac;
    bool w_cycle;
    // Store level-0 null vectors for warm-start rebuilds
    std::vector<Vec> null_vecs_l0;
    // Cached D*null_vec for perturbation-based evolution (force_evolve)
    std::vector<Vec> Dv_l0;
    std::vector<double> null_evals_l0; // Rayleigh quotients of null vecs
    // Sparse coarse operator (optional, for large coarse grids)
    SparseCoarseOp sparse_Ac;
    bool use_sparse_coarse = false;

    void update_coarsest_deflation(int n_defl, int lobpcg_iters = 3);
    void set_symmetric(double damping = 0.8);
    Vec precondition(const Vec& b);
    void rebuild_deeper_levels();
    void rebind_prolongator_lambdas();  // rebind restrict/prolong after P changes
    Vec prolong_to_fine(const Vec& v_coarse) const;
    std::pair<std::vector<Vec>, std::vector<double>>
    build_fine_deflation(int k, const OpApply& fine_op, int fine_dim,
                         int n_refine = 5,
                         std::vector<Vec>* warm_X_coarse = nullptr,
                         std::vector<Vec>* warm_X_fine = nullptr);

    // Setup sparse coarse operator at level 0 with TRLM deflation.
    // Replaces the coarsest-level dense LU solve with deflated CG
    // on the sparse stencil operator.
    //   n_defl: number of deflation eigenvectors (from TRLM)
    //   cg_tol: CG convergence tolerance for coarse solves
    //   max_cg_iter: max CG iterations per coarse solve
    void setup_sparse_coarse(const OpApply& fine_op, int fine_dim,
                              int n_defl = 16, double cg_tol = 1e-12,
                              int max_cg_iter = 200);

    // Refresh prolongator via RR of null vectors against new D†D.
    // Rotates null_vecs_l0, rebuilds P, Galerkin coarse ops, cascades.
    // Returns RR result (includes rotation matrix for generator extraction).
    RREvolveResult refresh_prolongator_rr(const DiracOp& D_new);

    // Refresh prolongator using a predicted rotation (no RR, no fine matvecs for rotation).
    // Rotates null_vecs_l0 by R_pred, rebuilds P, Galerkin coarse ops, cascades.
    void refresh_prolongator_forecast(const DiracOp& D_new,
                                      const std::vector<Vec>& R_pred);

    // Perturbation-based null space evolution (Strategy C).
    // Uses delta_D from gauge update to rotate null vecs via force_evolve.
    // Zero full matvecs — only k sparse delta_D applications.
    // Rebuilds P + Galerkin from rotated null vecs.
    void refresh_prolongator_perturbation(
        const DiracOp& D_new,
        const std::array<RVec, 2>& pi, double dt);

    // Initialise Dv cache: compute D*v for each null vector
    void init_Dv_cache(const DiracOp& D);
};

std::vector<Vec> compute_near_null_space(const DiracOp& D, int k,
                                         int outer_iters, std::mt19937& rng,
                                         const std::vector<Vec>* warm_start = nullptr);

std::vector<Vec> compute_near_null_space_generic(
    const OpApply& A, int dim, int k, int outer_iters, std::mt19937& rng,
    const std::vector<Vec>* warm_start = nullptr);

MGHierarchy build_mg_hierarchy(
    const DiracOp& D, int n_levels,
    int block_size, int k_null, int coarse_block_agg,
    int null_iters, std::mt19937& rng,
    bool w_cycle = true,
    int pre_smooth = 3, int post_smooth = 3,
    bool verbose = true,
    const std::vector<Vec>* warm_start = nullptr);

MGHierarchy build_mg_hierarchy_warm(
    const DiracOp& D, int n_levels,
    int block_size, int k_null, int coarse_block_agg,
    int null_iters, std::mt19937& rng,
    const std::vector<Vec>& warm_null_vecs,
    bool w_cycle = true,
    int pre_smooth = 3, int post_smooth = 3,
    bool verbose = false);

std::vector<Vec> refresh_from_coarse_eigvecs(
    const DiracOp& D, const Prolongator& P, const CoarseOp& Ac,
    int k, int smooth_iters = 5);
