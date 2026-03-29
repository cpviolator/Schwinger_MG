#include "multigrid.h"

// =====================================================================
//  MULTIGRID V-CYCLE (one-level)
// =====================================================================
// One-level: smooth -> restrict residual -> coarse solve -> prolong -> correct -> smooth
Vec mg_vcycle(const DiracOp& D, Prolongator& P, CoarseOp& Ac,
              const Vec& b, int pre_smooth, int post_smooth) {
    int n = D.lat.ndof;

    // pre-smooth
    Vec x = zeros(n);
    mr_smooth(D, x, b, pre_smooth);

    // compute residual r = b - Ax
    Vec Ax(n);
    D.apply_DdagD(x, Ax);
    Vec r(n);
    #pragma omp parallel for schedule(static) if(n > OMP_MIN_SIZE)
    for (int i = 0; i < n; i++) r[i] = b[i] - Ax[i];

    // restrict
    Vec rc = P.restrict_vec(r);

    // coarse solve
    Vec ec = Ac.solve(rc);

    // prolong and correct
    Vec e = P.prolong(ec);
    #pragma omp parallel for schedule(static) if(n > OMP_MIN_SIZE)
    for (int i = 0; i < n; i++) x[i] += e[i];

    // post-smooth
    mr_smooth(D, x, b, post_smooth);

    return x;
}

// =====================================================================
//  RECURSIVE MULTIGRID CYCLE (V-cycle or W-cycle)
// =====================================================================
Vec mg_cycle(std::vector<MGLevel>& levels, int lev, const Vec& b,
             bool w_cycle) {
    int n = levels[lev].dim;
    int last = (int)levels.size() - 1;

    // Base case: coarsest level -- direct or iterative solve
    if (lev == last) {
        if (levels[lev].coarse_solve)
            return levels[lev].coarse_solve(b);
        return levels[lev].Ac.solve(b);
    }

    // Pre-smooth
    Vec x = zeros(n);
    if (levels[lev].richardson_omega > 0)
        richardson_smooth_op(levels[lev].op, x, b,
                             levels[lev].pre_smooth, levels[lev].richardson_omega);
    else
        mr_smooth_op(levels[lev].op, x, b, levels[lev].pre_smooth);

    // Compute residual r = b - A*x
    Vec Ax(n);
    levels[lev].op(x, Ax);
    Vec r(n);
    #pragma omp parallel for schedule(static) if(n > OMP_MIN_SIZE)
    for (int i = 0; i < n; i++) r[i] = b[i] - Ax[i];

    // Restrict residual to next coarser level
    Vec rc = levels[lev].restrict_fn(r);

    // Recursive coarse solve (first call)
    Vec ec = mg_cycle(levels, lev + 1, rc, w_cycle);

    // W-cycle: second recursive call on the coarse residual
    if (w_cycle && lev + 1 < last) {
        int nc = (int)rc.size();
        Vec Ac_ec(nc);
        levels[lev + 1].op(ec, Ac_ec);
        Vec rc2(nc);
        #pragma omp parallel for schedule(static) if(nc > OMP_MIN_SIZE)
        for (int i = 0; i < nc; i++) rc2[i] = rc[i] - Ac_ec[i];
        Vec ec2 = mg_cycle(levels, lev + 1, rc2, w_cycle);
        #pragma omp parallel for schedule(static) if(nc > OMP_MIN_SIZE)
        for (int i = 0; i < nc; i++) ec[i] += ec2[i];
    }

    // Prolong and correct
    Vec e = levels[lev].prolong_fn(ec);
    #pragma omp parallel for schedule(static) if(n > OMP_MIN_SIZE)
    for (int i = 0; i < n; i++) x[i] += e[i];

    // Post-smooth
    if (levels[lev].richardson_omega > 0)
        richardson_smooth_op(levels[lev].op, x, b,
                             levels[lev].post_smooth, levels[lev].richardson_omega);
    else
        mr_smooth_op(levels[lev].op, x, b, levels[lev].post_smooth);

    return x;
}

// =====================================================================
//  MGHierarchy methods
// =====================================================================

// Update coarsest-level deflation vectors via LOBPCG.
// The coarsest operator is small (dense), so this is very cheap:
// no MG preconditioner needed, just dense matvec.
void MGHierarchy::update_coarsest_deflation(int n_defl, int lobpcg_iters) {
    int last = (int)levels.size() - 1;
    if (last < 0) return;
    auto& Ac = levels[last].Ac;
    int dim = Ac.dim;
    if (dim <= 0 || n_defl <= 0) return;

    // Operator for the coarsest level
    OpApply coarse_op = [&Ac](const Vec& s, Vec& d) { d = Ac.apply(s); };

    // No preconditioner at coarsest level (it's already small/dense)
    auto identity_precond = [](const Vec& v) { return v; };

    // Warm start from previous deflation vectors, or cold start
    std::vector<Vec> X0(n_defl);
    if ((int)levels[last].defl_vecs.size() >= n_defl) {
        for (int i = 0; i < n_defl; i++)
            X0[i] = levels[last].defl_vecs[i];
    } else {
        // Cold start: random vectors (avoid expensive Jacobi on large Ac)
        std::mt19937 defl_rng(42);
        for (int i = 0; i < n_defl; i++)
            X0[i] = random_vec(dim, defl_rng);
    }

    auto res = lobpcg_update(coarse_op, dim, n_defl, X0,
                              identity_precond, lobpcg_iters, 1e-10);

    levels[last].defl_vecs = std::move(res.eigvecs);
    levels[last].defl_vals = std::move(res.eigvals);
}

// Make the preconditioner symmetric by switching to Richardson smoothing
// with a fixed omega = damping / lambda_max at each level.
// This makes the MG cycle a linear, self-adjoint operator suitable for CG.
void MGHierarchy::set_symmetric(double damping) {
    for (int l = 0; l < (int)levels.size() - 1; l++) {
        // Estimate lambda_max at this level
        double lmax = estimate_lambda_max(levels[l].op, levels[l].dim, 30);
        levels[l].richardson_omega = damping / lmax;
    }
}

// Convenience: call the cycle
Vec MGHierarchy::precondition(const Vec& b) {
    return mg_cycle(levels, 0, b, w_cycle);
}

// Cascade-rebuild coarse operators at levels 1..n_levels-1.
// Assumes level-0 Ac has already been rebuilt for the current gauge/prolongator.
// This re-does the Galerkin projections P†AP at each deeper level using the
// EXISTING prolongators (no new near-null vectors needed).
void MGHierarchy::rebuild_deeper_levels() {
    int n_levels = (int)levels.size();
    if (n_levels <= 2) return;  // only 2 levels: nothing deeper to rebuild

    for (int l = 0; l < (int)coarse_prolongators.size(); l++) {
        // The operator at level l+1 is already referencing the coarse op
        // from level l (via captured reference in the lambda). Since we
        // rebuilt that coarse op, the level-(l+1) operator is now current.
        // But we need to rebuild the coarse operator at level l+1:
        //   Ac_{l+1} = P_{l+1}† A_{l+1} P_{l+1}
        auto& Pc = coarse_prolongators[l];
        int fine_dim = levels[l + 1].dim;
        intermediate_Ac[l].build_generic(levels[l + 1].op, Pc,
                                          fine_dim, Pc.coarse_dim);
        // Also update the Ac stored in the level struct
        levels[l + 1].Ac = intermediate_Ac[l];
    }

    // Update coarsest-level Ac (used for direct solve)
    levels[n_levels - 1].Ac = intermediate_Ac.back();

    // If sparse coarse is in use, rebuild it too
    if (use_sparse_coarse && !geo_prolongators.empty()) {
        auto& P0 = geo_prolongators[0];
        int fine_dim = levels[0].dim;
        sparse_Ac.build(P0, levels[0].op, fine_dim);
        // Re-run TRLM with warm start from previous deflation vectors
        int n_defl = (int)sparse_Ac.defl_vecs.size();
        if (n_defl > 0)
            sparse_Ac.setup_deflation(n_defl);
    }
}

// ---------------------------------------------------------------
// Refresh prolongator via Rayleigh-Ritz of null vectors
// ---------------------------------------------------------------
RREvolveResult MGHierarchy::refresh_prolongator_rr(const DiracOp& D_new) {
    int k = (int)null_vecs_l0.size();
    int ndof = D_new.lat.ndof;

    // RR: project D†D onto null vector subspace, diagonalise, rotate
    OpApply A_new = [&D_new](const Vec& s, Vec& d) { D_new.apply_DdagD(s, d); };
    auto rr = rr_evolve(A_new, null_vecs_l0, ndof);

    // Update null vectors with rotated versions
    null_vecs_l0 = std::move(rr.eigvecs);

    // Rebuild prolongator from rotated null vectors
    auto& P = geo_prolongators[0];
    P.build_from_vectors(null_vecs_l0);

    // Rebuild level-0 coarse operator: Ac = P†(D†D)P
    levels[0].Ac.build(D_new, P);

    // Cascade to deeper levels
    rebuild_deeper_levels();

    return rr;
}

// ---------------------------------------------------------------
// Refresh prolongator using a predicted rotation (no RR)
// ---------------------------------------------------------------
void MGHierarchy::refresh_prolongator_forecast(const DiracOp& D_new,
                                                const std::vector<Vec>& R_pred) {
    int ndof = D_new.lat.ndof;

    // Rotate null vectors by predicted rotation
    apply_rotation(null_vecs_l0, R_pred, ndof);

    // Rebuild prolongator from rotated null vectors
    auto& P = geo_prolongators[0];
    P.build_from_vectors(null_vecs_l0);

    // Rebuild level-0 coarse operator: Ac = P†(D†D)P
    levels[0].Ac.build(D_new, P);

    // Cascade to deeper levels
    rebuild_deeper_levels();
}

// ---------------------------------------------------------------
// Initialise Dv cache: D*v for each null vector
// ---------------------------------------------------------------
void MGHierarchy::init_Dv_cache(const DiracOp& D) {
    int k = (int)null_vecs_l0.size();
    int ndof = D.lat.ndof;
    Dv_l0.resize(k);
    null_evals_l0.resize(k);
    for (int i = 0; i < k; i++) {
        Dv_l0[i].resize(ndof);
        D.apply(null_vecs_l0[i], Dv_l0[i]);
        // Rayleigh quotient: v†(D†D)v = ||Dv||²
        null_evals_l0[i] = std::real(dot(Dv_l0[i], Dv_l0[i]));
    }
}

// ---------------------------------------------------------------
// Perturbation-based null space evolution (Strategy C)
// Uses force_evolve: delta_D from gauge update (momentum pi, step dt)
// Zero full matvecs — only k sparse delta_D applications
// ---------------------------------------------------------------
void MGHierarchy::refresh_prolongator_perturbation(
    const DiracOp& D_new,
    const std::array<RVec, 2>& pi, double dt)
{
    int k = (int)null_vecs_l0.size();
    int ndof = D_new.lat.ndof;

    // delta_D = D_new - D_old (from gauge update exp(i*dt*pi))
    auto apply_dD = [&](const Vec& src, Vec& dst) {
        D_new.apply_delta_D(src, dst, pi, dt);
    };
    // delta_D† = gamma5 * delta_D(-pi*dt) * gamma5
    // For the perturbation formula we need delta_D†, but force_evolve
    // doesn't actually use it separately — it computes dot(dDv_i, Dv_j)
    // which only needs delta_D applied forward. So pass a dummy.
    auto apply_dD_dag = apply_dD;  // not used by force_evolve in practice

    auto result = force_evolve(null_vecs_l0, null_evals_l0, Dv_l0,
                                apply_dD, apply_dD_dag, ndof);

    null_vecs_l0 = std::move(result.eigvecs);
    null_evals_l0 = std::move(result.eigvals);
    Dv_l0 = std::move(result.Dv);

    // Rebuild prolongator from rotated null vectors
    auto& P = geo_prolongators[0];
    P.build_from_vectors(null_vecs_l0);

    // Rebuild level-0 coarse operator: Ac = P†(D†D)P
    levels[0].Ac.build(D_new, P);

    // Cascade to deeper levels
    rebuild_deeper_levels();
}

// Prolong a coarsest-level vector all the way to the fine grid
// by chaining prolong_fn calls from coarsest to finest.
Vec MGHierarchy::prolong_to_fine(const Vec& v_coarse) const {
    int last = (int)levels.size() - 1;
    Vec v = v_coarse;
    for (int l = last - 1; l >= 0; l--) {
        v = levels[l].prolong_fn(v);
    }
    return v;
}

// Build fine-grid deflation space via coarse->prolong->refine pipeline:
//   1. Compute smallest eigenpairs of Ac via LOBPCG (cheap, O(coarse_dim^3))
//   2. Prolong through full MG hierarchy to fine grid
//   3. Refine with a few MG-preconditioned LOBPCG iterations on D†D
// The coarse eigenvectors provide a warm start that's much better than
// random, so only a few fine-grid refinement steps are needed.
//
// fine_op: fine-grid operator (D†D)
// fine_dim: fine-grid dimension
// n_refine: number of fine-grid LOBPCG refinement iterations (0 = skip)
// warm_X_coarse: optional warm-start coarse eigenvectors from previous call
// warm_X_fine: optional warm-start fine eigenvectors from previous call
std::pair<std::vector<Vec>, std::vector<double>>
MGHierarchy::build_fine_deflation(int k, const OpApply& fine_op, int fine_dim,
                                   int n_refine,
                                   std::vector<Vec>* warm_X_coarse,
                                   std::vector<Vec>* warm_X_fine) {
    int last = (int)levels.size() - 1;
    if (last < 0) return {{}, {}};

    auto& Ac = levels[last].Ac;
    if (Ac.dim <= 0) return {{}, {}};

    int nk = std::min(k, Ac.dim);

    // Step 1: Coarse eigensolve via direct Lanczos on Ac
    // (Ac is small and dense -- Lanczos gives machine-precision eigenvectors)
    auto coarse_evecs = Ac.smallest_eigenvectors(nk);
    if (warm_X_coarse) *warm_X_coarse = coarse_evecs;

    // Step 2: Prolong to fine grid + orthonormalise (double MGS)
    std::vector<Vec> fine_vecs(nk);
    for (int i = 0; i < nk; i++)
        fine_vecs[i] = prolong_to_fine(coarse_evecs[i]);

    for (int i = 0; i < nk; i++) {
        for (int pass = 0; pass < 2; pass++)
            for (int j = 0; j < i; j++) {
                cx proj = dot(fine_vecs[j], fine_vecs[i]);
                axpy(-proj, fine_vecs[j], fine_vecs[i]);
            }
        double nv = norm(fine_vecs[i]);
        if (nv > 1e-14) scale(fine_vecs[i], cx(1.0/nv));
    }

    // Step 3: Refine via Chebyshev-filtered subspace iteration
    // Chebyshev filter amplifies small-eigenvalue components without
    // a preconditioner; only uses matvecs. Much more robust than
    // inverse iteration with MG preconditioner.
    if (n_refine > 0) {
        // Use previous fine vectors as warm start if available
        if (warm_X_fine && (int)warm_X_fine->size() >= nk) {
            fine_vecs = *warm_X_fine;
        }

        auto cheb_res = chebyshev_subspace_iteration(
            fine_op, fine_dim, nk, fine_vecs,
            /*poly_deg=*/20, /*max_iter=*/n_refine, /*tol=*/1e-8);
        fine_vecs = std::move(cheb_res.eigvecs);

        if (warm_X_fine) *warm_X_fine = fine_vecs;
    }

    // Compute fine-grid Rayleigh quotients
    std::vector<double> fine_evals(nk);
    for (int i = 0; i < nk; i++) {
        Vec Av(fine_dim);
        fine_op(fine_vecs[i], Av);
        fine_evals[i] = std::real(dot(fine_vecs[i], Av));
    }

    return {std::move(fine_vecs), std::move(fine_evals)};
}

// =====================================================================
//  MG SETUP: INITIAL NEAR-NULL SPACE CONSTRUCTION
// =====================================================================
// Use inverse iteration to find near-null vectors of D†D.
// Each outer step solves D†D x_new ~ x_old (approximately, via MR),
// which amplifies the smallest-eigenvalue components by ~1/lambda_min
// per iteration.  This is the expensive step done once at the start.
std::vector<Vec> compute_near_null_space(const DiracOp& D, int k,
                                         int outer_iters, std::mt19937& rng,
                                         const std::vector<Vec>* warm_start)
{
    int n = D.lat.ndof;
    std::vector<Vec> null_vecs(k);
    const int inner_mr = 5;  // MR steps per approximate inverse solve

    for (int i = 0; i < k; i++) {
        if (warm_start && i < (int)warm_start->size())
            null_vecs[i] = (*warm_start)[i];
        else
            null_vecs[i] = random_vec(n, rng);
        double nv = norm(null_vecs[i]);
        scale(null_vecs[i], cx(1.0/nv));

        // Inverse iteration: solve D†D x_new ~ x_old repeatedly
        for (int iter = 0; iter < outer_iters; iter++) {
            Vec rhs_inv = null_vecs[i];
            Vec x_new = zeros(n);
            mr_smooth(D, x_new, rhs_inv, inner_mr);

            // Orthogonalise against previously converged vectors
            for (int j = 0; j < i; j++) {
                cx proj = dot(null_vecs[j], x_new);
                axpy(-proj, null_vecs[j], x_new);
            }

            nv = norm(x_new);
            if (nv > 1e-14) {
                scale(x_new, cx(1.0/nv));
                null_vecs[i] = x_new;
            } else {
                break;
            }
        }
    }

    // Final orthogonalisation
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < i; j++) {
            cx proj = dot(null_vecs[j], null_vecs[i]);
            axpy(-proj, null_vecs[j], null_vecs[i]);
        }
        double nv = norm(null_vecs[i]);
        if (nv > 1e-14) scale(null_vecs[i], cx(1.0/nv));
    }

    return null_vecs;
}

// =====================================================================
//  GENERIC NEAR-NULL SPACE (for coarse levels)
// =====================================================================
std::vector<Vec> compute_near_null_space_generic(
    const OpApply& A, int dim, int k, int outer_iters, std::mt19937& rng,
    const std::vector<Vec>* warm_start)
{
    std::vector<Vec> null_vecs(k);
    const int inner_mr = 5;

    for (int i = 0; i < k; i++) {
        if (warm_start && i < (int)warm_start->size())
            null_vecs[i] = (*warm_start)[i];
        else
            null_vecs[i] = random_vec(dim, rng);
        double nv = norm(null_vecs[i]);
        scale(null_vecs[i], cx(1.0/nv));

        for (int iter = 0; iter < outer_iters; iter++) {
            Vec rhs_inv = null_vecs[i];
            Vec x_new = zeros(dim);
            mr_smooth_op(A, x_new, rhs_inv, inner_mr);
            for (int j = 0; j < i; j++) {
                cx proj = dot(null_vecs[j], x_new);
                axpy(-proj, null_vecs[j], x_new);
            }
            nv = norm(x_new);
            if (nv > 1e-14) {
                scale(x_new, cx(1.0/nv));
                null_vecs[i] = x_new;
            } else {
                break;
            }
        }
    }

    for (int i = 0; i < k; i++) {
        for (int j = 0; j < i; j++) {
            cx proj = dot(null_vecs[j], null_vecs[i]);
            axpy(-proj, null_vecs[j], null_vecs[i]);
        }
        double nv = norm(null_vecs[i]);
        if (nv > 1e-14) scale(null_vecs[i], cx(1.0/nv));
    }
    return null_vecs;
}

// =====================================================================
//  MULTI-LEVEL MG SETUP
// =====================================================================
// Build an N-level multigrid hierarchy.
// Level 0 uses the geometric Prolongator; deeper levels use CoarseProlongator
// with algebraic (linear) blocking.
MGHierarchy build_mg_hierarchy(
    const DiracOp& D, int n_levels,
    int block_size, int k_null, int coarse_block_agg,
    int null_iters, std::mt19937& rng,
    bool w_cycle,
    int pre_smooth, int post_smooth,
    bool verbose,
    const std::vector<Vec>* warm_start)
{
    MGHierarchy mg;
    mg.w_cycle = w_cycle;
    mg.levels.resize(n_levels);

    // --- Level 0: fine grid ---
    OpApply fine_op = [&D](const Vec& s, Vec& d){ D.apply_DdagD(s, d); };
    mg.levels[0].op = fine_op;
    mg.levels[0].dim = D.lat.ndof;
    mg.levels[0].pre_smooth = pre_smooth;
    mg.levels[0].post_smooth = post_smooth;

    // Compute near-null vectors for level 0
    auto null_vecs = compute_near_null_space(D, k_null, null_iters, rng, warm_start);
    mg.null_vecs_l0 = null_vecs;  // store for warm-start rebuilds

    // Build geometric prolongator
    mg.geo_prolongators.emplace_back(D.lat, block_size, block_size, k_null);
    mg.geo_prolongators.back().build_from_vectors(null_vecs);

    auto& P0 = mg.geo_prolongators.back();
    mg.levels[0].restrict_fn = [&P0](const Vec& v){ return P0.restrict_vec(v); };
    mg.levels[0].prolong_fn  = [&P0](const Vec& v){ return P0.prolong(v); };
    mg.levels[0].Ac.build(D, P0);

    int prev_dim = P0.coarse_dim;

    if (verbose) std::cout << "  Level 0: dim=" << D.lat.ndof << " -> coarse_dim=" << prev_dim << "\n";

    // --- Intermediate levels 1 .. n_levels-2 ---
    for (int l = 1; l < n_levels - 1; l++) {
        // Operator at this level = coarse op from level above
        CoarseOp& Ac_above = (l == 1) ? mg.levels[0].Ac
                                       : mg.intermediate_Ac.back();
        OpApply level_op = [&Ac_above](const Vec& s, Vec& d){ d = Ac_above.apply(s); };

        mg.levels[l].op = level_op;
        mg.levels[l].dim = prev_dim;
        mg.levels[l].pre_smooth = pre_smooth;
        mg.levels[l].post_smooth = post_smooth;

        // Block size for algebraic prolongator: aggregate coarse_block_agg
        // consecutive coarse DOFs into one block
        int blk = coarse_block_agg;
        // Ensure divisibility
        while (prev_dim % blk != 0 && blk > 1) blk--;
        if (blk < 1) blk = 1;

        // Compute near-null vectors at this level
        auto coarse_null = compute_near_null_space_generic(
            level_op, prev_dim, k_null, null_iters, rng);

        // Build algebraic prolongator
        mg.coarse_prolongators.emplace_back(prev_dim, blk, k_null);
        mg.coarse_prolongators.back().build_from_vectors(coarse_null);

        auto& Pc = mg.coarse_prolongators.back();
        mg.levels[l].restrict_fn = [&Pc](const Vec& v){ return Pc.restrict_vec(v); };
        mg.levels[l].prolong_fn  = [&Pc](const Vec& v){ return Pc.prolong(v); };

        // Build coarse operator via Galerkin: P†AP
        mg.intermediate_Ac.emplace_back();
        mg.intermediate_Ac.back().build_generic(level_op, Pc, prev_dim, Pc.coarse_dim);
        mg.levels[l].Ac = mg.intermediate_Ac.back();

        if (verbose) std::cout << "  Level " << l << ": dim=" << prev_dim
                  << " block=" << blk << " -> coarse_dim=" << Pc.coarse_dim << "\n";

        prev_dim = Pc.coarse_dim;
    }

    // --- Coarsest level: direct solve ---
    if (n_levels >= 2) {
        CoarseOp& Ac_above = (n_levels == 2) ? mg.levels[0].Ac
                                               : mg.intermediate_Ac.back();
        OpApply coarsest_op = [&Ac_above](const Vec& s, Vec& d){ d = Ac_above.apply(s); };
        mg.levels[n_levels - 1].op = coarsest_op;
        mg.levels[n_levels - 1].dim = prev_dim;
        mg.levels[n_levels - 1].Ac = Ac_above;
        mg.levels[n_levels - 1].pre_smooth = 0;
        mg.levels[n_levels - 1].post_smooth = 0;
    }

    return mg;
}

// Warm-started hierarchy rebuild: uses previous null vectors as starting point
// with fewer iterations, dramatically reducing setup cost.
MGHierarchy build_mg_hierarchy_warm(
    const DiracOp& D, int n_levels,
    int block_size, int k_null, int coarse_block_agg,
    int null_iters, std::mt19937& rng,
    const std::vector<Vec>& warm_null_vecs,
    bool w_cycle,
    int pre_smooth, int post_smooth,
    bool verbose)
{
    return build_mg_hierarchy(D, n_levels, block_size, k_null, coarse_block_agg,
                              null_iters, rng, w_cycle, pre_smooth, post_smooth,
                              verbose, &warm_null_vecs);
}

// =====================================================================
//  COARSE-EIGENVECTOR PROLONGATOR REFRESH
// =====================================================================
// Compute k smallest eigenvectors of the coarse operator Ac, prolong
// to the fine grid, and apply a few MR smoothing steps to adapt them
// to the current gauge field. These become new near-null vectors.
// Cost: k x (1 prolong + smooth_iters MR) ~ k x (smooth_iters+1) matvecs.
std::vector<Vec> refresh_from_coarse_eigvecs(
    const DiracOp& D, const Prolongator& P, const CoarseOp& Ac,
    int k, int smooth_iters)
{
    int n = D.lat.ndof;

    // Get k smallest eigenvectors of the coarse operator
    auto coarse_evecs = Ac.smallest_eigenvectors(k);

    std::vector<Vec> null_vecs(k);
    for (int i = 0; i < k; i++) {
        // Prolong coarse eigenvector to fine grid
        null_vecs[i] = P.prolong(coarse_evecs[i]);

        // Smooth with MR on D†D to adapt to current gauge field
        Vec rhs = null_vecs[i];
        Vec x = zeros(n);
        mr_smooth(D, x, rhs, smooth_iters);
        double nv = norm(x);
        if (nv > 1e-14) {
            scale(x, cx(1.0 / nv));
            null_vecs[i] = std::move(x);
        }

        // Orthogonalise against previous vectors
        for (int j = 0; j < i; j++) {
            cx proj = dot(null_vecs[j], null_vecs[i]);
            axpy(-proj, null_vecs[j], null_vecs[i]);
        }
        nv = norm(null_vecs[i]);
        if (nv > 1e-14) scale(null_vecs[i], cx(1.0 / nv));
    }

    return null_vecs;
}

// =====================================================================
//  SPARSE COARSE OPERATOR SETUP
// =====================================================================
// Build sparse stencil-based coarse operator from the level-0 geometric
// prolongator, run TRLM to find deflation eigenvectors, and wire up
// the coarsest-level solve to use deflated CG.
void MGHierarchy::setup_sparse_coarse(const OpApply& fine_op, int fine_dim,
                                       int n_defl, double cg_tol,
                                       int max_cg_iter) {
    if (geo_prolongators.empty()) return;
    auto& P0 = geo_prolongators[0];

    // Build sparse coarse op from the same Galerkin projection
    sparse_Ac.build(P0, fine_op, fine_dim);

    // Run TRLM to find low modes for deflation
    std::cout << "  Sparse coarse: dim=" << sparse_Ac.dim
              << " (nbx=" << sparse_Ac.nbx << " nby=" << sparse_Ac.nby
              << " k_vec=" << sparse_Ac.k_vec << ")\n";
    std::cout << "  Running TRLM for " << n_defl << " deflation vectors...\n";

    sparse_Ac.setup_deflation(n_defl);

    std::cout << "  TRLM eigenvalues:";
    for (int i = 0; i < std::min(n_defl, (int)sparse_Ac.defl_vals.size()); i++)
        std::cout << " " << sparse_Ac.defl_vals[i];
    std::cout << "\n";

    // Wire up the coarsest level to use sparse deflated CG
    int last = (int)levels.size() - 1;
    if (last < 0) return;

    // Capture by pointer (sparse_Ac lives in MGHierarchy, same lifetime)
    auto* sac = &sparse_Ac;
    int miter = max_cg_iter;
    double ctol = cg_tol;
    levels[last].coarse_solve = [sac, miter, ctol](const Vec& b) -> Vec {
        return sac->solve(b, miter, ctol);
    };
    use_sparse_coarse = true;

    std::cout << "  Coarsest-level solve: deflated CG (sparse, "
              << n_defl << " deflation vectors)\n";
}
