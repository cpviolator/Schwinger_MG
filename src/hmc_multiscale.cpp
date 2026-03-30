#include "hmc.h"
#include "eigensolver.h"
#include "multigrid.h"
#include <cmath>
#include <chrono>
#include <iostream>
#include <iomanip>

void lowmode_fermion_force(const DiracOp& D,
    const DeflationState& defl,
    const Vec& phi,
    std::array<RVec, 2>& force)
{
    // Project phi onto deflation subspace to get low-mode solution
    Vec X_low = defl.deflated_initial_guess(phi);
    fermion_force(D, X_low, force);
}

// ---------------------------------------------------------------
//  Evolve deflation state (between trajectories)
// ---------------------------------------------------------------
void evolve_deflation_state(DeflationState& defl,
    const DiracOp& D_new, bool fresh_trlm)
{
    if (!defl.valid || defl.eigvecs.empty()) return;
    int nd = (int)defl.eigvecs.size();

    if (fresh_trlm) {
        OpApply A = [&D_new](const Vec& src, Vec& dst) {
            D_new.apply_DdagD(src, dst);
        };
        auto result = trlm_eigensolver(A, D_new.lat.ndof, nd,
                                        std::min(2*nd + 10, D_new.lat.ndof),
                                        100, 1e-10);
        defl.eigvecs = std::move(result.eigvecs);
        defl.eigvals = std::move(result.eigvals);
    } else {
        OpApply A = [&D_new](const Vec& src, Vec& dst) {
            D_new.apply_DdagD(src, dst);
        };
        auto rr = rr_evolve(A, defl.eigvecs, D_new.lat.ndof);
        defl.eigvecs = std::move(rr.eigvecs);
        defl.eigvals = std::move(rr.eigvals);
    }

    defl.update_cache(D_new);
}

// ---------------------------------------------------------------
//  Multi-timescale HMC trajectory (Sexton-Weingarten)
//  Inner = low-mode force (cheap, from eigenvectors)
//  Outer = gauge + high-mode fermion force (expensive, CG solve)
// ---------------------------------------------------------------
MultiScaleResult hmc_trajectory_multiscale(
    GaugeField& gauge, const Lattice& lat, double mass, double wilson_r,
    const MultiScaleParams& params, DeflationState& defl,
    std::mt19937& rng,
    const std::function<Vec(const Vec&)>* precond)
{
    using Clock = std::chrono::high_resolution_clock;
    using Dur = std::chrono::duration<double>;

    double dt_outer = params.tau / params.n_outer;
    double dt_inner = dt_outer / params.n_inner;
    double c_sw = params.c_sw;
    double mu_t = params.mu_t;
    int total_cg = 0;
    int lowmode_evals = 0;
    double highmode_time = 0, lowmode_time = 0;

    // Save gauge for reject
    GaugeField gauge_old(lat);
    gauge_old.U[0] = gauge.U[0];
    gauge_old.U[1] = gauge.U[1];

    // Generate momenta and pseudofermion
    MomentumField mom(lat);
    mom.randomise(rng);

    DiracOp D_init(lat, gauge, mass, wilson_r, c_sw, mu_t);
    Vec phi;
    generate_pseudofermion(D_init, rng, phi);

    // Update Dv cache for initial gauge
    defl.update_cache(D_init);

    // --- Compute initial Hamiltonian (exact, full CG) ---
    double KE_init = mom.kinetic_energy();
    double SG_init = gauge_action(gauge, params.beta);
    double SF_init;
    {
        OpApply A = [&D_init](const Vec& src, Vec& dst) { D_init.apply_DdagD(src, dst); };
        CGResult res;
        if (precond)
            res = cg_solve_precond(A, lat.ndof, phi, *precond, params.cg_maxiter, params.cg_tol);
        else
            res = cg_solve(A, lat.ndof, phi, params.cg_maxiter, params.cg_tol);
        SF_init = std::real(dot(phi, res.solution));
        total_cg += res.iterations;
    }
    double H_init = KE_init + SG_init + SF_init;

    // === OUTER HALF-KICK (slow forces: gauge + high-mode fermion) ===
    // High-mode = full fermion force - low-mode force
    {
        auto t0 = Clock::now();
        DiracOp D(lat, gauge, mass, wilson_r, c_sw, mu_t);
        std::array<RVec, 2> gf, ff_full, fl;
        gauge_force(gauge, params.beta, gf);

        OpApply A = [&D](const Vec& src, Vec& dst) { D.apply_DdagD(src, dst); };
        Vec x0 = defl.deflated_initial_guess(phi);
        CGResult res = cg_solve_x0(A, lat.ndof, phi, x0, params.cg_maxiter, params.cg_tol);
        total_cg += res.iterations;
        fermion_force(D, res.solution, ff_full);
        lowmode_fermion_force(D, defl, phi, fl);

        #pragma omp parallel for collapse(2) schedule(static)
        for (int mu = 0; mu < 2; mu++)
            for (int s = 0; s < lat.V; s++)
                mom.pi[mu][s] += 0.5 * dt_outer * (gf[mu][s] + ff_full[mu][s] - fl[mu][s]);
        highmode_time += Dur(Clock::now() - t0).count();
    }

    // === OUTER LOOP ===
    for (int o = 0; o < params.n_outer; o++) {

        // --- INNER HALF-KICK (fast: low-mode force) ---
        {
            auto t0 = Clock::now();
            DiracOp D(lat, gauge, mass, wilson_r, c_sw, mu_t);
            std::array<RVec, 2> fl;
            lowmode_fermion_force(D, defl, phi, fl);
            #pragma omp parallel for collapse(2) schedule(static)
            for (int mu = 0; mu < 2; mu++)
                for (int s = 0; s < lat.V; s++)
                    mom.pi[mu][s] += 0.5 * dt_inner * fl[mu][s];
            lowmode_evals++;
            lowmode_time += Dur(Clock::now() - t0).count();
        }

        // --- INNER LOOP ---
        for (int i = 0; i < params.n_inner; i++) {
            // Full gauge update
            #pragma omp parallel for collapse(2) schedule(static)
            for (int mu = 0; mu < 2; mu++)
                for (int s = 0; s < lat.V; s++)
                    gauge.U[mu][s] *= std::exp(cx(0, dt_inner * mom.pi[mu][s]));

            if (i < params.n_inner - 1) {
                // Full inner kick
                auto t0 = Clock::now();
                DiracOp D(lat, gauge, mass, wilson_r, c_sw, mu_t);
                std::array<RVec, 2> fl;
                lowmode_fermion_force(D, defl, phi, fl);
                #pragma omp parallel for collapse(2) schedule(static)
                for (int mu = 0; mu < 2; mu++)
                    for (int s = 0; s < lat.V; s++)
                        mom.pi[mu][s] += dt_inner * fl[mu][s];
                lowmode_evals++;
                lowmode_time += Dur(Clock::now() - t0).count();
            }
        }

        // --- FINAL INNER HALF-KICK ---
        {
            auto t0 = Clock::now();
            DiracOp D(lat, gauge, mass, wilson_r, c_sw, mu_t);
            std::array<RVec, 2> fl;
            lowmode_fermion_force(D, defl, phi, fl);
            #pragma omp parallel for collapse(2) schedule(static)
            for (int mu = 0; mu < 2; mu++)
                for (int s = 0; s < lat.V; s++)
                    mom.pi[mu][s] += 0.5 * dt_inner * fl[mu][s];
            lowmode_evals++;
            lowmode_time += Dur(Clock::now() - t0).count();
        }

        // --- OUTER KICK (slow: gauge + high-mode = full - low) ---
        {
            auto t0 = Clock::now();
            DiracOp D(lat, gauge, mass, wilson_r, c_sw, mu_t);
            std::array<RVec, 2> gf, ff_full, fl;
            gauge_force(gauge, params.beta, gf);

            OpApply A = [&D](const Vec& src, Vec& dst) { D.apply_DdagD(src, dst); };
            Vec x0 = defl.deflated_initial_guess(phi);
            CGResult res = cg_solve_x0(A, lat.ndof, phi, x0, params.cg_maxiter, params.cg_tol);
            total_cg += res.iterations;
            fermion_force(D, res.solution, ff_full);
            lowmode_fermion_force(D, defl, phi, fl);

            double kick = (o < params.n_outer - 1) ? dt_outer : 0.5 * dt_outer;
            #pragma omp parallel for collapse(2) schedule(static)
            for (int mu = 0; mu < 2; mu++)
                for (int s = 0; s < lat.V; s++)
                    mom.pi[mu][s] += kick * (gf[mu][s] + ff_full[mu][s] - fl[mu][s]);
            highmode_time += Dur(Clock::now() - t0).count();
        }
    }

    // --- Compute final Hamiltonian (exact, full CG) ---
    double KE_final = mom.kinetic_energy();
    double SG_final = gauge_action(gauge, params.beta);
    double SF_final;
    {
        DiracOp D(lat, gauge, mass, wilson_r, c_sw, mu_t);
        OpApply A = [&D](const Vec& src, Vec& dst) { D.apply_DdagD(src, dst); };
        CGResult res;
        if (precond)
            res = cg_solve_precond(A, lat.ndof, phi, *precond, params.cg_maxiter, params.cg_tol);
        else
            res = cg_solve(A, lat.ndof, phi, params.cg_maxiter, params.cg_tol);
        SF_final = std::real(dot(phi, res.solution));
        total_cg += res.iterations;
    }
    double H_final = KE_final + SG_final + SF_final;

    double dH = H_final - H_init;
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    bool accept = (dH < 0) || (uniform(rng) < std::exp(-dH));

    if (!accept) {
        gauge.U[0] = gauge_old.U[0];
        gauge.U[1] = gauge_old.U[1];
    }

    return {accept, dH, KE_final - KE_init, SG_final - SG_init, SF_final - SF_init,
            total_cg, lowmode_evals, highmode_time, lowmode_time};
}

// ---------------------------------------------------------------
//  MG-based multi-timescale HMC
//  Low-mode force via coarse-grid deflation (restrict-project-prolong)
// ---------------------------------------------------------------

void coarse_lowmode_force(const DiracOp& D,
    const CoarseDeflState& cdefl,
    const Prolongator& P,
    const Vec& phi,
    std::array<RVec, 2>& force,
    int smooth_iters)
{
    int n = D.lat.ndof;

    if (smooth_iters <= 0) {
        // Raw restrict-project-prolong (no smoothing)
        Vec X_low = cdefl.lowmode_solution(phi, P);
        fermion_force(D, X_low, force);
        return;
    }

    // Cheap MG-like cycle: pre-smooth → restrict residual → coarse deflation → prolong → post-smooth
    OpApply A = [&D](const Vec& s, Vec& d) { D.apply_DdagD(s, d); };

    // Pre-smooth: MR iterations on D†D x = phi
    Vec x = zeros(n);
    mr_smooth_op(A, x, phi, smooth_iters);

    // Compute residual r = phi - D†D x
    Vec Ax(n);
    A(x, Ax);
    Vec r(n);
    for (int i = 0; i < n; i++) r[i] = phi[i] - Ax[i];

    // Restrict residual to coarse, deflation-project, prolong correction
    Vec e_coarse = cdefl.lowmode_solution(r, P);

    // Apply correction
    for (int i = 0; i < n; i++) x[i] += e_coarse[i];

    // Post-smooth: MR on D†D x = phi starting from current x
    mr_smooth_op(A, x, phi, smooth_iters);

    fermion_force(D, x, force);
}

void evolve_coarse_deflation(CoarseDeflState& cdefl,
    const SparseCoarseOp& Ac_new,
    EigenForecastState* forecast)
{
    if (cdefl.eigvecs.empty()) return;
    int k = (int)cdefl.eigvecs.size();

    // Pre-rotate if forecast has enough history
    std::vector<Vec> R_pred;
    bool did_forecast = false;
    if (forecast && forecast->history_len >= 2) {
        R_pred = forecast_rotation(*forecast);
        apply_rotation(cdefl.eigvecs, R_pred, Ac_new.dim);
        did_forecast = true;
    }

    // RR on (possibly pre-rotated) eigenvectors
    OpApply op = Ac_new.as_op();
    auto rr = rr_evolve(op, cdefl.eigvecs, Ac_new.dim);

    // Extract generator from the rotation
    if (forecast) {
        std::vector<Vec> H_full;
        if (did_forecast) {
            // Full rotation = U_correction × R_pred
            std::vector<Vec> U_full;
            mat_mul_kk(rr.rotation, R_pred, U_full, k);
            extract_generator(U_full, k, H_full);

            // Print diagnostic: ||H_correction|| vs ||H_full||
            std::vector<Vec> H_corr;
            extract_generator(rr.rotation, k, H_corr);
            double norm_corr = frobenius_norm(H_corr, k);
            double norm_full = frobenius_norm(H_full, k);
            std::cout << "  [forecast] ||H||=" << std::scientific << std::setprecision(3) << norm_full
                      << "  ||H_corr||=" << norm_corr
                      << "  ratio=" << std::fixed << std::setprecision(4)
                      << (norm_full > 1e-30 ? norm_corr / norm_full : 0.0) << "\n";
        } else {
            // No forecast applied — U from RR is the full rotation
            extract_generator(rr.rotation, k, H_full);
            double norm_full = frobenius_norm(H_full, k);
            std::cout << "  [forecast] ||H||=" << std::scientific << std::setprecision(3) << norm_full
                      << "  (collecting history " << forecast->history_len + 1 << "/"
                      << EigenForecastState::max_history << ")\n";
        }

        // Store in history (push front, drop oldest if full)
        forecast->k = k;
        if (forecast->history_len < EigenForecastState::max_history) {
            forecast->H_history.insert(forecast->H_history.begin(), std::move(H_full));
            forecast->history_len++;
        } else {
            forecast->H_history.pop_back();
            forecast->H_history.insert(forecast->H_history.begin(), std::move(H_full));
        }
    }

    cdefl.eigvecs = std::move(rr.eigvecs);
    cdefl.eigvals = std::move(rr.eigvals);
}

MGMultiScaleResult hmc_trajectory_mg_multiscale(
    GaugeField& gauge, const Lattice& lat, double mass, double wilson_r,
    const MGMultiScaleParams& params,
    CoarseDeflState& cdefl,
    Prolongator& P,
    std::function<Vec(const Vec&)>& mg_precond,
    std::mt19937& rng,
    EigenForecastState* forecast,
    const std::function<void()>* pre_solve,
    MGHierarchy* mg_hierarchy)
{
    using Clock = std::chrono::high_resolution_clock;
    using Dur = std::chrono::duration<double>;

    double dt_outer = params.tau / params.n_outer;
    double c_sw = params.c_sw;
    double mu_t = params.mu_t;
    int total_cg = 0;
    int lowmode_evals = 0;
    double highmode_time = 0, lowmode_time = 0;

    // Save gauge for reject
    GaugeField gauge_old(lat);
    gauge_old.U[0] = gauge.U[0];
    gauge_old.U[1] = gauge.U[1];

    MomentumField mom(lat);
    mom.randomise(rng);

    DiracOp D_init(lat, gauge, mass, wilson_r, c_sw, mu_t);
    // Multi-timescale: e/o not supported (force splitting requires consistent action).
    // The e/o Schur complement action differs from D†D, making F_outer - F_inner
    // inconsistent when inner uses full-lattice phi and outer uses Schur phi_o.
    // E/O works correctly for the standard HMC path (no force splitting).
    bool eo = false;
    int n_half = 2 * lat.V_half;
    Vec phi;
    generate_pseudofermion(D_init, rng, phi);
    if (!eo) {
        generate_pseudofermion(D_init, rng, phi);
    }

    // --- Initial Hamiltonian ---
    double KE_init = mom.kinetic_energy();
    double SG_init = gauge_action(gauge, params.beta);
    double SF_init;
    if (eo) {
        EvenOddDiracOp eoD(D_init);
        Vec phi_o_init = eoD.gather_odd(phi);
        OpApply A = [&eoD](const Vec& s, Vec& d) { eoD.apply_schur_dag_schur(s, d); };
        auto res = cg_solve(A, n_half, phi_o_init, params.cg_maxiter, params.cg_tol);
        Vec x_full = eoD.reconstruct_full(res.solution);
        SF_init = std::real(dot(phi, x_full));
        total_cg += res.iterations;
    } else {
        OpApply A = [&D_init](const Vec& s, Vec& d) { D_init.apply_DdagD(s, d); };
        auto res = cg_solve_precond(A, lat.ndof, phi, mg_precond,
                                     params.cg_maxiter, params.cg_tol);
        SF_init = std::real(dot(phi, res.solution));
        total_cg += res.iterations;
    }
    double H_init = KE_init + SG_init + SF_init;

    // ── Primitives ──

    auto compute_outer_force = [&](std::array<RVec, 2>& f_out) {
        auto t0 = Clock::now();
        if (pre_solve) (*pre_solve)();  // refresh MG before CG solve
        DiracOp D(lat, gauge, mass, wilson_r, c_sw, mu_t);
        std::array<RVec, 2> gf, ff_full, fl;
        gauge_force(gauge, params.beta, gf);

        if (eo) {
            // E/O CG for faster convergence, but full-lattice force for
            // consistency with the inner force splitting
            EvenOddDiracOp eoD(D);
            Vec phi_o_cur = eoD.gather_odd(phi);
            OpApply A = [&eoD](const Vec& s, Vec& d) { eoD.apply_schur_dag_schur(s, d); };
            auto res = cg_solve(A, n_half, phi_o_cur, params.cg_maxiter, params.cg_tol);
            total_cg += res.iterations;
            Vec x_full = eoD.reconstruct_full(res.solution);
            fermion_force(D, x_full, ff_full);
        } else {
            OpApply A = [&D](const Vec& s, Vec& d) { D.apply_DdagD(s, d); };
            auto res = cg_solve_precond(A, lat.ndof, phi, mg_precond,
                                         params.cg_maxiter, params.cg_tol);
            total_cg += res.iterations;
            fermion_force(D, res.solution, ff_full);
        }

        // Inner force (coarse deflation) — always on full lattice
        coarse_lowmode_force(D, cdefl, P, phi, fl, params.inner_smooth);
        f_out[0].resize(lat.V);
        f_out[1].resize(lat.V);
        #pragma omp parallel for collapse(2) schedule(static)
        for (int mu = 0; mu < 2; mu++)
            for (int s = 0; s < lat.V; s++)
                f_out[mu][s] = gf[mu][s] + ff_full[mu][s] - fl[mu][s];
        highmode_time += Dur(Clock::now() - t0).count();
    };

    auto kick_mom = [&](const std::array<RVec, 2>& f, double dt) {
        #pragma omp parallel for collapse(2) schedule(static)
        for (int mu = 0; mu < 2; mu++)
            for (int s = 0; s < lat.V; s++)
                mom.pi[mu][s] += dt * f[mu][s];
    };

    auto update_gauge = [&](double dt) {
        #pragma omp parallel for collapse(2) schedule(static)
        for (int mu = 0; mu < 2; mu++)
            for (int s = 0; s < lat.V; s++)
                gauge.U[mu][s] *= std::exp(cx(0, dt * mom.pi[mu][s]));
    };

    auto compute_inner_force = [&](std::array<RVec, 2>& fl) {
        auto t0 = Clock::now();
        DiracOp D(lat, gauge, mass, wilson_r, c_sw, mu_t);
        coarse_lowmode_force(D, cdefl, P, phi, fl, params.inner_smooth);
        lowmode_evals++;
        lowmode_time += Dur(Clock::now() - t0).count();
    };

    // Track inner steps for periodic deflation refresh
    int inner_step_counter = 0;

    auto maybe_refresh_deflation = [&]() {
        if (inner_step_counter == 0) return;
        bool do_full = (params.defl_refresh > 0 &&
                        inner_step_counter % params.defl_refresh == 0);
        if (do_full) {
            // Expensive: rebuild coarse operator + RR evolve + extract generator
            DiracOp D_ref(lat, gauge, mass, wilson_r, c_sw, mu_t);
            OpApply A_ref = [&D_ref](const Vec& s, Vec& d) { D_ref.apply_DdagD(s, d); };
            SparseCoarseOp sac_tmp;
            sac_tmp.build(P, A_ref, lat.ndof);
            evolve_coarse_deflation(cdefl, sac_tmp, forecast);
        } else if (forecast && forecast->history_len >= 2) {
            // Cheap: apply predicted rotation only (no coarse op rebuild)
            auto R_pred = forecast_rotation(*forecast);
            apply_rotation(cdefl.eigvecs, R_pred, (int)cdefl.eigvecs[0].size());
        }
    };

    // Perturbation-based MG prolongator refresh: compute delta_D before gauge
    // update, then force_evolve + rebuild P after. Zero full matvecs.
    auto maybe_perturb_mg = [&](double dti) {
        if (!mg_hierarchy || params.mg_perturb_freq <= 0) return;
        if (inner_step_counter == 0) return;
        if (inner_step_counter % params.mg_perturb_freq != 0) return;

        auto& mg = *mg_hierarchy;
        int k = (int)mg.null_vecs_l0.size();
        if (k == 0 || mg.Dv_l0.empty()) return;

        // Compute delta_D * v_i BEFORE gauge update (uses current/old gauge)
        DiracOp D_old(lat, gauge, mass, wilson_r, c_sw, mu_t);
        std::vector<Vec> dDv(k);
        for (int j = 0; j < k; j++) {
            dDv[j].resize(lat.ndof);
            D_old.apply_delta_D(mg.null_vecs_l0[j], dDv[j], mom.pi, dti);
        }

        // Gauge update happens in caller after this returns
        // We store dDv for use after gauge update
        // ... actually, we need to do the full sequence here.
        // Do the gauge update, force_evolve, rebuild P, then the caller
        // skips its own gauge update.

        // Better approach: just apply the perturbation AFTER gauge update
        // by using the fact that delta_D was computed on old gauge.
        // We pass dDv to force_evolve_precomputed, which only needs the
        // inner products and doesn't re-apply delta_D.
        // But we need to update gauge first for the Galerkin rebuild...

        // Cleanest: do gauge update HERE, then force_evolve + rebuild
        update_gauge(dti);

        auto result = force_evolve_precomputed(
            mg.null_vecs_l0, mg.null_evals_l0, mg.Dv_l0, dDv, lat.ndof);
        mg.null_vecs_l0 = std::move(result.eigvecs);
        mg.null_evals_l0 = std::move(result.eigvals);
        mg.Dv_l0 = std::move(result.Dv);

        // Rebuild P from rotated null vecs
        mg.geo_prolongators[0].build_from_vectors(mg.null_vecs_l0);
        // Rebuild Galerkin coarse op
        DiracOp D_new(lat, gauge, mass, wilson_r, c_sw, mu_t);
        mg.levels[0].Ac.build(D_new, mg.geo_prolongators[0]);
        mg.rebuild_deeper_levels();
    };

    // ── Inner sub-integrator: N_inner leapfrog steps with low-mode + gauge force ──
    auto inner_integrator = [&](double dt_total) {
        int ni = params.n_inner;
        double dti = dt_total / ni;

        std::array<RVec, 2> fl;
        compute_inner_force(fl);
        kick_mom(fl, 0.5 * dti);
        for (int i = 0; i < ni; i++) {
            bool did_perturb = false;
            if (mg_hierarchy && params.mg_perturb_freq > 0 &&
                inner_step_counter > 0 &&
                (inner_step_counter + 1) % params.mg_perturb_freq == 0) {
                // Perturbation refresh includes gauge update
                maybe_perturb_mg(dti);
                did_perturb = true;
            }
            if (!did_perturb) update_gauge(dti);
            inner_step_counter++;
            maybe_refresh_deflation();
            compute_inner_force(fl);
            kick_mom(fl, (i < ni - 1) ? dti : 0.5 * dti);
        }
    };

    // ── Outer integrator on the EXPENSIVE force ──
    std::array<RVec, 2> F_outer;
    double h = dt_outer;

    if (params.outer_type == OuterIntegrator::Leapfrog) {
        compute_outer_force(F_outer);
        kick_mom(F_outer, 0.5 * h);
        for (int o = 0; o < params.n_outer; o++) {
            inner_integrator(h);
            compute_outer_force(F_outer);
            kick_mom(F_outer, (o < params.n_outer - 1) ? h : 0.5 * h);
        }

    } else if (params.outer_type == OuterIntegrator::Omelyan) {
        double lam = 0.1932;
        compute_outer_force(F_outer);
        kick_mom(F_outer, lam * h);
        for (int o = 0; o < params.n_outer; o++) {
            inner_integrator(0.5 * h);
            compute_outer_force(F_outer);
            kick_mom(F_outer, (1.0 - 2.0 * lam) * h);
            inner_integrator(0.5 * h);
            compute_outer_force(F_outer);
            kick_mom(F_outer, (o < params.n_outer - 1) ? (2.0 * lam * h) : (lam * h));
        }

    } else {
        // =====================================================
        //  MILC-style nested FGI (PQPQP_FGI)
        //  Yin & Mawhinney arXiv:1111.5059, Kennedy et al.
        //
        //  P(λh) inner(h/2) FG((1-2λ)h, ξh³) inner(h/2) P(λh)
        //
        //  P = EXPENSIVE force kick (fermion, CG solve)
        //  inner = CHEAP sub-integrator (gauge + low-mode, leapfrog)
        //  FG = Hessian-free force gradient on the EXPENSIVE force:
        //    0. Save gauge & momentum, zero momentum
        //    1. Compute expensive force → kick momentum by (ξh³/((1-2λ)h)) = ξh²/(1-2λ)
        //    2. Update gauge by the resulting momentum (unit step)
        //    3. Restore momentum
        //    4. Compute expensive force at displaced gauge → kick by (1-2λ)h
        //    5. Restore gauge
        //
        //  λ=1/6, ξ=1/72
        // =====================================================
        double lam = 1.0 / 6.0;
        double xi = 1.0 / 72.0;
        double one_m_2lam = 1.0 - 2.0 * lam;  // = 2/3
        double xi_h3 = 2.0 * xi * h * h * h;  // = 2 * h³/72 = h³/36

        // FG step on the expensive force (following MILC force_gradient())
        auto fg_step = [&]() {
            auto t0 = Clock::now();

            // Save gauge and momentum
            GaugeField gauge_save(lat);
            gauge_save.U[0] = gauge.U[0];
            gauge_save.U[1] = gauge.U[1];
            MomentumField mom_save(lat);
            mom_save.pi[0] = mom.pi[0];
            mom_save.pi[1] = mom.pi[1];

            // Zero momentum
            #pragma omp parallel for collapse(2) schedule(static)
            for (int mu = 0; mu < 2; mu++)
                for (int s = 0; s < lat.V; s++)
                    mom.pi[mu][s] = 0.0;

            // Compute expensive force, kick by eps_ttt/eps_t = xi_h3 / (one_m_2lam * h)
            std::array<RVec, 2> f_tmp;
            compute_outer_force(f_tmp);
            double fg_kick_coeff = xi_h3 / (one_m_2lam * h);
            kick_mom(f_tmp, fg_kick_coeff);

            // Update gauge by unit step (U' = exp(i * 1.0 * pi) * U)
            // pi now contains only the FG displacement
            update_gauge(1.0);

            // Restore momentum
            mom.pi[0] = mom_save.pi[0];
            mom.pi[1] = mom_save.pi[1];

            // Compute expensive force at displaced gauge, kick by (1-2λ)h
            compute_outer_force(f_tmp);
            kick_mom(f_tmp, one_m_2lam * h);

            // Restore gauge
            gauge.U[0] = gauge_save.U[0];
            gauge.U[1] = gauge_save.U[1];

            highmode_time += Dur(Clock::now() - t0).count();
        };

        // Main loop: P(λh) inner(h/2) FG inner(h/2) P(λh)
        compute_outer_force(F_outer);
        kick_mom(F_outer, lam * h);

        for (int o = 0; o < params.n_outer; o++) {
            inner_integrator(0.5 * h);
            fg_step();
            inner_integrator(0.5 * h);

            compute_outer_force(F_outer);
            kick_mom(F_outer, (o < params.n_outer - 1) ? (2.0 * lam * h) : (lam * h));
        }
    }

    // --- Final Hamiltonian ---
    if (pre_solve) (*pre_solve)();  // refresh MG before final H CG
    double KE_final = mom.kinetic_energy();
    double SG_final = gauge_action(gauge, params.beta);
    double SF_final;
    if (eo) {
        DiracOp D(lat, gauge, mass, wilson_r, c_sw, mu_t);
        EvenOddDiracOp eoD(D);
        Vec phi_o_fin = eoD.gather_odd(phi);
        OpApply A = [&eoD](const Vec& s, Vec& d) { eoD.apply_schur_dag_schur(s, d); };
        auto res = cg_solve(A, n_half, phi_o_fin, params.cg_maxiter, params.cg_tol);
        Vec x_full = eoD.reconstruct_full(res.solution);
        SF_final = std::real(dot(phi, x_full));
        total_cg += res.iterations;
    } else {
        DiracOp D(lat, gauge, mass, wilson_r, c_sw, mu_t);
        OpApply A = [&D](const Vec& s, Vec& d) { D.apply_DdagD(s, d); };
        auto res = cg_solve_precond(A, lat.ndof, phi, mg_precond,
                                     params.cg_maxiter, params.cg_tol);
        SF_final = std::real(dot(phi, res.solution));
        total_cg += res.iterations;
    }
    double H_final = KE_final + SG_final + SF_final;

    double dH = H_final - H_init;
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    bool accept = (dH < 0) || (uniform(rng) < std::exp(-dH));

    if (!accept) {
        gauge.U[0] = gauge_old.U[0];
        gauge.U[1] = gauge_old.U[1];
    }

    return {accept, dH, KE_final - KE_init, SG_final - SG_init, SF_final - SF_init,
            total_cg, lowmode_evals, highmode_time, lowmode_time};
}

// ---------------------------------------------------------------
//  Factored MD evolution for reversibility test
// ---------------------------------------------------------------
static int mg_ms_md_evolve(
    GaugeField& gauge, MomentumField& mom,
    const Lattice& lat, double mass, double wilson_r, const Vec& phi,
    const MGMultiScaleParams& params, CoarseDeflState& cdefl,
    Prolongator& P, std::function<Vec(const Vec&)>& mg_precond)
{
    double h = params.tau / params.n_outer;
    double c_sw = params.c_sw;
    double mu_t = params.mu_t;
    int total_cg = 0;
    auto oforce = [&](std::array<RVec,2>& f) {
        DiracOp D(lat,gauge,mass,wilson_r,c_sw);
        std::array<RVec,2> gf,ff,fl;
        gauge_force(gauge,params.beta,gf);
        OpApply A=[&D](const Vec&s,Vec&d){D.apply_DdagD(s,d);};
        auto r=cg_solve_precond(A,lat.ndof,phi,mg_precond,params.cg_maxiter,params.cg_tol);
        total_cg+=r.iterations;
        fermion_force(D,r.solution,ff);
        coarse_lowmode_force(D,cdefl,P,phi,fl,params.inner_smooth);
        f[0].resize(lat.V); f[1].resize(lat.V);
        for(int mu=0;mu<2;mu++) for(int s=0;s<lat.V;s++) f[mu][s]=gf[mu][s]+ff[mu][s]-fl[mu][s];
    };
    auto kmom = [&](const std::array<RVec,2>&f,double dt){
        for(int mu=0;mu<2;mu++) for(int s=0;s<lat.V;s++) mom.pi[mu][s]+=dt*f[mu][s]; };
    auto ug = [&](double dt){
        for(int mu=0;mu<2;mu++) for(int s=0;s<lat.V;s++) gauge.U[mu][s]*=std::exp(cx(0,dt*mom.pi[mu][s])); };
    auto iforce=[&](std::array<RVec,2>&fl){ DiracOp D(lat,gauge,mass,wilson_r,c_sw); coarse_lowmode_force(D,cdefl,P,phi,fl,params.inner_smooth); };
    auto inner=[&](double dt_total){
        int ni=params.n_inner; double dti=dt_total/ni; std::array<RVec,2> fl;
        iforce(fl); kmom(fl,0.5*dti);
        for(int i=0;i<ni;i++){ ug(dti); iforce(fl); kmom(fl,(i<ni-1)?dti:0.5*dti); } };
    std::array<RVec,2> F;
    if(params.outer_type==OuterIntegrator::Leapfrog){
        oforce(F); kmom(F,0.5*h);
        for(int o=0;o<params.n_outer;o++){ inner(h); oforce(F); kmom(F,(o<params.n_outer-1)?h:0.5*h); }
    } else if(params.outer_type==OuterIntegrator::Omelyan){
        double lam=0.1932; oforce(F); kmom(F,lam*h);
        for(int o=0;o<params.n_outer;o++){
            inner(0.5*h); oforce(F); kmom(F,(1-2*lam)*h); inner(0.5*h); oforce(F);
            kmom(F,(o<params.n_outer-1)?(2*lam*h):(lam*h)); }
    } else {
        double lam=1./6,xi=1./72,om=1-2*lam,xh3=2*xi*h*h*h;
        auto fgs=[&](){
            GaugeField gs(lat); gs.U[0]=gauge.U[0]; gs.U[1]=gauge.U[1];
            MomentumField ms(lat); ms.pi[0]=mom.pi[0]; ms.pi[1]=mom.pi[1];
            for(int mu=0;mu<2;mu++) for(int s=0;s<lat.V;s++) mom.pi[mu][s]=0;
            std::array<RVec,2> ft; oforce(ft); kmom(ft,xh3/(om*h)); ug(1.0);
            mom.pi[0]=ms.pi[0]; mom.pi[1]=ms.pi[1]; oforce(ft); kmom(ft,om*h);
            gauge.U[0]=gs.U[0]; gauge.U[1]=gs.U[1]; };
        oforce(F); kmom(F,lam*h);
        for(int o=0;o<params.n_outer;o++){
            inner(0.5*h); fgs(); inner(0.5*h); oforce(F);
            kmom(F,(o<params.n_outer-1)?(2*lam*h):(lam*h)); }
    }
    return total_cg;
}

ReversibilityResult reversibility_test_mg_multiscale(
    GaugeField& gauge, const Lattice& lat, double mass, double wilson_r,
    const MGMultiScaleParams& params, CoarseDeflState& cdefl,
    Prolongator& P, std::function<Vec(const Vec&)>& mg_precond, std::mt19937& rng)
{
    double c_sw = params.c_sw;
    double mu_t = params.mu_t;
    int total_cg=0;
    GaugeField gi(lat); gi.U[0]=gauge.U[0]; gi.U[1]=gauge.U[1];
    MomentumField mom(lat); mom.randomise(rng);
    MomentumField mi(lat); mi.pi[0]=mom.pi[0]; mi.pi[1]=mom.pi[1];
    DiracOp Di(lat,gauge,mass,wilson_r,c_sw); Vec phi; generate_pseudofermion(Di,rng,phi);
    auto compH=[&]()->double{
        double H=mom.kinetic_energy()+gauge_action(gauge,params.beta);
        DiracOp D(lat,gauge,mass,wilson_r,c_sw); OpApply A=[&D](const Vec&s,Vec&d){D.apply_DdagD(s,d);};
        auto r=cg_solve_precond(A,lat.ndof,phi,mg_precond,params.cg_maxiter,params.cg_tol);
        total_cg+=r.iterations; H+=std::real(dot(phi,r.solution)); return H; };
    double H0=compH();
    total_cg+=mg_ms_md_evolve(gauge,mom,lat,mass,wilson_r,phi,params,cdefl,P,mg_precond);
    double H1=compH(); double dH_fwd=H1-H0;
    for(int mu=0;mu<2;mu++) for(int s=0;s<lat.V;s++) mom.pi[mu][s]*=-1;
    total_cg+=mg_ms_md_evolve(gauge,mom,lat,mass,wilson_r,phi,params,cdefl,P,mg_precond);
    double H2=compH(); double dH_bwd=H2-H1;
    double diff=0,ref=0;
    for(int mu=0;mu<2;mu++) for(int s=0;s<lat.V;s++){
        diff+=std::norm(gauge.U[mu][s]-gi.U[mu][s]); ref+=std::norm(gi.U[mu][s]); }
    double md=0,mr=0;
    for(int mu=0;mu<2;mu++) for(int s=0;s<lat.V;s++){
        double d=mom.pi[mu][s]+mi.pi[mu][s]; md+=d*d; mr+=mi.pi[mu][s]*mi.pi[mu][s]; }
    gauge.U[0]=gi.U[0]; gauge.U[1]=gi.U[1];
    return {std::sqrt(diff/ref),std::sqrt(md/mr),dH_fwd,dH_bwd,total_cg};
}
