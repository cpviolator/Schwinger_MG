#include "hmc.h"
#include "eigensolver.h"
#include <cmath>
#include <chrono>
#include <iostream>

// Helper: get EigenTracker from TrackingState's opaque pointer
static EigenTracker* get_tracker(TrackingState* ts) {
    return static_cast<EigenTracker*>(ts->tracker_ptr);
}
static const EigenTracker* get_tracker(const TrackingState* ts) {
    return static_cast<const EigenTracker*>(ts->tracker_ptr);
}

std::vector<Vec> TrackingState::get_null_vectors() const {
    auto* tracker = get_tracker(this);
    if (!tracker_initialized || !tracker || tracker->pool.empty())
        return {};
    int k = std::min(n_ev, (int)tracker->pool.size());
    return std::vector<Vec>(tracker->pool.begin(), tracker->pool.begin() + k);
}
#include <iomanip>

void generate_pseudofermion(const DiracOp& D, std::mt19937& rng, Vec& phi) {
    Vec eta = random_vec(D.lat.ndof, rng);
    phi.resize(D.lat.ndof);
    D.apply_dag(eta, phi);
}

HMCResult hmc_trajectory(
    GaugeField& gauge, const Lattice& lat, double mass, double wilson_r,
    const HMCParams& params, std::mt19937& rng,
    const std::function<Vec(const Vec&)>* precond,
    TrackingState* tracking)
{
    double dt = params.tau / params.n_steps;
    double c_sw = params.c_sw;
    double mu_t = params.mu_t;
    int total_cg = 0;

    GaugeField gauge_old(lat);
    gauge_old.U[0] = gauge.U[0];
    gauge_old.U[1] = gauge.U[1];

    MomentumField mom(lat);
    mom.randomise(rng);

    // === Even-Odd preconditioned path ===
    // Uses Schur complement M†M on odd sites for CG, with dedicated e/o force
    // For clover: includes log-det and clover diagonal derivative terms
    if (params.use_eo) {
        int n_half = 2 * lat.V_half;

        // Generate pseudofermion in Schur space
        DiracOp D_init(lat, gauge, mass, wilson_r, c_sw, mu_t);
        EvenOddDiracOp eoD_init(D_init);
        Vec phi_o = eoD_init.generate_pseudofermion_eo(rng);

        // Solve + force helper
        auto solve_and_force = [&](std::array<RVec, 2>& ff) {
            DiracOp D(lat, gauge, mass, wilson_r, c_sw, mu_t);
            EvenOddDiracOp eoD(D);
            OpApply A = [&eoD](const Vec& s, Vec& d) { eoD.apply_schur_dag_schur(s, d); };
            auto res = cg_solve(A, n_half, phi_o, params.cg_maxiter, params.cg_tol);
            total_cg += res.iterations;
            Vec y_o(n_half);
            eoD.apply_schur(res.solution, y_o);
            eo_fermion_force(D, eoD, res.solution, y_o, ff);
            return res;
        };

        // Compute per-monomial Hamiltonian components
        struct HComponents { double KE, SG, SF, LD; };
        auto compute_H_components = [&]() -> HComponents {
            HComponents h;
            h.KE = mom.kinetic_energy();
            h.SG = gauge_action(gauge, params.beta);
            DiracOp D(lat, gauge, mass, wilson_r, c_sw, mu_t);
            EvenOddDiracOp eoD(D);
            OpApply A = [&eoD](const Vec& s, Vec& d) { eoD.apply_schur_dag_schur(s, d); };
            auto res = cg_solve(A, n_half, phi_o, params.cg_maxiter, params.cg_tol);
            total_cg += res.iterations;
            h.SF = std::real(dot(phi_o, res.solution));
            h.LD = (c_sw != 0.0) ? -2.0 * eoD.log_det_ee() : 0.0;
            return h;
        };

        auto H_i = compute_H_components();
        double H_init = H_i.KE + H_i.SG + H_i.SF + H_i.LD;

        // Leapfrog: half-step, full steps, half-step
        std::array<RVec, 2> gf, ff;
        gauge_force(gauge, params.beta, gf);
        auto res0 = solve_and_force(ff);
        #pragma omp parallel for collapse(2) schedule(static)
        for (int mu = 0; mu < 2; mu++)
            for (int s = 0; s < lat.V; s++)
                mom.pi[mu][s] += 0.5 * dt * (gf[mu][s] + ff[mu][s]);

        for (int step = 0; step < params.n_steps - 1; step++) {
            #pragma omp parallel for collapse(2) schedule(static)
            for (int mu = 0; mu < 2; mu++)
                for (int s = 0; s < lat.V; s++)
                    gauge.U[mu][s] *= std::exp(cx(0, dt * mom.pi[mu][s]));
            gauge_force(gauge, params.beta, gf);
            solve_and_force(ff);
            #pragma omp parallel for collapse(2) schedule(static)
            for (int mu = 0; mu < 2; mu++)
                for (int s = 0; s < lat.V; s++)
                    mom.pi[mu][s] += dt * (gf[mu][s] + ff[mu][s]);
        }

        #pragma omp parallel for collapse(2) schedule(static)
        for (int mu = 0; mu < 2; mu++)
            for (int s = 0; s < lat.V; s++)
                gauge.U[mu][s] *= std::exp(cx(0, dt * mom.pi[mu][s]));

        gauge_force(gauge, params.beta, gf);
        solve_and_force(ff);
        #pragma omp parallel for collapse(2) schedule(static)
        for (int mu = 0; mu < 2; mu++)
            for (int s = 0; s < lat.V; s++)
                mom.pi[mu][s] += 0.5 * dt * (gf[mu][s] + ff[mu][s]);

        auto H_f = compute_H_components();
        double H_final = H_f.KE + H_f.SG + H_f.SF + H_f.LD;
        double dH = H_final - H_init;
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        bool accept = (dH < 0) || (uniform(rng) < std::exp(-dH));
        if (!accept) {
            gauge.U[0] = gauge_old.U[0];
            gauge.U[1] = gauge_old.U[1];
        }
        return {accept, dH, H_f.KE - H_i.KE, H_f.SG - H_i.SG,
                H_f.SF - H_i.SF, H_f.LD - H_i.LD, total_cg};
    }

    // === Full-lattice path ===
    DiracOp D_init(lat, gauge, mass, wilson_r, c_sw, mu_t);
    Vec phi;
    generate_pseudofermion(D_init, rng, phi);

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

    // MD evolution (leapfrog or Omelyan)
    if (tracking) tracking->reset_trajectory();
    total_cg += plain_leapfrog_evolve(gauge, mom, lat, mass, wilson_r, phi, params, precond, tracking);

    // Final Hamiltonian
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

    return {accept, dH, KE_final - KE_init, SG_final - SG_init,
            SF_final - SF_init, 0.0, total_cg};
}

// ---------------------------------------------------------------
//  Plain leapfrog MD evolution (factored out for reversibility)
//  Evolves gauge and mom in-place using the standard VV leapfrog.
//  phi is the pseudofermion (fixed throughout the trajectory).
//  Returns total CG iterations used.
// ---------------------------------------------------------------
int plain_leapfrog_evolve(
    GaugeField& gauge, MomentumField& mom,
    const Lattice& lat, double mass, double wilson_r,
    const Vec& phi, const HMCParams& params,
    const std::function<Vec(const Vec&)>* precond,
    TrackingState* tracking)
{
    double dt = params.tau / params.n_steps;
    double c_sw = params.c_sw;
    double mu_t = params.mu_t;
    int total_cg = 0;

    auto solve_force = [&](std::array<RVec, 2>& gf, std::array<RVec, 2>& ff) {
        DiracOp D(lat, gauge, mass, wilson_r, c_sw, mu_t);
        gauge_force(gauge, params.beta, gf);
        OpApply A = [&D](const Vec& src, Vec& dst) { D.apply_DdagD(src, dst); };

        if (tracking) {
            // --- One-time tracker initialization from first CG solve ---
            auto* tracker = get_tracker(tracking);
            if (!tracking->tracker_initialized && !tracker) {
                // Run quick TRLM to seed the pool
                auto trlm = trlm_eigensolver(A, lat.ndof, tracking->n_ev,
                    std::min(2 * tracking->n_ev + 10, lat.ndof), 100, 1e-8);
                if (trlm.converged) {
                    tracker = new EigenTracker();
                    tracking->tracker_ptr = tracker;
                    auto apply_D_init = [&D](const Vec& s, Vec& d) { D.apply(s, d); };
                    tracker->init(trlm, apply_D_init, lat.ndof,
                                  tracking->n_ev, tracking->pool_capacity);
                    tracking->tracker_initialized = true;
                }
            }

            // Chronological initial guess
            const Vec* x0_ptr = tracking->has_prev_solution
                ? &tracking->prev_solution : nullptr;

            // Tracked CG: x0 + preconditioner + Ritz extraction
            // Cap Lanczos vectors to 3×n_ritz to control memory/compute
            int max_lcz = tracking->n_ritz > 0 ? 3 * tracking->n_ritz : 0;
            auto res = cg_solve_tracked(A, lat.ndof, phi,
                x0_ptr, precond, params.cg_maxiter, params.cg_tol,
                tracking->n_ritz, max_lcz);
            total_cg += res.iterations;

            // Save solution for next chronological guess
            tracking->prev_solution = res.solution;
            tracking->has_prev_solution = true;

            // Absorb Ritz vectors + solution into EigenTracker pool
            tracker = get_tracker(tracking);
            if (tracking->tracker_initialized && tracker) {
                auto apply_D = [&D](const Vec& s, Vec& d) { D.apply(s, d); };

                // Absorb Ritz vectors
                if (!res.ritz_pairs.empty()) {
                    std::vector<Vec> ritz_vecs;
                    for (auto& rp : res.ritz_pairs)
                        ritz_vecs.push_back(std::move(rp.vector));
                    int n_abs = tracker->absorb(ritz_vecs, apply_D);
                    tracking->total_ritz_absorbed += n_abs;
                }

                // Absorb normalised solution vector
                Vec x_norm = res.solution;
                double xn = norm(x_norm);
                if (xn > 1e-14) {
                    scale(x_norm, cx(1.0/xn));
                    tracker->absorb({x_norm}, apply_D);
                    tracking->total_solutions_absorbed++;
                }
            }

            tracking->force_eval_count++;
            fermion_force(D, res.solution, ff);
        } else {
            // Original path: no tracking
            CGResult res;
            if (precond)
                res = cg_solve_precond(A, lat.ndof, phi, *precond, params.cg_maxiter, params.cg_tol);
            else
                res = cg_solve(A, lat.ndof, phi, params.cg_maxiter, params.cg_tol);
            total_cg += res.iterations;
            fermion_force(D, res.solution, ff);
        }
    };

    auto update_mom = [&](std::array<RVec, 2>& gf, std::array<RVec, 2>& ff, double scale) {
        #pragma omp parallel for collapse(2) schedule(static)
        for (int mu = 0; mu < 2; mu++)
            for (int s = 0; s < lat.V; s++)
                mom.pi[mu][s] += scale * (gf[mu][s] + ff[mu][s]);
    };

    auto update_gauge = [&](double scale) {
        #pragma omp parallel for collapse(2) schedule(static)
        for (int mu = 0; mu < 2; mu++)
            for (int s = 0; s < lat.V; s++)
                gauge.U[mu][s] *= std::exp(cx(0, scale * mom.pi[mu][s]));
    };

    std::array<RVec, 2> gf, ff;

    if (params.omelyan) {
        // Omelyan (2MN) integrator: PQPQP with λ=0.1932
        // One step: P(λε) Q(ε/2) P((1-2λ)ε) Q(ε/2) P(λε)
        // N steps combined: P(λε) [Q(ε/2) P((1-2λ)ε) Q(ε/2) P(2λε)]^{N-1} Q(ε/2) P((1-2λ)ε) Q(ε/2) P(λε)
        constexpr double lambda = 0.1932;
        int N = params.n_steps;

        // Initial P(λε)
        solve_force(gf, ff);
        update_mom(gf, ff, lambda * dt);

        for (int step = 0; step < N; step++) {
            // Q(ε/2)
            update_gauge(0.5 * dt);
            // P((1-2λ)ε)
            solve_force(gf, ff);
            update_mom(gf, ff, (1.0 - 2.0 * lambda) * dt);
            // Q(ε/2)
            update_gauge(0.5 * dt);
            // P(2λε) or P(λε) for last step
            solve_force(gf, ff);
            if (step < N - 1)
                update_mom(gf, ff, 2.0 * lambda * dt);
            else
                update_mom(gf, ff, lambda * dt);
        }
    } else {
        // Standard leapfrog (VV): P(dt/2) [Q(dt) P(dt)]^{N-1} Q(dt) P(dt/2)
        solve_force(gf, ff);
        update_mom(gf, ff, 0.5 * dt);

        for (int step = 0; step < params.n_steps - 1; step++) {
            update_gauge(dt);
            solve_force(gf, ff);
            update_mom(gf, ff, dt);
        }

        update_gauge(dt);
        solve_force(gf, ff);
        update_mom(gf, ff, 0.5 * dt);
    }

    return total_cg;
}

// ---------------------------------------------------------------
//  Plain-HMC reversibility test
//  Forward trajectory → negate momenta → backward trajectory.
//  Gauge and momenta should return to O(machine epsilon).
// ---------------------------------------------------------------
ReversibilityResult reversibility_test_plain(
    GaugeField& gauge, const Lattice& lat, double mass, double wilson_r,
    const HMCParams& params, std::mt19937& rng,
    const std::function<Vec(const Vec&)>* precond)
{
    double c_sw = params.c_sw;
    double mu_t = params.mu_t;

    // Save initial gauge
    GaugeField gauge_save(lat);
    gauge_save.U[0] = gauge.U[0];
    gauge_save.U[1] = gauge.U[1];

    // Generate pseudofermion and momenta
    DiracOp D_init(lat, gauge, mass, wilson_r, c_sw, mu_t);
    Vec phi;
    generate_pseudofermion(D_init, rng, phi);
    MomentumField mom(lat);
    mom.randomise(rng);

    // Save initial momenta
    std::array<RVec, 2> pi_save = {mom.pi[0], mom.pi[1]};

    // Compute H_init
    double KE_init = mom.kinetic_energy();
    double SG_init = gauge_action(gauge, params.beta);
    double SF_init;
    {
        OpApply A = [&D_init](const Vec& src, Vec& dst) { D_init.apply_DdagD(src, dst); };
        auto res = cg_solve(A, lat.ndof, phi, params.cg_maxiter, params.cg_tol);
        SF_init = std::real(dot(phi, res.solution));
    }
    double H_init = KE_init + SG_init + SF_init;

    // Forward trajectory
    int cg_fwd = plain_leapfrog_evolve(gauge, mom, lat, mass, wilson_r, phi, params, precond);

    // Compute H after forward
    double KE_fwd = mom.kinetic_energy();
    double SG_fwd = gauge_action(gauge, params.beta);
    double SF_fwd;
    {
        DiracOp D(lat, gauge, mass, wilson_r, c_sw, mu_t);
        OpApply A = [&D](const Vec& src, Vec& dst) { D.apply_DdagD(src, dst); };
        auto res = cg_solve(A, lat.ndof, phi, params.cg_maxiter, params.cg_tol);
        SF_fwd = std::real(dot(phi, res.solution));
    }
    double dH_fwd = (KE_fwd + SG_fwd + SF_fwd) - H_init;

    // Negate momenta
    for (int mu = 0; mu < 2; mu++)
        for (int s = 0; s < lat.V; s++)
            mom.pi[mu][s] = -mom.pi[mu][s];

    // Backward trajectory
    int cg_bwd = plain_leapfrog_evolve(gauge, mom, lat, mass, wilson_r, phi, params, precond);

    // Compute H after backward
    double KE_bwd = mom.kinetic_energy();
    double SG_bwd = gauge_action(gauge, params.beta);
    double SF_bwd;
    {
        DiracOp D(lat, gauge, mass, wilson_r, c_sw, mu_t);
        OpApply A = [&D](const Vec& src, Vec& dst) { D.apply_DdagD(src, dst); };
        auto res = cg_solve(A, lat.ndof, phi, params.cg_maxiter, params.cg_tol);
        SF_bwd = std::real(dot(phi, res.solution));
    }
    double dH_bwd = (KE_bwd + SG_bwd + SF_bwd) - (KE_fwd + SG_fwd + SF_fwd);

    // Compare gauge to saved
    double gauge_norm = 0, gauge_diff = 0;
    for (int mu = 0; mu < 2; mu++)
        for (int s = 0; s < lat.V; s++) {
            gauge_norm += std::norm(gauge_save.U[mu][s]);
            gauge_diff += std::norm(gauge.U[mu][s] - gauge_save.U[mu][s]);
        }
    double gauge_delta = std::sqrt(gauge_diff / std::max(gauge_norm, 1e-30));

    // Compare momenta (backward negates, so compare -mom to saved)
    double mom_norm = 0, mom_diff = 0;
    for (int mu = 0; mu < 2; mu++)
        for (int s = 0; s < lat.V; s++) {
            mom_norm += pi_save[mu][s] * pi_save[mu][s];
            double d = (-mom.pi[mu][s]) - pi_save[mu][s];
            mom_diff += d * d;
        }
    double mom_delta = std::sqrt(mom_diff / std::max(mom_norm, 1e-30));

    // Restore gauge
    gauge.U[0] = gauge_save.U[0];
    gauge.U[1] = gauge_save.U[1];

    return {gauge_delta, mom_delta, dH_fwd, dH_bwd, cg_fwd + cg_bwd};
}

// ---------------------------------------------------------------
//  Low-mode fermion force from eigenpairs of D†D
//  Computes X_low = Σ_i (v_i†φ / λ_i) v_i  (low-mode part of solution)
//  Then calls the standard fermion_force on X_low.
// ---------------------------------------------------------------
