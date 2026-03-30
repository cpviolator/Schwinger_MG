#include "hmc.h"
#include "eigensolver.h"
#include <cmath>
#include <chrono>
#include <iostream>
#include <iomanip>

void generate_pseudofermion(const DiracOp& D, std::mt19937& rng, Vec& phi) {
    Vec eta = random_vec(D.lat.ndof, rng);
    phi.resize(D.lat.ndof);
    D.apply_dag(eta, phi);
}

HMCResult hmc_trajectory(
    GaugeField& gauge, const Lattice& lat, double mass, double wilson_r,
    const HMCParams& params, std::mt19937& rng,
    const std::function<Vec(const Vec&)>* precond)
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

    // === Full-lattice path (original) ===
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

    // Half-step momenta
    {
        DiracOp D(lat, gauge, mass, wilson_r, c_sw, mu_t);
        std::array<RVec, 2> gf, ff;
        gauge_force(gauge, params.beta, gf);

        OpApply A = [&D](const Vec& src, Vec& dst) { D.apply_DdagD(src, dst); };
        CGResult res;
        if (precond)
            res = cg_solve_precond(A, lat.ndof, phi, *precond, params.cg_maxiter, params.cg_tol);
        else
            res = cg_solve(A, lat.ndof, phi, params.cg_maxiter, params.cg_tol);
        total_cg += res.iterations;
        fermion_force(D, res.solution, ff);

        #pragma omp parallel for collapse(2) schedule(static)
        for (int mu = 0; mu < 2; mu++)
            for (int s = 0; s < lat.V; s++)
                mom.pi[mu][s] += 0.5 * dt * (gf[mu][s] + ff[mu][s]);
    }

    // Full steps
    for (int step = 0; step < params.n_steps - 1; step++) {
        #pragma omp parallel for collapse(2) schedule(static)
        for (int mu = 0; mu < 2; mu++)
            for (int s = 0; s < lat.V; s++)
                gauge.U[mu][s] *= std::exp(cx(0, dt * mom.pi[mu][s]));

        DiracOp D(lat, gauge, mass, wilson_r, c_sw, mu_t);
        std::array<RVec, 2> gf, ff;
        gauge_force(gauge, params.beta, gf);

        OpApply A = [&D](const Vec& src, Vec& dst) { D.apply_DdagD(src, dst); };
        CGResult res;
        if (precond)
            res = cg_solve_precond(A, lat.ndof, phi, *precond, params.cg_maxiter, params.cg_tol);
        else
            res = cg_solve(A, lat.ndof, phi, params.cg_maxiter, params.cg_tol);
        total_cg += res.iterations;
        fermion_force(D, res.solution, ff);

        #pragma omp parallel for collapse(2) schedule(static)
        for (int mu = 0; mu < 2; mu++)
            for (int s = 0; s < lat.V; s++)
                mom.pi[mu][s] += dt * (gf[mu][s] + ff[mu][s]);
    }

    // Final gauge update
    #pragma omp parallel for collapse(2) schedule(static)
    for (int mu = 0; mu < 2; mu++)
        for (int s = 0; s < lat.V; s++)
            gauge.U[mu][s] *= std::exp(cx(0, dt * mom.pi[mu][s]));

    // Half-step momenta (final)
    {
        DiracOp D(lat, gauge, mass, wilson_r, c_sw, mu_t);
        std::array<RVec, 2> gf, ff;
        gauge_force(gauge, params.beta, gf);

        OpApply A = [&D](const Vec& src, Vec& dst) { D.apply_DdagD(src, dst); };
        CGResult res;
        if (precond)
            res = cg_solve_precond(A, lat.ndof, phi, *precond, params.cg_maxiter, params.cg_tol);
        else
            res = cg_solve(A, lat.ndof, phi, params.cg_maxiter, params.cg_tol);
        total_cg += res.iterations;
        fermion_force(D, res.solution, ff);

        #pragma omp parallel for collapse(2) schedule(static)
        for (int mu = 0; mu < 2; mu++)
            for (int s = 0; s < lat.V; s++)
                mom.pi[mu][s] += 0.5 * dt * (gf[mu][s] + ff[mu][s]);
    }

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
//  Low-mode fermion force from eigenpairs of D†D
//  Computes X_low = Σ_i (v_i†φ / λ_i) v_i  (low-mode part of solution)
//  Then calls the standard fermion_force on X_low.
// ---------------------------------------------------------------
