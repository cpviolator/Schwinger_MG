#include "hmc.h"
#include <cmath>
#include <iostream>
#include <iomanip>
#include <cstdio>

void perturb_gauge(GaugeField& g, std::mt19937& rng, double epsilon) {
    std::normal_distribution<double> dist(0, epsilon);
    for (int mu = 0; mu < 2; mu++)
        for (int s = 0; s < g.lat.V; s++)
            g.U[mu][s] *= std::exp(cx(0, dist(rng)));
}

bool save_gauge(const GaugeField& g, double beta, double mass,
                const std::string& filename) {
    FILE* f = fopen(filename.c_str(), "wb");
    if (!f) { std::cerr << "Cannot open " << filename << " for writing\n"; return false; }

    GaugeHeader hdr;
    hdr.magic = 0x53574E47;
    hdr.L = g.lat.L;
    hdr.beta = beta;
    hdr.mass = mass;
    hdr.avg_plaq = g.avg_plaq();

    fwrite(&hdr, sizeof(GaugeHeader), 1, f);
    for (int mu = 0; mu < 2; mu++)
        fwrite(g.U[mu].data(), sizeof(cx), g.lat.V, f);
    fclose(f);
    return true;
}

bool load_gauge(GaugeField& g, GaugeHeader& hdr, const std::string& filename) {
    FILE* f = fopen(filename.c_str(), "rb");
    if (!f) { std::cerr << "Cannot open " << filename << " for reading\n"; return false; }

    fread(&hdr, sizeof(GaugeHeader), 1, f);
    if (hdr.magic != 0x53574E47) {
        std::cerr << "Bad magic number in " << filename << "\n";
        fclose(f); return false;
    }
    if ((int)hdr.L != g.lat.L) {
        std::cerr << "Lattice size mismatch: file has L=" << hdr.L
                  << " but current L=" << g.lat.L << "\n";
        fclose(f); return false;
    }

    for (int mu = 0; mu < 2; mu++)
        fread(g.U[mu].data(), sizeof(cx), g.lat.V, f);
    fclose(f);

    for (int mu = 0; mu < 2; mu++)
        for (int s = 0; s < g.lat.V; s++)
            g.U[mu][s] /= std::abs(g.U[mu][s]);

    return true;
}

void MomentumField::randomise(std::mt19937& rng) {
    std::normal_distribution<double> dist(0.0, 1.0);
    for (int mu = 0; mu < 2; mu++)
        for (int s = 0; s < lat.V; s++)
            pi[mu][s] = dist(rng);
}

double MomentumField::kinetic_energy() const {
    double ke = 0.0;
    for (int mu = 0; mu < 2; mu++)
        for (int s = 0; s < lat.V; s++)
            ke += pi[mu][s] * pi[mu][s];
    return 0.5 * ke;
}

double gauge_action(const GaugeField& g, double beta) {
    double s = 0.0;
    for (int site = 0; site < g.lat.V; site++)
        s += std::real(g.plaq(site));
    return -beta * s;
}

void gauge_force(const GaugeField& g, double beta,
                 std::array<RVec, 2>& force) {
    const Lattice& lat = g.lat;
    force[0].resize(lat.V);
    force[1].resize(lat.V);

    for (int s = 0; s < lat.V; s++) {
        {
            int sx = lat.xp(s), sy = lat.yp(s), smy = lat.ym(s);
            int sxmy = lat.ym(sx);
            cx staple = g.U[1][sx] * std::conj(g.U[0][sy]) * std::conj(g.U[1][s])
                      + std::conj(g.U[1][sxmy]) * std::conj(g.U[0][smy]) * g.U[1][smy];
            force[0][s] = -beta * std::imag(g.U[0][s] * staple);
        }
        {
            int sy = lat.yp(s), sx = lat.xp(s), smx = lat.xm(s);
            int symx = lat.xm(sy);
            cx staple = g.U[0][sy] * std::conj(g.U[1][sx]) * std::conj(g.U[0][s])
                      + std::conj(g.U[0][symx]) * std::conj(g.U[1][smx]) * g.U[0][smx];
            force[1][s] = -beta * std::imag(g.U[1][s] * staple);
        }
    }
}

FermionActionResult fermion_action(const DiracOp& D, const Vec& phi,
                                    int max_iter, double tol) {
    OpApply A = [&D](const Vec& src, Vec& dst) { D.apply_DdagD(src, dst); };
    auto res = cg_solve(A, D.lat.ndof, phi, max_iter, tol);
    double act = std::real(dot(phi, res.solution));
    return {act, std::move(res.solution), res.iterations};
}

FermionActionResult fermion_action_precond(
    const DiracOp& D, const Vec& phi,
    const std::function<Vec(const Vec&)>& precond,
    int max_iter, double tol) {
    OpApply A = [&D](const Vec& src, Vec& dst) { D.apply_DdagD(src, dst); };
    auto res = cg_solve_precond(A, D.lat.ndof, phi, precond, max_iter, tol);
    double act = std::real(dot(phi, res.solution));
    return {act, std::move(res.solution), res.iterations};
}

void fermion_force(const DiracOp& D, const Vec& X,
                   std::array<RVec, 2>& force) {
    const Lattice& lat = D.lat;
    force[0].assign(lat.V, 0.0);
    force[1].assign(lat.V, 0.0);

    Vec chi(lat.ndof);
    D.apply(X, chi);

    for (int s = 0; s < lat.V; s++) {
        cx chi_s0 = chi[2*s], chi_s1 = chi[2*s+1];
        cx x_s0 = X[2*s], x_s1 = X[2*s+1];
        cx ii(0,1);

        {
            int sf = lat.xp(s);
            cx u = D.gauge.U[0][s];
            cx ud = std::conj(u);

            cx ux0 = u * X[2*sf], ux1 = u * X[2*sf+1];
            cx fwd0 = D.r * ux0 - ux1;
            cx fwd1 = -ux0 + D.r * ux1;
            cx c_fwd = std::conj(chi_s0) * fwd0 + std::conj(chi_s1) * fwd1;

            cx udx0 = ud * x_s0, udx1 = ud * x_s1;
            cx bwd0 = D.r * udx0 + udx1;
            cx bwd1 = udx0 + D.r * udx1;
            cx c_bwd = std::conj(chi[2*sf]) * bwd0 + std::conj(chi[2*sf+1]) * bwd1;

            force[0][s] = std::real(ii * c_bwd - ii * c_fwd);
        }

        {
            int sf = lat.yp(s);
            cx u = D.gauge.U[1][s];
            cx ud = std::conj(u);

            cx ux0 = u * X[2*sf], ux1 = u * X[2*sf+1];
            cx fwd0 = D.r * ux0 + ii * ux1;
            cx fwd1 = -ii * ux0 + D.r * ux1;
            cx c_fwd = std::conj(chi_s0) * fwd0 + std::conj(chi_s1) * fwd1;

            cx udx0 = ud * x_s0, udx1 = ud * x_s1;
            cx bwd0 = D.r * udx0 - ii * udx1;
            cx bwd1 = ii * udx0 + D.r * udx1;
            cx c_bwd = std::conj(chi[2*sf]) * bwd0 + std::conj(chi[2*sf+1]) * bwd1;

            force[1][s] = std::real(ii * c_bwd - ii * c_fwd);
        }
    }
}

void verify_forces(const GaugeField& g, double beta, double mass, double wilson_r,
                   int max_iter, double tol) {
    const Lattice& lat = g.lat;
    double eps = 1e-5;
    std::mt19937 rng(12345);

    DiracOp D0(lat, g, mass, wilson_r);
    Vec phi(lat.ndof);
    Vec eta = random_vec(lat.ndof, rng);
    D0.apply_dag(eta, phi);

    std::array<RVec, 2> gf_ana;
    gauge_force(g, beta, gf_ana);

    OpApply A0 = [&D0](const Vec& src, Vec& dst) { D0.apply_DdagD(src, dst); };
    auto res0 = cg_solve(A0, lat.ndof, phi, max_iter, tol);
    std::array<RVec, 2> ff_ana;
    fermion_force(D0, res0.solution, ff_ana);

    double max_gf_err = 0, max_ff_err = 0;
    double max_gf_val = 0, max_ff_val = 0;

    for (int test = 0; test < 10; test++) {
        int mu = test % 2;
        int s = test * 7 % lat.V;

        GaugeField g_plus(lat);
        g_plus.U[0] = g.U[0]; g_plus.U[1] = g.U[1];
        g_plus.U[mu][s] *= std::exp(cx(0, eps));

        GaugeField g_minus(lat);
        g_minus.U[0] = g.U[0]; g_minus.U[1] = g.U[1];
        g_minus.U[mu][s] *= std::exp(cx(0, -eps));

        double sg_plus = gauge_action(g_plus, beta);
        double sg_minus = gauge_action(g_minus, beta);
        double gf_num = -(sg_plus - sg_minus) / (2 * eps);

        double gf_err = std::abs(gf_ana[mu][s] - gf_num);
        max_gf_err = std::max(max_gf_err, gf_err);
        max_gf_val = std::max(max_gf_val, std::abs(gf_ana[mu][s]));

        DiracOp D_plus(lat, g_plus, mass, wilson_r);
        DiracOp D_minus(lat, g_minus, mass, wilson_r);
        OpApply Ap = [&D_plus](const Vec& src, Vec& dst) { D_plus.apply_DdagD(src, dst); };
        OpApply Am = [&D_minus](const Vec& src, Vec& dst) { D_minus.apply_DdagD(src, dst); };
        auto rp = cg_solve(Ap, lat.ndof, phi, max_iter, tol);
        auto rm = cg_solve(Am, lat.ndof, phi, max_iter, tol);
        double sf_plus = std::real(dot(phi, rp.solution));
        double sf_minus = std::real(dot(phi, rm.solution));
        double ff_num = -(sf_plus - sf_minus) / (2 * eps);

        double ff_err = std::abs(ff_ana[mu][s] - ff_num);
        max_ff_err = std::max(max_ff_err, ff_err);
        max_ff_val = std::max(max_ff_val, std::abs(ff_ana[mu][s]));

        std::cout << "  link(" << mu << "," << s << "): "
                  << "gf_ana=" << std::setw(10) << gf_ana[mu][s]
                  << " gf_num=" << std::setw(10) << gf_num
                  << " | ff_ana=" << std::setw(10) << ff_ana[mu][s]
                  << " ff_num=" << std::setw(10) << ff_num << "\n";
    }
    std::cout << "Gauge force: max_err=" << max_gf_err
              << " rel=" << max_gf_err / std::max(max_gf_val, 1e-30) << "\n";
    std::cout << "Fermion force: max_err=" << max_ff_err
              << " rel=" << max_ff_err / std::max(max_ff_val, 1e-30) << "\n";
}

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
    int total_cg = 0;

    GaugeField gauge_old(lat);
    gauge_old.U[0] = gauge.U[0];
    gauge_old.U[1] = gauge.U[1];

    MomentumField mom(lat);
    mom.randomise(rng);

    DiracOp D_init(lat, gauge, mass, wilson_r);
    Vec phi;
    generate_pseudofermion(D_init, rng, phi);

    double H_init = mom.kinetic_energy() + gauge_action(gauge, params.beta);
    {
        OpApply A = [&D_init](const Vec& src, Vec& dst) { D_init.apply_DdagD(src, dst); };
        CGResult res;
        if (precond)
            res = cg_solve_precond(A, lat.ndof, phi, *precond, params.cg_maxiter, params.cg_tol);
        else
            res = cg_solve(A, lat.ndof, phi, params.cg_maxiter, params.cg_tol);
        H_init += std::real(dot(phi, res.solution));
        total_cg += res.iterations;
    }

    // Half-step momenta
    {
        DiracOp D(lat, gauge, mass, wilson_r);
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

        for (int mu = 0; mu < 2; mu++)
            for (int s = 0; s < lat.V; s++)
                mom.pi[mu][s] += 0.5 * dt * (gf[mu][s] + ff[mu][s]);
    }

    // Full steps
    for (int step = 0; step < params.n_steps - 1; step++) {
        for (int mu = 0; mu < 2; mu++)
            for (int s = 0; s < lat.V; s++)
                gauge.U[mu][s] *= std::exp(cx(0, dt * mom.pi[mu][s]));

        DiracOp D(lat, gauge, mass, wilson_r);
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

        for (int mu = 0; mu < 2; mu++)
            for (int s = 0; s < lat.V; s++)
                mom.pi[mu][s] += dt * (gf[mu][s] + ff[mu][s]);
    }

    // Final gauge update
    for (int mu = 0; mu < 2; mu++)
        for (int s = 0; s < lat.V; s++)
            gauge.U[mu][s] *= std::exp(cx(0, dt * mom.pi[mu][s]));

    // Half-step momenta (final)
    {
        DiracOp D(lat, gauge, mass, wilson_r);
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

        for (int mu = 0; mu < 2; mu++)
            for (int s = 0; s < lat.V; s++)
                mom.pi[mu][s] += 0.5 * dt * (gf[mu][s] + ff[mu][s]);
    }

    // Final Hamiltonian
    double H_final = mom.kinetic_energy() + gauge_action(gauge, params.beta);
    {
        DiracOp D(lat, gauge, mass, wilson_r);
        OpApply A = [&D](const Vec& src, Vec& dst) { D.apply_DdagD(src, dst); };
        CGResult res;
        if (precond)
            res = cg_solve_precond(A, lat.ndof, phi, *precond, params.cg_maxiter, params.cg_tol);
        else
            res = cg_solve(A, lat.ndof, phi, params.cg_maxiter, params.cg_tol);
        H_final += std::real(dot(phi, res.solution));
        total_cg += res.iterations;
    }

    double dH = H_final - H_init;
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    bool accept = (dH < 0) || (uniform(rng) < std::exp(-dH));

    if (!accept) {
        gauge.U[0] = gauge_old.U[0];
        gauge.U[1] = gauge_old.U[1];
    }

    return {accept, dH, 0.0, total_cg};
}
