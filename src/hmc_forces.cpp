#include "hmc.h"
#include "eigensolver.h"
#include "multigrid.h"
#include <cmath>
#include <chrono>
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

    #pragma omp parallel for collapse(2) schedule(static)
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
    #pragma omp parallel for collapse(2) reduction(+:ke) schedule(static)
    for (int mu = 0; mu < 2; mu++)
        for (int s = 0; s < lat.V; s++)
            ke += pi[mu][s] * pi[mu][s];
    return 0.5 * ke;
}

double gauge_action(const GaugeField& g, double beta) {
    double s = 0.0;
    #pragma omp parallel for schedule(static) if(g.lat.V > OMP_MIN_SIZE/4)
    for (int site = 0; site < g.lat.V; site++)
        s += std::real(g.plaq(site));
    return -beta * s;
}

void gauge_force(const GaugeField& g, double beta,
                 std::array<RVec, 2>& force) {
    const Lattice& lat = g.lat;
    force[0].resize(lat.V);
    force[1].resize(lat.V);

    #pragma omp parallel for schedule(static) if(lat.V > OMP_MIN_SIZE/4)
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

void fermion_force_bilinear(const DiracOp& D, const Vec& chi, const Vec& X,
                            std::array<RVec, 2>& force);

void fermion_force(const DiracOp& D, const Vec& X,
                   std::array<RVec, 2>& force) {
    Vec chi(D.lat.ndof);
    D.apply(X, chi);
    fermion_force_bilinear(D, chi, X, force);
}

void fermion_force_bilinear(const DiracOp& D, const Vec& chi, const Vec& X,
                            std::array<RVec, 2>& force) {
    const Lattice& lat = D.lat;
    force[0].assign(lat.V, 0.0);
    force[1].assign(lat.V, 0.0);

    // Hopping term contribution (standard Wilson)
    #pragma omp parallel for schedule(static) if(lat.V > OMP_MIN_SIZE/4)
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

    // Clover term contribution (factored into separate function)
    if (D.c_sw != 0.0)
        fermion_force_clover(D, chi, X, force);
}

// Clover force: derivative of the clover field w.r.t. gauge links
// Accumulates into force (does not zero it)
void fermion_force_clover(const DiracOp& D, const Vec& chi, const Vec& X,
                          std::array<RVec, 2>& force) {
    const Lattice& lat = D.lat;
    // The clover adds (c_sw/2) F_01(x) σ_3 to the Dirac operator.
    // The force from D†D involves d(D†D)/dA which gets contributions from
    // dD_clov/dA. For U(1), dF_01(x)/dA_mu(s) comes from differentiating
    // the 4 plaquettes in the clover at x that contain link (mu,s).
    //
    // For each link (mu,s), we compute d(S_fermion_clover)/dA_mu(s) numerically
    // via the chain: F_01(x) depends on plaquettes, plaquettes depend on links.
    // The clover contribution to the D†D force is:
    //   Σ_x [dF_01(x)/dA_mu(s)] × (c_sw/2) × (chi†σ_3 X + X†σ_3 chi)(x)
    //   = Σ_x [dF_01(x)/dA_mu(s)] × c_sw × Re(χ_0* X_0 - χ_1* X_1)(x)
    //
    // where the σ_3 weight is w(x) = Re(χ_0* X_0 - χ_1* X_1).
    //
    // dF_01(x)/dA_mu(s) = (1/4) × d(Im(Q_01(x)))/dA_mu(s)
    // For U(1), d(Im(P))/dA_mu(s) = Re(i × dP/dA_mu(s))
    //   = Re(i × (±i) × P) = ∓Re(P) for forward/backward links.
    {
        // Precompute σ_3 weight at each site: w(x) = Re(χ_0* X_0 - χ_1* X_1)
        RVec w(lat.V);
        #pragma omp parallel for schedule(static) if(lat.V > OMP_MIN_SIZE/4)
        for (int s = 0; s < lat.V; s++)
            w[s] = std::real(std::conj(chi[2*s]) * X[2*s]
                           - std::conj(chi[2*s+1]) * X[2*s+1]);

        double csw = D.c_sw;

        #pragma omp parallel for schedule(static) if(lat.V > OMP_MIN_SIZE/4)
        for (int s = 0; s < lat.V; s++) {
            int sxp = lat.xp(s), sxm = lat.xm(s);
            int syp = lat.yp(s), sym = lat.ym(s);
            int sxpym = lat.ym(sxp), sxmyp = lat.yp(sxm);
            int sxmym = lat.ym(sxm);

            // For link U_0(s): find all plaquettes containing U_0(s) in any clover
            // U_0(s) appears forward in clovers at 4 sites, but in U(1) the
            // derivative d(Im(P))/dA = ∓Re(P) depending on forward/backward.
            //
            // The 4 clover sites whose Q_01 contains U_0(s) forward:
            //   x=s (P_UR), x=s+x (P_UL), x=s+x+y (P_LL), x=s+y (P_LR)
            // And U_0†(s) backward:
            //   x=s (P_LR has U_0†(s)), x=s-y (P_UR of s-y has... )
            //
            // For U(1) this simplifies: the clover force from link (0,s) is:
            // (c_sw/2) × (1/4) × Σ over affected sites x of:
            //   d(Im(P))/dA_0(s) × w(x)
            //
            // Each plaquette P containing U_0(s) forward contributes:
            //   d(Im(P))/dA_0(s) = Re(i × i × P) = -Re(P)
            //   (since d/dA [e^{iA}] = i e^{iA}, so dP/dA = i P when U_0 is first)
            //
            // For backward (U_0†(s)): d(Im(P))/dA_0(s) = Re(P)

            // Enumerate all plaquette contributions for mu=0 link at s:
            // Forward plaquettes (U_0(s) appears as e^{iA}):
            //   P_UR(s): U_0(s)·U_1(sxp)·U_0†(syp)·U_1†(s)  → site s, dIm/dA = -Re(P)
            //   P_UL(sxp): contains U_0(s) as last link → site sxp, same contrib
            //   P_LL(sxp+y): U_0(s) → site sxp+y... hmm getting complicated
            //
            // Simpler approach: compute dQ_01(x)/dA_0(s) for each clover site x

            // Actually, the cleanest approach for U(1) in 2D:
            // For mu=0 link at s, the clover derivative is the imaginary part of
            // the "clover staple" — the sum of all paths that complete the
            // plaquettes touching U_0(s), weighted by w at the clover site.

            // Upper staple (plaquettes in the +y direction):
            // Staple_up = U_1(sxp)·U_0†(syp)·U_1†(s) (completes P_UR at site s)
            // This plaquette appears in Q_01(s) and Q_01(sxp) (as P_UL)
            cx staple_up = D.gauge.U[1][sxp] * std::conj(D.gauge.U[0][syp])
                         * std::conj(D.gauge.U[1][s]);
            // Full plaquette: P = U_0(s) × staple_up
            // Sites where this plaquette appears: s (P_UR) and sxp (P_UL of sxp)
            // Weight contribution: w(s) + w(sxp)... but P_UL at sxp uses U_0(s) differently.
            // Actually for U(1), the clover at site x has 4 plaquettes, and U_0(s) contributes
            // the same way to each one it appears in (just a phase factor).

            // Let me use the direct approach: for each plaquette touching U_0(s),
            // identify which clover site it belongs to, and accumulate w.

            // Plaquette P_UR(s) contains U_0(s) forward, belongs to clover at s.
            // Plaquette P_UL(sxp) contains U_0(s) forward, belongs to clover at sxp.
            // These are the SAME plaquette (cyclic in U(1))!
            // Similarly for the other directions.
            // So actually, U_0(s) forward appears in 2 distinct plaquettes:
            //   P_up = U_0(s)·U_1(sxp)·U_0†(syp)·U_1†(s)  → in clovers at s and sxp
            //   P_down = ... the plaquette below

            // P_up (going right then up): in clover at s (P_UR) and sxp (P_UL)
            cx P_up = D.gauge.U[0][s] * staple_up;

            // P_down (going right then down):
            // U_0(s)·U_1†(sym)·... wait, this isn't a standard plaquette starting from s.
            // The plaquette below s in the -y direction containing U_0(s) is:
            // U_0(s)·U_1†(sxpym)·U_0†(sym)·U_1(sym)... no.
            // Let me just directly compute it:
            // Going: right(U_0(s)), down(U_1†(sxpym)), left(U_0†(sym)), up(U_1(sym))
            // Wait, that's not a plaquette. Let me think again.
            //
            // Actually, the plaquette that goes right then down is:
            // P = U_0(s) · U_1†(sxp-y) · U_0†(s-y) · U_1(s-y)
            // Hmm, this doesn't look right either.
            //
            // The plaquette in the lower-right quadrant at site s:
            // Start at s, go right, down, left, up:
            // But we can't "go down" — we use U_1† in the backward direction.
            // P_down = U_1†(sym) · U_0(sym) · U_1(sxpym) · U_0†(s)
            // This is P_LR(s) and it contains U_0†(s), not U_0(s).
            //
            // So U_0(s) as a FORWARD link appears only in plaquettes going right-then-up
            // and in the conjugate arrangement. In 2D, each oriented link (μ,s) appears
            // forward in exactly ONE standard plaquette:
            //   P(s) = U_0(s) · U_1(sxp) · U_0†(syp) · U_1†(s)
            //
            // And U_0†(s) appears in plaquettes at:
            //   P_LR(s) = U_1†(sym)·U_0(sym)·U_1(sxpym)·U_0†(s)  (lower-right at s)
            //   P_LR(syp) contains U_0†(syp)... no, U_0†(s) specifically.

            // OK let me just be systematic. For U(1) in 2D with the 4-plaquette clover:
            //
            // dF_01(x)/dA_0(s) ≠ 0 only when one of the 4 plaquettes in Q_01(x)
            // contains U_0(s) or U_0†(s).
            //
            // P_UR(x) = U_0(x)·U_1(x+x̂)·U_0†(x+ŷ)·U_1†(x)
            //   Contains U_0(x) when x=s, and U_0†(x+ŷ) when x+ŷ=s → x=s-ŷ.
            //
            // P_UL(x) = U_1(x)·U_0†(x+ŷ-x̂)·U_1†(x-x̂)·U_0(x-x̂)
            //   Contains U_0†(x+ŷ-x̂) when x+ŷ-x̂=s → x=s+x̂-ŷ.
            //   Contains U_0(x-x̂) when x-x̂=s → x=s+x̂.
            //
            // P_LL(x) = U_0†(x-x̂)·U_1†(x-x̂-ŷ)·U_0(x-x̂-ŷ)·U_1(x-ŷ)
            //   Contains U_0†(x-x̂) when x-x̂=s → x=s+x̂.
            //   Contains U_0(x-x̂-ŷ) when x-x̂-ŷ=s → x=s+x̂+ŷ.
            //
            // P_LR(x) = U_1†(x-ŷ)·U_0(x-ŷ)·U_1(x+x̂-ŷ)·U_0†(x)
            //   Contains U_0(x-ŷ) when x-ŷ=s → x=s+ŷ.
            //   Contains U_0†(x) when x=s.
            //
            // Summary of clover sites x where dQ_01(x)/dA_0(s) ≠ 0:
            //
            // U_0(s) forward (d/dA gives +i factor):
            //   P_UR(s):      x=s
            //   P_UL(s+x̂):   x=s+x̂
            //   P_LL(s+x̂+ŷ): x=s+x̂+ŷ
            //   P_LR(s+ŷ):   x=s+ŷ
            //
            // U_0†(s) backward (d/dA gives -i factor):
            //   P_UR(s-ŷ):    x=s-ŷ
            //   P_UL(s+x̂-ŷ): x=s+x̂-ŷ
            //   P_LL(s+x̂):   x=s+x̂
            //   P_LR(s):      x=s

            // For each, the derivative of Im(plaquette) w.r.t. A gives:
            //   d(Im(P))/dA = Re(dP/dA) where dP/dA = ±i × P (U(1))
            //   Forward: dP/dA = i×P → d(Im(P))/dA = Re(i×P) = -Im(P)... wait.
            //   Actually d(Im(z))/dθ where z = e^{iθ} × rest = Re(i × z) = -Im(z)?
            //   Hmm. Let P = U_0(s) × S where S is the staple.
            //   d/dA_0(s) [Im(P)] = d/dA [Im(e^{iA} S)] = Im(i e^{iA} S) = Im(iP) = Re(P)
            //
            //   For backward: P = ... × U_0†(s) × ... = ... × e^{-iA} × ...
            //   d/dA [Im(P)] = Im(-iP) = -Re(P)

            // So the clover force from mu=0 at site s is:
            // (c_sw/2) × (1/4) × [
            //   Σ_{forward sites x} Re(P_x) × w(x)     // P_x is the specific plaquette
            // - Σ_{backward sites x} Re(P_x) × w(x)
            // ]
            // But we need to be careful: each plaquette contribution is Re(P) where P
            // is computed at the specific site x.

            // For simplicity, use the fact that for U(1), the plaquettes in forward
            // and backward are just complex conjugates of each other traversed
            // in opposite directions.

            // Let me compute this directly. The 4 forward plaquettes give:
            // P_UR(s) = U_0(s)·U_1(sxp)·U_0†(syp)·U_1†(s)
            // For the clover at site s, P_UL at s+x̂, P_LL at s+x̂+ŷ, P_LR at s+ŷ
            // These are 4 different plaquettes but in U(1) 2D the plaquette is the
            // same regardless of starting corner. So Re(P) is the same for all 4.

            // Actually no — the 4 plaquettes are at different locations!
            // P_UR(s) uses links at s, sxp, syp, s
            // P_UL(sxp) uses links at sxp, sxmyp... wait, P_UL at x=sxp:
            //   P_UL(sxp) = U_1(sxp)·U_0†(syp)·U_1†(s)·U_0(s) — SAME plaquette as P_UR(s)!

            // In U(1), cyclic permutation of a product of phases doesn't change the value.
            // So P_UR(s) = P_UL(sxp) = P_LL(sxp+y=sxpyp) = P_LR(syp) as complex numbers.
            // They're all the same plaquette value, just different starting points.

            // So the forward contribution is: Re(P_upper) × (w(s) + w(sxp) + w(sxpyp) + w(syp))
            // where P_upper = gauge.plaq(s) (the standard plaquette at s).

            // Similarly, for the backward (U_0†(s)):
            // P_UR(s-ŷ) = plaquette at s-ŷ
            // All 4 backward sites share the plaquette at s-ŷ (P at sym)
            // The backward contribution is: -Re(P_lower) × (w(sym) + w(sxpym) + w(sxp) + w(s))

            // Wait, the backward sites are: s-ŷ, s+x̂-ŷ, s+x̂, s.
            // P_UR(s-ŷ) = plaq(s-ŷ), but P_UL(s+x̂-ŷ) is a different plaquette!
            // P_UL(x) uses U_0†(x+ŷ-x̂). For x=s+x̂-ŷ: U_0†(s+x̂-ŷ+ŷ-x̂) = U_0†(s). ✓
            // But P_UL(s+x̂-ŷ) = U_1(s+x̂-ŷ)·U_0†(s)·U_1†(s-ŷ)·U_0(s-ŷ)
            // This is a different plaquette from P_UR(s-ŷ)!

            // OK I was wrong about them all being the same plaquette. They ARE different
            // plaquettes at different locations. Let me just compute all 8 contributions
            // (4 forward + 4 backward) explicitly.

            // FORWARD: U_0(s) appears, derivative gives +Re(P)
            // 1. P_UR(s): at site s. P = U_0(s)·U_1(sxp)·U_0†(syp)·U_1†(s) = plaq(s)
            cx P_UR_s = D.gauge.U[0][s] * D.gauge.U[1][sxp]
                      * std::conj(D.gauge.U[0][syp]) * std::conj(D.gauge.U[1][s]);

            // 2. P_UL(sxp): U_0(s) is last link. P = U_1(sxp)·U_0†(syp)·U_1†(s)·U_0(s)
            // Same value as P_UR_s for U(1). But the clover site is sxp, so weight is w(sxp).

            // 3. P_LL(sxp+ŷ): U_0(s) = U_0((sxp+ŷ)-x̂-ŷ) = U_0(s). Hmm...
            // P_LL(x) has U_0(x-x̂-ŷ). For x=s+x̂+ŷ: U_0(s+x̂+ŷ-x̂-ŷ) = U_0(s). ✓
            // P_LL(sxp+ŷ) = U_0†(syp)·U_1†(s)·U_0(s)·U_1(sxp... no wait)
            // P_LL(x) = U_0†(x-x̂)·U_1†(x-x̂-ŷ)·U_0(x-x̂-ŷ)·U_1(x-ŷ)
            // For x = sxp_yp (=lat.yp(sxp)):
            // Need x-x̂ = syp, x-x̂-ŷ = s
            // P = U_0†(syp)·U_1†(s)·U_0(s)·U_1(sxp)
            // For U(1): same product as P_UR_s (cyclic), so same value.
            // Clover site is sxp_yp = lat.yp(sxp).
            int sxpyp = lat.yp(sxp);

            // 4. P_LR(syp): U_0(s) = U_0(syp-ŷ) = U_0(s). ✓
            // P_LR(x) = U_1†(x-ŷ)·U_0(x-ŷ)·U_1(x+x̂-ŷ)·U_0†(x)
            // For x = syp: P_LR(syp) = U_1†(s)·U_0(s)·U_1(sxp)·U_0†(syp)
            // Same value as P_UR_s for U(1).

            // So all 4 forward plaquettes have the same value P_UR_s (U(1) abelian)
            double re_P_fwd = std::real(P_UR_s);
            double w_fwd = w[s] + w[sxp] + w[sxpyp] + w[syp];

            // BACKWARD: U_0†(s) appears, derivative gives -Re(P)
            // 5. P_UR(s-ŷ): U_0†(s) = U_0†((s-ŷ)+ŷ). P_UR(sym) = plaq(sym)
            cx P_UR_sym = D.gauge.U[0][sym] * D.gauge.U[1][sxpym]
                        * std::conj(D.gauge.U[0][s]) * std::conj(D.gauge.U[1][sym]);
            // 6. P_UL(sxp-ŷ): clover at sxpym
            // 7. P_LL(sxp): clover at sxp
            // 8. P_LR(s): clover at s
            // For U(1), these all have the same value as P_UR_sym (cyclic).
            double re_P_bwd = std::real(P_UR_sym);
            double w_bwd = w[sym] + w[sxpym] + w[sxp] + w[s];

            // Total clover force for mu=0 at site s:
            // F_clov = (c_sw/2) × (1/4) × (re_P_fwd × w_fwd - re_P_bwd × w_bwd)
            force[0][s] += (csw / 4.0) * (re_P_fwd * w_fwd - re_P_bwd * w_bwd);

            // Similarly for mu=1 link at s:
            // By symmetry, exchange x↔y directions.
            // Forward plaquettes containing U_1(s):
            // P = U_0(s)·U_1(sxp)·U_0†(syp)·U_1†(s) contains U_1†(s), not U_1(s).
            // The plaquette containing U_1(s) forward is:
            // Going up then right: U_1(s)·U_0(syp)·U_1†(sxp)·U_0†(s)
            // Wait, that's a plaquette in the (1,0) plane = conjugate of (0,1) plane.
            // Let me just swap x and y.

            // For mu=1: the relevant plaquettes are in the (1,0) orientation.
            // But our clover Q_01 only has (0,1) plaquettes.
            // In 2D there's only one plane, so mu=1 links also participate.

            // P_UR(x) has U_1†(x). So U_1(s) appears conjugated in P_UR(s): backward.
            // P_UR(x) has U_1(x+x̂). So U_1(s) forward when x+x̂=s → x=s-x̂=sxm.

            // Let me enumerate properly for mu=1:
            // Forward (U_1(s)):
            //   P_UR(x) has U_1(x+x̂): x+x̂=s → x=sxm. Site: sxm.
            //   P_UL(x) has U_1(x): x=s. Site: s.
            //   P_LL(x) has U_1(x-ŷ): x-ŷ=s → x=syp. Site: syp.
            //   P_LR(x) has U_1(x+x̂-ŷ): x+x̂-ŷ=s → x=sxm+ŷ=sxmyp. Site: sxmyp.

            // All forward plaquettes for U_1(s):
            // P_UR(sxm) = U_0(sxm)·U_1(s)·U_0†(sxm+ŷ=sxmyp)·U_1†(sxm)
            cx P_fwd_1 = D.gauge.U[0][sxm] * D.gauge.U[1][s]
                       * std::conj(D.gauge.U[0][sxmyp]) * std::conj(D.gauge.U[1][sxm]);
            // For U(1), all 4 forward plaquettes have the same value (cyclic).
            double re_P_fwd_1 = std::real(P_fwd_1);
            double w_fwd_1 = w[sxm] + w[s] + w[syp] + w[sxmyp];

            // Backward (U_1†(s)):
            //   P_UR(x) has U_1†(x): x=s. Site: s.
            //   P_UL(x) has U_1†(x-x̂): x-x̂=s → x=sxp. Site: sxp.
            //   P_LL(x) has U_1†(x-x̂-ŷ): x-x̂-ŷ=s → x=sxp+ŷ=sxpyp. Site: sxpyp.
            //   P_LR(x) has U_1†(x-ŷ): x-ŷ=s → x=syp. Site: syp.

            // P_UR(s) = plaq(s) — same as P_UR_s above (contains U_1†(s)).
            // For U(1), all 4 backward plaquettes have the same value as plaq(s).
            double re_P_bwd_1 = std::real(P_UR_s); // Same plaquette!
            double w_bwd_1 = w[s] + w[sxp] + w[sxpyp] + w[syp];

            force[1][s] += (csw / 4.0) * (re_P_fwd_1 * w_fwd_1 - re_P_bwd_1 * w_bwd_1);
        }
    }
}

// ---------------------------------------------------------------
//  Common clover plaquette derivative insertion
//  Accumulates factor × w[x] × dF_01(x)/dA_mu(s) into force
// ---------------------------------------------------------------
void clover_deriv_insert(const DiracOp& D, const RVec& w, double factor,
                         std::array<RVec, 2>& force) {
    const Lattice& lat = D.lat;
    #pragma omp parallel for schedule(static) if(lat.V > OMP_MIN_SIZE/4)
    for (int s = 0; s < lat.V; s++) {
        int sxp = lat.xp(s), sxm = lat.xm(s);
        int syp = lat.yp(s), sym = lat.ym(s);
        int sxpym = lat.ym(sxp), sxmyp = lat.yp(sxm);
        int sxmym = lat.ym(sxm);
        int sxpyp = lat.yp(sxp);

        // mu=0: forward plaq containing U_0(s)
        cx P_fwd = D.gauge.U[0][s] * D.gauge.U[1][sxp]
                 * std::conj(D.gauge.U[0][syp]) * std::conj(D.gauge.U[1][s]);
        double re_fwd = std::real(P_fwd);
        double w_fwd = w[s] + w[sxp] + w[sxpyp] + w[syp];
        // mu=0: backward plaq containing U_0†(s)
        cx P_bwd = D.gauge.U[0][sym] * D.gauge.U[1][sxpym]
                 * std::conj(D.gauge.U[0][s]) * std::conj(D.gauge.U[1][sym]);
        double re_bwd = std::real(P_bwd);
        double w_bwd = w[sym] + w[sxpym] + w[sxp] + w[s];
        force[0][s] += factor * (re_fwd * w_fwd - re_bwd * w_bwd);

        // mu=1: forward plaq containing U_1(s)
        cx P_fwd1 = D.gauge.U[0][sxm] * D.gauge.U[1][s]
                  * std::conj(D.gauge.U[0][sxmyp]) * std::conj(D.gauge.U[1][sxm]);
        double re_fwd1 = std::real(P_fwd1);
        double w_fwd1 = w[sxm] + w[s] + w[syp] + w[sxmyp];
        // mu=1: backward plaq containing U_1†(s)
        double re_bwd1 = std::real(P_fwd);  // same plaq as mu=0 forward
        double w_bwd1 = w[s] + w[sxp] + w[sxpyp] + w[syp];
        force[1][s] += factor * (re_fwd1 * w_fwd1 - re_bwd1 * w_bwd1);
    }
}

// ---------------------------------------------------------------
//  Hopping-only bilinear force: Re(chi† dD_hop/dA X) per link
//  Same as fermion_force_bilinear but WITHOUT clover contribution
// ---------------------------------------------------------------
void hopping_force_bilinear(const DiracOp& D, const Vec& chi, const Vec& X,
                            std::array<RVec, 2>& force) {
    const Lattice& lat = D.lat;
    force[0].assign(lat.V, 0.0);
    force[1].assign(lat.V, 0.0);
    #pragma omp parallel for schedule(static) if(lat.V > OMP_MIN_SIZE/4)
    for (int s = 0; s < lat.V; s++) {
        cx chi_s0 = chi[2*s], chi_s1 = chi[2*s+1];
        cx x_s0 = X[2*s], x_s1 = X[2*s+1];
        cx ii(0,1);
        { int sf = lat.xp(s);
          cx u = D.gauge.U[0][s]; cx ud = std::conj(u);
          cx ux0=u*X[2*sf], ux1=u*X[2*sf+1];
          cx fwd0=D.r*ux0-ux1; cx fwd1=-ux0+D.r*ux1;
          cx c_fwd = std::conj(chi_s0)*fwd0 + std::conj(chi_s1)*fwd1;
          cx udx0=ud*x_s0, udx1=ud*x_s1;
          cx bwd0=D.r*udx0+udx1; cx bwd1=udx0+D.r*udx1;
          cx c_bwd = std::conj(chi[2*sf])*bwd0 + std::conj(chi[2*sf+1])*bwd1;
          force[0][s] = std::real(ii*c_bwd - ii*c_fwd); }
        { int sf = lat.yp(s);
          cx u = D.gauge.U[1][s]; cx ud = std::conj(u);
          cx ux0=u*X[2*sf], ux1=u*X[2*sf+1];
          cx fwd0=D.r*ux0+ii*ux1; cx fwd1=-ii*ux0+D.r*ux1;
          cx c_fwd = std::conj(chi_s0)*fwd0 + std::conj(chi_s1)*fwd1;
          cx udx0=ud*x_s0, udx1=ud*x_s1;
          cx bwd0=D.r*udx0-ii*udx1; cx bwd1=ii*udx0+D.r*udx1;
          cx c_bwd = std::conj(chi[2*sf])*bwd0 + std::conj(chi[2*sf+1])*bwd1;
          force[1][s] = std::real(ii*c_bwd - ii*c_fwd); }
    }
}

// ---------------------------------------------------------------
//  Log-det force from det(D_ee) for clover even-odd HMC
//  F = +2 d(log det(D_ee))/dA
//  = 2 Σ_{even e} [dC_e/dA × (1/(2r+m+C_e) - 1/(2r+m-C_e))]
//  Same plaquette structure as clover force, but with logdet weight
//  only at even sites.
// ---------------------------------------------------------------
void logdet_ee_force(const DiracOp& D,
                     std::array<RVec, 2>& force) {
    const Lattice& lat = D.lat;
    force[0].assign(lat.V, 0.0);
    force[1].assign(lat.V, 0.0);
    if (D.c_sw == 0.0) return;

    double csw = D.c_sw;
    double diag = 2.0 * D.r + D.mass;
    double mu2 = D.mu_t * D.mu_t;

    // Precompute logdet weight at each site (zero for odd sites)
    // d(Re log det)/dC_e = d0/(d0²+μ²) - d1/(d1²+μ²)
    // where d0 = diag+C, d1 = diag-C
    // When μ=0: reduces to 1/(diag+C) - 1/(diag-C) = -2C/((diag+C)(diag-C))
    RVec w(lat.V, 0.0);
    #pragma omp parallel for schedule(static) if(lat.V_half > OMP_MIN_SIZE/8)
    for (int ie = 0; ie < lat.V_half; ie++) {
        int e = lat.even_sites[ie];
        double C = 0.5 * csw * D.clover_field[e];
        double d0 = diag + C, d1 = diag - C;
        w[e] = 0.5 * csw * (d0/(d0*d0 + mu2) - d1/(d1*d1 + mu2));
    }

    // Same plaquette enumeration as fermion_force_clover
    #pragma omp parallel for schedule(static) if(lat.V > OMP_MIN_SIZE/4)
    for (int s = 0; s < lat.V; s++) {
        int sxp = lat.xp(s), sxm = lat.xm(s);
        int syp = lat.yp(s), sym = lat.ym(s);
        int sxpym = lat.ym(sxp), sxmyp = lat.yp(sxm);
        int sxmym = lat.ym(sxm);
        int sxpyp = lat.yp(sxp);

        // mu=0: forward plaquette
        cx P_fwd = D.gauge.U[0][s] * D.gauge.U[1][sxp]
                 * std::conj(D.gauge.U[0][syp]) * std::conj(D.gauge.U[1][s]);
        double re_P_fwd = std::real(P_fwd);
        double w_fwd = w[s] + w[sxp] + w[sxpyp] + w[syp];

        // mu=0: backward plaquette
        cx P_bwd = D.gauge.U[0][sym] * D.gauge.U[1][sxpym]
                 * std::conj(D.gauge.U[0][s]) * std::conj(D.gauge.U[1][sym]);
        double re_P_bwd = std::real(P_bwd);
        double w_bwd = w[sym] + w[sxpym] + w[sxp] + w[s];

        force[0][s] += 0.5 * (re_P_fwd * w_fwd - re_P_bwd * w_bwd);

        // mu=1: forward plaquette (containing U_1(s))
        cx P_fwd_1 = D.gauge.U[0][sxm] * D.gauge.U[1][s]
                   * std::conj(D.gauge.U[0][sxmyp]) * std::conj(D.gauge.U[1][sxm]);
        double re_P_fwd_1 = std::real(P_fwd_1);
        double w_fwd_1 = w[sxm] + w[s] + w[syp] + w[sxmyp];

        // mu=1: backward plaquette (containing U_1†(s))
        double re_P_bwd_1 = std::real(P_fwd);  // same plaquette value as mu=0 forward
        double w_bwd_1 = w[s] + w[sxp] + w[sxpyp] + w[syp];

        force[1][s] += 0.5 * (re_P_fwd_1 * w_fwd_1 - re_P_bwd_1 * w_bwd_1);
    }
}

// Compute the clover derivative contribution for e/o Schur complement force.
// This handles the dD_oo/dA and d(D_ee⁻¹)/dA terms that are missing from
// the hopping-only eo_fermion_force for clover (c_sw != 0).
// Also includes the log-det force from -2 log det(D_ee).
// Accumulates into force (does not zero it).
static void eo_clover_force(const DiracOp& D, const EvenOddDiracOp& eoD,
                             const Vec& x_o, const Vec& y_o,
                             std::array<RVec, 2>& force) {
    if (D.c_sw == 0.0) return;
    const Lattice& lat = D.lat;
    int n_half = 2 * lat.V_half;
    double csw = D.c_sw;
    double diag = 2.0 * D.r + D.mass;

    // Precompute vectors needed for the clover weights
    // a_e = D_ee⁻¹ D_oe x_o (even-site reconstruction)
    Vec tmp_e(n_half);
    eoD.apply_oe(x_o, tmp_e);
    Vec a_e(n_half);
    eoD.apply_ee_inv(tmp_e, a_e);

    // w_e = D_ee⁻¹ D_eo† y_o = D_ee⁻¹ γ₅ D_oe(γ₅ y_o)
    Vec g5y(n_half);
    #pragma omp parallel for schedule(static) if(lat.V_half > OMP_MIN_SIZE/8)
    for (int i = 0; i < lat.V_half; i++) {
        g5y[2*i]   =  y_o[2*i];
        g5y[2*i+1] = -y_o[2*i+1];
    }
    Vec hop_e(n_half);
    eoD.apply_oe(g5y, hop_e);
    Vec g5hop(n_half);
    #pragma omp parallel for schedule(static) if(lat.V_half > OMP_MIN_SIZE/8)
    for (int i = 0; i < lat.V_half; i++) {
        g5hop[2*i]   =  hop_e[2*i];
        g5hop[2*i+1] = -hop_e[2*i+1];
    }
    Vec w_e(n_half);
    eoD.apply_ee_inv(g5hop, w_e);

    // Build the combined clover weight at each site:
    // - Odd sites: (c_sw/4) × Re(y_o† σ₃ x_o) from dD_oo/dA
    // - Even sites: -(c_sw/4) × Re(w_e† σ₃ D_ee⁻¹ a_e) from d(D_ee⁻¹)/dA
    //               + (1/2) × logdet weight from -2 log det(D_ee)
    RVec w(lat.V, 0.0);

    // Odd-site clover weight (from dD_oo/dA)
    #pragma omp parallel for schedule(static) if(lat.V_half > OMP_MIN_SIZE/8)
    for (int io = 0; io < lat.V_half; io++) {
        int s = lat.odd_sites[io];
        w[s] = (csw / 4.0) * std::real(
            std::conj(y_o[2*io]) * x_o[2*io] -
            std::conj(y_o[2*io+1]) * x_o[2*io+1]);
    }

    // Even-site weights (from d(D_ee⁻¹)/dA + logdet)
    #pragma omp parallel for schedule(static) if(lat.V_half > OMP_MIN_SIZE/8)
    for (int ie = 0; ie < lat.V_half; ie++) {
        int e = lat.even_sites[ie];
        double C = 0.5 * csw * D.clover_field[e];
        double dee_inv_0 = 1.0 / (diag + C);
        double dee_inv_1 = 1.0 / (diag - C);

        // d(D_ee⁻¹)/dA term in Schur complement: +D_oe D_ee⁻¹ (dD_ee/dA) D_ee⁻¹ D_eo
        // The force contribution: +Re(w_e† D_ee⁻¹ σ₃ a_e) × (c_sw/2) × (1/4)
        // = +(c_sw/4) × Re(w_e₀* dee_inv_0 a_e₀ - w_e₁* dee_inv_1 a_e₁)
        double w_dee_inv = +(csw / 4.0) * std::real(
            std::conj(w_e[2*ie]) * dee_inv_0 * a_e[2*ie] -
            std::conj(w_e[2*ie+1]) * dee_inv_1 * a_e[2*ie+1]);

        // Log-det weight: (1/2) × (-c_sw × C / ((diag+C)(diag-C)))
        double w_logdet = 0.5 * (-csw * C * dee_inv_0 * dee_inv_1);

        w[e] = w_dee_inv + w_logdet;
    }

    // Now compute the force using the same plaquette enumeration as fermion_force_clover
    // but with the combined weight w[] (which has both odd and even contributions)
    #pragma omp parallel for schedule(static) if(lat.V > OMP_MIN_SIZE/4)
    for (int s = 0; s < lat.V; s++) {
        int sxp = lat.xp(s), sxm = lat.xm(s);
        int syp = lat.yp(s), sym = lat.ym(s);
        int sxpym = lat.ym(sxp), sxmyp = lat.yp(sxm);
        int sxmym = lat.ym(sxm);
        int sxpyp = lat.yp(sxp);

        // mu=0
        cx P_fwd = D.gauge.U[0][s] * D.gauge.U[1][sxp]
                 * std::conj(D.gauge.U[0][syp]) * std::conj(D.gauge.U[1][s]);
        double re_P_fwd = std::real(P_fwd);
        double w_fwd = w[s] + w[sxp] + w[sxpyp] + w[syp];

        cx P_bwd = D.gauge.U[0][sym] * D.gauge.U[1][sxpym]
                 * std::conj(D.gauge.U[0][s]) * std::conj(D.gauge.U[1][sym]);
        double re_P_bwd = std::real(P_bwd);
        double w_bwd = w[sym] + w[sxpym] + w[sxp] + w[s];

        force[0][s] += re_P_fwd * w_fwd - re_P_bwd * w_bwd;

        // mu=1
        cx P_fwd_1 = D.gauge.U[0][sxm] * D.gauge.U[1][s]
                   * std::conj(D.gauge.U[0][sxmyp]) * std::conj(D.gauge.U[1][sxm]);
        double re_P_fwd_1 = std::real(P_fwd_1);
        double w_fwd_1 = w[sxm] + w[s] + w[syp] + w[sxmyp];

        double re_P_bwd_1 = std::real(P_fwd);
        double w_bwd_1 = w[s] + w[sxp] + w[sxpyp] + w[syp];

        force[1][s] += re_P_fwd_1 * w_fwd_1 - re_P_bwd_1 * w_bwd_1;
    }
}

// ---------------------------------------------------------------
//  Even-odd Schur complement force
//  Following Chroma eoprec_logdet_linop.h:114-157 and
//  two_flavor_monomial_w.h:618-633
//
//  F = -[schur_deriv(y_o, x_o) + schur_deriv(γ₅y_o, γ₅x_o)] - 2×logdet_force
//
//  where schur_deriv computes chi† (dM/dA) psi with 4 terms:
//    T1: +derivOddOddLinOp     (clover at odd sites)
//    T2: -derivOddEvenLinOp    (hopping oe derivative)
//    T3: +derivEvenEvenLinOp   (clover at even sites, chain rule)
//    T4: -derivEvenOddLinOp    (hopping eo derivative)
// ---------------------------------------------------------------
void schur_deriv_plus(const DiracOp& D, const EvenOddDiracOp& eoD,
                              const Vec& chi_o, const Vec& psi_o,
                              std::array<RVec, 2>& force) {
    const Lattice& lat = D.lat;
    int n_half = 2 * lat.V_half;

    // Pre-compute (Chroma lines 138-147):
    // tmp2 = A_ee⁻¹ D_eo psi  (even-site reconstruction of psi)
    Vec tmp1_e(n_half), tmp2(n_half);
    eoD.apply_oe(psi_o, tmp1_e);
    eoD.apply_ee_inv(tmp1_e, tmp2);

    // tmp3 = (A_ee†)⁻¹ D_oe† chi  (even-site reconstruction of chi via adjoint)
    // D_oe† chi = γ₅ D_eo(γ₅ chi) (hopping is γ₅-Hermitian regardless of twist)
    // (A_ee†)⁻¹ = A_ee(-μ)⁻¹ (adjoint inverse negates the twist)
    Vec g5chi(n_half);
    #pragma omp parallel for schedule(static) if(lat.V_half > OMP_MIN_SIZE/8)
    for (int i = 0; i < lat.V_half; i++) {
        g5chi[2*i] = chi_o[2*i]; g5chi[2*i+1] = -chi_o[2*i+1];
    }
    Vec hop_e(n_half);
    eoD.apply_oe(g5chi, hop_e);
    Vec g5hop(n_half);
    #pragma omp parallel for schedule(static) if(lat.V_half > OMP_MIN_SIZE/8)
    for (int i = 0; i < lat.V_half; i++) {
        g5hop[2*i] = hop_e[2*i]; g5hop[2*i+1] = -hop_e[2*i+1];
    }
    Vec tmp3(n_half);
    eoD.apply_diag_inv_impl(g5hop, tmp3, lat.even_sites, -D.mu_t);

    // Terms 1 & 3: clover diagonal derivatives (only for c_sw != 0)
    if (D.c_sw != 0.0) {
        RVec w(lat.V, 0.0);
        // Term 1: +derivOddOddLinOp — weight at odd sites = Re(chi† σ₃ psi)
        #pragma omp parallel for schedule(static) if(lat.V_half > OMP_MIN_SIZE/8)
        for (int io = 0; io < lat.V_half; io++) {
            int s = lat.odd_sites[io];
            w[s] = std::real(std::conj(chi_o[2*io]) * psi_o[2*io]
                           - std::conj(chi_o[2*io+1]) * psi_o[2*io+1]);
        }
        // Term 3: +derivEvenEvenLinOp — weight at even sites = Re(tmp3† σ₃ tmp2)
        #pragma omp parallel for schedule(static) if(lat.V_half > OMP_MIN_SIZE/8)
        for (int ie = 0; ie < lat.V_half; ie++) {
            int e = lat.even_sites[ie];
            w[e] = std::real(std::conj(tmp3[2*ie]) * tmp2[2*ie]
                           - std::conj(tmp3[2*ie+1]) * tmp2[2*ie+1]);
        }
        // Insert: factor = (c_sw/2) × (1/4) = c_sw/8
        // dA/dU = (c_sw/2) σ₃ dF_01/dA, and dF_01/dA = (1/4) d(Im(Q))/dA
        clover_deriv_insert(D, w, D.c_sw / 8.0, force);
    }

    // Terms 2 & 4: hopping derivatives
    // T2: -chi† D'_oe tmp2  → hopping force of chi_full(odd=chi, even=tmp3) × psi_full(even=tmp2, odd=psi)
    // T4: -tmp3† D'_eo psi  → same hopping force (scatter + hopping_force)
    Vec chi_full = eoD.scatter(tmp3, chi_o);
    Vec psi_full = eoD.scatter(tmp2, psi_o);
    std::array<RVec, 2> ff_hop;
    hopping_force_bilinear(D, chi_full, psi_full, ff_hop);
    // hopping_force_bilinear gives 2× because it sums fwd+bwd at each link.
    // Terms 2&4 each contribute once, so divide by 2.
    #pragma omp parallel for collapse(2) schedule(static)
    for (int mu = 0; mu < 2; mu++)
        for (int s = 0; s < lat.V; s++)
            force[mu][s] -= 0.5 * ff_hop[mu][s];
}

void eo_fermion_force(const DiracOp& D, const EvenOddDiracOp& eoD,
                      const Vec& x_o, const Vec& y_o,
                      std::array<RVec, 2>& force) {
    const Lattice& lat = D.lat;
    int n_half = 2 * lat.V_half;
    force[0].assign(lat.V, 0.0);
    force[1].assign(lat.V, 0.0);

    // Chroma two_flavor_monomial_w.h:618-633:
    // F = -[M.deriv(X, Y, MINUS) + M.deriv(Y, X, PLUS)]
    //
    // M.deriv(chi, psi, PLUS) = chi† dM/dA psi  → schur_deriv_plus(chi, psi)
    // M.deriv(chi, psi, MINUS) = chi† dM†/dA psi
    //   For γ₅-hermitian M: M† = γ₅ M γ₅
    //   chi† dM†/dA psi = chi† γ₅ dM/dA γ₅ psi = schur_deriv_plus(γ₅chi, γ₅psi)

    // Force = -dS/dA where S = φ†(M†M)⁻¹φ
    // dS/dA = -x†d(M†M)/dA x = -2Re(y†dM/dA x) where y = Mx
    // So force = -dS/dA = +2Re(y†dM/dA x)
    //
    // d(x†M†Mx)/dA = 2Re(y†dM/dA x) where y = Mx.
    // schur_deriv_plus gives Re(y†dM/dA x), so multiply by 2.
    schur_deriv_plus(D, eoD, y_o, x_o, force);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int mu = 0; mu < 2; mu++)
        for (int s = 0; s < lat.V; s++)
            force[mu][s] *= 2.0;

    // Log-det: +2 × derivLogDetEvenEven (in our force convention: -d(-2 log det)/dA = +2 d(log det)/dA)
    if (D.c_sw != 0.0) {
        std::array<RVec, 2> ff_logdet;
        logdet_ee_force(D, ff_logdet);
        #pragma omp parallel for collapse(2) schedule(static)
        for (int mu = 0; mu < 2; mu++)
            for (int s = 0; s < lat.V; s++)
                force[mu][s] += ff_logdet[mu][s];
    }
}

bool verify_forces(const GaugeField& g, double beta, double mass, double wilson_r,
                   int max_iter, double tol, double c_sw, bool use_eo,
                   double err_threshold, double mu_t) {
    const Lattice& lat = g.lat;
    double eps = 1e-5;
    std::mt19937 rng(12345);
    int n_tests = 20;  // test 20 links for thorough coverage

    std::string label = use_eo ? "E/O" : "Full";
    std::cout << "\n=== Force Verification (" << label
              << ", c_sw=" << c_sw << ", mu_t=" << mu_t << ") ===\n";

    DiracOp D0(lat, g, mass, wilson_r, c_sw, mu_t);

    // --- Gauge force (same for both full and e/o) ---
    std::array<RVec, 2> gf_ana;
    gauge_force(g, beta, gf_ana);

    // --- Fermion force: standard D†D or Schur complement ---
    Vec phi;                    // pseudofermion
    Vec phi_o;                  // odd-site pseudofermion (e/o only)
    std::array<RVec, 2> ff_ana;

    if (!use_eo) {
        // Standard: S_F = φ†(D†D)⁻¹φ
        Vec eta = random_vec(lat.ndof, rng);
        phi.resize(lat.ndof);
        D0.apply_dag(eta, phi);
        OpApply A0 = [&D0](const Vec& src, Vec& dst) { D0.apply_DdagD(src, dst); };
        auto res0 = cg_solve(A0, lat.ndof, phi, max_iter, tol);
        fermion_force(D0, res0.solution, ff_ana);
    } else {
        // E/O: S_eo = φ_o†(M†M)⁻¹φ_o - 2 log det(D_ee)
        EvenOddDiracOp eoD0(D0);
        int n_half = 2 * lat.V_half;
        Vec eta = random_vec(n_half, rng);
        phi_o.resize(n_half);
        eoD0.apply_schur_dag(eta, phi_o);
        OpApply A0 = [&eoD0](const Vec& src, Vec& dst) { eoD0.apply_schur_dag_schur(src, dst); };
        auto res0 = cg_solve(A0, n_half, phi_o, max_iter, tol);
        Vec y_o(n_half);
        eoD0.apply_schur(res0.solution, y_o);
        eo_fermion_force(D0, eoD0, res0.solution, y_o, ff_ana);
    }

    double max_gf_err = 0, max_ff_err = 0;
    double max_gf_val = 0, max_ff_val = 0;

    // Lambda to compute the action for a given gauge field
    auto compute_fermion_action = [&](const GaugeField& gf) -> double {
        DiracOp Dp(lat, gf, mass, wilson_r, c_sw, mu_t);
        if (!use_eo) {
            OpApply Ap = [&Dp](const Vec& s, Vec& d) { Dp.apply_DdagD(s, d); };
            auto r = cg_solve(Ap, lat.ndof, phi, max_iter, tol);
            return std::real(dot(phi, r.solution));
        } else {
            EvenOddDiracOp eoDp(Dp);
            int n_half = 2 * lat.V_half;
            OpApply Ap = [&eoDp](const Vec& s, Vec& d) { eoDp.apply_schur_dag_schur(s, d); };
            auto r = cg_solve(Ap, n_half, phi_o, max_iter, tol);
            double S = std::real(dot(phi_o, r.solution));
            if (c_sw != 0.0) S -= 2.0 * eoDp.log_det_ee();
            return S;
        }
    };

    for (int test = 0; test < n_tests; test++) {
        int mu = test % 2;
        int s = (test * 7 + test / 2 * 13) % lat.V;

        GaugeField g_plus(lat);
        g_plus.U[0] = g.U[0]; g_plus.U[1] = g.U[1];
        g_plus.U[mu][s] *= std::exp(cx(0, eps));

        GaugeField g_minus(lat);
        g_minus.U[0] = g.U[0]; g_minus.U[1] = g.U[1];
        g_minus.U[mu][s] *= std::exp(cx(0, -eps));

        // Gauge force check
        double sg_plus = gauge_action(g_plus, beta);
        double sg_minus = gauge_action(g_minus, beta);
        double gf_num = -(sg_plus - sg_minus) / (2 * eps);
        double gf_err = std::abs(gf_ana[mu][s] - gf_num);
        max_gf_err = std::max(max_gf_err, gf_err);
        max_gf_val = std::max(max_gf_val, std::abs(gf_ana[mu][s]));

        // Fermion force check
        double sf_plus = compute_fermion_action(g_plus);
        double sf_minus = compute_fermion_action(g_minus);
        double ff_num = -(sf_plus - sf_minus) / (2 * eps);
        double ff_err = std::abs(ff_ana[mu][s] - ff_num);
        max_ff_err = std::max(max_ff_err, ff_err);
        max_ff_val = std::max(max_ff_val, std::abs(ff_ana[mu][s]));

        std::cout << "  link(" << mu << "," << std::setw(3) << s << "): "
                  << "gf_ana=" << std::setw(12) << gf_ana[mu][s]
                  << " gf_num=" << std::setw(12) << gf_num
                  << " | ff_ana=" << std::setw(12) << ff_ana[mu][s]
                  << " ff_num=" << std::setw(12) << ff_num << "\n";
    }

    double gf_rel = max_gf_err / std::max(max_gf_val, 1e-30);
    double ff_rel = max_ff_err / std::max(max_ff_val, 1e-30);
    bool gf_pass = gf_rel < err_threshold;
    bool ff_pass = ff_rel < err_threshold;

    std::cout << "Gauge force:   max_err=" << std::scientific << max_gf_err
              << " rel=" << gf_rel << (gf_pass ? " PASS" : " FAIL") << "\n";
    std::cout << "Fermion force: max_err=" << std::scientific << max_ff_err
              << " rel=" << ff_rel << (ff_pass ? " PASS" : " FAIL") << "\n";

    return gf_pass && ff_pass;
}

