#include "hmc.h"
#include "eigensolver.h"
#include "smoother.h"
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

    // Hopping term contribution (standard Wilson)
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

    // Clover term contribution
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
    if (D.c_sw != 0.0) {
        // Precompute σ_3 weight at each site: w(x) = Re(χ_0* X_0 - χ_1* X_1)
        RVec w(lat.V);
        for (int s = 0; s < lat.V; s++)
            w[s] = std::real(std::conj(chi[2*s]) * X[2*s]
                           - std::conj(chi[2*s+1]) * X[2*s+1]);

        double csw = D.c_sw;

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

// ========================================
// Even-odd preconditioned HMC functions
// ========================================

void generate_pseudofermion_eo(const DiracOp& D, std::mt19937& rng, Vec& phi_even) {
    // Generate η_e ~ N(0,1) on even sites, then φ_e = D̂ η_e
    Vec eta_e = random_vec(D.lat.ndof2, rng);
    phi_even.resize(D.lat.ndof2);
    D.apply_schur(eta_e, phi_even);
}

FermionActionResult fermion_action_eo(const DiracOp& D, const Vec& phi_even,
                                       int max_iter, double tol) {
    // Solve D̂†D̂ x_e = φ_e on even parity
    OpApply A = [&D](const Vec& src, Vec& dst) { D.apply_schur_DdagD(src, dst); };
    auto res = cg_solve(A, D.lat.ndof2, phi_even, max_iter, tol);

    // Reconstruct full solution: x_o = -D_oo^{-1} D_oe x_e
    Vec x_odd;
    D.reconstruct_odd(res.solution, x_odd);

    // Scatter to full lattice
    Vec X_full(D.lat.ndof, cx(0));
    D.scatter_even(res.solution, X_full);
    D.scatter_odd(x_odd, X_full);

    // Action S_f = Re(φ_e† x_e)
    // Note: this is the Schur complement action, NOT divided by anything.
    // The 2-flavour partition function with EO is:
    //   |det D|² = |det D_ee|² |det D_oo|² |det D̂|²
    // The pseudofermion action for D̂ is:
    //   S_pf = φ_e† (D̂†D̂)^{-1} φ_e = Re(φ_e† x_e)
    double act = std::real(dot(phi_even, res.solution));

    return {act, std::move(X_full), res.iterations};
}

void fermion_force_eo(const DiracOp& D, const Vec& x_even, const Vec& Y_even,
                      std::array<RVec, 2>& force) {
    // Compute dS_eo/dU where S_eo = φ_e†(D̂†D̂)^{-1}φ_e = x_e†D̂†D̂ x_e
    // Force F(s,μ) = -∂S_eo/∂θ(s,μ) = -2 Re[Y_e†(∂D̂/∂θ)x_e]
    //
    // For Wilson (no clover): D̂ = D_ee - D_eo D_oo^{-1} D_oe
    // ∂D̂/∂θ = -(∂D_eo/∂θ) D_oo^{-1} D_oe - D_eo D_oo^{-1} (∂D_oe/∂θ)
    //
    // We need two odd-site intermediates:
    //   p_o = D_oo^{-1} D_oe x_e     (odd part of the reconstructed solution)
    //   q_o = D_oo^{-1} D_oe^dag Y_e (adjoint hop of Y_e to odd, with D_oo^{-1})
    //
    // For γ₅-hermiticity: D_oe^dag = γ₅ D_eo γ₅
    // So q_o = D_oo^{-1} γ₅ D_eo γ₅ Y_e
    //
    // Then the force at link (s,μ) is:
    //   F = 2Re[Y_e† (∂D_eo/∂θ) p_o] + 2Re[q_o† (∂D_oe/∂θ) x_e]
    //     (with appropriate signs from D̂ = D_ee - ...)

    const Lattice& lat = D.lat;
    int V2 = lat.V2;
    force[0].assign(lat.V, 0.0);
    force[1].assign(lat.V, 0.0);

    // Compute p_o = D_oo^{-1} D_oe x_e
    Vec D_oe_x;
    D.apply_oe(x_even, D_oe_x);
    Vec p_odd(lat.ndof2);
    D.invert_diag(D_oe_x, p_odd, false);

    // Compute q_o = D_oo^{-1} γ₅ D_eo γ₅ Y_e
    // First: g5_Y = γ₅ Y_e
    Vec g5_Y(lat.ndof2);
    for (int h = 0; h < V2; h++) { g5_Y[2*h] = Y_even[2*h]; g5_Y[2*h+1] = -Y_even[2*h+1]; }
    // D_eo applied to g5_Y (this treats g5_Y as an odd vector, producing even)
    // Wait — D_eo goes odd→even. We need D_eo^dag which maps even→odd.
    // D_eo^dag = γ₅_o D_oe γ₅_e. So: q_o = D_oo^{-1} γ₅ D_oe γ₅ Y_e
    // But D_oe maps even→odd. So D_oe(γ₅ Y_e) gives odd, then γ₅, then D_oo^{-1}.
    Vec D_oe_g5Y;
    D.apply_oe(g5_Y, D_oe_g5Y);
    // Apply γ₅ to the odd result
    Vec g5_D_oe_g5Y(lat.ndof2);
    for (int h = 0; h < V2; h++) { g5_D_oe_g5Y[2*h] = D_oe_g5Y[2*h]; g5_D_oe_g5Y[2*h+1] = -D_oe_g5Y[2*h+1]; }
    Vec q_odd(lat.ndof2);
    D.invert_diag(g5_D_oe_g5Y, q_odd, false);

    // Now the force. For each link (s,μ), the D_eo/D_oe hopping terms give:
    //
    // Term 1: -Y_e†(∂D_eo/∂θ)p_o  (the ∂D_eo/∂θ term in dD̂/dθ, with overall - from D̂ = D_ee - ...)
    //   This involves: Y_e at even site, p_o at odd neighbor (or vice versa)
    //   ∂D_eo/∂θ(s,μ) = derivative of the hop from odd s+μ to even s through link U_μ(s)
    //
    // Term 2: -q_o†(∂D_oe/∂θ)x_e  (the D_eo D_oo^{-1} ∂D_oe/∂θ term)
    //   This involves: q_o at odd site, x_e at even neighbor
    //
    // The outer product structure is IDENTICAL to the full fermion_force,
    // but using (Y_e, q_o) as "chi" and (x_e, p_o) as "X",
    // with the fields living on their respective parities.
    //
    // Actually, let me construct full-lattice vectors:
    //   X_full = (x_e on even, p_o on odd)
    //   chi_full = (Y_e on even, q_o on odd)
    // Then the force is the same outer product as fermion_force(D, X_full),
    // but with chi_full instead of D*X_full!

    // Scatter to full lattice
    Vec X_full(lat.ndof, cx(0));
    Vec chi_full(lat.ndof, cx(0));
    for (int h = 0; h < V2; h++) {
        int se = lat.even_sites[h];
        X_full[2*se] = x_even[2*h]; X_full[2*se+1] = x_even[2*h+1];
        chi_full[2*se] = Y_even[2*h]; chi_full[2*se+1] = Y_even[2*h+1];

        int so = lat.odd_sites[h];
        X_full[2*so] = p_odd[2*h]; X_full[2*so+1] = p_odd[2*h+1];
        chi_full[2*so] = q_odd[2*h]; chi_full[2*so+1] = q_odd[2*h+1];
    }

    // Now use the SAME outer product structure as fermion_force,
    // but with chi_full instead of D*X_full.
    // The overall sign is NEGATIVE because D̂ = D_ee - D_eo D_oo^{-1} D_oe,
    // and the outer products compute d(D_eo D_oo^{-1} D_oe)/dθ (without the minus).
    // The force F = -∂S/∂θ = -2Re[Y†(∂D̂/∂θ)x] = +2Re[Y†(d(hop chain)/dθ)x].
    // But fermion_force convention returns -∂S/∂θ directly, so we negate.
    cx ii(0,1);
    for (int s = 0; s < lat.V; s++) {
        cx chi_s0 = chi_full[2*s], chi_s1 = chi_full[2*s+1];
        cx x_s0 = X_full[2*s], x_s1 = X_full[2*s+1];

        // mu=0 (x-direction)
        {
            int sf = lat.xp(s);
            cx u = D.gauge.U[0][s];
            cx ud = std::conj(u);

            cx ux0 = u * X_full[2*sf], ux1 = u * X_full[2*sf+1];
            cx fwd0 = D.r * ux0 - ux1;
            cx fwd1 = -ux0 + D.r * ux1;
            cx c_fwd = std::conj(chi_s0) * fwd0 + std::conj(chi_s1) * fwd1;

            cx udx0 = ud * x_s0, udx1 = ud * x_s1;
            cx bwd0 = D.r * udx0 + udx1;
            cx bwd1 = udx0 + D.r * udx1;
            cx c_bwd = std::conj(chi_full[2*sf]) * bwd0 + std::conj(chi_full[2*sf+1]) * bwd1;

            force[0][s] = std::real(ii * c_bwd - ii * c_fwd);
        }

        // mu=1 (y-direction)
        {
            int sf = lat.yp(s);
            cx u = D.gauge.U[1][s];
            cx ud = std::conj(u);

            cx ux0 = u * X_full[2*sf], ux1 = u * X_full[2*sf+1];
            cx fwd0 = D.r * ux0 + ii * ux1;
            cx fwd1 = -ii * ux0 + D.r * ux1;
            cx c_fwd = std::conj(chi_s0) * fwd0 + std::conj(chi_s1) * fwd1;

            cx udx0 = ud * x_s0, udx1 = ud * x_s1;
            cx bwd0 = D.r * udx0 - ii * udx1;
            cx bwd1 = ii * udx0 + D.r * udx1;
            cx c_bwd = std::conj(chi_full[2*sf]) * bwd0 + std::conj(chi_full[2*sf+1]) * bwd1;

            force[1][s] = std::real(ii * c_bwd - ii * c_fwd);
        }
    }

    // Apply overall minus sign: D̂ = D_ee - (hop chain), so ∂D̂/∂θ = -∂(hop chain)/∂θ.
    // The outer products above compute +∂(hop chain)/∂θ contribution.
    // F = -∂S/∂θ = -2Re[Y†(∂D̂/∂θ)x] = +2Re[Y†(∂(hop chain)/∂θ)x] = what we computed above.
    // But wait — the non-EO fermion_force also computes F = -∂S/∂θ and returns positive values
    // that match -(S+-S-)/(2eps). Let me check the sign convention more carefully.
    //
    // Actually, the numerical force computes: ff_num = -(S+ - S-)/(2eps) = -∂S/∂θ.
    // The analytical force should equal ff_num. Currently ff_ana = -ff_num.
    // So negate:
    for (int mu = 0; mu < 2; mu++)
        for (int s = 0; s < lat.V; s++)
            force[mu][s] = -force[mu][s];

    // Note: no clover force term here — for clover, dD_oo/dU ≠ 0
    // and would require additional sigma-trace contributions.
}

void verify_forces_eo(const GaugeField& g, double beta, double mass, double wilson_r,
                      int max_iter, double tol, double c_sw) {
    const Lattice& lat = g.lat;
    double eps = 1e-5;
    std::mt19937 rng(12345);

    DiracOp D0(lat, g, mass, wilson_r, c_sw);
    Vec phi_e;
    generate_pseudofermion_eo(D0, rng, phi_e);

    // Solve D̂†D̂ x_e = phi_e to get x_even, then compute Y_e = D̂ x_e
    OpApply A0 = [&D0](const Vec& src, Vec& dst) { D0.apply_schur_DdagD(src, dst); };
    auto cg_res = cg_solve(A0, lat.ndof2, phi_e, max_iter, tol);
    Vec x_even = cg_res.solution;
    Vec Y_even(lat.ndof2);
    D0.apply_schur(x_even, Y_even);

    // Analytical EO force
    std::array<RVec, 2> ff_ana;
    fermion_force_eo(D0, x_even, Y_even, ff_ana);

    std::array<RVec, 2> gf_ana;
    gauge_force(g, beta, gf_ana);

    double max_gf_err = 0, max_ff_err = 0;
    double max_gf_val = 0, max_ff_val = 0;

    std::cout << "\n=== EO Force Verification ===\n";
    for (int test = 0; test < 10; test++) {
        int mu = test % 2;
        int s = test * 7 % lat.V;

        GaugeField g_plus(lat); g_plus.U[0] = g.U[0]; g_plus.U[1] = g.U[1];
        g_plus.U[mu][s] *= std::exp(cx(0, eps));
        GaugeField g_minus(lat); g_minus.U[0] = g.U[0]; g_minus.U[1] = g.U[1];
        g_minus.U[mu][s] *= std::exp(cx(0, -eps));

        // Numerical gauge force
        double sg_p = gauge_action(g_plus, beta), sg_m = gauge_action(g_minus, beta);
        double gf_num = -(sg_p - sg_m) / (2 * eps);
        double gf_err = std::abs(gf_ana[mu][s] - gf_num);
        max_gf_err = std::max(max_gf_err, gf_err);
        max_gf_val = std::max(max_gf_val, std::abs(gf_ana[mu][s]));

        // Numerical fermion force (using EO action)
        DiracOp Dp(lat, g_plus, mass, wilson_r, c_sw);
        DiracOp Dm(lat, g_minus, mass, wilson_r, c_sw);
        auto rp = fermion_action_eo(Dp, phi_e, max_iter, tol);
        auto rm = fermion_action_eo(Dm, phi_e, max_iter, tol);
        double ff_num = -(rp.action - rm.action) / (2 * eps);

        double ff_err = std::abs(ff_ana[mu][s] - ff_num);
        max_ff_err = std::max(max_ff_err, ff_err);
        max_ff_val = std::max(max_ff_val, std::abs(ff_ana[mu][s]));

        std::cout << "  link(" << mu << "," << s << "): "
                  << "gf_ana=" << std::setw(10) << gf_ana[mu][s]
                  << " gf_num=" << std::setw(10) << gf_num
                  << " | ff_ana=" << std::setw(10) << ff_ana[mu][s]
                  << " ff_num=" << std::setw(10) << ff_num << "\n";
    }
    std::cout << "EO Gauge force: max_err=" << max_gf_err
              << " rel=" << max_gf_err / std::max(max_gf_val, 1e-30) << "\n";
    std::cout << "EO Fermion force: max_err=" << max_ff_err
              << " rel=" << max_ff_err / std::max(max_ff_val, 1e-30) << "\n";
}

HMCResult hmc_trajectory_eo(
    GaugeField& gauge, const Lattice& lat, double mass, double wilson_r,
    const HMCParams& params, std::mt19937& rng)
{
    double dt = params.tau / params.n_steps;
    double c_sw = params.c_sw;
    int total_cg = 0;

    GaugeField gauge_old(lat);
    gauge_old.U[0] = gauge.U[0]; gauge_old.U[1] = gauge.U[1];

    MomentumField mom(lat);
    mom.randomise(rng);

    // Generate EO pseudofermion
    DiracOp D_init(lat, gauge, mass, wilson_r, c_sw);
    Vec phi_e;
    generate_pseudofermion_eo(D_init, rng, phi_e);

    // Initial Hamiltonian
    double H_init = mom.kinetic_energy() + gauge_action(gauge, params.beta);
    { auto res = fermion_action_eo(D_init, phi_e, params.cg_maxiter, params.cg_tol);
      H_init += res.action; total_cg += res.cg_iters; }

    // Leapfrog: half-kick → [drift → kick]^{N-1} → drift → half-kick
    auto kick = [&](double step) {
        DiracOp D(lat, gauge, mass, wilson_r, c_sw);
        std::array<RVec, 2> gf, ff;
        gauge_force(gauge, params.beta, gf);

        // EO CG solve
        OpApply A = [&D](const Vec& src, Vec& dst) { D.apply_schur_DdagD(src, dst); };
        auto cg_res = cg_solve(A, lat.ndof2, phi_e, params.cg_maxiter, params.cg_tol);
        total_cg += cg_res.iterations;

        // Compute Y_e = D̂ x_e for the EO force
        Vec Y_e(lat.ndof2);
        D.apply_schur(cg_res.solution, Y_e);

        // EO fermion force
        fermion_force_eo(D, cg_res.solution, Y_e, ff);

        for (int mu = 0; mu < 2; mu++)
            for (int s = 0; s < lat.V; s++)
                mom.pi[mu][s] += step * (gf[mu][s] + ff[mu][s]);
    };
    auto drift = [&](double step) {
        for (int mu = 0; mu < 2; mu++)
            for (int s = 0; s < lat.V; s++)
                gauge.U[mu][s] *= std::exp(cx(0, step * mom.pi[mu][s]));
    };

    kick(0.5 * dt);
    for (int step = 0; step < params.n_steps - 1; step++) {
        drift(dt);
        kick(dt);
    }
    drift(dt);
    kick(0.5 * dt);

    // Final Hamiltonian
    double H_final = mom.kinetic_energy() + gauge_action(gauge, params.beta);
    { DiracOp D(lat, gauge, mass, wilson_r, c_sw);
      auto res = fermion_action_eo(D, phi_e, params.cg_maxiter, params.cg_tol);
      H_final += res.action; total_cg += res.cg_iters; }

    double dH = H_final - H_init;

    // Metropolis
    std::uniform_real_distribution<double> udist(0.0, 1.0);
    bool accepted = (udist(rng) < std::exp(-dH));
    if (!accepted) { gauge.U[0] = gauge_old.U[0]; gauge.U[1] = gauge_old.U[1]; }

    static int n_acc = 0, n_tot = 0;
    n_tot++; if (accepted) n_acc++;

    return {accepted, dH, (double)n_acc / n_tot, total_cg};
}

void verify_forces(const GaugeField& g, double beta, double mass, double wilson_r,
                   int max_iter, double tol, double c_sw) {
    const Lattice& lat = g.lat;
    double eps = 1e-5;
    std::mt19937 rng(12345);

    DiracOp D0(lat, g, mass, wilson_r, c_sw);
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

        DiracOp D_plus(lat, g_plus, mass, wilson_r, c_sw);
        DiracOp D_minus(lat, g_minus, mass, wilson_r, c_sw);
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
    double c_sw = params.c_sw;
    int total_cg = 0;

    GaugeField gauge_old(lat);
    gauge_old.U[0] = gauge.U[0];
    gauge_old.U[1] = gauge.U[1];

    MomentumField mom(lat);
    mom.randomise(rng);

    DiracOp D_init(lat, gauge, mass, wilson_r, c_sw);
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
        DiracOp D(lat, gauge, mass, wilson_r, c_sw);
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

        DiracOp D(lat, gauge, mass, wilson_r, c_sw);
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
        DiracOp D(lat, gauge, mass, wilson_r, c_sw);
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
        DiracOp D(lat, gauge, mass, wilson_r, c_sw);
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

// ---------------------------------------------------------------
//  Low-mode fermion force from eigenpairs of D†D
//  Computes X_low = Σ_i (v_i†φ / λ_i) v_i  (low-mode part of solution)
//  Then calls the standard fermion_force on X_low.
// ---------------------------------------------------------------
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

    DiracOp D_init(lat, gauge, mass, wilson_r, c_sw);
    Vec phi;
    generate_pseudofermion(D_init, rng, phi);

    // Update Dv cache for initial gauge
    defl.update_cache(D_init);

    // --- Compute initial Hamiltonian (exact, full CG) ---
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

    // === OUTER HALF-KICK (slow forces: gauge + high-mode fermion) ===
    // High-mode = full fermion force - low-mode force
    {
        auto t0 = Clock::now();
        DiracOp D(lat, gauge, mass, wilson_r, c_sw);
        std::array<RVec, 2> gf, ff_full, fl;
        gauge_force(gauge, params.beta, gf);

        OpApply A = [&D](const Vec& src, Vec& dst) { D.apply_DdagD(src, dst); };
        Vec x0 = defl.deflated_initial_guess(phi);
        CGResult res = cg_solve_x0(A, lat.ndof, phi, x0, params.cg_maxiter, params.cg_tol);
        total_cg += res.iterations;
        fermion_force(D, res.solution, ff_full);
        lowmode_fermion_force(D, defl, phi, fl);

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
            DiracOp D(lat, gauge, mass, wilson_r, c_sw);
            std::array<RVec, 2> fl;
            lowmode_fermion_force(D, defl, phi, fl);
            for (int mu = 0; mu < 2; mu++)
                for (int s = 0; s < lat.V; s++)
                    mom.pi[mu][s] += 0.5 * dt_inner * fl[mu][s];
            lowmode_evals++;
            lowmode_time += Dur(Clock::now() - t0).count();
        }

        // --- INNER LOOP ---
        for (int i = 0; i < params.n_inner; i++) {
            // Full gauge update
            for (int mu = 0; mu < 2; mu++)
                for (int s = 0; s < lat.V; s++)
                    gauge.U[mu][s] *= std::exp(cx(0, dt_inner * mom.pi[mu][s]));

            if (i < params.n_inner - 1) {
                // Full inner kick
                auto t0 = Clock::now();
                DiracOp D(lat, gauge, mass, wilson_r, c_sw);
                std::array<RVec, 2> fl;
                lowmode_fermion_force(D, defl, phi, fl);
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
            DiracOp D(lat, gauge, mass, wilson_r, c_sw);
            std::array<RVec, 2> fl;
            lowmode_fermion_force(D, defl, phi, fl);
            for (int mu = 0; mu < 2; mu++)
                for (int s = 0; s < lat.V; s++)
                    mom.pi[mu][s] += 0.5 * dt_inner * fl[mu][s];
            lowmode_evals++;
            lowmode_time += Dur(Clock::now() - t0).count();
        }

        // --- OUTER KICK (slow: gauge + high-mode = full - low) ---
        {
            auto t0 = Clock::now();
            DiracOp D(lat, gauge, mass, wilson_r, c_sw);
            std::array<RVec, 2> gf, ff_full, fl;
            gauge_force(gauge, params.beta, gf);

            OpApply A = [&D](const Vec& src, Vec& dst) { D.apply_DdagD(src, dst); };
            Vec x0 = defl.deflated_initial_guess(phi);
            CGResult res = cg_solve_x0(A, lat.ndof, phi, x0, params.cg_maxiter, params.cg_tol);
            total_cg += res.iterations;
            fermion_force(D, res.solution, ff_full);
            lowmode_fermion_force(D, defl, phi, fl);

            double kick = (o < params.n_outer - 1) ? dt_outer : 0.5 * dt_outer;
            for (int mu = 0; mu < 2; mu++)
                for (int s = 0; s < lat.V; s++)
                    mom.pi[mu][s] += kick * (gf[mu][s] + ff_full[mu][s] - fl[mu][s]);
            highmode_time += Dur(Clock::now() - t0).count();
        }
    }

    // --- Compute final Hamiltonian (exact, full CG) ---
    double H_final = mom.kinetic_energy() + gauge_action(gauge, params.beta);
    {
        DiracOp D(lat, gauge, mass, wilson_r, c_sw);
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

    return {accept, dH, total_cg, lowmode_evals, highmode_time, lowmode_time};
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
    const SparseCoarseOp& Ac_new)
{
    if (cdefl.eigvecs.empty()) return;
    OpApply op = Ac_new.as_op();
    auto rr = rr_evolve(op, cdefl.eigvecs, Ac_new.dim);
    cdefl.eigvecs = std::move(rr.eigvecs);
    cdefl.eigvals = std::move(rr.eigvals);
}

MGMultiScaleResult hmc_trajectory_mg_multiscale(
    GaugeField& gauge, const Lattice& lat, double mass, double wilson_r,
    const MGMultiScaleParams& params,
    CoarseDeflState& cdefl,
    Prolongator& P,
    std::function<Vec(const Vec&)>& mg_precond,
    std::mt19937& rng)
{
    using Clock = std::chrono::high_resolution_clock;
    using Dur = std::chrono::duration<double>;

    double dt_outer = params.tau / params.n_outer;
    double c_sw = params.c_sw;
    int total_cg = 0;
    int lowmode_evals = 0;
    double highmode_time = 0, lowmode_time = 0;

    // Save gauge for reject
    GaugeField gauge_old(lat);
    gauge_old.U[0] = gauge.U[0];
    gauge_old.U[1] = gauge.U[1];

    MomentumField mom(lat);
    mom.randomise(rng);

    DiracOp D_init(lat, gauge, mass, wilson_r, c_sw);
    Vec phi;
    generate_pseudofermion(D_init, rng, phi);

    // --- Initial Hamiltonian ---
    double H_init = mom.kinetic_energy() + gauge_action(gauge, params.beta);
    {
        OpApply A = [&D_init](const Vec& s, Vec& d) { D_init.apply_DdagD(s, d); };
        auto res = cg_solve_precond(A, lat.ndof, phi, mg_precond,
                                     params.cg_maxiter, params.cg_tol);
        H_init += std::real(dot(phi, res.solution));
        total_cg += res.iterations;
    }

    // ── Primitives ──

    auto compute_outer_force = [&](std::array<RVec, 2>& f_out) {
        auto t0 = Clock::now();
        DiracOp D(lat, gauge, mass, wilson_r, c_sw);
        std::array<RVec, 2> gf, ff_full, fl;
        gauge_force(gauge, params.beta, gf);
        OpApply A = [&D](const Vec& s, Vec& d) { D.apply_DdagD(s, d); };
        auto res = cg_solve_precond(A, lat.ndof, phi, mg_precond,
                                     params.cg_maxiter, params.cg_tol);
        total_cg += res.iterations;
        fermion_force(D, res.solution, ff_full);
        coarse_lowmode_force(D, cdefl, P, phi, fl, params.inner_smooth);
        f_out[0].resize(lat.V);
        f_out[1].resize(lat.V);
        for (int mu = 0; mu < 2; mu++)
            for (int s = 0; s < lat.V; s++)
                f_out[mu][s] = gf[mu][s] + ff_full[mu][s] - fl[mu][s];
        highmode_time += Dur(Clock::now() - t0).count();
    };

    auto kick_mom = [&](const std::array<RVec, 2>& f, double dt) {
        for (int mu = 0; mu < 2; mu++)
            for (int s = 0; s < lat.V; s++)
                mom.pi[mu][s] += dt * f[mu][s];
    };

    auto update_gauge = [&](double dt) {
        for (int mu = 0; mu < 2; mu++)
            for (int s = 0; s < lat.V; s++)
                gauge.U[mu][s] *= std::exp(cx(0, dt * mom.pi[mu][s]));
    };

    auto compute_inner_force = [&](std::array<RVec, 2>& fl) {
        auto t0 = Clock::now();
        DiracOp D(lat, gauge, mass, wilson_r, c_sw);
        coarse_lowmode_force(D, cdefl, P, phi, fl, params.inner_smooth);
        lowmode_evals++;
        lowmode_time += Dur(Clock::now() - t0).count();
    };

    // Track inner steps for periodic deflation refresh
    int inner_step_counter = 0;

    auto maybe_refresh_deflation = [&]() {
        if (params.defl_refresh > 0 && inner_step_counter > 0
            && inner_step_counter % params.defl_refresh == 0) {
            // Rebuild coarse operator for current gauge and RR-evolve eigenvectors
            DiracOp D_ref(lat, gauge, mass, wilson_r, c_sw);
            OpApply A_ref = [&D_ref](const Vec& s, Vec& d) { D_ref.apply_DdagD(s, d); };
            // Rebuild sparse coarse op stencil from current gauge
            SparseCoarseOp sac_tmp;
            sac_tmp.build(P, A_ref, lat.ndof);
            // RR evolve coarse eigenvectors on the new operator
            evolve_coarse_deflation(cdefl, sac_tmp);
        }
    };

    // ── Inner sub-integrator: N_inner leapfrog steps with low-mode + gauge force ──
    auto inner_integrator = [&](double dt_total) {
        int ni = params.n_inner;
        double dti = dt_total / ni;

        std::array<RVec, 2> fl;
        compute_inner_force(fl);
        kick_mom(fl, 0.5 * dti);
        for (int i = 0; i < ni; i++) {
            update_gauge(dti);
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
    double H_final = mom.kinetic_energy() + gauge_action(gauge, params.beta);
    {
        DiracOp D(lat, gauge, mass, wilson_r, c_sw);
        OpApply A = [&D](const Vec& s, Vec& d) { D.apply_DdagD(s, d); };
        auto res = cg_solve_precond(A, lat.ndof, phi, mg_precond,
                                     params.cg_maxiter, params.cg_tol);
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

    return {accept, dH, total_cg, lowmode_evals, highmode_time, lowmode_time};
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
