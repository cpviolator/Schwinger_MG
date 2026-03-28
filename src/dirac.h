#pragma once
#include "types.h"
#include "lattice.h"
#include "gauge.h"
#include <omp.h>

struct DiracOp {
    const Lattice& lat;
    const GaugeField& gauge;
    double mass;
    double r;
    double c_sw;       // clover coefficient (0 = pure Wilson)
    RVec clover_field; // precomputed F_01(s) at each site

    DiracOp(const Lattice& lat_, const GaugeField& g_, double m_,
            double r_ = 1.0, double csw_ = 0.0)
        : lat(lat_), gauge(g_), mass(m_), r(r_), c_sw(csw_)
    {
        if (c_sw != 0.0) compute_clover_field();
    }

    // Compute the clover field strength F_01(s) = Im(Q_01(s)) / 4
    // Q_01(s) = sum of 4 oriented plaquettes meeting at site s
    void compute_clover_field() {
        int V = lat.V;
        clover_field.resize(V);
        #pragma omp parallel for schedule(static) if(V > OMP_MIN_SIZE)
        for (int s = 0; s < V; s++) {
            int sxp = lat.xp(s), sxm = lat.xm(s);
            int syp = lat.yp(s), sym = lat.ym(s);
            // Upper-right: U_0(s) U_1(s+x) U_0†(s+y) U_1†(s)
            cx P_ur = gauge.U[0][s] * gauge.U[1][sxp]
                    * std::conj(gauge.U[0][syp]) * std::conj(gauge.U[1][s]);
            // Upper-left: U_1(s) U_0†(s+y-x) U_1†(s-x) U_0(s-x)
            int sxmyp = lat.yp(sxm);
            cx P_ul = gauge.U[1][s] * std::conj(gauge.U[0][sxmyp])
                    * std::conj(gauge.U[1][sxm]) * gauge.U[0][sxm];
            // Lower-left: U_0†(s-x) U_1†(s-x-y) U_0(s-x-y) U_1(s-y)
            int sxmym = lat.ym(sxm);
            cx P_ll = std::conj(gauge.U[0][sxm]) * std::conj(gauge.U[1][sxmym])
                    * gauge.U[0][sxmym] * gauge.U[1][sym];
            // Lower-right: U_1†(s-y) U_0(s-y) U_1(s+x-y) U_0†(s)
            int sxpym = lat.ym(sxp);
            cx P_lr = std::conj(gauge.U[1][sym]) * gauge.U[0][sym]
                    * gauge.U[1][sxpym] * std::conj(gauge.U[0][s]);

            cx Q = P_ur + P_ul + P_ll + P_lr;
            clover_field[s] = std::imag(Q) / 4.0;
        }
    }

    // Core hopping kernel: accumulate the 4 Wilson hops at site s into (d0, d1).
    // For apply(): all scales = 1 (default).
    // For apply_delta_D(): fwd_scale = (e^{iθ}-1), bwd_scale = (e^{-iθ}-1).
    // The fwd_scale multiplies U, the bwd_scale multiplies U† (NOT conjugated with U).
    static inline void accumulate_hops(
        cx& d0, cx& d1,
        int s, const Vec& src, const GaugeField& gauge, const Lattice& lat, double r,
        cx fwd_x = 1.0, cx bwd_x = 1.0,
        cx fwd_y = 1.0, cx bwd_y = 1.0)
    {
        cx ii(0,1);
        // mu=0 forward: fwd_x * U * src
        { int sf = lat.xp(s);
          cx u = fwd_x * gauge.U[0][s];
          cx up0 = u*src[2*sf], up1 = u*src[2*sf+1];
          d0 -= 0.5*(r*up0 - up1);
          d1 -= 0.5*(-up0 + r*up1); }
        // mu=0 backward: bwd_x * U† * src
        { int sb = lat.xm(s);
          cx ud = bwd_x * std::conj(gauge.U[0][sb]);
          cx udq0 = ud*src[2*sb], udq1 = ud*src[2*sb+1];
          d0 -= 0.5*(r*udq0 + udq1);
          d1 -= 0.5*(udq0 + r*udq1); }
        // mu=1 forward: fwd_y * U * src
        { int sf = lat.yp(s);
          cx u = fwd_y * gauge.U[1][s];
          cx up0 = u*src[2*sf], up1 = u*src[2*sf+1];
          d0 -= 0.5*(r*up0 + ii*up1);
          d1 -= 0.5*(-ii*up0 + r*up1); }
        // mu=1 backward: bwd_y * U† * src
        { int sb = lat.ym(s);
          cx ud = bwd_y * std::conj(gauge.U[1][sb]);
          cx udq0 = ud*src[2*sb], udq1 = ud*src[2*sb+1];
          d0 -= 0.5*(r*udq0 - ii*udq1);
          d1 -= 0.5*(ii*udq0 + r*udq1); }
    }

    void apply(const Vec& src, Vec& dst) const {
        int V = lat.V;
        bool has_clover = (c_sw != 0.0);
        #pragma omp parallel for schedule(static) if(V > OMP_MIN_SIZE)
        for (int s = 0; s < V; s++) {
            cx s0 = src[2*s], s1 = src[2*s+1];
            // Diagonal: mass + Wilson + clover
            // σ_01 = [[-1,0],[0,1]], so clover adds +C to ψ_0, -C to ψ_1
            // where C = (c_sw/2) × F_01(s)
            double diag = 2.0*r + mass;
            if (has_clover) {
                double C = 0.5 * c_sw * clover_field[s];
                dst[2*s]   = (diag + C) * s0;
                dst[2*s+1] = (diag - C) * s1;
            } else {
                dst[2*s]   = diag * s0;
                dst[2*s+1] = diag * s1;
            }

            accumulate_hops(dst[2*s], dst[2*s+1], s, src, gauge, lat, r);
        }
    }

    void apply_dag(const Vec& src, Vec& dst) const {
        int V = lat.V;
        Vec g5src(src.size()), g5dst(src.size());
        #pragma omp parallel for schedule(static) if(V > OMP_MIN_SIZE)
        for (int s = 0; s < V; s++) {
            g5src[2*s]   = src[2*s];
            g5src[2*s+1] = -src[2*s+1];
        }
        apply(g5src, g5dst);
        #pragma omp parallel for schedule(static) if(V > OMP_MIN_SIZE)
        for (int s = 0; s < V; s++) {
            dst[2*s]   = g5dst[2*s];
            dst[2*s+1] = -g5dst[2*s+1];
        }
    }

    void apply_DdagD(const Vec& src, Vec& dst) const {
        Vec tmp(src.size());
        apply(src, tmp);
        apply_dag(tmp, dst);
    }

    // Compute δD v = (D_new - D_old) v, where D_new has gauge links
    // U_μ(x) → exp(i dt π_μ(x)) U_μ(x).
    // Only the hopping terms change; the mass/diagonal term cancels.
    // Factor per forward hop: (e^{i dt π} - 1) U
    // Factor per backward hop: (e^{-i dt π} - 1) U†
    void apply_delta_D(const Vec& src, Vec& dst,
                       const std::array<RVec, 2>& pi, double dt) const {
        int V = lat.V;
        #pragma omp parallel for schedule(static) if(V > OMP_MIN_SIZE)
        for (int s = 0; s < V; s++) {
            dst[2*s]   = 0.0;
            dst[2*s+1] = 0.0;
            // Phase factors: fwd = (e^{i dt π} - 1), bwd = (e^{-i dt π} - 1)
            cx fwd_x = std::exp(cx(0,  dt * pi[0][s])) - 1.0;
            cx bwd_x = std::exp(cx(0, -dt * pi[0][lat.xm(s)])) - 1.0;
            cx fwd_y = std::exp(cx(0,  dt * pi[1][s])) - 1.0;
            cx bwd_y = std::exp(cx(0, -dt * pi[1][lat.ym(s)])) - 1.0;
            accumulate_hops(dst[2*s], dst[2*s+1], s, src, gauge, lat, r,
                           fwd_x, bwd_x, fwd_y, bwd_y);
        }
    }
};

// ---------------------------------------------------------------
//  Even-Odd (Schur complement) preconditioned Dirac operator
//  Operates on half-lattice vectors (odd sites only for the solve)
//  M_pc = D_oo - D_oe D_ee^{-1} D_eo  (Schur complement on odd sites)
// ---------------------------------------------------------------
struct EvenOddDiracOp {
    const DiracOp& D;
    const Lattice& lat;

    EvenOddDiracOp(const DiracOp& D_) : D(D_), lat(D_.lat) {}

    // Hopping term: read from src_parity sites, write to dst_parity sites
    // dst_sites/src_sites are the site lists for the respective parities
    void apply_hop(const Vec& src, Vec& dst,
                   const std::vector<int>& dst_sites,
                   const std::vector<int>& src_sites) const {
        int nh = (int)dst_sites.size();
        cx ii(0,1);
        #pragma omp parallel for schedule(static) if(nh > OMP_MIN_SIZE)
        for (int id = 0; id < nh; id++) {
            int s = dst_sites[id];
            dst[2*id]   = 0.0;
            dst[2*id+1] = 0.0;

            // mu=0 forward: hop to xp(s) — which is in src_parity
            {
                int sf = lat.xp(s);
                int is = lat.eo_index[sf];
                cx u = D.gauge.U[0][s];
                cx p0 = src[2*is], p1 = src[2*is+1];
                cx up0 = u*p0, up1 = u*p1;
                dst[2*id]   -= 0.5 * (D.r*up0 - up1);
                dst[2*id+1] -= 0.5 * (-up0 + D.r*up1);
            }
            // mu=0 backward: hop to xm(s)
            {
                int sb = lat.xm(s);
                int is = lat.eo_index[sb];
                cx ud = std::conj(D.gauge.U[0][sb]);
                cx q0 = src[2*is], q1 = src[2*is+1];
                cx udq0 = ud*q0, udq1 = ud*q1;
                dst[2*id]   -= 0.5 * (D.r*udq0 + udq1);
                dst[2*id+1] -= 0.5 * (udq0 + D.r*udq1);
            }
            // mu=1 forward: hop to yp(s)
            {
                int sf = lat.yp(s);
                int is = lat.eo_index[sf];
                cx u = D.gauge.U[1][s];
                cx p0 = src[2*is], p1 = src[2*is+1];
                cx up0 = u*p0, up1 = u*p1;
                dst[2*id]   -= 0.5 * (D.r*up0 + ii*up1);
                dst[2*id+1] -= 0.5 * (-ii*up0 + D.r*up1);
            }
            // mu=1 backward: hop to ym(s)
            {
                int sb = lat.ym(s);
                int is = lat.eo_index[sb];
                cx ud = std::conj(D.gauge.U[1][sb]);
                cx q0 = src[2*is], q1 = src[2*is+1];
                cx udq0 = ud*q0, udq1 = ud*q1;
                dst[2*id]   -= 0.5 * (D.r*udq0 - ii*udq1);
                dst[2*id+1] -= 0.5 * (ii*udq0 + D.r*udq1);
            }
        }
    }

    // D_eo: hop from even sites to odd sites (hopping only, no diagonal)
    void apply_eo(const Vec& src_e, Vec& dst_o) const {
        apply_hop(src_e, dst_o, lat.odd_sites, lat.even_sites);
    }

    // D_oe: hop from odd sites to even sites
    void apply_oe(const Vec& src_o, Vec& dst_e) const {
        apply_hop(src_o, dst_e, lat.even_sites, lat.odd_sites);
    }

    // D_ee^{-1}: inverse of even-site diagonal (mass + Wilson + clover)
    void apply_ee_inv(const Vec& src, Vec& dst) const {
        apply_diag_inv(src, dst, lat.even_sites);
    }

    // D_oo^{-1}: inverse of odd-site diagonal
    void apply_oo_inv(const Vec& src, Vec& dst) const {
        apply_diag_inv(src, dst, lat.odd_sites);
    }

    void apply_diag_inv(const Vec& src, Vec& dst,
                        const std::vector<int>& sites) const {
        int nh = (int)sites.size();
        bool has_clover = (D.c_sw != 0.0);
        #pragma omp parallel for schedule(static) if(nh > OMP_MIN_SIZE)
        for (int i = 0; i < nh; i++) {
            if (has_clover) {
                int s = sites[i];
                double C = 0.5 * D.c_sw * D.clover_field[s];
                dst[2*i]   = src[2*i]   / (2.0*D.r + D.mass + C);
                dst[2*i+1] = src[2*i+1] / (2.0*D.r + D.mass - C);
            } else {
                double inv = 1.0 / (2.0*D.r + D.mass);
                dst[2*i]   = inv * src[2*i];
                dst[2*i+1] = inv * src[2*i+1];
            }
        }
    }

    // D_oo: diagonal of odd sites (mass + Wilson + clover)
    void apply_oo(const Vec& src, Vec& dst) const {
        int nh = lat.V_half;
        bool has_clover = (D.c_sw != 0.0);
        #pragma omp parallel for schedule(static) if(nh > OMP_MIN_SIZE)
        for (int i = 0; i < nh; i++) {
            if (has_clover) {
                int s = lat.odd_sites[i];
                double C = 0.5 * D.c_sw * D.clover_field[s];
                dst[2*i]   = (2.0*D.r + D.mass + C) * src[2*i];
                dst[2*i+1] = (2.0*D.r + D.mass - C) * src[2*i+1];
            } else {
                double diag = 2.0*D.r + D.mass;
                dst[2*i]   = diag * src[2*i];
                dst[2*i+1] = diag * src[2*i+1];
            }
        }
    }

    // Schur complement: M = D_oo - D_oe D_ee^{-1} D_eo
    // Applied to odd-site vector, produces odd-site vector
    void apply_schur(const Vec& src_o, Vec& dst_o) const {
        int n_half = 2 * lat.V_half;
        Vec tmp_e(n_half), tmp_e2(n_half);
        apply_oe(src_o, tmp_e);         // D_oe: odd → even
        apply_ee_inv(tmp_e, tmp_e2);    // D_ee^{-1}: even → even
        apply_eo(tmp_e2, dst_o);        // D_eo: even → odd
        // dst_o = D_oo src_o - dst_o
        int nh = lat.V_half;
        bool has_clover = (D.c_sw != 0.0);
        #pragma omp parallel for schedule(static) if(nh > OMP_MIN_SIZE)
        for (int i = 0; i < nh; i++) {
            double diag = 2.0*D.r + D.mass;
            if (has_clover) {
                int s = lat.odd_sites[i];
                double C = 0.5 * D.c_sw * D.clover_field[s];
                dst_o[2*i]   = (diag + C) * src_o[2*i]   - dst_o[2*i];
                dst_o[2*i+1] = (diag - C) * src_o[2*i+1] - dst_o[2*i+1];
            } else {
                dst_o[2*i]   = diag * src_o[2*i]   - dst_o[2*i];
                dst_o[2*i+1] = diag * src_o[2*i+1] - dst_o[2*i+1];
            }
        }
    }

    // M† = D_oo† - D_eo† D_ee^{-1†} D_oe†
    // For γ₅-Hermitian D: M† = γ₅ M γ₅ (applied per odd site)
    void apply_schur_dag(const Vec& src_o, Vec& dst_o) const {
        int n_half = 2 * lat.V_half;
        // γ₅ = [[1,0],[0,-1]] flips sign of second spinor component
        Vec g5src(n_half);
        #pragma omp parallel for schedule(static) if(lat.V_half > OMP_MIN_SIZE/8)
        for (int i = 0; i < lat.V_half; i++) {
            g5src[2*i]   =  src_o[2*i];
            g5src[2*i+1] = -src_o[2*i+1];
        }
        Vec g5dst(n_half);
        apply_schur(g5src, g5dst);
        #pragma omp parallel for schedule(static) if(lat.V_half > OMP_MIN_SIZE/8)
        for (int i = 0; i < lat.V_half; i++) {
            dst_o[2*i]   =  g5dst[2*i];
            dst_o[2*i+1] = -g5dst[2*i+1];
        }
    }

    // M†M for CG (HPD operator on odd sites)
    void apply_schur_dag_schur(const Vec& src_o, Vec& dst_o) const {
        int n_half = 2 * lat.V_half;
        Vec tmp(n_half);
        apply_schur(src_o, tmp);
        apply_schur_dag(tmp, dst_o);
    }

    // Scatter half-lattice vectors into a full-lattice vector
    Vec scatter(const Vec& half_e, const Vec& half_o) const {
        Vec full(lat.ndof, 0.0);
        #pragma omp parallel for schedule(static) if(lat.V_half > OMP_MIN_SIZE/8)
        for (int i = 0; i < lat.V_half; i++) {
            int se = lat.even_sites[i];
            full[2*se]   = half_e[2*i];
            full[2*se+1] = half_e[2*i+1];
        }
        #pragma omp parallel for schedule(static) if(lat.V_half > OMP_MIN_SIZE/8)
        for (int i = 0; i < lat.V_half; i++) {
            int so = lat.odd_sites[i];
            full[2*so]   = half_o[2*i];
            full[2*so+1] = half_o[2*i+1];
        }
        return full;
    }

    // Gather even/odd components from a full-lattice vector
    Vec gather_even(const Vec& full) const {
        Vec half(2 * lat.V_half);
        #pragma omp parallel for schedule(static) if(lat.V_half > OMP_MIN_SIZE/8)
        for (int i = 0; i < lat.V_half; i++) {
            int s = lat.even_sites[i];
            half[2*i]   = full[2*s];
            half[2*i+1] = full[2*s+1];
        }
        return half;
    }

    Vec gather_odd(const Vec& full) const {
        Vec half(2 * lat.V_half);
        #pragma omp parallel for schedule(static) if(lat.V_half > OMP_MIN_SIZE/8)
        for (int i = 0; i < lat.V_half; i++) {
            int s = lat.odd_sites[i];
            half[2*i]   = full[2*s];
            half[2*i+1] = full[2*s+1];
        }
        return half;
    }

    // === Asymmetric even-odd HMC (QUDA/openQCD approach) ===
    //
    // Pseudofermion lives in the Schur complement space (odd sites).
    // det(D) = det(D_ee) × det(M) where M = D_oo - D_oe D_ee⁻¹ D_eo.
    // For Wilson (c_sw=0), det(D_ee) is constant and drops out.
    // For clover, det(D_ee) must be included separately (log-det + force).
    //
    // HMC workflow:
    //   1. Generate η_o random on odd sites
    //   2. φ_o = M† η_o  (pseudofermion)
    //   3. CG: M†M x_o = φ_o → x_o = (M†M)⁻¹ φ_o
    //   4. Action: S = φ_o† x_o
    //   5. Reconstruct full x from x_o for force computation
    //   6. Force = fermion_force(D, x_full) + clover_logdet_force (if c_sw≠0)

    // Generate pseudofermion: φ_o = M† η_o
    Vec generate_pseudofermion_eo(std::mt19937& rng) const {
        int n_half = 2 * lat.V_half;
        Vec eta_o = random_vec(n_half, rng);
        Vec phi_o(n_half);
        apply_schur_dag(eta_o, phi_o);
        return phi_o;
    }

    // Reconstruct full-lattice solution from odd-site CG solution x_o:
    // For Mx_o = something, the full D solution has:
    //   x_e = -D_ee⁻¹ D_eo x_o    (hop odd→even, then diagonal inverse)
    Vec reconstruct_full(const Vec& x_o) const {
        int n_half = 2 * lat.V_half;
        Vec hop_e(n_half);
        apply_oe(x_o, hop_e);          // D_oe: odd→even hopping
        Vec x_e(n_half);
        for (int i = 0; i < n_half; i++) hop_e[i] = -hop_e[i];
        apply_ee_inv(hop_e, x_e);
        return scatter(x_e, x_o);
    }

    // Compute log det(D_ee) — needed for clover HMC Hamiltonian
    // For Wilson (c_sw=0): returns 2*V_half*log(2r+m) (constant, can be ignored in ΔH)
    double log_det_ee() const {
        double logdet = 0;
        #pragma omp parallel for schedule(static) if(lat.V_half > OMP_MIN_SIZE/8)
        for (int i = 0; i < lat.V_half; i++) {
            int s = lat.even_sites[i];
            double diag = 2.0*D.r + D.mass;
            if (D.c_sw != 0.0) {
                double C = 0.5 * D.c_sw * D.clover_field[s];
                logdet += std::log(diag + C) + std::log(diag - C);
            } else {
                logdet += 2.0 * std::log(diag);
            }
        }
        return logdet;
    }

    // Reconstruct full solution from odd-site CG solution x_o and pseudofermion φ:
    // x_e = D_ee^{-1} (φ_e - D_eo x_o)  [note: D_eo here means odd→even hopping]
    Vec reconstruct(const Vec& x_o, const Vec& phi) const {
        Vec phi_e = gather_even(phi);
        int n_half = 2 * lat.V_half;
        Vec hop_e(n_half);
        apply_oe(x_o, hop_e);          // D_oe: odd → even (hopping of x_o to even sites)
        // tmp = φ_e - hop_e
        for (int i = 0; i < n_half; i++)
            hop_e[i] = phi_e[i] - hop_e[i];
        Vec x_e(n_half);
        apply_ee_inv(hop_e, x_e);       // x_e = D_ee^{-1} (φ_e - D_oe x_o)
        return scatter(x_e, x_o);
    }
};
