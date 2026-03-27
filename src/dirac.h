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

            // mu=0 (x-direction), gamma_1 = [[0,1],[1,0]]
            {
                int sf = lat.xp(s);
                cx u = gauge.U[0][s];
                cx p0 = src[2*sf], p1 = src[2*sf+1];
                cx up0 = u * p0, up1 = u * p1;
                dst[2*s]   -= 0.5 * (r*up0 - up1);
                dst[2*s+1] -= 0.5 * (-up0 + r*up1);

                int sb = lat.xm(s);
                cx ud = std::conj(gauge.U[0][sb]);
                cx q0 = src[2*sb], q1 = src[2*sb+1];
                cx udq0 = ud * q0, udq1 = ud * q1;
                dst[2*s]   -= 0.5 * (r*udq0 + udq1);
                dst[2*s+1] -= 0.5 * (udq0 + r*udq1);
            }

            // mu=1 (y-direction), gamma_2 = [[0,-i],[i,0]]
            {
                int sf = lat.yp(s);
                cx u = gauge.U[1][s];
                cx p0 = src[2*sf], p1 = src[2*sf+1];
                cx up0 = u * p0, up1 = u * p1;
                cx ii(0,1);
                dst[2*s]   -= 0.5 * (r*up0 + ii*up1);
                dst[2*s+1] -= 0.5 * (-ii*up0 + r*up1);

                int sb = lat.ym(s);
                cx ud = std::conj(gauge.U[1][sb]);
                cx q0 = src[2*sb], q1 = src[2*sb+1];
                cx udq0 = ud * q0, udq1 = ud * q1;
                dst[2*s]   -= 0.5 * (r*udq0 - ii*udq1);
                dst[2*s+1] -= 0.5 * (ii*udq0 + r*udq1);
            }
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

    // ========================================
    // Even-odd preconditioned operations
    // ========================================
    // Half-vectors: ndof2 = V/2 * 2 (2 spinor components per site, V/2 sites per parity)
    // Indexing: half_vec[2*i + spin] for half-site index i, spin 0 or 1

    // Diagonal mass + clover on even sites: D_ee
    double diag_ee(int half_s, int spin) const {
        int s = lat.even_sites[half_s];
        double d = 2.0 * r + mass;
        if (c_sw != 0.0) {
            double C = 0.5 * c_sw * clover_field[s];
            return d + (spin == 0 ? C : -C);
        }
        return d;
    }
    // Diagonal on odd sites: D_oo
    double diag_oo(int half_s, int spin) const {
        int s = lat.odd_sites[half_s];
        double d = 2.0 * r + mass;
        if (c_sw != 0.0) {
            double C = 0.5 * c_sw * clover_field[s];
            return d + (spin == 0 ? C : -C);
        }
        return d;
    }

    // Apply D_eo: even ← odd hopping
    void apply_eo(const Vec& src_odd, Vec& dst_even) const {
        int V2 = lat.V2;
        dst_even.assign(lat.ndof2, cx(0));
        for (int he = 0; he < V2; he++) {
            int s = lat.even_sites[he];  // full-lattice even site
            cx d0(0), d1(0);

            // All neighbors of an even site are odd
            // mu=0 forward
            { int sf = lat.xp(s); int ho = lat.eo_map[sf];
              cx u = gauge.U[0][s], p0 = src_odd[2*ho], p1 = src_odd[2*ho+1];
              cx up0 = u*p0, up1 = u*p1;
              d0 -= 0.5*(r*up0 - up1); d1 -= 0.5*(-up0 + r*up1); }
            // mu=0 backward
            { int sb = lat.xm(s); int ho = lat.eo_map[sb];
              cx ud = std::conj(gauge.U[0][sb]), q0 = src_odd[2*ho], q1 = src_odd[2*ho+1];
              cx udq0 = ud*q0, udq1 = ud*q1;
              d0 -= 0.5*(r*udq0 + udq1); d1 -= 0.5*(udq0 + r*udq1); }
            // mu=1 forward
            { int sf = lat.yp(s); int ho = lat.eo_map[sf]; cx ii(0,1);
              cx u = gauge.U[1][s], p0 = src_odd[2*ho], p1 = src_odd[2*ho+1];
              cx up0 = u*p0, up1 = u*p1;
              d0 -= 0.5*(r*up0 + ii*up1); d1 -= 0.5*(-ii*up0 + r*up1); }
            // mu=1 backward
            { int sb = lat.ym(s); int ho = lat.eo_map[sb]; cx ii(0,1);
              cx ud = std::conj(gauge.U[1][sb]), q0 = src_odd[2*ho], q1 = src_odd[2*ho+1];
              cx udq0 = ud*q0, udq1 = ud*q1;
              d0 -= 0.5*(r*udq0 - ii*udq1); d1 -= 0.5*(ii*udq0 + r*udq1); }

            dst_even[2*he] = d0;
            dst_even[2*he+1] = d1;
        }
    }

    // Apply D_oe: odd ← even hopping
    void apply_oe(const Vec& src_even, Vec& dst_odd) const {
        int V2 = lat.V2;
        dst_odd.assign(lat.ndof2, cx(0));
        for (int ho = 0; ho < V2; ho++) {
            int s = lat.odd_sites[ho];
            cx d0(0), d1(0);

            // mu=0 forward
            { int sf = lat.xp(s); int he = lat.eo_map[sf];
              cx u = gauge.U[0][s], p0 = src_even[2*he], p1 = src_even[2*he+1];
              cx up0 = u*p0, up1 = u*p1;
              d0 -= 0.5*(r*up0 - up1); d1 -= 0.5*(-up0 + r*up1); }
            // mu=0 backward
            { int sb = lat.xm(s); int he = lat.eo_map[sb];
              cx ud = std::conj(gauge.U[0][sb]), q0 = src_even[2*he], q1 = src_even[2*he+1];
              cx udq0 = ud*q0, udq1 = ud*q1;
              d0 -= 0.5*(r*udq0 + udq1); d1 -= 0.5*(udq0 + r*udq1); }
            // mu=1 forward
            { int sf = lat.yp(s); int he = lat.eo_map[sf]; cx ii(0,1);
              cx u = gauge.U[1][s], p0 = src_even[2*he], p1 = src_even[2*he+1];
              cx up0 = u*p0, up1 = u*p1;
              d0 -= 0.5*(r*up0 + ii*up1); d1 -= 0.5*(-ii*up0 + r*up1); }
            // mu=1 backward
            { int sb = lat.ym(s); int he = lat.eo_map[sb]; cx ii(0,1);
              cx ud = std::conj(gauge.U[1][sb]), q0 = src_even[2*he], q1 = src_even[2*he+1];
              cx udq0 = ud*q0, udq1 = ud*q1;
              d0 -= 0.5*(r*udq0 - ii*udq1); d1 -= 0.5*(ii*udq0 + r*udq1); }

            dst_odd[2*ho] = d0;
            dst_odd[2*ho+1] = d1;
        }
    }

    // Invert diagonal: x = D_ee^{-1} b or D_oo^{-1} b
    void invert_diag(const Vec& src, Vec& dst, bool even) const {
        int V2 = lat.V2;
        dst.resize(lat.ndof2);
        for (int h = 0; h < V2; h++) {
            for (int spin = 0; spin < 2; spin++) {
                double d = even ? diag_ee(h, spin) : diag_oo(h, spin);
                dst[2*h + spin] = src[2*h + spin] / d;
            }
        }
    }

    // Schur complement: D̂_ee = D_ee - D_eo D_oo^{-1} D_oe
    // Operates on even-parity half-vectors
    void apply_schur(const Vec& src_even, Vec& dst_even) const {
        // D_oe src → tmp_odd
        Vec tmp_odd;
        apply_oe(src_even, tmp_odd);
        // D_oo^{-1} tmp → tmp_odd
        Vec inv_tmp;
        invert_diag(tmp_odd, inv_tmp, false);
        // D_eo (D_oo^{-1} D_oe src) → tmp_even
        Vec tmp_even;
        apply_eo(inv_tmp, tmp_even);
        // D_ee src - D_eo D_oo^{-1} D_oe src
        dst_even.resize(lat.ndof2);
        for (int h = 0; h < lat.V2; h++)
            for (int spin = 0; spin < 2; spin++) {
                double d = diag_ee(h, spin);
                dst_even[2*h+spin] = d * src_even[2*h+spin] - tmp_even[2*h+spin];
            }
    }

    // D̂†D̂ on even-parity vectors
    void apply_schur_DdagD(const Vec& src, Vec& dst) const {
        Vec tmp(lat.ndof2);
        apply_schur(src, tmp);
        // D̂† = γ5 D̂ γ5 (γ5-hermiticity)
        Vec g5tmp(lat.ndof2), g5dst(lat.ndof2);
        for (int h = 0; h < lat.V2; h++) { g5tmp[2*h] = tmp[2*h]; g5tmp[2*h+1] = -tmp[2*h+1]; }
        apply_schur(g5tmp, g5dst);
        for (int h = 0; h < lat.V2; h++) { dst[2*h] = g5dst[2*h]; dst[2*h+1] = -g5dst[2*h+1]; }
    }

    // Reconstruct full solution from even-parity x_e:
    // x_o = D_oo^{-1} (b_o - D_oe x_e)  where b is the full source
    // For HMC with pseudofermion φ_e on even parity:
    //   b_o = 0 → x_o = -D_oo^{-1} D_oe x_e
    void reconstruct_odd(const Vec& x_even, Vec& x_odd) const {
        Vec tmp;
        apply_oe(x_even, tmp);
        invert_diag(tmp, x_odd, false);
        for (auto& v : x_odd) v = -v;
    }

    // Scatter even-parity half-vector to full-lattice vector
    void scatter_even(const Vec& half, Vec& full) const {
        full.assign(lat.ndof, cx(0));
        for (int h = 0; h < lat.V2; h++) {
            int s = lat.even_sites[h];
            full[2*s] = half[2*h];
            full[2*s+1] = half[2*h+1];
        }
    }
    // Scatter odd-parity
    void scatter_odd(const Vec& half, Vec& full) const {
        for (int h = 0; h < lat.V2; h++) {
            int s = lat.odd_sites[h];
            full[2*s] = half[2*h];
            full[2*s+1] = half[2*h+1];
        }
    }
    // Gather even-parity from full vector
    void gather_even(const Vec& full, Vec& half) const {
        half.resize(lat.ndof2);
        for (int h = 0; h < lat.V2; h++) {
            int s = lat.even_sites[h];
            half[2*h] = full[2*s];
            half[2*h+1] = full[2*s+1];
        }
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

            // mu=0 (x-direction), gamma_1 = [[0,1],[1,0]]
            {
                int sf = lat.xp(s);
                cx phase_fwd = std::exp(cx(0, dt * pi[0][s])) - 1.0;
                cx u = gauge.U[0][s];
                cx p0 = src[2*sf], p1 = src[2*sf+1];
                cx dup0 = phase_fwd * u * p0, dup1 = phase_fwd * u * p1;
                dst[2*s]   -= 0.5 * (r*dup0 - dup1);
                dst[2*s+1] -= 0.5 * (-dup0 + r*dup1);

                int sb = lat.xm(s);
                cx phase_bwd = std::exp(cx(0, -dt * pi[0][sb])) - 1.0;
                cx ud = std::conj(gauge.U[0][sb]);
                cx q0 = src[2*sb], q1 = src[2*sb+1];
                cx dudq0 = phase_bwd * ud * q0, dudq1 = phase_bwd * ud * q1;
                dst[2*s]   -= 0.5 * (r*dudq0 + dudq1);
                dst[2*s+1] -= 0.5 * (dudq0 + r*dudq1);
            }

            // mu=1 (y-direction), gamma_2 = [[0,-i],[i,0]]
            {
                int sf = lat.yp(s);
                cx phase_fwd = std::exp(cx(0, dt * pi[1][s])) - 1.0;
                cx u = gauge.U[1][s];
                cx p0 = src[2*sf], p1 = src[2*sf+1];
                cx dup0 = phase_fwd * u * p0, dup1 = phase_fwd * u * p1;
                cx ii(0,1);
                dst[2*s]   -= 0.5 * (r*dup0 + ii*dup1);
                dst[2*s+1] -= 0.5 * (-ii*dup0 + r*dup1);

                int sb = lat.ym(s);
                cx phase_bwd = std::exp(cx(0, -dt * pi[1][sb])) - 1.0;
                cx ud = std::conj(gauge.U[1][sb]);
                cx q0 = src[2*sb], q1 = src[2*sb+1];
                cx dudq0 = phase_bwd * ud * q0, dudq1 = phase_bwd * ud * q1;
                dst[2*s]   -= 0.5 * (r*dudq0 - ii*dudq1);
                dst[2*s+1] -= 0.5 * (ii*dudq0 + r*dudq1);
            }
        }
    }
};
