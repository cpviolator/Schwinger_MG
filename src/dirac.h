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
