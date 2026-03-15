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

    DiracOp(const Lattice& lat_, const GaugeField& g_, double m_, double r_ = 1.0)
        : lat(lat_), gauge(g_), mass(m_), r(r_) {}

    void apply(const Vec& src, Vec& dst) const {
        int V = lat.V;
        #pragma omp parallel for schedule(static) if(V > OMP_MIN_SIZE)
        for (int s = 0; s < V; s++) {
            cx s0 = src[2*s], s1 = src[2*s+1];
            dst[2*s]   = (2.0*r + mass) * s0;
            dst[2*s+1] = (2.0*r + mass) * s1;

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
};
