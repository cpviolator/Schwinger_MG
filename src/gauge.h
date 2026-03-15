#pragma once
#include "types.h"
#include "lattice.h"
#include <array>
#include <random>
#include <cmath>
#include <omp.h>

struct GaugeField {
    const Lattice& lat;
    std::array<Vec, 2> U;

    GaugeField(const Lattice& lat_) : lat(lat_) {
        U[0].resize(lat.V, 1.0);
        U[1].resize(lat.V, 1.0);
    }

    void randomise(std::mt19937& rng, double width = M_PI) {
        std::uniform_real_distribution<double> dist(-width, width);
        for (int mu = 0; mu < 2; mu++)
            for (int s = 0; s < lat.V; s++)
                U[mu][s] = std::exp(cx(0, dist(rng)));
    }

    cx plaq(int s) const {
        int sp0 = lat.xp(s), sp1 = lat.yp(s);
        return U[0][s] * U[1][sp0] * std::conj(U[0][sp1]) * std::conj(U[1][s]);
    }

    double avg_plaq() const {
        double s = 0;
        #pragma omp parallel for reduction(+:s) schedule(static) if(lat.V > OMP_MIN_SIZE)
        for (int i = 0; i < lat.V; i++) s += std::real(plaq(i));
        return s / lat.V;
    }
};
