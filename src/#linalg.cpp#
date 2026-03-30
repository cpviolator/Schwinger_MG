#include "linalg.h"
#include <cmath>
#include <omp.h>

long long g_matvec_count = 0;

cx dot(const Vec& a, const Vec& b) {
    double re = 0, im = 0;
    int n = (int)a.size();
    #pragma omp parallel for reduction(+:re,im) schedule(static) if(n > OMP_MIN_SIZE)
    for (int i = 0; i < n; i++) {
        cx prod = std::conj(a[i]) * b[i];
        re += std::real(prod);
        im += std::imag(prod);
    }
    return cx(re, im);
}

double norm(const Vec& v) { return std::sqrt(std::real(dot(v,v))); }

void axpy(cx a, const Vec& x, Vec& y) {
    int n = (int)x.size();
    #pragma omp parallel for schedule(static) if(n > OMP_MIN_SIZE)
    for (int i = 0; i < n; i++) y[i] += a * x[i];
}

Vec add(const Vec& a, const Vec& b, cx alpha, cx beta) {
    int n = (int)a.size();
    Vec r(n);
    #pragma omp parallel for schedule(static) if(n > OMP_MIN_SIZE)
    for (int i = 0; i < n; i++) r[i] = alpha*a[i] + beta*b[i];
    return r;
}

void scale(Vec& v, cx a) {
    int n = (int)v.size();
    #pragma omp parallel for schedule(static) if(n > OMP_MIN_SIZE)
    for (int i = 0; i < n; i++) v[i] *= a;
}

Vec zeros(int n) { return Vec(n, 0.0); }

Vec random_vec(int n, std::mt19937& rng) {
    std::normal_distribution<double> dist(0.0, 1.0);
    Vec v(n);
    for (auto& x : v) x = cx(dist(rng), dist(rng));
    return v;
}
