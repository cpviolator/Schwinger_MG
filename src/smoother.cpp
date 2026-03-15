#include "smoother.h"
#include "linalg.h"
#include <omp.h>
#include <cmath>
#include <random>

// Simple Minimal Residual (MR) smoother: cheaper than SSOR for D†D.
// x <- x + omega (A r) / (A r, A r) * r   where r = b - Ax
void mr_smooth(const DiracOp& D, Vec& x, const Vec& b, int niter) {
    int n = D.lat.ndof;
    Vec r(n), Ar(n);
    for (int it = 0; it < niter; it++) {
        D.apply_DdagD(x, r);
        #pragma omp parallel for schedule(static) if(n > OMP_MIN_SIZE)
        for (int i = 0; i < n; i++) r[i] = b[i] - r[i];
        D.apply_DdagD(r, Ar);
        cx rAr = dot(Ar, r);
        cx ArAr = dot(Ar, Ar);
        if (std::abs(ArAr) < 1e-30) break;
        cx omega = rAr / ArAr;
        axpy(omega, r, x);
    }
}

// Standard MR: adaptive omega = <Ar,r>/<Ar,Ar> (nonlinear, NOT symmetric)
void mr_smooth_op(const OpApply& A, Vec& x, const Vec& b, int niter) {
    int n = (int)x.size();
    Vec r(n), Ar(n);
    for (int it = 0; it < niter; it++) {
        A(x, r);
        #pragma omp parallel for schedule(static) if(n > OMP_MIN_SIZE)
        for (int i = 0; i < n; i++) r[i] = b[i] - r[i];
        A(r, Ar);
        cx rAr = dot(Ar, r);
        cx ArAr = dot(Ar, Ar);
        if (std::abs(ArAr) < 1e-30) break;
        cx omega = rAr / ArAr;
        axpy(omega, r, x);
    }
}

// Estimate lambda_max of operator A via power iteration
double estimate_lambda_max(const OpApply& A, int n, int niter) {
    std::mt19937 rng_local(12345);
    Vec v = random_vec(n, rng_local);
    double v_norm = norm(v);
    scale(v, 1.0/v_norm);

    Vec Av(n);
    double lambda = 0;
    for (int i = 0; i < niter; i++) {
        A(v, Av);
        lambda = norm(Av);
        if (lambda < 1e-30) break;
        scale(Av, 1.0/lambda);
        v = Av;
    }
    return lambda;
}

// Richardson smoother: fixed omega (linear, symmetric for HPD A)
// omega should be < 2/lambda_max(A) for stability
void richardson_smooth_op(const OpApply& A, Vec& x, const Vec& b,
                          int niter, double omega) {
    int n = (int)x.size();
    Vec r(n);
    for (int it = 0; it < niter; it++) {
        A(x, r);
        #pragma omp parallel for schedule(static) if(n > OMP_MIN_SIZE)
        for (int i = 0; i < n; i++)
            x[i] += omega * (b[i] - r[i]);
    }
}
