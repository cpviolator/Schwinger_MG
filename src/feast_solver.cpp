#include "feast_solver.h"
#include "linalg.h"
#include <cstring>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>

// Call FEAST Fortran functions directly (the feast.h C header uses char* for ijob
// but the Fortran interface uses INTEGER — mismatch causes ijob to be misread)
extern "C" {
    void feastinit_(int* fpm);
    void zfeast_hrci_(int* ijob, int* N, double* Ze, double* work, double* workc,
                      double* zAq, double* zSq, int* fpm, double* epsout, int* loop,
                      double* Emin, double* Emax, int* M0, double* lambda,
                      double* q, int* mode, double* res, int* info);
}
static void feast_init(int* fpm) { feastinit_(fpm); }

// ---------------------------------------------------------------
// Preconditioned BiCGStab for shifted system (z*I - A)*x = b
// z is complex, A is Hermitian → shifted op is non-Hermitian
// precond: optional left preconditioner M^{-1} (e.g., MG V-cycle)
// ---------------------------------------------------------------
Vec shifted_solve(const OpApply& A, cx z, const Vec& b, int n,
                  int max_iter, double tol,
                  const std::function<Vec(const Vec&)>* precond) {
    // Shifted operator: T*x = z*x - A*x
    auto T = [&](const Vec& src, Vec& dst) {
        A(src, dst);
        for (int i = 0; i < n; i++)
            dst[i] = z * src[i] - dst[i];
    };

    // Preconditioner: M^{-1} approximation (identity if none)
    auto apply_precond = [&](const Vec& src) -> Vec {
        if (precond) {
            // The preconditioner approximates (D†D)^{-1}, not (zI-D†D)^{-1}.
            // For BiCGStab, any SPD preconditioner helps. The MG V-cycle
            // approximates (D†D)^{-1} which is a good preconditioner for
            // the shifted system when |z| is not too large.
            return (*precond)(src);
        }
        return src;
    };

    Vec x = zeros(n);
    Vec r(n);
    T(x, r);
    for (int i = 0; i < n; i++) r[i] = b[i] - r[i];

    Vec r_hat = r;
    cx rho_old = 1.0, alpha = 1.0, omega = 1.0;
    Vec v = zeros(n), p = zeros(n);

    double b_norm = norm(b);
    if (b_norm < 1e-30) return x;

    for (int iter = 0; iter < max_iter; iter++) {
        cx rho = dot(r_hat, r);
        if (std::abs(rho) < 1e-30) break;

        cx beta = (rho / rho_old) * (alpha / omega);
        for (int i = 0; i < n; i++)
            p[i] = r[i] + beta * (p[i] - omega * v[i]);

        // Preconditioned: p_hat = M^{-1} p, then v = T p_hat
        Vec p_hat = apply_precond(p);
        T(p_hat, v);
        alpha = rho / dot(r_hat, v);

        Vec s(n);
        for (int i = 0; i < n; i++) s[i] = r[i] - alpha * v[i];

        double s_norm = norm(s);
        if (s_norm / b_norm < tol) {
            for (int i = 0; i < n; i++) x[i] += alpha * p_hat[i];
            return x;
        }

        // Preconditioned: s_hat = M^{-1} s, then t = T s_hat
        Vec s_hat = apply_precond(s);
        Vec t(n);
        T(s_hat, t);
        omega = dot(t, s) / dot(t, t);

        for (int i = 0; i < n; i++) {
            x[i] += alpha * p_hat[i] + omega * s_hat[i];
            r[i] = s[i] - omega * t[i];
        }

        rho_old = rho;

        if (norm(r) / b_norm < tol) return x;
    }
    return x;
}

// ---------------------------------------------------------------
// FEAST eigensolver via RCI (Reverse Communication Interface)
// Uses zfeast_hrci for complex Hermitian eigenproblems
// ---------------------------------------------------------------
TRLMResult feast_eigensolver(
    const OpApply& A, int n,
    double Emin, double Emax, int M0,
    int n_contour, double tol, int max_iter,
    const std::vector<Vec>* warm_start,
    const std::function<Vec(const Vec&)>* precond)
{
    TRLMResult result;
    result.converged = false;
    result.iterations = 0;
    result.num_restarts = 0;

    // FEAST parameter array
    int fpm[128];
    std::memset(fpm, 0, sizeof(fpm));
    feast_init(fpm);
    fpm[0] = 1;         // verbose output
    fpm[1] = n_contour;  // half-contour quadrature points
    fpm[2] = -(int)std::log10(std::max(tol, 1e-16)); // tolerance exponent
    fpm[3] = max_iter;   // max FEAST iterations
    fpm[4] = warm_start ? 1 : 0;  // 0=random init, 1=user-provided

    // Allocate FEAST work arrays
    // zfeast_hrci uses complex work arrays (stored as double pairs)
    // work:  N × M0 complex (output: filtered subspace)
    // workc: N × M0 complex (RHS / solve workspace)
    // q:     N × M0 complex (eigenvectors)
    // Aq, Sq: M0 × M0 complex (projected matrices)
    std::vector<double> work(2 * n * M0, 0.0);
    std::vector<double> workc(2 * n * M0, 0.0);
    std::vector<double> q(2 * n * M0, 0.0);
    std::vector<double> Aq(2 * M0 * M0, 0.0);
    std::vector<double> Sq(2 * M0 * M0, 0.0);
    std::vector<double> lambda(M0, 0.0);
    std::vector<double> res(M0, 0.0);

    // Warm start: copy initial eigenvectors into q
    if (warm_start) {
        int k = std::min((int)warm_start->size(), M0);
        for (int j = 0; j < k; j++) {
            for (int i = 0; i < n && i < (int)(*warm_start)[j].size(); i++) {
                q[2*(j*n + i)]     = std::real((*warm_start)[j][i]);
                q[2*(j*n + i) + 1] = std::imag((*warm_start)[j][i]);
            }
        }
    }

    double Ze[2];  // complex shift (real, imag)
    int mode = 0;  // number of eigenvalues found
    double epsout = 0.0;
    int loop = 0;
    int info = 0;
    int ijob = -1;  // Fortran INTEGER, init to -1 for first call

    int total_matvecs = 0;
    int total_solves = 0;

    // RCI loop
    while (true) {
        zfeast_hrci_(&ijob, &n, Ze, work.data(), workc.data(),
                     Aq.data(), Sq.data(), fpm, &epsout, &loop,
                     &Emin, &Emax, &M0, lambda.data(), q.data(),
                     &mode, res.data(), &info);

        if (ijob == 0) break;  // done

        if (ijob == 10 || ijob == 20) {
            // Factorize: prepare to solve (Ze*I - A) or (Ze*B - A)
            // For standard eigenproblem (B=I), nothing to pre-factorize
            // The solve happens at ijob=11/21
            continue;
        }

        if (ijob == 11 || ijob == 21) {
            // Solve: (Ze*I - A) * solution = workc
            // workc contains M0 right-hand sides (columns of N×M0 complex matrix)
            // Result goes back into workc
            cx z_shift(Ze[0], Ze[1]);

            for (int col = 0; col < M0; col++) {
                // Extract column from workc
                Vec rhs(n);
                for (int i = 0; i < n; i++)
                    rhs[i] = cx(workc[2*(col*n + i)], workc[2*(col*n + i) + 1]);

                // Solve (z*I - A)*x = rhs
                Vec sol = shifted_solve(A, z_shift, rhs, n, 500, 1e-12, precond);
                total_solves++;

                // Write solution back to workc
                for (int i = 0; i < n; i++) {
                    workc[2*(col*n + i)]     = std::real(sol[i]);
                    workc[2*(col*n + i) + 1] = std::imag(sol[i]);
                }
            }
            continue;
        }

        if (ijob == 30) {
            // Multiply: A * q → work  (M0 columns)
            for (int col = 0; col < M0; col++) {
                Vec src(n);
                for (int i = 0; i < n; i++)
                    src[i] = cx(q[2*(col*n + i)], q[2*(col*n + i) + 1]);

                Vec dst(n);
                A(src, dst);
                total_matvecs++;

                for (int i = 0; i < n; i++) {
                    work[2*(col*n + i)]     = std::real(dst[i]);
                    work[2*(col*n + i) + 1] = std::imag(dst[i]);
                }
            }
            continue;
        }

        if (ijob == 40) {
            // Multiply: B * q → work  (B=I for standard eigenproblem)
            std::memcpy(work.data(), q.data(), 2 * n * M0 * sizeof(double));
            continue;
        }

        // Unknown ijob — shouldn't happen
        std::cerr << "FEAST RCI: unknown ijob=" << ijob << "\n";
        break;
    }

    // Extract results
    std::cout << "  FEAST: info=" << info << " mode=" << mode
              << " loops=" << loop << " epsout=" << std::scientific
              << epsout << " solves=" << total_solves
              << " matvecs=" << total_matvecs << "\n";

    if (info != 0 && info != 1) {
        std::cerr << "  FEAST warning: info=" << info << "\n";
    }

    result.iterations = total_matvecs;
    result.num_restarts = loop;
    result.converged = (info == 0);

    // Copy eigenvalues and eigenvectors (only the 'mode' found)
    int n_found = std::max(0, mode);
    result.eigvals.resize(n_found);
    result.eigvecs.resize(n_found);

    for (int j = 0; j < n_found; j++) {
        result.eigvals[j] = lambda[j];
        result.eigvecs[j].resize(n);
        for (int i = 0; i < n; i++)
            result.eigvecs[j][i] = cx(q[2*(j*n + i)], q[2*(j*n + i) + 1]);
    }

    // Sort by eigenvalue (ascending)
    if (n_found > 1) {
        std::vector<int> idx(n_found);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(),
                  [&](int a, int b) { return result.eigvals[a] < result.eigvals[b]; });
        std::vector<double> sorted_vals(n_found);
        std::vector<Vec> sorted_vecs(n_found);
        for (int i = 0; i < n_found; i++) {
            sorted_vals[i] = result.eigvals[idx[i]];
            sorted_vecs[i] = std::move(result.eigvecs[idx[i]]);
        }
        result.eigvals = std::move(sorted_vals);
        result.eigvecs = std::move(sorted_vecs);
    }

    return result;
}
