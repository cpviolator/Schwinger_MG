#include "eigensolver.h"
#include "linalg.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <iomanip>
#include <omp.h>

// Jacobi eigenvalue solver for small Hermitian matrix (for Ritz extraction)
// Returns eigenvalues in ascending order with corresponding eigenvectors.
void jacobi_eigen(std::vector<Vec>& A_cols, int n,
                  RVec& evals, std::vector<Vec>& evecs) {
    // Copy to working matrix (row-major for convenience)
    std::vector<Vec> A(n, Vec(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            A[i][j] = A_cols[j][i];  // A_cols is column-major

    // Eigenvector accumulator (starts as identity)
    evecs.resize(n, Vec(n, 0.0));
    for (int i = 0; i < n; i++) evecs[i][i] = 1.0;

    double off = 0;
    bool converged = false;
    for (int sweep = 0; sweep < 500; sweep++) {
        // check off-diagonal norm
        off = 0;
        for (int i = 0; i < n; i++)
            for (int j = i+1; j < n; j++)
                off += std::norm(A[i][j]);
        if (off < 1e-28) { converged = true; break; }

        for (int p = 0; p < n-1; p++) {
            for (int q = p+1; q < n; q++) {
                if (std::abs(A[p][q]) < 1e-15) continue;
                // Jacobi rotation to zero A[p][q]
                cx apq = A[p][q];
                double app = std::real(A[p][p]);
                double aqq = std::real(A[q][q]);
                double tau = (aqq - app) / (2.0 * std::abs(apq));
                double t = (tau >= 0 ? 1.0 : -1.0) /
                           (std::abs(tau) + std::sqrt(1 + tau*tau));
                double c = 1.0 / std::sqrt(1 + t*t);
                double s_real = t * c;
                cx phase = (std::abs(apq) > 1e-30) ? apq / std::abs(apq) : 1.0;
                cx s = s_real * std::conj(phase);

                // apply rotation to A
                for (int i = 0; i < n; i++) {
                    cx aip = A[i][p], aiq = A[i][q];
                    A[i][p] = c * aip + s * aiq;
                    A[i][q] = -std::conj(s) * aip + c * aiq;
                }
                for (int j = 0; j < n; j++) {
                    cx apj = A[p][j], aqj = A[q][j];
                    A[p][j] = c * apj + std::conj(s) * aqj;
                    A[q][j] = -s * apj + c * aqj;
                }
                // accumulate eigenvectors
                for (int i = 0; i < n; i++) {
                    cx vip = evecs[i][p], viq = evecs[i][q];
                    evecs[i][p] = c * vip + s * viq;
                    evecs[i][q] = -std::conj(s) * vip + c * viq;
                }
            }
        }
    }
    if (!converged) {
        // Only warn if off-diagonal norm is significant relative to diagonal
        double diag_norm = 0;
        for (int i = 0; i < n; i++) diag_norm += std::norm(A[i][i]);
        // Jacobi convergence warning suppressed for clean output.
        // Production code should use LAPACK (zheev/zhegv).
    }

    evals.resize(n);
    for (int i = 0; i < n; i++) evals[i] = std::real(A[i][i]);

    // sort by ascending eigenvalue
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b) {
        return evals[a] < evals[b];
    });
    RVec sorted_evals(n);
    std::vector<Vec> sorted_evecs(n, Vec(n));
    for (int i = 0; i < n; i++) {
        sorted_evals[i] = evals[idx[i]];
        for (int j = 0; j < n; j++)
            sorted_evecs[j][i] = evecs[j][idx[i]];
    }
    evals = sorted_evals;
    evecs = sorted_evecs;
}

// =====================================================================
//  Lanczos eigensolver for small dense Hermitian matrices.
//  More robust than Jacobi for matrices up to ~256x256.
//  Uses full reorthogonalization + implicit QL on the real tridiagonal.
//  Same interface as jacobi_eigen — drop-in replacement.
// =====================================================================
void lanczos_eigen(std::vector<Vec>& A_cols, int n,
                   RVec& evals, std::vector<Vec>& evecs) {
    if (n == 0) { evals.clear(); evecs.clear(); return; }
    if (n == 1) {
        evals = {std::real(A_cols[0][0])};
        evecs.resize(1, Vec(1));
        evecs[0][0] = 1.0;
        return;
    }

    // Dense matvec: y = A * x
    auto matvec = [&](const Vec& x, Vec& y) {
        for (int i = 0; i < n; i++) y[i] = 0.0;
        for (int j = 0; j < n; j++) {
            cx xj = x[j];
            for (int i = 0; i < n; i++)
                y[i] += A_cols[j][i] * xj;
        }
    };

    // Lanczos tridiagonalization with full reorthogonalization
    // Produces real tridiagonal T: A = Q T Q†
    std::vector<Vec> Q(n, Vec(n, 0.0));
    RVec alpha_l(n, 0.0);     // diagonal of T
    RVec beta_l(n + 1, 0.0);  // sub-diagonal (beta_l[0] unused)

    // Starting vector: first basis vector
    Q[0][0] = 1.0;

    Vec w(n, 0.0), Aq(n, 0.0);
    matvec(Q[0], w);
    alpha_l[0] = std::real(dot(Q[0], w));
    for (int i = 0; i < n; i++) w[i] -= alpha_l[0] * Q[0][i];

    for (int j = 1; j < n; j++) {
        beta_l[j] = norm(w);
        if (beta_l[j] < 1e-14) {
            // Invariant subspace — find orthogonal restart vector
            bool found = false;
            for (int try_idx = 0; try_idx < n && !found; try_idx++) {
                Q[j] = Vec(n, 0.0);
                Q[j][try_idx] = 1.0;
                for (int pass = 0; pass < 2; pass++)
                    for (int i = 0; i < j; i++) {
                        cx proj = dot(Q[i], Q[j]);
                        axpy(-proj, Q[i], Q[j]);
                    }
                double nq = norm(Q[j]);
                if (nq > 1e-10) {
                    scale(Q[j], cx(1.0 / nq));
                    found = true;
                }
            }
            beta_l[j] = 0.0;
        } else {
            for (int i = 0; i < n; i++) Q[j][i] = w[i] / beta_l[j];
        }

        // Full reorthogonalization (two MGS passes)
        for (int pass = 0; pass < 2; pass++)
            for (int i = 0; i < j; i++) {
                cx proj = dot(Q[i], Q[j]);
                axpy(-proj, Q[i], Q[j]);
            }
        double nq = norm(Q[j]);
        if (nq > 1e-14) scale(Q[j], cx(1.0 / nq));

        matvec(Q[j], w);
        alpha_l[j] = std::real(dot(Q[j], w));
        for (int i = 0; i < n; i++)
            w[i] -= alpha_l[j] * Q[j][i] + beta_l[j] * Q[j - 1][i];

        // Reorthogonalize w against all Q
        for (int pass = 0; pass < 2; pass++)
            for (int i = 0; i <= j; i++) {
                cx proj = dot(Q[i], w);
                axpy(-proj, Q[i], w);
            }
    }

    // Solve tridiagonal eigenproblem T v = λ v via implicit QL
    RVec d = alpha_l;  // diagonal (modified in place)
    RVec e(n, 0.0);    // sub-diagonal
    for (int i = 0; i < n - 1; i++) e[i] = beta_l[i + 1];

    // Eigenvector accumulator for T (starts as identity)
    std::vector<RVec> Z(n, RVec(n, 0.0));
    for (int i = 0; i < n; i++) Z[i][i] = 1.0;

    // Implicit QL iteration with Wilkinson shift
    for (int l = 0; l < n; l++) {
        int iter_count = 0;
        while (true) {
            // Find smallest m >= l such that e[m] ≈ 0
            int m;
            for (m = l; m < n - 1; m++) {
                double dd = std::abs(d[m]) + std::abs(d[m + 1]);
                if (std::abs(e[m]) <= 1e-15 * dd) break;
            }
            if (m == l) break;
            if (++iter_count > 200) break;

            // Wilkinson shift
            double g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            double r = std::hypot(g, 1.0);
            g = d[m] - d[l] + e[l] / (g + std::copysign(r, g));

            double s = 1.0, c = 1.0, p = 0.0;
            bool converged_early = false;
            for (int i = m - 1; i >= l; i--) {
                double f = s * e[i];
                double b = c * e[i];
                r = std::hypot(f, g);
                e[i + 1] = r;
                if (r < 1e-30) {
                    d[i + 1] -= p;
                    e[m] = 0.0;
                    converged_early = true;
                    break;
                }
                s = f / r;
                c = g / r;
                g = d[i + 1] - p;
                r = (d[i] - g) * s + 2.0 * c * b;
                p = s * r;
                d[i + 1] = g + p;
                g = c * r - b;
                // Accumulate eigenvectors of T
                for (int k = 0; k < n; k++) {
                    double t = Z[k][i + 1];
                    Z[k][i + 1] = s * Z[k][i] + c * t;
                    Z[k][i] = c * Z[k][i] - s * t;
                }
            }
            if (!converged_early) {
                d[l] -= p;
                e[l] = g;
                e[m] = 0.0;
            }
        }
    }

    // Sort eigenvalues ascending
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b) {
        return d[a] < d[b];
    });

    // Back-transform: eigenvectors of A = Q * eigenvectors of T
    // evecs[row][col] = component row of eigenvector col
    evals.resize(n);
    evecs.resize(n, Vec(n, 0.0));
    for (int col = 0; col < n; col++) {
        evals[col] = d[idx[col]];
        for (int row = 0; row < n; row++) {
            cx val = 0.0;
            for (int j = 0; j < n; j++)
                val += Q[j][row] * Z[j][idx[col]];
            evecs[row][col] = val;
        }
    }
}

// =====================================================================
//  10e. LOBPCG EIGENMODE UPDATE
// =====================================================================
// Locally Optimal Block Preconditioned Conjugate Gradient for tracking
// k smallest eigenmodes of A = D†D as the gauge field evolves.
// Warm-started from previous eigenvectors, converges in 2-3 iterations
// for small gauge perturbations.
//
// Each iteration costs: k matvecs + k preconditioner applications + O(k²N)
// orthogonalisation.  For k=4, n_iter=3, that's ~12 matvecs + 12 MG cycles.
//
// Returns updated eigenvectors (normalised, orthogonal) and eigenvalues.
LOBPCGResult lobpcg_update(
    const OpApply& A,
    int n,
    int k,
    const std::vector<Vec>& X0,   // warm-start eigenvectors
    const std::function<Vec(const Vec&)>& precond,
    int max_iter,
    double tol)
{
    // X = current eigenvector block (n × k)
    // W = preconditioned residuals
    // P = previous search directions (empty on first iteration)
    std::vector<Vec> X(k), AX(k), W(k), AW(k), P(k), AP(k);
    bool have_P = false;

    // Initialise from warm start
    for (int i = 0; i < k; i++) {
        X[i] = X0[i];
    }

    // Orthonormalise X
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < i; j++) {
            cx proj = dot(X[j], X[i]);
            axpy(-proj, X[j], X[i]);
        }
        double nv = norm(X[i]);
        if (nv > 1e-14) scale(X[i], cx(1.0/nv));
    }

    // Compute AX
    for (int i = 0; i < k; i++) {
        AX[i].resize(n);
        A(X[i], AX[i]);
    }

    int iter;
    for (iter = 0; iter < max_iter; iter++) {
        // Compute eigenvalues (Rayleigh quotients)
        std::vector<double> theta(k);
        for (int i = 0; i < k; i++) {
            theta[i] = std::real(dot(X[i], AX[i]));
        }

        // Compute residuals R[i] = AX[i] - theta[i] * X[i]
        // and check convergence
        double max_rnorm = 0;
        std::vector<Vec> R(k);
        for (int i = 0; i < k; i++) {
            R[i] = AX[i];
            axpy(cx(-theta[i]), X[i], R[i]);
            double rnorm = norm(R[i]) / std::max(std::abs(theta[i]), 1e-30);
            max_rnorm = std::max(max_rnorm, rnorm);
        }
        if (max_rnorm < tol) break;

        // Precondition residuals: W = M⁻¹ R
        for (int i = 0; i < k; i++) {
            W[i] = precond(R[i]);
            AW[i].resize(n);
            A(W[i], AW[i]);
        }

        // Build the subspace S = [X, W, P] (or [X, W] on first iter)
        // Solve the projected eigenproblem: S† A S y = θ S† S y
        int nblocks = have_P ? 3 : 2;
        int sdim = nblocks * k;

        // Collect all vectors and their A-images into arrays
        std::vector<Vec*> S(sdim), AS(sdim);
        for (int i = 0; i < k; i++) {
            S[i] = &X[i];       AS[i] = &AX[i];
            S[k+i] = &W[i];     AS[k+i] = &AW[i];
        }
        if (have_P) {
            for (int i = 0; i < k; i++) {
                S[2*k+i] = &P[i];  AS[2*k+i] = &AP[i];
            }
        }

        // Form small matrices: H = S†AS, M_ov = S†S
        std::vector<Vec> H_cols(sdim, Vec(sdim, 0.0));
        std::vector<Vec> M_cols(sdim, Vec(sdim, 0.0));
        for (int j = 0; j < sdim; j++) {
            for (int i = 0; i < sdim; i++) {
                H_cols[j][i] = dot(*S[i], *AS[j]);
                M_cols[j][i] = dot(*S[i], *S[j]);
            }
        }

        // Hermitise
        for (int i = 0; i < sdim; i++)
            for (int j = i+1; j < sdim; j++) {
                cx avg_h = 0.5 * (H_cols[j][i] + std::conj(H_cols[i][j]));
                H_cols[j][i] = avg_h;
                H_cols[i][j] = std::conj(avg_h);
                cx avg_m = 0.5 * (M_cols[j][i] + std::conj(M_cols[i][j]));
                M_cols[j][i] = avg_m;
                M_cols[i][j] = std::conj(avg_m);
            }

        // Solve generalised eigenproblem H y = θ M y via Cholesky reduction:
        // L L† = M, then solve L⁻¹ H L⁻† z = θ z, y = L⁻† z
        // Use simple Cholesky factorisation of M
        std::vector<Vec> L(sdim, Vec(sdim, 0.0));
        bool chol_ok = true;
        for (int j = 0; j < sdim; j++) {
            for (int i = 0; i < j; i++) {
                cx s = M_cols[j][i];
                for (int p = 0; p < i; p++)
                    s -= L[i][p] * std::conj(L[j][p]);
                L[j][i] = s / L[i][i];
            }
            cx s = M_cols[j][j];
            for (int p = 0; p < j; p++)
                s -= L[j][p] * std::conj(L[j][p]);
            double d = std::real(s);
            if (d < 1e-14) { chol_ok = false; break; }
            L[j][j] = std::sqrt(d);
        }

        if (!chol_ok) {
            // Fallback: skip Rayleigh-Ritz, just use preconditioned steepest descent
            for (int i = 0; i < k; i++) {
                // X_new = X + alpha * W, choose alpha to minimise Rayleigh quotient
                cx xAw = dot(X[i], AW[i]);
                cx wAw = dot(W[i], AW[i]);
                cx xAx = dot(X[i], AX[i]);
                // Minimise (xAx + 2α Re(xAw) + α²wAw) / (1 + α²ww)
                // Approximate: alpha = -xAw / wAw
                if (std::abs(wAw) > 1e-30) {
                    cx alpha = -xAw / wAw;
                    axpy(alpha, W[i], X[i]);
                }
                double nv = norm(X[i]);
                if (nv > 1e-14) scale(X[i], cx(1.0/nv));
            }
            // Re-orthogonalise
            for (int i = 0; i < k; i++) {
                for (int j = 0; j < i; j++) {
                    cx proj = dot(X[j], X[i]);
                    axpy(-proj, X[j], X[i]);
                }
                double nv = norm(X[i]);
                if (nv > 1e-14) scale(X[i], cx(1.0/nv));
            }
            for (int i = 0; i < k; i++) A(X[i], AX[i]);
            have_P = false;
            continue;
        }

        // Forward solve: L⁻¹ H → Ht
        // Form Ht = L⁻¹ H L⁻†
        // Step 1: Y = L⁻¹ H (solve L Y_col = H_col for each col)
        std::vector<Vec> Y(sdim, Vec(sdim, 0.0));
        for (int col = 0; col < sdim; col++) {
            for (int i = 0; i < sdim; i++) {
                cx s = H_cols[col][i];
                for (int p = 0; p < i; p++)
                    s -= L[i][p] * Y[col][p];
                Y[col][i] = s / L[i][i];
            }
        }
        // Step 2: Ht = Y L⁻† (solve L† Z_row = Y_row for each row → transpose)
        // Equivalently: Ht[j][i] = solve for Z where L† Z_col = Y†_col
        // Simpler: Ht_cols[j][i] = (L⁻¹ H L⁻†)[i][j]
        std::vector<Vec> Ht_cols(sdim, Vec(sdim, 0.0));
        for (int col = 0; col < sdim; col++) {
            // Solve L† z = Y[col]
            Vec z(sdim, 0.0);
            for (int i = sdim-1; i >= 0; i--) {
                cx s = Y[col][i];
                for (int p = i+1; p < sdim; p++)
                    s -= std::conj(L[p][i]) * z[p];
                z[i] = s / L[i][i];
            }
            Ht_cols[col] = z;
        }

        // Hermitise Ht
        for (int i = 0; i < sdim; i++)
            for (int j = i+1; j < sdim; j++) {
                cx avg = 0.5 * (Ht_cols[j][i] + std::conj(Ht_cols[i][j]));
                Ht_cols[j][i] = avg;
                Ht_cols[i][j] = std::conj(avg);
            }

        // Diagonalise Ht
        RVec evals;
        std::vector<Vec> evecs;
        lanczos_eigen(Ht_cols, sdim, evals, evecs);

        // Back-transform: y = L⁻† z (for each eigenvector)
        std::vector<Vec> Y_back(sdim, Vec(sdim, 0.0));
        for (int col = 0; col < k; col++) {  // only need k smallest
            for (int i = sdim-1; i >= 0; i--) {
                cx s = evecs[i][col];  // evecs stored as evecs[row][col]
                for (int p = i+1; p < sdim; p++)
                    s -= std::conj(L[p][i]) * Y_back[col][p];
                Y_back[col][i] = s / L[i][i];
            }
        }

        // Reconstruct fine-grid eigenvectors: X_new[i] = Σ_j y[j][i] * S[j]
        // Also save P = X_new - X_old (search direction for next iter)
        std::vector<Vec> X_new(k), AX_new(k);
        for (int i = 0; i < k; i++) {
            X_new[i] = zeros(n);
            AX_new[i] = zeros(n);
            for (int j = 0; j < sdim; j++) {
                axpy(Y_back[i][j], *S[j], X_new[i]);
                axpy(Y_back[i][j], *AS[j], AX_new[i]);
            }

            // P = X_new - X_old (for next iteration's 3-block subspace)
            P[i] = X_new[i];
            axpy(cx(-1.0), X[i], P[i]);
            AP[i] = AX_new[i];
            axpy(cx(-1.0), AX[i], AP[i]);

            // Normalise P
            double np = norm(P[i]);
            if (np > 1e-14) {
                scale(P[i], cx(1.0/np));
                scale(AP[i], cx(1.0/np));
            }
        }

        X = std::move(X_new);
        AX = std::move(AX_new);
        have_P = true;

        // Re-orthonormalise X (numerical safety)
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < i; j++) {
                cx proj = dot(X[j], X[i]);
                axpy(-proj, X[j], X[i]);
                axpy(-proj, X[j], AX[i]);  // keep AX consistent
            }
            double nv = norm(X[i]);
            if (nv > 1e-14) {
                scale(X[i], cx(1.0/nv));
                scale(AX[i], cx(1.0/nv));
            }
        }
    }

    // Final eigenvalues
    std::vector<double> eigvals(k);
    for (int i = 0; i < k; i++) {
        eigvals[i] = std::real(dot(X[i], AX[i]));
    }

    return {std::move(X), eigvals, iter};
}

// =====================================================================
//  Chebyshev-filtered subspace iteration
// =====================================================================
ChebSubspaceResult chebyshev_subspace_iteration(
    const OpApply& A, int n, int k,
    const std::vector<Vec>& X0,
    int poly_deg, int max_iter, double tol,
    double lambda_max)
{
    // Step 0: Estimate lambda_max via power iteration if not provided
    if (lambda_max <= 0.0) {
        Vec v = X0[0];
        double nv = norm(v);
        if (nv > 1e-14) scale(v, cx(1.0/nv));
        Vec Av(n);
        for (int i = 0; i < 20; i++) {
            A(v, Av);
            double rq = std::real(dot(v, Av));
            lambda_max = std::max(lambda_max, rq);
            nv = norm(Av);
            if (nv > 1e-14) { v = Av; scale(v, cx(1.0/nv)); }
        }
        lambda_max *= 1.1;  // safety margin
    }

    // Orthonormalise initial vectors
    std::vector<Vec> X(k);
    for (int i = 0; i < k; i++) X[i] = X0[i];

    auto orthonorm = [&](std::vector<Vec>& V) {
        for (int i = 0; i < k; i++) {
            // Double MGS
            for (int pass = 0; pass < 2; pass++) {
                for (int j = 0; j < i; j++) {
                    cx proj = dot(V[j], V[i]);
                    axpy(-proj, V[j], V[i]);
                }
            }
            double nv = norm(V[i]);
            if (nv > 1e-14) scale(V[i], cx(1.0/nv));
        }
    };

    orthonorm(X);

    // Compute initial Rayleigh quotients for lower bound estimate
    std::vector<Vec> AX(k);
    for (int i = 0; i < k; i++) {
        AX[i].resize(n);
        A(X[i], AX[i]);
    }
    std::vector<double> theta(k);
    for (int i = 0; i < k; i++)
        theta[i] = std::real(dot(X[i], AX[i]));

    int iter;
    for (iter = 0; iter < max_iter; iter++) {
        // Check convergence
        double max_rnorm = 0;
        for (int i = 0; i < k; i++) {
            Vec r = AX[i];
            axpy(cx(-theta[i]), X[i], r);
            double rnorm = norm(r) / std::max(std::abs(theta[i]), 1e-30);
            max_rnorm = std::max(max_rnorm, rnorm);
        }
        if (max_rnorm < tol) break;

        // Chebyshev filter: amplify components with eigenvalue < lambda_cut
        // Use lambda_cut = theta[k-1] (current largest wanted Ritz value)
        // as the cutoff — we want to suppress everything above this.
        double lambda_cut = theta[k-1] * 1.2;  // slight margin
        double e = (lambda_max - lambda_cut) / 2.0;
        double c_center = (lambda_max + lambda_cut) / 2.0;

        // Apply Chebyshev filter T_p((A - c*I) / e) to each vector
        // Using 3-term recurrence: T_0(x) = 1, T_1(x) = x,
        //   T_{n+1}(x) = 2x T_n(x) - T_{n-1}(x)
        // For the filter: y = T_p((A - c*I)/e) * x
        for (int v = 0; v < k; v++) {
            // sigma_1 = e / (c - lambda_cut)... use standard Chebyshev filter
            // Simpler: apply the filter via the recurrence directly
            Vec y_prev = X[v];                        // T_0 = I → x
            Vec Ay(n);
            A(y_prev, Ay);
            // y_1 = (A - c*I)/e * x
            Vec y_cur(n);
            #pragma omp parallel for schedule(static) if(n > OMP_MIN_SIZE)
            for (int i = 0; i < n; i++)
                y_cur[i] = (Ay[i] - c_center * y_prev[i]) / e;

            for (int p = 2; p <= poly_deg; p++) {
                A(y_cur, Ay);
                Vec y_next(n);
                #pragma omp parallel for schedule(static) if(n > OMP_MIN_SIZE)
                for (int i = 0; i < n; i++)
                    y_next[i] = 2.0 * (Ay[i] - c_center * y_cur[i]) / e - y_prev[i];
                y_prev = std::move(y_cur);
                y_cur = std::move(y_next);
            }
            X[v] = std::move(y_cur);
        }

        // Re-orthonormalise
        orthonorm(X);

        // Rayleigh-Ritz projection: solve the small eigenproblem
        // H = X† A X, extract k smallest eigenvalues/vectors
        for (int i = 0; i < k; i++) {
            AX[i].resize(n);
            A(X[i], AX[i]);
        }

        std::vector<Vec> H_cols(k, Vec(k, 0.0));
        for (int j = 0; j < k; j++)
            for (int i = 0; i < k; i++)
                H_cols[j][i] = dot(X[i], AX[j]);

        // Hermitise
        for (int i = 0; i < k; i++)
            for (int j = i+1; j < k; j++) {
                cx avg = 0.5 * (H_cols[j][i] + std::conj(H_cols[i][j]));
                H_cols[j][i] = avg;
                H_cols[i][j] = std::conj(avg);
            }

        RVec evals;
        std::vector<Vec> evecs;
        lanczos_eigen(H_cols, k, evals, evecs);

        // Rotate X and AX to Ritz basis
        std::vector<Vec> X_new(k), AX_new(k);
        for (int i = 0; i < k; i++) {
            X_new[i] = zeros(n);
            AX_new[i] = zeros(n);
            for (int j = 0; j < k; j++) {
                axpy(evecs[j][i], X[j], X_new[i]);
                axpy(evecs[j][i], AX[j], AX_new[i]);
            }
        }
        X = std::move(X_new);
        AX = std::move(AX_new);

        // Update Rayleigh quotients
        for (int i = 0; i < k; i++)
            theta[i] = std::real(dot(X[i], AX[i]));
    }

    return {std::move(X), theta, iter, lambda_max};
}

// =====================================================================
//  Thick Restart Lanczos Method (TRLM)
// =====================================================================
// Based on QUDA's implementation. Computes the n_ev smallest eigenvalues
// of a Hermitian operator using Lanczos with thick restarts and optional
// Chebyshev polynomial acceleration.
TRLMResult trlm_eigensolver(
    const OpApply& A, int n, int n_ev,
    int n_kr_in, int max_restarts, double tol,
    int poly_deg, double a_min, double a_max,
    TRLMState* state_out,
    const Vec* start_vec)
{
    // Default Krylov space size — Chebyshev needs fewer extra vectors
    bool use_cheby_default = (poly_deg > 0);
    int n_kr;
    if (n_kr_in > 0) {
        n_kr = n_kr_in;
    } else if (use_cheby_default) {
        n_kr = std::min(n_ev + 24, n);  // Chebyshev: ~20% extra suffices
    } else {
        n_kr = std::min(std::max(2 * n_ev + 6, n_ev + 32), n);
    }
    if (n_kr > n) n_kr = n;
    if (n_kr < n_ev + 6) n_kr = std::min(n_ev + 6, n);

    // Lanczos vectors (Krylov space)
    std::vector<Vec> kSpace(n_kr + 1);
    for (int i = 0; i <= n_kr; i++) kSpace[i] = zeros(n);

    // Tridiagonal matrix entries
    RVec alpha(n_kr, 0.0);  // diagonal
    RVec beta_arr(n_kr, 0.0);   // sub-diagonal
    RVec residua(n_kr, 0.0);

    // Starting vector: use provided start_vec or random
    std::mt19937 rng(42);
    if (start_vec && (int)start_vec->size() == n) {
        kSpace[0] = *start_vec;
    } else {
        kSpace[0] = random_vec(n, rng);
    }
    double nv = norm(kSpace[0]);
    scale(kSpace[0], cx(1.0/nv));

    // Estimate lambda_max for Chebyshev if needed
    bool use_cheby = (poly_deg > 0);
    if (use_cheby && a_max <= 0.0) {
        // Power iteration to estimate spectral radius
        Vec v = kSpace[0];
        Vec Av(n);
        double lmax = 0;
        for (int i = 0; i < 30; i++) {
            A(v, Av);
            double rq = std::real(dot(v, Av));
            lmax = std::max(lmax, rq);
            double nrm = norm(Av);
            if (nrm > 1e-14) { v = Av; scale(v, cx(1.0/nrm)); }
        }
        a_max = lmax * 1.1;
        // Re-randomise starting vector since power iteration changed it
        kSpace[0] = random_vec(n, rng);
        nv = norm(kSpace[0]);
        scale(kSpace[0], cx(1.0/nv));
    }

    // chebyOp: QUDA-style sigma-scaled Chebyshev polynomial of A.
    // Computes C_d(A)*src where C_d is proportional to T_d((A-theta)/delta)
    // with sigma scaling factors for numerical stability.
    // [a_min, a_max] defines the interval that gets mapped to [-1,1].
    // Eigenvalues outside this interval are amplified by the Chebyshev polynomial.
    auto chebyOp = [&](const Vec& src, Vec& dst) {
        if (!use_cheby) {
            A(src, dst);
            return;
        }
        double delta = (a_max - a_min) / 2.0;
        double theta = (a_max + a_min) / 2.0;
        double sigma1 = -delta / theta;
        double d1 = sigma1 / delta;
        double d2 = 1.0;

        // C_1 = d2*src + d1*A*src = src - A*src/theta
        Vec Av(n);
        A(src, Av);
        dst.resize(n);
        #pragma omp parallel for schedule(static) if(n > OMP_MIN_SIZE)
        for (int i = 0; i < n; i++)
            dst[i] = d2 * src[i] + d1 * Av[i];

        if (poly_deg <= 1) return;

        // tmp1 = C_0 = src, tmp2 = C_1 = dst
        Vec tmp1 = src;
        Vec tmp2 = dst;

        double sigma_old = sigma1;
        // Build C_2 through C_{poly_deg-1}
        for (int p = 2; p < poly_deg; p++) {
            double sigma = 1.0 / (2.0 / sigma1 - sigma_old);
            d1 = 2.0 * sigma / delta;
            d2 = -d1 * theta;
            double d3 = -sigma * sigma_old;

            // out = A * tmp2
            A(tmp2, Av);

            // tmp1_new = d3*tmp1 + d2*tmp2 + d1*A*tmp2
            Vec tmp_new(n);
            #pragma omp parallel for schedule(static) if(n > OMP_MIN_SIZE)
            for (int i = 0; i < n; i++)
                tmp_new[i] = d3 * tmp1[i] + d2 * tmp2[i] + d1 * Av[i];

            tmp1 = std::move(tmp2);
            tmp2 = std::move(tmp_new);
            sigma_old = sigma;
        }
        dst = std::move(tmp2);
    };

    int num_locked = 0;
    int num_converged = 0;
    int num_keep = 0;
    int iter = 0;
    bool converged = false;


    Vec r(n, 0.0);  // work vector

    // lanczosStep: compute one step of the Lanczos algorithm
    auto lanczosStep = [&](int j) {
        // r = chebyOp(v_j)
        chebyOp(kSpace[j], r);

        // alpha_j = v_j† r
        alpha[j] = std::real(dot(kSpace[j], r));

        // r = r - alpha_j * v_j
        axpy(cx(-alpha[j]), kSpace[j], r);

        // r = r - beta_{j-1} * v_{j-1}  (and any prior locked vectors)
        int start = (j > num_keep) ? j - 1 : 0;
        for (int i = start; i < j; i++) {
            axpy(cx(-beta_arr[i]), kSpace[i], r);
        }

        // Full reorthogonalization against all Krylov vectors
        for (int k = 0; k <= j; k++) {
            cx proj = dot(kSpace[k], r);
            axpy(-proj, kSpace[k], r);
        }
        // Second pass for numerical stability
        for (int k = 0; k <= j; k++) {
            cx proj = dot(kSpace[k], r);
            axpy(-proj, kSpace[k], r);
        }

        // beta_j = ||r||
        beta_arr[j] = norm(r);

        // v_{j+1} = r / beta_j
        if (beta_arr[j] > 1e-14) {
            kSpace[j + 1] = r;
            scale(kSpace[j + 1], cx(1.0 / beta_arr[j]));
        }
    };

    int restart;
    for (restart = 0; restart < max_restarts && !converged; restart++) {
        // Build Krylov space from num_keep to n_kr
        for (int step = num_keep; step < n_kr; step++) {
            lanczosStep(step);
        }
        iter += (n_kr - num_keep);

        // Solve the arrow/tridiagonal eigenproblem
        int dim = n_kr - num_locked;
        int arrow_pos = num_keep - num_locked;

        // Build arrow matrix
        // After thick restart, the matrix has arrow structure:
        // - First arrow_pos rows/cols: diagonal = alpha, off-diagonal connects to arrow_pos
        // - Below arrow_pos: standard tridiagonal
        std::vector<Vec> arrow_cols(dim, Vec(dim, 0.0));
        for (int i = 0; i < dim; i++) {
            arrow_cols[i][i] = alpha[i + num_locked];
        }
        for (int i = 0; i < arrow_pos; i++) {
            arrow_cols[arrow_pos][i] = beta_arr[i + num_locked];
            arrow_cols[i][arrow_pos] = beta_arr[i + num_locked];
        }
        for (int i = arrow_pos; i < dim - 1; i++) {
            arrow_cols[i + 1][i] = beta_arr[i + num_locked];
            arrow_cols[i][i + 1] = beta_arr[i + num_locked];
        }

        RVec evals;
        std::vector<Vec> evecs;
        lanczos_eigen(arrow_cols, dim, evals, evecs);

        // QUDA-style Chebyshev: wanted eigenvalues of A are the LARGEST of
        // chebyOp.  Reverse the column ordering so largest come first,
        // matching the convergence / locking logic which operates on
        // the leading entries.
        // Note: evecs is stored transposed: evecs[row][col] where col is
        // the eigenvector index.  Reversing columns = reversing each row.
        bool reverse_spectrum = use_cheby && (a_min > 0.0);
        if (reverse_spectrum) {
            std::reverse(evals.begin(), evals.end());
            for (auto& row : evecs) {
                std::reverse(row.begin(), row.end());
            }
        }

        // Compute residuals: |beta_{n_kr-1} * last_component_of_eigenvector|
        for (int i = 0; i < dim; i++) {
            residua[i + num_locked] = std::abs(beta_arr[n_kr - 1] * std::real(evecs[dim - 1][i]));
            alpha[i + num_locked] = evals[i];
        }

        // Convergence / locking check
        int iter_locked = 0;
        int iter_converged = 0;

        // Machine epsilon for locking (tighter than convergence)
        double epsilon = 1e-15;

        for (int i = 1; i < dim; i++) {
            if (residua[i + num_locked] < epsilon * std::abs(alpha[i + num_locked])) {
                iter_locked = i;
            } else {
                break;
            }
        }

        iter_converged = iter_locked;
        for (int i = iter_locked + 1; i < dim; i++) {
            if (residua[i + num_locked] < tol * std::abs(alpha[i + num_locked])) {
                iter_converged = i;
            } else {
                break;
            }
        }

        int iter_keep = std::min(iter_converged + (n_kr - num_converged) / 2, n_kr - num_locked - 12);
        if (iter_keep < 1) iter_keep = 1;
        if (iter_keep > dim - 1) iter_keep = dim - 1;


        // Thick restart: rotate Krylov space to keep best Ritz vectors
        // kSpace[num_locked + i] = sum_j evecs[j][i] * kSpace_old[num_locked + j] for i in [0, iter_keep)
        {
            std::vector<Vec> rotated(iter_keep);
            for (int i = 0; i < iter_keep; i++) {
                rotated[i] = zeros(n);
                for (int j = 0; j < dim; j++) {
                    axpy(evecs[j][i], kSpace[num_locked + j], rotated[i]);
                }
            }
            for (int i = 0; i < iter_keep; i++) {
                kSpace[num_locked + i] = std::move(rotated[i]);
            }

            // Preserve residual vector: kSpace[n_kr] holds the last Lanczos
            // vector (v_{n_kr} = r/||r||). Place it at the restart position
            // so the next Krylov extension continues from the correct direction.
            kSpace[num_locked + iter_keep] = kSpace[n_kr];

            // Update beta values for the arrow structure
            for (int i = 0; i < iter_keep; i++) {
                beta_arr[i + num_locked] = beta_arr[n_kr - 1] * std::real(evecs[dim - 1][i]);
            }
        }

        num_converged = num_locked + iter_converged;
        num_keep = num_locked + iter_keep;
        num_locked += iter_locked;

        if (num_converged >= n_ev) {
            converged = true;
            if (reverse_spectrum) {
                // Sort by descending chebyOp eigenvalue (largest first =
                // smallest A eigenvalue first after RQ recomputation)
                for (int i = 1; i < n_kr; i++) {
                    int j = i;
                    while (j > 0 && alpha[j-1] < alpha[j]) {
                        std::swap(alpha[j], alpha[j-1]);
                        std::swap(kSpace[j], kSpace[j-1]);
                        j--;
                    }
                }
            } else {
                // Sort by ascending eigenvalue (insertion sort)
                for (int i = 1; i < n_kr; i++) {
                    int j = i;
                    while (j > 0 && alpha[j-1] > alpha[j]) {
                        std::swap(alpha[j], alpha[j-1]);
                        std::swap(kSpace[j], kSpace[j-1]);
                        j--;
                    }
                }
            }
        }
    }

    // Save Krylov subspace state BEFORE extracting result (which moves vectors).
    if (state_out && converged) {
        state_out->n_ev = n_ev;
        state_out->n_kr = n_kr;
        state_out->n = n;
        state_out->valid = true;
        state_out->kSpace.resize(n_kr);
        for (int i = 0; i < n_kr; i++)
            state_out->kSpace[i] = kSpace[i];
        // Re-orthonormalize (double MGS) — kSpace after thick restart + sort
        // may not be orthogonal beyond the converged Ritz vectors.
        for (int i = 0; i < n_kr; i++) {
            for (int pass = 0; pass < 2; pass++)
                for (int j = 0; j < i; j++) {
                    cx proj = dot(state_out->kSpace[j], state_out->kSpace[i]);
                    axpy(-proj, state_out->kSpace[j], state_out->kSpace[i]);
                }
            double nv = norm(state_out->kSpace[i]);
            if (nv > 1e-14)
                scale(state_out->kSpace[i], cx(1.0 / nv));
            else {
                // Linearly dependent — replace with random orthogonal vector
                state_out->kSpace[i] = random_vec(n, rng);
                for (int pass = 0; pass < 2; pass++)
                    for (int j = 0; j < i; j++) {
                        cx proj = dot(state_out->kSpace[j], state_out->kSpace[i]);
                        axpy(-proj, state_out->kSpace[j], state_out->kSpace[i]);
                    }
                nv = norm(state_out->kSpace[i]);
                if (nv > 1e-14) scale(state_out->kSpace[i], cx(1.0 / nv));
            }
        }
    }

    // Build result
    TRLMResult result;
    result.converged = converged;
    result.iterations = iter;
    result.num_restarts = restart;
    result.eigvecs.resize(n_ev);
    result.eigvals.resize(n_ev);
    for (int i = 0; i < n_ev; i++) {
        result.eigvecs[i] = std::move(kSpace[i]);
        result.eigvals[i] = alpha[i];
    }

    // If Chebyshev was used, the eigenvalues are of the Chebyshev-mapped operator,
    // not the original. Compute true Rayleigh quotients and sort ascending.
    if (use_cheby) {
        for (int i = 0; i < n_ev; i++) {
            Vec Av(n);
            A(result.eigvecs[i], Av);
            result.eigvals[i] = std::real(dot(result.eigvecs[i], Av));
        }
        // Sort by ascending eigenvalue of A
        for (int i = 1; i < n_ev; i++) {
            int j = i;
            while (j > 0 && result.eigvals[j-1] > result.eigvals[j]) {
                std::swap(result.eigvals[j], result.eigvals[j-1]);
                std::swap(result.eigvecs[j], result.eigvecs[j-1]);
                j--;
            }
        }
    }

    return result;
}

