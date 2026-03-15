#include "solvers.h"
#include "multigrid.h"
#include "eigensolver.h"
#include "linalg.h"
#include "coarse_op.h"
#include <cmath>
#include <algorithm>
#include <iostream>

// ---------------------------------------------------------------
//  Unpreconditioned CG for HPD operator (D†D)
// ---------------------------------------------------------------
CGResult cg_solve(
    const OpApply& A,
    int n,
    const Vec& rhs,
    int max_iter,
    double tol
) {
    double rhs_norm = norm(rhs);
    if (rhs_norm < 1e-30) return {zeros(n), 0, 0.0};

    Vec x = zeros(n);
    // r = rhs - A*x = rhs (since x=0)
    Vec r(rhs);
    Vec p(r);
    cx rr = dot(r, r);
    int iter = 0;

    while (iter < max_iter) {
        Vec Ap(n);
        A(p, Ap);
        cx pAp = dot(p, Ap);
        cx alpha = rr / pAp;

        #pragma omp parallel for schedule(static) if(n > OMP_MIN_SIZE)
        for (int i = 0; i < n; i++) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }
        iter++;

        double rnorm = norm(r);
        if (rnorm / rhs_norm < tol) break;

        cx rr_new = dot(r, r);
        cx beta = rr_new / rr;
        rr = rr_new;

        #pragma omp parallel for schedule(static) if(n > OMP_MIN_SIZE)
        for (int i = 0; i < n; i++)
            p[i] = r[i] + beta * p[i];
    }

    double final_res = norm(r) / rhs_norm;
    return {x, iter, final_res};
}

// ---------------------------------------------------------------
//  Preconditioned CG for HPD operator with HPD preconditioner
//  Requires M^{-1} to be hermitian positive definite (symmetric MG).
// ---------------------------------------------------------------
CGResult cg_solve_precond(
    const OpApply& A,
    int n,
    const Vec& rhs,
    const std::function<Vec(const Vec&)>& precond,
    int max_iter,
    double tol
) {
    double rhs_norm = norm(rhs);
    if (rhs_norm < 1e-30) return {zeros(n), 0, 0.0};

    Vec x = zeros(n);
    Vec r(rhs);
    Vec z = precond(r);
    Vec p(z);
    cx rz = dot(r, z);
    int iter = 0;

    while (iter < max_iter) {
        Vec Ap(n);
        A(p, Ap);
        cx pAp = dot(p, Ap);
        cx alpha = rz / pAp;

        #pragma omp parallel for schedule(static) if(n > OMP_MIN_SIZE)
        for (int i = 0; i < n; i++) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }
        iter++;

        double rnorm = norm(r);
        if (rnorm / rhs_norm < tol) break;

        z = precond(r);
        cx rz_new = dot(r, z);
        cx beta = rz_new / rz;
        rz = rz_new;

        #pragma omp parallel for schedule(static) if(n > OMP_MIN_SIZE)
        for (int i = 0; i < n; i++)
            p[i] = z[i] + beta * p[i];
    }

    double final_res = norm(r) / rhs_norm;
    return {x, iter, final_res};
}

// ---------------------------------------------------------------
//  Deflated Preconditioned CG for HPD operator
//  Uses deflation vectors U to project out small eigenvalue components
//  before running preconditioned CG on the deflated system.
//
//  Algorithm (Saad et al.):
//    1. Project initial guess: x0 = U * (U†AU)^{-1} * U† * b
//    2. Deflated residual: r0 = b - A*x0
//    3. Run preconditioned CG with deflation projection applied
//       to the preconditioned residual at each step:
//       z = (I - U*(U†AU)^{-1}*U†*A) * M^{-1} * r
//       This ensures CG doesn't waste work on deflated components.
// ---------------------------------------------------------------
CGResult cg_solve_deflated(
    const OpApply& A,
    int n,
    const Vec& rhs,
    const std::function<Vec(const Vec&)>& precond,
    const std::vector<Vec>& defl_vecs,   // orthonormal deflation vectors
    const std::vector<double>& defl_vals, // eigenvalues (for initial projection)
    int max_iter,
    double tol
) {
    int ndefl = (int)defl_vecs.size();
    double rhs_norm = norm(rhs);
    if (rhs_norm < 1e-30) return {zeros(n), 0, 0.0};

    // Precompute AU and E = U†AU (small dense ndefl × ndefl matrix)
    std::vector<Vec> AU(ndefl);
    for (int i = 0; i < ndefl; i++) {
        AU[i].resize(n);
        A(defl_vecs[i], AU[i]);
    }
    std::vector<Vec> E_cols(ndefl, Vec(ndefl, 0.0));
    for (int i = 0; i < ndefl; i++)
        for (int j = 0; j < ndefl; j++)
            E_cols[j][i] = dot(defl_vecs[i], AU[j]);

    // E^{-1}: solve via dense LU (E is small HPD matrix)
    // Build CoarseOp wrapper for E
    CoarseOp E_op;
    E_op.dim = ndefl;
    E_op.mat = E_cols;

    // Precompute E^{-1} * (AU)† for cheap projection.
    // Projection: P_defl(z) = z - U * E^{-1} * (AU)† * z
    //   = z - U * W * z  where W[i][j] = sum_k E^{-1}[i][k] * conj(AU[k][j])
    // This avoids matvecs during CG — only dot products.

    // Initial guess from deflation space: x0 = U * E^{-1} * U† * b
    Vec Utb(ndefl, 0.0);
    for (int i = 0; i < ndefl; i++)
        Utb[i] = dot(defl_vecs[i], rhs);
    Vec c0 = E_op.solve(Utb);
    Vec x = zeros(n);
    for (int i = 0; i < ndefl; i++)
        axpy(c0[i], defl_vecs[i], x);

    // r = b - A*x
    Vec Ax(n);
    A(x, Ax);
    Vec r(n);
    for (int i = 0; i < n; i++) r[i] = rhs[i] - Ax[i];

    // Deflation-projected preconditioned residual:
    // z = M^{-1}r - U * E^{-1} * U† * A * M^{-1}r
    //   = M^{-1}r - U * E^{-1} * (AU)† * M^{-1}r  (since E = U†AU is Hermitian)
    // Cost: ndefl dot products + ndefl×ndefl solve + ndefl axpys (no matvec!)
    auto apply_defl_projection = [&](const Vec& z_in) -> Vec {
        // Compute (AU)† * z_in
        Vec UtAz(ndefl, 0.0);
        for (int i = 0; i < ndefl; i++)
            UtAz[i] = dot(AU[i], z_in);
        // Solve E * c = UtAz
        Vec c = E_op.solve(UtAz);
        // z_in - U*c
        Vec result = z_in;
        for (int i = 0; i < ndefl; i++)
            axpy(-c[i], defl_vecs[i], result);
        return result;
    };

    Vec z = apply_defl_projection(precond(r));
    Vec p(z);
    cx rz = dot(r, z);
    int iter = 0;

    while (iter < max_iter) {
        Vec Ap(n);
        A(p, Ap);
        cx pAp = dot(p, Ap);
        cx alpha = rz / pAp;

        #pragma omp parallel for schedule(static) if(n > OMP_MIN_SIZE)
        for (int i = 0; i < n; i++) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }
        iter++;

        double rnorm = norm(r);
        if (rnorm / rhs_norm < tol) break;

        z = apply_defl_projection(precond(r));
        cx rz_new = dot(r, z);
        cx beta = rz_new / rz;
        rz = rz_new;

        #pragma omp parallel for schedule(static) if(n > OMP_MIN_SIZE)
        for (int i = 0; i < n; i++)
            p[i] = z[i] + beta * p[i];
    }

    double final_res = norm(r) / rhs_norm;
    return {x, iter, final_res};
}

// ---------------------------------------------------------------
//  GCR (Generalized Conjugate Residual) with truncation
//  Handles non-symmetric preconditioners (like MG with MR smoother)
//  while exploiting HPD structure of A.
//  Equivalent to FGMRES but with explicit direction storage and
//  minimization via A-inner products on the residual.
//  No restart needed — truncation window limits memory.
// ---------------------------------------------------------------
FCGResult fcg_solve(
    const OpApply& A,
    int n,
    const Vec& rhs,
    const std::function<Vec(const Vec&)>& precond,
    int max_iter,
    double tol,
    int trunc   // truncation window (like FGMRES restart length)
) {
    double rhs_norm = norm(rhs);
    if (rhs_norm < 1e-30) return {zeros(n), 0, 0.0};

    Vec x = zeros(n);
    Vec r(rhs);

    // Direction storage (sliding window)
    std::vector<Vec> P_dirs;    // preconditioned search directions
    std::vector<Vec> AP_dirs;   // A * P_dirs (for orthogonalization)
    std::vector<double> ApAp;   // <A p_i, A p_i>

    int iter = 0;
    while (iter < max_iter) {
        double rnorm = norm(r);
        if (rnorm / rhs_norm < tol) break;

        // Precondition
        Vec z = precond(r);

        // Compute Az
        Vec Az(n);
        A(z, Az);

        // Orthogonalise Az against stored AP directions
        // and apply same coefficients to z (maintain correspondence)
        Vec p = z;
        Vec Ap = Az;
        int nd = (int)AP_dirs.size();
        for (int i = 0; i < nd; i++) {
            cx beta = dot(Az, AP_dirs[i]) / ApAp[i];
            #pragma omp parallel for schedule(static) if(n > OMP_MIN_SIZE)
            for (int j = 0; j < n; j++) {
                p[j] -= beta * P_dirs[i][j];
                Ap[j] -= beta * AP_dirs[i][j];
            }
        }

        // Step size: minimise |r - alpha * Ap|^2
        // alpha = <Ap, r> / <Ap, Ap>
        double ApAp_val = dot(Ap, Ap).real();
        if (ApAp_val < 1e-30) {
            // Direction collapsed — clear history and restart
            P_dirs.clear(); AP_dirs.clear(); ApAp.clear();
            continue;
        }
        cx alpha = dot(Ap, r) / ApAp_val;

        // Update
        #pragma omp parallel for schedule(static) if(n > OMP_MIN_SIZE)
        for (int j = 0; j < n; j++) {
            x[j] += alpha * p[j];
            r[j] -= alpha * Ap[j];
        }
        iter++;

        // Store in sliding window
        if (nd >= trunc) {
            P_dirs.erase(P_dirs.begin());
            AP_dirs.erase(AP_dirs.begin());
            ApAp.erase(ApAp.begin());
        }
        P_dirs.push_back(std::move(p));
        AP_dirs.push_back(std::move(Ap));
        ApAp.push_back(ApAp_val);
    }

    double final_res = norm(r) / rhs_norm;
    return {x, iter, final_res};
}

FGMRESResult fgmres_solve_generic(
    const OpApply& A,
    int n,
    const Vec& rhs,
    const std::function<Vec(const Vec&)>& precond,
    int restart,
    int max_iter,
    double tol,
    int n_ritz_harvest
) {
    double rhs_norm = norm(rhs);
    if (rhs_norm < 1e-30) return {zeros(n), 0, 0.0, {}};

    Vec x = zeros(n);
    int total_iter = 0;

    std::vector<RitzPair> all_ritz;

    while (total_iter < max_iter) {
        // compute residual
        Vec Ax(n);
        A(x, Ax);
        Vec r(n);
        #pragma omp parallel for schedule(static) if(n > OMP_MIN_SIZE)
        for (int i = 0; i < n; i++) r[i] = rhs[i] - Ax[i];
        double beta = norm(r);
        if (beta / rhs_norm < tol) break;

        int m = std::min(restart, max_iter - total_iter);

        // Arnoldi basis V[0..m], preconditioned vectors Z[0..m-1]
        std::vector<Vec> V(m+1), Z(m);
        V[0] = r;
        scale(V[0], cx(1.0/beta));

        // Hessenberg matrix H[i][j], i=0..m, j=0..m-1
        // H_raw stores the unreduced Hessenberg (before Givens) for Ritz extraction
        std::vector<Vec> H(m+1, Vec(m, 0.0));
        std::vector<Vec> H_raw(m+1, Vec(m, 0.0));

        int k;
        Vec g(m+1, 0.0);
        g[0] = beta;

        // Givens rotations
        Vec cs(m, 0.0), sn(m, 0.0);

        for (k = 0; k < m; k++) {
            // precondition: Z[k] = M^{-1} V[k]
            Z[k] = precond(V[k]);

            // Intra-solve deflation: augment preconditioned vector with
            // deflation from Ritz pairs accumulated in previous restart cycles.
            // Within a single solve the operator is fixed, so these are valid.
            for (const auto& dp : all_ritz) {
                if (dp.value > 1e-14) {
                    cx coeff = dot(dp.vector, V[k]) / dp.value;
                    axpy(coeff, dp.vector, Z[k]);
                }
            }

            // w = A * Z[k]
            Vec w(n);
            A(Z[k], w);

            // Arnoldi orthogonalisation (modified Gram-Schmidt with
            // DGKS-style reorthogonalisation for numerical robustness)
            for (int j = 0; j <= k; j++) {
                H[j][k] = dot(V[j], w);
                axpy(-H[j][k], V[j], w);
            }
            // Second MGS pass: correct for loss of orthogonality
            for (int j = 0; j <= k; j++) {
                cx correction = dot(V[j], w);
                H[j][k] += correction;
                axpy(-correction, V[j], w);
            }
            H[k+1][k] = norm(w);

            // Save unreduced Hessenberg column for Ritz extraction
            for (int j = 0; j <= k+1; j++) H_raw[j][k] = H[j][k];

            if (std::abs(H[k+1][k]) < 1e-14) {
                k++;
                break;
            }
            V[k+1] = w;
            scale(V[k+1], cx(1.0 / H[k+1][k]));

            // Apply previous Givens rotations to new column of H.
            // Convention: rotation matrix is [[conj(cs), conj(sn)], [-sn, cs]]
            // applied to [h_j; h_{j+1}].  This is the unitary form for complex
            // Hessenberg QR (cf. Saad, "Iterative Methods for Sparse Linear
            // Systems", adapted for complex entries).
            for (int j = 0; j < k; j++) {
                cx h_j  = H[j][k];
                cx h_j1 = H[j+1][k];
                H[j][k]   = std::conj(cs[j]) * h_j + std::conj(sn[j]) * h_j1;
                H[j+1][k] = -sn[j] * h_j + cs[j] * h_j1;
            }

            // Compute new Givens rotation to zero H[k+1][k].
            // cs[k] and sn[k] are complex; denom is real and positive.
            double a = std::abs(H[k][k]);
            double b = std::abs(H[k+1][k]);
            double denom = std::sqrt(a*a + b*b);
            cs[k] = H[k][k] / denom;
            sn[k] = H[k+1][k] / denom;

            g[k+1] = -sn[k] * g[k];
            g[k]   = std::conj(cs[k]) * g[k];

            H[k][k]   = denom;  // Real and positive: sqrt(|H[k][k]|^2 + |H[k+1][k]|^2)
            H[k+1][k] = 0.0;

            total_iter++;

            if (std::abs(g[k+1]) / rhs_norm < tol) {
                k++;
                break;
            }
        }

        int kk = k;  // actual number of Arnoldi steps

        // Solve the upper triangular system H y = g
        Vec y(kk, 0.0);
        for (int i = kk-1; i >= 0; i--) {
            cx s = g[i];
            for (int j = i+1; j < kk; j++) s -= H[i][j] * y[j];
            y[i] = s / H[i][i];
        }

        // Update solution: x += Z * y
        for (int j = 0; j < kk; j++)
            axpy(y[j], Z[j], x);

        // =========================================================
        //  RITZ EXTRACTION FROM HESSENBERG (zero additional matvecs)
        // =========================================================
        //  The Arnoldi relation gives A Z_k = V_{k+1} H̃_k, so the
        //  Rayleigh quotient of A in the Z-basis is:
        //    G = Z† A Z = Z† V_{k+1} H̃_k
        //  Since V is orthonormal: V†V = I, and the Z vectors are
        //  the preconditioned V vectors.
        //
        //  For standard Ritz extraction we form G_k = H̃_k^† H̃_k
        //  (the kk×kk matrix) from the unreduced Hessenberg H_raw.
        //  Its eigenvalues approximate eigenvalues of A=D†D and its
        //  eigenvectors, mapped back via Z, give the Ritz vectors.
        //  This requires ZERO additional D†D applications.
        // =========================================================
        if (kk >= n_ritz_harvest && n_ritz_harvest > 0) {
            // Form G = H̃† H̃  (kk×kk Hermitian matrix)
            // H̃ is (kk+1)×kk stored in H_raw[0..kk][0..kk-1]
            // G[j][i] = Σ_{p=0}^{kk} conj(H_raw[p][i]) * H_raw[p][j]
            std::vector<Vec> G_cols(kk, Vec(kk, 0.0));
            for (int j = 0; j < kk; j++)
                for (int i = 0; i < kk; i++)
                    for (int p = 0; p <= kk; p++)
                        G_cols[j][i] += std::conj(H_raw[p][i]) * H_raw[p][j];

            // Hermitise (should be exact, but enforce numerically)
            for (int i = 0; i < kk; i++)
                for (int j = i+1; j < kk; j++) {
                    cx avg = 0.5 * (G_cols[j][i] + std::conj(G_cols[i][j]));
                    G_cols[j][i] = avg;
                    G_cols[i][j] = std::conj(avg);
                }

            // Diagonalise G with Jacobi
            RVec evals;
            std::vector<Vec> evecs;
            lanczos_eigen(G_cols, kk, evals, evecs);

            // Reconstruct Ritz vectors on the fine grid via Z basis.
            // Ritz vector p = Σ_j evecs[j][p] * Z[j]
            int nh = std::min(n_ritz_harvest, kk);
            for (int p = 0; p < nh; p++) {
                RitzPair rp;
                rp.value = evals[p];
                rp.vector = zeros(n);
                for (int j = 0; j < kk; j++)
                    axpy(evecs[j][p], Z[j], rp.vector);
                double nv = norm(rp.vector);
                if (nv > 1e-14) scale(rp.vector, cx(1.0/nv));
                all_ritz.push_back(rp);
            }

            // Keep bounded: retain only the smallest-eigenvalue pairs
            if ((int)all_ritz.size() > 2 * n_ritz_harvest) {
                std::sort(all_ritz.begin(), all_ritz.end(),
                          [](const RitzPair& a, const RitzPair& b) {
                              return a.value < b.value;
                          });
                all_ritz.resize(n_ritz_harvest);
            }
        }

        if (std::abs(g[kk]) / rhs_norm < tol) break;
    }

    // final residual
    Vec Ax(n);
    A(x, Ax);
    double res = 0;
    #pragma omp parallel for reduction(+:res) schedule(static) if(n > OMP_MIN_SIZE)
    for (int i = 0; i < n; i++) res += std::norm(rhs[i] - Ax[i]);
    res = std::sqrt(res) / rhs_norm;

    if (total_iter >= max_iter && res > tol) {
        std::cerr << "[WARNING] FGMRES did not converge in " << max_iter
                  << " iterations. Residual: " << std::scientific << res
                  << " (target: " << tol << ")\n";
    }

    return {x, total_iter, res, all_ritz};
}

// Backward-compatible wrapper: single-level MG preconditioner
FGMRESResult fgmres_solve(
    const DiracOp& D, const Vec& rhs,
    Prolongator& P, CoarseOp& Ac,
    int restart, int max_iter, double tol,
    int pre_smooth, int post_smooth,
    int n_ritz_harvest)
{
    OpApply A = [&D](const Vec& s, Vec& d){ D.apply_DdagD(s, d); };
    auto precond = [&](const Vec& v) {
        return mg_vcycle(D, P, Ac, v, pre_smooth, post_smooth);
    };
    return fgmres_solve_generic(A, D.lat.ndof, rhs, precond,
                                restart, max_iter, tol, n_ritz_harvest);
}

// Multi-level MG preconditioner wrapper for FGMRES
FGMRESResult fgmres_solve_mg(
    const OpApply& A, int n, const Vec& rhs,
    MGHierarchy& mg,
    int restart, int max_iter, double tol,
    int n_ritz_harvest)
{
    auto precond = [&mg](const Vec& v) { return mg.precondition(v); };
    return fgmres_solve_generic(A, n, rhs, precond,
                                restart, max_iter, tol, n_ritz_harvest);
}

// FCG with MG preconditioner convenience wrapper
FCGResult fcg_solve_mg(
    const OpApply& A, int n, const Vec& rhs,
    MGHierarchy& mg,
    int max_iter, double tol,
    int truncation)
{
    auto precond = [&mg](const Vec& v) { return mg.precondition(v); };
    return fcg_solve(A, n, rhs, precond, max_iter, tol, truncation);
}
