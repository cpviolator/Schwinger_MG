#include "eigensolver.h"
#include "linalg.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <iomanip>

// =========================================================================
// trlm_update: warm-start TRLM with previous eigenvectors
// =========================================================================
// Uses the n_ev converged eigenvectors from the previous operator as the
// initial Krylov vectors (locked set) for TRLM on the new operator.
// The Lanczos extends from a residual direction, and the previous eigenvectors
// bias convergence toward the wanted eigenspace.
//
// Cost: same as TRLM but converges faster because the initial subspace
// is already close to the wanted eigenspace.
TRLMResult trlm_update(
    const OpApply& A_new, const TRLMState& state,
    int max_restarts, double tol,
    TRLMState* state_out)
{
    int n = state.n;
    int n_ev = state.n_ev;
    int n_kr = state.n_kr;

    // Use the first eigenvector (smallest eigenvalue) as the starting
    // Lanczos vector. This biases the Krylov space toward the low end
    // of the spectrum where our wanted eigenvectors live.
    return trlm_eigensolver(A_new, n, n_ev, n_kr,
                             max_restarts, tol,
                             /*poly_deg=*/0, /*a_min=*/0.0, /*a_max=*/0.0,
                             state_out, &state.kSpace[0]);
}

// =========================================================================
// rr_evolve: Rayleigh-Ritz eigenvector evolution
// =========================================================================
// Project A_new onto the subspace spanned by old_eigvecs, diagonalise the
// small projected matrix, and return the rotated eigenvectors with updated
// eigenvalues. This is exact within the subspace — no perturbation theory
// truncation — and costs only k matvecs + O(k²) inner products.
//
// When applied at every MD step, the subspace tracks the true eigenvectors
// as long as they don't leak out of span{v_1,...,v_k}. Residuals indicate
// subspace quality: small residuals mean the eigenvectors are well-contained.
RREvolveResult rr_evolve(
    const OpApply& A_new,
    const std::vector<Vec>& old_eigvecs,
    int n)
{
    int k = (int)old_eigvecs.size();
    RREvolveResult result;
    result.matvecs = k;

    // Compute A_new * v_i for each eigenvector
    std::vector<Vec> AV(k);
    for (int i = 0; i < k; i++) {
        AV[i].resize(n);
        A_new(old_eigvecs[i], AV[i]);
    }

    // Form M_ij = v_i† A_new v_j  (k × k Hermitian matrix)
    std::vector<Vec> M_cols(k, Vec(k, 0.0));
    for (int j = 0; j < k; j++)
        for (int i = 0; i <= j; i++) {
            cx val = dot(old_eigvecs[i], AV[j]);
            M_cols[j][i] = val;
            if (i != j) M_cols[i][j] = std::conj(val);
        }

    // Diagonalise M
    RVec evals;
    std::vector<Vec> evecs;
    lanczos_eigen(M_cols, k, evals, evecs);
    // evals sorted ascending, evecs[row][col] = row-th component of col-th eigenvector

    // Store rotation matrix (eigenvectors of M) for forecasting
    result.rotation = evecs;  // k×k, evecs[row][col]

    // Rotate eigenvectors: new_v_i = sum_j evecs[j][i] * old_v_j
    // Also rotate AV for residual computation
    result.eigvecs.resize(k);
    result.eigvals.resize(k);
    std::vector<Vec> AV_rot(k);
    for (int i = 0; i < k; i++) {
        result.eigvecs[i] = zeros(n);
        AV_rot[i] = zeros(n);
        for (int j = 0; j < k; j++) {
            axpy(evecs[j][i], old_eigvecs[j], result.eigvecs[i]);
            axpy(evecs[j][i], AV[j], AV_rot[i]);
        }
        result.eigvals[i] = evals[i];
    }

    // Compute residuals ||A v_i - λ_i v_i|| / ||A v_i||
    result.max_residual = 0;
    for (int i = 0; i < k; i++) {
        Vec r = AV_rot[i];
        double av_norm = norm(r);
        axpy(cx(-evals[i]), result.eigvecs[i], r);
        double rel_res = norm(r) / std::max(av_norm, 1e-30);
        result.max_residual = std::max(result.max_residual, rel_res);
    }

    return result;
}

// Force-based eigenvector evolution (zero full matvecs).
// Uses perturbation: δ(D†D) = δD† D + D† δD + δD† δD
// Arrow matrix update: T_ij = λ_i δ_ij + (δD v_i)†(D v_j) + (D v_i)†(δD v_j) + (δD v_i)†(δD v_j)
// Then diagonalise T and rotate eigenvectors + auxiliary Dv vectors.
ForceEvolveResult force_evolve(
    const std::vector<Vec>& eigvecs,
    const std::vector<double>& eigvals,
    const std::vector<Vec>& Dv,
    const std::function<void(const Vec&, Vec&)>& apply_deltaD,
    const std::function<void(const Vec&, Vec&)>& apply_deltaD_dag,
    int n)
{
    int k = (int)eigvecs.size();
    ForceEvolveResult result;
    result.matvecs = 0;

    // Compute δD v_i for each eigenvector
    std::vector<Vec> dDv(k);
    for (int i = 0; i < k; i++) {
        dDv[i].resize(n);
        apply_deltaD(eigvecs[i], dDv[i]);
    }

    // Build the k×k updated projected matrix:
    // δ(D†D) = (D+δD)†(D+δD) - D†D = δD†D + D†δD + δD†δD
    // v_i† δ(D†D) v_j = (δD v_i)†(D v_j) + (D v_i)†(δD v_j) + (δD v_i)†(δD v_j)
    std::vector<Vec> T_cols(k, Vec(k, 0.0));
    for (int j = 0; j < k; j++) {
        for (int i = 0; i <= j; i++) {
            cx val = dot(dDv[i], Dv[j]) + dot(Dv[i], dDv[j]) + dot(dDv[i], dDv[j]);
            if (i == j) val += eigvals[i];
            T_cols[j][i] = val;
            if (i != j) T_cols[i][j] = std::conj(val);
        }
    }

    // Diagonalise T
    RVec evals;
    std::vector<Vec> evecs;
    lanczos_eigen(T_cols, k, evals, evecs);

    // Rotate eigenvectors and update Dv to D_new v_new_i
    // D_new v_j = (D_old + δD) v_j = Dv[j] + dDv[j]
    result.eigvecs.resize(k);
    result.eigvals.resize(k);
    result.Dv.resize(k);
    for (int i = 0; i < k; i++) {
        result.eigvecs[i] = zeros(n);
        result.Dv[i] = zeros(n);
        for (int j = 0; j < k; j++) {
            cx c = evecs[j][i];
            axpy(c, eigvecs[j], result.eigvecs[i]);
            // D_new v_j ≈ Dv[j] + dDv[j]
            axpy(c, Dv[j], result.Dv[i]);
            axpy(c, dDv[j], result.Dv[i]);
        }
        result.eigvals[i] = evals[i];
    }

    // No residual computed (would require a matvec).
    // Caller should periodically verify with a real matvec.
    result.max_residual = -1;

    return result;
}

// Force-based evolution with pre-computed delta_D vectors
ForceEvolveResult force_evolve_precomputed(
    const std::vector<Vec>& eigvecs,
    const std::vector<double>& eigvals,
    const std::vector<Vec>& Dv,
    const std::vector<Vec>& dDv,
    int n)
{
    int k = (int)eigvecs.size();
    ForceEvolveResult result;
    result.matvecs = 0;

    std::vector<Vec> T_cols(k, Vec(k, 0.0));
    for (int j = 0; j < k; j++) {
        for (int i = 0; i <= j; i++) {
            cx val = dot(dDv[i], Dv[j]) + dot(Dv[i], dDv[j]) + dot(dDv[i], dDv[j]);
            if (i == j) val += eigvals[i];
            T_cols[j][i] = val;
            if (i != j) T_cols[i][j] = std::conj(val);
        }
    }

    RVec evals;
    std::vector<Vec> evecs;
    lanczos_eigen(T_cols, k, evals, evecs);

    result.eigvecs.resize(k);
    result.eigvals.resize(k);
    result.Dv.resize(k);
    for (int i = 0; i < k; i++) {
        result.eigvecs[i] = zeros(n);
        result.Dv[i] = zeros(n);
        for (int j = 0; j < k; j++) {
            cx c = evecs[j][i];
            axpy(c, eigvecs[j], result.eigvecs[i]);
            axpy(c, Dv[j], result.Dv[i]);
            axpy(c, dDv[j], result.Dv[i]);
        }
        result.eigvals[i] = evals[i];
    }
    result.max_residual = -1;
    return result;
}
