#pragma once
#include "types.h"
#include <vector>

struct LOBPCGResult {
    std::vector<Vec> eigvecs;
    std::vector<double> eigvals;
    int iterations;
};

struct ChebSubspaceResult {
    std::vector<Vec> eigvecs;
    std::vector<double> eigvals;
    int iterations;
    double lambda_max_used;
};

struct TRLMResult {
    std::vector<Vec> eigvecs;
    std::vector<double> eigvals;
    int iterations;
    int num_restarts;
    bool converged;
};

void jacobi_eigen(std::vector<Vec>& A_cols, int n,
                  RVec& evals, std::vector<Vec>& evecs);

void lanczos_eigen(std::vector<Vec>& A_cols, int n,
                   RVec& evals, std::vector<Vec>& evecs);

ChebSubspaceResult chebyshev_subspace_iteration(
    const OpApply& A, int n, int k,
    const std::vector<Vec>& X0,
    int poly_deg = 20, int max_iter = 10, double tol = 1e-8,
    double lambda_max = 0.0);

LOBPCGResult lobpcg_update(
    const OpApply& A, int n, int k,
    const std::vector<Vec>& X0,
    const std::function<Vec(const Vec&)>& precond,
    int max_iter = 5, double tol = 1e-8);

// Thick Restart Lanczos Method with optional Chebyshev acceleration.
// Computes n_ev smallest eigenvalues of Hermitian operator A.
//   A: operator to eigensolve
//   n: vector dimension
//   n_ev: number of eigenvalues wanted
//   n_kr: Krylov space size (must be >= n_ev + 6)
//   max_restarts: maximum restart iterations
//   tol: convergence tolerance
//   poly_deg: Chebyshev polynomial degree (0 = no Chebyshev)
//   a_min: Chebyshev lower bound (0 = auto)
//   a_max: Chebyshev upper bound (0 = auto-estimate)
TRLMResult trlm_eigensolver(
    const OpApply& A, int n, int n_ev,
    int n_kr = 0, int max_restarts = 100, double tol = 1e-10,
    int poly_deg = 0, double a_min = 0.0, double a_max = 0.0);
