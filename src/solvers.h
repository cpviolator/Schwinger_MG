#pragma once
#include "types.h"
#include "linalg.h"
#include "dirac.h"
#include "prolongator.h"
#include "coarse_op.h"
#include <vector>
#include <functional>

// Forward declarations
struct MGHierarchy;

struct RitzPair {
    double value;
    Vec vector;
};

struct FGMRESResult {
    Vec solution;
    int iterations;
    double final_residual;
    std::vector<RitzPair> ritz_pairs;
};

struct CGResult {
    Vec solution;
    int iterations;
    double final_residual;
};

struct FCGResult {
    Vec solution;
    int iterations;
    double final_residual;
};

CGResult cg_solve(
    const OpApply& A, int n, const Vec& rhs,
    int max_iter, double tol);

CGResult cg_solve_x0(
    const OpApply& A, int n, const Vec& rhs,
    const Vec& x0, int max_iter, double tol);

CGResult cg_solve_precond(
    const OpApply& A, int n, const Vec& rhs,
    const std::function<Vec(const Vec&)>& precond,
    int max_iter, double tol);

CGResult cg_solve_deflated(
    const OpApply& A, int n, const Vec& rhs,
    const std::function<Vec(const Vec&)>& precond,
    const std::vector<Vec>& defl_vecs,
    const std::vector<double>& defl_vals,
    int max_iter, double tol);

FCGResult fcg_solve(
    const OpApply& A, int n, const Vec& rhs,
    const std::function<Vec(const Vec&)>& precond,
    int max_iter, double tol, int trunc = 30);

FGMRESResult fgmres_solve_generic(
    const OpApply& A, int n, const Vec& rhs,
    const std::function<Vec(const Vec&)>& precond,
    int restart, int max_iter, double tol,
    int n_ritz_harvest = 4);

FGMRESResult fgmres_solve(
    const DiracOp& D, const Vec& rhs,
    Prolongator& P, CoarseOp& Ac,
    int restart, int max_iter, double tol,
    int pre_smooth = 3, int post_smooth = 3,
    int n_ritz_harvest = 4);

FGMRESResult fgmres_solve_mg(
    const OpApply& A, int n, const Vec& rhs,
    MGHierarchy& mg,
    int restart, int max_iter, double tol,
    int n_ritz_harvest = 4);

FCGResult fcg_solve_mg(
    const OpApply& A, int n, const Vec& rhs,
    MGHierarchy& mg,
    int max_iter, double tol, int truncation = 20);

// CG solve with Lanczos Ritz extraction.
// Exploits the CG–Lanczos equivalence: CG implicitly builds a Lanczos
// tridiagonal T from the α,β coefficients.  After convergence we diagonalise
// T and reconstruct Ritz vectors from the stored (normalised) residuals.
// Cost: identical to plain CG — the Ritz extraction is a post-processing
// step on already-computed data (O(m²) eigensolver + O(m·n) reconstruction,
// where m = number of CG iterations).
CGResult cg_solve_ritz(
    const OpApply& A, int n, const Vec& rhs,
    int max_iter, double tol,
    int n_ritz, std::vector<RitzPair>& ritz_out,
    int max_lanczos_vecs = 0);
