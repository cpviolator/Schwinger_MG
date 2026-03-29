#pragma once
#include "types.h"
#include "eigensolver.h"

// FEAST eigensolver: finds eigenvalues in [Emin, Emax] of Hermitian operator A.
// Uses the Reverse Communication Interface (RCI) — calls OpApply for A*x
// and solves shifted systems (z*I - A)x = b via BiCGStab internally.
// Returns TRLMResult for drop-in compatibility with TRLM.
TRLMResult feast_eigensolver(
    const OpApply& A, int n,
    double Emin, double Emax, int M0,
    int n_contour = 8, double tol = 1e-10, int max_iter = 20,
    const std::vector<Vec>* warm_start = nullptr);

// Solve (z*I - A)*x = b where z is complex, A is Hermitian.
// Uses BiCGStab since the shifted operator is non-Hermitian.
Vec shifted_solve(const OpApply& A, cx z, const Vec& b, int n,
                  int max_iter = 500, double tol = 1e-12);
