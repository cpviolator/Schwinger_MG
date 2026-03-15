#pragma once
#include "types.h"
#include "linalg.h"
#include "dirac.h"
#include <random>

void mr_smooth(const DiracOp& D, Vec& x, const Vec& b, int niter);
void mr_smooth_op(const OpApply& A, Vec& x, const Vec& b, int niter);
double estimate_lambda_max(const OpApply& A, int n, int niter = 20);
void richardson_smooth_op(const OpApply& A, Vec& x, const Vec& b,
                          int niter, double omega);
