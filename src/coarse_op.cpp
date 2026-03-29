#include "coarse_op.h"
#include "eigensolver.h"
#include "feast_solver.h"
#include <iostream>

// Setup deflation space for SparseCoarseOp.
// Finds the n_ev smallest eigenpairs of the sparse coarse operator.
// solver: "trlm" (default) or "feast"
// feast_emax: upper bound of FEAST spectral window (0 = auto-estimate)
void SparseCoarseOp::setup_deflation(int n_ev, int n_kr,
                                      int max_restarts, double tol,
                                      const std::string& solver,
                                      double feast_emax) {
    if (dim <= 0 || n_ev <= 0) return;

    OpApply op = as_op();

    if (solver == "feast") {
        // Auto-estimate Emax if not provided: use a few Lanczos steps
        double emax = feast_emax;
        if (emax <= 0.0) {
            // Quick estimate: find the n_ev-th eigenvalue via TRLM, set emax = 2x
            int quick_kr = std::min(2 * n_ev + 10, dim);
            auto quick = trlm_eigensolver(op, dim, n_ev, quick_kr, 50, 1e-4);
            if (!quick.eigvals.empty())
                emax = 2.0 * quick.eigvals.back();
            else
                emax = 1.0;  // fallback
            std::cout << "  FEAST auto Emax=" << emax << "\n";
        }
        int M0 = std::min((int)(1.5 * n_ev) + 4, dim);
        // Warm-start from existing deflation vectors if available
        const std::vector<Vec>* warm = (!defl_vecs.empty()) ? &defl_vecs : nullptr;
        auto result = feast_eigensolver(op, dim, 0.0, emax, M0, 8, tol, 20, warm);

        // Keep only n_ev smallest (FEAST may find more)
        int n_found = std::min((int)result.eigvecs.size(), n_ev);
        defl_vecs.assign(result.eigvecs.begin(), result.eigvecs.begin() + n_found);
        defl_vals.assign(result.eigvals.begin(), result.eigvals.begin() + n_found);
    } else {
        // TRLM (default)
        if (n_kr <= 0) n_kr = std::min(2 * n_ev + 10, dim);
        if (n_kr > dim) n_kr = dim;
        if (n_kr < n_ev + 2) n_kr = std::min(n_ev + 6, dim);

        auto result = trlm_eigensolver(op, dim, n_ev, n_kr, max_restarts, tol);
        if (!result.converged) {
            std::cerr << "SparseCoarseOp::setup_deflation: TRLM did not converge"
                      << " (restarts=" << result.num_restarts << ")\n";
        }
        defl_vecs = std::move(result.eigvecs);
        defl_vals = std::move(result.eigvals);
    }
}
