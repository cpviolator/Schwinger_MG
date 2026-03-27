#include "coarse_op.h"
#include "eigensolver.h"
#include <iostream>

// Setup deflation space for SparseCoarseOp via TRLM.
// Finds the n_ev smallest eigenpairs of the sparse coarse operator,
// which are then used by solve() for deflated CG.
void SparseCoarseOp::setup_deflation(int n_ev, int n_kr,
                                      int max_restarts, double tol) {
    if (dim <= 0 || n_ev <= 0) return;
    if (n_kr <= 0) n_kr = std::min(2 * n_ev + 10, dim);
    if (n_kr > dim) n_kr = dim;
    if (n_kr < n_ev + 2) n_kr = std::min(n_ev + 6, dim);

    OpApply op = as_op();
    auto result = trlm_eigensolver(op, dim, n_ev, n_kr, max_restarts, tol);

    if (!result.converged) {
        std::cerr << "SparseCoarseOp::setup_deflation: TRLM did not converge"
                  << " (restarts=" << result.num_restarts << ")\n";
    }

    defl_vecs = std::move(result.eigvecs);
    defl_vals = std::move(result.eigvals);
}
