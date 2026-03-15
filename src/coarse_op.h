#pragma once
#include "types.h"
#include "linalg.h"
#include "prolongator.h"
#include "dirac.h"
#include <algorithm>
#include <numeric>

// Forward declaration — defined in eigensolver.h
void lanczos_eigen(std::vector<Vec>& A_cols, int n, RVec& evals, std::vector<Vec>& evecs);

// =====================================================================
//  COARSE-GRID OPERATOR  (dense, since coarse dim is small)
// =====================================================================
struct CoarseOp {
    int dim;
    std::vector<Vec> mat;   // dense matrix stored as columns

    CoarseOp() : dim(0) {}

    // Build A_c = P† A P using a generic operator
    template<typename ProlongT>
    void build_generic(const OpApply& A, ProlongT& P, int fine_dim, int coarse_dim_) {
        dim = coarse_dim_;
        mat.resize(dim);
        for (int j = 0; j < dim; j++) {
            Vec ej(dim, 0.0);
            ej[j] = 1.0;
            Vec fine = P.prolong(ej);
            Vec Afine(fine_dim);
            A(fine, Afine);
            mat[j] = P.restrict_vec(Afine);
        }
    }

    // Build A_c = P† A P  where A = D†D (backward-compatible)
    void build(const DiracOp& D, Prolongator& P) {
        OpApply A = [&D](const Vec& s, Vec& d){ D.apply_DdagD(s, d); };
        build_generic(A, P, D.lat.ndof, P.coarse_dim);
    }

    // Apply A_c to a coarse vector
    Vec apply(const Vec& x) const {
        Vec y(dim, 0.0);
        for (int j = 0; j < dim; j++) {
            cx xj = x[j];
            for (int i = 0; i < dim; i++)
                y[i] += mat[j][i] * xj;
        }
        return y;
    }

    // In-place apply: dst = A_c * src (for OpApply interface)
    void apply_to(const Vec& src, Vec& dst) const {
        dst = apply(src);
    }

    // Return k eigenvectors with smallest eigenvalues.
    // Uses Jacobi eigensolver on the dense coarse matrix.
    // Cost: O(dim³) — negligible for dim ≤ 256.
    std::vector<Vec> smallest_eigenvectors(int k) const {
        // Build Hermitian matrix: (mat† + mat)/2 for safety
        // Actually mat = P†A P should already be Hermitian for A=D†D
        std::vector<Vec> H_cols(dim, Vec(dim, 0.0));
        for (int j = 0; j < dim; j++)
            for (int i = 0; i < dim; i++)
                H_cols[j][i] = mat[j][i];

        // Hermitise
        for (int i = 0; i < dim; i++)
            for (int j = i+1; j < dim; j++) {
                cx avg = 0.5 * (H_cols[j][i] + std::conj(H_cols[i][j]));
                H_cols[j][i] = avg;
                H_cols[i][j] = std::conj(avg);
            }

        RVec evals;
        std::vector<Vec> evecs;
        lanczos_eigen(H_cols, dim, evals, evecs);

        // Return k eigenvectors (columns of evecs) with smallest eigenvalues
        int nk = std::min(k, dim);
        std::vector<Vec> result(nk);
        for (int p = 0; p < nk; p++) {
            result[p].resize(dim);
            for (int j = 0; j < dim; j++)
                result[p][j] = evecs[j][p];
        }
        return result;
    }

    // Solve A_c x = b via dense LU with partial pivoting.
    // Complexity: O(n^3) where n = coarse_dim = nblocks * k_vec.
    // For the typical case (16x16 lattice, 4x4 blocks, 4 null vecs),
    // n = 64, so O(n^3) ~ 260K ops — negligible per V-cycle.
    // For coarse_dim > ~500, switch to a Krylov solver (e.g. CG,
    // since A_c = P^dag D^dag D P is Hermitian positive semi-definite).
    Vec solve(const Vec& b) const {
        int n = dim;
        // build augmented matrix
        std::vector<Vec> A(n);
        for (int i = 0; i < n; i++) {
            A[i].resize(n + 1);
            for (int j = 0; j < n; j++) A[i][j] = mat[j][i]; // row i, col j
            A[i][n] = b[i];
        }
        // forward elimination
        std::vector<int> piv(n);
        std::iota(piv.begin(), piv.end(), 0);
        for (int k = 0; k < n; k++) {
            // partial pivot
            double best = 0;
            int best_row = k;
            for (int i = k; i < n; i++) {
                double v = std::abs(A[i][k]);
                if (v > best) { best = v; best_row = i; }
            }
            std::swap(A[k], A[best_row]);
            if (best < 1e-30) continue;
            for (int i = k+1; i < n; i++) {
                cx factor = A[i][k] / A[k][k];
                for (int j = k; j <= n; j++)
                    A[i][j] -= factor * A[k][j];
            }
        }
        // back substitution
        Vec x(n, 0.0);
        for (int i = n-1; i >= 0; i--) {
            cx s = A[i][n];
            for (int j = i+1; j < n; j++) s -= A[i][j] * x[j];
            x[i] = (std::abs(A[i][i]) > 1e-30) ? s / A[i][i] : 0.0;
        }
        return x;
    }
};
