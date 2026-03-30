#pragma once
#include "types.h"
#include "linalg.h"
#include "prolongator.h"
#include "dirac.h"
#include <algorithm>
#include <numeric>
#include <array>
#include <string>
#include <iostream>

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

// =====================================================================
//  SPARSE COARSE-GRID OPERATOR  (stencil-based, for large coarse grids)
// =====================================================================
// Stores only the 5-point stencil (self + 4 neighbours) on the coarse
// lattice, with k_vec × k_vec dense blocks at each stencil point.
//
// Storage: O(5 × nblocks × k_vec²)  vs  O(dim²) for dense.
// Matvec:  O(5 × dim × k_vec)       vs  O(dim²) for dense.
//
// For k_vec=4, dim=1024: sparse matvec = 20K ops vs dense = 1M ops (50×).
// For 4D lattice QCD with dim=O(10⁴-10⁵), sparse is essential.
//
// Coarse solve uses deflated CG: TRLM finds low modes of the sparse op,
// then CG deflates them out for fast convergence.
struct SparseCoarseOp {
    int nbx, nby, nblocks;   // coarse lattice dimensions
    int k_vec;                // internal DOFs per coarse site
    int dim;                  // total coarse dim = nblocks * k_vec

    // Stencil storage: link[block][dir] = k_vec × k_vec dense matrix
    // stored row-major: entry (i,j) at index [i * k_vec + j].
    // Directions: 0=self, 1=+x, 2=-x, 3=+y, 4=-y,
    //             5=+x+y, 6=+x-y, 7=-x+y, 8=-x-y
    // D†D couples sites up to 2 hops apart, so diagonal coarse blocks
    // also couple — need 9-point stencil for correctness.
    static constexpr int N_DIR = 9;
    std::vector<std::array<Vec, N_DIR>> link;

    // Deflation space (computed by setup_deflation via TRLM)
    std::vector<Vec> defl_vecs;
    std::vector<double> defl_vals;

    // Validate deflation vectors: compute residuals ||Av - λv||/||Av||
    // and discard any with residual above threshold.  Also discards
    // vectors with non-positive eigenvalues (operator is HPD).
    // Returns number of vectors kept.
    int validate_deflation(double res_threshold = 0.1) {
        int kept = 0;
        int ndefl = (int)defl_vecs.size();
        for (int i = 0; i < ndefl; i++) {
            // Recompute Rayleigh quotient
            Vec Av(dim);
            apply_to(defl_vecs[i], Av);
            double rq = std::real(dot(defl_vecs[i], Av));
            Vec resid = Av;
            axpy(cx(-rq), defl_vecs[i], resid);
            double av_norm = norm(Av);
            double rel_res = (av_norm > 1e-30) ? norm(resid) / av_norm : 1.0;

            if (rq > 1e-14 && rel_res < res_threshold) {
                if (kept < i) {
                    defl_vecs[kept] = std::move(defl_vecs[i]);
                }
                defl_vals[kept] = rq;  // use fresh Rayleigh quotient
                kept++;
            }
        }
        defl_vecs.resize(kept);
        defl_vals.resize(kept);
        return kept;
    }

    SparseCoarseOp() : nbx(0), nby(0), nblocks(0), k_vec(0), dim(0) {}

    // Neighbour lookup with periodic BCs (9-point stencil)
    int neighbour(int b, int dir) const {
        int bxi = b % nbx, byi = b / nbx;
        int dx = 0, dy = 0;
        switch (dir) {
            case 0: break;                        // self
            case 1: dx =  1; break;               // +x
            case 2: dx = -1; break;               // -x
            case 3: dy =  1; break;               // +y
            case 4: dy = -1; break;               // -y
            case 5: dx =  1; dy =  1; break;      // +x+y
            case 6: dx =  1; dy = -1; break;      // +x-y
            case 7: dx = -1; dy =  1; break;      // -x+y
            case 8: dx = -1; dy = -1; break;      // -x-y
        }
        int nx = (bxi + dx + nbx) % nbx;
        int ny = (byi + dy + nby) % nby;
        return nx + ny * nbx;
    }

    // Reverse direction: if dir goes I→J, reverse_dir goes J→I
    static int reverse_dir(int dir) {
        // 0↔0, 1↔2, 3↔4, 5↔8, 6↔7
        static constexpr int rev[] = {0, 2, 1, 4, 3, 8, 7, 6, 5};
        return rev[dir];
    }

    // Find direction from block I to block J (returns -1 if not neighbours)
    int find_dir(int I, int J) const {
        for (int d = 0; d < N_DIR; d++)
            if (neighbour(I, d) == J) return d;
        return -1;
    }

    // Build from a geometric Prolongator via Galerkin projection A_c = P† A P.
    // Extracts only the stencil entries (block I couples to block J only if
    // they are the same or nearest-neighbour on the coarse lattice).
    void build(const Prolongator& P, const OpApply& A, int fine_dim) {
        nbx = P.nbx;
        nby = P.nby;
        nblocks = P.nblocks;
        k_vec = P.k_vec;
        dim = nblocks * k_vec;

        // Allocate stencil
        link.resize(nblocks);
        for (int b = 0; b < nblocks; b++)
            for (int d = 0; d < N_DIR; d++)
                link[b][d].assign(k_vec * k_vec, 0.0);

        // For each coarse basis vector e_j, compute column j of A_c = P† A P
        // and extract stencil entries.
        for (int J = 0; J < nblocks; J++) {
            for (int kj = 0; kj < k_vec; kj++) {
                int j = J * k_vec + kj;
                Vec ej(dim, 0.0);
                ej[j] = 1.0;

                // Prolong, apply fine op, restrict
                Vec fine_vec = P.prolong(ej);
                Vec Afine(fine_dim);
                A(fine_vec, Afine);
                Vec coarse_col = P.restrict_vec(Afine);

                // Extract stencil entries: for each block I that's J or a neighbour
                for (int d = 0; d < N_DIR; d++) {
                    int I = neighbour(J, d);
                    // Direction from I to J (the reverse)
                    int rd = reverse_dir(d);
                    for (int ki = 0; ki < k_vec; ki++) {
                        link[I][rd][ki * k_vec + kj] = coarse_col[I * k_vec + ki];
                    }
                }
            }
        }
    }

    // Sparse matvec: dst = A_c * src
    void apply_to(const Vec& src, Vec& dst) const {
        dst.assign(dim, 0.0);
        #pragma omp parallel for schedule(static) if(nblocks > OMP_MIN_SIZE/16)
        for (int b = 0; b < nblocks; b++) {
            for (int d = 0; d < N_DIR; d++) {
                int nb = neighbour(b, d);
                const auto& L = link[b][d];
                for (int ki = 0; ki < k_vec; ki++) {
                    cx sum = 0.0;
                    for (int kj = 0; kj < k_vec; kj++)
                        sum += L[ki * k_vec + kj] * src[nb * k_vec + kj];
                    dst[b * k_vec + ki] += sum;
                }
            }
        }
    }

    // OpApply functor
    OpApply as_op() const {
        return [this](const Vec& s, Vec& d) { apply_to(s, d); };
    }

    // Perturbation-based deflation update: use known δD from HMC to
    // compute δA_c v_i via fine-grid operations, then do RR projection
    // in the extended subspace {v_1,...,v_k, δA_c v_1,...,δA_c v_k}.
    //
    // Cost: n_defl × ~2.5 fine-grid matvecs (vs dim fine matvecs for
    // full Galerkin rebuild + TRLM).
    //
    // This updates defl_vecs and defl_vals IN PLACE without rebuilding
    // the sparse coarse operator stencil.
    //
    // Arguments:
    //   P: prolongator (for coarse ↔ fine mapping)
    //   apply_deltaD: v ↦ δD v  (known from gauge update)
    //   apply_D: v ↦ D_old v (current Dirac op before gauge update)
    //   apply_D_dag: v ↦ D_old† v
    //   fine_dim: fine-grid vector dimension
    //
    // After calling this, the stencil is still the OLD A_c, but
    // defl_vecs/defl_vals are updated to approximate the new eigenpairs.
    void perturbation_update(
        const Prolongator& P,
        const std::function<void(const Vec&, Vec&)>& apply_deltaD,
        const std::function<void(const Vec&, Vec&)>& apply_deltaD_dag,
        const std::function<void(const Vec&, Vec&)>& apply_D,
        const std::function<void(const Vec&, Vec&)>& apply_D_dag,
        int fine_dim)
    {
        int k = (int)defl_vecs.size();
        if (k == 0) return;

        // For each deflation vector v_i:
        //   Prolong w_i = P v_i
        //   Compute δ(D†D) w_i = δD†(D w_i) + D†(δD w_i) + δD†(δD w_i)
        //   Restrict c_i = P† [δ(D†D) w_i]
        std::vector<Vec> dAv(k);  // δA_c v_i in coarse space

        for (int i = 0; i < k; i++) {
            Vec wi = P.prolong(defl_vecs[i]);   // fine-grid

            // D w_i
            Vec Dwi(fine_dim);
            apply_D(wi, Dwi);

            // δD w_i
            Vec dDwi(fine_dim);
            apply_deltaD(wi, dDwi);

            // Term 1: δD†(D w_i)
            Vec term1(fine_dim);
            apply_deltaD_dag(Dwi, term1);

            // Term 2: D†(δD w_i)
            Vec term2(fine_dim);
            apply_D_dag(dDwi, term2);

            // Term 3: δD†(δD w_i)  (second order, small but free)
            Vec term3(fine_dim);
            apply_deltaD_dag(dDwi, term3);

            // δ(D†D) w_i = term1 + term2 + term3
            Vec delta_A_wi(fine_dim);
            for (int j = 0; j < fine_dim; j++)
                delta_A_wi[j] = term1[j] + term2[j] + term3[j];

            // Restrict to coarse
            dAv[i] = P.restrict_vec(delta_A_wi);
        }

        // Build the perturbation matrix in the deflation subspace:
        // M_ij = v_i† (A_c + δA_c) v_j = λ_i δ_ij + v_i† δA_c v_j
        // where A_c v_i = λ_i v_i
        std::vector<Vec> M_cols(k, Vec(k, 0.0));
        for (int i = 0; i < k; i++) {
            M_cols[i][i] = cx(defl_vals[i]);
            for (int j = 0; j <= i; j++) {
                cx dM = dot(defl_vecs[j], dAv[i]);
                M_cols[i][j] += dM;
                if (i != j) M_cols[j][i] += std::conj(dM);
            }
        }

        // Diagonalise M → rotated eigenvectors within the subspace
        RVec evals;
        std::vector<Vec> evecs;
        lanczos_eigen(M_cols, k, evals, evecs);

        // Rotate deflation vectors
        std::vector<Vec> new_defl(k);
        for (int i = 0; i < k; i++) {
            new_defl[i] = zeros(dim);
            for (int j = 0; j < k; j++)
                axpy(evecs[j][i], defl_vecs[j], new_defl[i]);
        }
        defl_vecs = std::move(new_defl);
        for (int i = 0; i < k; i++)
            defl_vals[i] = evals[i];

        // Also extract out-of-subspace residual directions and add to pool
        // r_i = δA_c v_i - Σ_j (v_j† δA_c v_i) v_j
        // These point where eigenvectors are leaking
        std::vector<Vec> residuals;
        for (int i = 0; i < k; i++) {
            Vec ri = dAv[i];
            // Project out the deflation subspace
            for (int j = 0; j < k; j++) {
                cx proj = dot(defl_vecs[j], ri);
                axpy(-proj, defl_vecs[j], ri);
            }
            double rnorm = norm(ri);
            if (rnorm > 1e-10) {
                scale(ri, cx(1.0 / rnorm));
                // Orthogonalise against existing residuals
                for (auto& prev : residuals) {
                    cx proj = dot(prev, ri);
                    axpy(-proj, prev, ri);
                }
                rnorm = norm(ri);
                if (rnorm > 0.1) {
                    scale(ri, cx(1.0 / rnorm));
                    residuals.push_back(std::move(ri));
                }
            }
        }

        // Add residual directions to the deflation space and re-diag
        if (!residuals.empty()) {
            int k_ext = k + (int)residuals.size();
            // Extended basis: {defl_vecs} ∪ {residuals}
            std::vector<Vec> ext_basis(k_ext);
            for (int i = 0; i < k; i++) ext_basis[i] = defl_vecs[i];
            for (int i = 0; i < (int)residuals.size(); i++)
                ext_basis[k + i] = residuals[i];

            // Build projected matrix in extended basis using stencil
            // Note: we use the OLD stencil A_c here. The perturbation
            // correction is already folded into the eigenvectors.
            // For the extended vectors (residuals), we use A_c directly.
            std::vector<Vec> A_ext(k_ext);
            for (int i = 0; i < k_ext; i++) {
                A_ext[i].resize(dim);
                apply_to(ext_basis[i], A_ext[i]);
                // Add perturbation for original defl vectors
                if (i < k) {
                    for (int j = 0; j < dim; j++)
                        A_ext[i][j] += dAv[i][j];
                }
            }

            // Build (k_ext × k_ext) Hermitian matrix
            std::vector<Vec> H_cols(k_ext, Vec(k_ext, 0.0));
            for (int i = 0; i < k_ext; i++)
                for (int j = 0; j <= i; j++) {
                    cx val = dot(ext_basis[j], A_ext[i]);
                    H_cols[i][j] = val;
                    if (i != j) H_cols[j][i] = std::conj(val);
                }

            // Diagonalise and keep k smallest
            RVec ext_evals;
            std::vector<Vec> ext_evecs;
            lanczos_eigen(H_cols, k_ext, ext_evals, ext_evecs);

            std::vector<Vec> final_defl(k);
            for (int i = 0; i < k; i++) {
                final_defl[i] = zeros(dim);
                for (int j = 0; j < k_ext; j++)
                    axpy(ext_evecs[j][i], ext_basis[j], final_defl[i]);
                defl_vals[i] = ext_evals[i];
            }
            defl_vecs = std::move(final_defl);
        }
    }

    // Setup deflation space via TRLM on the sparse operator.
    // Finds n_ev smallest eigenpairs using Krylov space of size n_kr.
    // This is called once after build(), then deflated CG uses the result.
    void setup_deflation(int n_ev, int n_kr = 0, int max_restarts = 100,
                         double tol = 1e-10,
                         const std::string& solver = "trlm",
                         double feast_emax = 0.0);

    // Solve A_c x = b via CG with deflated initial guess.
    // Deflation vectors are used ONLY for the initial guess:
    //   x0 = Σ (v_i†b / λ_i) v_i
    // This is safe with approximate eigenvectors — a bad guess just means
    // more iterations, never divergence (unlike projected deflation which
    // breaks CG if vectors are inaccurate).
    // Cost per solve: O(n_iter × 5 × dim × k_vec) sparse matvecs.
    Vec solve(const Vec& b, int max_iter = 200, double cg_tol = 1e-12) const {
        int ndefl = (int)defl_vecs.size();
        double bnorm = norm(b);
        if (bnorm < 1e-30) return zeros(dim);

        // Initial guess from deflation: x0 = Σ (v_i†b / λ_i) v_i
        Vec x = zeros(dim);
        if (ndefl > 0) {
            for (int i = 0; i < ndefl; i++) {
                if (defl_vals[i] > 1e-14) {
                    cx coeff = dot(defl_vecs[i], b) / defl_vals[i];
                    axpy(coeff, defl_vecs[i], x);
                }
            }
        }

        // Plain CG from initial guess
        Vec Ax(dim);
        apply_to(x, Ax);
        Vec r(dim);
        for (int i = 0; i < dim; i++) r[i] = b[i] - Ax[i];

        Vec p = r;
        cx rr_val = dot(r, r);
        int iter = 0;

        while (iter < max_iter) {
            Vec Ap(dim);
            apply_to(p, Ap);
            cx pAp = dot(p, Ap);
            if (std::abs(pAp) < 1e-30) break;
            cx alpha = rr_val / pAp;

            for (int i = 0; i < dim; i++) {
                x[i] += alpha * p[i];
                r[i] -= alpha * Ap[i];
            }
            iter++;

            double rnorm = norm(r);
            if (rnorm / bnorm < cg_tol) break;
            if (std::isnan(rnorm) || std::isinf(rnorm)) {
                // CG diverged — return best guess
                break;
            }

            cx rr_new = dot(r, r);
            cx beta = rr_new / rr_val;
            rr_val = rr_new;

            for (int i = 0; i < dim; i++)
                p[i] = r[i] + beta * p[i];
        }

        return x;
    }
};
