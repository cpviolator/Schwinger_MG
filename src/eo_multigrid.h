#pragma once
#include "types.h"
#include "linalg.h"
#include "lattice.h"
#include "dirac.h"
#include "smoother.h"
#include <vector>
#include <functional>
#include <random>
#include <iostream>

// Forward declaration (defined in multigrid.cpp)
std::vector<Vec> compute_near_null_space_generic(
    const OpApply& A, int dim, int k, int outer_iters, std::mt19937& rng,
    const std::vector<Vec>* warm_start);

// ---------------------------------------------------------------
//  Even-Odd Aware Multigrid
//  Builds MG hierarchy directly on the Schur complement M†M
//  - Near-null vectors live in odd-site (half-lattice) space
//  - Prolongator maps half-lattice ↔ coarse
//  - Smoother uses M†M (not D†D)
//  - Coarse operator = P_eo† M†M P_eo (Galerkin projection)
// ---------------------------------------------------------------

struct EvenOddProlongator {
    const Lattice& lat;
    int bx, by;
    int nbx, nby, nblocks;
    int k_vec;
    int coarse_dim;

    // block_vecs[b][k] = half-lattice vector restricted to odd sites in block b
    // stored as length (2 * n_odd_in_block) vectors
    std::vector<std::vector<Vec>> block_vecs;

    // map: odd-site half-index → block
    std::vector<int> odd_to_block;
    // map: odd-site half-index → local position within that block's odd sites
    std::vector<int> odd_to_local;
    // reverse map: block → list of half-indices of odd sites in that block
    std::vector<std::vector<int>> block_odd_indices;
    // number of odd sites per block
    std::vector<int> block_n_odd;

    EvenOddProlongator(const Lattice& lat_, int bx_, int by_, int k_)
        : lat(lat_), bx(bx_), by(by_), k_vec(k_)
    {
        nbx = lat.L / bx;
        nby = lat.L / by;
        nblocks = nbx * nby;
        coarse_dim = nblocks * k_vec;

        // Map each odd site (by half-index) to a block and local position
        odd_to_block.resize(lat.V_half);
        odd_to_local.resize(lat.V_half);
        block_odd_indices.resize(nblocks);
        block_n_odd.resize(nblocks, 0);

        for (int io = 0; io < lat.V_half; io++) {
            int s = lat.odd_sites[io];
            int x = s % lat.L, y = s / lat.L;
            int bi = x / bx, bj = y / by;
            int b = bi + nbx * bj;
            odd_to_block[io] = b;
            odd_to_local[io] = block_n_odd[b];
            block_odd_indices[b].push_back(io);
            block_n_odd[b]++;
        }

        block_vecs.resize(nblocks);
        for (int b = 0; b < nblocks; b++)
            block_vecs[b].resize(k_vec);
    }

    // Prolong: coarse → half-lattice (odd sites)
    Vec prolong(const Vec& coarse) const {
        int n_half = 2 * lat.V_half;
        Vec fine(n_half, 0.0);
        for (int b = 0; b < nblocks; b++) {
            for (int k = 0; k < k_vec; k++) {
                cx c = coarse[b * k_vec + k];
                const Vec& bv = block_vecs[b][k];
                for (int j = 0; j < (int)block_odd_indices[b].size(); j++) {
                    int io = block_odd_indices[b][j];
                    fine[2*io]   += c * bv[2*j];
                    fine[2*io+1] += c * bv[2*j+1];
                }
            }
        }
        return fine;
    }

    // Restrict: half-lattice (odd sites) → coarse
    Vec restrict_vec(const Vec& fine) const {
        Vec coarse(coarse_dim, 0.0);
        for (int b = 0; b < nblocks; b++) {
            for (int k = 0; k < k_vec; k++) {
                cx sum = 0;
                const Vec& bv = block_vecs[b][k];
                for (int j = 0; j < (int)block_odd_indices[b].size(); j++) {
                    int io = block_odd_indices[b][j];
                    sum += std::conj(bv[2*j])   * fine[2*io]
                         + std::conj(bv[2*j+1]) * fine[2*io+1];
                }
                coarse[b * k_vec + k] = sum;
            }
        }
        return coarse;
    }

    // Orthonormalise block vectors within each block
    void orthonormalise() {
        for (int b = 0; b < nblocks; b++) {
            int n_odd = block_n_odd[b];
            int bdim = 2 * n_odd;
            for (int k = 0; k < k_vec; k++) {
                Vec& v = block_vecs[b][k];
                if ((int)v.size() != bdim) v.resize(bdim, 0.0);
                // Gram-Schmidt against previous vectors in this block
                for (int j = 0; j < k; j++) {
                    cx proj = 0;
                    for (int i = 0; i < bdim; i++)
                        proj += std::conj(block_vecs[b][j][i]) * v[i];
                    for (int i = 0; i < bdim; i++)
                        v[i] -= proj * block_vecs[b][j][i];
                }
                // Normalise
                double nrm = 0;
                for (int i = 0; i < bdim; i++)
                    nrm += std::norm(v[i]);
                nrm = std::sqrt(nrm);
                if (nrm > 1e-14)
                    for (int i = 0; i < bdim; i++)
                        v[i] /= nrm;
            }
        }
    }

    // Build from half-lattice null vectors
    void build_from_vectors(const std::vector<Vec>& null_vecs) {
        for (int k = 0; k < k_vec; k++) {
            const Vec& nv = null_vecs[k];
            for (int b = 0; b < nblocks; b++) {
                int n_odd = block_n_odd[b];
                block_vecs[b][k].resize(2 * n_odd);
                for (int j = 0; j < n_odd; j++) {
                    int io = block_odd_indices[b][j];
                    block_vecs[b][k][2*j]   = nv[2*io];
                    block_vecs[b][k][2*j+1] = nv[2*io+1];
                }
            }
        }
        orthonormalise();
    }
};

// Dense coarse operator for e/o MG
struct EvenOddCoarseOp {
    int dim;
    std::vector<Vec> mat;  // column-major dense matrix

    void build(const OpApply& A_schur_dag_schur,
               EvenOddProlongator& P, int fine_half_dim) {
        dim = P.coarse_dim;
        mat.resize(dim);
        for (int j = 0; j < dim; j++) {
            Vec ej = zeros(dim);
            ej[j] = 1.0;
            Vec fj = P.prolong(ej);
            Vec Afj(fine_half_dim);
            A_schur_dag_schur(fj, Afj);
            mat[j] = P.restrict_vec(Afj);
        }
    }

    Vec apply(const Vec& x) const {
        Vec y = zeros(dim);
        for (int j = 0; j < dim; j++) {
            cx xj = x[j];
            for (int i = 0; i < dim; i++)
                y[i] += mat[j][i] * xj;
        }
        return y;
    }

    Vec solve(const Vec& b) const {
        // Dense LU solve
        int n = dim;
        std::vector<Vec> L(n, Vec(n, 0.0));
        std::vector<Vec> U(n);
        for (int j = 0; j < n; j++) U[j] = mat[j];

        for (int k = 0; k < n; k++) {
            L[k][k] = 1.0;
            for (int i = k+1; i < n; i++) {
                cx factor = U[k][i] / U[k][k];
                L[k][i] = factor;
                for (int j = k; j < n; j++)
                    U[j][i] -= factor * U[j][k];
            }
        }
        // Forward solve L y = b
        Vec y = b;
        for (int k = 0; k < n; k++)
            for (int i = k+1; i < n; i++)
                y[i] -= L[k][i] * y[k];
        // Backward solve U x = y
        Vec x(n);
        for (int k = n-1; k >= 0; k--) {
            x[k] = y[k];
            for (int j = k+1; j < n; j++)
                x[k] -= U[j][k] * x[j];
            x[k] /= U[k][k];
        }
        return x;
    }
};

// The complete e/o-aware MG preconditioner
struct EvenOddMG {
    EvenOddProlongator P_eo;
    EvenOddCoarseOp Ac;
    int fine_half_dim;
    int pre_smooth, post_smooth;
    double richardson_omega = 0.0;

    EvenOddMG(const Lattice& lat, int bx, int by, int k_vec,
              int pre = 3, int post = 3)
        : P_eo(lat, bx, by, k_vec), fine_half_dim(2 * lat.V_half),
          pre_smooth(pre), post_smooth(post) {}

    // Build the MG hierarchy from a DiracOp
    void build(const DiracOp& D, int null_iters, std::mt19937& rng) {
        EvenOddDiracOp eoD(D);
        OpApply A = [&eoD](const Vec& s, Vec& d) {
            eoD.apply_schur_dag_schur(s, d);
        };

        // Compute near-null vectors of M†M in half-lattice space
        auto null_vecs = compute_near_null_space_generic(
            A, fine_half_dim, P_eo.k_vec, null_iters, rng);

        P_eo.build_from_vectors(null_vecs);

        // Build Galerkin coarse operator: P† M†M P
        Ac.build(A, P_eo, fine_half_dim);

        // Estimate lambda_max for Richardson smoothing
        double lmax = estimate_lambda_max(A, fine_half_dim, 30);
        richardson_omega = 0.8 / lmax;

        std::cout << "  E/O MG: fine_half_dim=" << fine_half_dim
                  << " coarse_dim=" << Ac.dim
                  << " lambda_max=" << lmax << "\n";
    }

    // V-cycle preconditioner: operates on half-lattice (odd-site) vectors
    Vec precondition(const Vec& b, const OpApply& A_MdagM) const {
        int n = fine_half_dim;

        // Pre-smooth
        Vec x = zeros(n);
        richardson_smooth_op(A_MdagM, x, b, pre_smooth, richardson_omega);

        // Compute residual
        Vec Ax(n);
        A_MdagM(x, Ax);
        Vec r(n);
        for (int i = 0; i < n; i++) r[i] = b[i] - Ax[i];

        // Restrict to coarse
        Vec rc = P_eo.restrict_vec(r);

        // Coarse solve (dense LU)
        Vec ec = Ac.solve(rc);

        // Prolong and correct
        Vec e = P_eo.prolong(ec);
        for (int i = 0; i < n; i++) x[i] += e[i];

        // Post-smooth
        richardson_smooth_op(A_MdagM, x, b, post_smooth, richardson_omega);

        return x;
    }
};
