#pragma once
#include "types.h"
#include "lattice.h"
#include <cassert>
#include <cmath>

// =====================================================================
//  BLOCK-AGGREGATION PROLONGATOR
// =====================================================================
// Divide the L×L lattice into blocks of size bx×by.
// Each block has k_vec near-null vectors (restricted & orthonormalised).
// Prolongator P: coarse_vec → fine_vec.
// Coarse DOF index = block_index * k_vec + vec_index.
struct Prolongator {
    const Lattice& lat;
    int bx, by;                          // block size
    int nbx, nby, nblocks;              // number of blocks
    int k_vec;                           // near-null vectors per block
    int coarse_dim;                      // nblocks * k_vec

    // block_vecs[b][k] = fine-grid vector restricted to block b, for k-th null vec
    // stored as length (2 * bx * by) vectors
    std::vector<std::vector<Vec>> block_vecs;

    // map: fine site → (block_index, local_index)
    std::vector<int> site_to_block;
    std::vector<int> site_to_local;

    // reverse map: block_index → list of fine sites (for parallel restrict)
    std::vector<std::vector<int>> block_sites;

    // quality metric per block vector: Rayleigh quotient ||A v||/||v||
    std::vector<std::vector<double>> quality;

    Prolongator(const Lattice& lat_, int bx_, int by_, int k_)
        : lat(lat_), bx(bx_), by(by_), k_vec(k_)
    {
        assert(lat.L % bx == 0 && lat.L % by == 0);
        nbx = lat.L / bx;
        nby = lat.L / by;
        nblocks = nbx * nby;
        coarse_dim = nblocks * k_vec;

        block_vecs.resize(nblocks, std::vector<Vec>(k_vec));
        quality.resize(nblocks, std::vector<double>(k_vec, 1e10));

        site_to_block.resize(lat.V);
        site_to_local.resize(lat.V);
        block_sites.resize(nblocks);
        for (int s = 0; s < lat.V; s++) {
            int x = s % lat.L, y = s / lat.L;
            int bi = x / bx, bj = y / by;
            int li = x % bx, lj = y % by;
            site_to_block[s] = bi + nbx * bj;
            site_to_local[s] = li + bx * lj;
            block_sites[site_to_block[s]].push_back(s);
        }
    }

    int block_fine_dim() const { return 2 * bx * by; }

    // Restrict a fine-grid vector to a block: extract block components
    Vec restrict_to_block(const Vec& fine, int b) const {
        int bdim = block_fine_dim();
        Vec out(bdim, 0.0);
        for (int s = 0; s < lat.V; s++) {
            if (site_to_block[s] == b) {
                int loc = site_to_local[s];
                out[2*loc]   = fine[2*s];
                out[2*loc+1] = fine[2*s+1];
            }
        }
        return out;
    }

    // Prolong: coarse → fine
    Vec prolong(const Vec& coarse) const {
        Vec fine(lat.ndof, 0.0);
        #pragma omp parallel for schedule(static) if(lat.V > OMP_MIN_SIZE)
        for (int s = 0; s < lat.V; s++) {
            int b = site_to_block[s];
            int loc = site_to_local[s];
            for (int k = 0; k < k_vec; k++) {
                cx c = coarse[b * k_vec + k];
                fine[2*s]   += c * block_vecs[b][k][2*loc];
                fine[2*s+1] += c * block_vecs[b][k][2*loc+1];
            }
        }
        return fine;
    }

    // Restrict: fine → coarse  (P†)
    // Parallelise over blocks: each block accumulates independently (no races)
    Vec restrict_vec(const Vec& fine) const {
        Vec coarse(coarse_dim, 0.0);
        #pragma omp parallel for schedule(static) if(nblocks > OMP_MIN_SIZE/16)
        for (int b = 0; b < nblocks; b++) {
            for (int s : block_sites[b]) {
                int loc = site_to_local[s];
                for (int k = 0; k < k_vec; k++) {
                    coarse[b * k_vec + k] +=
                        std::conj(block_vecs[b][k][2*loc])   * fine[2*s] +
                        std::conj(block_vecs[b][k][2*loc+1]) * fine[2*s+1];
                }
            }
        }
        return coarse;
    }

    // Block-orthonormalise the vectors for block b (Gram-Schmidt)
    void orthonormalise_block(int b) {
        int bdim = block_fine_dim();
        for (int k = 0; k < k_vec; k++) {
            auto& v = block_vecs[b][k];
            // subtract projections on earlier vectors
            for (int j = 0; j < k; j++) {
                cx proj = 0;
                for (int i = 0; i < bdim; i++)
                    proj += std::conj(block_vecs[b][j][i]) * v[i];
                for (int i = 0; i < bdim; i++)
                    v[i] -= proj * block_vecs[b][j][i];
            }
            // normalise
            double n = 0;
            for (int i = 0; i < bdim; i++) n += std::norm(v[i]);
            n = std::sqrt(n);
            if (n > 1e-14)
                for (int i = 0; i < bdim; i++) v[i] /= n;
        }
    }

    // Build from a set of fine-grid near-null vectors
    void build_from_vectors(const std::vector<Vec>& null_vecs) {
        assert((int)null_vecs.size() >= k_vec);
        for (int b = 0; b < nblocks; b++) {
            for (int k = 0; k < k_vec; k++) {
                block_vecs[b][k] = restrict_to_block(null_vecs[k], b);
            }
            orthonormalise_block(b);
        }
    }
};

// =====================================================================
//  ALGEBRAIC PROLONGATOR (for coarse-to-coarser transfers)
// =====================================================================
// Unlike the geometric Prolongator above, this uses linear blocking:
// indices [b*block_k, (b+1)*block_k) form block b.  No lattice needed.
struct CoarseProlongator {
    int fine_dim;
    int block_k;      // DOFs per block
    int nblocks;
    int k_vec;         // near-null vectors per block
    int coarse_dim;

    std::vector<std::vector<Vec>> block_vecs;  // block_vecs[b][k], length block_k

    CoarseProlongator() : fine_dim(0), block_k(0), nblocks(0), k_vec(0), coarse_dim(0) {}

    CoarseProlongator(int fine_dim_, int block_k_, int k_vec_)
        : fine_dim(fine_dim_), block_k(block_k_), k_vec(k_vec_)
    {
        assert(fine_dim % block_k == 0);
        nblocks = fine_dim / block_k;
        coarse_dim = nblocks * k_vec;
        block_vecs.resize(nblocks, std::vector<Vec>(k_vec));
    }

    void orthonormalise_block(int b) {
        for (int k = 0; k < k_vec; k++) {
            auto& v = block_vecs[b][k];
            for (int j = 0; j < k; j++) {
                cx proj = 0;
                for (int i = 0; i < block_k; i++)
                    proj += std::conj(block_vecs[b][j][i]) * v[i];
                for (int i = 0; i < block_k; i++)
                    v[i] -= proj * block_vecs[b][j][i];
            }
            double n = 0;
            for (int i = 0; i < block_k; i++) n += std::norm(v[i]);
            n = std::sqrt(n);
            if (n > 1e-14)
                for (int i = 0; i < block_k; i++) v[i] /= n;
        }
    }

    Vec prolong(const Vec& coarse) const {
        Vec fine(fine_dim, 0.0);
        #pragma omp parallel for schedule(static) if(nblocks > OMP_MIN_SIZE/16)
        for (int b = 0; b < nblocks; b++) {
            int base = b * block_k;
            for (int k = 0; k < k_vec; k++) {
                cx c = coarse[b * k_vec + k];
                for (int i = 0; i < block_k; i++)
                    fine[base + i] += c * block_vecs[b][k][i];
            }
        }
        return fine;
    }

    Vec restrict_vec(const Vec& fine) const {
        Vec coarse(coarse_dim, 0.0);
        #pragma omp parallel for schedule(static) if(nblocks > OMP_MIN_SIZE/16)
        for (int b = 0; b < nblocks; b++) {
            int base = b * block_k;
            for (int k = 0; k < k_vec; k++) {
                cx s = 0;
                for (int i = 0; i < block_k; i++)
                    s += std::conj(block_vecs[b][k][i]) * fine[base + i];
                coarse[b * k_vec + k] = s;
            }
        }
        return coarse;
    }

    void build_from_vectors(const std::vector<Vec>& null_vecs) {
        assert((int)null_vecs.size() >= k_vec);
        for (int b = 0; b < nblocks; b++) {
            int base = b * block_k;
            for (int k = 0; k < k_vec; k++) {
                block_vecs[b][k].resize(block_k);
                for (int i = 0; i < block_k; i++)
                    block_vecs[b][k][i] = null_vecs[k][base + i];
            }
            orthonormalise_block(b);
        }
    }
};
