#include "eigensolver.h"
#include "linalg.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <iomanip>


// =========================================================================
// Chronological generator forecasting
// =========================================================================

// Extract Hermitian generator H = -i log(U) from unitary rotation U.
// Exact method: diagonalise S = (U - U†)/(2i) to get eigenvectors V and sin(θ),
// then compute θ_j = atan2(sin θ_j, Re(v_j† U v_j)) for exact angles.
// U stored as evecs[row][col] from lanczos_eigen.
void extract_generator(const std::vector<Vec>& U, int k,
                       std::vector<Vec>& H) {
    // Build Hermitian matrix S = (U - U†)/(2i)  [eigenvalues = sin(θ_j)]
    std::vector<Vec> S(k, Vec(k, 0.0));
    for (int r = 0; r < k; r++) {
        for (int c = 0; c < k; c++) {
            // U[row][col]: U_rc = U[r][c], U†_rc = conj(U[c][r])
            cx u_rc = U[r][c];
            cx ud_rc = std::conj(U[c][r]);
            S[c][r] = (u_rc - ud_rc) / cx(0, 2);  // (U - U†)/(2i), stored S[col][row]
        }
    }
    // Enforce exact Hermiticity
    for (int i = 0; i < k; i++) {
        for (int j = i + 1; j < k; j++) {
            cx avg = 0.5 * (S[j][i] + std::conj(S[i][j]));
            S[j][i] = avg; S[i][j] = std::conj(avg);
        }
        S[i][i] = std::real(S[i][i]);
    }

    // Diagonalise S: S = V diag(sin θ) V†
    RVec sin_theta;
    std::vector<Vec> V;
    lanczos_eigen(S, k, sin_theta, V);
    // V[row][col] = row-th component of col-th eigenvector

    // For each eigenvector v_j of S, compute the eigenvalue of U:
    // u_j = v_j† U v_j = cos(θ_j) + i sin(θ_j)
    // Then θ_j = atan2(Im(u_j), Re(u_j))
    RVec theta(k);
    for (int j = 0; j < k; j++) {
        cx u_j = 0.0;
        for (int r = 0; r < k; r++) {
            cx Uv_r = 0.0;  // (U v_j)_r = Σ_c U[r][c] * V[c][j]
            for (int c = 0; c < k; c++)
                Uv_r += U[r][c] * V[c][j];
            u_j += std::conj(V[r][j]) * Uv_r;  // v_j† U v_j
        }
        theta[j] = std::atan2(std::imag(u_j), std::real(u_j));
    }

    // H = V diag(θ) V†, stored as H[col][row]
    H.resize(k, Vec(k, 0.0));
    for (int r = 0; r < k; r++) {
        for (int c = 0; c < k; c++) {
            cx sum = 0.0;
            for (int m = 0; m < k; m++)
                sum += V[r][m] * theta[m] * std::conj(V[c][m]);
            H[c][r] = sum;
        }
    }
}

// Extrapolate generator history and construct R_pred = exp(i H_pred).
// Linear (2 pts): H_pred = 2 H[0] - H[1]
// Quadratic (3 pts): H_pred = 3 H[0] - 3 H[1] + H[2]
// Returns k×k unitary matrix as cols[col][row].
std::vector<Vec> forecast_rotation(const EigenForecastState& state) {
    int k = state.k;
    int n = state.history_len;

    // Need at least 1 generator to extrapolate
    if (n < 1 || k <= 0) return {};

    // Extrapolate H_pred
    std::vector<Vec> H_pred(k, Vec(k, 0.0));
    if (n >= 3) {
        // Quadratic: 3 H[n-1] - 3 H[n-2] + H[n-3]
        auto& H0 = state.H_history[n-1];
        auto& H1 = state.H_history[n-2];
        auto& H2 = state.H_history[n-3];
        for (int i = 0; i < k; i++)
            for (int j = 0; j < k; j++)
                H_pred[i][j] = 3.0*H0[i][j] - 3.0*H1[i][j] + H2[i][j];
    } else if (n >= 2) {
        // Linear: 2 H[n-1] - H[n-2]
        auto& H0 = state.H_history[n-1];
        auto& H1 = state.H_history[n-2];
        for (int i = 0; i < k; i++)
            for (int j = 0; j < k; j++)
                H_pred[i][j] = 2.0*H0[i][j] - H1[i][j];
    } else {
        // Constant: just use the last generator (n == 1)
        H_pred = state.H_history[0];
    }

    // Compute R = exp(i H_pred) via eigendecomposition of Hermitian H_pred
    RVec evals;
    std::vector<Vec> evecs;
    lanczos_eigen(H_pred, k, evals, evecs);
    // evecs[row][col] = row-th component of col-th eigenvector
    // evals = θ_j (eigenvalues of H_pred)

    // R = V diag(e^{iθ}) V†  where V = evecs
    // R[col][row] = Σ_m evecs[row][m] * e^{iθ_m} * conj(evecs[col][m])
    std::vector<Vec> R(k, Vec(k, 0.0));
    for (int i = 0; i < k; i++) {       // row
        for (int j = 0; j < k; j++) {   // col
            cx sum = 0.0;
            for (int m = 0; m < k; m++) {
                sum += evecs[i][m] * std::exp(cx(0, evals[m])) * std::conj(evecs[j][m]);
            }
            R[j][i] = sum;  // stored as R[col][row]
        }
    }
    return R;
}

// Apply k×k rotation R to eigenvectors: new_v_i = Σ_j R[j][i] * old_v_j
// R stored as R[col][row] matching evecs convention from lanczos_eigen
void apply_rotation(std::vector<Vec>& eigvecs,
                    const std::vector<Vec>& R, int n) {
    int k = (int)eigvecs.size();
    std::vector<Vec> rotated(k);
    for (int i = 0; i < k; i++) {
        rotated[i] = zeros(n);
        for (int j = 0; j < k; j++) {
            axpy(R[j][i], eigvecs[j], rotated[i]);
        }
    }
    eigvecs = std::move(rotated);
}

// Multiply two k×k matrices: C = A × B (stored as cols[col][row])
void mat_mul_kk(const std::vector<Vec>& A, const std::vector<Vec>& B,
                std::vector<Vec>& C, int k) {
    C.resize(k, Vec(k, 0.0));
    // C[col_c][row] = Σ_m A[m][row] * B[col_c][m]
    // In matrix terms: C_rc = Σ_m A_rm * B_mc
    for (int r = 0; r < k; r++) {
        for (int c = 0; c < k; c++) {
            cx sum = 0.0;
            for (int m = 0; m < k; m++) {
                sum += A[m][r] * B[c][m];
            }
            C[c][r] = sum;
        }
    }
}

// Frobenius norm of k×k matrix stored as cols[col][row]
double frobenius_norm(const std::vector<Vec>& M, int k) {
    double sum = 0.0;
    for (int j = 0; j < k; j++)
        for (int i = 0; i < k; i++)
            sum += std::norm(M[j][i]);
    return std::sqrt(sum);
}

// =========================================================================
// Hybrid eigenvector tracker
// =========================================================================

HybridTrackerState hybrid_tracker_init(
    const OpApply& A,
    const std::function<void(const Vec&, Vec&)>& apply_D,
    const TRLMResult& trlm_result,
    int n, int n_ev, int n_kr)
{
    HybridTrackerState state;
    state.n = n;
    state.n_ev = n_ev;
    state.n_kr = n_kr;
    state.valid = true;

    // Start with the converged eigenvectors as the first n_ev Krylov vectors.
    // Fill the remaining n_kr - n_ev slots with random orthogonal vectors
    // (they'll be replaced by Lanczos extension on the first step).
    state.kSpace.resize(n_kr);
    state.Dv.resize(n_kr);
    state.eigvals.resize(n_ev);

    int k_have = std::min((int)trlm_result.eigvecs.size(), n_kr);
    for (int i = 0; i < k_have; i++) {
        state.kSpace[i] = trlm_result.eigvecs[i];
        state.Dv[i].resize(n);
        apply_D(state.kSpace[i], state.Dv[i]);
    }
    for (int i = 0; i < std::min(n_ev, (int)trlm_result.eigvals.size()); i++)
        state.eigvals[i] = trlm_result.eigvals[i];

    // Fill remaining slots with random vectors orthogonal to existing
    std::mt19937 rng(12345);
    for (int i = k_have; i < n_kr; i++) {
        state.kSpace[i] = random_vec(n, rng);
        for (int pass = 0; pass < 2; pass++)
            for (int j = 0; j < i; j++) {
                cx proj = dot(state.kSpace[j], state.kSpace[i]);
                axpy(-proj, state.kSpace[j], state.kSpace[i]);
            }
        double nv = norm(state.kSpace[i]);
        if (nv > 1e-14) scale(state.kSpace[i], cx(1.0 / nv));
        state.Dv[i].resize(n);
        apply_D(state.kSpace[i], state.Dv[i]);
    }

    return state;
}

HybridTrackResult hybrid_force_step(
    HybridTrackerState& state,
    const std::function<void(const Vec&, Vec&)>& apply_deltaD)
{
    int k = state.n_kr;
    int n = state.n;
    HybridTrackResult result;
    result.matvecs = 0;

    // Compute δD v_i for each Krylov vector
    std::vector<Vec> dDv(k);
    for (int i = 0; i < k; i++) {
        dDv[i].resize(n);
        apply_deltaD(state.kSpace[i], dDv[i]);
    }

    // Build k×k projected matrix for the new operator:
    // T_ij = (D+δD)†(D+δD) projected onto kSpace
    //      = (Dv_i + δDv_i)†(Dv_j + δDv_j)
    // This is EXACT (not perturbative) for the projection.
    // Note: for non-eigenvector kSpace, T_ij ≠ λ_i δ_ij + correction.
    // We compute the full projection.
    std::vector<Vec> Dv_new(k);
    for (int i = 0; i < k; i++) {
        Dv_new[i].resize(n);
        for (int s = 0; s < n; s++)
            Dv_new[i][s] = state.Dv[i][s] + dDv[i][s];
    }

    // T_ij = (D_new v_i)† (D_new v_j) = v_i† D†_new D_new v_j
    std::vector<Vec> T_cols(k, Vec(k, 0.0));
    for (int j = 0; j < k; j++) {
        for (int i = 0; i <= j; i++) {
            cx val = dot(Dv_new[i], Dv_new[j]);
            T_cols[j][i] = val;
            if (i != j) T_cols[i][j] = std::conj(val);
        }
    }

    // Diagonalise
    RVec evals;
    std::vector<Vec> evecs;
    lanczos_eigen(T_cols, k, evals, evecs);

    // Rotate kSpace and Dv
    std::vector<Vec> new_kSpace(k);
    std::vector<Vec> new_Dv(k);
    for (int i = 0; i < k; i++) {
        new_kSpace[i] = zeros(n);
        new_Dv[i] = zeros(n);
        for (int j = 0; j < k; j++) {
            cx c = evecs[j][i];
            axpy(c, state.kSpace[j], new_kSpace[i]);
            axpy(c, Dv_new[j], new_Dv[i]);
        }
    }
    state.kSpace = std::move(new_kSpace);
    state.Dv = std::move(new_Dv);

    // Update eigvals for the wanted n_ev
    result.eigvals.resize(state.n_ev);
    for (int i = 0; i < state.n_ev && i < k; i++)
        result.eigvals[i] = evals[i];
    state.eigvals = result.eigvals;

    result.max_residual = -1; // not computed
    return result;
}

HybridTrackResult hybrid_lanczos_step(
    HybridTrackerState& state,
    const OpApply& A,
    const std::function<void(const Vec&, Vec&)>& apply_D,
    int n_ext)
{
    int n = state.n;
    int n_kr = state.n_kr;
    if (n_ext <= 0) n_ext = n_kr - state.n_ev; // extend to fill full space

    HybridTrackResult result;
    result.matvecs = 0;

    // Step 1: Compute residual of worst kept Ritz vector to get Lanczos seed
    // r = A v_{n_ev} - λ_{n_ev} v_{n_ev}, orthogonalised against kSpace
    Vec r(n);
    A(state.kSpace[state.n_ev - 1], r);
    result.matvecs++;
    double rq = std::real(dot(state.kSpace[state.n_ev - 1], r));
    axpy(cx(-rq), state.kSpace[state.n_ev - 1], r);
    // Orthogonalise against all kSpace vectors
    for (int pass = 0; pass < 2; pass++)
        for (int j = 0; j < n_kr; j++) {
            cx proj = dot(state.kSpace[j], r);
            axpy(-proj, state.kSpace[j], r);
        }
    double r_norm = norm(r);
    if (r_norm < 1e-14) {
        // Subspace is already invariant — use random direction
        std::mt19937 rng(99999);
        r = random_vec(n, rng);
        for (int pass = 0; pass < 2; pass++)
            for (int j = 0; j < n_kr; j++) {
                cx proj = dot(state.kSpace[j], r);
                axpy(-proj, state.kSpace[j], r);
            }
        r_norm = norm(r);
    }
    scale(r, cx(1.0 / r_norm));

    // Step 2: Lanczos extension — generate n_ext new orthogonal vectors
    // via Lanczos applied to A, starting from r
    std::vector<Vec> ext_vecs;
    ext_vecs.reserve(n_ext);
    ext_vecs.push_back(r);

    for (int step = 0; step < n_ext; step++) {
        Vec Av(n);
        A(ext_vecs[step], Av);
        result.matvecs++;

        // Orthogonalise against kSpace + previous ext_vecs (double MGS)
        for (int pass = 0; pass < 2; pass++) {
            for (int j = 0; j < n_kr; j++) {
                cx proj = dot(state.kSpace[j], Av);
                axpy(-proj, state.kSpace[j], Av);
            }
            for (int j = 0; j <= step; j++) {
                cx proj = dot(ext_vecs[j], Av);
                axpy(-proj, ext_vecs[j], Av);
            }
        }
        double av_norm = norm(Av);
        if (av_norm < 1e-14) break; // Krylov space exhausted
        scale(Av, cx(1.0 / av_norm));

        if (step + 1 < n_ext)
            ext_vecs.push_back(std::move(Av));
    }

    int n_ext_actual = (int)ext_vecs.size();
    int total_dim = n_kr + n_ext_actual;

    // Step 3: Build (n_kr + n_ext) × (n_kr + n_ext) projected matrix
    // Build full projected matrix
    // kSpace-kSpace block: use Dv to avoid n_kr matvecs
    // v_i† D†D v_j = (D v_i)†(D v_j) = Dv[i]† Dv[j]
    std::vector<Vec> T_cols(total_dim, Vec(total_dim, 0.0));

    for (int j = 0; j < n_kr; j++)
        for (int i = 0; i <= j; i++) {
            cx val = dot(state.Dv[i], state.Dv[j]);
            T_cols[j][i] = val;
            if (i != j) T_cols[i][j] = std::conj(val);
        }

    // kSpace-ext cross block: need A * ext_vecs
    std::vector<Vec> A_ext(n_ext_actual);
    for (int i = 0; i < n_ext_actual; i++) {
        A_ext[i].resize(n);
        A(ext_vecs[i], A_ext[i]);
        result.matvecs++;
    }

    for (int j = 0; j < n_ext_actual; j++)
        for (int i = 0; i < n_kr; i++) {
            cx val = dot(state.kSpace[i], A_ext[j]);
            T_cols[n_kr + j][i] = val;
            T_cols[i][n_kr + j] = std::conj(val);
        }

    // ext-ext block
    for (int j = 0; j < n_ext_actual; j++)
        for (int i = 0; i <= j; i++) {
            cx val = dot(ext_vecs[i], A_ext[j]);
            T_cols[n_kr + j][n_kr + i] = val;
            if (i != j) T_cols[n_kr + i][n_kr + j] = std::conj(val);
        }

    // Step 4: Diagonalise and keep best n_kr Ritz vectors
    RVec evals;
    std::vector<Vec> evecs;
    lanczos_eigen(T_cols, total_dim, evals, evecs);

    // Rotate: new kSpace[i] = sum over j of evecs[j][i] * (all_vecs[j])
    // where all_vecs = [kSpace[0..n_kr-1], ext_vecs[0..n_ext-1]]
    std::vector<Vec> new_kSpace(n_kr);
    for (int i = 0; i < n_kr; i++) {
        new_kSpace[i] = zeros(n);
        for (int j = 0; j < n_kr; j++)
            axpy(evecs[j][i], state.kSpace[j], new_kSpace[i]);
        for (int j = 0; j < n_ext_actual; j++)
            axpy(evecs[n_kr + j][i], ext_vecs[j], new_kSpace[i]);
    }
    state.kSpace = std::move(new_kSpace);

    // Recompute Dv for the new kSpace (needed for next force step)
    state.Dv.resize(n_kr);
    for (int i = 0; i < n_kr; i++) {
        state.Dv[i].resize(n);
        apply_D(state.kSpace[i], state.Dv[i]);
    }

    // Update eigvals
    result.eigvals.resize(state.n_ev);
    for (int i = 0; i < state.n_ev && i < total_dim; i++)
        result.eigvals[i] = evals[i];
    state.eigvals = result.eigvals;

    // Compute residual for the n_ev wanted eigenvectors
    result.max_residual = 0;
    for (int i = 0; i < state.n_ev; i++) {
        Vec Avi(n);
        A(state.kSpace[i], Avi);
        result.matvecs++;
        double av_norm = norm(Avi);
        axpy(cx(-result.eigvals[i]), state.kSpace[i], Avi);
        double res_i = norm(Avi) / std::max(av_norm, 1e-30);
        result.max_residual = std::max(result.max_residual, res_i);
    }

    return result;
}

// =================================================================
//  EigenTracker implementation
// =================================================================

void EigenTracker::init(
    const TRLMResult& trlm,
    const std::function<void(const Vec&, Vec&)>& apply_D,
    int n_, int n_ev_, int pool_capacity_)
{
    n = n_;
    n_ev = n_ev_;
    pool_capacity = pool_capacity_;

    int k_have = std::min((int)trlm.eigvecs.size(), pool_capacity);
    pool.resize(k_have);
    Dpool.resize(k_have);
    for (int i = 0; i < k_have; i++) {
        pool[i] = trlm.eigvecs[i];
        Dpool[i].resize(n);
        apply_D(pool[i], Dpool[i]);
    }

    // Fill remaining slots with random vectors orthogonal to existing
    std::mt19937 rng(54321);
    while ((int)pool.size() < pool_capacity) {
        Vec v = random_vec(n, rng);
        for (int pass = 0; pass < 2; pass++)
            for (int j = 0; j < (int)pool.size(); j++) {
                cx proj = dot(pool[j], v);
                axpy(-proj, pool[j], v);
            }
        double nv = norm(v);
        if (nv < 1e-14) continue;
        scale(v, cx(1.0 / nv));
        Vec Dv(n);
        apply_D(v, Dv);
        pool.push_back(std::move(v));
        Dpool.push_back(std::move(Dv));
    }

    eigvals.resize(n_ev);
    for (int i = 0; i < n_ev && i < (int)trlm.eigvals.size(); i++)
        eigvals[i] = trlm.eigvals[i];

    valid = true;
}

void EigenTracker::force_update(
    const std::function<void(const Vec&, Vec&)>& apply_deltaD)
{
    if (!valid) return;
    int k = (int)pool.size();

    // Update Dpool: Dv_new = Dv_old + δD v
    for (int i = 0; i < k; i++) {
        Vec dDv(n);
        apply_deltaD(pool[i], dDv);
        #pragma omp parallel for schedule(static) if(n > OMP_MIN_SIZE)
        for (int s = 0; s < n; s++)
            Dpool[i][s] += dDv[s];
    }

    // Re-project and re-diag using updated Dpool
    compress();
}

int EigenTracker::absorb(
    const std::vector<Vec>& new_vecs,
    const std::function<void(const Vec&, Vec&)>& apply_D)
{
    if (!valid) return 0;
    int absorbed = 0;

    for (const auto& v_in : new_vecs) {
        if ((int)pool.size() >= pool_capacity + (int)new_vecs.size())
            break; // will compress later

        Vec v = v_in;
        // Orthogonalise against existing pool (double MGS)
        for (int pass = 0; pass < 2; pass++)
            for (int j = 0; j < (int)pool.size(); j++) {
                cx proj = dot(pool[j], v);
                axpy(-proj, pool[j], v);
            }
        double nv = norm(v);
        if (nv < 0.1) continue; // too much overlap — skip

        scale(v, cx(1.0 / nv));
        Vec Dv(n);
        apply_D(v, Dv);
        pool.push_back(std::move(v));
        Dpool.push_back(std::move(Dv));
        absorbed++;
    }

    // If pool exceeds capacity, compress
    if ((int)pool.size() > pool_capacity)
        compress();

    return absorbed;
}

void EigenTracker::chebyshev_probe(
    const OpApply& A,
    const std::function<void(const Vec&, Vec&)>& apply_D,
    std::mt19937& rng,
    double lambda_max, int degree)
{
    if (!valid || degree < 1) return;

    // Generate random vector
    Vec x = random_vec(n, rng);

    // Chebyshev filter: amplifies components with eigenvalue < lambda_cut
    // Set lambda_cut just above the n_ev-th eigenvalue
    double lambda_cut = (n_ev < (int)eigvals.size()) ?
        eigvals[n_ev - 1] * 1.5 : lambda_max * 0.1;
    lambda_cut = std::min(lambda_cut, lambda_max * 0.5);

    double sigma = (lambda_max - lambda_cut) / 2.0;
    double c_center = (lambda_max + lambda_cut) / 2.0;
    if (sigma < 1e-14) return;

    // 3-term Chebyshev recurrence on scaled operator (A - c I) / σ
    // T_0(x) = x,  T_1 = Âx = (Ax - cx)/σ,  T_{k+1} = 2Â T_k - T_{k-1}
    Vec y_prev = x;         // T_0
    Vec Ax_buf(n);
    A(y_prev, Ax_buf);
    Vec y_curr(n);
    #pragma omp parallel for schedule(static) if(n > OMP_MIN_SIZE)
    for (int s = 0; s < n; s++)
        y_curr[s] = (Ax_buf[s] - c_center * y_prev[s]) / sigma;  // T_1

    for (int d = 2; d <= degree; d++) {
        A(y_curr, Ax_buf);
        Vec y_next(n);
        #pragma omp parallel for schedule(static) if(n > OMP_MIN_SIZE)
        for (int s = 0; s < n; s++)
            y_next[s] = 2.0 * (Ax_buf[s] - c_center * y_curr[s]) / sigma
                      - y_prev[s];
        y_prev = std::move(y_curr);
        y_curr = std::move(y_next);
    }

    // Orthogonalise filtered vector against pool
    for (int pass = 0; pass < 2; pass++)
        for (int j = 0; j < (int)pool.size(); j++) {
            cx proj = dot(pool[j], y_curr);
            axpy(-proj, pool[j], y_curr);
        }
    double nv = norm(y_curr);
    if (nv < 1e-14) return;
    scale(y_curr, cx(1.0 / nv));

    // Compute D(filtered vector) and add to pool
    Vec Dv(n);
    apply_D(y_curr, Dv);
    pool.push_back(std::move(y_curr));
    Dpool.push_back(std::move(Dv));

    // Compress if over capacity
    if ((int)pool.size() > pool_capacity)
        compress();
}

void EigenTracker::compress()
{
    if (!valid) return;
    int k = (int)pool.size();
    if (k <= 1) return;

    // Build k×k projected matrix: T_ij = Dpool[i]† Dpool[j]
    std::vector<Vec> T_cols(k, Vec(k, 0.0));
    for (int j = 0; j < k; j++)
        for (int i = 0; i <= j; i++) {
            cx val = dot(Dpool[i], Dpool[j]);
            T_cols[j][i] = val;
            if (i != j) T_cols[i][j] = std::conj(val);
        }

    // Diagonalise
    RVec evals;
    std::vector<Vec> evecs;
    lanczos_eigen(T_cols, k, evals, evecs);

    // Keep at most pool_capacity vectors (smallest eigenvalue first)
    int keep = std::min(k, pool_capacity);
    std::vector<Vec> new_pool(keep);
    std::vector<Vec> new_Dpool(keep);
    for (int i = 0; i < keep; i++) {
        new_pool[i] = zeros(n);
        new_Dpool[i] = zeros(n);
        for (int j = 0; j < k; j++) {
            cx c = evecs[j][i];
            axpy(c, pool[j], new_pool[i]);
            axpy(c, Dpool[j], new_Dpool[i]);
        }
        // Renormalise pool vector (Dpool follows the same rotation)
        double nv = norm(new_pool[i]);
        if (nv > 1e-14 && std::abs(nv - 1.0) > 1e-10) {
            cx s = cx(1.0 / nv);
            scale(new_pool[i], s);
            scale(new_Dpool[i], s);
        }
    }
    pool = std::move(new_pool);
    Dpool = std::move(new_Dpool);

    // Update eigenvalue estimates
    eigvals.resize(std::min(n_ev, keep));
    for (int i = 0; i < (int)eigvals.size(); i++)
        eigvals[i] = evals[i];
}

void EigenTracker::perturbation_extend(
    const std::function<void(const Vec&, Vec&)>& apply_deltaD,
    const std::function<void(const Vec&, Vec&)>& apply_deltaD_dag,
    const std::function<void(const Vec&, Vec&)>& apply_D_dag,
    const std::function<void(const Vec&, Vec&)>& apply_D)
{
    if (!valid) return;
    int k_ev = std::min(n_ev, (int)pool.size());

    for (int i = 0; i < k_ev; i++) {
        // Compute perturbation direction: δ(D†D) v_i
        //   = δD†(D v_i) + D†(δD v_i) + δD†(δD v_i)

        // δD v_i  (sparse, ~1/4 D cost)
        Vec dDv(n);
        apply_deltaD(pool[i], dDv);

        // Term 1: δD†(D v_i)  — sparse, use stored Dpool[i]
        Vec t1(n);
        apply_deltaD_dag(Dpool[i], t1);

        // Term 2: D†(δD v_i)  — full D† application (most expensive)
        Vec t2(n);
        apply_D_dag(dDv, t2);

        // Term 3: δD†(δD v_i)  — sparse, second-order
        Vec t3(n);
        apply_deltaD_dag(dDv, t3);

        // r_i = t1 + t2 + t3  (full perturbation direction)
        Vec r(n);
        #pragma omp parallel for schedule(static) if(n > OMP_MIN_SIZE)
        for (int s = 0; s < n; s++)
            r[s] = t1[s] + t2[s] + t3[s];

        // Project out subspace component (double MGS)
        for (int pass = 0; pass < 2; pass++)
            for (int j = 0; j < (int)pool.size(); j++) {
                cx proj = dot(pool[j], r);
                axpy(-proj, pool[j], r);
            }

        double nr = norm(r);
        if (nr < 1e-14) continue;  // eigenvector stays in subspace
        scale(r, cx(1.0 / nr));

        // Compute D(r) for Dpool cache
        Vec Dr(n);
        apply_D(r, Dr);

        pool.push_back(std::move(r));
        Dpool.push_back(std::move(Dr));
    }

    // Compress if over capacity
    if ((int)pool.size() > pool_capacity)
        compress();
}

double EigenTracker::max_residual(const OpApply& A) const
{
    double max_res = 0;
    int k = std::min(n_ev, (int)pool.size());
    for (int i = 0; i < k; i++) {
        Vec Av(n);
        A(pool[i], Av);
        double av_norm = norm(Av);
        // r = Av - λv
        Vec resid(Av);
        axpy(cx(-eigvals[i]), pool[i], resid);
        double res_i = norm(resid) / std::max(av_norm, 1e-30);
        max_res = std::max(max_res, res_i);
    }
    return max_res;
}
