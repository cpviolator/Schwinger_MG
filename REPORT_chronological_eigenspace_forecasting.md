# Chronological Forecasting for Eigenspace Evolution

## 1. Problem Statement

In multi-timescale HMC with coarse-grid deflation, we maintain k eigenvectors of the
coarse operator A_c. As the gauge field evolves (between trajectories and within MD
integration), A_c changes and the eigenvectors must be updated.

Currently we use **Rayleigh-Ritz (RR)** projection:
1. Apply A_c^new to each old eigenvector: AV_i = A_c^new v_i  (**k matvecs**)
2. Build projected matrix: M_ij = v_i^dag A_c^new v_j  (**k^2 inner products**)
3. Diagonalise M = U Lambda U^dag  (**O(k^3) Lanczos/QL**)
4. Rotate: w_j = sum_i U_ij v_i  (**k^2 work in n-dimensional space**)

The matrix U is **unitary** (k x k) and represents the optimal rotation within the
subspace. The question: can we **predict U** from its history, applying the predicted
rotation before (or instead of) RR?

## 2. Structure of the Rotation Matrix

### 2.1 Unitary Eigendecomposition

U is unitary, so its eigenvalues lie on the unit circle:

    U = V diag(e^{i theta_1}, ..., e^{i theta_k}) V^dag

where V is unitary and theta_j are real **eigenangles**. For small gauge changes,
theta_j ~ 0 (U ~ I). The angles theta_j are the natural coordinates for forecasting.

### 2.2 Hermitian Generator

Equivalently, U = exp(iH) where H is the k x k Hermitian **generator**:

    H = V diag(theta_1, ..., theta_k) V^dag

For small gauge changes: H ~ (M - Lambda) / Lambda (off-diagonal part of the
normalised projected matrix). The generator is a smooth function of the gauge field,
making it suitable for polynomial extrapolation.

### 2.3 Information Available for Free

At each RR step, we already compute:
- M (the k x k projected matrix)
- U (eigenvectors of M) = the rotation matrix
- Lambda (eigenvalues of M) = new eigenvalues

So extracting U costs nothing extra. The eigendecomposition of U (to get theta_j, V)
costs O(k^3) but k is typically 8-16, so this is negligible.

## 3. Forecasting Strategies

### Strategy A: Forecast Eigenangles theta_j

**Idea**: Track the eigenangles theta_j^(t) across trajectories. Extrapolate:
theta_j^(t+1) ~ c_0 theta_j^(t) + c_1 theta_j^(t-1) + ... (polynomial/linear).

Construct predicted rotation: R_pred = V^(t) diag(e^{i theta_pred}) V^(t) dag.
Apply R_pred to pre-rotate eigenvectors before RR.

**Subtlety**: The eigenvectors V of U change between trajectories (they're in different
bases). Two approaches:
- (a) Assume V is slowly varying; use V from the last trajectory
- (b) Track V in a cumulative frame by composing rotations: V_cumul = V^(t) V^(t-1) ...

**Complexity**: O(k^3) to diagonalise U + O(k) scalar extrapolations
**FLOPs**: ~k^3 + k multiplications (negligible vs. k coarse matvecs)
**Benefit**: Pre-rotation makes the RR correction small. If the prediction is good
enough, we can **skip RR entirely** on some trajectories, saving k coarse matvecs.

**Risk**: If V changes significantly, the predicted rotation is in the wrong basis.
Phase ambiguity in eigenvector ordering can cause discontinuities in theta_j.

### Strategy B: Forecast the Generator H

**Idea**: Instead of decomposing U into angles and V, work with the generator
H = -i log(U) directly. H is Hermitian, so linear combinations preserve Hermiticity.

Forecast: H^(t+1) ~ 2 H^(t) - H^(t-1) (linear), or higher-order polynomial.
Predicted rotation: R_pred = exp(i H_pred).

For k = 8-16, the matrix exponential is computed via eigendecomposition of H_pred
(O(k^3)), which is negligible.

**Complexity**: O(k^3) for matrix log + O(k^2) for extrapolation
**FLOPs**: ~2 k^3 (log + exp)
**Benefit**: Same as Strategy A, but avoids the phase-ordering ambiguity. H is a
smooth matrix-valued function of the gauge field.

**Advantage over A**: No need to track V separately. H encodes the full rotation
including the basis. Linear extrapolation in the space of Hermitian matrices is natural
and preserves the Lie algebra structure.

**Risk**: For large rotations (theta_j ~ pi), log(U) has branch cuts. In practice,
HMC gauge changes are small, so theta_j << 1 and log(U) ~ U - I is well-behaved.

### Strategy C: Perturbation Theory Forecast

**Idea**: Use first-order perturbation theory to predict the rotation:

    U_ij^(1) ~ delta_ij + (v_i^dag delta_A v_j) / (lambda_i - lambda_j)   for i != j

where delta_A = A_c^new - A_c^old is the change in the coarse operator.

If we can predict delta_A (e.g., from the MD forces), we get a predicted rotation
without any matvecs. This is related to the existing `force_evolve` function which
uses the fine-grid delta_D to compute the perturbative correction.

**Complexity**: Medium — need delta_A or its projection onto the eigenvector subspace
**FLOPs**: O(k^2) inner products if delta_A v_i is available
**Benefit**: Zero-matvec prediction. Good for within-trajectory updates.

**Risk**: First-order perturbation theory breaks down when eigenvalues are nearly
degenerate (lambda_i ~ lambda_j). The formula diverges, and mixing between
near-degenerate modes is poorly predicted.

### Strategy D: Chebyshev-Window Tracking

**Idea**: With Chebyshev acceleration, TRLM maps eigenvalues to [0, 1] via a
polynomial filter. The wanted eigenvalues (smallest of the original A_c) become the
**largest** of the Chebyshev-filtered operator. As the gauge evolves:

- Eigenvalues within the window shift smoothly
- Eigenvalues near the window boundary may enter or exit
- The Chebyshev mapping amplifies separation, making tracking easier

Track the **Ritz values in the Chebyshev-mapped spectrum** across trajectories. When
a Ritz value approaches the window boundary from outside, predict that a new
eigenvector is about to enter. Pre-emptively extend the deflation subspace with a
Chebyshev-filtered random vector to capture it.

**Complexity**: High — requires monitoring spectrum boundaries + Chebyshev filtering
**FLOPs**: O(poly_deg) matvecs per candidate vector
**Benefit**: High when eigenvalue crossings occur (topology changes, phase transitions,
or at light quark masses where the spectrum is dense near zero). Prevents sudden
degradation of deflation quality.

**Risk**: Over-engineering for smooth gauge evolution where no crossings occur.
Cost of Chebyshev-filtering extra vectors may exceed the benefit.

### Strategy E: Chronological Forecasting on M directly

**Idea**: Track the k x k projected matrix M^(t) across trajectories. But M is
always expressed in the current eigenbasis (after rotation), so M^(t) = Lambda^(t)
(diagonal). Not directly useful.

Instead, track the **off-diagonal structure of M before diagonalisation** — i.e.,
the raw projected matrix in the old eigenbasis:

    M_raw^(t) = old_v_i^dag A_c^(t) old_v_j

Forecast M_raw^(t+1), diagonalise to get the predicted rotation, apply it.

**Complexity**: O(k^2) storage per trajectory
**FLOPs**: O(k^2) extrapolation + O(k^3) diagonalisation
**Benefit**: Equivalent to Strategy B (forecasting the generator) since
H ~ (M_raw - Lambda_old) for small changes. But M_raw has a clearer physical
interpretation.

**Risk**: Same basis-dependence issue as Strategy A. After rotation, the next
M_raw is in a different basis.

## 4. Comparison Table

| Strategy | Complexity | Extra FLOPs | Can Skip RR? | Handles Crossings? | Basis-Invariant? |
|----------|-----------|-------------|-------------|-------------------|-----------------|
| A: theta_j forecast | Low | ~k^3 | Yes (if accurate) | No | No (phase ambiguity) |
| B: Generator H | Low-Med | ~2k^3 | Yes (if accurate) | No | Yes (Lie algebra) |
| C: Perturbation | Medium | ~k^2 n_coarse | Partial | No (diverges at crossings) | N/A |
| D: Chebyshev window | High | ~poly_deg matvecs | No (different purpose) | Yes | N/A |
| E: M_raw forecast | Low | ~k^3 | Yes (if accurate) | No | No |

## 5. Recommended Approach: Strategy B + D Hybrid

### Primary: Generator Forecasting (Strategy B)

1. After each RR, extract the rotation matrix U (already computed).
2. Compute the generator: H = -i log(U). For small rotations, H ~ -i(U - I).
   For general U, diagonalise U to get eigenangles, construct H = V diag(theta) V^dag.
3. Store H in a circular buffer (last N_history generators).
4. Forecast: H_pred = extrapolate(H_history). Linear (2-point) or quadratic (3-point).
5. Compute predicted rotation: R_pred = exp(i H_pred). For small H, R_pred ~ I + iH_pred.
6. Pre-rotate eigenvectors: v_pred = R_pred v_old.
7. Do RR on v_pred (or skip if residuals are small).

**Cost**: negligible (k=8-16, all operations are O(k^3) ~ 4096 FLOPs).

**Benefit assessment**:
- If theta_j ~ 0.01-0.1 rad/trajectory (typical), linear extrapolation predicts theta
  to ~0.001-0.01 accuracy. The RR correction is then a rotation of ~0.001-0.01 rad.
- This means the projected matrix M in the pre-rotated basis is nearly diagonal,
  so RR converges trivially. The main saving is that we can **skip RR** on
  trajectories where the prediction is accurate, checking via a cheap residual estimate.

### Supplement: Chebyshev Window Monitoring (Strategy D)

Track the k-th eigenvalue (boundary of the deflation window) and the (k+1)-th
Ritz value (first unwanted mode). When they approach each other, trigger a fresh
TRLM to capture the crossing.

This addresses the failure mode where forecasting can't help: a new eigenmode
entering the subspace that wasn't there before.

## 6. Implementation Sketch

```cpp
struct EigenForecastState {
    std::vector<std::vector<Vec>> H_history;  // circular buffer of generators
    int history_len = 0;
    int max_history = 3;  // for quadratic extrapolation

    // Extract generator from rotation matrix U (k x k)
    // H = V diag(theta) V^dag where U = V diag(e^{i theta}) V^dag
    std::vector<Vec> extract_generator(const std::vector<Vec>& U_evecs,
                                        const RVec& U_evals_re,
                                        const RVec& U_evals_im, int k);

    // Extrapolate generator and construct predicted rotation
    std::vector<Vec> predict_rotation(int k);

    // Apply rotation to eigenvectors
    void pre_rotate(std::vector<Vec>& eigvecs, const std::vector<Vec>& R_pred, int n);
};
```

The `evolve_coarse_deflation` function would change to:

```cpp
void evolve_coarse_deflation(CoarseDeflState& cdefl,
    const SparseCoarseOp& Ac_new,
    EigenForecastState& forecast)
{
    if (cdefl.eigvecs.empty()) return;

    // 1. Predict rotation from history
    if (forecast.history_len > 0) {
        auto R_pred = forecast.predict_rotation(k);
        forecast.pre_rotate(cdefl.eigvecs, R_pred, Ac_new.dim);
    }

    // 2. RR on pre-rotated vectors (correction should be small)
    OpApply op = Ac_new.as_op();
    auto rr = rr_evolve(op, cdefl.eigvecs, Ac_new.dim);

    // 3. Extract generator from the RR rotation U and store
    forecast.store_generator(rr_rotation_matrix, k);

    // 4. Update state
    cdefl.eigvecs = std::move(rr.eigvecs);
    cdefl.eigvals = std::move(rr.eigvals);
}
```

## 7. When is This Useful?

### Useful:
- **Long HMC runs** where RR is called every trajectory — amortised cost matters
- **Large k** (16-32 deflation vectors) where k coarse matvecs dominate
- **Within-trajectory updates** where RR is too expensive to call at every inner step
- **Adaptive RR skipping**: use the prediction to decide when RR is needed

### Not useful:
- **k is small** (8) and coarse matvecs are cheap — RR is already fast
- **Large gauge changes** per trajectory (cold start, or very long tau) — prediction
  is inaccurate, RR is needed anyway
- **Eigenvalue crossings** — prediction fails, need fresh TRLM

## 8. Relation to Chronological CG Forecasting

The standard chronological forecasting technique (Brower et al., hep-lat/9509012)
predicts the CG initial guess from past solutions. The same mathematical framework
applies here:

**CG forecasting**: x_pred = argmin ||x - sum_i c_i x_past_i||^2 s.t. sum c_i = 1
**Rotation forecasting**: H_pred = argmin ||H - sum_i c_i H_past_i||_F^2

Both exploit the smoothness of the trajectory through configuration space. The
rotation forecasting is actually simpler because H lives in a k-dimensional vector
space (k^2 matrix entries), while the CG solution lives in n-dimensional space
(much larger).

The key difference: CG forecasting helps with the **solve** (initial guess).
Rotation forecasting helps with the **eigenvector maintenance** (deflation quality).
They are complementary and can be used together.

## 9. Open Questions

1. **Optimal extrapolation order**: Linear (2-point) vs quadratic (3-point) vs
   higher. In CG forecasting, 3-5 past solutions work best.

2. **When to skip RR**: Need a cheap residual estimator that doesn't require matvecs.
   Possible: use the forecasted eigenvalues to estimate residuals.

3. **Phase tracking for Strategy A**: How to handle eigenangle reordering when
   eigenvalues cross. Strategy B (generator) avoids this naturally.

4. **Interaction with fresh TRLM**: After a fresh TRLM, the eigenvectors jump
   discontinuously. The forecast history should be reset.

5. **Within-trajectory application**: Can we forecast the rotation during MD
   evolution (between inner steps) using the gauge-field momentum as the
   derivative? This would be Strategy C (perturbation) applied continuously.
