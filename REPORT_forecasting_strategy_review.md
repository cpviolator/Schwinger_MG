# Eigenspace Forecasting Strategy: Review and Analysis

## What are we predicting?

The current forecasting (eigen_forecast.cpp) predicts the **rotation of eigenvectors
within a fixed k-dimensional subspace**. It does NOT predict:
- New eigenvector directions (outside the current span)
- Eigenvalue trajectories (these come from Rayleigh quotients after rotation)
- Subspace expansion or contraction

### The prediction pipeline

Given eigenvectors {v_1, ..., v_k} of D†D at gauge config U_t:

1. Gauge evolves: U_t → U_{t+1}
2. **RR projection**: project new D†D_{t+1} into span{v_1,...,v_k}
   - M_ij = v_i† D†D_{t+1} v_j  (k×k projected matrix, costs k matvecs)
   - Diagonalise: M = U Λ U†
   - New eigenvectors: w_i = Σ_j U_ji v_j  (rotation within span)
   - New eigenvalues: Λ_ii  (Rayleigh quotients within span)
3. **Generator extraction**: H = -i log(U) where U = exp(iH), H Hermitian
4. **History storage**: store H_t in circular buffer
5. **Forecasting**: extrapolate H_{t+1} from {H_t, H_{t-1}, ...}
   - Constant (1 sample): H_pred = H_t
   - Linear (2 samples): H_pred = 2H_t - H_{t-1}
   - Quadratic (3 samples): H_pred = 3H_t - 3H_{t-1} + H_{t-2}
6. **Pre-rotation**: R_pred = exp(i H_pred), then v_i → Σ_j R_pred_ji v_j

### What the generator H actually encodes

H is a k×k Hermitian matrix with eigenvalues θ_1, ..., θ_k (the rotation angles).
For small gauge changes (dt << 1), H ≈ O(dt) and the rotation is near-identity.

The off-diagonal elements of H encode **mixing between eigenvectors** — how much
v_1 rotates into v_2, etc. This mixing is driven by the change δ(D†D) in the
operator. First-order perturbation theory gives:

    H_ij ≈ <v_i| δ(D†D) |v_j> / (λ_j - λ_i)   for i ≠ j

So H is largest when eigenvectors are nearly degenerate (small λ_j - λ_i) and the
perturbation has large matrix elements between them.

## Fundamental flaw: subspace leakage

**The critical issue**: RR operates within span{v_1,...,v_k}. After the gauge
changes, the TRUE eigenvectors of D†D_{t+1} have components OUTSIDE this span.
The residual measures this leakage:

    ||D†D v - λv|| / ||D†D v|| ≈ sin(angle between v and true eigenvector)

Our tracking experiment showed:
- Step 0: residual = 1e-12 (converged)
- Step 1: residual = 0.87 (catastrophic — 87% of the eigenvector is outside the span)
- Steps 2-20: residual stays 0.87-0.97 (span is essentially random w.r.t. true eigenvectors)

**Forecasting within this degraded span is meaningless.** Whether we predict the
rotation or not, we're rotating vectors that are already 87% wrong. This explains
why forecast+RR = bare RR in every metric (CG, residual, eigenvalue error).

## What would work: subspace expansion

The problem is not prediction accuracy — it's that the k-dimensional span is too
small. After one gauge step, the true eigenvectors have leaked into the orthogonal
complement (the other N-k directions). We need to bring in information from outside
the span.

### Strategy 1: TRLM warm-start with larger Krylov space

TRLM builds a Krylov space of dimension n_kr >> n_ev during convergence. This
Krylov basis spans a larger subspace that includes the near-null directions PLUS
buffer directions. After gauge evolution:

- RR in the full n_kr space (not just n_ev) captures much of the leakage
- The arrow matrix (tridiagonal from Lanczos) encodes coupling between converged
  and unconverged Ritz vectors — this IS the spectral drift information
- Forecasting in the n_kr space is more useful because the drift stays within
  the tracked space for longer

Cost: n_kr matvecs per step instead of n_ev. For n_kr = 2*n_ev + 16 and n_ev = 4,
this is 24 matvecs vs 4 — 6× more expensive but captures the drift.

### Strategy 2: Periodic warm inverse iteration

Our tracking experiment showed that warm inverse iteration (3 steps from RR-tracked
vectors every 5 gauge steps) achieves CG=40 vs fresh CG=39. This works because:

- MR smoothing in inverse iteration brings in NEW directions from the operator
- 3 steps amplify near-null components by ~1/λ_min^3
- The warm start means we only need ~3 iterations instead of 20

Cost: 3 × k × 5 MR steps = 60 MR applications every 5 gauge steps = 12 MR/step.
Much cheaper than n_kr matvecs for Strategy 1.

### Strategy 3: Combined forecast + expand

The ideal approach:
1. Track n_kr vectors with RR (captures drift within the larger space)
2. Forecast the n_kr × n_kr rotation (smooth, predictable in the larger space)
3. Periodically refresh the n_kr - n_ev buffer with Lanczos extension (brings
   truly new directions from the operator)
4. At CG solve time, extract the n_ev best vectors from the tracked n_kr pool

This is the **Hybrid Tracker** architecture already in eigensolver.h.

## Eigenvalue vs eigenvector prediction

### Eigenvalue prediction (currently NOT done independently)

Eigenvalues λ_i of D†D change smoothly with the gauge field. In principle, we could
predict λ_i(t+1) from λ_i(t), λ_i(t-1), ... independently of the eigenvectors.
But this is not useful for MG because:

- MG needs the eigenvectors (for the prolongator), not the eigenvalues
- Correct eigenvalues with wrong eigenvectors don't help CG convergence
- The eigenvalues come for free from RR (Rayleigh quotients)

### Eigenvector prediction (current approach)

The generator H predicts how eigenvectors ROTATE within the span. This is:
- Cheap to compute (k×k eigendecomposition, O(k³))
- Smooth and predictable when the span is correct
- Useless when the span has drifted

### What we should predict instead

Rather than predicting the rotation within a fixed span, we should predict which
NEW DIRECTIONS will enter the near-null space. Perturbation theory gives:

    δv_i ≈ Σ_{j≠i} <v_j| δ(D†D) |v_i> / (λ_i - λ_j) × v_j

The dominant correction comes from the eigenvectors v_j with eigenvalues closest
to λ_i (small denominator). These are exactly the vectors in the TRLM buffer
(indices n_ev+1, ..., n_kr). By tracking these buffer vectors and predicting which
ones will become important, we can proactively expand the span.

## Recommendations

1. **Stop forecasting the rotation in the n_ev span.** It doesn't help when the
   span leaks (which happens after 1 gauge step at our parameters).

2. **Track a larger subspace** (n_kr = 2-3× n_ev). Use RR in this larger space
   to capture drift. The overhead is n_kr matvecs per step.

3. **Use warm inverse iteration** (3 steps from tracked vectors every 5 gauge
   steps) to bring in genuinely new directions from the operator. This is the
   most cost-effective approach at our lattice sizes.

4. **Forecast in the larger space** only if the n_kr RR residuals stay small.
   The generator in the n_kr space is smoother and more predictable.

5. **At CG time**: extract the n_ev best vectors from the tracked pool and build
   the MG prolongator. The remaining n_kr - n_ev vectors serve as a quality
   buffer for the next forecasting cycle.

## Experimental evidence

From the L=32 tracking experiment (20 MD steps, dt=0.05, m=0.1, k=4):

| Method | Final max_res | ev0 error | MG CG | vs Fresh |
|--------|-------------|-----------|-------|----------|
| Stale | 0.87 | 0.021 | 52 | +33% |
| RR-every-step | 0.87 | 1.51 | 52 | +33% |
| Forecast+RR | 0.87 | 1.51 | 52 | +33% |
| RR + warm/5 | 0.57 | 0.16 | 40 | +3% |
| Fresh TRLM | ~1e-11 | 0 | 39 | baseline |

RR and forecast+RR are identical — confirming that rotation prediction within a
leaked subspace adds no value. Warm refinement every 5 steps nearly matches fresh
TRLM, confirming that new directions from the operator are what matters.
