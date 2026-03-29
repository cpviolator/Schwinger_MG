# Prolongator Refresh Study Results

## Test Configuration
- L=32, beta=2.0, mass=0.05, Wilson (c_sw=0)
- MG: 2 levels, block=4, k_null=8, n_defl=16
- Integrator: Nested FGI, n_outer=10, n_inner=5, total_steps=50
- defl_refresh=5, maxiter=1000, tau=1.0
- Thermalised from gauge_L32_b2.00_t50.bin (plaq=0.5986)
- seed=42, same RNG (seed+3000) for all arms
- 1 OMP thread (single-threaded runs)

## Established Baselines (20 trajectories, completed)

| Arm | Accept | Avg|dH| | AvgCG | AvgTime | CG_ratio |
|-----|--------|---------|-------|---------|----------|
| Stale | 65% | 0.764 | 2559 | 15.08s | 1.000 |
| Rebuild/5 | 70% | 0.795 | 1863 | 10.13s | 0.728 |
| RR+Rebuild/5 | 45% | 1.09 | 1963 | 10.79s | 0.767 |

Source: bwdnsnw2a (completed)

## Earlier Results (n_outer=5, n_inner=3, total_steps=15)

| Arm | Accept | Avg|dH| | AvgCG | AvgTime | CG_ratio |
|-----|--------|---------|-------|---------|----------|
| Stale | 55% | 2.55 | 1373 | 7.95s | 1.000 |
| Rebuild/2 | 60% | 1.67 | 971 | 5.22s | 0.707 |
| RR+Rebuild/5 | 55% | 2.15 | 1035 | 5.73s | 0.754 |
| RR+Rebuild/10 | 60% | 2.03 | 1149 | 6.21s | 0.836 |

Source: b9g7ishqn (completed)

## Broken Perturbation Results (fixed since)

Perturbation arms with the bug (Dv cache drift):
| Arm | Accept | Avg|dH| | AvgCG | Notes |
|-----|--------|---------|-------|-------|
| Perturb/1+Rb/5 | 60% | 0.69 | 2117 | Partially worked |
| Perturb/2+Rb/5 | 0% | 611 | 848 | Broken — Dv drift |
| Perturb/5+Rb/5 | 0% | 110 | 1040 | Broken — Dv drift |

Bug fixed: now saves gauge at last perturbation point and computes
full delta_D = D_current - D_saved.

## Corrected Results (lambda fix + sparse coarse, identical physics)

Config: n_outer=10, n_inner=5, total_steps=50, defl_refresh=0
        Dense coarse initially, sparse+TRLM at rebuild points
        8 OMP threads, 10 trajectories per arm
        Physics IDENTICAL: dH=0.0051, 100% acceptance all arms

| Arm | AvgCG | AvgTime | CG_ratio | Notes |
|-----|-------|---------|----------|-------|
| Stale | 2585 | 13.98s | 1.000 | CG grows 1143→3129 |
| Rebuild/5 | 2159 | 10.86s | 0.835 | CG drops to 1176 at rebuild, grows back |
| LieAlg+Rb/5 | 2289 | 11.55s | 0.885 | RR rotation HURTS — worse than stale t1-4 |

Source: bmc8xvblx (completed, lambda fix + sparse coarse)

Key finding: RR rotation of null vectors degrades MG quality.
Only warm rebuild (fresh inverse iteration) improves CG.
Rotation-based forecasting (Strategies A-E) cannot substitute for
fresh near-null directions.

## FEAST Integration Status

FEAST v2.0 integrated via CMake FetchContent (GitHub certik/feast).
Built with gfortran + macOS Accelerate.

What works:
- FEAST on coarse operator (dim=64-512): identical eigenvalues to TRLM,
  1-iteration warm-start convergence, ~1s
- FEAST for fine null space initial build (L=16, dim=512): identical
  CG counts to inverse iteration
- CLI: --eigensolver feast --feast-emax <float>

What's broken (commits a148acb-ec1fcd1):
- FEAST-MG fine refresh in HMC study: trajectory hangs for arms with
  feast_fine_refresh=true. Bug is in the study arm infrastructure
  (lambda capture or MG copy issue), NOT in FEAST itself.
- gamma5*D FEAST (ec1fcd1): correct eigenvalues on L=16 but not
  tested in HMC study due to above hang.

Next steps:
- Fix study arm hang (likely stale lambda capture in coarse_solve
  after mg_arm = mg copy — same class of bug as the restrict/prolong
  lambda issue fixed in af7013d)
- Test MG-preconditioned FEAST on gamma5*D for fine null space refresh
- Compare FEAST warm-start (seeded from TRLM) vs warm inverse iteration
