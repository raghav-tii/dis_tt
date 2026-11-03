# TT-SVD vs DIS-TT on functions and their derivatives (2026-11-03)

Companion to `FINAL_REPORT.md`. Script: `experiments/plot_ttsvd_vs_dis.py` (about 40 s on CPU).
Figures and raw numbers: `docs/figures/*.png`, `docs/figures/ttsvd_vs_dis.json`.

## Setup

Each case is a parametric family u(x; m) with m ~ N(0, I_p), sampled on a QTT grid. Both methods
run the same greedy left-to-right sweep (`dis_tt_dense`, D = d, the same rank cap `rmax` on every
bond). They differ only in their snapshots:

| Method | Snapshots Z (Z Z^T is the target operator) | What it is built to capture |
| --- | --- | --- |
| TT-SVD (PCA-TT) | (u_i - u_bar) / sqrt(N) | function values (output covariance C_Y) |
| DIS-TT | du/dm_j (m_i) / sqrt(N), all j | derivatives (H_Y = E[J J^T]) |

Error metric: held-out (200 test samples, 60 for the PDE) relative squared projection error
||(I - UU^T) A||_F^2 / ||A||_F^2. A holds the centred test values ("values") or the stacked test
Jacobians ("derivatives"). The analytic Jacobians come from the complex step and are exact to
round-off. The PDE Jacobian is the whitened D_E F = DF L, from the repo's `SemilinearProblem` JVPs.
Training uses N = 200 samples (100 for the PDE).

| Case | Formula | p | Grid |
| --- | --- | --- | --- |
| bump | a exp(-(x-c)^2 / 2w^2), c = 0.5 + 0.08 m1, w = 0.04 e^{0.25 m2}, a = 1 + 0.2 m3 | 3 | 2^10 |
| chirp | (1 + 0.3 m3 x) sin(2π(6 + 0.8 m1) x + 3 m2 x^2) | 3 | 2^10 |
| tanh-field | tanh(3 Σ_j m_j sqrt(2) sin(jπx) / j) | 24 | 2^10 |
| semilinear-PDE | paper Sec. 4.1 (FD, `problems/semilinear.py`) | 256 | 2^8 |

## Figures

Each per-case figure has four panels: (a) realisations, (b) one test value and its projection
onto both r-column TT bases, (c) one test derivative column and its projections, (d) held-out
error against basis size. Legend errors in (b) and (c) are relative L2 errors, not squared.

- `docs/figures/bump.png`
- `docs/figures/chirp.png`
- `docs/figures/tanh-field.png`
- `docs/figures/semilinear.png`
- `docs/figures/summary_rmax8.png`: both errors for every case at rmax = 8.

## Results at matched basis size (identical TT parameter counts)

| Case | r | Values: TT-SVD | Values: DIS-TT | Derivatives: TT-SVD | Derivatives: DIS-TT |
| --- | --- | --- | --- | --- | --- |
| bump | 4 | **1.25e-1** | 2.38e-1 | 3.04e-1 | **2.27e-1** |
| bump | 8 | **1.32e-2** | 4.13e-2 | 6.86e-2 | **5.38e-2** |
| bump | 16 | **2.46e-4** | 6.21e-4 | **3.08e-3** | 3.54e-3 |
| chirp | 4 | **2.64e-1** | 3.88e-1 | 2.79e-1 | **1.74e-1** |
| chirp | 8 | **3.61e-2** | 6.62e-2 | 4.47e-2 | **2.76e-2** |
| chirp | 24 | 1.92e-7 | **4.69e-8** | 3.14e-7 | **4.70e-8** |
| tanh-field | 8 | **1.32e-1** | 1.85e-1 | 3.67e-1 | **3.30e-1** |
| tanh-field | 32 | **4.94e-3** | 6.96e-3 | 4.61e-2 | **3.93e-2** |
| semilinear-PDE | 8 | **4.09e-3** | 6.05e-3 | 2.71e-2 | **2.08e-2** |
| semilinear-PDE | 32 | 1.22e-5 | **9.58e-6** | 2.15e-5 | **9.43e-6** |

Full sweep (r = 1 to 32, with per-bond ranks and parameter counts): `docs/figures/ttsvd_vs_dis.json`.

## Findings

1. **Each method wins the quantity it targets at moderate r.** At r = 8, DIS-TT has 10-38 % lower
   derivative error than TT-SVD on all four cases. TT-SVD has 1.4-3x lower value error. At r = 4
   the PDE is an exception: TT-SVD is better on both (derivatives 0.117 vs 0.158).
   Figure panels (b) and (c) show the same split on single test samples.
2. **The derivative advantage is modest, not dramatic.** Under the same rank cap it is at most
   about 1.6x (chirp r = 4 to 8). On the bump it reverses at r >= 12: TT-SVD is better on both
   quantities there (r = 16: 3.08e-3 vs 3.54e-3).
3. **Chirp at high rank: DIS-TT wins on both.** At r = 24, DIS-TT is 4x better on values and 7x
   better on derivatives. At the largest cap it reaches 1e-13, against about 1e-8 for TT-SVD.
   The Jacobian columns contain x·cos(·) and x^2·cos(·) terms, so they span the value manifold's
   tangent directions, and the value snapshots resolve those directions less well. TT-SVD's
   last rank also saturates at r = 28 against DIS-TT's 30.
4. **PDE: DIS-TT is never worse on derivatives for r >= 8, and is about equal on values for r >= 12.**
   This matches the paper's output-side "parity" prediction and repo exp3.
5. **The tanh-field has slow spectral decay for both methods.** With 24 active parameters
   and saturation, neither basis reaches 1e-2 derivative error by r = 32. TT rank, not the choice
   of snapshots, is the limit here.
6. **Single-function TT-SVD is a different question.** Compressing one function u(x) with
   `tt.from_dense` has no derivative information to use, so DIS-TT only applies to parametric
   families. That is what these figures compare.

## Caveats

- One seed, one training set per case. The differences at r <= 8 are consistent across all four
  cases, but no error bars were computed.
- Bases are compared at an equal rank cap (`rmax`). Parameter counts are identical except at
  high caps where a bond rank saturates (at most 7 %, chirp r = 32; see the JSON). Comparisons at matched eps would give different ranks; see repo exp3/exp5.
- The analytic families use a whitened prior (L = I), so DIS-TT reduces to the uncentred TT-SVD
  of the Jacobian tensor (n_1, ..., n_d, N·p).
