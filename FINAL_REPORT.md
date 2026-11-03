# DIS-TT implementation: final report (2026-11-03)

Integration documentation; it is outside the per-task file ownership in AGENTS.md. Per-task details,
design choices and AGENTS-format reports are in `notes/T1.md` ... `notes/T10.md`. The independent audit
is in `notes/CODEX_REVIEW.md`.

## 1. Status

| Task | Module | Status | Acceptance (original test file) |
| --- | --- | --- | --- |
| T1 | `src/dis_tt/tt.py` | done | `tests/test_tt.py` 7 passed |
| T2 | `src/dis_tt/linalg.py` | done | `tests/test_linalg.py` 14 passed |
| T3 | `src/dis_tt/snapshots.py` | done | `tests/test_snapshots.py -m "slow or not slow"` 7 passed |
| T4 | `src/dis_tt/dense.py` | done | `tests/test_dense.py` 13 passed |
| T5 | `src/dis_tt/native.py` | done | `tests/test_native.py` 6 passed |
| T6 | `src/dis_tt/input_side.py` | done; wide L unsupported | `tests/test_input_side.py` 3 passed |
| T6b | `src/dis_tt/bounds.py` | done | `tests/test_bounds.py` 4 passed |
| T7 | `src/dis_tt/rbno.py` (data, loss) | done | `tests/test_rbno.py` 2 passed |
| T8 | `rbno.train_latent_network` (JAX, optional) | done; overfit test deviates from Table 2 (see 5) | own `tests/test_train.py` 3 passed |
| T9 | `problems/semilinear.py`, `experiments/run_checklist.py` | done; FD approximation of the paper setup | own `tests/test_semilinear.py` 6 passed; exp1-exp6 executed |
| T10 | `src/dis_tt/refine.py` | optional, NOT implemented (stub remains) | none |

Ground truth `tests/test_toy_problem.py`: 5 passed. All 61 original tests (baseline `d764b45`) pass.

Remaining stubs: `refine.als_refine` raises `NotImplementedError("T10")`. No other stubs remain.

## 2. Test evidence (final `main`, run on 2026-11-03)
Environment: repository-local `.venv` (Python 3.12.3, numpy 2.5.4, scipy 1.18.1, pytest 9.2.0,
jax 0.12.0.1 + jaxlib 0.12.0.1, optax 0.2.8; CPU only).

```
.venv/bin/python -m pytest -q                          -> 167 passed, 3 deselected in 10.98s
.venv/bin/python -m pytest -q -m "slow or not slow"    -> 170 passed in 48.84s
```
Per file (`-m "slow or not slow"`): bounds 4, bounds_extra 7, codex_resume_extra 5, codex_review_extra 16,
dense 13, dense_extra 13, input_side 3, input_side_extra 8, linalg 14, linalg_extra 17, native 6,
native_extra 12, rbno 2, rbno_extra 4, semilinear 6, snapshots 7, snapshots_extra 8, toy_problem 5,
train 3 (overfit 44 s), tt 7, tt_extra 10.
Protected files: `git diff d764b45 -- tests/conftest.py src/dis_tt/problems/toy.py <all baseline test files>`
is empty. No original test or tolerance was edited.
`tests/test_train.py` is skipped automatically if jax/optax are missing.

## 3. Architecture and usage
```
snapshots.py  pre-scaled Z (Z Z^T = H_hat): output DIS (random/kl/exact), PCA, input DIS (L^T side), mixed
dense.py      Algorithm A: greedy left-to-right DIS-TT of dense Z (eps / rmax / ranks, D = d or d-1, backends)
native.py     Algorithm B: same result from TT snapshots, never densifies
input_side.py Algorithm C: whitened input basis W, encoder W^T L^{-1} x, decoder L W xi (square L only)
bounds.py     projection error, unprojected tails, Prop. 1 report, grouped bootstrap
rbno.py       Algorithm D: (xi, eta, J) data via min(r_X, r_Y) JVP/VJP solves; Sobolev loss; JAX training
problems/     toy.py (exact fixture), semilinear.py (paper Sec. 4.1 by FD)
```
Example:
```python
from dis_tt import snapshots as sn
from dis_tt.dense import dis_tt_dense
from dis_tt.problems.semilinear import SemilinearProblem
p = SemilinearProblem(levels=8)
xs = p.prior.sample(np.random.default_rng(0), 40)
Z = sn.output_dis_snapshots(p, p.prior, xs, direction_kind="exact").Z
res = dis_tt_dense(Z, p.out_modes, eps=0.1, precompress=True)   # res.ranks, res.tails, res.cert
```
Experiments: `PYTHONPATH=src .venv/bin/python experiments/run_checklist.py exp1 exp2 exp3 exp4 exp5 exp6`
(writes `results/*.csv|json`, git-ignored; tracked copies in `notes/T9_results/`). Runtimes on this machine:
exp1 0.2 s, exp2 89 s, exp3 133 s, exp4 25 s, exp5 3.7 s, exp6 410 s.

## 4. Numerical results (details: `notes/T4.md`, `T5.md`, `T6.md`, `T6b.md`, `T8.md`, `T9.md`)
- Toy, eps = 0.05: ranks (2,4,8,9,7,6,5); sum(tails) 0.1456518 equals ||(I - Pi) Z||^2 to 4.6e-13 rel; cert 8.15e-4.
- Toy bound report, ranks (2,3,4,4,4,3,3): lower 2.4945 <= train 3.6378 (= sum tails) <= upper 11.434; test 4.1349.
- Native timing linear in T (x1.87, x2.06 per doubling) and ~linear in d; dense Algorithm A is faster at these sizes.
- exp1: full-rank DIS-TT reproduces dense DIS: toy eig err 2.6e-9, sin <= 2.9e-6; semilinear 3.7e-15 / 1.3e-14.
- exp2: training identity exact (4.4e-11). Held-out error > training upper bound for N = 10 at eps 0.03 and 0.01.
  Plug-in Prop. 2 sampling term (2.4e3 to 2.7e4) is vacuous against errors <= 220.
- exp3: input DIS beats input KL/PCA (r = 8: 1.42e-3 vs 4.84e-3; r = 32: 4.8e-6 vs 3.4e-5). Output PCA-TT and
  DIS-TT are near parity in L2 at equal parameters; DIS is better in H1 at low budgets.
- exp4: budget 1280 linear solves: random 2.72e-3 +- 2.9e-4, KL 2.69e-3 +- 3.6e-4, exact (N = 5) 4.55e-3 +- 9.4e-4.
- exp5: max ranks saturate with depth (d = 9 -> 10 unchanged). DIS-TT ranks exceed PCA-TT ranks (eps 0.1:
  14 vs 10 at d = 10). Per-bond DIS spectra decay more slowly (caveat 4 confirmed).
- exp6: RBNO at r = 10 / 20 under the Table 2 protocol: L2 4.3e-3 / 1.5e-3; H1 0.18-0.34 (network-limited,
  reconstruction 2e-3 / 5e-5). TT bases at d = 8 have more parameters than dense bases (18 480 vs 5 120 at r = 10).
- T8: overfit 20 toy samples: loss 4.51e-7 (< 1e-3) with width 16r and 60 000 epochs. Exact Table 2 budget: 0.215.

## 5. Deviations and interpretations
- T3: extra `batch` kwarg; T6b: extra `n_groups`, `backend` kwargs (defaults preserve behaviour).
- T8: LR-halving spacing (paper: "halved five times starting at epoch 2250"; spacing unstated) taken as
  even, at 2250, 2400, ..., 2850. Width = 2 max(r_X, r_Y). Initialisation N(0, 1/fan_in). The overfit test
  uses width 16r and 60 000 epochs because the paper configuration cannot reach 1e-3 here.
- T9: FD/finite volumes and Euclidean inner products instead of FEniCS FE with L^2 mass. The prior's
  Laplacian boundary conditions are not given in the paper; Neumann was assumed, with nodal covariance
  A_h^{-1}/h. Samples are much smaller than the paper's, with one NN seed. These are NOT faithful
  reproductions of the paper's figures.
- Codex audit (merged `b493763`):
  - T1: scale-safe truncation norms.
  - T2: randomized null-complement fallback.
  - T4: precompress keeps all singular values and is skipped for the randomized backend.
  - T3: input validation.
  - T5: no stored bond blocks.
  - T6b: positive bootstrap arguments.
  - 21 new regression tests. Details in `notes/CODEX_REVIEW.md`.

## 6. Limitations and findings (not tuned away)
1. **Wide whitening factor L (T6): unsupported.** After TT truncation, W leaks into ker(L), so
   Q_TT = L W W^T L^+ is not a projector (e.g. ||Q^2 - Q|| = 0.15 at eps 0.1) and V is not E-orthonormal.
   The API does not detect this. Projecting onto range(L^T) restores the projector but loses the TT form.
2. Prop. 2 sampling term is vacuous at these N (exp2); bootstrap spreads are diagnostics only.
3. DIS per-bond spectra decay more slowly than PCA's, so DIS-TT ranks are larger (exp5, caveat 4).
4. At d = 8 the TT bases do not compress relative to dense bases (exp6).
5. Algorithm B is a Python loop over snapshots; it is slower than A whenever densifying is possible.
6. Randomized-backend tails are the actual residual mass of the sketched projector, not optimal tails;
   finite-precision limits are listed in `CODEX_REVIEW.md`.
7. The TT-specific propositions (Props. 1-2, input-side extension) are unpublished derivations from the note;
   the experiments test them, they are not established results.

## 7. Provenance
- Paper: Luo, O'Leary-Roseberry, Chen, Ghattas, arXiv:2504.08730v1. TeX source from
  https://arxiv.org/src/2504.08730, sha256 6e061170a00210a0b5cfb056afd07073e14cf5b4e0c7a2ef93dff5930d8b455f
  (not stored in the repo). Settings were read from `sections/numerical_results.tex` (PDE, prior, Table 2,
  training protocol) and `sections/mathematical_setup.tex` (network definition, Sobolev loss and weights).
- The full note linked in AGENTS.md (claude.ai artifact) was NOT accessed; `docs/theory.md` was used instead.

## 8. Git
`main` history (first parent): d764b45 baseline -> 76903a6 T1 -> 7751a04 T2 -> e8c1278 T3 -> 3f6d006 T4
-> 91fac0a T5 -> bf30f0d T6 -> e700890 T6b -> b493763 Codex audit -> 971f3a0 T7 -> acd7a04 T8 -> e2220f2 T9
-> integration/final-report merge (this report, README, notes). Task branches `task/T1..T9`,
`review/codex-core` and its worktree `.worktrees/codex-core` are retained. `.venv/`, `observations/`,
`skills/`, `.worktrees/` are excluded via `.git/info/exclude`; `results/` via `.gitignore`.

## 9. Resume
```
cd <repo root>; .venv/bin/python -m pytest -q -m "slow or not slow"      # expect 170 passed
uv pip install --python .venv/bin/python numpy scipy pytest jax optax     # if recreating the venv
PYTHONPATH=src .venv/bin/python experiments/run_checklist.py exp1          # any of exp1..exp6
```
Open work: T10 (design decision in `notes/T10.md`); wide-L support (`notes/T6.md`); larger-N / FE-mass
experiments for paper fidelity; batching in Algorithm B.
