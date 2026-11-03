# AGENTS.md: implementation tasks for DIS-TT

This repository is scaffolding. Every algorithm raises `NotImplementedError("Tn")`, where Tn
is the task below. The math lives in `docs/theory.md`; the full note is at
https://claude.ai/artifact/Jjr4SEdweverCdqwKStCyn. The acceptance tests in `tests/` are the
spec. They have been checked to be satisfiable by a reference implementation that is NOT in
the repo.

## Ground rules (all tasks)

1. **Do not edit** `tests/conftest.py`, `tests/test_toy_problem.py`, `src/dis_tt/problems/toy.py`,
   or any existing assertion or tolerance. If you believe a test is wrong, stop. Add an
   `ISSUE` entry to `notes/<task>.md` with a minimal reproduction and the mathematical argument, and
   report it. Do not work around it.
2. You MAY add tests (new files `tests/test_<task>_extra.py`). Add them for edge cases you
   handle: rank caps, D = d-1, T = 1, n_k != 2, empty tails.
3. Keep every public signature and dataclass field. Extra keyword arguments with defaults are fine.
4. Runtime dependencies are numpy and scipy only (T8 may add jax or torch as an optional extra).
5. Respect the stated complexity. Never densify in `native.py`, and never form the
   d_Y x d_Y operator H anywhere in library code (`problems/toy.py` references are test-only).
6. Conventions are binding (below). Most likely bug: the Kronecker or reshape order.
7. Log every deviation, design choice and surprise in `notes/<task>.md` (e.g. `notes/T4.md`).
8. Finish with the report format at the bottom. "Tests pass" alone is not a report.

## Conventions

- Dense vector <-> tensor: `y.reshape(mode_sizes)` in **C order**. Mode 1 is the slowest
  index (top QTT bit).
- TT core k has shape `(r_{k-1}, n_k, r_k)`. Left-orthogonal means
  `G.reshape(-1, r_k)` has orthonormal columns.
- The left frame `U_{<=k}` has rows `(i_1, ..., i_k)` in C order; see
  `tests/conftest.py::ref_left_frame`. That function is the reference.
- Snapshot matrices are pre-scaled: `Z @ Z.T` is the operator estimate. Scaling happens only
  in `snapshots.py`.
- The snapshot index is the **last** tensor mode: tensor shape `(n_1, ..., n_d, T)`.
- Whitening factor L with C_X = L L^T. Whitened Gram is `L^T DF^T DF L` (note L^T on the
  adjoint side).
- `DISTTResult.tails[k]` is the discarded eigenvalue mass at bond k of the PROJECTED operator
  H~_k. Then `sum(tails) == ||(I - Pi_D) Z||_F^2` exactly.

## Task graph

```
T1 tt.py ─────────┐
T2 linalg.py ──── T4 dense.py ──┬── T5 native.py      (needs T1)
T3 snapshots.py ──┘             ├── T6 input_side.py  (needs T1, T3)
                                └── T6b bounds.py     (needs T1)
                                       T6 ── T7 rbno.py data ── T8 NN training
all of the above ── T9 experiments + PDE problems ── T10 ALS refinement (optional)
```

T1, T2 and T3 can run in parallel. T5, T6 and T6b can run in parallel after T4.

| Task | Files you own | Acceptance | Notes |
| --- | --- | --- | --- |
| T1 | `src/dis_tt/tt.py` | `pytest tests/test_tt.py` | numpy only; `from_dense` is plain TT-SVD |
| T2 | `src/dis_tt/linalg.py` | `pytest tests/test_linalg.py` | randomized backend: tails via trace minus top sum |
| T3 | `src/dis_tt/snapshots.py` | `pytest tests/test_snapshots.py -m "slow or not slow"` | batch JVPs/VJPs per x |
| T4 | `src/dis_tt/dense.py` | `pytest tests/test_dense.py` | reshape trap in module docstring |
| T5 | `src/dis_tt/native.py` | `pytest tests/test_native.py` | import `from . import tt`; no densify |
| T6 | `src/dis_tt/input_side.py` | `pytest tests/test_input_side.py` | wide-L case: add a test before supporting it |
| T6b | `src/dis_tt/bounds.py` | `pytest tests/test_bounds.py` | dense reference tools |
| T7 | `src/dis_tt/rbno.py` (data, loss) | `pytest tests/test_rbno.py` | both branches must agree |
| T8 | `rbno.train_latent_network` | your own tests | paper Table 2 hyperparameters |
| T9 | `experiments/`, `src/dis_tt/problems/<pde>.py` | experiment outputs in `results/` | report numbers; don't tune theory away |
| T10 | `src/dis_tt/refine.py` | your own tests: monotone, <= greedy | multi-target ALS |

Full suite: `pytest -q -m "slow or not slow"`.

## Prompts for implementing agents

Copy one block per agent. Each block is self-contained.

### T1: TT toolkit
> You are implementing `src/dis_tt/tt.py` in the dis_tt repo. Read `AGENTS.md`
> (Ground rules, Conventions), then the docstrings in `tt.py`, then `tests/test_tt.py` and
> `tests/conftest.py::ref_left_frame`. Implement every stub in `tt.py` with numpy only.
> Done when `pytest tests/test_tt.py` passes and you have added
> `tests/test_tt_extra.py` covering n_k != 2, d = 1 and rank caps. Do not modify other files
> except `notes/<task>.md`. Report using the format at the end of AGENTS.md.

### T2: bond linear algebra
> Implement `select_rank` and `top_eigh_gram` in `src/dis_tt/linalg.py`. Read `AGENTS.md`,
> the docstrings and `tests/test_linalg.py`. The randomized backend must return
> `complete=False`, and callers compute tails as `trace - sum(top eigenvalues)`; make
> `select_rank(..., trace=...)` support that. Done when `pytest tests/test_linalg.py` passes.
> Add extra tests for all-zero input and k > m. Report per AGENTS.md.

### T3: snapshot estimators
> Implement `src/dis_tt/snapshots.py` (note Sec. 5; `docs/theory.md` table). Scaling: random
> directions 1/sqrt(N S); KL and exact columns 1/sqrt(N); PCA 1/sqrt(N) with the biased mean.
> The input side uses `prior.apply_sqrt_T` (L^T), not `apply_sqrt`. One `model.solve` per
> sample, then batched JVP/VJP calls. Done when
> `pytest tests/test_snapshots.py -m "slow or not slow"` passes. Report per AGENTS.md.

### T4: Algorithm A (dense DIS-TT)
> Implement `dis_tt_dense` in `src/dis_tt/dense.py` exactly as the pseudocode in its
> docstring, using `linalg.select_rank` and `linalg.top_eigh_gram`. Fill
> `spectra_complete`. With `precompress=True`, nothing observable may change. Read the
> RESHAPE TRAP note. Done when `pytest tests/test_dense.py` passes. Then add
> `tests/test_dense_extra.py` with: n_k != 2 modes, D = d-1 with eps, rmax cap binding, and a
> timing check that cost scales linearly in T (two sizes, ratio < 3 for 2x T). Report per
> AGENTS.md, including the per-bond ranks and tails for `eps=0.05` on the toy fixture.

### T5: Algorithm B (TT-native DIS-TT)
> Implement `dis_tt_native` in `src/dis_tt/native.py` following its docstring: right-orthogonalise
> each snapshot once, keep per-snapshot left environments, stream the bond Gram. Never call
> `tt.to_dense` or `tt.block_to_dense` (a test monkeypatches them). Done when
> `pytest tests/test_native.py` passes. Add a test with mixed snapshot TT ranks. Report the
> measured runtime vs T (3 sizes) and vs d (QTT depth 6, 8, 10) per AGENTS.md.

### T6: input side
> Implement `WhitenedInputBasis` and `input_dis_tt` in `src/dis_tt/input_side.py`.
> Use `tt.block_to_dense` for `W()` (dense is fine here). Encoder `W^T L^{-1} x`, decoder
> `L W xi`. Done when `pytest tests/test_input_side.py` passes. Then investigate the OPEN
> ISSUE in the docstring (wide L): write a small prior with a wide factor, test whether
> `Q_TT` is a projector, and record the outcome in notes/<task>.md. Don't change the
> square-L code path. Report per AGENTS.md.

### T6b: bounds and certificates
> Implement `src/dis_tt/bounds.py`. Done when `pytest tests/test_bounds.py` passes. The bootstrap
> resamples groups of `group_size` columns and preserves the estimator scaling (state how in
> notes/<task>.md). Report per AGENTS.md, including `bound_report` numbers on the toy fixture at
> `ranks=(2,3,4,4,4,3,3)`.

### T7: RBNO data assembly
> Implement `assemble_rbno_data` and `sobolev_loss` in `src/dis_tt/rbno.py` (paper eqs. 9-10;
> per-sample normalisation as in paper Sec. 4.3). The VJP branch uses L^T. Done when
> `pytest tests/test_rbno.py` passes. Report per AGENTS.md.

### T8: latent network training
> Implement `train_latent_network` with JAX (preferred) or PyTorch as an optional dependency.
> Use paper Table 2 settings (softplus, width 2r, depth 4-6, Adam, batch 25, LR halving
> schedule). The loss is `sobolev_loss` with the network Jacobian from autodiff. Write
> `tests/test_train.py`: overfit 20 samples of the toy RBNO data to loss < 1e-3. Report per
> AGENTS.md.

### T9: experiments and PDE problem
> First implement `src/dis_tt/problems/semilinear.py`: the paper's 1D semilinear elliptic
> problem (Sec. 4.1) by finite differences on 2^d interior nodes. Use a Newton solver with
> exact JVP (linearised solve) and VJP (adjoint solve), verified by finite differences and
> adjointness tests in the style of `tests/test_toy_problem.py`. Then implement
> `experiments/run_checklist.py` exp1-exp6 in order. Report numbers. Where a result
> contradicts `docs/theory.md`, that is a finding: report it with the evidence and do NOT
> change the theory or the library to hide it.

### T10: ALS refinement (optional)
> Implement `als_refine` in `src/dis_tt/refine.py` (docstring). Tests: training error
> non-increasing per microstep, final error <= greedy error, left-orthogonality kept.
> Report per AGENTS.md.

## Report format (every task)

```
TASK: Tn
STATUS: done | blocked | partial
TESTS: <exact pytest command> -> <N passed, M failed>   (paste the summary line)
EXTRA TESTS ADDED: <files, one line each>
NUMBERS: <task-specific numbers requested above>
DEVIATIONS: <none | list, each with reason>   (also in notes/<task>.md)
ISSUES: <none | suspected test/theory problems with minimal repro>
NEXT: <one line>
```
