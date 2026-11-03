# T6b: bounds and certificates (`src/dis_tt/bounds.py`)

## Plan
1. `projection_error`: B = `tt.subspace_matrix(cores, mode_sizes)`, return ||Z - B B^T Z||_F^2
   from the residual (not ||Z||^2 - ||B^T Z||^2, which cancels).
2. `unprojected_tails`: per k, singular values of the left unfolding `Z.reshape(prod(n_<=k), -1)`
   (snapshot index last), tail = sum_{i>r_k} s_i^2.
3. `bound_report`: train/test projection errors, per-bond unprojected tails at `result.ranks`,
   upper = sum, lower = max, sum_tails = sum(result.tails).
4. `bootstrap_tails`: columns split into T/group_size contiguous groups (one x per group, matching
   the per-sample column order of `snapshots.py`); draw the same number of groups with
   replacement, scale sqrt(G / n_drawn) (= 1 here), rerun `dis_tt_dense(ranks=...)`, return the
   sample std of per-bond tails.
5. Extras in `tests/test_bounds_extra.py`; report `bound_report` on the toy at RANKS.

## Design choices
- `projection_error` uses the residual Z - B(B^T Z) (no cancellation); B = `tt.subspace_matrix`
  (dense, small problems only, as the module states). Works for D = d and D = d-1.
- `unprojected_tails` uses singular values of the left unfolding (s_i^2 = eigenvalues of H^(k);
  missing eigenvalues are exact zeros). Agrees with `eigvalsh` reference to rtol 1e-8.
- `bound_report.meta`: total_train, total_test, train_rel, test_rel, cert, ranks, tails.
- Extra kwargs on `bootstrap_tails` (defaults keep the original behaviour):
  `n_groups=None` (number of groups drawn, default G) and `backend="auto"` (passed to Algorithm A).

## Bootstrap scaling (required statement)
Columns are split into G = T / group_size CONTIGUOUS groups. With group_size = S (directions per
x) each group is exactly one sample x, because `snapshots.py` stores all directions of x_1, then
x_2, ... Z already carries the weight 1/sqrt(N S) (or 1/sqrt(N)) per column, so
Z Z^T = sum_g Z_g Z_g^T. Drawing n groups i.i.d. uniformly with replacement and multiplying by
sqrt(G / n) gives E*[Z* Z*^T] = (G/n) * n * (1/G) sum_g Z_g Z_g^T = Z Z^T: the resample is an
unbiased estimate of H_hat with the same scaling as the original. The default n = G gives
factor 1, i.e. no rescale. group_size must divide T (else ValueError). Std uses ddof = 1 (0 if
n_boot = 1). Tested directly by capturing the resamples (`test_bootstrap_preserves_scaling`).

## Numbers: toy fixture (Z_exact / Z_exact_test, 128 x 2560), ranks = (2, 3, 4, 4, 4, 3, 3)
greedy tails      (0, 2.4945, 0.59323, 0.35797, 0.088733, 0.10321, 9.9374e-05)
unprojected tails (0, 2.4945, 1.9739, 1.9540, 0.81373, 2.1291, 2.0690)
lower_unprojected = 2.494522  <=  train_error = 3.637771 (= sum_tails, rel gap 2.7e-14)
                              <=  upper_unprojected = 11.43422
test_error = 4.134923 (held-out, 40 fresh samples); totals: train 178.7203, test 177.4296;
relative: train 2.036e-2, test 2.331e-2 (test/train = 1.14).
bootstrap std (n_boot = 50, rng 0):
  group_size = 64 (one x per group): (0, 0.6555, 0.1464, 0.1706, 0.04222, 0.04871, 4.46e-5), sum 1.064
  group_size = 1 (column-level):     (0, 0.1522, 0.03642, 0.03770, 0.009721, 0.01073, 1.16e-5), sum 0.247
Observation: column-level resampling understates the spread ~4x because the 64 directions of one x
are strongly correlated; group_size = S is the right choice. Here
train + sum(bootstrap std) = 4.70 >= test 4.13 (informal; not the Prop. 2 bound, whose M_k is unknown).

## Deviations
None (two extra keyword arguments with defaults, see above).

## Result
`pytest tests/test_bounds.py` -> 4 passed; `tests/test_bounds_extra.py` -> 7 passed.

## Independent audit, 2026-11-03
Verified grouped resampling scaling and added positive group/draw/replicate validation. Fresh toy and bootstrap numbers match prior findings.
Full evidence, deviations and exact results: `notes/CODEX_REVIEW.md`.

## Report (AGENTS.md format)
TASK: T6b
STATUS: done
TESTS: .venv/bin/python -m pytest -q tests/test_bounds.py -> 4 passed
EXTRA TESTS ADDED: tests/test_bounds_extra.py (7)
NUMBERS: ranks (2,3,4,4,4,3,3): lower 2.494522 <= train 3.637771 (= sum tails) <= upper 11.43422; test 4.134923;
         bootstrap std sum 1.064 (group 64) vs 0.247 (group 1)
DEVIATIONS: extra kwargs n_groups, backend (defaults keep behaviour)
ISSUES: none
NEXT: done
