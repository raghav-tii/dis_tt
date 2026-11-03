# Independent final validation

Target: `main` at `1f50554`, baseline `d764b45`, 2026-11-03.

Plan before changes:
1. Independently run root `.venv` fast and inclusive suites and verify protected blobs.
2. Audit T7/T8 mathematics, semilinear adjoints, and all six saved experiment records.
3. Fix concrete defects on `review/final-validation` with new regression tests only.
4. Run gates, integrate to main, and record limitations without moving milestones.

Fix plan: reject exhausted Newton solves instead of silently treating them as PDE solutions;
add new regression tests for exhaustion and zero forcing; qualify experiment labels in reports.
Dense semilinear reference helpers are a strict ground-rule violation, but removing them would
break the instruction to preserve public signatures. Record this unresolved API/architecture issue
explicitly rather than disguise it by moving an import or changing the return type.

## Independently executed evidence

Initial `git branch --show-current` was `main`, `git log -5 --oneline` began at `1f50554`,
and `git status --porcelain=v1 --untracked-files=all` was empty before this audit wrote files.
All Python commands used the root `.venv` (Python 3.12.3, numpy 2.5.4, scipy 1.18.1,
pytest 9.2.0, jax/jaxlib 0.12.0.1, optax 0.2.8). No dependencies were installed.

On unchanged main `1f50554`:

```
.venv/bin/python -m pytest -q
167 passed, 3 deselected in 11.39s
.venv/bin/python -m pytest -q -m "slow or not slow"
170 passed in 49.86s
.venv/bin/python -m pytest -q tests/test_toy_problem.py
5 passed in 0.09s
```

The inclusive suite ran the real `test_overfit_20_samples` (not skipped): 20 actual toy samples,
autodiff network Jacobians, depth 6, width 64, 60000 epochs. It independently computes the NumPy
Sobolev loss, compares to the trainer's loss, and requires <1e-3. The test did not print the exact
loss; 4.51e-7 and default-budget 0.2153 remain prior reported measurements, not newly measured
numbers. The changed width, epoch count, and shifted LR schedule are disclosed. No claim is made
that the default Table 2 budget meets the overfit target, nor that a single failed run proves it cannot.

On `review/final-validation`, before the solver change:

```
.venv/bin/python -m pytest -q tests/test_final_validation_extra.py
2 failed, 1 passed in 0.23s
```

Both failures demonstrate Newton returning without raising after budgets 1 and 0 on unit forcing.
The fix adds the loop-exhaustion error; the existing successful convergence criteria are unchanged.
New tests also protect zero forcing and convergence on the last allowed iteration.

After the change:

```
.venv/bin/python -m pytest -q tests/test_final_validation_extra.py tests/test_semilinear.py
9 passed in 0.66s
.venv/bin/python -m pytest -q
170 passed, 3 deselected in 10.98s
.venv/bin/python -m pytest -q -m "slow or not slow"
173 passed in 51.83s
.venv/bin/python notes/validate_saved_results.py
PASS: 13 original test/fixture/protected files byte-identical to d764b45
PASS: all six saved CSV/JSON pairs match results/ byte-for-byte
PASS: saved summary statistics and CSV/JSON fields agree; no experiments rerun
```

The durable checker contains the exact byte comparison and numerical recomputation operations.
It enumerates all ten original test/fixture files from `git ls-tree`, plus protected toy.py,
AGENTS.md, and docs/theory.md; no original assertion or tolerance changed.
`git diff --check` passed. `rg -n 'NotImplementedError|pass$|TODO' src experiments` found only the
T10 raise and stale package-level explanatory text. All required T1–T9 stubs are removed.

## Saved evidence audit

All six result sets are present as CSV and JSON in both `results/` and tracked `notes/T9_results/`.
CSV row counts are 14, 12, 62, 60, 38, 4. SHA256 of the tracked JSONs:

```
exp1 ece774065a67bd92bd4c9c2919c8dc9b8f19577793fcd846812c8cc53780b83d
exp2 dc1bd62049967ac49fe4cee38388fb79f3bb48e7b63ed30ab55b5e739dd6d643
exp3 ab1172183a978d8a4b4db05353b7a2310d1f09e55a8851ba90e88b626575ab4a
exp4 b5560bded1a16862dd8f9b294fa3e123307087e27e95fae4fe2850fb2c5e32a9
exp5 7366f0601ee90b233267b25f0ccca04cc9da7b60a87b5665adb2a42a60fd2e81
exp6 8375a8e1154313046ef4f05f8a23dfe77fea51b6a732a9019fd1baccdf0449f4
```

Headline values agree with report rounding: exp1 toy eigenvalue/subspace errors 2.5515e-9/2.9083e-6;
semilinear 3.7112e-15/1.3087e-14. exp2 maximum identity gap 4.4092e-11, training sandwich 12/12,
held-out <= training upper 10/12, <= upper + plug-in 12/12. exp3 input r8 KL 0.00484389 vs
DIS 0.00142249. exp4 means 0.00271735/0.00268697/0.00454580 (random/KL/exact), 1280 solves each.
exp5 DIS maximum ranks at d9/d10 repeat 14/21/28; PCA eps0.1 changes 11 to 10.
exp6 dense/TT parameter counts 5120/18480 at r10 and 10240/20920 at r20; L2 errors
0.00430873/0.00469821 and 0.00152465/0.00166438. H1 derivative errors span 0.183156–0.337122.
The checker verifies summary-to-CSV consistency with allowances for the CSV's six-significant-digit
serialization. It does not rerun exp1–exp6 or establish the provenance of prior execution beyond the
saved artifacts and matching code. No outputs were rewritten.

## Mathematical and claim audit

- T7: xi=W^T L^-1 x, eta=U^T(y-ybar), J=U^T DF L W. The VJP implementation computes
  (W^T L^T DF^T U)^T, with the required apply_sqrt_T. Both branch sizes, including nonsymmetric
  L, pass the explicit dense reference tests. One batched derivative call per sample is implemented.
- Loss: mean of squared value residual divided by ||eta_s||^2+tau plus squared Jacobian residual
  divided by ||J_s||_F^2+tau. Global uses sums over the dataset, yielding 2/N for zero predictions;
  it is not the same as dividing by mean target energy. JAX reuses full-dataset weights and uses
  jacfwd plus vmap; existing finite-difference and loss checks pass.
- Semilinear residual Jacobian A is symmetric, but DF=A^-1 diag(m) is not. JVP solves A^-1(m v),
  VJP returns m A^-1 w. Half-cell m is on the correct side. Finite differences, batched adjointness,
  large forcing, linear solution, and prior scaling tests all ran. Exhausted Newton solves now fail.
  The existing tiny-step convergence criterion can still terminate on stagnation; callers needing
  a strict residual guarantee should inspect res_rel. Solver/prior argument validation is limited.
- exp2 uses grouped exact columns with group_size=dX and sample-block variance; these are consistent
  with snapshot ordering. Its plug-in sampling term is vacuous here. An expectation theorem is not
  validated by checking one held-out set against that number.
- exp3 does not run exact matched-parameter comparisons. exp4 matches derivative RHS counts, not
  wall time or nonlinear-solve counts. exp5 varies independent random samples with grid depth; the
  finite rank plateau is not an asymptotic result. These limitations do not change saved values.
- exp6 performs dense array encoder/decoder work after expanding TT-derived bases. It is an
  end-to-end RBNO training demonstration, not TT-native execution or a speedup demonstration.
  H1 columns measure derivative error alone, using squared Euclidean/whitened norms, not spatial
  H1 or the full Sobolev loss. Basis reconstruction agreement is numerical, not an exact identity.
- The FD/FV discretization, Euclidean mass, assumed prior Neumann BC, smaller N, and single NN seed
  remain deviations. The paper TeX/source hash is prior provenance; this audit did not independently
  fetch it. No full paper reproduction is claimed.

## Remaining constraints and status

**Functional gates pass; strict repository compliance remains qualified/partial for T9.**
Semilinear H_Y_reference/H_X_white_reference allocate dense Gram operators in src. The rule only
exempts protected toy.py. Their experiment-only labels do not waive the constraint. Exp5 calls
H_Y_reference. Removal or changing the ndarray return contract conflicts with preserving public
signatures, so this audit records the issue rather than hiding it. A follow-up API decision should
move reference construction to experiments while specifying compatibility expectations.

Wide-L support remains explicitly unsupported; optional T10 remains unimplemented. No milestones,
original tests, theory, protected files, or saved experiment outputs were changed. FINAL_REPORT.md
and notes/T9.md now qualify overclaims. The only numerical library change is Newton failure reporting;
three new regression tests were added.

Git initially rejected a branch lock because .git was read-only. The explicitly authorized operation
succeeded via escalated execution; no approval question was sent. The final main gate and commit
provenance are recorded below after integration.

Additional protection check enumerated `git ls-tree -r --name-only 1f50554 tests` and compared each
blob to the worktree: all 22 existing test/fixture files unchanged. Enumerating baseline test paths
with the same command at d764b45 and running `.venv/bin/python -m pytest -q -m "slow or not slow"`
on those nine test files gave **61 passed in 3.19s**.

## Final main integration and gates

`git commit -m "Validate consolidated head and report exhausted Newton solves"` created
`9691447` on `review/final-validation`. `git switch main` followed by
`git merge --ff-only review/final-validation` safely advanced main from `1f50554` to `9691447`.
The review branch is retained. Independently rerun from main at that commit:

```
.venv/bin/python -m pytest -q
170 passed, 3 deselected in 14.97s
.venv/bin/python -m pytest -q -m "slow or not slow"
173 passed in 52.12s
.venv/bin/python notes/validate_saved_results.py
PASS: all protected comparisons and all six saved result checks
```

`git status --porcelain=v1 --untracked-files=all` was empty after integration and these gates.
The explicit baseline protected-path diff and diffs of test_train.py/test_semilinear.py against
1f50554 were empty. A subsequent documentation-only commit records these final gate results in
this note and FINAL_REPORT.md; the tested source/tests remain exactly those in `9691447`.

TASK: final independent validation (T1–T9 audit)
STATUS: partial (functional gates pass; strict T9 dense-reference constraint remains unresolved)
TESTS: .venv/bin/python -m pytest -q -m "slow or not slow" -> 173 passed in 52.12s on main
EXTRA TESTS ADDED: tests/test_final_validation_extra.py (3 Newton failure/convergence regressions)
NUMBERS: 61 original tests pass; 22 pre-existing test/fixture files unchanged; six result sets verified
DEVIATIONS: reference API preserved despite dense-Gram location conflict; paper deviations retained
ISSUES: dense reference methods in src violate AGENTS.md ground rule 5; see constraints above
NEXT: resolve reference-helper API placement; optional T10 remains unimplemented
