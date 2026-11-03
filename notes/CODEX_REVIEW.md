# Independent review of T1–T6b

## Plan (before edits)
1. Read theory, implementations, acceptance/extra tests and task notes; run ground-truth and full baseline in the isolated worktree.
2. Add adversarial tests for randomized rank completion, precompression invariance, numerical TT scaling, estimator validation and bootstrap scaling.
3. Fix demonstrated implementation defects only; preserve existing tests and square-factor mathematics.
4. Re-run targeted and full suites, measure complexity and toy certificates, record remaining limitations and commit for integration.

Baseline: main e700890. Shared absolute venv; worktree .worktrees/codex-core, branch review/codex-core.
Ground truth: python -m pytest -q tests/test_toy_problem.py -> 5 passed in 0.09s.
Full baseline: python -m pytest -q -m 'slow or not slow' -> 2 failed, 134 passed in 6.98s.
Both failures are existing T7 NotImplementedError in test_rbno.py, outside this review.

Wide-L policy: existing investigation tests intentionally invoke unsafe truncated wide-L bases and measure non-idempotence. They are mathematically sound. A default fail-fast guard conflicts with keeping those tests passing and unmodified. Propose rejecting rectangular factors at input_dis_tt and WhitenedInputBasis construction (physical dimension can be obtained from apply_sqrt(zeros(white_dim))); migrate investigation to explicitly unchecked algebra in a separately authorized change. No test-specific bypass or silent projection is acceptable. This review leaves wide-L behavior unchanged and reports it as an unresolved safety limitation.

## Resumed independent audit — 2026-11-03 (authoritative results)

The previous attempt ended with HTTP401; its report was not a successful audit. The text above
and six implementation diffs plus `tests/test_codex_review_extra.py` were already present as
uncommitted work. They were preserved, read and independently tested in this run. No earlier
reported test result is used as evidence below.

### Starting state and isolation
- Shared main: e700890, `git status --short --branch` showed only untracked `.worktrees/`.
- Git history and task notes show T1–T6b merged; T7 remains a stub, not an assumed running task.
- Existing branch review/codex-core also pointed at e700890. Its registered worktree was the
  obsolete `/home/raghavendra/InnoviumAI/Projects/dis_tt/.worktrees/codex-core` (prunable).
  Actual files were at `.worktrees/codex-core`; `git worktree repair .worktrees/codex-core`
  repaired the registration without replacing files or resetting anything.
- Starting modifications: bounds.py, dense.py, linalg.py, native.py, snapshots.py, tt.py;
  untracked CODEX_REVIEW.md and test_codex_review_extra.py. These changes were retained after
  verification. Fresh tests are in test_codex_resume_extra.py (five cases).
- Read AGENTS.md, agents/skills/dis-tt-implementer/SKILL.md, docs/theory.md, task notes,
  implementations, acceptance tests, conftest and factor investigations. Used the existing
  root .venv, not a new environment. No delegation or merge performed.

### Verified findings and improvements
1. **T1 scaling defect fixed.** On diag(1,.1,.01,.001) at eps=.02, multiplying the vector by
   1e-200 incorrectly kept rank four; 1e200 overflowed the norm. Scale-normalized norms and
   singular-value suffix sums now yield rank two for both from_dense and tt_round. Rank caps
   can still prevent the requested tolerance; eps=0 retains the documented numerical cutoff.
2. **T2 randomized fixed-rank defect fixed.** One snapshot with modes (2,4) and requested ranks
   (2,5) returned (2,1). A range sketch cannot produce the required null complement. The SVD
   fallback now supplies it while keeping spectra_complete=False. This fallback has exact-SVD
   cost, not the sketch-only cost; it is needed only when k exceeds min(m,c).
3. **T4 precompression defects fixed.** Numerical-rank pruning discarded a representable
   1e-30 tail from diag(1,1e-15), violating fixed-rank tail accounting. Retain all thin-SVD
   singular values. Changing column coordinates also changes a seeded randomized sketch:
   baseline projector entries differed by as much as .40494. Skip precompression for that
   backend to preserve its output. This trades acceleration for invariance: T_used is now
   min(T,d_Y), rather than numerical rank, for nonzero dense exact inputs. Exact degenerate
   eigenspaces can still rotate; literal elementwise core invariance is not mathematically
   meaningful. Metadata intentionally records the requested option and actual T_used.
4. **T2/T4 randomized tails verified with a poor sketch.** Each tail agrees with the squared
   loss between successive nested projections, including D=d-1. Trace minus retained Ritz
   energy gives the actual residual mass of the approximate projector. It is not necessarily
   the optimal discarded eigenvalue sum. Thus the exact-greedy upper sandwich should not be
   advertised for arbitrary poor sketches. Floating-point subtraction loses relative accuracy
   for tails near machine epsilon times total; the SVD suffix-sum backend is preferable there.
5. **T3 scaling and orientation verified.** Exact/KL use 1/sqrt(N), random directions use
   1/sqrt(NS), PCA uses the biased mean and 1/sqrt(N). Existing batch-count tests pass.
   Fresh explicit 3x2 linear Jacobian / 2x4 nonsymmetric factor checks at N=1 and N=7 reproduce
   both J L L^T J^T and L^T J^T J L. Empty sample sets and zero random directions now fail
   clearly before solve calls instead of dividing by zero or failing indirectly.
6. **T5 memory improvement verified.** Removed the list of T bond blocks (T*r*n*R storage);
   recompute each block for its environment update. The unavoidable environments cost T*r*R;
   the documentation now includes this term. Native uses only core contractions and bond
   Grams; there is no call to dense helpers or construction of a global sensitivity operator.
   A 40-mode, rank-one test forbids all dense helpers and succeeds (physical size 2^40).
   Existing mixed-rank, nonbinary and dense/native comparison tests all pass.
7. **T6 square-factor algebra verified; wide-L remains unsupported.** No change to square-L
   behavior. For P=L^+L, Q=LWW^T L^+ obeys Q^2=LW(W^T P W)W^T L^+. When LW has full column
   rank, idempotence requires W^T P W=I. Fresh 2x4 counterexample confirms a defect >.01
   after TT truncation. Existing investigation tests are mathematically sound, not erroneous
   tests to bypass. Current APIs accept wide factors but provide no guard or correction;
   users cannot infer a physical projector or E-orthonormality from the white-space certificate.
   Projecting and reorthogonalizing P W can destroy TT structure. This remains a documented
   limitation, not newly claimed support. The full-column-rank qualification matters: a basis
   wholly in ker(L) produces Q=0, an idempotent map despite failing E-orthonormality.
8. **T6b bootstrap verified.** G contiguous groups, n draws, scale sqrt(G/n): expected resampled
   operator equals the original, since (G/n)*n*(1/G)=1. Single-group repeated-draw regression
   directly checks the operator. Reject zero groups/draws/replicates. Use group_size=S for
   directions belonging to one sample. Mixed snapshots concatenate PCA and DIS blocks, so
   their columns are not automatically contiguous per physical sample; callers must regroup
   before grouped bootstrap. Bootstrap spreads are diagnostics, not Prop. 2 confidence bounds.

### Complexity and numerical limits
Dense fixed-rank bond Gram work is linear in T with ranks/modes fixed. Precompression adds a
thin SVD of the full snapshot matrix. Native cost stays linear in T and d at fixed TT ranks,
with the stated polynomial rank factors and Python per-snapshot overhead. No global H is
explicitly assembled by the audited algorithms; bond Grams can nevertheless approach global
dimensions when intermediate ranks are untruncated. Dense bounds tools intentionally expand
basis matrices and are not scalable native routines.

`tt.from_dense` uses full thin SVDs before truncation: the docstring O(d_Y*n*r) is not a general
bound for arbitrary inputs with a small requested rmax. Actual work sums
O(m_k*c_k*min(m_k,c_k)); rmax reduces later unfoldings, not the current SVD cost. TT rounding
works on existing small core ranks. No algorithm was replaced to conceal this qualification.
Gram conditioning squares singular-value conditioning. Extreme-scale squared energies in
DIS/bounds can still underflow/overflow, and tiny trace-difference tails can cancel. The T1
scale fix does not claim arbitrary-scale robustness for every public routine.

Reproduction: `OPENBLAS_NUM_THREADS=1 PYTHONPATH=src ../../.venv/bin/python notes/codex_review_measure.py`
from the review worktree. Seed 81, one warmup, median of five; seconds:

| algorithm | dimensions | time |
| --- | --- | --- |
| dense, rmax=6 | d=10, T=2000 | .021155659 |
| dense, rmax=6 | d=10, T=4000 | .049513915 (2.340x) |
| native, R=3, rmax=4 | d=8, T=100 | .033799295 |
| native | d=8, T=200 | .068378177 (2.023x) |
| native | d=8, T=400 | .141881164 (2.075x) |
| native | d=6, T=200 | .051409016 |
| native | d=10, T=200 | .090137995 |

Toy eps=.05 ranks: (2,4,8,9,7,6,5); tails:
(0,0,0,.04078594078153802,.039704773587825926,.04154219487596226,.02361893939710861).
At ranks (2,3,4,4,4,3,3): tails
(0,2.494522129578633,.5932344905109305,.3579700035636348,.08873296761174676,
.10321191102499827,.00009937436720072954).
Bound lower/train/upper/test = 2.4945221295786917 / 3.6377708766571133 /
11.434217936849121 / 4.1349229374663485.
Bootstrap 50 replicates, seed 0: grouped std sum ~1.063485 versus column-level ~.246793.
These reproduce the prior task numbers independently; the TT-specific propositions remain
unpublished derivations, not established paper results.

### Exact test results from this run
Commands below are from the review worktree unless labeled root.
- `../../.venv/bin/python -m pytest -q tests/test_toy_problem.py`
  -> `5 passed in 0.07s`.
- Root baseline: `.venv/bin/python -m pytest -q -m 'slow or not slow'`
  -> `2 failed, 134 passed in 8.56s`.
- Regression comparison against baseline: root .venv Python, insert `src` in sys.path,
  import dis_tt.dense first, then pytest.main with `-q -o pythonpath=` and both new review
  test paths, `--tb=short` -> `12 failed, 9 passed, 5 warnings in 0.16s`.
- `../../.venv/bin/python -m pytest -q tests/test_codex_resume_extra.py tests/test_codex_review_extra.py`
  -> `21 passed in 0.08s`.
- `../../.venv/bin/python -m pytest -q --ignore=tests/test_rbno.py -m 'slow or not slow'`
  -> `155 passed in 6.16s`.
- `../../.venv/bin/python -m pytest -q --tb=no`
  -> `2 failed, 153 passed, 2 deselected in 6.10s`.
- `../../.venv/bin/python -m pytest -q -m 'slow or not slow' --tb=no`
  -> `2 failed, 155 passed in 7.95s`.
Both full-suite failures, also on untouched baseline, are test_rbno.py's
`test_reduced_jacobian_branches_agree` and `test_sobolev_loss_zero_at_truth`, raising
NotImplementedError("T7") at rbno.py:54. No completed-task regression and no suspected
incorrect assertion were found. T7 implementation is outside this audit.

TASK: T1–T6b independent audit
STATUS: done (audit and core fixes; full repository still awaits T7)
TESTS: see exact commands and summaries above
EXTRA TESTS ADDED: test_codex_review_extra.py (16 preserved, verified cases); test_codex_resume_extra.py (5 fresh cases)
NUMBERS: toy certificates, bootstrap and timing table above
DEVIATIONS: skip randomized precompression; retain exact thin-SVD spectrum; exact fallback for null complements; native recontraction; documented complexity qualifications
ISSUES: no test ISSUE; wide-L limitation and finite-precision qualifications remain
NEXT: review this branch for integration; do not merge main as part of this audit
