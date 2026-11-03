# Implementation notes

One file per task: `notes/T1.md`, `notes/T4.md`, ... (separate files avoid merge conflicts between parallel agents). In each file record design choices, deviations from the docstrings, surprises,
and any ISSUE with a test or with `docs/theory.md` (minimal reproduction plus argument).

## Scaffold (done)

- Tests were validated against a private reference implementation (61/61 pass, including
  `-m slow`). Subspace-angle tolerances are 1e-6, not 1e-8: Gram-based eigenvectors are only
  accurate to ~sqrt(machine eps)/gap once round-off modes are dropped at intermediate bonds.
- The note's pseudocode writes C^{1/2} on the adjoint side (Alg. C line 6, Alg. D line 8).
  That is correct for the symmetric square root. The code generalises to any factor
  C_X = L L^T and therefore uses L^T there (`GaussianPrior.apply_sqrt_T`).
- Open: wide whitening factors (input_side.py docstring), owned by T6.
