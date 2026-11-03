"""Experiments from note Sec. 8 'First experiments, in order' (TASK T9: run AFTER T1-T7 pass).

Each function writes ``results/<name>.csv`` (+ optional figure ``results/<name>.png``) and
returns a dict of headline numbers. No function may change library code; failures of a
theoretical claim are RESULTS to report, not bugs to hide.

Run:  python experiments/run_checklist.py exp1 exp2 ...
"""
from __future__ import annotations

import sys


def exp1_consistency():
    """D = d, full ranks reproduce dense DIS (eigenvalues, basis) to round-off.

    Problems: toy (ToyProblem(8, 7)) and, once T9-PDE exists, the paper's 1D semilinear
    elliptic problem on 256 interior nodes (QTT d = 8).
    Report: max rel. eigenvalue error (top 10), sin(theta_max) of the top-r subspaces, r = 2..8.
    """
    raise NotImplementedError("T9")


def exp2_bound_check():
    """Prop. 1 / Prop. 2 on held-out data.

    For eps in {0.3, 0.1, 0.03, 0.01} and N in {10, 40, 160}: train_error (== sum tails),
    test_error, lower_unprojected, upper_unprojected, bootstrap std per bond.
    Pass criterion for the THEORY (report, don't assert): test_error <= upper + sampling term.
    """
    raise NotImplementedError("T9")


def exp3_dis_vs_pca_vs_mixed():
    """Held-out L2 and H1 reconstruction error vs total TT parameter count.

    Output side: PCA-TT, DIS-TT, mixed alpha in {0.25, 0.5, 0.75}. Input side: KL (input PCA)
    vs DIS-TT. Paper's prediction: clear input-side DIS win, output-side parity.
    """
    raise NotImplementedError("T9")


def exp4_direction_sets():
    """Random (S), KL (K) and exact columns at matched solve budget: tail error and its variance over 20 seeds."""
    raise NotImplementedError("T9")


def exp5_refinement():
    """Increase QTT depth d at a fixed physical problem; record max_k r_k at fixed eps.

    DIS-TT is useful only if ranks saturate in d. Also record per-bond spectra of H_Y vs C_Y
    (caveat 4: derivative spectra may decay more slowly).
    """
    raise NotImplementedError("T9")


def exp6_end_to_end():
    """Train the RBNO (T8) with TT bases vs dense DIS bases at equal (r_X, r_Y); paper Sec. 4.4.3 protocol."""
    raise NotImplementedError("T9")


if __name__ == "__main__":
    EXPS = {k.split("_")[0]: v for k, v in globals().items() if k.startswith("exp")}
    for key in sys.argv[1:] or ["exp1"]:
        print(key, EXPS[key]())
