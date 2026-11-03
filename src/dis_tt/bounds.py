"""Certificates and bound checks (TASK T6b; doc Sec. 4, experiment 'Bound check').

Prop. 1, empirical form (training snapshots Z, result from Algorithm A/B):
    ||(I - Pi_D) Z||_F^2 == sum_k tail_k                       (exact)
    max_k tail_{r_k}(H^{(k)})  <=  min over TT bases  <=  sum_k tail_k  <=  sum_k tail_{r_k}(H^{(k)})
where H^{(k)} = Tr_{>k}(Z Z^T) is the UNPROJECTED partial trace, i.e. the Gram
of the k-th left unfolding of the tensor (n_1, ..., n_d, T).

Prop. 2 (sampling): held-out ||(I - Pi_D) Z_test||_F^2 versus the training
certificate; per-bond spread estimated by bootstrap over snapshot columns
(doc Sec. 8, item 5).

All functions here are DENSE reference tools for small problems (d_Y <= ~2^14).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from . import tt
from .dense import dis_tt_dense
from .interfaces import BoundReport, DISTTResult


def projection_error(result: DISTTResult, Z: np.ndarray) -> float:
    """||(I - Pi_D) Z||_F^2 using the dense subspace matrix of ``result``."""
    Z = _as_matrix(Z)
    B = tt.subspace_matrix(result.cores, result.mode_sizes)
    R = Z - B @ (B.T @ Z)
    return float(np.sum(R * R))


def unprojected_tails(Z: np.ndarray, mode_sizes: Sequence[int], ranks: Sequence[int]) -> np.ndarray:
    """tail_{r_k}(H^{(k)}) for k = 1..len(ranks), H^{(k)} = Gram of left unfolding k of (n_1..n_d, T)."""
    Z = _as_matrix(Z)
    modes = tuple(int(n) for n in mode_sizes)
    if len(ranks) > len(modes):
        raise ValueError("more ranks than modes")
    out = np.empty(len(ranks))
    for k, r in enumerate(ranks):
        M = Z.reshape(int(np.prod(modes[: k + 1])), -1)  # rows (i_1..i_k), columns (i_{k+1}..i_d, tau)
        s2 = np.linalg.svd(M, compute_uv=False) ** 2       # eigenvalues of H^(k) (rest are zero)
        out[k] = float(np.sum(s2[int(r):]))
    return out


def bound_report(result: DISTTResult, Z_train: np.ndarray, Z_test: np.ndarray) -> BoundReport:
    """Assemble the Prop. 1 / Prop. 2 numbers. Z_test pre-scaled like Z_train."""
    Z_train, Z_test = _as_matrix(Z_train), _as_matrix(Z_test)
    per_bond = unprojected_tails(Z_train, result.mode_sizes, result.ranks)
    train, test = projection_error(result, Z_train), projection_error(result, Z_test)
    total_test = float(np.sum(Z_test * Z_test))
    return BoundReport(
        train_error=train, test_error=test,
        upper_unprojected=float(per_bond.sum()), lower_unprojected=float(per_bond.max()),
        sum_tails=float(np.sum(result.tails)), per_bond_unprojected=per_bond,
        meta={"total_train": result.total, "total_test": total_test,
              "train_rel": train / result.total if result.total > 0 else 0.0,
              "test_rel": test / total_test if total_test > 0 else 0.0,
              "cert": result.cert, "ranks": result.ranks, "tails": np.asarray(result.tails).copy()})


def bootstrap_tails(Z: np.ndarray, mode_sizes: Sequence[int], ranks: Sequence[int],
                    n_boot: int = 50, rng: np.random.Generator | None = None,
                    group_size: int = 1, n_groups: int | None = None,
                    backend: str = "auto") -> np.ndarray:
    """Bootstrap std of per-bond tails at FIXED ranks. Returns (D,).

    Resample snapshot columns with replacement in groups of ``group_size``
    (= S, so all directions of one x stay together), rescale to keep the
    estimator unbiased, rerun Algorithm A with ``ranks`` fixed.

    Scaling: with G = T / group_size groups and ``n_groups`` draws (default G), the resample is
    multiplied by sqrt(G / n_groups), so E*[Z* Z*^T] = Z Z^T (the 1/sqrt(N S) weights inside
    Z are already per group). With the default this factor is exactly 1.
    """
    Z = _as_matrix(Z)
    T = Z.shape[1]
    g = int(group_size)
    if g < 1 or T % g:
        raise ValueError(f"group_size={group_size} must divide the number of columns T={T}")
    G = T // g
    n_draw = G if n_groups is None else int(n_groups)
    rng = np.random.default_rng(0) if rng is None else rng
    scale = np.sqrt(G / n_draw)
    groups = Z.reshape(Z.shape[0], G, g)  # columns grouped contiguously, one x per group
    tails = np.empty((int(n_boot), len(ranks)))
    for b in range(int(n_boot)):
        idx = rng.integers(0, G, size=n_draw)
        Zb = scale * groups[:, idx, :].reshape(Z.shape[0], n_draw * g)
        tails[b] = dis_tt_dense(Zb, mode_sizes, ranks=ranks, D=len(ranks), backend=backend).tails
    return tails.std(axis=0, ddof=1) if n_boot > 1 else np.zeros(len(ranks))


def _as_matrix(Z: np.ndarray) -> np.ndarray:
    Z = np.asarray(Z, dtype=float)
    return Z[:, None] if Z.ndim == 1 else Z
