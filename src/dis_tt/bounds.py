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

from .interfaces import BoundReport, DISTTResult


def projection_error(result: DISTTResult, Z: np.ndarray) -> float:
    """||(I - Pi_D) Z||_F^2 using the dense subspace matrix of ``result``."""
    raise NotImplementedError("T6b")


def unprojected_tails(Z: np.ndarray, mode_sizes: Sequence[int], ranks: Sequence[int]) -> np.ndarray:
    """tail_{r_k}(H^{(k)}) for k = 1..len(ranks), H^{(k)} = Gram of left unfolding k of (n_1..n_d, T)."""
    raise NotImplementedError("T6b")


def bound_report(result: DISTTResult, Z_train: np.ndarray, Z_test: np.ndarray) -> BoundReport:
    """Assemble the Prop. 1 / Prop. 2 numbers. Z_test pre-scaled like Z_train."""
    raise NotImplementedError("T6b")


def bootstrap_tails(Z: np.ndarray, mode_sizes: Sequence[int], ranks: Sequence[int],
                    n_boot: int = 50, rng: np.random.Generator | None = None,
                    group_size: int = 1) -> np.ndarray:
    """Bootstrap std of per-bond tails at FIXED ranks. Returns (D,).

    Resample snapshot columns with replacement in groups of ``group_size``
    (= S, so all directions of one x stay together), rescale to keep the
    estimator unbiased, rerun Algorithm A with ``ranks`` fixed.
    """
    raise NotImplementedError("T6b")
