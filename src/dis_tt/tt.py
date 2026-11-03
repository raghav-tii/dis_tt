"""Minimal tensor-train toolkit (TASK T1).

Only what Algorithms A-D need. Deliberately numpy-only so the DIS-TT logic is
testable without an external TT library; an adapter to ttpy / teneva / the
user's ``tensnet`` can be added later (T9) but must not replace these tests.

Conventions: core shape (r_{k-1}, n_k, r_k); dense <-> tensor via C-order
reshape (mode 1 slowest). See :mod:`dis_tt.interfaces`.

Block TT
--------
A block TT with r columns is a list of cores whose LAST core has shape
(r_{D-1}, n_D, r) instead of (r_{D-1}, n_D, 1). Its dense form is the basis
matrix U of shape (n_1...n_D, r). This is the output of DIS-TT with D = d.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class TensorTrain:
    """Plain TT: cores[k] shape (r_{k-1}, n_k, r_k), r_0 = r_d = 1."""

    cores: list[np.ndarray]

    @property
    def d(self) -> int:
        return len(self.cores)

    @property
    def mode_sizes(self) -> tuple[int, ...]:
        return tuple(int(G.shape[1]) for G in self.cores)

    @property
    def ranks(self) -> tuple[int, ...]:
        """(r_0, r_1, ..., r_d)."""
        return (int(self.cores[0].shape[0]),) + tuple(int(G.shape[2]) for G in self.cores)

    def copy(self) -> "TensorTrain":
        return TensorTrain([G.copy() for G in self.cores])


# --------------------------------------------------------------------------
# T1 stubs
# --------------------------------------------------------------------------
def from_dense(y: np.ndarray, mode_sizes: Sequence[int], eps: float = 0.0, rmax: int | None = None) -> TensorTrain:
    """Standard TT-SVD of a single dense vector (Oseledets 2011).

    Relative Frobenius error <= eps, using per-bond threshold eps*||y||/sqrt(d-1).
    eps = 0 -> exact (drop only singular values below 1e-14 * sigma_1).

    Complexity O(d_Y * n * r) with thin SVDs. Used by tests to build TT snapshots;
    NOT the DIS-TT algorithm (that one works on snapshot ensembles).
    """
    raise NotImplementedError("T1")


def to_dense(tt: TensorTrain) -> np.ndarray:
    """Contract a TT to a dense vector of length prod(n_k) (C order)."""
    raise NotImplementedError("T1")


def block_to_dense(cores: Sequence[np.ndarray]) -> np.ndarray:
    """Dense basis matrix of a left frame / block TT.

    Given cores G_1..G_D (shapes (r_{k-1}, n_k, r_k), r_0 = 1) return
    U_{<=D} of shape (n_1...n_D, r_D), rows in C order. Recursion:
    U_{<=k} = (U_{<=k-1} (x) I_{n_k}) reshape(G_k, (r_{k-1} n_k, r_k)).
    Check the Kronecker ordering against the C-order convention in the tests.
    """
    raise NotImplementedError("T1")


def subspace_matrix(cores: Sequence[np.ndarray], mode_sizes: Sequence[int]) -> np.ndarray:
    """Orthonormal basis of range(Pi_D) as a dense matrix of shape (d_Y, dim).

    D = d   -> returns block_to_dense(cores)            (dim = r_d)
    D < d   -> returns U_{<=D} (x) I_{n_{D+1}...n_d}     (dim = r_D * n_{D+1}...n_d)
    Dense; for tests and small problems only.
    """
    raise NotImplementedError("T1")


def right_orthogonalize(tt: TensorTrain) -> TensorTrain:
    """Return an equal TT whose cores 2..d are right-orthogonal.

    Right-orthogonal: reshape(G_k, (r_{k-1}, n_k r_k)) has orthonormal rows.
    Afterwards ||tt||_F = ||cores[0]||_F and, for every bond k, the partial trace
    Tr_{>k}(z z^T) = L_k L_k^T with L_k = contraction of cores 1..k
    (Algorithm B line 3). Cost O(d n R^3) via RQ/LQ.
    """
    raise NotImplementedError("T1")


def project_block(cores: Sequence[np.ndarray], tt: TensorTrain) -> np.ndarray:
    """Coefficients U^T z for a block TT U (r columns) and a TT z. Returns (r,).

    Left-to-right environment contraction, never densifies:
    cost O(d n r R (r + R)).
    """
    raise NotImplementedError("T1")


def block_apply(cores: Sequence[np.ndarray], coeffs: np.ndarray) -> TensorTrain:
    """U @ coeffs as a plain TT (absorb coeffs into the last core). (Algorithm D line 10.)"""
    raise NotImplementedError("T1")


def tt_add(a: TensorTrain, b: TensorTrain, alpha: float = 1.0, beta: float = 1.0) -> TensorTrain:
    """alpha a + beta b, rank r_a + r_b (block-diagonal cores), no rounding."""
    raise NotImplementedError("T1")


def tt_round(tt: TensorTrain, eps: float, rmax: int | None = None) -> TensorTrain:
    """TT rounding (orthogonalise right-to-left, truncated SVD left-to-right)."""
    raise NotImplementedError("T1")


def left_orthonormality_defect(cores: Sequence[np.ndarray]) -> float:
    """max_k ||G_k^T G_k - I||_2 over the left-unfoldings (diagnostic, tests)."""
    raise NotImplementedError("T1")
