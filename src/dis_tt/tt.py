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


def _truncation_rank(s: np.ndarray, delta: float, rmax: int | None) -> int:
    """Smallest r >= 1 with ||s[r:]||_2 <= delta (delta = 0: drop s < 1e-14 s_1), capped at rmax."""
    if s.size == 0 or s[0] == 0.0:
        r = 1
    elif delta > 0:
        tail = np.sqrt(np.cumsum((s ** 2)[::-1]))[::-1]  # tail[j] = ||s[j:]||
        r = max(1, int(np.sum(tail > delta)))
    else:
        r = max(1, int(np.sum(s > 1e-14 * s[0])))
    if rmax is not None:
        r = min(r, int(rmax))
    return max(1, min(r, s.size))


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
    y = np.asarray(y, dtype=float).ravel()
    modes = tuple(int(n) for n in mode_sizes)
    if y.size != int(np.prod(modes)):
        raise ValueError(f"len(y)={y.size} != prod(mode_sizes)={int(np.prod(modes))}")
    d = len(modes)
    delta = eps * np.linalg.norm(y) / np.sqrt(d - 1) if (eps > 0 and d > 1) else 0.0
    cores: list[np.ndarray] = []
    C, r_prev = y, 1
    for n in modes[:-1]:
        M = C.reshape(r_prev * n, -1)
        Uk, s, Vt = np.linalg.svd(M, full_matrices=False)
        r = _truncation_rank(s, delta, rmax)
        cores.append(Uk[:, :r].reshape(r_prev, n, r))
        C = s[:r, None] * Vt[:r]
        r_prev = r
    cores.append(C.reshape(r_prev, modes[-1], 1))
    return TensorTrain(cores)


def to_dense(tt: TensorTrain) -> np.ndarray:
    """Contract a TT to a dense vector of length prod(n_k) (C order)."""
    return block_to_dense(tt.cores).reshape(-1)


def block_to_dense(cores: Sequence[np.ndarray]) -> np.ndarray:
    """Dense basis matrix of a left frame / block TT.

    Given cores G_1..G_D (shapes (r_{k-1}, n_k, r_k), r_0 = 1) return
    U_{<=D} of shape (n_1...n_D, r_D), rows in C order. Recursion:
    U_{<=k} = (U_{<=k-1} (x) I_{n_k}) reshape(G_k, (r_{k-1} n_k, r_k)).
    Check the Kronecker ordering against the C-order convention in the tests.
    """
    U = cores[0].reshape(-1, cores[0].shape[2])  # rows (alpha_0, i_1), alpha_0 slowest; r_0 = 1
    for G in cores[1:]:
        # (U (x) I_{n_k}) G_k with rows (i_1..i_{k-1}, i_k) in C order
        U = np.einsum("Ia,air->Iir", U, G).reshape(-1, G.shape[2])
    return U


def subspace_matrix(cores: Sequence[np.ndarray], mode_sizes: Sequence[int]) -> np.ndarray:
    """Orthonormal basis of range(Pi_D) as a dense matrix of shape (d_Y, dim).

    D = d   -> returns block_to_dense(cores)            (dim = r_d)
    D < d   -> returns U_{<=D} (x) I_{n_{D+1}...n_d}     (dim = r_D * n_{D+1}...n_d)
    Dense; for tests and small problems only.
    """
    U = block_to_dense(cores)
    D = len(cores)
    if D >= len(mode_sizes):
        return U
    rest = int(np.prod(mode_sizes[D:]))
    return np.kron(U, np.eye(rest))


def right_orthogonalize(tt: TensorTrain) -> TensorTrain:
    """Return an equal TT whose cores 2..d are right-orthogonal.

    Right-orthogonal: reshape(G_k, (r_{k-1}, n_k r_k)) has orthonormal rows.
    Afterwards ||tt||_F = ||cores[0]||_F and, for every bond k, the partial trace
    Tr_{>k}(z z^T) = L_k L_k^T with L_k = contraction of cores 1..k
    (Algorithm B line 3). Cost O(d n R^3) via RQ/LQ.
    """
    cores = [G.copy() for G in tt.cores]
    for k in range(len(cores) - 1, 0, -1):
        G = cores[k]
        r0, n, r1 = G.shape
        Q, R = np.linalg.qr(G.reshape(r0, n * r1).T)  # G_k^(r) = R^T Q^T
        cores[k] = Q.T.reshape(-1, n, r1)
        cores[k - 1] = np.einsum("anb,cb->anc", cores[k - 1], R)
    return TensorTrain(cores)


def project_block(cores: Sequence[np.ndarray], tt: TensorTrain) -> np.ndarray:
    """Coefficients U^T z for a block TT U (r columns) and a TT z. Returns (r,).

    Left-to-right environment contraction, never densifies:
    cost O(d n r R (r + R)).
    """
    if len(cores) != tt.d:
        raise ValueError("block and TT must have the same number of modes")
    E = np.ones((1, 1))  # (alpha_U, beta_z)
    for G, H in zip(cores, tt.cores):
        E = np.einsum("ab,air,bis->rs", E, G, H, optimize=True)
    return E[:, 0]


def block_apply(cores: Sequence[np.ndarray], coeffs: np.ndarray) -> TensorTrain:
    """U @ coeffs as a plain TT (absorb coeffs into the last core). (Algorithm D line 10.)"""
    coeffs = np.asarray(coeffs, dtype=float)
    out = [G.copy() for G in cores[:-1]]
    out.append(np.tensordot(cores[-1], coeffs, axes=([2], [0])).reshape(cores[-1].shape[0], cores[-1].shape[1], 1))
    return TensorTrain(out)


def tt_add(a: TensorTrain, b: TensorTrain, alpha: float = 1.0, beta: float = 1.0) -> TensorTrain:
    """alpha a + beta b, rank r_a + r_b (block-diagonal cores), no rounding."""
    if a.mode_sizes != b.mode_sizes:
        raise ValueError("mode sizes differ")
    d = a.d
    if d == 1:
        return TensorTrain([alpha * a.cores[0] + beta * b.cores[0]])
    cores = []
    for k, (A, B) in enumerate(zip(a.cores, b.cores)):
        if k == 0:
            cores.append(np.concatenate([alpha * A, beta * B], axis=2))
        elif k == d - 1:
            cores.append(np.concatenate([A, B], axis=0))
        else:
            ra0, n, ra1 = A.shape
            rb0, _, rb1 = B.shape
            C = np.zeros((ra0 + rb0, n, ra1 + rb1), dtype=np.result_type(A, B))
            C[:ra0, :, :ra1] = A
            C[ra0:, :, ra1:] = B
            cores.append(C)
    return TensorTrain(cores)


def tt_round(tt: TensorTrain, eps: float, rmax: int | None = None) -> TensorTrain:
    """TT rounding (orthogonalise right-to-left, truncated SVD left-to-right)."""
    R = right_orthogonalize(tt)
    cores = R.cores
    d = len(cores)
    nrm = np.linalg.norm(cores[0])  # = ||tt||_F after right-orthogonalisation
    delta = eps * nrm / np.sqrt(d - 1) if (eps > 0 and d > 1) else 0.0
    for k in range(d - 1):
        r0, n, r1 = cores[k].shape
        Uk, s, Vt = np.linalg.svd(cores[k].reshape(r0 * n, r1), full_matrices=False)
        r = _truncation_rank(s, delta, rmax)
        cores[k] = Uk[:, :r].reshape(r0, n, r)
        cores[k + 1] = np.einsum("ab,bnc->anc", s[:r, None] * Vt[:r], cores[k + 1])
    return TensorTrain(cores)


def left_orthonormality_defect(cores: Sequence[np.ndarray]) -> float:
    """max_k ||G_k^T G_k - I||_2 over the left-unfoldings (diagnostic, tests)."""
    out = 0.0
    for G in cores:
        M = G.reshape(-1, G.shape[2])
        out = max(out, float(np.linalg.norm(M.T @ M - np.eye(M.shape[1]), 2)))
    return out
