"""Algorithm B: greedy DIS-TT from snapshots already in TT format (TASK T5).

Equivalent to :func:`dis_tt.dense.dis_tt_dense` applied to the densified
snapshots (same spectra, tails and subspaces up to round-off and eigenvector
sign/rotation inside degenerate eigenspaces) but never forms a dense vector.

Pseudocode (doc Sec. 6, Algorithm B)
------------------------------------
    for tau = 1..T:
        right-orthogonalise z_tau (cores d..2)       # Tr_{>k}(z z^T) = L_k L_k^T
        Phi_tau <- [[1]]                             # left environment U_{<=0}^T L_0
    total <- sum_tau ||Z_tau^(1)||_F^2 ; delta2 <- eps^2 * total / D
    for k = 1..D:
        for tau: B_tau[(a,i), b] <- sum_c Phi_tau[a,c] Z_tau^(k)[c,i,b]   # (r_{k-1} n_k) x R_k
        Hk <- sum_tau B_tau B_tau^T                  # stream; (r_{k-1} n_k)^2 memory
        (Lam, V) <- eigh(Hk) descending
        r_k, tail_k <- select_rank(...)
        G_k <- V[:, :r_k]
        for tau: Phi_tau <- G_k^T B_tau              # r_k x R_k
    return DISTTResult

Row ordering of B_tau must be (a, i) with a = alpha_{k-1} slowest, matching
reshape(G_k, (r_{k-1} n_k, r_k)) and Algorithm A. At k = d every R_d = 1.

Complexity (doc Sec. 7): O(T d (n R^3 + r n R^2 + r^2 n^2 R)) + O(d r^3 n^3);
memory O(T d n R^2 + (r n)^2). Do NOT build the stacked rank-(T R) TT
(that is the O(d n (T R)^3) naive alternative).

Snapshots are PRE-SCALED (sum_tau z z^T = H_hat); see dis_tt.snapshots.

Import TT helpers as ``from . import tt`` and call ``tt.right_orthogonalize``
etc.; test_native.py monkeypatches ``tt.to_dense`` to prove nothing is densified.
"""
from __future__ import annotations

from typing import Sequence

from .interfaces import DISTTResult
from .tt import TensorTrain


def dis_tt_native(snapshots: Sequence[TensorTrain], *, eps: float | None = None,
                  rmax: int | None = None, ranks: Sequence[int] | None = None,
                  D: int | None = None) -> DISTTResult:
    """Greedy DIS-TT on a list of TT snapshots with common mode sizes."""
    raise NotImplementedError("T5")
