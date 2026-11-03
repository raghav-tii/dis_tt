"""Algorithm C: input DIS-TT, i.e. a whitened TT active subspace (TASK T6).

Math (doc Sec. 3 'Input side', Sec. 5)
--------------------------------------
Whitened coordinates xi = L^{-1} x (C_X = L L^T). Whitened sensitivity
    H_X^w = E[L^T DF^T DF L].
Its top eigenvectors W (Euclidean-orthonormal, here a block TT over the
white-space modes) map to the paper's E-orthonormal input basis V = L W, with
    encoder(x)  = W^T L^{-1} x          (paper eq. 33 with C_X^{-1} v_i = L^{-T} w_i)
    decoder(xi) = L W xi
    Q_TT        = L W W^T L^{-1}        (E-orthogonal projector).
Check: V^T C_X^{-1} V = W^T L^T L^{-T} L^{-1} L W = I.

Pseudocode (doc Sec. 6, Algorithm C) -- the compression itself is Algorithm A
or B applied to input_dis_snapshots(...); this module only wraps the result.

OPEN ISSUE (verify, do not assume): for a WIDE factor L (doc Sec. 8 item 3,
L = A^{-1}[sqrt(c) grad^T, sqrt(kappa) I]) the white space is larger than X,
L^{-1} must be the pseudo-inverse L^+ = L^T C^{-1}, and W may contain
components in ker(L). Then Q_TT above is still a projector only if
range(W) is orthogonal to ker(L); otherwise project W onto range(L^T) first.
Write a test for the wide case before supporting it.

Complexity: snapshot cost N * |dirs| adjoint solves + one L^T application each;
compression as Algorithm A/B on the white-space modes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from . import tt
from .dense import dis_tt_dense
from .interfaces import DISTTResult, GaussianPrior, SnapshotSet


@dataclass
class WhitenedInputBasis:
    """Input DIS-TT basis plus its encoder/decoder (dense apply for now)."""

    result: DISTTResult
    prior: GaussianPrior

    def W(self) -> np.ndarray:
        """Dense white-space basis (white_dim, r_X). Requires result.is_basis."""
        if not self.result.is_basis:
            raise ValueError("W() needs an r-column basis (D = d)")
        return tt.block_to_dense(self.result.cores)

    def encode(self, x: np.ndarray) -> np.ndarray:
        """xi = W^T L^{-1} x; x (d_X,) or (d_X, n) -> (r_X,) or (r_X, n)."""
        return self.W().T @ self.prior.apply_inv_sqrt(np.asarray(x, dtype=float))

    def decode(self, xi: np.ndarray) -> np.ndarray:
        """x = L W xi."""
        return self.prior.apply_sqrt(self.W() @ np.asarray(xi, dtype=float))

    def V(self) -> np.ndarray:
        """Physical E-orthonormal basis L W, shape (d_X, r_X)."""
        return self.prior.apply_sqrt(self.W())


def input_dis_tt(snaps: SnapshotSet, white_mode_sizes: Sequence[int], prior: GaussianPrior,
                 **dense_kwargs) -> WhitenedInputBasis:
    """Run Algorithm A on whitened VJP snapshots (D = d forced) and wrap the result."""
    modes = tuple(int(n) for n in white_mode_sizes)
    if snaps.Z.shape[0] != int(np.prod(modes)):
        raise ValueError(f"snapshots have {snaps.Z.shape[0]} rows, prod(white_mode_sizes) = {int(np.prod(modes))}")
    D = dense_kwargs.pop("D", None)
    if D is not None and int(D) != len(modes):
        raise ValueError("input_dis_tt needs an r-column basis: D = d is forced")
    result = dis_tt_dense(snaps.Z, modes, D=len(modes), **dense_kwargs)
    return WhitenedInputBasis(result=result, prior=prior)
