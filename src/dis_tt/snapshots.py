"""Matrix-free estimators: snapshot builders (TASK T3; doc Sec. 5).

Every builder returns a PRE-SCALED :class:`~dis_tt.interfaces.SnapshotSet`
with Z Z^T equal to the estimator below. Compression routines never rescale.

Output DIS (doc Sec. 5, 'Output side'):
    H_Y = E_{x,h~gamma}[z z^T],  z = DF(x) h
    direction_kind  columns per x        column scale       estimator
    "random"        S prior draws h      1/sqrt(N S)        unbiased
    "kl"            sqrt(mu_j) phi_j     1/sqrt(N)          exact up to KL tail
    "exact"         L e_j, j<=white_dim  1/sqrt(N)          exact (d_X small only)

Output PCA:   columns (y_s - y_bar)/sqrt(N), y_bar the sample mean (biased
              estimator, paper eq. 17).

Input DIS, whitened (doc Sec. 5, 'Input side'):
    H_X^w = E[g g^T],  g = L^T DF(x)^T eta
    eta_kind "gaussian": S draws eta ~ N(0, I_{d_Y}), scale 1/sqrt(N S) -> H_X^w
    eta_kind "basis"   : eta = u_a, a = 1..r_Y (columns of output basis U),
                         scale 1/sqrt(N) -> H_X^{w,U} = E[L^T DF^T U U^T DF L]
    NOTE L^T, not L: for a non-symmetric / rectangular factor the whitened
    Jacobian is DF L, so its Gram is L^T DF^T DF L. For the toy prior L = L^T.

Mixed (doc Sec. 4, 'Mixed objective'):
    H_alpha = alpha C_Y / Tr C_Y + (1 - alpha) H_Y / Tr H_Y
    -> concatenate sqrt(alpha / ||Z_pca||_F^2) Z_pca and
       sqrt((1-alpha) / ||Z_dis||_F^2) Z_dis.

Cost: N nonlinear solves + (columns per x) * N linearised solves; storage d * T.
"""
from __future__ import annotations

import numpy as np

from .interfaces import ForwardModel, GaussianPrior, SnapshotSet


def output_dis_snapshots(model: ForwardModel, prior: GaussianPrior, xs: np.ndarray,
                         direction_kind: str = "random", n_dirs: int = 8,
                         rng: np.random.Generator | None = None) -> SnapshotSet:
    """JVP snapshots for H_Y. xs: (d_X, N). One ``model.solve`` per column of xs.

    n_dirs = S for "random", K for "kl", ignored for "exact".
    Batch the JVPs per x (pass an (d_X, k) block) when the model supports it.
    """
    raise NotImplementedError("T3")


def output_pca_snapshots(model: ForwardModel, xs: np.ndarray) -> tuple[SnapshotSet, np.ndarray]:
    """Centred, scaled output snapshots and the sample mean y_bar."""
    raise NotImplementedError("T3")


def input_dis_snapshots(model: ForwardModel, prior: GaussianPrior, xs: np.ndarray,
                        eta_kind: str = "gaussian", n_dirs: int = 8,
                        U_out: np.ndarray | None = None,
                        rng: np.random.Generator | None = None) -> SnapshotSet:
    """Whitened VJP snapshots for H_X^w (eta_kind "gaussian") or H_X^{w,U} ("basis").

    U_out: dense (d_Y, r_Y) orthonormal output basis, required for "basis".
    """
    raise NotImplementedError("T3")


def mixed_snapshots(pca: SnapshotSet, dis: SnapshotSet, alpha: float) -> SnapshotSet:
    """Weighted concatenation for H_alpha (both inputs pre-scaled)."""
    raise NotImplementedError("T3")
