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
                         rng: np.random.Generator | None = None, batch: bool = True) -> SnapshotSet:
    """JVP snapshots for H_Y. xs: (d_X, N). One ``model.solve`` per column of xs.

    n_dirs = S for "random", K for "kl", ignored for "exact".
    Batch the JVPs per x (pass an (d_X, k) block) when the model supports it;
    ``batch=False`` falls back to one JVP per column. ``rng`` defaults to default_rng(0).
    """
    xs = _as_columns(xs)
    N = xs.shape[1]
    if direction_kind == "random":
        rng = np.random.default_rng(0) if rng is None else rng
        S = int(n_dirs)
        if S < 1:
            raise ValueError("n_dirs must be positive")
        dirs = lambda: prior.sample(rng, S)  # fresh h ~ N(0, C_X) per x
        scale = 1.0 / np.sqrt(N * S)
    elif direction_kind == "kl":
        mu, phi = prior.kl_modes(int(n_dirs))
        fixed = phi * np.sqrt(np.clip(mu, 0.0, None))[None, :]
        S, dirs, scale = fixed.shape[1], (lambda: fixed), 1.0 / np.sqrt(N)
    elif direction_kind == "exact":
        fixed = prior.apply_sqrt(np.eye(prior.white_dim))  # columns L e_j
        S, dirs, scale = fixed.shape[1], (lambda: fixed), 1.0 / np.sqrt(N)
    else:
        raise ValueError(f"unknown direction_kind {direction_kind!r}")
    blocks = []
    for x in xs.T:
        state = model.solve(x)
        blocks.append(_apply_block(model.jvp, state, dirs(), batch))
    Z = scale * np.concatenate(blocks, axis=1)
    return SnapshotSet(Z=Z, kind="dis_out", n_samples=N, n_dirs=S, direction_kind=direction_kind,
                       meta={"scale": scale})


def output_pca_snapshots(model: ForwardModel, xs: np.ndarray) -> tuple[SnapshotSet, np.ndarray]:
    """Centred, scaled output snapshots and the sample mean y_bar."""
    xs = _as_columns(xs)
    N = xs.shape[1]
    Y = np.stack([np.asarray(model.output(model.solve(x)), dtype=float) for x in xs.T], axis=1)
    y_bar = Y.mean(axis=1)
    scale = 1.0 / np.sqrt(N)  # biased covariance estimator (paper eq. 17)
    Z = scale * (Y - y_bar[:, None])
    return SnapshotSet(Z=Z, kind="pca_out", n_samples=N, n_dirs=1, meta={"scale": scale}), y_bar


def input_dis_snapshots(model: ForwardModel, prior: GaussianPrior, xs: np.ndarray,
                        eta_kind: str = "gaussian", n_dirs: int = 8,
                        U_out: np.ndarray | None = None,
                        rng: np.random.Generator | None = None, batch: bool = True) -> SnapshotSet:
    """Whitened VJP snapshots for H_X^w (eta_kind "gaussian") or H_X^{w,U} ("basis").

    U_out: dense (d_Y, r_Y) orthonormal output basis, required for "basis".
    ``batch=False`` calls the VJP column by column. ``rng`` defaults to default_rng(0).
    """
    xs = _as_columns(xs)
    N = xs.shape[1]
    if eta_kind == "gaussian":
        rng = np.random.default_rng(0) if rng is None else rng
        S = int(n_dirs)
        if S < 1:
            raise ValueError("n_dirs must be positive")
        scale = 1.0 / np.sqrt(N * S)
    elif eta_kind == "basis":
        if U_out is None:
            raise ValueError('eta_kind="basis" requires U_out')
        U_out = np.asarray(U_out, dtype=float)
        S = U_out.shape[1]
        scale = 1.0 / np.sqrt(N)
    else:
        raise ValueError(f"unknown eta_kind {eta_kind!r}")
    blocks = []
    for x in xs.T:
        state = model.solve(x)
        if eta_kind == "gaussian":
            d_y = np.asarray(model.output(state)).shape[0]
            eta = rng.standard_normal((d_y, S))
        else:
            eta = U_out
        g = _apply_block(model.vjp, state, eta, batch)       # DF^T eta, (d_X, S)
        blocks.append(prior.apply_sqrt_T(g))                  # L^T DF^T eta, (white_dim, S)
    Z = scale * np.concatenate(blocks, axis=1)
    return SnapshotSet(Z=Z, kind="dis_in", n_samples=N, n_dirs=S, direction_kind=eta_kind,
                       meta={"scale": scale})


def mixed_snapshots(pca: SnapshotSet, dis: SnapshotSet, alpha: float) -> SnapshotSet:
    """Weighted concatenation for H_alpha (both inputs pre-scaled)."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    if pca.Z.shape[0] != dis.Z.shape[0]:
        raise ValueError("pca and dis snapshots live in different spaces")
    def _w(Z, a):
        nrm2 = float(np.sum(Z * Z))
        return np.sqrt(a / nrm2) * Z if nrm2 > 0 else np.zeros_like(Z)
    Z = np.concatenate([_w(pca.Z, alpha), _w(dis.Z, 1.0 - alpha)], axis=1)
    return SnapshotSet(Z=Z, kind="mixed", n_samples=pca.n_samples, n_dirs=pca.n_dirs + dis.n_dirs,
                       direction_kind=dis.direction_kind,
                       meta={"alpha": alpha, "n_pca_cols": pca.Z.shape[1], "n_dis_cols": dis.Z.shape[1]})


def _as_columns(xs: np.ndarray) -> np.ndarray:
    xs = np.asarray(xs, dtype=float)
    xs = xs[:, None] if xs.ndim == 1 else xs
    if xs.ndim != 2 or xs.shape[1] == 0:
        raise ValueError("xs must contain at least one sample in a column matrix")
    return xs


def _apply_block(op, state, V: np.ndarray, batch: bool) -> np.ndarray:
    """op(state, V) for a (dim, k) block: one batched call, or column by column if batch=False."""
    if batch:
        out = np.asarray(op(state, V), dtype=float)
        return out[:, None] if out.ndim == 1 else out
    return np.stack([np.asarray(op(state, v), dtype=float) for v in V.T], axis=1)
