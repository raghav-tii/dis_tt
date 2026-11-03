"""Toy test problem with exact Jacobians (IMPLEMENTED -- test fixture, not a task).

Map
---
    F(x) = A tanh(B x + b0),        F : R^{d_X} -> R^{d_Y}
    DF(x) h = A diag(sech^2(B x + b0)) B h
    DF(x)^T w = B^T diag(sech^2(B x + b0)) A^T w

* d_Y = 2**out_levels (QTT-friendly). Columns of A are smooth or localised
  functions on a uniform grid of (0, 1), so outputs, JVPs and the DIS basis have
  low QTT rank. range(DF) is contained in range(A), so rank(H_Y) <= p.
* d_X = 2**in_levels. Rows of B are localised averaging kernels on the input
  grid, so the input sensitivity is concentrated in a sub-interval and input
  DIS differs visibly from input PCA (as in the paper, Sec. 4.4.1).

Prior
-----
    C_X = (c * L_h + kappa * I)^{-1},   L_h = 1D Dirichlet Laplacian (FD, h = 1/(d_X+1))

i.e. the (-c Delta + kappa)^{-1} family used by both 1D problems in
Luo et al. (arXiv:2504.08730, Sec. 4.1). Dense eigendecomposition gives exact
C^{1/2}, C^{-1/2} and KL modes; fine for d_X <= ~2^11.

Everything here is dense and exact on purpose: it is the ground truth the
DIS-TT implementation is tested against.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LaplacePrior:
    """Gaussian prior N(0, C) with C = (c L_h + kappa I)^{-1} on a uniform 1D grid.

    Implements the ``GaussianPrior`` protocol of :mod:`dis_tt.interfaces` with
    the symmetric factor L = C^{1/2}.
    """

    dim: int
    c: float = 0.1
    kappa: float = 1.0
    _mu: np.ndarray = field(init=False, repr=False)  # KL eigenvalues, descending
    _phi: np.ndarray = field(init=False, repr=False)  # KL modes, columns, orthonormal

    def __post_init__(self) -> None:
        h = 1.0 / (self.dim + 1)
        lap = (2.0 * np.eye(self.dim) - np.eye(self.dim, k=1) - np.eye(self.dim, k=-1)) / h**2
        prec = self.c * lap + self.kappa * np.eye(self.dim)
        lam, q = np.linalg.eigh(prec)  # ascending precision eigenvalues
        self._mu = 1.0 / lam  # descending covariance eigenvalues
        self._phi = q

    # --- GaussianPrior protocol -------------------------------------------
    def covariance(self) -> np.ndarray:
        return (self._phi * self._mu) @ self._phi.T

    def apply_sqrt(self, v: np.ndarray) -> np.ndarray:
        """L v with L = C^{1/2} (symmetric square root). v: (dim,) or (dim, k)."""
        return self._phi @ (np.sqrt(self._mu)[:, None] * (self._phi.T @ _as2d(v))).reshape(_shape(v))

    def apply_sqrt_T(self, v: np.ndarray) -> np.ndarray:
        """L^T v (= L v here, L symmetric)."""
        return self.apply_sqrt(v)

    def apply_inv_sqrt(self, v: np.ndarray) -> np.ndarray:
        """L^{-1} v."""
        return self._phi @ ((1.0 / np.sqrt(self._mu))[:, None] * (self._phi.T @ _as2d(v))).reshape(_shape(v))

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """n draws, shape (dim, n)."""
        return self.apply_sqrt(rng.standard_normal((self.dim, n)))

    def kl_modes(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Top-k KL pairs (mu_j, phi_j); phi has shape (dim, k)."""
        return self._mu[:k].copy(), self._phi[:, :k].copy()

    @property
    def white_dim(self) -> int:
        """Dimension of the white-noise space (columns of L)."""
        return self.dim


@dataclass
class ToyProblem:
    """F(x) = A tanh(B x + b0) with exact JVP/VJP; see module docstring.

    Implements the ``ForwardModel`` protocol of :mod:`dis_tt.interfaces`.
    """

    out_levels: int = 8
    in_levels: int = 7
    p: int = 10
    seed: int = 0
    prior: LaplacePrior = field(init=False)
    A: np.ndarray = field(init=False, repr=False)
    B: np.ndarray = field(init=False, repr=False)
    b0: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        d_y, d_x = 2**self.out_levels, 2**self.in_levels
        t = (np.arange(d_y) + 0.5) / d_y
        cols = []
        for j in range(self.p):
            if j % 2 == 0:  # smooth global mode, decaying amplitude
                cols.append(np.sin((j // 2 + 1) * np.pi * t) * 0.7 ** j)
            else:  # localised bump
                c0 = rng.uniform(0.15, 0.85)
                cols.append(np.exp(-((t - c0) / 0.05) ** 2) * 0.7 ** j)
        self.A = np.stack(cols, axis=1)  # (d_Y, p)

        s = (np.arange(1, d_x + 1)) / (d_x + 1)
        rows = []
        for j in range(self.p):
            c0 = 0.55 + 0.35 * rng.uniform()  # sensitivity concentrated in (0.55, 0.9)
            w = np.exp(-((s - c0) / 0.04) ** 2)
            rows.append(3.0 * w / w.sum() * np.sqrt(d_x))
        self.B = np.stack(rows, axis=0)  # (p, d_X)
        self.b0 = rng.uniform(-0.5, 0.5, self.p)
        self.prior = LaplacePrior(d_x)

    # --- sizes ------------------------------------------------------------
    @property
    def d_y(self) -> int:
        return 2**self.out_levels

    @property
    def d_x(self) -> int:
        return 2**self.in_levels

    @property
    def out_modes(self) -> tuple[int, ...]:
        return (2,) * self.out_levels

    @property
    def in_modes(self) -> tuple[int, ...]:
        return (2,) * self.in_levels

    # --- ForwardModel protocol -------------------------------------------
    def solve(self, x: np.ndarray) -> dict:
        """Forward solve; the returned state carries the linearisation point."""
        u = self.B @ x + self.b0
        return {"x": x, "u": u, "y": self.A @ np.tanh(u)}

    def output(self, state: dict) -> np.ndarray:
        return state["y"]

    def jvp(self, state: dict, h: np.ndarray) -> np.ndarray:
        """DF(x) h; h shape (d_X,) or (d_X, k)."""
        s2 = 1.0 / np.cosh(state["u"]) ** 2
        return self.A @ (_bcast(s2, h) * (self.B @ h))

    def vjp(self, state: dict, w: np.ndarray) -> np.ndarray:
        """DF(x)^T w; w shape (d_Y,) or (d_Y, k)."""
        s2 = 1.0 / np.cosh(state["u"]) ** 2
        return self.B.T @ (_bcast(s2, w) * (self.A.T @ w))

    # --- test-only references (never call from library code) -------------
    def jacobian(self, state: dict) -> np.ndarray:
        """Dense DF(x), shape (d_Y, d_X)."""
        s2 = 1.0 / np.cosh(state["u"]) ** 2
        return self.A @ (s2[:, None] * self.B)

    def H_Y_reference(self, xs: np.ndarray) -> np.ndarray:
        """Empirical H_Y = (1/N) sum DF C DF^T over columns of xs (dense)."""
        C = self.prior.covariance()
        H = np.zeros((self.d_y, self.d_y))
        for x in xs.T:
            J = self.jacobian(self.solve(x))
            H += J @ C @ J.T
        return H / xs.shape[1]

    def H_X_white_reference(self, xs: np.ndarray) -> np.ndarray:
        """Empirical whitened H_X^w = (1/N) sum L^T DF^T DF L (dense)."""
        L = self.prior.apply_sqrt(np.eye(self.d_x))
        H = np.zeros((self.d_x, self.d_x))
        for x in xs.T:
            JL = self.jacobian(self.solve(x)) @ L
            H += JL.T @ JL
        return H / xs.shape[1]


def _as2d(v: np.ndarray) -> np.ndarray:
    return v[:, None] if v.ndim == 1 else v


def _shape(v: np.ndarray) -> tuple[int, ...]:
    return v.shape


def _bcast(s: np.ndarray, v: np.ndarray) -> np.ndarray:
    return s if v.ndim == 1 else s[:, None]
