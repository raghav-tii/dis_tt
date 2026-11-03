"""1D semilinear elliptic problem of Luo et al. (arXiv:2504.08730, Sec. 4.1) by finite differences (TASK T9).

Paper statement (TeX source, ``sections/numerical_results.tex``, paragraph "Semilinear elliptic PDE")
-----------------------------------------------------------------------------------------------
    -(c_1(s) y'(s))' + c_2 y(s)^3 = x(s),  s in (0, 1),   y(0) = 0,   y'(1) = 0,
    c_1(s) = 0.0001 + 0.01 * chi_(0.5, 1)(s),   c_2 = 0.1,   x ~ N(0, (-2 Delta + 10)^{-1}).
The paper uses 256 linear finite elements (d_X = d_Y = 257, FEniCS/hIPPYlib).

This discretisation (deviations, see notes/T9.md)
--------------------------------------------------
* Finite volumes / FD on n = 2**levels unknowns s_j = j h, h = 1/n, j = 1..n (the Dirichlet node s = 0
  is eliminated; s = 1 carries the Neumann condition through a half cell). For levels = 8 this is the
  paper's 257-node mesh minus the Dirichlet node. c_1 is evaluated at cell midpoints s_{j-1/2}.
* Residual  R(y, x) = S y + m * (c_2 y^3 - x),  S symmetric tridiagonal (flux form), m_j = h (j < n),
  m_n = h/2. Newton with backtracking; Jacobian J_y = S + 3 c_2 diag(m y^2) is SPD.
  DF(x) = J_y^{-1} diag(m):   JVP = J_y^{-1}(m * h),   VJP = m * J_y^{-1} w   (J_y symmetric).
* Input and output are nodal vectors with the EUCLIDEAN inner product (identity mass), as the library
  assumes everywhere (theory caveat 6). The paper's spaces are L^2(0, 1) with the FE mass matrix.
* Prior: the paper gives C = (-2 Delta + 10)^{-1} without boundary conditions for Delta. Here Delta is the
  homogeneous-Neumann FD Laplacian on the same n nodes (an interpretation), and the NODAL covariance is
  C = A_h^{-1} / h with A_h = 2 K_h + 10 I, i.e. the Green's-function values G(s_i, s_j) (mesh-independent
  pointwise variance ~ 1 / (2 sqrt(20)) ~ 0.11). Factor L = C^{1/2} (symmetric, dense eigendecomposition).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import solveh_banded


@dataclass
class EllipticPrior:
    """N(0, C), C = (a_lap K_h + a_id I)^{-1} / h, K_h the Neumann FD Laplacian on n nodes; GaussianPrior protocol."""

    dim: int
    a_lap: float = 2.0
    a_id: float = 10.0
    _mu: np.ndarray = field(init=False, repr=False)
    _phi: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        n, h = self.dim, 1.0 / self.dim
        K = 2.0 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
        K[0, 0] = K[-1, -1] = 1.0  # homogeneous Neumann at both ends (interpretation)
        A = self.a_lap * K / h ** 2 + self.a_id * np.eye(n)
        lam, q = np.linalg.eigh(A)          # ascending precision eigenvalues
        self._mu = 1.0 / (lam * h)            # descending covariance eigenvalues
        self._phi = q

    def covariance(self) -> np.ndarray:
        return (self._phi * self._mu) @ self._phi.T

    def apply_sqrt(self, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=float)
        return (self._phi @ (np.sqrt(self._mu)[:, None] * (self._phi.T @ v.reshape(self.dim, -1)))).reshape(v.shape)

    def apply_sqrt_T(self, v: np.ndarray) -> np.ndarray:
        return self.apply_sqrt(v)

    def apply_inv_sqrt(self, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=float)
        return (self._phi @ ((1.0 / np.sqrt(self._mu))[:, None] * (self._phi.T @ v.reshape(self.dim, -1)))).reshape(v.shape)

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return self.apply_sqrt(rng.standard_normal((self.dim, n)))

    def kl_modes(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        return self._mu[:k].copy(), self._phi[:, :k].copy()

    @property
    def white_dim(self) -> int:
        return self.dim


@dataclass
class SemilinearProblem:
    """Paper Sec. 4.1 semilinear elliptic map x -> y on 2**levels FD unknowns; ForwardModel protocol."""

    levels: int = 8
    c2: float = 0.1
    diff_low: float = 1e-4
    diff_jump: float = 1e-2
    newton_tol: float = 1e-12
    newton_maxit: int = 50
    prior: EllipticPrior = field(init=False)

    def __post_init__(self) -> None:
        n = 2 ** self.levels
        self.h = 1.0 / n
        self.s = np.arange(1, n + 1) * self.h
        mid = (np.arange(1, n + 1) - 0.5) * self.h                       # s_{j-1/2}, j = 1..n
        c = self.diff_low + self.diff_jump * (mid > 0.5)
        self.c_mid = c
        main = np.empty(n)
        main[:-1] = (c[:-1] + c[1:]) / self.h                             # fluxes at j-1/2 and j+1/2
        main[-1] = c[-1] / self.h                                         # Neumann: only the left flux
        self._S_main, self._S_off = main, -c[1:] / self.h                 # off[j] couples j, j+1
        self.m = np.full(n, self.h)
        self.m[-1] = 0.5 * self.h
        self.prior = EllipticPrior(n)

    # --- sizes -------------------------------------------------------------
    @property
    def d_y(self) -> int:
        return 2 ** self.levels

    @property
    def d_x(self) -> int:
        return 2 ** self.levels

    @property
    def out_modes(self) -> tuple[int, ...]:
        return (2,) * self.levels

    @property
    def in_modes(self) -> tuple[int, ...]:
        return (2,) * self.levels

    # --- discrete operators --------------------------------------------------
    def _S_apply(self, y: np.ndarray) -> np.ndarray:
        out = self._S_main * y
        out[:-1] += self._S_off * y[1:]
        out[1:] += self._S_off * y[:-1]
        return out

    def residual(self, y: np.ndarray, x: np.ndarray) -> np.ndarray:
        return self._S_apply(y) + self.m * (self.c2 * y ** 3 - x)

    def _jac_banded(self, y: np.ndarray) -> np.ndarray:
        ab = np.zeros((2, y.size))                     # upper form for solveh_banded
        ab[0, 1:] = self._S_off
        ab[1] = self._S_main + 3.0 * self.c2 * self.m * y ** 2
        return ab

    # --- ForwardModel protocol ---------------------------------------------
    def solve(self, x: np.ndarray) -> dict:
        x = np.asarray(x, dtype=float)
        y = np.zeros_like(x)
        r = self.residual(y, x)
        r0 = np.linalg.norm(r)
        it = 0
        for it in range(1, self.newton_maxit + 1):
            dy = solveh_banded(self._jac_banded(y), -r)
            t = 1.0
            while True:  # backtracking on ||R||
                y_new = y + t * dy
                r_new = self.residual(y_new, x)
                if np.linalg.norm(r_new) <= (1 - 1e-4 * t) * np.linalg.norm(r) or t < 1e-8:
                    break
                t *= 0.5
            y, r = y_new, r_new
            if np.linalg.norm(r) <= self.newton_tol * max(r0, 1e-300) or np.linalg.norm(t * dy) <= 1e-15 * (1 + np.linalg.norm(y)):
                break
        else:
            raise RuntimeError(
                f"Newton failed to converge in {self.newton_maxit} iterations "
                f"(relative residual {np.linalg.norm(r) / max(r0, 1e-300):.6g})"
            )
        return {"x": x, "y": y, "ab": self._jac_banded(y), "newton_iters": it,
                "res_rel": float(np.linalg.norm(r) / max(r0, 1e-300))}

    def output(self, state: dict) -> np.ndarray:
        return state["y"]

    def jvp(self, state: dict, h: np.ndarray) -> np.ndarray:
        """DF(x) h = J_y^{-1} (m * h) (linearised forward solve); h (d_X,) or (d_X, k)."""
        h = np.asarray(h, dtype=float)
        m = self.m if h.ndim == 1 else self.m[:, None]
        return solveh_banded(state["ab"], m * h)

    def vjp(self, state: dict, w: np.ndarray) -> np.ndarray:
        """DF(x)^T w = m * J_y^{-1} w (adjoint solve, J_y symmetric); w (d_Y,) or (d_Y, k)."""
        w = np.asarray(w, dtype=float)
        m = self.m if w.ndim == 1 else self.m[:, None]
        return m * solveh_banded(state["ab"], w)

    # --- test / experiment references (dense; never used by library code) ---
    def jacobian(self, state: dict) -> np.ndarray:
        return self.jvp(state, np.eye(self.d_x))

    def H_Y_reference(self, xs: np.ndarray) -> np.ndarray:
        L = self.prior.apply_sqrt(np.eye(self.d_x))
        H = np.zeros((self.d_y, self.d_y))
        for x in xs.T:
            JL = self.jacobian(self.solve(x)) @ L
            H += JL @ JL.T
        return H / xs.shape[1]

    def H_X_white_reference(self, xs: np.ndarray) -> np.ndarray:
        L = self.prior.apply_sqrt(np.eye(self.d_x))
        H = np.zeros((self.d_x, self.d_x))
        for x in xs.T:
            JL = self.jacobian(self.solve(x)) @ L
            H += JL.T @ JL
        return H / xs.shape[1]
