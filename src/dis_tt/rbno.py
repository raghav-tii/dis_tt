"""Algorithm D: RBNO training data with TT encoder/decoder (TASK T7).

Surrogate (paper Sec. 2.3):  F~(x) = y_bar + U_Y phi_theta(W^T L^{-1} x).
Sobolev loss (paper eq. 10), per sample s:
    ||eta_s - phi(xi_s)||^2 / a0_s + ||J_s - D phi(xi_s)||_F^2 / a1_s
with
    xi_s  = W^T L^{-1} x_s                (r_X,)
    eta_s = U_Y^T (y_s - y_bar)           (r_Y,)
    J_s   = U_Y^T DF(x_s) L W             (r_Y, r_X)   reduced whitened Jacobian
and the paper's per-sample normalisation a0_s = ||eta_s||^2, a1_s = ||J_s||_F^2
(Sec. 4.3; optional tau > 0 added to both).

Pseudocode (doc Sec. 6, Algorithm D)
------------------------------------
    for s:
        state <- solve(x_s); y_s <- output(state)
        xi_s <- encode(x_s); eta_s <- U_Y^T (y_s - y_bar)
        if r_X <= r_Y:  J_s[:, b] <- U_Y^T jvp(state, L W e_b),   b = 1..r_X
        else:           J_s[a, :] <- W^T L^T vjp(state, U_Y e_a), a = 1..r_Y
    -> min(r_X, r_Y) linearised solves per sample.

NOTE the transpose: the VJP branch needs L^T (doc pseudocode wrote C^{1/2},
which coincides only for symmetric L). Both branches must agree to 1e-10 on
the toy problem (test_rbno.py).

Complexity: N_train [C_fwd + min(r_X, r_Y) C_lin] solves; dense projections
O(N_train min(r_X, r_Y) d r); TT-native projections O(... d n r R (r+R)).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .input_side import WhitenedInputBasis
from .interfaces import ForwardModel


@dataclass
class RBNODataset:
    xi: np.ndarray       # (N, r_X)
    eta: np.ndarray      # (N, r_Y)
    J: np.ndarray        # (N, r_Y, r_X)
    y_bar: np.ndarray    # (d_Y,)
    meta: dict


def assemble_rbno_data(model: ForwardModel, U_out: np.ndarray, in_basis: WhitenedInputBasis,
                       xs: np.ndarray, y_bar: np.ndarray, branch: str = "auto") -> RBNODataset:
    """Build (xi, eta, J) for Sobolev training. branch in {"auto", "jvp", "vjp"}.

    U_out: dense orthonormal (d_Y, r_Y). A block-TT overload is T9 work.
    """
    raise NotImplementedError("T7")


def sobolev_loss(eta_pred: np.ndarray, J_pred: np.ndarray, data: RBNODataset,
                 normalise: str = "per_sample", tau: float = 0.0) -> float:
    """Paper eq. 10 averaged over samples. normalise in {"per_sample", "global", "none"}."""
    raise NotImplementedError("T7")


def train_latent_network(data: RBNODataset, **kwargs):
    """Fit phi_theta: R^{r_X} -> R^{r_Y} (paper Table 2: depth 4-6, width 2r, softplus, Adam).

    Framework is the implementer's choice (JAX or PyTorch; neither is a hard
    dependency of this package). Must use the exact Jacobian D phi via autodiff
    in the loss. Out of scope for the core tests; see AGENTS.md T8.
    """
    raise NotImplementedError("T8")
