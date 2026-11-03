"""Protocols and result containers shared by all algorithms (IMPLEMENTED: types only).

Conventions (binding for every module; see AGENTS.md section "Conventions")
--------------------------------------------------------------------------
* Dense vector <-> tensor: ``y.reshape(mode_sizes)`` in C order. Mode 1 is the
  slowest index; for QTT (n_k = 2) i_1 is the most significant bit.
* TT core k: ndarray of shape (r_{k-1}, n_k, r_k).
* Snapshot matrices are PRE-SCALED: Z has shape (d, T) and Z @ Z.T equals the
  empirical sensitivity operator (H_hat_Y, H_hat_X^w or the mixed H_alpha).
  All weights 1/sqrt(N S) etc. are applied by :mod:`dis_tt.snapshots`, never by
  the compression routines.
* Whitening uses a factor L with C_X = L L^T (L = C^{1/2} for the toy prior).
  Whitened input coordinates: xi = L^{-1} x. Physical input basis: V = L W.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np


# --------------------------------------------------------------------------
# Model / prior protocols
# --------------------------------------------------------------------------
@runtime_checkable
class ForwardModel(Protocol):
    """Parameter-to-observable map F with linearised solves (dense vectors).

    ``solve`` returns an opaque state holding whatever the linearisation needs
    (e.g. the PDE solution and factorised Jacobian), so that several JVP/VJP
    calls at the same x reuse one nonlinear solve (doc Sec. 5).
    """

    def solve(self, x: np.ndarray) -> Any: ...
    def output(self, state: Any) -> np.ndarray: ...
    def jvp(self, state: Any, h: np.ndarray) -> np.ndarray: ...   # DF(x) h   (linearised forward solve)
    def vjp(self, state: Any, w: np.ndarray) -> np.ndarray: ...   # DF(x)^T w (adjoint solve)


@runtime_checkable
class GaussianPrior(Protocol):
    """N(0, C_X) with a factor L, C_X = L L^T. L may be rectangular (doc Sec. 8, item 3)."""

    white_dim: int

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray: ...   # (d_X, n)
    def apply_sqrt(self, v: np.ndarray) -> np.ndarray: ...                   # L v
    def apply_sqrt_T(self, v: np.ndarray) -> np.ndarray: ...                 # L^T v
    def apply_inv_sqrt(self, v: np.ndarray) -> np.ndarray: ...               # L^{-1} v; L^+ v = L^T C^{-1} v if L is wide
    def kl_modes(self, k: int) -> tuple[np.ndarray, np.ndarray]: ...        # (mu (k,), phi (d_X, k))


@runtime_checkable
class TTForwardModel(Protocol):
    """TT-native variant (Algorithm B / doc Sec. 5 'TT-native snapshots').

    Inputs and outputs are :class:`dis_tt.tt.TensorTrain`. jvp/vjp are linear
    TT solves (AMEn/ALS) with the Jacobian operator from the last Newton step.
    Implementations live outside this package (e.g. a QTT PDE solver); T9.
    """

    def solve(self, x: Any) -> Any: ...
    def output(self, state: Any) -> Any: ...
    def jvp(self, state: Any, h: Any, tol: float) -> Any: ...
    def vjp(self, state: Any, w: Any, tol: float) -> Any: ...


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
@dataclass
class DISTTResult:
    """Output of the greedy DIS-TT / PCA-TT sweep (Algorithms A and B).

    Attributes
    ----------
    cores : list of left-orthogonal cores G_1..G_D, G_k shape (r_{k-1}, n_k, r_k),
        with reshape(G_k, (r_{k-1} n_k, r_k)) having orthonormal columns.
        D = d   -> r-column block-TT basis U (last core's r_D = number of columns).
        D = d-1 -> TT subspace U_{<=d-1} (x) I_{n_d}.
    mode_sizes : (n_1, ..., n_d) of the FULL tensor (length d even if D = d-1).
    tails : (D,) discarded eigenvalue mass per bond, tail_k = sum_{i>r_k} lambda_i(H~_k).
    spectra : per bond, ALL eigenvalues of the projected operator H~_k, descending
        (diagnostics; length r_{k-1} n_k, possibly truncated by the backend - then
        ``spectra_complete[k] = False``).
    total : Tr(H_hat) = ||Z||_F^2.
    spectra_complete : per bond, whether ``spectra[k]`` is the full spectrum.
    """

    cores: list[np.ndarray]
    mode_sizes: tuple[int, ...]
    tails: np.ndarray
    spectra: list[np.ndarray]
    total: float
    spectra_complete: list[bool] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def D(self) -> int:
        return len(self.cores)

    @property
    def ranks(self) -> tuple[int, ...]:
        """(r_1, ..., r_D)."""
        return tuple(int(G.shape[2]) for G in self.cores)

    @property
    def cert(self) -> float:
        """Certified empirical relative squared error  sum_k tail_k / total  (doc Sec. 4, 'Exact training error')."""
        return float(np.sum(self.tails) / self.total) if self.total > 0 else 0.0

    @property
    def is_basis(self) -> bool:
        """True if D = d (r-column basis), False if TT subspace (D = d-1)."""
        return self.D == len(self.mode_sizes)


@dataclass
class SnapshotSet:
    """Pre-scaled snapshot matrix plus provenance (built by :mod:`dis_tt.snapshots`)."""

    Z: np.ndarray                 # (d, T), Z Z^T = target operator estimate
    kind: str                     # "dis_out", "pca_out", "dis_in", "mixed"
    n_samples: int                # N
    n_dirs: int                   # S (or K, or d_X, or r_Y)
    direction_kind: str = ""      # "random", "kl", "exact", "gaussian", "basis"
    meta: dict = field(default_factory=dict)


@dataclass
class BoundReport:
    """Numbers for the doc's 'Bound check' experiment (Prop. 1 / Prop. 2)."""

    train_error: float            # ||(I-P) Z_train||_F^2  (must equal sum(tails))
    test_error: float             # ||(I-P) Z_test||_F^2
    upper_unprojected: float      # sum_k tail_{r_k}(H^{(k)})   (dense, small problems only)
    lower_unprojected: float      # max_k tail_{r_k}(H^{(k)})
    sum_tails: float
    per_bond_unprojected: np.ndarray
    meta: dict = field(default_factory=dict)


Snapshots = Sequence[Any]  # sequence of TensorTrain for Algorithm B
