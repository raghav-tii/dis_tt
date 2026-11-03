"""Small linear-algebra kernels used at every bond (TASK T2)."""
from __future__ import annotations

import numpy as np


def select_rank(eigs_desc: np.ndarray, delta2: float | None, rmax: int | None,
                rfixed: int | None = None, trace: float | None = None) -> tuple[int, float]:
    """Rank rule of doc Sec. 4 ('Rank selection'). Returns (r, tail_r).

    tail_r = sum_{i>r} eigs_desc[i], or trace - sum_{i<=r} eigs_desc[i] when
    ``trace`` is given (needed for the randomized backend, whose spectrum is
    incomplete). Clip negative round-off eigenvalues and negative tails to 0.

    r = smallest value with tail_r <= delta2, then capped by rmax and by
    len(eigs_desc); r >= 1 always. ``rfixed`` overrides delta2/rmax (still capped
    by len(eigs_desc)). delta2 = rfixed = None -> full rank minus numerically zero
    eigenvalues (< 1e-14 * eigs_desc[0]).
    """
    raise NotImplementedError("T2")


def top_eigh_gram(M: np.ndarray, k: int | None = None, backend: str = "gram",
                  oversample: int = 10, power_iters: int = 1,
                  rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray, bool]:
    """Leading eigenpairs of M M^T without forming it when that is cheaper.

    Parameters
    ----------
    M : (m, c) matrix (the bond-k working matrix of Algorithm A, m = r_{k-1} n_k).
    k : number of pairs wanted (None = all m).
    backend :
        "gram"       eigh(M @ M.T); cost O(m^2 c + m^3). Exact full spectrum.
        "svd"        thin SVD of M; cost O(m c min(m, c)). Exact full spectrum.
        "auto"       "gram" if c >= m else "svd".
        "randomized" Halko-Martinsson-Tropp range finder with k + oversample
                     columns and ``power_iters`` power iterations; cost
                     O((k+p) m c). Spectrum NOT complete -> tails must be
                     computed as trace(M M^T) - sum(top eigenvalues), which is
                     exact for the tail SUM even though individual tail
                     eigenvalues are unknown.

    Returns
    -------
    eigvals (descending), eigvecs (m, len(eigvals)), complete (bool).
    """
    raise NotImplementedError("T2")
