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
    e = np.clip(np.asarray(eigs_desc, dtype=float).ravel(), 0.0, None)
    n = e.size
    if n == 0:
        raise ValueError("select_rank needs at least one eigenvalue")
    if trace is None:
        # tails[r] = sum_{i>=r} e[i], r = 0..n (suffix sums, tails[n] = 0)
        tails = np.concatenate([np.cumsum(e[::-1])[::-1], [0.0]])
    else:
        tails = np.clip(float(trace) - np.concatenate([[0.0], np.cumsum(e)]), 0.0, None)
    if rfixed is not None:
        r = int(rfixed)
    else:
        if delta2 is None:
            r = int(np.sum(e > 1e-14 * e[0])) if e[0] > 0 else 1
        else:
            ok = np.nonzero(tails[1:] <= delta2)[0]
            r = int(ok[0]) + 1 if ok.size else n
        if rmax is not None:
            r = min(r, int(rmax))
    r = max(1, min(r, n))
    return r, float(tails[r])


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
    M = np.asarray(M, dtype=float)
    m, c = M.shape
    k = m if k is None else max(0, min(int(k), m))
    if backend == "auto":
        backend = "gram" if c >= m else "svd"
    if backend == "gram":
        lam, V = np.linalg.eigh(M @ M.T)
        lam, V = lam[::-1], V[:, ::-1]
        complete = True
    elif backend == "svd":
        full = k > min(m, c)  # need the null-space complement of range(M) as well
        U, s, _ = np.linalg.svd(M, full_matrices=full)
        lam = np.zeros(U.shape[1])
        lam[: s.size] = s * s
        V = U
        complete = True
    elif backend == "randomized":
        if k > min(m, c):
            # The requested subspace includes null directions. A range sketch
            # cannot supply these; complete the basis with the exact SVD path.
            lam, V, _ = top_eigh_gram(M, k=k, backend="svd")
            return lam, V, False
        rng = np.random.default_rng(0) if rng is None else rng
        ell = min(k + int(oversample), m, c) if c > 0 else 0
        if ell == 0:
            return np.zeros(0), np.zeros((m, 0)), False
        Q, _ = np.linalg.qr(M @ rng.standard_normal((c, ell)))
        for _ in range(int(power_iters)):
            Q, _ = np.linalg.qr(M.T @ Q)
            Q, _ = np.linalg.qr(M @ Q)
        Ub, s, _ = np.linalg.svd(Q.T @ M, full_matrices=False)
        lam, V = s * s, Q @ Ub
        k = min(k, lam.size)
        return lam[:k], V[:, :k], False
    else:
        raise ValueError(f"unknown backend {backend!r}")
    lam = np.clip(lam, 0.0, None)
    return lam[:k], V[:, :k], complete and k == m
