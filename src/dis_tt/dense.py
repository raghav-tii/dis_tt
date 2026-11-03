"""Algorithm A: greedy DIS-TT / PCA-TT from a dense snapshot matrix (TASK T4).

Math (doc Sec. 3-4)
-------------------
Nested left frames U_{<=k} = (U_{<=k-1} (x) I_{n_k}) G_k, projectors
Pi_k = U_{<=k} U_{<=k}^T (x) I_{>k}. Telescoping
    ||(I - Pi_D) Z||_F^2 = sum_k Tr[(I - G_k G_k^T) H~_k],
    H~_k = W_k^T Tr_{>k}(Z Z^T) W_k,   W_k = U_{<=k-1} (x) I_{n_k}.
Greedy choice G_k = top-r_k eigenvectors of H~_k. On the training snapshots
the identity is EXACT:  ||(I - Pi_D) Z||_F^2 = sum_k tail_k  (Prop. 1).

Pseudocode (doc Sec. 6, Algorithm A, lines 7-19)
------------------------------------------------
    total  <- ||Z||_F^2
    delta2 <- eps^2 * total / D
    M <- reshape(Z, [n_1, n_2...n_d * T])            # r_0 = 1
    for k = 1..D:
        # M : (r_{k-1} n_k) x (n_{k+1}...n_d * T)
        (Lam, V) <- top eigenpairs of M M^T          # projected partial trace H~_k
        r_k <- select_rank(Lam, delta2, rmax)
        G_k <- V[:, :r_k];  tail_k <- sum_{i>r_k} Lam_i
        M <- G_k^T M                                 # r_k x (n_{k+1}...n_d * T)
        if k < D: M <- reshape(M, [r_k n_{k+1}, n_{k+2}...n_d * T])

RESHAPE TRAP: Z has shape (d_Y, T) with the snapshot index LAST. The tensor
view is (n_1, ..., n_d, T). The bond-k matrix must have rows (alpha_{k-1}, i_k)
and columns (i_{k+1}, ..., i_d, tau) -- i.e. reshape Z to (n_1, rest) first,
NOT Z.T. Keep tau as the trailing (fastest) column index throughout.

Complexity (doc Sec. 7): Gram backend O(d_Y T (n + r^2)) + O(d r^3 n^3);
randomized backend O((r+p) d_Y T (1 + r/n)); memory O(d_Y T).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .interfaces import DISTTResult
from .linalg import select_rank, top_eigh_gram


def dis_tt_dense(Z: np.ndarray, mode_sizes: Sequence[int], *,
                 eps: float | None = None, rmax: int | None = None,
                 ranks: Sequence[int] | None = None, D: int | None = None,
                 backend: str = "auto", oversample: int = 10, power_iters: int = 1,
                 rng: np.random.Generator | None = None,
                 precompress: bool = False) -> DISTTResult:
    """Greedy left-to-right DIS-TT of a pre-scaled snapshot matrix.

    Parameters
    ----------
    Z : (prod(mode_sizes), T) pre-scaled snapshots (Z Z^T = H_hat).
        Pass JVP snapshots for DIS-TT, centred outputs for PCA-TT, a
        concatenation for the mixed objective. The routine is agnostic.
    mode_sizes : (n_1, ..., n_d).
    eps : relative tolerance on sqrt(cert); per-bond delta2 = eps^2 * total / D.
    rmax : rank cap for every bond.
    ranks : fixed (r_1, ..., r_D); overrides eps/rmax.
    D : d (default, r-column basis; r_d = ranks[-1] or eps-chosen) or d-1
        (TT subspace).
    backend : see :func:`dis_tt.linalg.top_eigh_gram`.
    precompress : replace Z by Q Sigma from a thin SVD first (doc Alg. A line 6);
        only worth it when T > rank(Z). Must not change any output.

    Returns
    -------
    DISTTResult with cores, tails, spectra, total.
    """
    Z = np.asarray(Z, dtype=float)
    modes = tuple(int(n) for n in mode_sizes)
    d = len(modes)
    if Z.ndim == 1:
        Z = Z[:, None]
    if Z.shape[0] != int(np.prod(modes)):
        raise ValueError(f"Z has {Z.shape[0]} rows, prod(mode_sizes) = {int(np.prod(modes))}")
    if D is None:
        D = len(ranks) if ranks is not None else d
    D = int(D)
    if not 1 <= D <= d:
        raise ValueError(f"D must lie in 1..{d}, got {D}")
    if ranks is not None and len(ranks) != D:
        raise ValueError(f"len(ranks) = {len(ranks)} != D = {D}")

    total = float(np.sum(Z * Z))
    delta2 = None if (eps is None or ranks is not None) else float(eps) ** 2 * total / D
    rng = np.random.default_rng(0) if rng is None else rng

    T_in = Z.shape[1]
    if precompress:
        Z = _precompress(Z)

    cores: list[np.ndarray] = []
    tails, spectra, complete_flags = [], [], []
    M = Z.reshape(modes[0], -1)  # rows i_1, columns (i_2..i_d, tau); r_0 = 1
    r_prev = 1
    for k in range(D):
        rfixed = None if ranks is None else int(ranks[k])
        if backend == "randomized":
            lam, V, r, tail = _randomized_bond(M, delta2, rmax, rfixed, oversample, power_iters, rng)
            complete = False
        else:
            lam, V, complete = top_eigh_gram(M, backend=backend)
            r, tail = select_rank(lam, delta2, rmax, rfixed=rfixed)
        G = V[:, :r]
        cores.append(G.reshape(r_prev, modes[k], r))
        tails.append(tail)
        spectra.append(lam)
        complete_flags.append(bool(complete))
        M = G.T @ M  # (r_k, n_{k+1}..n_d * T)
        if k < D - 1:
            M = M.reshape(r * modes[k + 1], -1)
        r_prev = r

    return DISTTResult(cores=cores, mode_sizes=modes, tails=np.asarray(tails, dtype=float),
                       spectra=spectra, total=total, spectra_complete=complete_flags,
                       meta={"backend": backend, "eps": eps, "rmax": rmax, "D": D,
                             "precompress": precompress, "T_in": T_in, "T_used": Z.shape[1]})


def _precompress(Z: np.ndarray) -> np.ndarray:
    """Q Sigma from a thin SVD of Z (same Z Z^T), keeping the numerical rank; no-op if it does not shrink T."""
    if Z.shape[1] <= Z.shape[0]:
        return Z  # rank(Z) <= T already; replacing Z cannot reduce the column count meaningfully
    U, s, _ = np.linalg.svd(Z, full_matrices=False)
    if s.size == 0 or s[0] == 0.0:
        return Z[:, :1] * 0.0
    keep = int(np.sum(s > s[0] * np.finfo(float).eps * max(Z.shape)))
    return U[:, :keep] * s[:keep]


def _randomized_bond(M, delta2, rmax, rfixed, oversample, power_iters, rng):
    """Randomized eigenpairs of M M^T; tails via trace(M M^T) - sum(top), exact as a sum.

    Fixed rank -> one sketch of size rfixed. Tolerance mode -> double the number of requested
    pairs until the trace-based tail meets delta2, or rmax / min(m, c) is reached.
    """
    m, c = M.shape
    trace = float(np.sum(M * M))
    kmax = min(m, c) if rmax is None else min(int(rmax), m, c)
    if rfixed is not None:
        k = max(1, min(int(rfixed), m))
    elif delta2 is None:
        k = max(1, kmax)
    else:
        k = max(1, min(4, kmax))
    while True:
        lam, V, _ = top_eigh_gram(M, k=k, backend="randomized", oversample=oversample,
                                  power_iters=power_iters, rng=rng)
        if lam.size == 0:  # c = 0 cannot happen for valid input; keep r >= 1 contract
            lam, V = np.zeros(1), np.eye(m, 1)
        r, tail = select_rank(lam, delta2, rmax, rfixed=rfixed, trace=trace)
        if rfixed is not None or delta2 is None or tail <= delta2 or k >= kmax:
            return lam, V, r, tail
        k = min(2 * k, kmax)
