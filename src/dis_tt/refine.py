"""Optional ALS refinement of a greedy DIS-TT basis (TASK T10, after everything else passes).

Doc Sec. 8, item 2: maximise Tr(Pi H_hat) over block-TT bases with fixed ranks
by alternating sweeps. Microstep at core k (others fixed, left part
left-orthogonal, right part right-orthogonal, block index carried by core k):
    G_k <- top eigenvectors of the projected Gram  E_k^T Z Z^T E_k,
    E_k = U_{<=k-1} (x) I_{n_k} (x) (right frame)
This is the multi-target DMRG/ALS eigenproblem. Each microstep cannot increase
||(I - Pi) Z||_F^2 -> test monotonicity, and that the final error is <= the
greedy one. The block index must be moved between cores with an SVD (as in
block-TT / DMRG-for-many-states) when sweeping.
"""
from __future__ import annotations

import numpy as np

from .interfaces import DISTTResult


def als_refine(result: DISTTResult, Z: np.ndarray, n_sweeps: int = 2) -> tuple[DISTTResult, list[float]]:
    """Return refined result and the training error after every microstep."""
    raise NotImplementedError("T10")
