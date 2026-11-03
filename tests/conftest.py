"""Shared fixtures and INDEPENDENT numpy references (never import the code under test here).

The reference helpers below also pin down the index conventions: if an
implementation disagrees with ``ref_left_frame`` it has the wrong Kronecker
ordering, not the test.
"""
from __future__ import annotations

import numpy as np
import pytest

from dis_tt.problems.toy import ToyProblem


# ---------------------------------------------------------------- fixtures
@pytest.fixture(scope="session")
def toy() -> ToyProblem:
    # d_Y = 2^7 = 128, d_X = 2^6 = 64, p = 8
    return ToyProblem(out_levels=7, in_levels=6, p=8, seed=1)


@pytest.fixture(scope="session")
def xs_train(toy) -> np.ndarray:
    return toy.prior.sample(np.random.default_rng(10), 40)


@pytest.fixture(scope="session")
def xs_test(toy) -> np.ndarray:
    return toy.prior.sample(np.random.default_rng(11), 40)


@pytest.fixture(scope="session")
def Z_exact(toy, xs_train) -> np.ndarray:
    """Exact-column output DIS snapshots: Z Z^T = H_Y_hat (pre-scaled)."""
    return ref_output_dis_exact(toy, xs_train)


@pytest.fixture(scope="session")
def Z_exact_test(toy, xs_test) -> np.ndarray:
    return ref_output_dis_exact(toy, xs_test)


# ---------------------------------------------------------------- references
def ref_output_dis_exact(toy: ToyProblem, xs: np.ndarray) -> np.ndarray:
    L = toy.prior.apply_sqrt(np.eye(toy.d_x))
    cols = [toy.jacobian(toy.solve(x)) @ L for x in xs.T]
    return np.concatenate(cols, axis=1) / np.sqrt(xs.shape[1])


def ref_left_frame(cores) -> np.ndarray:
    """U_{<=D} with rows (i_1, ..., i_D) in C order (i_1 slowest)."""
    U = cores[0].reshape(cores[0].shape[1], cores[0].shape[2])  # r_0 = 1
    for G in cores[1:]:
        U = np.einsum("Ia,air->Iir", U, G).reshape(-1, G.shape[2])
    return U


def ref_subspace(cores, mode_sizes) -> np.ndarray:
    """Orthonormal basis of range(Pi_D) = U_{<=D} (x) I_{rest}."""
    U = ref_left_frame(cores)
    rest = int(np.prod(mode_sizes[len(cores):])) if len(cores) < len(mode_sizes) else 1
    return np.kron(U, np.eye(rest))


def ref_unfold_gram(Z: np.ndarray, mode_sizes, k: int) -> np.ndarray:
    """H^{(k)} = Tr_{>k}(Z Z^T): Gram of left unfolding k of tensor (n_1..n_d, T)."""
    M = Z.reshape(int(np.prod(mode_sizes[:k])), -1)
    return M @ M.T


def ref_tail(eigs_desc: np.ndarray, r: int) -> float:
    return float(np.clip(eigs_desc[r:], 0, None).sum())


def eig_desc(H: np.ndarray) -> np.ndarray:
    return np.linalg.eigvalsh(H)[::-1]


def proj_err(B: np.ndarray, Z: np.ndarray) -> float:
    """||(I - B B^T) Z||_F^2 for orthonormal B."""
    R = Z - B @ (B.T @ Z)
    return float(np.sum(R * R))


def max_principal_angle_sin(A: np.ndarray, B: np.ndarray) -> float:
    """sin of the largest principal angle between ranges of orthonormal A and B (same #cols)."""
    return float(np.linalg.norm(B - A @ (A.T @ B), 2))  # = sin(theta_max), no cancellation


def left_orth_defect(cores) -> float:
    out = 0.0
    for G in cores:
        M = G.reshape(-1, G.shape[2])
        out = max(out, float(np.linalg.norm(M.T @ M - np.eye(M.shape[1]), 2)))
    return out
