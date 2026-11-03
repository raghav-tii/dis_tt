"""T1 acceptance: TT toolkit and index conventions."""
import numpy as np
import pytest
from conftest import ref_left_frame, ref_subspace

from dis_tt import tt

MODES = (2, 3, 2, 4, 2)


def _smooth_vec(modes, seed=0):
    n = int(np.prod(modes))
    t = np.linspace(0, 1, n)
    return np.sin(5 * t) + 0.3 * np.cos(17 * t) + 0.01 * np.random.default_rng(seed).standard_normal(n)


def _random_left_orth_cores(modes, ranks, seed=0):
    rng = np.random.default_rng(seed)
    cores, r_prev = [], 1
    for n, r in zip(modes, ranks):
        Q, _ = np.linalg.qr(rng.standard_normal((r_prev * n, r)))
        cores.append(Q.reshape(r_prev, n, r))
        r_prev = r
    return cores


def test_from_dense_roundtrip_exact():
    y = _smooth_vec(MODES)
    T = tt.from_dense(y, MODES)
    assert T.mode_sizes == MODES and T.ranks[0] == T.ranks[-1] == 1
    np.testing.assert_allclose(tt.to_dense(T), y, atol=1e-12)


def test_to_dense_c_order():
    """Rank-1 TT with e_{i_k} cores must give the C-order flat index."""
    idx = (1, 2, 0, 3, 1)
    cores = [np.eye(n)[i].reshape(1, n, 1) for n, i in zip(MODES, idx)]
    v = tt.to_dense(tt.TensorTrain(cores))
    assert np.argmax(v) == np.ravel_multi_index(idx, MODES) and v.sum() == 1


@pytest.mark.parametrize("eps", [1e-2, 1e-4])
def test_from_dense_tolerance(eps):
    y = _smooth_vec(MODES, 1)
    T = tt.from_dense(y, MODES, eps=eps)
    assert np.linalg.norm(tt.to_dense(T) - y) <= eps * np.linalg.norm(y) * (1 + 1e-10)


def test_block_to_dense_and_subspace_match_reference():
    cores = _random_left_orth_cores(MODES, (2, 4, 5, 3, 3))
    np.testing.assert_allclose(tt.block_to_dense(cores), ref_left_frame(cores), atol=1e-13)
    for D in (3, 5):
        np.testing.assert_allclose(tt.subspace_matrix(cores[:D], MODES), ref_subspace(cores[:D], MODES), atol=1e-13)
    assert tt.left_orthonormality_defect(cores) < 1e-12


def test_right_orthogonalize_and_partial_trace():
    y = _smooth_vec(MODES, 2)
    T = tt.right_orthogonalize(tt.from_dense(y, MODES))
    np.testing.assert_allclose(tt.to_dense(T), y, atol=1e-12)
    for G in T.cores[1:]:
        M = G.reshape(G.shape[0], -1)
        np.testing.assert_allclose(M @ M.T, np.eye(M.shape[0]), atol=1e-12)
    # Tr_{>k}(y y^T) = L_k L_k^T with L_k = contraction of cores 1..k
    for k in range(1, len(MODES)):
        Lk = ref_left_frame(T.cores[:k])
        Yk = y.reshape(int(np.prod(MODES[:k])), -1)
        np.testing.assert_allclose(Lk @ Lk.T, Yk @ Yk.T, atol=1e-11)


def test_project_block_apply_add_round():
    cores = _random_left_orth_cores(MODES, (2, 4, 5, 3, 3), seed=3)
    U = ref_left_frame(cores)
    y = _smooth_vec(MODES, 4)
    T = tt.from_dense(y, MODES)
    np.testing.assert_allclose(tt.project_block(cores, T), U.T @ y, atol=1e-11)
    c = np.array([1.0, -2.0, 0.5])
    np.testing.assert_allclose(tt.to_dense(tt.block_apply(cores, c)), U @ c, atol=1e-12)
    S = tt.tt_add(T, T, 2.0, -0.5)
    np.testing.assert_allclose(tt.to_dense(S), 1.5 * y, atol=1e-11)
    R = tt.tt_round(S, 1e-10)
    assert max(R.ranks) <= max(T.ranks)
    np.testing.assert_allclose(tt.to_dense(R), 1.5 * y, atol=1e-8)
