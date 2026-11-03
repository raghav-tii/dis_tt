"""T1 extra tests: n_k != 2, d = 1, rank caps, zero input, rounding error bound."""
import numpy as np
import pytest
from conftest import ref_left_frame, ref_subspace

from dis_tt import tt


def _rand(n, seed=0):
    return np.random.default_rng(seed).standard_normal(n)


@pytest.mark.parametrize("modes", [(3, 5, 4), (7,), (2, 1, 3), (5, 5)])
def test_roundtrip_mixed_modes(modes):
    y = _rand(int(np.prod(modes)), 1)
    T = tt.from_dense(y, modes)
    assert T.mode_sizes == modes and T.ranks[0] == T.ranks[-1] == 1
    np.testing.assert_allclose(tt.to_dense(T), y, atol=1e-12)
    R = tt.right_orthogonalize(T)
    np.testing.assert_allclose(tt.to_dense(R), y, atol=1e-12)
    assert np.isclose(np.linalg.norm(R.cores[0]), np.linalg.norm(y))


def test_d1_operations():
    y = _rand(6, 2)
    T = tt.from_dense(y, (6,), eps=0.5)  # d = 1: no bonds to truncate, exact
    assert T.ranks == (1, 1)
    np.testing.assert_allclose(tt.to_dense(T), y)
    np.testing.assert_allclose(tt.to_dense(tt.tt_add(T, T, 1.0, 2.0)), 3 * y)
    np.testing.assert_allclose(tt.to_dense(tt.tt_round(T, 0.1)), y)
    Q, _ = np.linalg.qr(np.random.default_rng(3).standard_normal((6, 3)))
    cores = [Q.reshape(1, 6, 3)]
    np.testing.assert_allclose(tt.project_block(cores, T), Q.T @ y, atol=1e-13)
    np.testing.assert_allclose(tt.subspace_matrix(cores, (6,)), Q)
    np.testing.assert_allclose(tt.to_dense(tt.block_apply(cores, np.ones(3))), Q.sum(1))


def test_rank_cap_binding_and_optimal_truncation():
    modes = (4, 3, 5)
    y = _rand(60, 4)
    T = tt.from_dense(y, modes, rmax=2)
    assert max(T.ranks) <= 2
    # first bond of TT-SVD with rmax is the best rank-2 approx of the first unfolding
    s = np.linalg.svd(y.reshape(4, 15), compute_uv=False)
    err1 = np.linalg.norm(tt.to_dense(tt.from_dense(y, (4, 15), rmax=2)) - y)
    assert np.isclose(err1, np.linalg.norm(s[2:]))
    R = tt.tt_round(tt.from_dense(y, modes), 0.0, rmax=2)
    assert max(R.ranks) <= 2


def test_round_error_bound_and_rank_reduction():
    modes = (3, 4, 2, 5)
    y = _rand(120, 5)
    T = tt.from_dense(y, modes)
    S = tt.tt_add(T, T, 1.0, 1.0)
    assert max(S.ranks) == 2 * max(T.ranks)
    for eps in (0.3, 0.05):
        R = tt.tt_round(S, eps)
        assert np.linalg.norm(tt.to_dense(R) - 2 * y) <= eps * np.linalg.norm(2 * y) * (1 + 1e-10)
    assert tt.tt_round(S, 1e-12).ranks == T.ranks


def test_zero_vector():
    T = tt.from_dense(np.zeros(24), (2, 3, 4), eps=1e-3)
    assert all(r == 1 for r in T.ranks)
    assert np.all(tt.to_dense(T) == 0)


def test_subspace_matrix_mixed_modes_against_reference():
    modes, ranks = (3, 5, 2, 4), (3, 4, 4)
    rng = np.random.default_rng(6)
    cores, r_prev = [], 1
    for n, r in zip(modes, ranks):
        Q, _ = np.linalg.qr(rng.standard_normal((r_prev * n, r)))
        cores.append(Q.reshape(r_prev, n, r))
        r_prev = r
    for D in (1, 2, 3):
        B = tt.subspace_matrix(cores[:D], modes)
        np.testing.assert_allclose(B, ref_subspace(cores[:D], modes), atol=1e-13)
        np.testing.assert_allclose(B.T @ B, np.eye(B.shape[1]), atol=1e-12)
    np.testing.assert_allclose(tt.block_to_dense(cores), ref_left_frame(cores), atol=1e-13)


def test_from_dense_rejects_wrong_size():
    with pytest.raises(ValueError):
        tt.from_dense(np.zeros(10), (2, 3))
