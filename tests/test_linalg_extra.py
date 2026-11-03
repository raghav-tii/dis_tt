"""T2 extra tests: all-zero input, k > m, randomized tails via trace, rank caps."""
import numpy as np
import pytest

from dis_tt.linalg import select_rank, top_eigh_gram


@pytest.mark.parametrize("delta2", [None, 0.0, 1.0])
def test_select_rank_all_zero(delta2):
    r, tail = select_rank(np.zeros(4), delta2, None)
    assert r == 1 and tail == 0.0
    r, tail = select_rank(np.zeros(4), delta2, None, trace=0.0)
    assert r == 1 and tail == 0.0


def test_select_rank_rmax_and_rfixed_with_trace():
    eigs = np.array([5.0, 3.0, 1.0])
    r, tail = select_rank(eigs, 0.0, 2, trace=10.0)  # tail never <= 0 -> full, then capped
    assert (r, tail) == (2, pytest.approx(2.0))
    r, tail = select_rank(eigs, None, None, rfixed=1, trace=10.0)
    assert (r, tail) == (1, pytest.approx(5.0))
    # trace smaller than the top sum by round-off -> tail clipped to 0
    r, tail = select_rank(eigs, 0.0, None, trace=9.0 - 1e-15)
    assert r == 3 and tail == 0.0


@pytest.mark.parametrize("backend", ["gram", "svd", "auto", "randomized"])
def test_top_eigh_all_zero(backend):
    lam, V, complete = top_eigh_gram(np.zeros((5, 7)), k=3, backend=backend)
    assert np.all(lam == 0) and V.shape == (5, lam.size)
    np.testing.assert_allclose(V.T @ V, np.eye(V.shape[1]), atol=1e-12)


@pytest.mark.parametrize("backend", ["gram", "svd", "auto"])
@pytest.mark.parametrize("shape", [(6, 3), (4, 9)])
def test_top_eigh_k_larger_than_m(backend, shape):
    m = shape[0]
    M = np.random.default_rng(2).standard_normal(shape)
    lam, V, complete = top_eigh_gram(M, k=m + 5, backend=backend)
    assert lam.size == m and V.shape == (m, m) and complete
    np.testing.assert_allclose(V.T @ V, np.eye(m), atol=1e-10)
    np.testing.assert_allclose(lam, np.clip(np.linalg.eigvalsh(M @ M.T)[::-1], 0, None), atol=1e-10)
    assert np.all(lam >= 0)


def test_top_eigh_k_truncated_marks_incomplete():
    M = np.random.default_rng(3).standard_normal((6, 10))
    lam, V, complete = top_eigh_gram(M, k=2, backend="gram")
    assert lam.size == 2 and V.shape == (6, 2) and not complete


def test_randomized_tail_via_trace_is_exact_sum():
    rng = np.random.default_rng(4)
    M = rng.standard_normal((30, 4)) @ rng.standard_normal((4, 80)) + 1e-3 * rng.standard_normal((30, 80))
    lam, V, complete = top_eigh_gram(M, k=4, backend="randomized", rng=rng, power_iters=2)
    assert not complete
    ref = np.linalg.eigvalsh(M @ M.T)[::-1]
    r, tail = select_rank(lam, 1.0, None, trace=float(np.sum(M * M)))
    assert r == 4
    assert tail == pytest.approx(ref[4:].sum(), rel=1e-6)


def test_randomized_k_larger_than_m():
    M = np.random.default_rng(5).standard_normal((5, 20))
    lam, V, complete = top_eigh_gram(M, k=12, backend="randomized")
    assert not complete and lam.size == 5
    np.testing.assert_allclose(lam, np.linalg.eigvalsh(M @ M.T)[::-1], rtol=1e-10)
