"""T2 acceptance: rank rule and bond eigensolvers."""
import numpy as np
import pytest

from dis_tt.linalg import select_rank, top_eigh_gram

EIGS = np.array([10.0, 4.0, 1.0, 0.5, 0.1, 0.0])


@pytest.mark.parametrize("delta2,rmax,expected", [
    (0.6, None, 3),      # tail_3 = 0.6 <= 0.6
    (0.59, None, 4),     # tail_4 = 0.1
    (0.0, None, 5),      # tail_5 = 0
    (100.0, None, 1),    # r >= 1
    (0.0, 2, 2),         # rmax cap
])
def test_select_rank_tolerance(delta2, rmax, expected):
    r, tail = select_rank(EIGS, delta2, rmax)
    assert r == expected
    assert tail == pytest.approx(EIGS[r:].sum())


def test_select_rank_fixed_and_full():
    assert select_rank(EIGS, 1e9, 1, rfixed=4)[0] == 4
    assert select_rank(EIGS, None, None, rfixed=99)[0] == len(EIGS)
    assert select_rank(EIGS, None, None)[0] == 5  # exact zero dropped


def test_select_rank_with_trace_and_negative_roundoff():
    top = EIGS[:3]
    r, tail = select_rank(top, 0.7, None, trace=EIGS.sum())
    assert (r, tail) == (3, pytest.approx(0.6))
    r, tail = select_rank(np.array([1.0, 1e-3, -1e-17]), None, None)
    assert r == 2 and tail >= 0


@pytest.mark.parametrize("backend", ["gram", "svd", "auto"])
@pytest.mark.parametrize("shape", [(6, 40), (40, 6)])
def test_top_eigh_exact_backends(backend, shape):
    rng = np.random.default_rng(0)
    M = rng.standard_normal(shape)
    lam, V, complete = top_eigh_gram(M, backend=backend)
    ref = np.linalg.eigvalsh(M @ M.T)[::-1]
    assert complete
    np.testing.assert_allclose(lam[: min(shape)], ref[: min(shape)], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(V.T @ V, np.eye(V.shape[1]), atol=1e-10)
    k = min(shape)
    np.testing.assert_allclose((M @ M.T) @ V[:, :k], V[:, :k] * lam[:k], atol=1e-8)


def test_top_eigh_randomized_low_rank():
    rng = np.random.default_rng(1)
    M = rng.standard_normal((50, 5)) @ rng.standard_normal((5, 300))
    lam, V, complete = top_eigh_gram(M, k=5, backend="randomized", rng=rng)
    assert not complete
    ref = np.linalg.eigvalsh(M @ M.T)[::-1][:5]
    np.testing.assert_allclose(lam[:5], ref, rtol=1e-8)
