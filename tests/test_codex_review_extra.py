"""Independent adversarial regressions for the completed core tasks."""
import numpy as np
import pytest

from dis_tt import tt, snapshots, bounds
from dis_tt.dense import dis_tt_dense
from dis_tt.linalg import top_eigh_gram


def test_randomized_fixed_rank_includes_null_complement():
    Z = np.arange(1., 9.)[:, None]
    result = dis_tt_dense(Z, (2, 4), ranks=(2, 5), backend="randomized")
    assert result.ranks == (2, 5)
    B = tt.block_to_dense(result.cores)
    np.testing.assert_allclose(B.T @ B, np.eye(5), atol=1e-12)
    assert np.linalg.norm(Z - B @ (B.T @ Z)) < 1e-12
    assert not any(result.spectra_complete)


def test_randomized_precompression_preserves_projector():
    Z = np.random.default_rng(7).normal(size=(30, 100))
    kw = dict(ranks=(3, 4), backend="randomized", oversample=0, power_iters=0)
    a = dis_tt_dense(Z, (5, 6), **kw)
    b = dis_tt_dense(Z, (5, 6), precompress=True, **kw)
    A, B = tt.block_to_dense(a.cores), tt.block_to_dense(b.cores)
    np.testing.assert_allclose(A @ A.T, B @ B.T, atol=1e-12)
    np.testing.assert_allclose(a.tails, b.tails, atol=1e-10)


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_tt_relative_tolerance_is_scale_invariant(scale):
    y = np.diag([1., .1, .01, .001]).ravel()
    with np.errstate(over="raise", invalid="raise"):
        a = tt.from_dense(scale * y, (4, 4), eps=.02)
        b = tt.tt_round(tt.from_dense(scale * y, (4, 4)), .02)
    for z in (a, b):
        assert z.ranks == (1, 2, 1)
        assert np.linalg.norm(tt.to_dense(z) / scale - y) <= .02 * np.linalg.norm(y)


@pytest.mark.parametrize("kw", [dict(n_boot=0), dict(n_groups=0), dict(n_groups=-1)])
def test_bootstrap_rejects_empty_resampling(kw):
    with pytest.raises(ValueError):
        bounds.bootstrap_tails(np.eye(4), (2, 2), (1, 1), **kw)


def test_randomized_residual_tail_with_deliberately_poor_sketch():
    Z = np.random.default_rng(4).normal(size=(30, 50))
    r = dis_tt_dense(Z, (5, 6), ranks=(2, 3), backend="randomized",
                     oversample=0, power_iters=0)
    B = tt.block_to_dense(r.cores)
    err = np.sum((Z - B @ (B.T @ Z)) ** 2)
    assert r.tails.sum() == pytest.approx(err, rel=1e-12)
    assert not any(r.spectra_complete)


def test_bootstrap_single_group_exact_scaling(monkeypatch):
    Z = np.arange(1., 17.).reshape(4, 4)
    seen = []
    original = bounds.dis_tt_dense
    def capture(z, *args, **kwargs):
        seen.append(z)
        return original(z, *args, **kwargs)
    monkeypatch.setattr(bounds, "dis_tt_dense", capture)
    bounds.bootstrap_tails(Z, (2, 2), (1, 1), n_boot=2, group_size=4, n_groups=3)
    for z in seen:
        np.testing.assert_allclose(z @ z.T, Z @ Z.T, rtol=1e-14)


@pytest.mark.parametrize("builder", ["output_dis_snapshots", "input_dis_snapshots", "output_pca_snapshots"])
def test_snapshot_empty_sample_set_rejected_before_solve(builder):
    with pytest.raises(ValueError, match="sample"):
        if builder == "output_pca_snapshots":
            getattr(snapshots, builder)(None, np.empty((2, 0)))
        else:
            getattr(snapshots, builder)(None, None, np.empty((2, 0)))


@pytest.mark.parametrize("builder", ["output_dis_snapshots", "input_dis_snapshots"])
def test_zero_random_directions_rejected_before_solve(builder):
    with pytest.raises(ValueError, match="n_dirs"):
        getattr(snapshots, builder)(None, None, np.ones((2, 1)), n_dirs=0)


def test_native_high_depth_without_dense_storage(monkeypatch):
    from dis_tt.native import dis_tt_native
    rng = np.random.default_rng(9)
    cores = [rng.normal(size=(1, 2, 1)) for _ in range(40)]
    cores = [g / np.linalg.norm(g) for g in cores]
    z = tt.TensorTrain(cores)
    def forbidden(*args, **kwargs):
        raise AssertionError("native path densified")
    monkeypatch.setattr(tt, "to_dense", forbidden)
    monkeypatch.setattr(tt, "block_to_dense", forbidden)
    monkeypatch.setattr(tt, "subspace_matrix", forbidden)
    result = dis_tt_native([z, z.copy()], ranks=(1,) * 40)
    assert result.total == pytest.approx(2.)
    assert result.tails.sum() < 1e-12
    assert tt.left_orthonormality_defect(result.cores) < 1e-12


def test_precompress_does_not_discard_small_but_requested_tail():
    Z = np.zeros((2, 100))
    Z[0, 0], Z[1, 1] = 1., 1e-15
    for precompress in (False, True):
        r = dis_tt_dense(Z, (2,), ranks=(1,), backend="svd", precompress=precompress)
        assert r.tails[0] == pytest.approx(1e-30, rel=1e-12, abs=0.)
