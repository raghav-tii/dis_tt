"""T5 extra tests: mixed snapshot TT ranks, n_k != 2, D = d-1, T = 1, rank caps, no densification."""
import numpy as np
import pytest
from conftest import left_orth_defect, max_principal_angle_sin, proj_err, ref_left_frame, ref_subspace

from dis_tt import tt
from dis_tt.dense import dis_tt_dense
from dis_tt.native import dis_tt_native


def _random_tt(modes, ranks, rng, scale=1.0):
    rs = (1,) + tuple(ranks) + (1,)
    return tt.TensorTrain([scale * rng.standard_normal((rs[k], n, rs[k + 1])) for k, n in enumerate(modes)])


def _mixed_rank_snaps(modes, seed=0):
    rng = np.random.default_rng(seed)
    d = len(modes)
    rank_choices = [(1,) * (d - 1), (2,) * (d - 1), (3,) * (d - 1),
                    tuple(min(k + 1, d - k, 4) for k in range(1, d))]
    snaps = [_random_tt(modes, rank_choices[t % 4], rng, scale=0.5 ** (t % 3)) for t in range(13)]
    return snaps


def _dense(snaps):
    return np.stack([tt.to_dense(z) for z in snaps], axis=1)


@pytest.mark.parametrize("modes", [(2,) * 6, (3, 4, 2, 5)])
@pytest.mark.parametrize("kw", [dict(), dict(eps=0.2), dict(eps=0.05, D=-1), dict(eps=1e-3, rmax=3)])
def test_mixed_snapshot_ranks_match_dense(modes, kw):
    kw = dict(kw)
    if kw.get("D") == -1:
        kw["D"] = len(modes) - 1
    snaps = _mixed_rank_snaps(modes)
    assert len({z.ranks for z in snaps}) >= 3  # genuinely mixed TT ranks
    Z = _dense(snaps)
    a = dis_tt_dense(Z, modes, **kw)
    b = dis_tt_native(snaps, **kw)
    assert a.ranks == b.ranks
    np.testing.assert_allclose(b.tails, a.tails, rtol=1e-7, atol=1e-13 * a.total)
    assert b.total == pytest.approx(a.total, rel=1e-10)
    for sa, sb in zip(a.spectra, b.spectra):
        np.testing.assert_allclose(sb, sa, rtol=1e-7, atol=1e-12 * a.total)
    assert proj_err(ref_subspace(b.cores, modes), Z) == pytest.approx(b.tails.sum(), rel=1e-7, abs=1e-13 * a.total)
    assert left_orth_defect(b.cores) < 1e-10
    if "rmax" in kw:
        assert max(b.ranks) <= kw["rmax"]


def test_full_rank_subspace_matches_dense_mixed_ranks():
    modes = (3, 4, 2, 5)
    snaps = _mixed_rank_snaps(modes, seed=1)
    Z = _dense(snaps)
    a, b = dis_tt_dense(Z, modes, ranks=(3, 5, 6, 4)), dis_tt_native(snaps, ranks=(3, 5, 6, 4))
    assert max_principal_angle_sin(ref_left_frame(a.cores), ref_left_frame(b.cores)) < 1e-6


def test_single_snapshot():
    modes = (2, 3, 4)
    z = _random_tt(modes, (2, 3), np.random.default_rng(2))
    b = dis_tt_native([z], eps=1e-8)
    a = dis_tt_dense(tt.to_dense(z)[:, None], modes, eps=1e-8)
    assert a.ranks == b.ranks
    np.testing.assert_allclose(b.tails, a.tails, atol=1e-12 * a.total)


def test_mixed_ranks_never_densify(monkeypatch):
    snaps = _mixed_rank_snaps((3, 4, 2, 5), seed=3)

    def boom(*a, **k):
        raise AssertionError("densified inside Algorithm B")

    monkeypatch.setattr(tt, "to_dense", boom)
    monkeypatch.setattr(tt, "block_to_dense", boom)
    monkeypatch.setattr(tt, "subspace_matrix", boom)
    dis_tt_native(snaps, eps=0.1)
    dis_tt_native(snaps, ranks=(2, 2, 2))


def test_input_validation():
    a = _random_tt((2, 3), (2,), np.random.default_rng(4))
    b = _random_tt((3, 2), (2,), np.random.default_rng(5))
    with pytest.raises(ValueError):
        dis_tt_native([])
    with pytest.raises(ValueError):
        dis_tt_native([a, b])
    with pytest.raises(ValueError):
        dis_tt_native([a], ranks=(1,), D=2)
