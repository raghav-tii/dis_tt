"""T5 acceptance: Algorithm B equals Algorithm A on densifiable problems."""
import numpy as np
import pytest
from conftest import max_principal_angle_sin, proj_err, ref_left_frame, ref_subspace

from dis_tt import tt
from dis_tt.dense import dis_tt_dense
from dis_tt.native import dis_tt_native

MODES = (2,) * 7


@pytest.fixture(scope="module")
def tt_snaps(Z_exact):
    # keep the test small: the first 300 columns, exact TT conversion
    return [tt.from_dense(z, MODES) for z in Z_exact[:, :300].T], Z_exact[:, :300]


@pytest.mark.parametrize("kw", [dict(eps=0.1), dict(eps=0.01), dict(ranks=(2, 3, 4, 4, 4, 3, 3)), dict(eps=0.05, D=6)])
def test_native_matches_dense(tt_snaps, kw):
    snaps, Z = tt_snaps
    a = dis_tt_dense(Z, MODES, **kw)
    b = dis_tt_native(snaps, **kw)
    assert a.ranks == b.ranks
    np.testing.assert_allclose(b.tails, a.tails, rtol=1e-7, atol=1e-13 * a.total)
    assert b.total == pytest.approx(a.total, rel=1e-10)
    for sa, sb in zip(a.spectra, b.spectra):
        n = min(len(sa), len(sb))
        np.testing.assert_allclose(sb[:n], sa[:n], rtol=1e-7, atol=1e-13 * a.total)
    assert proj_err(ref_subspace(b.cores, MODES), Z) == pytest.approx(b.tails.sum(), rel=1e-7, abs=1e-13 * a.total)


def test_native_full_rank_basis(tt_snaps):
    snaps, Z = tt_snaps
    b = dis_tt_native(snaps)
    V = np.linalg.eigh(Z @ Z.T)[1][:, ::-1]
    assert max_principal_angle_sin(ref_left_frame(b.cores)[:, :4], V[:, :4]) < 1e-6


def test_native_does_not_densify(tt_snaps, monkeypatch):
    """Algorithm B must never call to_dense / block_to_dense on snapshots."""
    snaps, _ = tt_snaps

    def boom(*a, **k):
        raise AssertionError("densified inside Algorithm B")

    monkeypatch.setattr(tt, "to_dense", boom)
    monkeypatch.setattr(tt, "block_to_dense", boom)
    dis_tt_native(snaps[:20], eps=0.1)
