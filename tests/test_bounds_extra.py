"""T6b extra tests: bootstrap scaling and grouping, D = d-1, mixed modes, rank caps, validation."""
import numpy as np
import pytest
from conftest import eig_desc, ref_tail, ref_unfold_gram

from dis_tt.bounds import bootstrap_tails, bound_report, projection_error, unprojected_tails
from dis_tt.dense import dis_tt_dense

MIXED = (3, 4, 2, 5)


def _Z(T=24, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, int(np.prod(MIXED)))
    B = np.stack([np.sin((j + 1) * np.pi * t) * 0.6 ** j for j in range(6)], 1)
    return B @ rng.standard_normal((6, T)) + 1e-3 * rng.standard_normal((t.size, T))


@pytest.mark.parametrize("D", [3, 4])
def test_mixed_modes_sandwich_and_identity(D):
    Z = _Z()
    res = dis_tt_dense(Z, MIXED, eps=0.05, D=D)
    rep = bound_report(res, Z, _Z(seed=1))
    tol = 1e-10 * res.total
    assert rep.train_error == pytest.approx(rep.sum_tails, rel=1e-8)
    assert rep.lower_unprojected <= rep.train_error + tol <= rep.upper_unprojected + 2 * tol
    assert rep.per_bond_unprojected.shape == (D,)
    assert np.all(res.tails <= rep.per_bond_unprojected + tol)
    ref = [ref_tail(eig_desc(ref_unfold_gram(Z, MIXED, k + 1)), r) for k, r in enumerate(res.ranks)]
    np.testing.assert_allclose(rep.per_bond_unprojected, ref, rtol=1e-8, atol=1e-14 * res.total)


def test_unprojected_tails_rank_above_unfolding_rank_is_zero():
    Z = _Z(T=3)
    np.testing.assert_allclose(unprojected_tails(Z, MIXED, (3, 12, 24, 3)), 0.0, atol=1e-20)
    assert unprojected_tails(Z, MIXED, (1,))[0] > 0


def test_projection_error_of_full_subspace_is_zero():
    Z = _Z()
    res = dis_tt_dense(Z, MIXED, ranks=(3, 12, 24), D=3)  # D = d-1 at full ranks: Pi = I
    assert projection_error(res, Z) < 1e-20 * res.total


def test_bootstrap_preserves_scaling(monkeypatch):
    """E*[Z* Z*^T] = Z Z^T: capture the resamples bootstrap_tails feeds to Algorithm A."""
    from dis_tt import bounds
    Z = _Z(T=40, seed=2)
    seen = []

    def capture(Zb, *a, **k):
        seen.append(Zb.copy())
        return dis_tt_dense(Zb, *a, **k)

    monkeypatch.setattr(bounds, "dis_tt_dense", capture)
    H = Z @ Z.T
    for n_groups in (None, 25):
        seen.clear()
        bootstrap_tails(Z, MIXED, (1,), n_boot=3000, rng=np.random.default_rng(0), group_size=4,
                        n_groups=n_groups)
        assert all(Zb.shape[1] == 4 * (n_groups or 10) for Zb in seen)
        H_boot = sum(Zb @ Zb.T for Zb in seen) / len(seen)
        assert np.linalg.norm(H_boot - H) / np.linalg.norm(H) < 0.03
        # every resampled group is an intact, unscaled-up-to-sqrt(G/m) block of 4 original columns
        blocks = {tuple(np.round(Z[:, 4 * j], 12)) for j in range(10)}
        fac = np.sqrt(10 / (n_groups or 10))
        assert all(tuple(np.round(Zb[:, c] / fac, 12)) in blocks for Zb in seen[:20] for c in range(0, Zb.shape[1], 4))


def test_bootstrap_groups_stay_together():
    """If every group is identical, any grouped resample equals Z up to column order -> zero spread."""
    block = _Z(T=4, seed=3)
    Z = np.concatenate([block] * 6, axis=1) / np.sqrt(6)
    s = bootstrap_tails(Z, MIXED, (2, 3, 3), n_boot=8, rng=np.random.default_rng(1), group_size=4)
    np.testing.assert_allclose(s, 0.0, atol=1e-10 * np.sum(Z ** 2))
    # ungrouped resampling of the same columns does spread
    s1 = bootstrap_tails(Z, MIXED, (2, 3, 3), n_boot=8, rng=np.random.default_rng(1), group_size=1)
    assert s1.max() > 1e-6


def test_bootstrap_reproducible_and_validates():
    Z = _Z(T=12)
    a = bootstrap_tails(Z, MIXED, (2, 3, 3, 2), n_boot=5, rng=np.random.default_rng(4), group_size=3)
    b = bootstrap_tails(Z, MIXED, (2, 3, 3, 2), n_boot=5, rng=np.random.default_rng(4), group_size=3)
    np.testing.assert_array_equal(a, b)
    assert bootstrap_tails(Z, MIXED, (2, 3), n_boot=1).shape == (2,)
    with pytest.raises(ValueError):
        bootstrap_tails(Z, MIXED, (2, 3), group_size=5)  # does not divide T = 12
