"""T4 extra tests: n_k != 2, D = d-1 with eps, binding rmax, T = 1, randomized trace tails, linear cost in T."""
import time

import numpy as np
import pytest
from conftest import (eig_desc, left_orth_defect, max_principal_angle_sin, proj_err, ref_left_frame,
                      ref_subspace, ref_tail, ref_unfold_gram)

from dis_tt.dense import dis_tt_dense

MIXED = (3, 4, 2, 5)


def _lowrank_Z(modes, T, rank=6, seed=0):
    rng = np.random.default_rng(seed)
    n = int(np.prod(modes))
    t = np.linspace(0, 1, n)
    basis = np.stack([np.sin((j + 1) * np.pi * t) * 0.6 ** j for j in range(rank)], axis=1)
    return basis @ rng.standard_normal((rank, T)) + 1e-4 * rng.standard_normal((n, T))


@pytest.mark.parametrize("D", [3, 4])
@pytest.mark.parametrize("eps", [0.2, 0.02])
def test_mixed_modes_identity_and_tolerance(D, eps):
    Z = _lowrank_Z(MIXED, 30)
    res = dis_tt_dense(Z, MIXED, eps=eps, D=D)
    assert [G.shape[1] for G in res.cores] == list(MIXED[:D])
    assert res.is_basis == (D == 4)
    B = ref_subspace(res.cores, MIXED)
    assert proj_err(B, Z) == pytest.approx(res.tails.sum(), rel=1e-8, abs=1e-14 * res.total)
    assert res.cert <= eps ** 2 * (1 + 1e-10)
    assert left_orth_defect(res.cores) < 1e-10
    for k, lam in enumerate(res.spectra):  # projected operator has r_{k-1} n_k eigenvalues
        r_prev = 1 if k == 0 else res.ranks[k - 1]
        assert lam.size == r_prev * MIXED[k]


def test_D_d_minus_1_with_eps_on_toy(Z_exact):
    modes = (2,) * 7
    res = dis_tt_dense(Z_exact, modes, eps=0.05, D=6)
    assert res.D == 6 and not res.is_basis
    B = ref_subspace(res.cores, modes)
    assert B.shape[1] == res.ranks[-1] * 2
    assert proj_err(B, Z_exact) == pytest.approx(res.tails.sum(), rel=1e-8)
    assert np.all(res.tails <= 0.05 ** 2 * res.total / 6 * (1 + 1e-12))


def test_rmax_binding():
    Z = _lowrank_Z(MIXED, 25, rank=8, seed=1)
    free = dis_tt_dense(Z, MIXED, eps=1e-6)
    assert max(free.ranks) > 3
    res = dis_tt_dense(Z, MIXED, eps=1e-6, rmax=3)
    assert max(res.ranks) <= 3 and res.cert > 1e-12  # cap binds, tolerance no longer met
    assert proj_err(ref_subspace(res.cores, MIXED), Z) == pytest.approx(res.tails.sum(), rel=1e-8)
    # bond 1 is plain truncated eigendecomposition of H^(1): tail equals the reference tail
    lam1 = eig_desc(ref_unfold_gram(Z, MIXED, 1))
    assert res.tails[0] == pytest.approx(ref_tail(lam1, res.ranks[0]), rel=1e-8)


def test_fixed_ranks_override_eps_and_rmax():
    Z = _lowrank_Z(MIXED, 20)
    res = dis_tt_dense(Z, MIXED, ranks=(2, 3, 4, 2), eps=0.5, rmax=1)
    assert res.ranks == (2, 3, 4, 2)
    res = dis_tt_dense(Z, MIXED, ranks=(3, 4, 2))  # D inferred from len(ranks)
    assert res.D == 3


def test_single_snapshot_T1_matches_tt_svd_ranks():
    from dis_tt import tt
    y = _lowrank_Z(MIXED, 1, seed=2)
    res = dis_tt_dense(y, MIXED, eps=1e-3, D=3)
    ref = tt.from_dense(y[:, 0], MIXED, eps=1e-3)  # threshold eps ||y|| / sqrt(d-1), d-1 = 3 = D
    assert res.ranks == ref.ranks[1:-1]
    assert proj_err(ref_subspace(res.cores, MIXED), y) <= 1e-6 * np.sum(y ** 2) * (1 + 1e-10)


def test_randomized_trace_tails_are_exact_and_flagged(Z_exact):
    modes = (2,) * 7
    a = dis_tt_dense(Z_exact, modes, eps=0.05)
    c = dis_tt_dense(Z_exact, modes, eps=0.05, backend="randomized", power_iters=2, rng=np.random.default_rng(1))
    assert not any(c.spectra_complete) and all(a.spectra_complete)
    assert c.ranks == a.ranks
    err = proj_err(ref_subspace(c.cores, modes), Z_exact)
    assert err == pytest.approx(c.tails.sum(), rel=1e-8)  # trace-based tail is the exact discarded mass
    np.testing.assert_allclose(c.tails, a.tails, rtol=1e-6, atol=1e-12 * a.total)


def test_precompress_invisible_mixed_modes_and_rank_deficient():
    Z = _lowrank_Z(MIXED, 200, rank=5, seed=3)
    Z[:, 100:] = 0.0  # rank-deficient, T >> rank
    for kw in [dict(eps=0.05), dict(eps=0.05, D=3), dict(ranks=(2, 3, 3, 2))]:
        a = dis_tt_dense(Z, MIXED, **kw)
        b = dis_tt_dense(Z, MIXED, precompress=True, **kw)
        assert a.ranks == b.ranks and b.meta["T_used"] < Z.shape[1]
        np.testing.assert_allclose(a.tails, b.tails, rtol=1e-8, atol=1e-14 * a.total)
        assert a.total == b.total
        Ua, Ub = ref_subspace(a.cores, MIXED), ref_subspace(b.cores, MIXED)
        assert max_principal_angle_sin(Ua, Ub) < 1e-6


def test_all_zero_snapshots():
    res = dis_tt_dense(np.zeros((40, 3)), (2, 4, 5), eps=0.1)
    assert res.ranks == (1, 1, 1) and res.total == 0 and res.cert == 0
    assert left_orth_defect(res.cores) < 1e-12


def test_invalid_arguments():
    Z = np.zeros((24, 2))
    with pytest.raises(ValueError):
        dis_tt_dense(Z, (2, 3, 5))
    with pytest.raises(ValueError):
        dis_tt_dense(Z, (2, 3, 4), ranks=(1, 1), D=3)
    with pytest.raises(ValueError):
        dis_tt_dense(Z, (2, 3, 4), D=4)


def test_cost_linear_in_T():
    """Doubling T at fixed d_Y and ranks must not much more than double the runtime (ratio < 3)."""
    modes = (2,) * 10
    rng = np.random.default_rng(4)
    Z2 = rng.standard_normal((2 ** 10, 4000))
    Z1 = Z2[:, :2000].copy()

    def best(Z):
        dis_tt_dense(Z, modes, rmax=6)  # warm-up
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            dis_tt_dense(Z, modes, rmax=6)
            times.append(time.perf_counter() - t0)
        return min(times)

    t1, t2 = best(Z1), best(Z2)
    assert t2 / t1 < 3.0, (t1, t2)
