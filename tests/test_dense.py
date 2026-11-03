"""T4 acceptance: Algorithm A (doc Sec. 4 Prop. 1, Sec. 6)."""
import numpy as np
import pytest
from conftest import (eig_desc, left_orth_defect, max_principal_angle_sin, proj_err, ref_left_frame,
                      ref_subspace, ref_tail, ref_unfold_gram)

from dis_tt.dense import dis_tt_dense

MODES = (2,) * 7


def test_full_ranks_reproduce_dense_dis(Z_exact):
    """D = d with no truncation before the last bond == the paper's (non-TT) DIS basis."""
    res = dis_tt_dense(Z_exact, MODES)  # eps = None -> full ranks
    assert res.D == 7 and res.is_basis
    lam = eig_desc(Z_exact @ Z_exact.T)
    r = 7
    np.testing.assert_allclose(res.spectra[-1][:r], lam[:r], rtol=1e-8)
    U = ref_left_frame(res.cores)
    V = np.linalg.eigh(Z_exact @ Z_exact.T)[1][:, ::-1]
    assert max_principal_angle_sin(U[:, :4], V[:, :4]) < 1e-6  # Gram eigvecs: ~sqrt(eps) accuracy after dropping round-off modes
    assert res.cert < 1e-12


@pytest.mark.parametrize("D", [6, 7])
@pytest.mark.parametrize("eps", [0.3, 0.1, 0.03])
def test_exact_training_identity_and_tolerance(Z_exact, D, eps):
    res = dis_tt_dense(Z_exact, MODES, eps=eps, D=D)
    B = ref_subspace(res.cores, MODES)
    err = proj_err(B, Z_exact)
    assert err == pytest.approx(res.tails.sum(), rel=1e-8, abs=1e-14 * res.total)
    assert res.total == pytest.approx(np.sum(Z_exact**2), rel=1e-12)
    assert res.cert <= eps**2 * (1 + 1e-10)
    assert left_orth_defect(res.cores) < 1e-10
    assert len(res.tails) == D and len(res.spectra) == D


@pytest.mark.parametrize("ranks", [(2, 3, 3, 3, 3, 3, 3), (2, 4, 6, 6, 5, 4, 2)])
def test_prop1_sandwich(Z_exact, ranks):
    """max_k tail(H^(k)) <= greedy error = sum tails <= sum_k tail(H^(k))."""
    res = dis_tt_dense(Z_exact, MODES, ranks=ranks)
    assert res.ranks == ranks
    unproj = np.array([ref_tail(eig_desc(ref_unfold_gram(Z_exact, MODES, k + 1)), r) for k, r in enumerate(ranks)])
    err = proj_err(ref_subspace(res.cores, MODES), Z_exact)
    scale = res.total * 1e-10
    assert unproj.max() <= err + scale
    assert err <= unproj.sum() + scale
    assert np.all(res.tails <= unproj + scale)  # per-bond Poincare separation


def test_backends_agree(Z_exact):
    ranks = (2, 3, 4, 4, 4, 3, 3)
    a = dis_tt_dense(Z_exact, MODES, ranks=ranks, backend="gram")
    b = dis_tt_dense(Z_exact, MODES, ranks=ranks, backend="svd")
    c = dis_tt_dense(Z_exact, MODES, ranks=ranks, backend="randomized", oversample=10,
                     power_iters=2, rng=np.random.default_rng(0))
    np.testing.assert_allclose(a.tails, b.tails, rtol=1e-8, atol=1e-14 * a.total)
    np.testing.assert_allclose(a.tails.sum(), c.tails.sum(), rtol=1e-5)
    Ua, Ub = ref_left_frame(a.cores), ref_left_frame(b.cores)
    assert max_principal_angle_sin(Ua, Ub) < 1e-6


def test_precompress_is_invisible(Z_exact):
    a = dis_tt_dense(Z_exact, MODES, eps=0.05)
    b = dis_tt_dense(Z_exact, MODES, eps=0.05, precompress=True)
    assert a.ranks == b.ranks
    np.testing.assert_allclose(a.tails, b.tails, rtol=1e-8, atol=1e-14 * a.total)


def test_single_snapshot_is_tt_svd():
    """T = 1, D = d-1: PCA-TT of one vector is plain TT-SVD with the eps/sqrt(D) rule."""
    t = np.linspace(0, 1, 2**8)
    y = (np.sin(7 * t) + np.exp(-((t - 0.3) / 0.05) ** 2))[:, None]
    res = dis_tt_dense(y, (2,) * 8, eps=1e-3, D=7)
    B = ref_subspace(res.cores, (2,) * 8)
    assert proj_err(B, y) <= 1e-6 * np.sum(y**2) * (1 + 1e-10)


def test_snapshot_column_order_irrelevant(Z_exact):
    perm = np.random.default_rng(0).permutation(Z_exact.shape[1])
    a = dis_tt_dense(Z_exact, MODES, eps=0.05)
    b = dis_tt_dense(Z_exact[:, perm], MODES, eps=0.05)
    np.testing.assert_allclose(a.tails, b.tails, rtol=1e-8, atol=1e-14 * a.total)
