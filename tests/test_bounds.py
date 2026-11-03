"""T6b acceptance: certificates and bound report (doc Sec. 4)."""
import numpy as np
import pytest
from conftest import eig_desc, ref_tail, ref_unfold_gram

from dis_tt.bounds import bootstrap_tails, bound_report, projection_error, unprojected_tails
from dis_tt.dense import dis_tt_dense

MODES = (2,) * 7
RANKS = (2, 3, 4, 4, 4, 3, 3)


def test_projection_error_equals_certificate(Z_exact):
    res = dis_tt_dense(Z_exact, MODES, ranks=RANKS)
    assert projection_error(res, Z_exact) == pytest.approx(res.tails.sum(), rel=1e-8)


def test_unprojected_tails_reference(Z_exact):
    got = unprojected_tails(Z_exact, MODES, RANKS)
    ref = [ref_tail(eig_desc(ref_unfold_gram(Z_exact, MODES, k + 1)), r) for k, r in enumerate(RANKS)]
    np.testing.assert_allclose(got, ref, rtol=1e-8, atol=1e-14 * np.sum(Z_exact**2))


def test_bound_report_ordering(Z_exact, Z_exact_test):
    res = dis_tt_dense(Z_exact, MODES, ranks=RANKS)
    rep = bound_report(res, Z_exact, Z_exact_test)
    tol = 1e-10 * res.total
    assert rep.train_error == pytest.approx(rep.sum_tails, rel=1e-8)
    assert rep.lower_unprojected <= rep.train_error + tol
    assert rep.train_error <= rep.upper_unprojected + tol
    assert rep.test_error > 0


def test_bootstrap_shape(Z_exact):
    s = bootstrap_tails(Z_exact, MODES, RANKS, n_boot=10, rng=np.random.default_rng(0), group_size=64)
    assert s.shape == (len(RANKS),) and np.all(np.isfinite(s)) and np.all(s >= 0)
