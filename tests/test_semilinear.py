"""T9: semilinear elliptic problem (paper Sec. 4.1, FD). FD, adjointness and solver checks."""
import numpy as np
import pytest

from dis_tt.problems.semilinear import EllipticPrior, SemilinearProblem


@pytest.fixture(scope="module")
def prob():
    return SemilinearProblem(levels=6)


@pytest.fixture(scope="module")
def x0(prob):
    return prob.prior.sample(np.random.default_rng(0), 1)[:, 0]


def test_newton_converges(prob, x0):
    st = prob.solve(x0)
    assert st["res_rel"] < 1e-11 and st["newton_iters"] < 20
    big = prob.solve(30 * x0)  # strongly nonlinear regime
    assert np.linalg.norm(prob.residual(big["y"], 30 * x0)) < 1e-9 * np.linalg.norm(prob.m * 30 * x0)


def test_paper_coefficients(prob):
    mid = (np.arange(1, prob.d_x + 1) - 0.5) * prob.h
    np.testing.assert_allclose(prob.c_mid, np.where(mid > 0.5, 0.0101, 0.0001))
    assert prob.c2 == 0.1 and prob.prior.a_lap == 2.0 and prob.prior.a_id == 10.0


def test_jvp_matches_central_differences(prob, x0):
    st = prob.solve(x0)
    v = prob.prior.sample(np.random.default_rng(1), 1)[:, 0]
    for eps in (1e-4, 1e-5):
        fd = (prob.solve(x0 + eps * v)["y"] - prob.solve(x0 - eps * v)["y"]) / (2 * eps)
        assert np.linalg.norm(fd - prob.jvp(st, v)) <= 1e-7 * np.linalg.norm(fd)


def test_adjointness_and_batching(prob, x0):
    st = prob.solve(x0)
    rng = np.random.default_rng(2)
    V, Wm = rng.standard_normal((prob.d_x, 5)), rng.standard_normal((prob.d_y, 4))
    lhs = Wm.T @ prob.jvp(st, V)
    rhs = prob.vjp(st, Wm).T @ V
    np.testing.assert_allclose(lhs, rhs, rtol=1e-12, atol=1e-14 * np.abs(lhs).max())
    np.testing.assert_allclose(prob.jvp(st, V)[:, 2], prob.jvp(st, V[:, 2]), rtol=1e-13)
    np.testing.assert_allclose(prob.vjp(st, Wm)[:, 1], prob.vjp(st, Wm[:, 1]), rtol=1e-13)
    np.testing.assert_allclose(prob.jacobian(st) @ V, prob.jvp(st, V), rtol=1e-12, atol=1e-16)


def test_linear_limit_second_order_convergence():
    """c2 = 0, constant c1 = c, x = 1: exact y = (s - s^2/2)/c. FD error must decrease ~h^2 (or be exact)."""
    errs = []
    for lv in (4, 5, 6):
        p = SemilinearProblem(levels=lv, c2=0.0, diff_low=0.5, diff_jump=0.0)
        y = p.solve(np.ones(p.d_x))["y"]
        errs.append(np.max(np.abs(y - (p.s - p.s ** 2 / 2) / 0.5)))
    assert errs[-1] < 1e-10 or errs[-1] < errs[0] / 10


def test_prior_scaling_and_factor():
    mid_var = []
    for n in (64, 256):
        pr = EllipticPrior(n)
        C = pr.covariance()
        L = pr.apply_sqrt(np.eye(n))
        np.testing.assert_allclose(L @ L.T, C, atol=1e-12)
        np.testing.assert_allclose(pr.apply_inv_sqrt(L), np.eye(n), atol=1e-8)
        mid_var.append(np.diag(C)[n // 2])
        mu, phi = pr.kl_modes(3)
        np.testing.assert_allclose(C @ phi, phi * mu, atol=1e-10)

    # nodal covariance = Green's function values: pointwise variance is mesh independent (2%),
    # larger than the free-space 1/(2 sqrt(20)) because of the Neumann images (corr. length 0.45)
    assert abs(mid_var[0] / mid_var[1] - 1) < 0.02 and mid_var[1] > 1 / (2 * np.sqrt(20))
