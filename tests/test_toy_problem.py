"""Fixture sanity (passes before any task is done). If these fail, stop: the ground truth is broken."""
import numpy as np


def test_jvp_matches_finite_difference(toy):
    rng = np.random.default_rng(0)
    x = toy.prior.sample(rng, 1)[:, 0]
    h = toy.prior.sample(rng, 1)[:, 0]
    st = toy.solve(x)
    t = 1e-6
    fd = (toy.output(toy.solve(x + t * h)) - toy.output(toy.solve(x - t * h))) / (2 * t)
    np.testing.assert_allclose(toy.jvp(st, h), fd, rtol=1e-6, atol=1e-9)


def test_vjp_is_adjoint_of_jvp(toy):
    rng = np.random.default_rng(1)
    st = toy.solve(toy.prior.sample(rng, 1)[:, 0])
    h, w = rng.standard_normal(toy.d_x), rng.standard_normal(toy.d_y)
    assert abs(w @ toy.jvp(st, h) - toy.vjp(st, w) @ h) < 1e-10 * (1 + abs(w @ toy.jvp(st, h)))


def test_batched_jvp_vjp(toy):
    rng = np.random.default_rng(2)
    st = toy.solve(toy.prior.sample(rng, 1)[:, 0])
    H = rng.standard_normal((toy.d_x, 3))
    W = rng.standard_normal((toy.d_y, 3))
    J = toy.jacobian(st)
    np.testing.assert_allclose(toy.jvp(st, H), J @ H, atol=1e-12)
    np.testing.assert_allclose(toy.vjp(st, W), J.T @ W, atol=1e-12)


def test_prior_factor(toy):
    p = toy.prior
    L = p.apply_sqrt(np.eye(p.dim))
    np.testing.assert_allclose(L @ L.T, p.covariance(), rtol=1e-10, atol=1e-14)
    v = np.random.default_rng(3).standard_normal(p.dim)
    np.testing.assert_allclose(p.apply_inv_sqrt(p.apply_sqrt(v)), v, rtol=1e-8, atol=1e-10)
    mu, phi = p.kl_modes(p.dim)
    assert np.all(np.diff(mu) <= 0)
    np.testing.assert_allclose((phi * mu) @ phi.T, p.covariance(), atol=1e-12)


def test_problem_is_nontrivial(toy, Z_exact):
    """DIS spectrum must have a usable gap and the output DIS basis must be low QTT rank-ish."""
    lam = np.linalg.eigvalsh(Z_exact @ Z_exact.T)[::-1]
    assert lam[0] > 0
    assert lam[toy.p] < 1e-10 * lam[0]  # rank(H_Y) <= p
    assert lam[3] - lam[4] > 1e-3 * lam[0]  # gap used by r = 4 tests
