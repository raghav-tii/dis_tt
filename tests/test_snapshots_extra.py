"""T3 extra tests: non-symmetric and wide whitening factors, solve counts, batching, N = 1."""
import numpy as np
import pytest

from dis_tt import snapshots as sn


class FactorPrior:
    """Wraps the toy prior with factor L' = L Q: same C_X = L' L'^T, L' non-symmetric or wide."""

    def __init__(self, base, Q):
        self.base, self.Q = base, Q
        self.white_dim = Q.shape[1]
        self.Lf = base.apply_sqrt(np.eye(base.dim)) @ Q

    def sample(self, rng, n):
        return self.Lf @ rng.standard_normal((self.white_dim, n))

    def apply_sqrt(self, v):
        return self.Lf @ v

    def apply_sqrt_T(self, v):
        return self.Lf.T @ v

    def apply_inv_sqrt(self, v):
        return np.linalg.pinv(self.Lf) @ v

    def kl_modes(self, k):
        return self.base.kl_modes(k)


class CountingModel:
    def __init__(self, model):
        self.m, self.n_solve, self.n_jvp, self.n_vjp = model, 0, 0, 0

    def solve(self, x):
        self.n_solve += 1
        return self.m.solve(x)

    def output(self, s):
        return self.m.output(s)

    def jvp(self, s, h):
        self.n_jvp += 1
        return self.m.jvp(s, h)

    def vjp(self, s, w):
        self.n_vjp += 1
        return self.m.vjp(s, w)


def _orth(n, m, seed):
    """(n, m) with orthonormal rows if m >= n (so Q Q^T = I_n)."""
    Q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((m, n)))
    return Q.T


@pytest.fixture(scope="module")
def xs(toy):
    return toy.prior.sample(np.random.default_rng(3), 6)


@pytest.mark.parametrize("wide", [False, True])
def test_exact_output_dis_invariant_to_factor(toy, xs, wide):
    prior = FactorPrior(toy.prior, _orth(toy.d_x, 2 * toy.d_x if wide else toy.d_x, 1))
    if wide:
        assert prior.Lf.shape == (toy.d_x, 2 * toy.d_x)
    else:
        assert not np.allclose(prior.Lf, prior.Lf.T)
    S = sn.output_dis_snapshots(toy, prior, xs, direction_kind="exact")
    assert S.Z.shape == (toy.d_y, prior.white_dim * xs.shape[1])
    H = toy.H_Y_reference(xs)
    np.testing.assert_allclose(S.Z @ S.Z.T, H, rtol=1e-9, atol=1e-12 * np.abs(H).max())


@pytest.mark.parametrize("wide", [False, True])
def test_input_dis_uses_LT_for_nonsymmetric_factor(toy, xs, wide):
    prior = FactorPrior(toy.prior, _orth(toy.d_x, 2 * toy.d_x if wide else toy.d_x, 2))
    S = sn.input_dis_snapshots(toy, prior, xs, eta_kind="basis", U_out=np.eye(toy.d_y))
    H = sum((toy.jacobian(toy.solve(x)) @ prior.Lf).T @ (toy.jacobian(toy.solve(x)) @ prior.Lf)
            for x in xs.T) / xs.shape[1]
    assert S.Z.shape[0] == prior.white_dim
    np.testing.assert_allclose(S.Z @ S.Z.T, H, rtol=1e-9, atol=1e-12 * np.abs(H).max())


def test_one_solve_and_one_batched_call_per_sample(toy, xs):
    N = xs.shape[1]
    for call, attr in [
        (lambda m: sn.output_dis_snapshots(m, toy.prior, xs, "random", n_dirs=5), "n_jvp"),
        (lambda m: sn.output_dis_snapshots(m, toy.prior, xs, "kl", n_dirs=5), "n_jvp"),
        (lambda m: sn.input_dis_snapshots(m, toy.prior, xs, "gaussian", n_dirs=5), "n_vjp"),
        (lambda m: sn.output_pca_snapshots(m, xs), None),
    ]:
        m = CountingModel(toy)
        call(m)
        assert m.n_solve == N
        if attr:
            assert getattr(m, attr) == N


def test_batch_false_matches_batched(toy, xs):
    for kw in [dict(direction_kind="kl", n_dirs=6), dict(direction_kind="random", n_dirs=3)]:
        a = sn.output_dis_snapshots(toy, toy.prior, xs, rng=np.random.default_rng(1), **kw).Z
        b = sn.output_dis_snapshots(toy, toy.prior, xs, rng=np.random.default_rng(1), batch=False, **kw).Z
        np.testing.assert_allclose(a, b, atol=1e-14)
    a = sn.input_dis_snapshots(toy, toy.prior, xs, "gaussian", 4, rng=np.random.default_rng(2)).Z
    b = sn.input_dis_snapshots(toy, toy.prior, xs, "gaussian", 4, rng=np.random.default_rng(2), batch=False).Z
    np.testing.assert_allclose(a, b, atol=1e-14)


def test_single_sample_and_column_grouping(toy, xs):
    x = xs[:, :1]
    S1 = sn.output_dis_snapshots(toy, toy.prior, x, direction_kind="kl", n_dirs=4)
    SN = sn.output_dis_snapshots(toy, toy.prior, xs, direction_kind="kl", n_dirs=4)
    assert S1.n_samples == 1 and S1.Z.shape == (toy.d_y, 4)
    # columns grouped per sample: first 4 columns of the N-sample set are x_1's, rescaled
    np.testing.assert_allclose(SN.Z[:, :4] * np.sqrt(xs.shape[1]), S1.Z, atol=1e-14)
    P, y_bar = sn.output_pca_snapshots(toy, x)
    assert np.allclose(P.Z, 0) and np.allclose(y_bar, toy.output(toy.solve(x[:, 0])))


def test_mixed_endpoints_and_validation(toy, xs):
    P, _ = sn.output_pca_snapshots(toy, xs)
    Dd = sn.output_dis_snapshots(toy, toy.prior, xs, direction_kind="kl", n_dirs=4)
    M0 = sn.mixed_snapshots(P, Dd, 0.0)
    assert np.all(M0.Z[:, : P.Z.shape[1]] == 0) and np.sum(M0.Z ** 2) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        sn.mixed_snapshots(P, Dd, 1.5)
    with pytest.raises(ValueError):
        sn.input_dis_snapshots(toy, toy.prior, xs, eta_kind="basis")
