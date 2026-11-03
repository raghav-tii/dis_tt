"""T7 extra tests: non-symmetric factor (L^T in the VJP branch), auto branch, normalisations, validation."""
import numpy as np
import pytest

from dis_tt import snapshots as sn
from dis_tt.input_side import input_dis_tt
from dis_tt.rbno import assemble_rbno_data, sobolev_loss


class FactorPrior:
    """Square non-symmetric factor L' = C^{1/2} Q (same covariance as the toy prior)."""

    def __init__(self, base, seed=3):
        Q = np.linalg.qr(np.random.default_rng(seed).standard_normal((base.dim, base.dim)))[0]
        self.base, self.white_dim = base, base.dim
        self.Lf = base.apply_sqrt(np.eye(base.dim)) @ Q
        self.Linv = np.linalg.inv(self.Lf)

    def sample(self, rng, n):
        return self.Lf @ rng.standard_normal((self.white_dim, n))

    def apply_sqrt(self, v):
        return self.Lf @ v

    def apply_sqrt_T(self, v):
        return self.Lf.T @ v

    def apply_inv_sqrt(self, v):
        return self.Linv @ v

    def kl_modes(self, k):
        return self.base.kl_modes(k)


def _setup(toy, xs, prior, r_Y, ranks):
    Z = sn.output_dis_snapshots(toy, prior, xs, direction_kind="exact").Z
    U = np.linalg.eigh(Z @ Z.T)[1][:, ::-1][:, :r_Y]
    S = sn.input_dis_snapshots(toy, prior, xs, eta_kind="basis", U_out=U)
    basis = input_dis_tt(S, toy.in_modes, prior, ranks=ranks)
    y_bar = np.mean([toy.output(toy.solve(x)) for x in xs.T], axis=0)
    return U, basis, y_bar


@pytest.mark.parametrize("r_Y,ranks", [(4, (2, 4, 4, 4, 4, 3)), (2, (2, 4, 4, 4, 4, 5))])
def test_branches_agree_nonsymmetric_factor(toy, xs_train, r_Y, ranks):
    pr = FactorPrior(toy.prior)
    assert not np.allclose(pr.Lf, pr.Lf.T)
    U, basis, y_bar = _setup(toy, xs_train[:, :20], pr, r_Y, ranks)
    xs = xs_train[:, 20:24]
    a = assemble_rbno_data(toy, U, basis, xs, y_bar, branch="jvp")
    b = assemble_rbno_data(toy, U, basis, xs, y_bar, branch="vjp")
    c = assemble_rbno_data(toy, U, basis, xs, y_bar)
    np.testing.assert_allclose(a.J, b.J, atol=1e-10)
    assert c.meta["branch"] == ("jvp" if basis.W().shape[1] <= r_Y else "vjp")
    assert c.meta["lin_solves_per_sample"] == min(r_Y, basis.W().shape[1])
    for s, x in enumerate(xs.T):
        np.testing.assert_allclose(a.J[s], U.T @ toy.jacobian(toy.solve(x)) @ pr.Lf @ basis.W(), atol=1e-10)


def test_loss_normalisations(toy, xs_train):
    U, basis, y_bar = _setup(toy, xs_train[:, :20], toy.prior, 3, (2, 4, 4, 4, 4, 3))
    d = assemble_rbno_data(toy, U, basis, xs_train[:, 20:30], y_bar)
    z_eta, z_J = np.zeros_like(d.eta), np.zeros_like(d.J)
    assert sobolev_loss(z_eta, d.J, d) == pytest.approx(1.0)        # each sample contributes 1
    assert sobolev_loss(z_eta, z_J, d) == pytest.approx(2.0)
    assert sobolev_loss(z_eta, z_J, d, normalise="global") == pytest.approx(2.0 / d.eta.shape[0])
    none = sobolev_loss(z_eta, z_J, d, normalise="none")
    assert none == pytest.approx(np.mean(np.sum(d.eta ** 2, 1) + np.sum(d.J ** 2, (1, 2))))
    assert sobolev_loss(z_eta, z_J, d, tau=1e6) < 1e-3
    with pytest.raises(ValueError):
        sobolev_loss(z_eta, z_J, d, normalise="bogus")


def test_validation(toy, xs_train):
    U, basis, y_bar = _setup(toy, xs_train[:, :10], toy.prior, 3, (2, 4, 4, 4, 4, 3))
    with pytest.raises(ValueError):
        assemble_rbno_data(toy, U, basis, xs_train[:, :2], y_bar, branch="fd")
    d = assemble_rbno_data(toy, U, basis, xs_train[:, 0], y_bar)  # single 1-D sample
    assert d.J.shape == (1, 3, 3)
    d.eta[:] = 0.0
    with pytest.raises(ValueError):
        sobolev_loss(d.eta, d.J, d)  # zero a0 without tau
