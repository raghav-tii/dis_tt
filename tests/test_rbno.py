"""T7 acceptance: RBNO data assembly with both JVP and VJP branches."""
import numpy as np
import pytest

from dis_tt import snapshots as sn
from dis_tt.input_side import input_dis_tt
from dis_tt.rbno import assemble_rbno_data, sobolev_loss


@pytest.fixture(scope="module")
def setup(toy, xs_train, Z_exact):
    U = np.linalg.eigh(Z_exact @ Z_exact.T)[1][:, ::-1][:, :4]  # dense output DIS, r_Y = 4
    S = sn.input_dis_snapshots(toy, toy.prior, xs_train, eta_kind="basis", U_out=U)
    basis = input_dis_tt(S, toy.in_modes, toy.prior, ranks=(2, 4, 4, 4, 4, 3))  # r_X = 3
    y_bar = np.mean([toy.output(toy.solve(x)) for x in xs_train.T], axis=0)
    return U, basis, y_bar


def test_reduced_jacobian_branches_agree(toy, xs_test, setup):
    U, basis, y_bar = setup
    xs = xs_test[:, :5]
    a = assemble_rbno_data(toy, U, basis, xs, y_bar, branch="jvp")
    b = assemble_rbno_data(toy, U, basis, xs, y_bar, branch="vjp")
    np.testing.assert_allclose(a.J, b.J, atol=1e-10)
    LW = basis.V()
    for s, x in enumerate(xs.T):
        st = toy.solve(x)
        np.testing.assert_allclose(a.J[s], U.T @ toy.jacobian(st) @ LW, atol=1e-10)
        np.testing.assert_allclose(a.eta[s], U.T @ (toy.output(st) - y_bar), atol=1e-12)
        np.testing.assert_allclose(a.xi[s], basis.encode(x), atol=1e-12)
    assert a.J.shape == (5, 4, 3)


def test_sobolev_loss_zero_at_truth(toy, xs_test, setup):
    U, basis, y_bar = setup
    d = assemble_rbno_data(toy, U, basis, xs_test[:, :5], y_bar)
    assert sobolev_loss(d.eta, d.J, d) == pytest.approx(0.0, abs=1e-20)
    assert sobolev_loss(d.eta * 0, d.J, d) > 0
