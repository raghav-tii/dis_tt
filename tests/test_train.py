"""T8: latent network training (optional JAX). Overfit 20 toy RBNO samples to loss < 1e-3."""
import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("optax")

from dis_tt import snapshots as sn
from dis_tt.input_side import input_dis_tt
from dis_tt.rbno import assemble_rbno_data, sobolev_loss, train_latent_network


@pytest.fixture(scope="module")
def data20(toy, xs_train, Z_exact):
    U = np.linalg.eigh(Z_exact @ Z_exact.T)[1][:, ::-1][:, :4]
    S = sn.input_dis_snapshots(toy, toy.prior, xs_train, eta_kind="basis", U_out=U)
    basis = input_dis_tt(S, toy.in_modes, toy.prior, ranks=(2, 4, 4, 4, 4, 3))
    y_bar = np.mean([toy.output(toy.solve(x)) for x in xs_train.T], axis=0)
    return assemble_rbno_data(toy, U, basis, xs_train[:, :20], y_bar)


def test_paper_protocol_configuration(data20):
    """Paper Table 2 (semilinear row) defaults: depth 6, width 2r, softplus, Adam, batch 25, LR 1e-3,
    3000 epochs halved 5x from epoch 2250. Short run checks the wiring only."""
    net = train_latent_network(data20, epochs=4, halving_start=1, n_halvings=2)
    assert net.meta["depth"] == 6 and net.meta["width"] == 8 and net.meta["activation"] == "softplus"
    assert net.meta["batch_size"] == 20 and net.meta["lr"] == 1e-3 and net.meta["lr_halving_epochs"] == [1, 2]
    full = train_latent_network(data20, epochs=0)  # schedule only, no steps
    assert full.meta["epochs"] == 0


@pytest.mark.slow
def test_overfit_20_samples(data20):
    """Overfit to < 1e-3. Exact Table 2 budget (width 2r = 8, 3000 epochs = 3000 Adam steps at N=20)
    reaches only ~0.215 here, so this test widens to 16 r and trains 60000 epochs (see notes/T8.md)."""
    net = train_latent_network(data20, depth=6, width_factor=16, lr=1e-3, epochs=60000, seed=0)
    assert net.meta["width"] == 64
    assert net.meta["lr_halving_epochs"] == [45000, 48000, 51000, 54000, 57000]
    loss = sobolev_loss(net.predict(data20.xi), net.jacobian(data20.xi), data20)
    assert loss == pytest.approx(net.meta["final_train_loss"], rel=1e-6)
    assert loss < 1e-3, loss


def test_autodiff_jacobian_matches_finite_differences(data20):
    net = train_latent_network(data20, depth=4, epochs=5, seed=1)
    xi = data20.xi[:3]
    Jn = net.jacobian(xi)
    h = 1e-6
    for b in range(xi.shape[1]):
        e = np.zeros(xi.shape[1]); e[b] = h
        fd = (net.predict(xi + e) - net.predict(xi - e)) / (2 * h)
        np.testing.assert_allclose(Jn[:, :, b], fd, atol=1e-7)
