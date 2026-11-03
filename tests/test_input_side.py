"""T6 acceptance: whitened input DIS-TT and its encoder/decoder."""
import numpy as np
import pytest
from conftest import eig_desc, max_principal_angle_sin

from dis_tt import snapshots as sn
from dis_tt.input_side import input_dis_tt


@pytest.fixture(scope="module")
def in_snaps(toy, xs_train):
    return sn.input_dis_snapshots(toy, toy.prior, xs_train, eta_kind="basis", U_out=np.eye(toy.d_y))


def test_full_rank_matches_dense_whitened_dis(toy, xs_train, in_snaps):
    basis = input_dis_tt(in_snaps, toy.in_modes, toy.prior)
    H = toy.H_X_white_reference(xs_train)
    np.testing.assert_allclose(basis.result.spectra[-1][:6], eig_desc(H)[:6], rtol=1e-8)
    V_ref = np.linalg.eigh(H)[1][:, ::-1][:, :3]
    assert max_principal_angle_sin(basis.W()[:, :3], V_ref) < 1e-6


def test_encoder_decoder_algebra(toy, in_snaps):
    basis = input_dis_tt(in_snaps, toy.in_modes, toy.prior, ranks=(2, 4, 4, 4, 4, 3))
    V = basis.V()
    Cinv = np.linalg.inv(toy.prior.covariance())
    np.testing.assert_allclose(V.T @ Cinv @ V, np.eye(3), atol=1e-8)       # E-orthonormal
    xi = np.random.default_rng(0).standard_normal(3)
    np.testing.assert_allclose(basis.encode(basis.decode(xi)), xi, atol=1e-10)
    x = toy.prior.sample(np.random.default_rng(1), 2)
    Qx = basis.decode(basis.encode(x))
    np.testing.assert_allclose(basis.decode(basis.encode(Qx)), Qx, atol=1e-9)  # idempotent


def test_input_dis_beats_input_pca_on_derivatives(toy, xs_train, xs_test, in_snaps):
    """Paper Sec. 4.4.1 qualitative claim, on held-out data, at r_X = 3."""
    basis = input_dis_tt(in_snaps, toy.in_modes, toy.prior, ranks=(2, 4, 4, 4, 4, 3))
    L = toy.prior.apply_sqrt(np.eye(toy.d_x))
    W_dis = basis.W()
    # Input PCA = KL modes v_i = sqrt(mu_i) phi_i (E-normalised); whitened: L^{-1} v_i = phi_i (symmetric L)
    _, W_pca = toy.prior.kl_modes(3)

    def err(W):
        tot = 0.0
        for x in xs_test.T:
            JL = toy.jacobian(toy.solve(x)) @ L
            tot += np.sum((JL - JL @ W @ W.T) ** 2)
        return tot

    assert err(W_dis) < err(W_pca)
