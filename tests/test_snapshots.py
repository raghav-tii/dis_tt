"""T3 acceptance: estimators reproduce the target operators (doc Sec. 5)."""
import numpy as np
import pytest

from dis_tt import snapshots as sn


def _H_Y_from_Z(Z):
    return Z @ Z.T


def test_output_exact_and_full_kl_equal_reference(toy, xs_train):
    H_ref = toy.H_Y_reference(xs_train)
    for kind, k in [("exact", None), ("kl", toy.d_x)]:
        S = sn.output_dis_snapshots(toy, toy.prior, xs_train, direction_kind=kind, n_dirs=k or 0)
        assert S.Z.shape[0] == toy.d_y and S.n_samples == xs_train.shape[1]
        np.testing.assert_allclose(_H_Y_from_Z(S.Z), H_ref, rtol=1e-9, atol=1e-12 * np.abs(H_ref).max())


def test_output_kl_truncated_is_lower_bound(toy, xs_train):
    H_ref = toy.H_Y_reference(xs_train)
    S = sn.output_dis_snapshots(toy, toy.prior, xs_train, direction_kind="kl", n_dirs=8)
    assert S.Z.shape[1] == 8 * xs_train.shape[1]
    gap = np.linalg.eigvalsh(H_ref - S.Z @ S.Z.T)
    assert gap.min() > -1e-10 * np.trace(H_ref)  # H_ref - H_KL is PSD


@pytest.mark.slow
def test_output_random_is_unbiased_in_trace(toy, xs_train):
    H_ref = toy.H_Y_reference(xs_train)
    S = sn.output_dis_snapshots(toy, toy.prior, xs_train, direction_kind="random", n_dirs=400,
                                rng=np.random.default_rng(5))
    assert S.Z.shape[1] == 400 * xs_train.shape[1]
    assert abs(np.sum(S.Z**2) / np.trace(H_ref) - 1) < 0.05


def test_output_pca(toy, xs_train):
    S, y_bar = sn.output_pca_snapshots(toy, xs_train)
    Y = np.stack([toy.output(toy.solve(x)) for x in xs_train.T], axis=1)
    np.testing.assert_allclose(y_bar, Y.mean(1), atol=1e-13)
    C = (Y - y_bar[:, None]) @ (Y - y_bar[:, None]).T / Y.shape[1]
    np.testing.assert_allclose(S.Z @ S.Z.T, C, atol=1e-12)


def test_input_basis_kind_exact(toy, xs_train):
    """eta = columns of U; with U = I this is exactly H_X^w."""
    S = sn.input_dis_snapshots(toy, toy.prior, xs_train, eta_kind="basis", U_out=np.eye(toy.d_y))
    H_ref = toy.H_X_white_reference(xs_train)
    np.testing.assert_allclose(S.Z @ S.Z.T, H_ref, rtol=1e-9, atol=1e-12 * np.abs(H_ref).max())
    # projected version with a 3-column U
    U = np.linalg.qr(np.random.default_rng(0).standard_normal((toy.d_y, 3)))[0]
    S3 = sn.input_dis_snapshots(toy, toy.prior, xs_train, eta_kind="basis", U_out=U)
    L = toy.prior.apply_sqrt(np.eye(toy.d_x))
    H3 = sum((U.T @ toy.jacobian(toy.solve(x)) @ L).T @ (U.T @ toy.jacobian(toy.solve(x)) @ L)
             for x in xs_train.T) / xs_train.shape[1]
    np.testing.assert_allclose(S3.Z @ S3.Z.T, H3, rtol=1e-9, atol=1e-14)


@pytest.mark.slow
def test_input_gaussian_unbiased_in_trace(toy, xs_train):
    S = sn.input_dis_snapshots(toy, toy.prior, xs_train, eta_kind="gaussian", n_dirs=400,
                               rng=np.random.default_rng(6))
    assert abs(np.sum(S.Z**2) / np.trace(toy.H_X_white_reference(xs_train)) - 1) < 0.05


def test_mixed_weights(toy, xs_train):
    P, _ = sn.output_pca_snapshots(toy, xs_train)
    Dd = sn.output_dis_snapshots(toy, toy.prior, xs_train, direction_kind="kl", n_dirs=8)
    M = sn.mixed_snapshots(P, Dd, alpha=0.3)
    assert M.Z.shape[1] == P.Z.shape[1] + Dd.Z.shape[1]
    assert np.sum(M.Z**2) == pytest.approx(1.0)
    assert np.sum(M.Z[:, : P.Z.shape[1]] ** 2) == pytest.approx(0.3)
