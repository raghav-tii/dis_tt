"""T6 extra tests: D forced, non-symmetric square factor, and the OPEN ISSUE (wide factor L).

Wide-L findings (see notes/T6.md): with L (d_X x w), w > d_X, C = L L^T, L^+ = L^T C^{-1},
P = L^+ L the orthogonal projector onto range(L^T) in white space:
    Q_TT = L W W^T L^+ is a projector  <=>  (I - P) W = 0   (when L W has full column rank).
The whitened snapshots lie in range(L^T), but greedy TT truncation does NOT keep W there.
The library does not support wide L; these tests document the behaviour.
"""
import numpy as np
import pytest

from dis_tt import snapshots as sn
from dis_tt.input_side import input_dis_tt


class FactorPrior:
    """Same C_X as the toy prior, factor L' = L Q (Q with orthonormal rows; wide if w > d_X)."""

    def __init__(self, base, Q):
        self.base, self.white_dim = base, Q.shape[1]
        self.Lf = base.apply_sqrt(np.eye(base.dim)) @ Q
        self.C = self.Lf @ self.Lf.T
        self.Lpinv = self.Lf.T @ np.linalg.inv(self.C)  # L^+ (= L^{-1} when square)

    def sample(self, rng, n):
        return self.Lf @ rng.standard_normal((self.white_dim, n))

    def apply_sqrt(self, v):
        return self.Lf @ v

    def apply_sqrt_T(self, v):
        return self.Lf.T @ v

    def apply_inv_sqrt(self, v):
        return self.Lpinv @ v

    def kl_modes(self, k):
        return self.base.kl_modes(k)


def _orth_rows(n, w, seed):
    return np.linalg.qr(np.random.default_rng(seed).standard_normal((w, n)))[0].T


@pytest.fixture(scope="module")
def wide(toy, xs_train):
    pr = FactorPrior(toy.prior, _orth_rows(toy.d_x, 2 * toy.d_x, 7))  # white_dim 128 = 2^7
    S = sn.input_dis_snapshots(toy, pr, xs_train, eta_kind="basis", U_out=np.eye(toy.d_y))
    return pr, S


def _diagnostics(basis, pr):
    W = basis.W()
    P = pr.Lpinv @ pr.Lf
    Q = pr.Lf @ W @ W.T @ pr.Lpinv
    return (np.linalg.norm(W - P @ W, 2), np.linalg.norm(Q @ Q - Q, 2),
            np.linalg.norm(basis.V().T @ np.linalg.inv(pr.C) @ basis.V() - np.eye(W.shape[1]), 2))


def test_D_is_forced(toy, xs_train):
    S = sn.input_dis_snapshots(toy, toy.prior, xs_train, eta_kind="basis", U_out=np.eye(toy.d_y))
    assert input_dis_tt(S, toy.in_modes, toy.prior, eps=0.1, D=len(toy.in_modes)).result.is_basis
    with pytest.raises(ValueError):
        input_dis_tt(S, toy.in_modes, toy.prior, eps=0.1, D=5)


def test_nonsymmetric_square_factor(toy, xs_train):
    """Square non-symmetric L: encoder uses L^{-1}, E-orthonormality and idempotence hold."""
    pr = FactorPrior(toy.prior, _orth_rows(toy.d_x, toy.d_x, 3))
    assert not np.allclose(pr.Lf, pr.Lf.T)
    S = sn.input_dis_snapshots(toy, pr, xs_train, eta_kind="basis", U_out=np.eye(toy.d_y))
    b = input_dis_tt(S, toy.in_modes, pr, ranks=(2, 4, 4, 4, 4, 3))
    ker, idem, eorth = _diagnostics(b, pr)
    assert ker < 1e-12 and idem < 1e-10 and eorth < 1e-10
    xi = np.random.default_rng(0).standard_normal((3, 2))
    np.testing.assert_allclose(b.encode(b.decode(xi)), xi, atol=1e-10)


def test_wide_snapshots_live_in_range_LT(wide):
    pr, S = wide
    P = pr.Lpinv @ pr.Lf
    assert np.linalg.norm(S.Z - P @ S.Z) < 1e-12 * np.linalg.norm(S.Z)


def test_wide_full_rank_is_projector(wide):
    """No truncation (numerical full rank): W spans range(Z) within range(L^T) -> projector."""
    pr, S = wide
    b = input_dis_tt(S, (2,) * 7, pr)
    ker, idem, eorth = _diagnostics(b, pr)
    assert ker < 1e-6 and idem < 1e-10 and eorth < 1e-10


@pytest.mark.parametrize("kw", [dict(eps=0.1), dict(eps=0.01), dict(ranks=(2, 4, 4, 4, 4, 4, 3))])
def test_wide_truncated_leaks_into_kernel(wide, kw):
    """Finding: TT truncation puts W partly in ker(L); Q_TT is then NOT a projector and V is not
    E-orthonormal. The criterion idempotent <=> (I - P) W = 0 is checked numerically."""
    pr, S = wide
    b = input_dis_tt(S, (2,) * 7, pr, **kw)
    ker, idem, eorth = _diagnostics(b, pr)
    assert ker > 0.1 and idem > 1e-2 and eorth > 1e-2
    # remedy suggested in the docstring (test-only, NOT library behaviour): project onto range(L^T)
    P = pr.Lpinv @ pr.Lf
    W = b.W()
    Wp = np.linalg.qr(P @ W)[0]
    Q = pr.Lf @ Wp @ Wp.T @ pr.Lpinv
    assert np.linalg.norm(Q @ Q - Q, 2) < 1e-10
    err = lambda B: float(np.sum((S.Z - B @ (B.T @ S.Z)) ** 2))
    assert err(Wp) <= err(W) * (1 + 1e-12)  # never worse on training data (Z in range(P))


def test_wide_rank_above_dX_must_hit_kernel(wide):
    """r_X > d_X = dim range(L^T): W cannot avoid ker(L), whatever the algorithm."""
    pr, S = wide
    b = input_dis_tt(S, (2,) * 7, pr, ranks=(2, 4, 8, 16, 32, 64, 70))
    assert _diagnostics(b, pr)[0] > 0.99
