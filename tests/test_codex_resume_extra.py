"""Fresh independent audit checks; existing tests remain untouched."""
import numpy as np
import pytest
from dis_tt import tt, snapshots
from dis_tt.dense import dis_tt_dense
from dis_tt.interfaces import SnapshotSet
from dis_tt.input_side import input_dis_tt


@pytest.mark.parametrize('backend', ['gram', 'svd', 'randomized'])
def test_each_projected_tail_matches_direct_energy_loss(backend):
    Z = np.random.default_rng(612).normal(size=(30, 17))
    result = dis_tt_dense(Z, (2, 3, 5), ranks=(1, 2), backend=backend,
                          oversample=0, power_iters=0)
    previous = Z
    for k, tail in enumerate(result.tails):
        B = tt.subspace_matrix(result.cores[:k + 1], (2, 3, 5))
        projected = B @ (B.T @ Z)
        assert tail == pytest.approx(np.sum((previous - projected)**2), rel=1e-12)
        previous = projected


class Prior:
    white_dim = 4
    L = np.array([[1., 2., 0., 0.], [0., 1., 1., 0.]])
    def apply_sqrt(self, v): return self.L @ v
    def apply_sqrt_T(self, v): return self.L.T @ v
    def apply_inv_sqrt(self, v): return np.linalg.pinv(self.L) @ v


class Linear:
    A = np.array([[1., 2.], [3., -1.], [2., 1.]])
    def solve(self, x): return x
    def output(self, x): return self.A @ x
    def jvp(self, x, h): return self.A @ h
    def vjp(self, x, h): return self.A.T @ h


def test_snapshot_scaling_and_transpose_on_explicit_rectangular_factor():
    m, p = Linear(), Prior()
    J = m.A @ p.L
    for N in (1, 7):
        xs = np.zeros((2, N))
        out = snapshots.output_dis_snapshots(m, p, xs, direction_kind='exact')
        inp = snapshots.input_dis_snapshots(m, p, xs, eta_kind='basis', U_out=np.eye(3))
        np.testing.assert_allclose(out.Z @ out.Z.T, J @ J.T, atol=1e-13)
        np.testing.assert_allclose(inp.Z @ inp.Z.T, J.T @ J, atol=1e-13)


def test_minimal_wide_factor_truncation_is_not_physical_projector():
    p = Prior()
    Z = p.L.T @ np.array([[1., 2.], [3., -1.]])
    s = SnapshotSet(Z=Z, kind='dis_in', n_samples=1, n_dirs=2)
    basis = input_dis_tt(s, (2, 2), p, ranks=(1, 1), backend='svd')
    Q = basis.decode(basis.encode(np.eye(2)))
    assert np.linalg.norm(Q @ Q - Q) > .01
    W = basis.W()
    P = np.linalg.pinv(p.L) @ p.L
    np.testing.assert_allclose(Q @ Q, p.L @ W @ (W.T @ P @ W) @ W.T @ np.linalg.pinv(p.L))
