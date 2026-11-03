"""Independent regressions for semilinear solve failure reporting."""
import numpy as np
import pytest

from dis_tt.problems.semilinear import SemilinearProblem


def test_newton_exhaustion_is_not_a_solution():
    p = SemilinearProblem(levels=4, newton_maxit=1)
    with pytest.raises(RuntimeError, match="Newton failed to converge"):
        p.solve(np.ones(p.d_x))


def test_zero_iteration_budget_is_not_a_solution():
    p = SemilinearProblem(levels=4, newton_maxit=0)
    with pytest.raises(RuntimeError, match="Newton failed to converge"):
        p.solve(np.ones(p.d_x))


def test_zero_forcing_and_converged_final_iteration():
    p = SemilinearProblem(levels=4, newton_maxit=1)
    st = p.solve(np.zeros(p.d_x))
    np.testing.assert_array_equal(st["y"], 0)
    assert st["res_rel"] == 0
    linear = SemilinearProblem(levels=4, c2=0, newton_maxit=1)
    st = linear.solve(np.ones(linear.d_x))
    assert st["res_rel"] < 1e-12
    np.testing.assert_allclose(st["y"], linear.jvp(st, np.ones(linear.d_x)), rtol=1e-12)
