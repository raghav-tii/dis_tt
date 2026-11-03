"""DIS-TT: derivative-informed subspace projectors for tensor-train compression.

Scaffolding for the method in the note "DIS-TT: Derivative-informed projectors
for tensor-train compression" (based on Luo, O'Leary-Roseberry, Chen, Ghattas,
arXiv:2504.08730). Algorithms raise NotImplementedError until their task in
AGENTS.md is done; ``dis_tt.problems.toy`` is complete and exact.

Module map (doc section -> module -> task):
    Sec. 3 TT projectors            tt.py, linalg.py      T1, T2
    Sec. 5 estimators               snapshots.py          T3
    Sec. 6 Alg. A (dense)           dense.py              T4
    Sec. 6 Alg. B (TT-native)       native.py             T5
    Sec. 6 Alg. C (input side)      input_side.py         T6
    Sec. 4 bounds / certificates    bounds.py             T6b
    Sec. 6 Alg. D (RBNO data)       rbno.py               T7, T8
    Sec. 8 ALS refinement           refine.py             T10
"""
from .interfaces import BoundReport, DISTTResult, ForwardModel, GaussianPrior, SnapshotSet, TTForwardModel

__all__ = ["BoundReport", "DISTTResult", "ForwardModel", "GaussianPrior", "SnapshotSet", "TTForwardModel"]
__version__ = "0.0.1"
