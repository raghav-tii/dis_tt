# dis_tt: derivative-informed projectors for tensor-train compression

Scaffolding for **DIS-TT**. It replaces the per-bond PCA step of TT-SVD with the
derivative-informed subspace (DIS) eigenproblem of Luo, O'Leary-Roseberry, Chen & Ghattas
(arXiv:2504.08730). The result is a TT basis whose H^1-seminorm truncation error is certified
by the discarded per-bond eigenvalues.

- Theory, condensed: `docs/theory.md`. Full note: https://claude.ai/artifact/Jjr4SEdweverCdqwKStCyn
- Tasks, conventions and agent prompts: `AGENTS.md`
- Running agents (skill, orchestrator, gates): `agents/README.md`
- Progress log and deviations: `notes/` (one file per task)

## Status

| Component | State |
| --- | --- |
| Interfaces, result types (`interfaces.py`) | done |
| Toy problem with exact Jacobians and prior (`problems/toy.py`) | done, tested |
| Acceptance tests (`tests/`) | done; verified satisfiable against a private reference implementation |
| TT toolkit, linalg, snapshots, Algorithms A-D, bounds, ALS | stubs, tasks T1-T10 |

```
pytest -q                       # 5 pass (fixtures); the rest fail until T1-T7 are done (no install needed)
pytest -q -m "slow or not slow" # include statistical tests
```

## Layout

```
src/dis_tt/
  interfaces.py   protocols (ForwardModel, GaussianPrior, TTForwardModel), DISTTResult, SnapshotSet
  tt.py           T1  minimal TT toolkit
  linalg.py       T2  rank rule, bond eigensolvers
  snapshots.py    T3  JVP / VJP / PCA / mixed snapshot builders (note Sec. 5)
  dense.py        T4  Algorithm A
  native.py       T5  Algorithm B
  input_side.py   T6  Algorithm C (whitened TT active subspace)
  bounds.py       T6b certificates, Prop. 1/2 checks, bootstrap
  rbno.py         T7/T8 Algorithm D (reduced Jacobians, Sobolev loss, training)
  refine.py       T10 ALS refinement (optional)
  problems/toy.py exact test problem
experiments/run_checklist.py   T9  the six experiments of note Sec. 8
agents/                        skill, orchestrator, run guide
notes/                         per-task implementation logs
```
