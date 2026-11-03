"""Run from the review worktree: OPENBLAS_NUM_THREADS=1 PYTHONPATH=src ../../.venv/bin/python notes/codex_review_measure.py."""
import time
import numpy as np
from dis_tt import tt, snapshots, bounds
from dis_tt.dense import dis_tt_dense
from dis_tt.native import dis_tt_native
from dis_tt.problems.toy import ToyProblem

def timing(fn):
    fn()
    times = []
    for _ in range(5):
        start = time.perf_counter(); fn(); times.append(time.perf_counter()-start)
    return float(np.median(times))

rng = np.random.default_rng(81)
for T in (2000, 4000):
    Z = rng.normal(size=(1024, T))
    print('dense', T, timing(lambda: dis_tt_dense(Z, (2,)*10, rmax=6)))

def make(d):
    ranks = (1,) + (3,)*(d-1) + (1,)
    return tt.TensorTrain([rng.normal(size=(ranks[k], 2, ranks[k+1])) / 3 for k in range(d)])
for d,T in ((8,100), (8,200), (8,400), (6,200), (10,200)):
    zs = [make(d) for _ in range(T)]
    print('native', d, T, timing(lambda: dis_tt_native(zs, rmax=4)))
p = ToyProblem(out_levels=7, in_levels=6, p=8, seed=1)
xs = p.prior.sample(np.random.default_rng(10),40)
xt = p.prior.sample(np.random.default_rng(11),40)
Z = snapshots.output_dis_snapshots(p,p.prior,xs,direction_kind='exact').Z
Zt = snapshots.output_dis_snapshots(p,p.prior,xt,direction_kind='exact').Z
r = dis_tt_dense(Z,p.out_modes,eps=.05)
print('eps .05',r.ranks, r.tails.tolist())
r = dis_tt_dense(Z,p.out_modes,ranks=(2,3,4,4,4,3,3))
b = bounds.bound_report(r,Z,Zt)
print('fixed tails',r.tails.tolist())
print('bounds lower/train/upper/test',b.lower_unprojected,b.train_error,b.upper_unprojected,b.test_error)
for g in (64,1):
    print('bootstrap',g,bounds.bootstrap_tails(Z,p.out_modes,r.ranks,n_boot=50,group_size=g).tolist())
