"""TT-SVD (PCA-TT) vs DIS-TT on parametric function families: values and derivatives.

For each family u(x; m), m ~ N(0, I_p) on a QTT grid, both methods build an r-column
block-TT basis U with ``dis_tt_dense`` (same greedy sweep, same rank cap):
  TT-SVD / PCA-TT : snapshots (u - u_bar) / sqrt(N)        -> captures values
  DIS-TT          : snapshots du/dm_j / sqrt(N), all j      -> captures derivatives
Held-out errors are relative squared projection errors of centred values and of
Jacobians. Analytic Jacobians use the complex step (exact to round-off).

Run:  PYTHONPATH=src python experiments/plot_ttsvd_vs_dis.py
Writes docs/figures/*.png and docs/figures/ttsvd_vs_dis.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dis_tt import snapshots as sn
from dis_tt.dense import dis_tt_dense
from dis_tt.problems.semilinear import SemilinearProblem
from dis_tt.tt import block_to_dense

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#8a8984"
RANKS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
LEVELS = 10
x = (np.arange(2 ** LEVELS) + 0.5) / 2 ** LEVELS
KL_J = np.arange(1, 25)


def bump(m):  # moving, breathing Gaussian pulse (p = 3)
    c, w, a = 0.5 + 0.08 * m[0], 0.04 * np.exp(0.25 * m[1]), 1 + 0.2 * m[2]
    return a * np.exp(-(x - c) ** 2 / (2 * w ** 2))


def chirp(m):  # chirp with random frequency, sweep rate and tilt (p = 3)
    return (1 + 0.3 * m[2] * x) * np.sin(2 * np.pi * (6 + 0.8 * m[0]) * x + 3 * m[1] * x ** 2)


def tanh_field(m):  # saturated Gaussian random field, 24 KL modes (p = 24)
    phi = np.sqrt(2) * np.sin(np.pi * np.outer(x, KL_J)) / KL_J
    return np.tanh(3 * phi @ m)


def complex_step_jac(f, m, h=1e-30):
    return np.stack([np.imag(f(m + 1j * h * e)) / h for e in np.eye(m.size)], axis=1)


class Analytic:
    """Minimal ForwardModel for an analytic f: R^p -> R^{2^LEVELS} with prior N(0, I)."""

    def __init__(self, f, p):
        self.f, self.p = f, p

    def outputs(self, ms):
        return np.stack([self.f(m) for m in ms.T], axis=1)

    def jacobians(self, ms):
        return [complex_step_jac(self.f, m) for m in ms.T]


def analytic_data(f, p, n_train=200, n_test=200, seed=0):
    rng = np.random.default_rng(seed)
    mtr, mte = rng.standard_normal((p, n_train)), rng.standard_normal((p, n_test))
    mdl = Analytic(f, p)
    Y, Yt = mdl.outputs(mtr), mdl.outputs(mte)
    ybar = Y.mean(axis=1)
    Jtr, Jte = mdl.jacobians(mtr), mdl.jacobians(mte)
    return dict(Zpca=(Y - ybar[:, None]) / np.sqrt(n_train), Zdis=np.hstack(Jtr) / np.sqrt(n_train),
                Yc_test=Yt - ybar[:, None], J_test=Jte, modes=(2,) * LEVELS)


def pde_data(n_train=100, n_test=60):
    prob = SemilinearProblem(levels=8)
    xs, xt = prob.prior.sample(np.random.default_rng(10), n_train), prob.prior.sample(np.random.default_rng(11), n_test)
    P, ybar = sn.output_pca_snapshots(prob, xs)
    Dz = sn.output_dis_snapshots(prob, prob.prior, xs, direction_kind="exact").Z
    Lw = prob.prior.apply_sqrt(np.eye(prob.prior.white_dim))  # whitened Jacobian D_E F = DF L
    Yt, Jt = [], []
    for xx in xt.T:
        st = prob.solve(xx)
        Yt.append(prob.output(st))
        Jt.append(prob.jvp(st, Lw))
    global x_pde
    x_pde = prob.s
    return dict(Zpca=P.Z, Zdis=Dz, Yc_test=np.stack(Yt, axis=1) - ybar[:, None], J_test=Jt, modes=prob.out_modes)


def rel_err(U, A):
    R = A - U @ (U.T @ A)
    return float(np.sum(R * R) / np.sum(A * A))


def basis(Z, modes, r):
    res = dis_tt_dense(Z, modes, rmax=r, precompress=True)
    return block_to_dense(res.cores), res


def sweep(data):
    Jall = np.hstack(data["J_test"])
    rows = []
    for r in RANKS:
        for name in ("pca", "dis"):
            U, res = basis(data["Z" + name], data["modes"], r)
            rows.append(dict(method=name, rmax=r, r=U.shape[1], ranks=res.ranks,
                             params=int(sum(G.size for G in res.cores)),
                             value_err=rel_err(U, data["Yc_test"]), deriv_err=rel_err(U, Jall)))
    return rows


def figure(title, grid, data, rows, r_show, j_show, fname, samples):
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.2))
    for a in ax.flat:
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(colors="#52514e", labelsize=8)
    a = ax[0, 0]
    for s in samples.T:
        a.plot(grid, s, lw=1.2, alpha=0.8)
    a.set_title("(a) six realisations u(x; m)", fontsize=10, loc="left")
    Up, _ = basis(data["Zpca"], data["modes"], r_show)
    Ud, _ = basis(data["Zdis"], data["modes"], r_show)
    v, J = data["Yc_test"][:, 0], data["J_test"][0][:, j_show]
    for a, tgt, lab in ((ax[0, 1], v, "centred value u - ū (test sample)"),
                        (ax[1, 0], J, f"derivative ∂u/∂m_{j_show + 1} (test sample)")):
        a.plot(grid, tgt, color=INK, lw=2.4, label="truth")
        for U, c, n in ((Up, BLUE, "TT-SVD (PCA-TT)"), (Ud, ORANGE, "DIS-TT")):
            proj = U @ (U.T @ tgt)
            e = np.linalg.norm(tgt - proj) / np.linalg.norm(tgt)
            a.plot(grid, proj, color=c, lw=1.6, ls="--", label=f"{n}: rel err {e:.2f}")
        a.set_title(f"({'b' if a is ax[0, 1] else 'c'}) {lab}, r = {r_show}", fontsize=10, loc="left")
        a.legend(fontsize=8, frameon=False)
    a = ax[1, 1]
    for name, c, n in (("pca", BLUE, "TT-SVD"), ("dis", ORANGE, "DIS-TT")):
        rr = [q for q in rows if q["method"] == name]
        a.semilogy([q["r"] for q in rr], [q["value_err"] for q in rr], "-o", color=c, ms=4, lw=2, label=f"{n}: values")
        a.semilogy([q["r"] for q in rr], [q["deriv_err"] for q in rr], "--s", color=c, ms=4, lw=2, label=f"{n}: derivatives")
    a.set_xlabel("basis size r (TT columns)", fontsize=9)
    a.set_ylabel("held-out rel. squared error", fontsize=9)
    a.set_title("(d) error vs basis size", fontsize=10, loc="left")
    a.grid(alpha=0.25)
    a.legend(fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=140)
    plt.close(fig)


def summary_figure(allrows):
    names = list(allrows)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    w = 0.38
    for k, key in enumerate(("value_err", "deriv_err")):
        a = ax[k]
        for i, (m, c, lab) in enumerate((("pca", BLUE, "TT-SVD"), ("dis", ORANGE, "DIS-TT"))):
            vals = [next(q[key] for q in allrows[n] if q["method"] == m and q["rmax"] == 8) for n in names]
            a.bar(np.arange(len(names)) + (i - 0.5) * w, vals, w * 0.95, color=c, label=lab)
        a.set_yscale("log")
        a.set_xticks(np.arange(len(names)), names, fontsize=9)
        a.set_title(("values" if k == 0 else "derivatives") + " — held-out rel. sq. error, rmax = 8", fontsize=10, loc="left")
        a.spines[["top", "right"]].set_visible(False)
        a.grid(axis="y", alpha=0.25)
    ax[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "summary_rmax8.png", dpi=140)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cases = {
        "bump": ("Moving Gaussian pulse  u = a·exp(-(x-c)²/2w²),  p = 3", bump, 3, 4, 0),
        "chirp": ("Random chirp  u = (1+0.3 m₃x)·sin(2π(6+0.8m₁)x + 3m₂x²),  p = 3", chirp, 3, 6, 1),
        "tanh-field": ("Saturated random field  u = tanh(3 Σ m_j √2 sin(jπx)/j),  p = 24", tanh_field, 24, 6, 0),
    }
    allrows = {}
    for key, (title, f, p, r_show, j_show) in cases.items():
        data = analytic_data(f, p)
        rows = sweep(data)
        allrows[key] = rows
        samples = np.stack([f(m) for m in np.random.default_rng(99).standard_normal((6, p))], axis=1)
        figure(title, x, data, rows, r_show, j_show, f"{key}.png", samples)
        print(key, "done")
    data = pde_data()
    rows = sweep(data)
    allrows["semilinear-PDE"] = rows
    samples = data["Yc_test"][:, :6] + 0  # centred outputs as realisations
    figure("Semilinear elliptic PDE (paper Sec. 4.1), whitened Jacobian, d_Y = 256 (centred outputs in (a))",
           x_pde, data, rows, 4, 128, "semilinear.png", samples)
    summary_figure(allrows)
    (OUT / "ttsvd_vs_dis.json").write_text(json.dumps(allrows, indent=1))
    for k, rows in allrows.items():
        print(k)
        for q in rows:
            print(f"  {q['method']} rmax={q['rmax']:2d} r={q['r']:2d} params={q['params']:5d} "
                  f"val={q['value_err']:.3e} der={q['deriv_err']:.3e}")


if __name__ == "__main__":
    main()
