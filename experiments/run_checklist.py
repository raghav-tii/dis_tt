"""Experiments from note Sec. 8 'First experiments, in order' (TASK T9: run AFTER T1-T7 pass).

Each function writes ``results/<name>.csv`` (+ ``results/<name>.json`` with headline numbers) and
returns a dict of headline numbers. No function may change library code; failures of a
theoretical claim are RESULTS to report, not bugs to hide.

Run:  PYTHONPATH=src python experiments/run_checklist.py exp1 exp2 ...

Problems: ``ToyProblem`` (exact, dense) and ``SemilinearProblem`` (paper Sec. 4.1 by FD, see its
module docstring for the deviations from the paper's FEM setup). Errors are normalised as in the
paper's reconstruction plots: squared error summed over test samples divided by the squared second
moment of the test data (||F||^2 for outputs, ||D F L||^2 for derivatives).
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

from dis_tt import snapshots as sn
from dis_tt.bounds import bootstrap_tails, bound_report, projection_error
from dis_tt.dense import dis_tt_dense
from dis_tt.input_side import input_dis_tt
from dis_tt.problems.semilinear import SemilinearProblem
from dis_tt.problems.toy import ToyProblem
from dis_tt.tt import block_to_dense

RESULTS = Path(__file__).resolve().parents[1] / "results"


# ------------------------------------------------------------------ helpers
def _write(name: str, rows: list[dict], summary: dict) -> dict:
    RESULTS.mkdir(exist_ok=True)
    if rows:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(RESULTS / f"{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: _fmt(v) for k, v in r.items()})
    with open(RESULTS / f"{name}.json", "w") as f:
        json.dump(_jsonable(summary), f, indent=2)
    return summary


def _fmt(v):
    if isinstance(v, (tuple, list, np.ndarray)):
        return " ".join(f"{x:.6g}" if isinstance(x, (float, np.floating)) else str(x) for x in np.asarray(v).ravel().tolist())
    return f"{v:.6g}" if isinstance(v, (float, np.floating)) else v


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    return o


def _sin(A, B):
    return float(np.linalg.norm(B - A @ (A.T @ B), 2))


def _params(cores) -> int:
    return int(sum(G.size for G in cores))


def _rel_proj(B, Z):
    R = Z - B @ (B.T @ Z)
    return float(np.sum(R * R) / np.sum(Z * Z))


def _outputs(prob, xs):
    return np.stack([prob.output(prob.solve(x)) for x in xs.T], axis=1)


def _exact_out(prob, xs):
    return sn.output_dis_snapshots(prob, prob.prior, xs, direction_kind="exact").Z


def _exact_in(prob, xs):
    return sn.input_dis_snapshots(prob, prob.prior, xs, eta_kind="basis", U_out=np.eye(prob.d_y)).Z


def _sqrt_factor(H):
    lam, V = np.linalg.eigh(0.5 * (H + H.T))
    keep = lam > 0
    return V[:, keep] * np.sqrt(lam[keep])  # Z Z^T = H (clipped), d_Y columns


# ------------------------------------------------------------------ experiments
def exp1_consistency():
    """D = d, full ranks reproduce dense DIS (eigenvalues, basis) to round-off.

    Problems: toy (ToyProblem(8, 7)) and, once T9-PDE exists, the paper's 1D semilinear
    elliptic problem on 256 interior nodes (QTT d = 8).
    Report: max rel. eigenvalue error (top 10), sin(theta_max) of the top-r subspaces, r = 2..8.
    """
    rows, summary = [], {}
    for name, prob in [("toy(8,7)", ToyProblem(out_levels=8, in_levels=7, p=10, seed=0)),
                       ("semilinear d=8", SemilinearProblem(levels=8))]:
        xs = prob.prior.sample(np.random.default_rng(100), 40)
        Z = _exact_out(prob, xs)
        t0 = time.perf_counter()
        res = dis_tt_dense(Z, prob.out_modes)  # eps=None: full numerical ranks, D = d
        t = time.perf_counter() - t0
        lam, V = np.linalg.eigh(Z @ Z.T)
        lam, V = lam[::-1], V[:, ::-1]
        U = block_to_dense(res.cores)
        k = min(10, len(res.spectra[-1]))
        eig_err = float(np.max(np.abs(res.spectra[-1][:k] - lam[:k]) / lam[:k]))
        sins = {r: _sin(U[:, :r], V[:, :r]) for r in range(2, 9)}
        gaps = {r: float((lam[r - 1] - lam[r]) / lam[0]) for r in range(2, 9)}
        for r in range(2, 9):
            rows.append({"problem": name, "r": r, "sin_theta_max": sins[r], "rel_gap_lam_r": gaps[r]})
        summary[name] = {"T": Z.shape[1], "ranks": res.ranks, "max_rel_eig_err_top10": eig_err,
                         "max_sin_r2_8": max(sins.values()), "sin_by_r": sins, "cert": res.cert,
                         "lam_top10": lam[:10], "seconds": t}
    return _write("exp1_consistency", rows, summary)


def exp2_bound_check():
    """Prop. 1 / Prop. 2 on held-out data.

    For eps in {0.3, 0.1, 0.03, 0.01} and N in {10, 40, 160}: train_error (== sum tails),
    test_error, lower_unprojected, upper_unprojected, bootstrap std per bond.
    Pass criterion for the THEORY (report, don't assert): test_error <= upper + sampling term.

    Sampling term (Prop. 2): sum_k sqrt(2 r_k M_k / N), M_k = E||Tr_{>k}(D_E F D_E F^*) - H^(k)||_HS^2
    is unknown; we report a PLUG-IN estimate from the training samples (sample variance of the
    per-sample partial traces about their mean). Problem: semilinear d = 8, exact directions.
    """
    prob = SemilinearProblem(levels=8)
    modes, dX = prob.out_modes, prob.d_x
    Z_test = _exact_out(prob, prob.prior.sample(np.random.default_rng(201), 200))
    rows, summary = [], {}
    for N in (10, 40, 160):
        Z = _exact_out(prob, prob.prior.sample(np.random.default_rng(200 + N), N))
        # plug-in M_k: per-sample Tr_{>k}(D D^*) = N * Gram of left unfolding of that sample's block
        blocks = Z.reshape(Z.shape[0], N, dX)
        M = []
        for k in range(1, len(modes) + 1):
            m = 2 ** k
            G = np.stack([(lambda B: B @ B.T)(blocks[:, s, :].reshape(m, -1)) for s in range(N)]) * N
            Hk = G.mean(0)
            M.append(float(np.mean(np.sum((G - Hk) ** 2, axis=(1, 2)))))
        M = np.array(M)
        for eps in (0.3, 0.1, 0.03, 0.01):
            res = dis_tt_dense(Z, modes, eps=eps, precompress=True)
            rep = bound_report(res, Z, Z_test)
            boot = bootstrap_tails(Z, modes, res.ranks, n_boot=20, rng=np.random.default_rng(N), group_size=dX)
            samp = float(np.sum(np.sqrt(2 * np.array(res.ranks) * M[: res.D] / N)))
            row = {"N": N, "eps": eps, "ranks": res.ranks, "params": _params(res.cores),
                   "total_train": res.total, "total_test": float(np.sum(Z_test ** 2)),
                   "train_error": rep.train_error, "sum_tails": rep.sum_tails, "test_error": rep.test_error,
                   "lower_unprojected": rep.lower_unprojected, "upper_unprojected": rep.upper_unprojected,
                   "sampling_term_plugin": samp, "boot_std_sum": float(boot.sum()), "boot_std": boot,
                   "train_rel": rep.train_error / res.total, "test_rel": rep.test_error / float(np.sum(Z_test ** 2)),
                   "cert": res.cert, "eps2": eps ** 2,
                   "test_le_upper": bool(rep.test_error <= rep.upper_unprojected),
                   "test_le_upper_plus_sampling": bool(rep.test_error <= rep.upper_unprojected + samp),
                   "test_le_train_plus_2boot": bool(rep.test_error <= rep.train_error + 2 * boot.sum()),
                   "identity_rel_gap": abs(rep.train_error - rep.sum_tails) / max(rep.sum_tails, 1e-300)}
            rows.append(row)
        summary[f"N={N}"] = {"M_k_plugin": M}
    summary["rows"] = rows
    summary["n_test_le_upper"] = sum(r["test_le_upper"] for r in rows)
    summary["n_test_le_upper_plus_sampling"] = sum(r["test_le_upper_plus_sampling"] for r in rows)
    summary["max_identity_rel_gap"] = max(r["identity_rel_gap"] for r in rows)
    return _write("exp2_bound_check", rows, summary)


def exp3_dis_vs_pca_vs_mixed():
    """Held-out L2 and H1 reconstruction error vs total TT parameter count.

    Output side: PCA-TT, DIS-TT, mixed alpha in {0.25, 0.5, 0.75}. Input side: KL (input PCA)
    vs DIS-TT. Paper's prediction: clear input-side DIS win, output-side parity.
    """
    prob = SemilinearProblem(levels=8)
    modes = prob.out_modes
    xs, xt = prob.prior.sample(np.random.default_rng(300), 160), prob.prior.sample(np.random.default_rng(301), 200)
    P, y_bar = sn.output_pca_snapshots(prob, xs)
    D = sn.output_dis_snapshots(prob, prob.prior, xs, direction_kind="exact")
    Yt = _outputs(prob, xt)
    Yc = Yt - y_bar[:, None]
    y2 = float(np.sum(Yt ** 2))
    Zt = _exact_out(prob, xt)
    sets = {"pca": P, "dis": D}
    for a in (0.25, 0.5, 0.75):
        sets[f"mixed{a}"] = sn.mixed_snapshots(P, D, a)
    rows = []
    eps_grid = (0.5, 0.3, 0.2, 0.1, 0.05, 0.03, 0.01, 0.003)
    for name, S in sets.items():
        for eps in eps_grid:
            res = dis_tt_dense(S.Z, modes, eps=eps, precompress=True)
            U = block_to_dense(res.cores)
            R = Yc - U @ (U.T @ Yc)
            rows.append({"side": "output", "basis": name, "eps": eps, "r": res.ranks[-1], "ranks": res.ranks,
                         "params": _params(res.cores), "L2_test_rel": float(np.sum(R * R) / y2),
                         "H1_test_rel": _rel_proj(U, Zt)})
    # input side: KL (input PCA) vs whitened input DIS-TT (dense input DIS as reference)
    Zin, Zin_t = _exact_in(prob, xs), _exact_in(prob, xt)
    _, phi = prob.prior.kl_modes(64)  # whitened KL basis (L symmetric): columns phi_i
    lamI, VI = np.linalg.eigh(Zin @ Zin.T)
    VI = VI[:, ::-1]
    for eps in eps_grid:
        b = input_dis_tt(sn.SnapshotSet(Z=Zin, kind="dis_in", n_samples=160, n_dirs=prob.d_y), prob.in_modes,
                         prob.prior, eps=eps, precompress=True)
        W = b.W()
        r = W.shape[1]
        rows.append({"side": "input", "basis": "dis_tt", "eps": eps, "r": r, "ranks": b.result.ranks,
                     "params": _params(b.result.cores), "H1_test_rel": _rel_proj(W, Zin_t)})
    for r in (1, 2, 4, 8, 16, 32, 64):
        rows.append({"side": "input", "basis": "kl_pca", "r": r, "params": r * prob.d_x,
                     "H1_test_rel": _rel_proj(phi[:, :r], Zin_t)})
        rows.append({"side": "input", "basis": "dis_dense", "r": r, "params": r * prob.d_x,
                     "H1_test_rel": _rel_proj(VI[:, :r], Zin_t)})
    # headline: errors at matched r for output (pca vs dis) and input (kl vs dis_dense)
    def at(side, basis, r, key):
        c = [x for x in rows if x["side"] == side and x["basis"] == basis and x["r"] == r and key in x]
        return c[0][key] if c else None
    summary = {"rows": rows,
               "input_r8": {"kl_pca": at("input", "kl_pca", 8, "H1_test_rel"), "dis_dense": at("input", "dis_dense", 8, "H1_test_rel")},
               "input_r16": {"kl_pca": at("input", "kl_pca", 16, "H1_test_rel"), "dis_dense": at("input", "dis_dense", 16, "H1_test_rel")}}
    return _write("exp3_dis_vs_pca_vs_mixed", rows, summary)


def exp4_direction_sets():
    """Random (S), KL (K) and exact columns at matched solve budget: tail error and its variance over 20 seeds.

    Budget B = 1280 linearised solves: random S = 32 and KL K = 32 with N = 40 samples; exact
    (d_X = 256 columns per sample) with N = 5. Ranks fixed to those of an exact N = 160, eps = 0.1
    reference. Metric: held-out relative H1 projection error (200 exact test samples).
    """
    prob = SemilinearProblem(levels=8)
    modes = prob.out_modes
    ref = dis_tt_dense(_exact_out(prob, prob.prior.sample(np.random.default_rng(400), 160)), modes, eps=0.1, precompress=True)
    ranks = ref.ranks
    Zt = _exact_out(prob, prob.prior.sample(np.random.default_rng(401), 200))
    U_ref = block_to_dense(ref.cores)
    err_ref = _rel_proj(U_ref, Zt)
    rows = []
    for seed in range(20):
        rng = np.random.default_rng(1000 + seed)
        for kind, N, S in (("random", 40, 32), ("kl", 40, 32), ("exact", 5, None)):
            xs = prob.prior.sample(rng, N)
            Z = sn.output_dis_snapshots(prob, prob.prior, xs, direction_kind=kind, n_dirs=S or 0,
                                        rng=np.random.default_rng(5000 + seed)).Z
            res = dis_tt_dense(Z, modes, ranks=ranks, precompress=True)
            U = block_to_dense(res.cores)
            rows.append({"seed": seed, "kind": kind, "N": N, "dirs": S or prob.d_x, "budget": N * (S or prob.d_x),
                         "H1_test_rel": _rel_proj(U, Zt), "train_cert": res.cert,
                         "sin_vs_ref": _sin(U_ref, U)})
    summary = {"ranks": ranks, "reference_H1_test_rel": err_ref}
    for kind in ("random", "kl", "exact"):
        e = np.array([r["H1_test_rel"] for r in rows if r["kind"] == kind])
        summary[kind] = {"mean": float(e.mean()), "std": float(e.std(ddof=1)), "min": float(e.min()), "max": float(e.max())}
    return _write("exp4_direction_sets", rows, summary)


def exp5_refinement():
    """Increase QTT depth d at a fixed physical problem; record max_k r_k at fixed eps.

    DIS-TT is useful only if ranks saturate in d. Also record per-bond spectra of H_Y vs C_Y
    (caveat 4: derivative spectra may decay more slowly).

    For each depth d = 6..10 an independent set of N = 40 prior samples is drawn; Z is the exact
    square-root factor of H_hat_Y = (1/N) sum J L L^T J^T (same Z Z^T as the exact-direction
    snapshots, so Algorithm A returns the same result) and the centred PCA snapshots for C_Y.
    """
    rows, spectra = [], {}
    for d in (6, 7, 8, 9, 10):
        prob = SemilinearProblem(levels=d)
        xs = prob.prior.sample(np.random.default_rng(500 + d), 40)
        Z_dis = _sqrt_factor(prob.H_Y_reference(xs))
        Z_pca, _ = sn.output_pca_snapshots(prob, xs)
        for name, Z in (("dis", Z_dis), ("pca", Z_pca.Z)):
            for eps in (0.1, 0.03, 0.01):
                res = dis_tt_dense(Z, prob.out_modes, eps=eps)
                rows.append({"d": d, "basis": name, "eps": eps, "max_rank": max(res.ranks), "ranks": res.ranks,
                             "r_last": res.ranks[-1], "params": _params(res.cores), "cert": res.cert})
            if d == 8:  # per-bond spectra of the projected operator at full ranks
                full = dis_tt_dense(Z, prob.out_modes)
                spectra[name] = [(s / s[0])[:16] if s[0] > 0 else s[:16] for s in full.spectra]
                lam = np.linalg.eigvalsh(Z @ Z.T)[::-1]
                spectra[name + "_global"] = lam[:40] / lam[0]
    for k in range(8):
        rows.append({"d": 8, "basis": "spectrum_bond", "bond": k + 1, "dis_norm_eigs": spectra["dis"][k],
                     "pca_norm_eigs": spectra["pca"][k]})
    summary = {"max_rank": {f"{b} eps={e}": [r["max_rank"] for r in rows if r.get("basis") == b and r.get("eps") == e]
                            for b in ("dis", "pca") for e in (0.1, 0.03, 0.01)},
               "depths": [6, 7, 8, 9, 10], "global_spectrum_d8": {"dis": spectra["dis_global"], "pca": spectra["pca_global"]}}
    return _write("exp5_refinement", rows, summary)


class _DenseInputBasis:
    """Duck-typed WhitenedInputBasis for a dense whitened basis W (experiment-only)."""

    def __init__(self, W, prior):
        self._W, self.prior = W, prior

    def W(self):
        return self._W

    def encode(self, x):
        return self._W.T @ self.prior.apply_inv_sqrt(np.asarray(x, dtype=float))


def exp6_end_to_end(ranks=(10, 20), n_train=640, n_test=500, n_basis=160, epochs=3000):
    """Train the RBNO (T8) with TT bases vs dense DIS bases at equal (r_X, r_Y); paper Sec. 4.4.3 protocol.

    Semilinear d = 8, paper Table 2 row (depth 6, width 2r, softplus, LR 1e-3, Adam, batch 25, 3000
    epochs, 5 halvings from epoch 2250). Bases from n_basis samples (exact directions); TT bases use
    intermediate ranks from eps = 1e-3 and final rank r. One run per configuration (the paper averages 10).
    """
    from dis_tt.rbno import assemble_rbno_data, train_latent_network

    prob = SemilinearProblem(levels=8)
    modes = prob.out_modes
    xb = prob.prior.sample(np.random.default_rng(600), n_basis)
    x_tr = prob.prior.sample(np.random.default_rng(601), n_train)
    x_te = prob.prior.sample(np.random.default_rng(602), n_test)
    Zo, Zi = _exact_out(prob, xb), _exact_in(prob, xb)
    y_bar = _outputs(prob, x_tr).mean(1)
    lo, Vo = np.linalg.eigh(Zo @ Zo.T)
    li, Vi = np.linalg.eigh(Zi @ Zi.T)
    Vo, Vi = Vo[:, ::-1], Vi[:, ::-1]
    L = prob.prior.apply_sqrt(np.eye(prob.d_x))
    test = []
    for x in x_te.T:
        st = prob.solve(x)
        test.append((prob.output(st), prob.jacobian(st) @ L))
    y2 = sum(float(np.sum(y ** 2)) for y, _ in test)
    J2 = sum(float(np.sum(J ** 2)) for _, J in test)
    out_int = dis_tt_dense(Zo, modes, eps=1e-3, precompress=True).ranks
    in_int = dis_tt_dense(Zi, prob.in_modes, eps=1e-3, precompress=True).ranks
    rows = []
    for r in ranks:
        def tt_ranks(base):
            rr = list(base[:-1])
            rr[-1] = max(rr[-1], int(np.ceil(r / 2)))
            return tuple(rr) + (r,)
        ro, ri = tt_ranks(out_int), tt_ranks(in_int)
        U_tt = block_to_dense(dis_tt_dense(Zo, modes, ranks=ro, precompress=True).cores)
        b_tt = input_dis_tt(sn.SnapshotSet(Z=Zi, kind="dis_in", n_samples=n_basis, n_dirs=prob.d_y), prob.in_modes,
                            prob.prior, ranks=ri, precompress=True)
        configs = {"dense_dis": (Vo[:, :r], _DenseInputBasis(Vi[:, :r], prob.prior)), "tt_dis": (U_tt, b_tt)}
        for name, (U, inb) in configs.items():
            W = inb.W()
            t0 = time.perf_counter()
            data = assemble_rbno_data(prob, U, inb, x_tr, y_bar)
            net = train_latent_network(data, depth=6, width_factor=2, lr=1e-3, epochs=epochs, seed=0)
            train_s = time.perf_counter() - t0
            eo = ed = 0.0
            recon_o = recon_d = 0.0
            for (y, JL), x in zip(test, x_te.T):
                xi = inb.encode(x)[None, :]
                y_hat = y_bar + U @ net.predict(xi)[0]
                J_hat = U @ net.jacobian(xi)[0] @ W.T
                eo += float(np.sum((y - y_hat) ** 2))
                ed += float(np.sum((JL - J_hat) ** 2))
                yc = y - y_bar
                recon_o += float(np.sum((yc - U @ (U.T @ yc)) ** 2))
                recon_d += float(np.sum((JL - U @ (U.T @ JL @ W) @ W.T) ** 2))
            rows.append({"r": r, "basis": name, "out_ranks": ro if name == "tt_dis" else "dense",
                         "in_ranks": ri if name == "tt_dis" else "dense",
                         "basis_params": (_params(dis_tt_dense(Zo, modes, ranks=ro, precompress=True).cores) + _params(b_tt.result.cores))
                         if name == "tt_dis" else 2 * r * prob.d_x,
                         "L2_test_rel": eo / y2, "H1_test_rel": ed / J2, "recon_L2_rel": recon_o / y2,
                         "recon_H1_rel": recon_d / J2, "final_train_loss": net.meta["final_train_loss"],
                         "seconds": train_s})
            print(rows[-1], flush=True)
    return _write("exp6_end_to_end", rows, {"rows": rows, "n_train": n_train, "n_test": n_test,
                                             "n_basis": n_basis, "epochs": epochs})


if __name__ == "__main__":
    EXPS = {k.split("_")[0]: v for k, v in globals().items() if k.startswith("exp")}
    for key in sys.argv[1:] or ["exp1"]:
        t0 = time.perf_counter()
        out = EXPS[key]()
        print(key, f"({time.perf_counter() - t0:.1f}s)", json.dumps(_jsonable({k: v for k, v in out.items() if k != "rows"}), default=str)[:3000])
