# DIS-TT theory reference (condensed)

Full note, with derivations, complexity tables and caveats:
https://claude.ai/artifact/Jjr4SEdweverCdqwKStCyn (section numbers below refer to it).
Source paper: Luo, O'Leary-Roseberry, Chen, Ghattas, *Dimension reduction for
derivative-informed operator learning*, arXiv:2504.08730 (2025).

**Status of claims.** The paper's results (DIS operators, Prop. 22 optimality, Lemma 39
sampling bounds, Thm. 10) are established. Every TT-specific statement here
(Props. 1-2, the corollary, the input-side extension) was derived in the note, not taken from the
paper, and is exactly what the test suite and experiments are meant to check.

## Objects (note Sec. 2)

| Symbol | Definition | Code |
| --- | --- | --- |
| D_E F | DF(x) L, whitened Jacobian, C_X = L L^T | `ForwardModel.jvp` + `GaussianPrior.apply_sqrt` |
| H_Y | E[DF C_X DF^T] (output sensitivity) | `snapshots.output_dis_snapshots` |
| H_X^w | E[L^T DF^T DF L] (whitened input sensitivity) | `snapshots.input_dis_snapshots` |
| C_Y | Cov(F) (output PCA) | `snapshots.output_pca_snapshots` |
| Z | pre-scaled snapshots, Z Z^T = estimate of H | `SnapshotSet.Z` |

Paper Prop. 22: E||(I-P)D_EF||_HS^2 = Tr((I-P) H_Y (I-P)), minimised by the top-r eigenvectors
of H_Y with value sum_{i>r} lambda_i. Everything below applies this bond by bond.

## TT projectors (note Sec. 3)

Left-orthogonal cores G_k (shape (r_{k-1}, n_k, r_k)); frames U_{<=k} = (U_{<=k-1} (x) I_{n_k}) G_k;
projectors Pi_k = U_{<=k} U_{<=k}^T (x) I_{>k}, which are nested. With D the last bond:

    ||(I - Pi_D) A||^2 = sum_{k=1}^D ||(Pi_{k-1} - Pi_k) A||^2                      (T1)
    E||(Pi_{k-1} - Pi_k) D_EF||^2 = Tr[(I - G_k G_k^T) H~_k]                         (T2)
    H~_k = W_k^T Tr_{>k}(H_Y) W_k,   W_k = U_{<=k-1} (x) I_{n_k}                     (T3)

In snapshot form, Tr_{>k}(Z Z^T) is the Gram of the k-th left unfolding of the tensor
(n_1, ..., n_d, T), so **DIS-TT = uncentred TT-SVD of the snapshot tensor with the snapshot
index last**.

D = d gives an r-column block-TT basis (RBNO decoder). D = d-1 gives a TT subspace
U_{<=d-1} (x) R^{n_d}.

## Error statements (note Sec. 4)

    Prop. 1:  max_k tail_{r_k}(H^(k)) <= min_TT err <= err_greedy = sum_k tail_{r_k}(H~_k)
                                                     <= sum_k tail_{r_k}(H^(k)),     H^(k) = Tr_{>k} H
    Exact on training data:  ||(I - Pi_D) Z||_F^2 == sum_k tails                    (tested to 1e-8)
    Corollary (Gaussian Poincare): E||(I-Pi)(F - Fbar)||^2 <= E||(I-Pi) D_EF||^2
    Prop. 2:  E_N err <= sum_k [ tail_{r_k}(H^(k)) + sqrt(2 r_k M_k / N) ],
              M_k = E||Tr_{>k}(D_EF D_EF^*) - H^(k)||_HS^2   (only the global branch is rigorous)
    Rank rule: r_k minimal with tail_k <= eps^2 Tr(H_hat) / D   =>  cert <= eps^2

Input side: identical with H_X^w; V = L W is E-orthonormal; encoder W^T L^{-1} x. The H^1
ridge bound needs the paper's Assumption 9, which holds per direction and so sums over any
E-orthonormal complement. Re-verify the paper's remaining proof steps before claiming it.

## Known failure modes (note Sec. 8)

1. Prop. 5 (linear constraints preserved by output DIS) does **not** survive TT truncation.
2. Greedy and order-dependent: the factor D is real. ALS sweeps (`refine.py`) can only help.
3. The whitening factor L must be cheap in TT form. A wide L needs the pseudo-inverse and a
   ker(L) check (`input_side.py` docstring).
4. Per-bond derivative spectra may decay more slowly than PCA spectra. Untested; measure it.
5. M_k is unknown, so use bootstrap spreads (`bounds.bootstrap_tails`).
6. Non-identity mass matrices break TT structure unless (nearly) Kronecker.
