# Neutral-prototype mass completion

This realization keeps the neutral local prototype intact and uses only its
unused clean-feature mass for the shared token.

For victim `v`, define

```text
p_v = -b_v / c_v
L_v = ||p_v||_1
M_v = ||x_v||_1
```

and construct every one of its identical trigger nodes as

```text
x_t(v) = p_v + eta (M_v - L_v) s,   0 <= eta <= 1.
```

The implementation uses each victim's measured `M_v`; it does not hard-code
one.  It fails explicitly if `L_v > M_v`, because nonnegative mass completion
is then infeasible without modifying the neutral carrier.  On row-normalized
Cora, `M_v = 1`.

Since `b_v + c_v p_v = 0`, the actual first-layer residual is

```text
r_v = eta c_v (M_v - L_v) s.
```

Thus the residual is directly collinear with the learned token.  `eta` only
controls how much unused mass is filled; it never deletes `p_v`.

## Run

```bash
./script/train_message_token_mass_completion.sh
```

The default Cora/GCN sweep uses `eta = 0.25, 0.50, 0.75, 1.00`.  A subset can
be selected with `TOKEN_ETAS`, for example:

```bash
TOKEN_ETAS="0.50 1.00" ./script/train_message_token_mass_completion.sh
```

## First complete sweep

| eta | ASR | clean accuracy | mean alpha | mean edge percentile | mean OOD percentile |
|---:|---:|---:|---:|---:|---:|
| 0.25 | 45.98% | 84.00% | 0.1316 | 73.67% | 28.99% |
| 0.50 | 80.44% | 84.22% | 0.2632 | 48.15% | 95.94% |
| 0.75 | 92.55% | 84.07% | 0.3947 | 35.33% | 99.63% |
| 1.00 | 95.72% | 84.07% | 0.5263 | 27.98% | 99.97% |

Mass completion recovers almost all of the previous exact-payload ASR
(96.31%) without exact residual inversion.  It also confirms that the low ASR
of prototype-mass interpolation was primarily an amplitude-budget problem.
The cost is visible in input-space OOD: high-eta, nearly one-hot tokens are
easy to distinguish by the current PCA diagnostic even though their feature
mass is normal and their edge percentile remains nonzero.

At `eta=1`, successful victims have mean `alpha=0.5312`; the remaining 58
failed victims have mean `alpha=0.4179`, mean degree 11.03, and mean `c_v=0.6365`.
Mass completion removes the old structural cap but does not eliminate
high-degree attenuation for the hardest residual failures.

## Verification

Nine unit tests cover old and new realizations.  The new tests include weighted
edges, adjacent victims, trigger sizes 1/3/5, all eta endpoints, exact carrier
preservation, row-mass identities, direct token collinearity, and analytic
residual recovery.  Across the complete real-graph sweep, maximum errors are:

```text
neutral message:       2.94e-8
carrier preservation:  2.97e-8
residual formula:       1.20e-7
token recovery:         4.70e-7
```
