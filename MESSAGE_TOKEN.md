# Neutral-prototype message token

This implementation adds a new mode without replacing the earlier exact-
residual ablations.

For a fixed attachment topology, the GCN input aggregation at victim `v` is

```text
m_v(x_t) = a_v + c_v x_t = m_v_clean + b_v + c_v x_t.
```

The neutral local prototype is computed with the same numerical GCN operator
used by the attack:

```text
p_v = -b_v / c_v.
```

Attaching copies of `p_v` therefore leaves the victim message unchanged.  A
single learned token `s` is a masked softmax over five common, low-semantic
training-feature coordinates.  The actual trigger is

```text
x_t(v) = ||p_v||_1 [(1-eta) p_v/||p_v||_1 + eta s].
```

It is nonnegative and preserves the prototype's L1 mass.  Its message residual
is not forced to have a common magnitude:

```text
r_v = eta c_v ||p_v||_1 (s - p_v/||p_v||_1).
```

Only the token logits are learned.  There is no `q` magnitude, `rho`, scale
cap, generator, QP, feasibility loss, semantic loss, CVaR loss, or residual-
consistency loss.  The bilevel shadow-training framework is retained and its
outer objective is target classification loss only.

## Run

```bash
./script/train_message_token.sh
```

Set `TOKEN_ETAS` to run a subset, for example:

```bash
TOKEN_ETAS="0.50" ./script/train_message_token.sh
```

The default sweep is `eta = 0.25, 0.50, 0.75`, with `K=5`, three trigger
nodes, ten poisoned training nodes, and the existing Cora/GCN protocol.

## Verification

```bash
python -m unittest discover -s tests -v
```

`tests/test_neutral_message_token.py` covers weighted edges, adjacent victims,
trigger sizes 1/3/5, feature-row ordering, nonnegativity, mass conservation,
neutrality, the analytic residual, and recovery of the same endpoint token.
The runtime diagnostic repeats the three central numerical checks on the real
training graph and on every evaluated victim.

## First Cora sweep

| eta | ASR | clean accuracy | mean clean-edge percentile | mean OOD percentile |
|---:|---:|---:|---:|---:|
| 0.25 | 32.40% | 84.22% | 89.39% | 0.04% |
| 0.50 | 51.51% | 84.22% | 47.05% | 29.03% |
| 0.75 | 75.50% | 84.15% | 17.52% | 80.68% |

This is the expected strength/stealth frontier: increasing `eta` improves ASR
while moving the trigger farther from its neutral local prototype.  All three
runs preserve trigger L1 mass exactly; across 1,355 evaluated victims, the
maximum neutral-message error is below `3.2e-8`, the maximum analytic-residual
error is below `6.2e-8`, and recovered-token cosine is effectively 1.
