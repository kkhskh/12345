# How each intervention changes the forward pass, gradients, updates and optimizer state

**Version.** This is v3 (after the pre-experiment reviews).
- v1, the version reviewed in round 1, is preserved as `logs/graph_audit19_v1.json`.
- v3 adds modes vsgo, frz, frzp, pnz and lbl (lbl audited with its noised labels), plus the development same-row control.
- v3 measures the update and the AdamW μ/ν change by running each mode's *actual* training step, which includes freezes. This addresses pre-review B5 and B R2-2.

**Status.** COR Phase 1.5, Step 1 (≈ 25%, unchanged). This is mandate stage 1. It was completed before any XIX transplant or confirmation run was scored.

**Artefacts.**
- `scripts/graph_audit19.py` → `logs/graph_audit19.json`: full per-tensor numbers.
- `scripts/graph_table19.py` → `logs/graph_table19.csv` / `.json`: the machine-readable table the mandate asks for.

**Setup.**
- Both bases (E: target route own0 at layer 0; M: target route own2 at layer 2).
- Two training states: step 0, and step 50 with the plain-training Adam state from the B development run, seed 801.
- One batch of 256 episodes (stream 97 900 000).
- Every intervention is compared with plain training at the same parameters.
- A gradient "changes" if its relative difference is > 1e-4. The forward pass "changes" if the maximum logit difference is > 1e-5.

## 1. Summary table (step 0; step 50 is qualitatively identical)

| mode | what it does | forward | d loss/d bias_T | d loss/d bias_C | routing WQ/WK at target layer | value WV at target layer | upstream layers | embeddings | downstream layers + readout |
|---|---|---|---|---|---|---|---|---|---|
| plain | — | — | E −0.0079, M −0.0061 | E +0.0079, M +0.0061 | — | — | — | — | — |
| block (gof B = 1e9) | T attention ≈ 0 | **changed** | 0 | 0 | changed | changed | changed | changed | **changed** |
| gof 2.3 | odds × 0.1 | **changed** | −0.0011 | +0.0011 | changed | changed | changed | changed | **changed** |
| gsg | stop-gradient on T logits | identical (≤ 1e-6) | **0** | **= plain** | changed | unchanged | changed (M) | changed | **unchanged** |
| gsv | gsg + stop-gradient on T value term | identical | 0 | E −0.0008, M +0.0090 | changed | changed | changed (M) | changed | unchanged |
| vsg (new) | stop-gradient on values read through T; routing intact | identical | **= plain** | = plain | **unchanged** | changed | changed (M) | changed | unchanged |
| vrg | values through T rolled across episodes + stop-gradient | **changed** | ≈ plain | ≈ plain | changed | changed | changed | changed | changed |
| vrs | values rolled, gradient kept | changed | ≈ plain | ≈ plain | changed | changed | changed | changed | changed |
| fst | straight-through block | changed | ≈ plain | ≈ plain | changed | changed | changed | changed | changed |

E has no upstream layers, because its target is layer 0. On M, "upstream" means all 24 tensors of layers 0–1.

## 2. Consequences for interpretation

**C1. gsg removes only the T-side half of the routing gradient.**
- A softmax is invariant to adding a constant to every logit. So under plain training, d loss/d bias_C = −d loss/d bias_T *exactly* (verified to 1e-9).
- gsg zeroes the T-side term and leaves the complement-side term untouched (+0.0079 on E, the same value as plain).
- The share of T in the attention softmax therefore still moves in the direction plain training would push it, through the complement logits.
- **Implication.** gsg is not a "no routing gradient" intervention. XVIII's reading of gsg as removing route recruitment is withdrawn at the mechanistic level (see `XVIII_FAILURES_AND_UNRESOLVED_CAUSES.md` §3). A complete removal of the routing signal needs both sides stopped (gsg + a complement stop). That combination is not used in XIX: it also removes all routing learning for the complement, so it is non-targeted by construction.

**C2. vsg is the clean value-only arm.**
- Forward pass identical.
- Routing gradients identical on both sides.
- The changed gradients are exactly the value pathway through T: target-layer WV and LN1, the embeddings and (on M) the upstream layers that feed those values.
- This is the arm XVIII lacked (confound 1).

**C3. vrg confounds three things.** Its forward pass differs (rolled content). Its routing gradient differs because the loss surface differs. Its value gradient is stopped. A vrg effect cannot be attributed to any one of these.

**C4. Every stop-gradient mode leaves downstream gradients unchanged at fixed parameters.** The effect of a stop-gradient mode on downstream layers arises only through the changed trajectory of upstream parameters. This is the sense in which these modes are "natural" (unchanged forward) but not "local" (they change the trajectory).

**C5. No mode is parameter-local on M.** The value and routing parameters of layer 2 are fed by layers 0–1, so every M intervention changes the gradients of all 24 upstream tensors. Parameter-level locality is possible only by freezing (workstream C), not by stopping gradients.

**C6. Optimizer state diverges for every mode.** Relative differences of the one-step update at step 50:
- 0.08–0.12 for the natural modes (gsg / gsv / vsg);
- 0.47–0.54 for block, gof and fst on E.

So by step 100 the Adam moments differ as well. This is why workstream A transplants optimizer state separately from parameters.

## 2b. Full table, v3 (step 50, from B's Adam state; step-0 rows in `logs/graph_table19.csv`)

Values are relative differences from plain training on the same batch. "grad changed" counts tensors whose gradient differs by more than 1e-4 (relative).

| base | mode | forward changed | dL/db_T | dL/db_C | grad changed: routing / value / target-other / upstream / shared / downstream | one-step update rel. diff: routing / value / target-other / upstream / downstream | μ rel. diff routing |
|---|---|---|---|---|---|---|---|
| E | block | yes | +0.0000 | -0.0000 | 2/2 / 1/1 / 9/9 / 0/0 / 5/5 / 27/27 | 0.12 / 0.14 / 0.24 / 0.00 / 0.54 | 0.12 |
| E | gof2.3 | yes | -0.0010 | +0.0010 | 2/2 / 1/1 / 9/9 / 0/0 / 5/5 / 27/27 | 0.10 / 0.12 / 0.21 / 0.00 / 0.47 | 0.10 |
| E | gsg | no | +0.0000 | +0.0042 | 2/2 / 0/1 / 2/9 / 0/0 / 4/5 / 0/27 | 0.09 / 0.00 / 0.06 / 0.00 / 0.00 | 0.08 |
| E | gsv | no | +0.0000 | -0.0000 | 2/2 / 1/1 / 2/9 / 0/0 / 5/5 / 0/27 | 0.09 / 0.10 / 0.08 / 0.00 / 0.00 | 0.09 |
| E | vsg | no | -0.0042 | +0.0042 | 0/2 / 1/1 / 2/9 / 0/0 / 3/5 / 0/27 | 0.00 / 0.10 / 0.09 / 0.00 / 0.00 | 0.00 |
| E | vsgo | no | -0.0042 | +0.0042 | 0/2 / 1/1 / 3/9 / 0/0 / 3/5 / 0/27 | 0.00 / 0.10 / 0.09 / 0.00 / 0.00 | 0.00 |
| E | vrg | yes | +0.0021 | -0.0021 | 2/2 / 1/1 / 9/9 / 0/0 / 5/5 / 27/27 | 0.06 / 0.11 / 0.10 / 0.00 / 0.06 | 0.06 |
| E | vrs | yes | +0.0021 | -0.0021 | 2/2 / 1/1 / 9/9 / 0/0 / 5/5 / 27/27 | 0.06 / 0.11 / 0.09 / 0.00 / 0.06 | 0.06 |
| E | fst | yes | -0.0068 | +0.0068 | 2/2 / 1/1 / 9/9 / 0/0 / 5/5 / 27/27 | 0.10 / 0.14 / 0.24 / 0.00 / 0.54 | 0.10 |
| E | frz | no | -0.0042 | +0.0042 | 0/2 / 0/1 / 0/9 / 0/0 / 0/5 / 0/27 | 1.00 / 0.00 / 0.00 / 0.00 / 0.00 | 0.21 |
| E | frzp | no | -0.0042 | +0.0042 | 0/2 / 0/1 / 0/9 / 0/0 / 0/5 / 0/27 | 1.00 / 0.00 / 0.00 / 0.00 / 0.00 | 0.00 |
| E | pnz | no | -0.0042 | +0.0042 | 0/2 / 0/1 / 0/9 / 0/0 / 0/5 / 0/27 | 0.00 / 0.00 / 0.00 / 0.00 / 0.00 | 0.00 |
| E | lbl | no | -0.0015 | +0.0015 | 2/2 / 1/1 / 9/9 / 0/0 / 5/5 / 27/27 | 0.18 / 0.14 / 0.44 / 0.00 / 0.94 | 0.15 |
| E | ctl_same_row | yes | -0.0157 | +0.0157 | 2/2 / 1/1 / 9/9 / 0/0 / 5/5 / 27/27 | 0.42 / 0.42 / 0.97 / 0.00 / 2.18 | 0.42 |
| M | block | yes | +0.0000 | -0.0000 | 2/2 / 1/1 / 9/9 / 24/24 / 5/5 / 3/3 | 0.11 / 0.11 / 0.13 / 0.12 / 0.16 | 0.08 |
| M | gof2.3 | yes | -0.0007 | +0.0007 | 2/2 / 1/1 / 9/9 / 24/24 / 5/5 / 3/3 | 0.10 / 0.10 / 0.11 / 0.10 / 0.14 | 0.07 |
| M | gsg | no | +0.0000 | +0.0034 | 2/2 / 0/1 / 2/9 / 24/24 / 5/5 / 0/3 | 0.12 / 0.00 / 0.10 / 0.05 / 0.00 | 0.11 |
| M | gsv | no | +0.0000 | -0.0003 | 2/2 / 1/1 / 2/9 / 24/24 / 5/5 / 0/3 | 0.12 / 0.10 / 0.14 / 0.09 / 0.00 | 0.12 |
| M | vsg | no | -0.0034 | +0.0034 | 0/2 / 1/1 / 2/9 / 24/24 / 5/5 / 0/3 | 0.00 / 0.10 / 0.12 / 0.07 / 0.00 | 0.00 |
| M | vsgo | no | -0.0034 | +0.0034 | 0/2 / 1/1 / 3/9 / 24/24 / 5/5 / 0/3 | 0.00 / 0.10 / 0.12 / 0.07 / 0.00 | 0.00 |
| M | vrg | yes | -0.0010 | +0.0010 | 2/2 / 1/1 / 9/9 / 24/24 / 5/5 / 3/3 | 0.05 / 0.10 / 0.13 / 0.08 / 0.04 | 0.04 |
| M | vrs | yes | -0.0010 | +0.0010 | 2/2 / 1/1 / 9/9 / 24/24 / 5/5 / 3/3 | 0.05 / 0.07 / 0.08 / 0.06 / 0.04 | 0.04 |
| M | fst | yes | -0.0051 | +0.0051 | 2/2 / 1/1 / 9/9 / 24/24 / 5/5 / 3/3 | 0.09 / 0.11 / 0.13 / 0.12 / 0.16 | 0.07 |
| M | frz | no | -0.0034 | +0.0034 | 0/2 / 0/1 / 0/9 / 0/24 / 0/5 / 0/3 | 1.00 / 0.00 / 0.00 / 0.00 / 0.00 | 0.50 |
| M | frzp | no | -0.0034 | +0.0034 | 0/2 / 0/1 / 0/9 / 0/24 / 0/5 / 0/3 | 1.00 / 0.00 / 0.00 / 0.00 / 0.00 | 0.00 |
| M | pnz | no | -0.0034 | +0.0034 | 0/2 / 0/1 / 0/9 / 0/24 / 0/5 / 0/3 | 0.00 / 0.00 / 0.00 / 0.00 / 0.00 | 0.00 |
| M | lbl | no | -0.0018 | +0.0018 | 2/2 / 1/1 / 9/9 / 24/24 / 5/5 / 3/3 | 0.22 / 0.19 / 0.35 / 0.32 / 0.39 | 0.20 |
| M | ctl_same_row | yes | -0.0054 | +0.0054 | 2/2 / 1/1 / 9/9 / 24/24 / 5/5 / 3/3 | 0.31 / 0.27 / 0.51 / 0.45 / 0.56 | 0.32 |

**Additional consequences (v3).**

- **C7. vsgo** keeps the forward pass and the routing gradient exactly as in plain training. It additionally changes WO (3 of 9 target-other tensors against 2 of 9 for vsg). The downstream readers of T's content still learn from it, and they cannot be stopped without changing the forward pass. This is a stated limitation.
- **C8. frz and frzp** change only the update of the frozen routing tensors (rel. diff 1.00).
  - frz also holds their μ (rel. diff 0.21 / 0.50 at step 50).
  - AdamW's step counter is shared, so frz's bias correction under-scales the held moments. At release this gives oversized steps: reviewer B measured 1.7–2.4× the baseline step length over steps 111–200.
  - frzp lets the moments update normally.
- **C9. pnz** changes no gradient, update or moment. It perturbs only the parameters, after each update. The per-step noise norm is 1.1–14.5× (E) and 0.3–6.4× (M) the plain update norm. The noise is matched in cumulative displacement, not in per-step size.
- **C10. lbl** leaves the forward pass and all edges untouched, but **every gradient changes**, because the targets change.
- **C11. Same-row controls** (msk oq0 on E, gof oq2 on M) change every gradient. They raise T's softmax share (pre-review B3), so they are **not** non-targeted. They are excluded from the confirmation.

## 3. What the audit does not show

- It shows **which quantities an intervention touches**. It does not show **which of them matter** for the organisation at step 1211. Answering that is the job of the transplant and vsg experiments.
- Gradient changes at fixed parameters at steps 0 and 50 do not bound how far the trajectories later diverge.
