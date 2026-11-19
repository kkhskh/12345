# How each intervention changes the forward pass, gradients, updates and optimizer state

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

## 3. What the audit does not show

- It shows **which quantities an intervention touches**. It does not show **which of them matter** for the organisation at step 1211. Answering that is the job of the transplant and vsg experiments.
- Gradient changes at fixed parameters at steps 0 and 50 do not bound how far the trajectories later diverge.
