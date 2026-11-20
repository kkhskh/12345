# Non-targeted optimization controls (mandate Workstream D)

**Status.** COR Phase 1.5, Step 1 (≈ 25%, unchanged).

**Evidence labels.** [dev] / [frozen].

## D1. Controls and predeclared matching

| control | what it disturbs | matching quantity, predeclared | gate | result of the gate on the 6 confirmation pairs |
|---|---|---|---|---|
| randnorm (state level, step 100) | B's θ plus a random direction with the per-group norm ‖θ_I − θ_B‖ (E: L0; M: L0+L2), B's moments | per-group parameter norm (exact by construction) | — | — |
| pnz (training level, window) | independent Gaussian noise on every tensor after each update; σ_k from the pair's own B/I states | total step-100 parameter displacement relative to I's | [0.67, 1.5] | ratios 1.003–1.007; all pass |
| lbl (training level, window) | 50% of the targets replaced by uniform labels (own RNG; data stream untouched) | step-100 loss excess (KL to Bayes) relative to I's; if I's excess < 0.005, the control needs ≥ 0.005 | ratio ≥ 0.5 | E: 0.98–1.02; M: 4.7–12.4 (much stronger than the block); all pass |

Which explanation each control addresses:
- **randnorm and pnz:** X8a (displacement of the same size in a random direction).
- **lbl:** X8b (loss disturbance, slowed learning, a delayed early route).

## D2. Non-targeted status, verified

**Edges.** None of the three touches any attention edge or softmax row specifically. The same-row controls msk oq0 and gof oq2 were rejected in review because they raise T's share (pre-review B3); their development results (E −0.01, M −0.17) are kept as development evidence only.

**Parameter groups** (`logs/graph_table19.csv`):
- pnz touches every tensor equally, relative to the block's displacement per tensor. It changes no gradient, update or moment.
- lbl changes every gradient.

**Task information.**
- pnz removes none.
- lbl corrupts the targets, not the inputs. Its step-100 route state partly resembles the block's. [dev E801] It leaves the early route non-functional at step 100 (nec_E 0.027 against B's 0.266), and its layer-0 displacement has cosine 0.76 with the block's.

**Strength.** These are not weak controls.
- pnz matches the block's displacement exactly.
- lbl matches the block's loss disturbance on E and exceeds it 5–12× on M.
- Isotropic noise alone cannot reproduce a loss disturbance. [dev] Even at 8× the matched scale, step-100 KL stays at B's level. This is why lbl was added.

## D3. Results [frozen]

| arm | E φ (821 / 822 / 823) | M φ | verdict |
|---|---|---|---|
| randnorm | 0.00 / 0.01 / −0.00 | −0.08 / 0.00 / −0.00 | SUPPORTED nil on both bases |
| pnz | 0.02 / 0.00 / 0.01 | −0.06 / −0.03 / 0.01 | SUPPORTED nil on both bases |
| lbl | −0.01 / 0.02 / 0.01 | −0.08 / −0.03 / 0.11 | SUPPORTED nil on both bases |

All 18 control models are competent and keep the B organisation.

## D4. What is eliminated, and what remains

**Eliminated, per base, within one parent and 3 data orders.**
- **X8a:** a random parameter displacement as large as the block's, applied at step 100 or accumulated during the window, does not select the organisation.
- **X8b:** a loss disturbance as large as the block's (E), or larger (M), during the window does not select it. That includes a disturbance that delays the early route's acquisition at step 100 (E).

**Remaining.** The general-disturbance explanation can survive only in a form these controls do not span: a disturbance *structured along the I − B direction*. That is no longer "general". [dev E801] Interpolating half of the layer-0 I − B difference does nothing, while all of it switches. So the selection is a threshold-like property of the specific learned direction.
