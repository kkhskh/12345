# Title candidates and selected title

All five avoid COR, Transformer-wide claims, and “complete recovery.”

1. **Causal Information Outside a Frozen Board Probe in Othello-GPT** (selected)
2. Probe-Preserving Residual Interventions and Donor-Free Transfer in Othello-GPT
3. Target-Conditioned Nullspace Steering in Othello-GPT and Its Limits
4. A Frozen Linear Board Probe Does Not Capture All Causal Move Information in Othello-GPT
5. Probe-Invisible Target Effects in Othello-GPT: Matched Interventions and Incomplete Transfer

Selected title (1): it names the object (frozen board probe), the finding (causal information outside it), and the setting (Othello-GPT). It does not promise a method paper or a recovered world state.

Candidate 4 is more explicit but long. Candidate 2 names two methods rather than the scientific question.

## Abstract (master)

Linear probes can recover Othello-GPT’s board with high accuracy, and prior work has shown that changing those probe-readable features changes the model’s legal-move predictions. That does not settle whether a frozen probe captures every residual-stream distinction the network uses. We study a single 8-layer Othello-GPT, a frozen 64-square linear board probe, and residual edits at `blocks.6.hook_resid_mid` that are constrained to the probe’s contrast nullspace so that the probe’s full occupancy-probability vector is preserved. Under that constraint, correct-target donor directions improved a legal-mass response score relative to wrong-target and random nullspace controls, both at matched residual L2 and, in a restored protocol, at matched output Jensen–Shannon divergence of 0.01. A 64-dimensional ridge map trained only on training-split activations produced a similar target-aligned advantage without using the target donor activation at evaluation; mean absolute scores remained negative (recorded \(Y_{\mathrm{learned}}\approx-0.58\)). About 56% of that map’s mean improvement over a probe-space editor was already obtained by a stage-only map. On a fresh 32-triplet cohort, mean target advantages were positive, but joint switching of both arms occurred in 13 of 32 triplets, below a 60% gate. A factorial test of visible/nullspace interaction yielded estimates near zero. The results support a limited claim: this frozen probe does not exhaust the residual information that affects legal-move outputs, and some of the remainder is predictable from symbolic state, but the recovered map is not a complete transferable board.
