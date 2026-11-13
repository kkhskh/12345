# Training and optimization (Problem C)

Development evidence (Conditions O/R learning curves: `LEARNABLE_COMPOSITION_EXPERIMENTS.md`; Condition Z variants:
`SELECTIVE_LEARNED_REPAIR.md`). Confirmation of the budget and phase claims: P2.

## C1. Budget

| reader (R, labels) | 1k | 4k | 16k | 48k |
|---|---|---|---|---|
| mlp | −1.65 | −1.44 | −0.46 (16k schedule) | +0.08 |
| attn | −1.65 | −0.26 | −0.08 | −0.04 |
| relnet | −1.54 | −0.04 | +0.07 | −0.01 |

The values are E_R1. In this range, the budget decides whether the concatenation MLP succeeds. For the relational readers
it only refines the result.

## C2. Objective: labels vs exact-Bayes soft targets

Soft targets speed up every reader: mlp −0.46 → +0.01 at 16k; attn −0.08 → −0.06; relnet +0.07 → +0.00. The function class
and the data distribution do not change, and labels are samples from the same posterior. This is a variance-reduction
effect on the gradient. Soft targets carry no extra relational information.

In Condition Z, Investigation VII already showed that soft targets improve, but do not repair, the original P-count
transfer (Csoft E −0.37 on xP80, −0.07 on xP81). The `bayes` repair spec was not rerun here.

## C3. Auxiliary relational supervision

`auxm`: a linear head on the final readout state predicts m_g, during training only, with weight 0.5. On xP80 at 4k steps
it gives **E_R1 −1.10, KL 0.118**, which is *worse* than the generic adapter (−0.89) and than Investigation VII's
original transfer. Asking the one-block downstream to also *report* m does not make it *use* m. A plausible reading is
that this is the decodable-but-unused pattern again: the head can be satisfied from features that the main readout
combines additively. That reading is not tested.

## C4. Order of learning (the separable phase)

Every learner without m first learns a response with the P-count sign. At 1k steps E_R1 ranges from −1.1 to −1.8, even
with Bayes targets (mlp −1.37, relnet −1.11, attn −0.53). Learners given m (O) start with a *positive* error instead.
Interpretation [N, development]:
- the count-based, group-separable part of the posterior is learned first, because it explains most of the loss
  (91.6 % of decisions have a single perfect group);
- the cross-group gate is learned later.

The P-count models of Investigations VI–VII look like models that stopped, or were frozen, in that first phase. The
look is the same; nothing shows the cause is the same.

## C5. Data frequency

Two perfect groups agreeing at q occur in 2.8 % of natural decisions. Development did not reweight the training
distribution. Investigation VII's `W_bal` (reweighting, Condition Z) moved E_R1 to −0.10 on xP80 but cost accuracy
(0.54). The cost of frequency is therefore visible as the delay in C4, not as a failure to learn eventually.

## X. Condition X: ordinary tokens, from scratch

`scripts/x_train.py` trains a Transformer from scratch on the full episode token sequence, with all parameters trainable
and generator labels.

| model | steps | KL | acc | dm on F45 R1 / R2 / R3 | F34 dm |
|---|---|---|---|---|---|
| 1 block | 6000 | 0.304 | 0.552 | 0.00 / 0.00 / 0.00 | 0.00 |
| 3 blocks | 6000 | 0.279 | 0.577 | 0.00 / 0.00 / 0.00 | 0.00 |

Both models respond to *no* edit: every family's E equals −dm_B. They have learned a coarse, edit-insensitive prior
(accuracy above the 1/3 baseline from marginal structure) and none of the count-dependent evidence, separable or
relational. **This is a budget-limited negative.**

- The pretrained P-count and T1 models of Investigations VI–VII needed curricula or pretraining tasks to reach count
  sensitivity at all.
- A 16k-step 3-block run was started and stopped after 28 min for compute (`logs/COMPUTE_LOG.txt`).
- Condition X therefore does not answer whether a raw-token Transformer would learn the relation at larger budgets. It
  shows that the relational problem sits on top of a harder token-binding problem, which the ingredient conditions
  (O/R) remove by construction.
