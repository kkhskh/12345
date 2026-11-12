# Selective learned repair without hand-computed m (Condition Z)

## Setting

The parent is a frozen pretrained model, either P-count or T1. Its representation feeds a fresh downstream Transformer
block that is trained on generator labels for 4000 steps, matched to Investigation VII's original transfer (C1). Each
repair changes one thing; `scripts/repair_train.py` gives the exact specifications.

| family | spec | uses m at inference? | what is learned |
|---|---|---|---|
| reference | orig | no | the downstream block only |
| reference (oracle) | injm | **yes** (hand-computed) | + a linear map of one-hot m |
| A: learned adapter | **radapt** | no | + a group-attention reader on the raw per-group ingredients (counts, q-reports, id); zero-init output added at the readout token |
| A: learned adapter | lca | no | + an MLP on the same ingredients (concatenated) |
| A control | gadapt | no | + the same MLP on matched random features (no task information) |
| B: native modification | unfreeze1, lora1, deep2 | no | train layer 1; rank-8 LoRA on layer 1; two fresh blocks |
| C: training only | auxm | no (training only) | + an auxiliary m-prediction head |
| C: training only | bayes | no | exact-Bayes soft targets |

**The radapt adapter reads ingredients that already suffice for the posterior** (counts + q-reports; §3 of the math
document). In the mandate's terms it is a *learned count+report injection*. Investigation VII's fixed count and
count+agreement injections (injn, injna) **fail**, with E −1.50 and −1.60 on xP80. A learnable non-relational version (lca)
is partial. Only the relational, attention-structured version repairs. What is learned is therefore the relation, not
access to the counts.

## Development results (xP80 / xP81 / xT70; eval_dev; 4000 steps)

| parent | spec | KL | E_R1 | dm_R1 | E_R2 | E_R3 | F34 E | F54o E | F0 / FLOC / FID \|E\| |
|---|---|---|---|---|---|---|---|---|---|
| xP80 | orig | 0.104 | **−0.895** | −0.66 | +0.93 | +0.20 | −0.12 | +0.90 | 0.17 / 0.47 / 0.15 |
| xP80 | gadapt | 0.102 | −0.889 | −0.66 | +0.86 | +0.21 | −0.14 | +0.87 | 0.17 / 0.47 / 0.14 |
| xP80 | auxm | 0.118 | −1.097 | −0.86 | +0.46 | +0.03 | +0.19 | +1.05 | 0.23 / 0.44 / 0.16 |
| xP80 | lca | 0.006 | −0.277 | −0.04 | −0.01 | −0.05 | +0.08 | +0.26 | 0.03 / 0.08 / 0.03 |
| xP80 | **radapt** | **0.002** | **−0.015** | +0.22 | −0.13 | −0.02 | +0.06 | +0.01 | 0.04 / 0.05 / 0.01 |
| xP81 | **radapt** | 0.002 | −0.286 | −0.05 | +0.04 | +0.00 | +0.07 | +0.27 | 0.02 / 0.03 / 0.02 |
| xT70 | **radapt** | 0.002 | −0.128 | +0.11 | +0.16 | +0.04 | +0.04 | +0.12 | 0.02 / 0.03 / 0.01 |
| xP80 | injm (Inv. VII) | 0.158 | +0.467 (overshoot) | +0.70 | | | | | |
| xP80 | Csoft (Inv. VII) | 0.134 | −0.369 | | | | | | |
| xP80 | W_bal (Inv. VII) | 0.415 | −0.104 | | | | | | |

`F54oE` is the regime-oriented version of F54o, the effect of removing a perfect partner. Source: `logs/run/*.out` and
`checkpoints/dev/*.json`.

## Reading (development)

1. **A learned relational adapter repairs the P-count transfer without m.** On xP80 the R1 error drops from −0.90 to
   −0.02, and R2, R3, F34 and F54o all move toward Bayes. The generic adapter with the same parameter budget does nothing
   (−0.89). Auxiliary m supervision makes things worse (−1.10).

2. **The repair is not "selective" in the narrow sense.** It does not leave everything else unchanged: it also cuts the
   control-family errors (FLOC 0.47 → 0.05) and KL (0.104 → 0.002). The original transfer is far from Bayes on many
   families. A module that computes the posterior's relational part corrects all of them, and nothing is made worse. Under
   the mandate's strict-preservation rule (no family worse in mean |E| or in distance to Bayes), the repair *passes*. A
   reader should still not picture it as a surgical edit of R1 alone.

3. **The oracle m-injection overshoots** (Investigation VII). The learned adapter does not. On average it *undershoots* on
   xP81 (−0.29). Its R1 accuracy varies between lineages more than its KL does: KL is 0.002 in all three parents, while
   E_R1 ranges from −0.02 to −0.29. This variation is why the frozen R1 bound was set at 0.25 rather than 0.15.

4. **Family B and the remaining family C runs were deferred** for compute (`logs/queue_deferred.txt`): unfreeze1, lora1,
   deep2, bayes, and auxm/gadapt/lca on xP81. They are reported in Part III / `OPEN_FOUNDATIONAL_QUESTIONS.md` if they
   were run. Without them, Investigation VIII cannot say whether the P-count *representation itself* can be made to
   support the relation by native modification.

## Confirmation (P4–P6)

See `FROZEN_PROTOCOL_AND_CONFIRMATION.md` Part II.
