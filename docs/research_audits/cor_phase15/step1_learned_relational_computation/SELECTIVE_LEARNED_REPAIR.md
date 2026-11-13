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

## Confirmation (P4–P6; frozen; eval_conf8)

Full table: `FROZEN_PROTOCOL_AND_CONFIRMATION.md` Part II.

- **P4: confirmed.** The original transfer fails on 8/8 new P-count lineages (mean E_R1 −0.74).
- **P5: confirmed.**
  - radapt beats orig on |E_R1| in 8/8 lineages (sign p = 0.004).
  - Mean |E_R1| is 0.057, and every preservation metric improves.
  - The generic adapter does nothing (mean |E_R1| 0.49, vs orig 0.50 on the same four lineages).
  - The oracle m-injection is erratic: +0.36 (overshoot) and −0.28.
- **P6: confirmed by its criterion. On R1 itself, however, radapt is not better than T1's mild original error** (1 paired
  win out of 4; zT1110 −0.34 vs −0.12).

## How the learned repair works: it bypasses the parent (exploratory, post-freeze)

Two ablations were applied to the trained radapt models.

| ablation | what is removed | P-count conf lineages (KL / acc) | dev xP80 / xP81 |
|---|---|---|---|
| none | — | 0.002 / 0.74 | 0.002 / 0.746 |
| adapter output zeroed (`repair_ablate.py`) | the learned relational module | **0.553–0.574 / 0.33–0.35** (all 8) | 0.561 / 0.335; 0.562 / 0.335 |
| parent state replaced by its batch mean (`repair_zablate.py`) | all episode-specific information from the frozen parent | **0.002–0.003 / 0.74** (zP100–103: E_R1 moves from −0.04, −0.13, −0.03, −0.08 to +0.08, +0.00, +0.13, −0.03) | see json |

The repaired model is at chance without the adapter, and **unchanged in quality without the parent's representation**.
The only trace of the parent is a small R1 shift: removing Z moves E_R1 up by about +0.10 on average. The parent
therefore still contributes a slight bias in the P-count (discount) direction, which the downstream block passes
through. The
downstream block reads the decision entirely from the adapter, which computes it from the raw ingredients. The
"repair" therefore leaves the P-count representation unused: it is a learned relational reader placed next to it.

**Consequences for the mandate's question.**

1. *Can a learned module, with no hand-computed m, supply the relational computation the original transfer lacks?*
   Yes, robustly (P5). It does so by computing the decision itself. "No m needed, and m injection is worse than learning"
   still stands.
2. *Can the frozen P-count representation be repaired, that is, made to support the relation?* **Not shown.** Every
   successful repair here ignores it. This is decided by the native-modification family (unfreeze1, lora1, deep2). Those
   were run on xP80 only, after the freeze, as exploratory development; see below.
3. The "selective" label should not be applied to radapt. Preservation holds because the bypass is close to Bayes
   everywhere, not because it edits R1 alone.

## Native modification (exploratory, xP80, post-freeze)

| spec | E_R1 | dm_R1 | E_R2 | E_R3 | F34 E | KL | FLOC \|E\| |
|---|---|---|---|---|---|---|---|
| orig | −0.895 | −0.66 | +0.93 | +0.20 | −0.12 | 0.104 | 0.47 |
| unfreeze1 (train layer 1 + fresh layer 2) | **−0.312** | −0.08 | +0.53 | +0.17 | −0.03 | 0.056 | 0.14 |
| lora1 (rank-8 on layer 1) | −0.439 | −0.21 | +0.57 | +0.11 | −0.07 | 0.066 | 0.20 |
| deep2 (two fresh blocks) | −0.546 | −0.31 | +0.63 | +0.11 | +0.01 | 0.061 | 0.18 |

Native modification of the pretrained network moves R1 about two thirds of the way from orig toward Bayes (unfreeze1),
or about half (LoRA). A second fresh block helps less (−0.55). None of them reaches Bayes within 4000 steps. R2 is still +0.5 off. Unlike radapt, these models *do*
use the parent's computation, because they modify it. They are the only evidence here that the P-count network can be
pushed toward the relation, and the evidence is one development parent, one seed, after the freeze.
