# Information access, attention competition and training disturbance (Experiment A)

**Status.** COR Phase 1.5, Step 1 (≈ 25%, unchanged).

**Labels:**
- [proof]: a mathematical result;
- [dev]: development data, D = 801, one run per cell;
- [frozen]: the frozen scorer's output;
- [post-hoc]: analysis written after the frozen output;
- [interp]: interpretation.

**Numbers and design.** All confirmation numbers are in FROZEN_PREDICTIONS_AND_RESULTS.md and `logs/score_XVIII.json`. The design is in PRE_EXPERIMENT_IDENTIFIABILITY_REVIEW.md §10.

## 1. What the XIV block does, mathematically [proof]

A −1e9 mask on edge set T removes three things at once during the window:
- **(F)** T's forward contribution;
- **(R)** the routing gradient on T's logits;
- **(V)** the value gradient through T in that head.

**Results that removed non-discriminating designs:**
- **L1.** Boosting the whole competing set in a row is identical to suppressing T, by softmax shift invariance.
- **L2.** Graded suppression moves information, share and gradient together. It cannot separate the hypotheses.
- **Pre-review A.** Under gsg, which stops the gradient on T's logits, the complement logits still carry part of the recruitment signal: 0.5–0.7 on own0, measured.

## 2. Arms and what each one sets

| arm | forward through T | content | routing gradient on T | value gradient through T | share of T |
|---|---|---|---|---|---|
| block | none | — | none | none | 0 |
| **vrg** | full mass | destroyed (another episode's values, stop-gradient) | yes | none | free |
| **gsg** | intact | real | none on T's logits | yes | free |
| **gsv** | intact | real | none | none | free |
| **sha-hi-rolled** | clamped to the paired baseline's per-head trajectory | destroyed | yes | none | = baseline's |
| **sha-lo-real** | clamped to the paired baseline's step-0 share | real | yes | yes | = initial |
| gof b = 2.3 | reduced (odds × 0.1) | real | scaled | scaled | reduced |

## 3. Results on the confirmation cohort

uP160, E and M bases, D = 811–813, measured on a fresh stream.

### 3.1 The frozen outcome

**The frozen primary analysis is INCONCLUSIVE for every arm.**
- The behavioural gate makes 4 of the 6 baselines non-unanimous, so only 2 pairs are evaluable.
- **On those 2 pairs, six of the eight hypotheses are CONTRADICTED** [frozen]: H1d, Hm_window, H2g_R, H2g_RV, H4 and H1d_plus_Hm.
- **The other two are INCONCLUSIVE:** Hm_release and H1d_plus_H2g.

### 3.2 Per pair, with the behavioural gate removed from evaluability [post-hoc; 5 evaluable pairs, the pre-registered sensitivity text applied in full]

| arm | E (3 pairs) | M (2–3 pairs) | reading |
|---|---|---|---|
| vrg | F, F, F (φ 0.99–1.04) | F, F (M\|813 missing on accuracy) | Destroying T's content with its attention mass intact selects the other organisation. |
| sha-hi-rolled | F, F, F | F, F | The same holds with T's forward share held on the baseline's own trajectory. |
| **sha-lo-real** | **N, N, N** (φ −0.03 to 0.03) | **N, N, N** | Holding T's forward share at its initial level, with real content and learning signal, selects **nothing**. |
| **gsv** | F, F, F (φ 0.81–0.85) | F, F (φ 0.98–1.02) | Removing all learning signal through T, with the forward pass intact, selects. |
| gsg | F, P, F (φ 0.65–0.89) | P, P, P (φ 0.36–0.43) | Removing only the routing gradient on T's logits gives a strong effect on E and a partial one on M. |
| gof b = 2.3 | F, F, F | F, F | A complete block is not necessary (mandate P1). |

### 3.3 Mediator: latent own-question share at release (step 100)

| arm | φ_share | outcome | consistent with "route established at release"? |
|---|---|---|---|
| vrg | 0.87–0.94 (block-like) | flips | yes |
| sha-hi-rolled | 0.86–0.90 | flips | yes |
| sha-lo-real (E) | −0.20 to −0.31: the latent share is **above** the baseline's | no flip | yes |
| sha-lo-real (M) | +0.07 to 0.08 | no flip | yes |
| gsv | 1.01–1.03 | flips | yes |
| **gsg on M** | **0.82–0.85** (block-like latent share) | **only partial** (φ_nec 0.36–0.43) | **no** |

**Clamp detail.** Under sha-lo-real the clamp holds the forward share low. The routing gradient, carrying real content, keeps raising T's latent logits, and at release the share jumps up: by 0.12–0.14 on E and 0.03 on M.

## 4. What is distinguished, and what is not

**Excluded** [post-hoc; on 2 evaluable pairs, frozen]:
- **Forward share or attention competition during formation, as a cause** (Hm_window, H2a). sha-lo-real keeps T's forward share at its initial level and selects nothing, in 6 of 6 pairs; this holds in the frozen analysis too. sha-hi-rolled keeps the baseline's share and selects, in 5 of 5 pairs.
- **"The forward content through T must be absent"** (H1d). gsv selects with the forward pass identical to plain training, in 5 of 5 pairs.
- **"The routing gradient alone"** (H2g_R) and **"any learning signal through T"** (H2g_RV). vrg and sha-hi-rolled select while the routing gradient on T is present.
- **The size of the immediate disturbance** (H3):
  - on M by XIV's data (L4);
  - on E post-hoc, because vrg selects in 3 of 3 E pairs with immediate Δloss 0.08–0.11, below the 0.18 of XIV's non-selecting noise.
  - The frozen H3 rule on E is **not** satisfied, because only one E pair is evaluable.

**What fits every arm** [interp; post-hoc; not pre-registered in this form]:

> The organisation is switched when, during formation, **route T receives no useful learning signal**. That happens when the signal is absent (block, gsv) or when it carries no task information (vrg, sha-hi-rolled: T's content belongs to another episode). It does not happen when T's attention share is merely held down while its content stays informative (sha-lo-real). In that case the route is still learned latently and established at release.

- **gsg is intermediate.** It removes the routing signal on T's logits but leaves the value signal and part of the recruitment signal through the complement logits. The arm is partial where the base relies on recruiting T (M) and near-full on E.
- **This description joins two pre-registered ideas:** H1d's "content" (through the *usefulness* of the signal) and H2g's "learning signal". It contradicts both in their exact pre-registered forms. It has to be tested prospectively before it can be claimed.

**Not distinguished:**
- **The value-gradient component.** Forward content and the value learning it enables cannot be separated: vsg was not run.
- **Whether the latent-share mediator is a cause or only a correlate.** No arm forces the latent share independently of content and learning signal. gsg on M is the one dissociation, and it goes against the mediator account.
- **Generality.** This is one parent and two bases (n_eff ≈ 2). The parent results are in INDEPENDENT_PARENT_GENERALIZATION.md.

## 5. Matching (mandate A5)

Interventions cannot be matched on every quantity at once. Each arm matches some quantities and gives up others:

| arm | what it matches | what it gives up |
|---|---|---|
| vrg | attention mass exactly | an immediate loss increase (E 0.08–0.11; M 0.01) |
| gsg / gsv | forward pass and loss identical to plain at each step, given the same parameters | trajectories diverge after the first update |
| sha-hi-rolled | forward share = the baseline's per head, by construction | the same loss increase as vrg |
| sha-lo-real | forward share = the initial level | real content kept |

The gradient energy over the window is comparable across arms in development (Reviewer B, N3). Per-group gradient norms are in each run's `window_log.json`.
