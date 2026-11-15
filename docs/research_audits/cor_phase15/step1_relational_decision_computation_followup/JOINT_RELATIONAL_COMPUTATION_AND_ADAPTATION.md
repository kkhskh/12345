# Joint relational computation and two-colluder adaptation (Workstream C)

Investigation XIII-B, Phase 1.5 Step 1. Labels: **[proof]**, **[num]** (numerical verification), **[dev]** (XIII seed-20 models),
**[conf]** (frozen seed-21 fine-tunes, see FROZEN_CONFIRMATION_AND_STATISTICS.md), **[expl]**.


> **Corrections after independent review 3** (`reviews/REVIEW_3_MECH.md`; see `reviews/ADJUDICATION.md`). Where the text below conflicts
> with these, the corrections govern.
> 1. **"E-based" means agreement-at-q-gated.** The D1/D2 edits separate E from A1 but not from the threshold-4 rule A2: distance
>    differences are ≤ 0.011.
> 2. **The ρ result is a robust class marker of the input-output law, but "a different combination operator" is one of several equivalent
>    readings.**
>    - The power mean is algebraically identical to tempered per-group evidence followed by output re-sharpening.
>    - A lapse floor moves ρ from 0.57 to 0.67 in uP160.
>    - ρ explains only about 15% of the early class's misfit relative to the mixture.
>    - Pure temperature and output saturation are ruled out.
>    - Bootstrap intervals for ρ exclude both 0 and 1 in the early class.
> 3. **"Combined mostly by the layer-2 MLP" is not identified.** Freezing components shows only that they lie on the serial path. In early
>    units, 40–54% of the relational effect at the layer-2 input sits on *other-question* source tokens and is read by layer-2 attention.
>    That route is in addition to the readout-token route.
> 4. **Class inheritance is partial.** Under one new seed, 2 of 5 parents moved by more than 0.3 in lik_RO1: uP163 early → hybrid, and uP164
>    hybrid → early. "uP164 is a hybrid" holds only for seed 20.
> 5. **Routing proportions vary with the fine-tuning seed within the mixture class** (uP161: route_R 0.28 → 0.61). The "categorical routing
>    law" is withdrawn. Q3b and Q4c are one event: route-Q necessity in uP161 seed 21. The Q3b miss (0.38 vs 0.4) is within stream-to-stream
>    noise (0.42 and 0.45 on two other streams).
> 6. **Adaptation caveats.**
>    - Capacity is not matched: early01 trains two blocks, the layer-2 arm trains one.
>    - "The frozen layer 2 reuses its mixture operation" is **untested** and is withdrawn as a claim.
>    - "J enters at resid1 from other-question tokens" is true by construction of the J-pairs.
>    - The decode_J ridge probe itself performs the cross-question sum, so it shows linearly accessible per-question cross-group equality,
>      not that the network computes n4.
>    - How much a layer-2-only repair uses J rises **graded** with lik_RO1 (review 3).

## C1. Which cross-group relation does the two-colluder decision need?

Generator: G = 3 groups of 2 sources, C = 2 colluders sharing one copy event and one copied value per question (probability β), Q = 6,
K = 3, α = 0.8. The exact posterior is `xii_bayes.posterior_shared`; XIII verified it exhaustively.

**Three observation granularities** (`scripts/censored_bayes.py`, exact Bayes for each censored observation, 200,000 episodes per β):

| β | FULL (all reports) | LOCALQ: joint values at q + self-agreement bits elsewhere | PERGROUP: independent-copying product rule (see note) |
|---|---|---|---|
| 1.0 | acc 0.801 | acc 0.801; **KL to FULL 0.000; 0.00% decisions differ** | acc 0.749; KL 0.054; 5.5% differ |
| 0.6 | acc 0.794 | acc 0.758; KL 0.106; **6.1% differ** | acc 0.500; KL 0.462; 37% differ |

**Note on PERGROUP (math review 1).** PERGROUP is numerically identical (to 4e-15) to the independent-copying product rule
(`genlib.posterior_brute`). It is **not** the exact posterior of any censored observation under the true generator, and **not** "the best
rule that never compares groups". A rule that keeps the shared copy event through *joint* self-agreement bits, while still comparing no
values across groups, reaches 0.671 at β = 0.6 (0.747 at β = 1). Most of PERGROUP's deficit at β = 0.6 therefore comes from ignoring the
shared copy event, not from the absence of value comparisons.

**Proposition (β = 1; scope: G = 3, C = 2, on-support inputs) [proof, corrected after math review 1].**
p(y | full reports) = p(y | LOCALQ observation).

*Proof.* With G = 3 and C = 2 the role is fixed by the reliable group g.

**FULL.** At β = 1 the colluder factor is K^{-6}·1[the other two groups report identically at all six questions]. That is the same
constant for every admissible role, so

  p(y | X) ∝ Σ_{g ∈ A(X)} W_g L_g(y),  where A(X) = {g : the other two groups are identical everywhere}

and W_g is the probability of g's reported values at q' ≠ q under "g reliable".

**LOCALQ.** An admissible role needs the other two groups to self-agree at every q' ≠ q, which has probability 1 under copying, and their
**four sources to be equal at q**. Then

  p(y | O) ∝ Σ_{g ∈ A'(O)} W'_g L_g(y),  with A(X) ⊆ A'(O),

where W'_g is the probability of g's *agreement bits* under "g reliable". In general W'_g ≠ W_g.

**Case 1: |A'(O)| = 1.** Then A(X) = A'(O) = {g*}. Both posteriors equal the normalised L_{g*}(y), so the weights do not matter.

**Case 2: |A'(O)| ≥ 2.** Let g' ∈ A'(O) with g' ≠ g*.
- g' ∈ A' means the four sources of the pair excluded from g', which contains g* and a true colluder h, are equal at q. So g* reports
  (v, v) at q, and so does h.
- g* is admissible, so the two true colluders h and g' are identical at q. Hence g' also reports (v, v).
- So all six reports at q equal v, L_g(y) is the same function for every g, and both posteriors equal that normalised function whatever the
  weights W_g, W'_g. ∎

Numerical check (review 1): max |FULL − LOCALQ| = 1e-16. Extra admissible roles occur in 2.8% of episodes, and all six reports at q are
equal in every one of them.

So, **for G = 3, C = 2, β = 1 and on-support inputs**, cross-group value equality at the *other* questions (the content of J_H beyond q)
is never needed. A **local** cross-group relation (equality of the groups' values at q) together with per-group self-agreement bits is
exactly sufficient. This sharpens XIII's statement that "J_H is needed on the S = 3 slice": what that slice
needs is the local equality at q.

At β = 0.6 the proposition fails (6.1% of decisions differ). The extra information is cross-group value equality at the questions other than q: the number of
questions at which all four sources of a pair agree, n4. It enters J_H^β = Π_q (1 − β + βK³·1[all four equal at q]).

**Valid isolating pairs (β < 1) [num].** Take x and form x' by relabelling one group's two sources with a non-identity permutation of the
three values at every question except q. This leaves every per-group statistic unchanged (self-agreement pattern, hence e_g, c_g and W_g)
and the values at q unchanged (hence L_g). Only cross-question identities change. On 1,500 such pairs:
- the exact posterior changes, by a mean |Δ centred log-prob| of 0.73;
- the exact decision changes on 21.5% of pairs;
- the independent-copying (per-group product) rule predicts a change of exactly 0 (max |Δ| = 0.000).

Both members of every pair are on the support at β = 0.6. At β = 1, 31.5% of pairs remain on the support, and in 99% of those the
relabelled group was the reliable one. Their exact change is exactly 0, as the Proposition requires. The remaining pairs are off-support.
So at β = 1, J-pairs are "off-support or zero exact change" (math review 1 corrected the earlier "all off-support"). The construction
uses only the corrected shared-copy posterior.

## C2. Is the relation present in frozen layers 0–1? (`scripts/decode_J.py`, ridge probes, β = 0.6)

Target: n4 for each of the three group pairs. The table reports test R² *beyond* what per-group statistics explain (dev):

| features | uP161 (late readout) | uP160 (early readout) | uP163 (early readout) |
|---|---|---|---|
| resid1, all 36 source tokens | 0.91–0.94 | 0.98–0.99 | 0.95–0.97 |
| resid2, all 36 source tokens | 0.81–0.91 | 0.97–0.98 | 0.92–0.95 |
| resid2, the 7 readout tokens | **0.39–0.58** | **0.80–0.87** | **0.81–0.84** |
| raw one-hot input (linear reference) | 0.00 | 0.00 | 0.00 |

Frozen C = 1 models compute the cross-group identity: it is linear in their features but not in the raw input. It is present at the
source tokens in every class. It reaches the readout tokens **strongly only in early-readout models**, consistent with their layer-1
readout heads pooling relational features from all questions (ALTERNATIVE_NATIVE_DECISION_ALGORITHMS §2).

**Decodable is not used.** In the native uP161, transplanting resid1 or resid2 of x' into x (J-pairs) changes the output by slope 0.00 at
every token set (`scripts/jpath.py`). The C = 1 network carries the information and ignores it, as its C = 1 objective allows.

## C3/C4. What adaptation learns, and whether layer 2 is necessary (`scripts/adapt2.py`, β = 0.6, 1000 steps per arm)

The J-pair slope is the response to J-pairs regressed on the exact change; 0 means the cross-question relation is ignored.
Exact-Bayes accuracy is 0.792 and the LOCALQ ceiling is 0.758.

| arm | uP161 (late) acc / J slope | uP160 (early) acc / J slope |
|---|---|---|
| native (C = 1) | 0.471 / 0.00 | 0.447 / −0.00 |
| readout only | 0.459 / 0.00 | 0.497 / 0.00 |
| layer-2 attention only | 0.703 / −0.00 | 0.771 / **0.49** |
| layer-2 MLP only | 0.610 / 0.00 | 0.728 / 0.03 |
| layer 2 + readout (full) | 0.719 / −0.00 | 0.779 / **0.61** |
| fresh layer 2 + readout | 0.720 / −0.00 | 0.781 / **0.64** |
| **layers 0–1 only, layer 2 + readout FROZEN** | **0.789 / 0.98** (R² 0.89) | (see FROZEN_CONFIRMATION; development rerun after OOM) |
| layers 0–1 + readout, layer 2 frozen | 0.748 / 0.32 | – |

Development findings:

1. **Necessity is reversed at β = 0.6 in the late-readout model.** Every layer-2-side repair of uP161, including a freshly initialised
   layer 2, stays below the LOCALQ ceiling and ignores the cross-question relation. Adapting **layers 0–1 with layer 2 frozen** reaches
   Bayes (0.789 vs 0.792) and uses J (slope 0.98). Layer-2 adaptation is therefore neither necessary nor, here, sufficient.
2. **Where early adaptation puts J** (`jpath.py`, uP161 early01).
   - After adaptation, the J-dependence enters at resid1 from the *other-question* source tokens (slope 0.99).
   - By resid2 it has been moved to the readout token (0.49) and the q source tokens (0.42). These are exactly the positions the frozen
     layer-2 mixture heads read.
   - The adapted layers 0–1 write the cross-question relation into the frozen mixture's inputs. The frozen layer 2 then reuses its C = 1
     mixture operation.
3. **Class conditions what a final-layer repair can use.**
   - In the early-readout uP160, the cross-question relation already reaches the readout token (decoding 0.80–0.87), and a layer-2 repair
     partly uses it (slope 0.49–0.64, acc 0.77–0.78 > the LOCALQ ceiling).
   - In the late-readout uP161 the readout token carries it only weakly (0.39–0.58), and no layer-2 repair uses it.
   - Q6 freezes this as a prediction.
4. **Optimisation versus capacity (control, dev).** With 4× the budget (4000 steps), the layer-2 repairs of uP161 reach 0.731 (full) and
   0.733 (fresh), against 0.719/0.720 at 1000 steps. Their J-pair slope stays at −0.001. In the late-readout model, more layer-2
   training does not unlock the cross-question relation. This is evidence against a simple optimisation-budget explanation. It is not a
   proof of architectural impossibility: other learning rates, longer training or other initialisations were not tried.

## C4 at β = 1 (dev, uP161; `logs/adapt2_b1/`)

| arm | acc | S = 3 not-all-agree acc |
|---|---|---|
| exact Bayes | 0.798 | – |
| layers 0–1 only, layer 2 + readout frozen | **0.801** (KL 0.008) | 0.973 |
| layer 2 + readout | **0.797** (KL 0.004) | 0.973 |
| uP160 (early readout): layers 0–1 only | **0.799** (KL 0.002) | 0.973 |
| uP160: layer 2 + readout | **0.798** (KL 0.003) | 0.973 |

At exact copying, both arms suffice. So a layer-2 change is **not necessary** at β = 1 either, and neither is an early-layer change. This
fits the Proposition: only the local relation at q is needed, and it is available to both kinds of repair. This holds in both classes (uP161 late, uP160 early).

## C5. Connection to the sufficiency theorem

The neural repairs follow the boundary.
- At β = 0.6 the accuracy gap between the LOCALQ ceiling (0.758) and Bayes (0.794; 200,000 episodes, s.e. of each accuracy ≈ 0.001) is
  attributable to cross-group equality at the other questions, which is the only information LOCALQ censors.
- The late-readout layer-2 repairs land at 0.72, below that ceiling, with no J use. Only repairs that use J (early01 in both classes;
  layer 2 in the early-readout class) exceed it.
- At β = 1 no cross-question relation is needed (Proposition). XIII's β = 1 adaptation results are consistent with that: layer-2
  attention-only or MLP-only adaptation reached Bayes in 5/5 parents.

Not claimed:
- that n4 / J_H^β is the *unique* sufficient intermediate (any statistic determining the pairwise identity counts would do);
- that the late-readout failure is an architectural impossibility rather than an optimisation limit, unless the long-training control
  says so.
