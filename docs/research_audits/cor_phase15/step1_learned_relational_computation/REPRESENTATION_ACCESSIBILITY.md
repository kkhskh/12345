# Representation accessibility (Problem B)

## 1. Definitions

The input or representation is called **Z**, the reader class **𝓡**, the training procedure and budget **𝒯**, and the target
function **f*** (the exact posterior, or its response pattern on R1/R2/R3).

- **Information-theoretic availability.** f* is a measurable function of Z. Equivalently, f* is constant on Z's level sets
  (the sufficiency test of the math document, §3).
- **Decodability (by a probe class 𝒫).** Some p ∈ 𝒫 recovers a named intermediate quantity (here m) from Z with high
  held-out accuracy.
- **Accessibility to (𝓡, 𝒯).** Training a reader of class 𝓡 on Z with procedure 𝒯 reaches f* within tolerance ε. This
  definition is *relative*: it depends on the reader class and on the budget, not only on Z.
- **Causal use.** In a trained reader, interventions on the component that encodes the quantity change the output the
  way f* requires. The component must be both necessary (ablation) and specific (matched controls do not do it).

These are four different properties. Each is strictly weaker than the next in the sense that none implies the next. Each
gap is observed in this investigation or in Investigation VII.

| gap | witness |
|---|---|
| available ⇏ decodable by a linear probe | not observed here (m is linearly decodable wherever it is available); kept as a logical possibility |
| decodable ⇏ accessible | **(i)** frozen P-count states: m decodable, one-block downstream fails (Investigation VII; here orig, auxm). **(ii)** Condition R concatenation MLP at 16k: m decodable at 0.99 from its hidden layer, E_R1 −0.46 |
| accessible to one (𝓡, 𝒯) ⇏ accessible to another | the same R features: attention/pairwise readers succeed at 16k; the MLP fails at 16k and succeeds at 48k or with Bayes targets; the linear reader never succeeds |
| accessible ⇏ causally used *through the named quantity* | not separable in successful attention readers: the m-subspace interchange carries the effect (share 1.84 / 0.51 vs random −0.23 / −0.40). A share above 1 shows the named subspace is not the *whole* account |

## 2. Representation × reader cross (development)

| representation \ reader | linear | concat MLP | group attention (1 / 2 blocks) | pairwise relnet | original 1 Transformer block | + learned relational adapter on raw ingredients |
|---|---|---|---|---|---|---|
| R: raw ingredients | fails, by Prop. C | fails at 16k; succeeds at 48k / Bayes targets | succeeds | succeeds | — | — |
| O: ingredients + m | fails (cannot form the product) | succeeds | succeeds | succeeds | — | — |
| Z: frozen P-count states | — | — | — | — | **fails** (Inv. VII C1: E −0.70 at 12k) | **succeeds** on xP80 (E −0.015); partial on xP81 (−0.29) |
| Z + m-injection (Inv. VII) | — | — | — | — | overshoots (E +0.47 / +0.13) | — |
| Z + aux m supervision (training only) | — | — | — | — | fails (E −1.10) | — |
| Z + generic adapter | — | — | — | — | fails (E −0.89) | — |
| X: ordinary tokens, from scratch | — | — | — | — | see `TRAINING_AND_OPTIMIZATION.md` §X | — |

"Succeeds" means |E_R1| ≤ 0.25 with dm_R1 > 0, |E| R2∪R3 ≤ 0.25 and KL ≤ 0.002.

## 3. What the cross says

1. **The ingredients make the relation available, and relational readers make it accessible.** Learning it requires a
   reader that can form a group-by-group multiplicative interaction (Prop. C excludes additive pooling). Attention and
   pairwise sums provide that interaction; plain concatenation provides it only after long training.
2. **The frozen P-count representation is the bottleneck for the original downstream. It is not missing the
   information.** A one-block downstream on Z fails. Giving it a target (auxm) or generic capacity (gadapt) does not help.
   Giving it a *learned relational module over the raw ingredients* does help: the downstream block can use a learned
   relational signal when one is supplied. The repair does not bypass the downstream block, which still has to combine
   the adapter's 64-d contribution with Z.
3. **Caveat on what "accessible from Z" means.** The successful repair reads the raw ingredients, not Z. The experiment
   therefore shows that *the composite (Z, learned relational adapter on raw evidence)* gives an accessible solution. It
   does **not** show that the relation can be accessed from Z alone with a better reader. The deferred runs unfreeze1,
   lora1 and deep2 (`logs/queue_deferred.txt`) were designed to test that. Their status is in
   `SELECTIVE_LEARNED_REPAIR.md`.

## 4. Theory: accessibility is a property of (Z, 𝓡, 𝒯), with one checkable sufficient condition

**Observation [P, elementary].** Suppose Z contains, for each group, an encoding from which a per-group map recovers
(n_g, a_g) exactly, where a_g is the q-report. Suppose 𝓡 contains the readers of Proposition B (gating sum, multiplied into a
per-group weight). Then f* is representable in 𝓡∘Z. This is a representability statement only. It says nothing about
whether 𝒯 finds the representation. The observations above (separable phase first; MLP needs 3× the steps) are about 𝒯,
and no theorem about them is claimed.

**No general accessibility theorem is claimed.** The development data rule out two simple candidates:

- "linear decodability of m implies accessibility to a one-block Transformer";
- "linear decodability implies accessibility to an MLP at a fixed budget".

The confirmation tests the second one again (P7).
