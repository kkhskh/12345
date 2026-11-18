# Higher-order complexity and scaling (Workstream C)

**Evidence labels:** [proof], [num], [dev], [frozen], [expl], [spec].

**Scope.** This workstream delivers the exact arity-4 mathematics. Arity-4 neural training was **not** authorised in this session, under a rule fixed before any confirmation outcome existed (FROZEN_PROTOCOL_XVI.md §3). The scaling conclusions below are therefore limited to what the mathematics and the arity-2/3 neural evidence support.

## 1. Construction (`src/xvitask.py`, S = 7 positions per question)

- **Gate:** sources 0–3; g = [x₀ + x₁ + x₂ + x₃ ≡ 0 mod 3], with P(g) = ½. x₀, x₁, x₂ are i.i.d. uniform, and x₃ is chosen to realise g.
- **Evidence:** positions 4, 5, 6, laid out like XV's 3, 4, 5:
  - g = 1: the pair (4, 5) reports Y with reliability b1, and 6 is noise;
  - g = 0: the single source 6 reports Y with reliability b0, and the pair is noise that agrees with probability ν.
- **Lower arities** use the same 7-position format, with irrelevant uniform sources filling positions a … 3.
- The generator's 6-position format (a ≤ 3) reproduces XV bit for bit (verification V1).

## 2. Exact properties

**Proposition 1 (no shortcut through a proper subset of gate sources) [proof + num].**
- Write the gate sources as free sources x₀ … x_{a−2} (i.i.d. uniform) and the last source x_{a−1} = −Σ free + c, where c = 0 if g = 1 and c is uniform on {1, 2} if g = 0.
- **A subset without x_{a−1}** contains only free sources. These are drawn independently of g.
- **A subset with x_{a−1}** must omit some free source x_j. Then x_{a−1} = −(included free sources) − x_j + c. Since x_j is uniform and independent of everything else, x_{a−1} is uniform given the included sources, whatever c is. So the subset is independent of g.
- Hence I(g; x_A) = 0 for every proper subset A, and I(g; all a gate sources) = ln 2.
- Numerically, the maximum over all proper subsets is about 3·10⁻¹⁵ nats for arity 4 (V5) and below 10⁻¹² for every neural condition (V4).

**Proposition 2 (minimal decision order a + 1) [proof of the upper bound; num for the lower bound].**
- **Upper bound.** The Bayes logit is the log of g·L_pair(y) + (1 − g)·L_single(y). This is a sum of tables over (all a gate sources, one evidence source, y). So it lies in the order-(a + 1) log-linear class.
- **Lower bound, numerically.** The expected-NLL-optimal rules of order ≤ a all have positive KL from Bayes, and order a + 1 reaches it exactly. For arity 4, the KL from Bayes by order is:

| order k | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| KL from Bayes | 0.3017 | 0.2773 | 0.2770 | 0.2770 | 1.8·10⁻¹⁴ |

- Every order ≤ 4 optimum has gate-flip slope G = 0 (|G| < 2·10⁻¹⁸). Order 5 gives G = 1.0000.

**Proposition 3 (value matched exactly across arity when a ≥ 3) [argument + num].**
- A table of order ≤ a can contain all a gate sources only if it contains no evidence. A table with any evidence source contains at most a − 1 gate sources, which are independent of g (Proposition 1).
- A proper gate subset is independent of (g, Y, evidence) jointly, because given g the gate sources are independent of everything else. A table over all gate sources and no evidence is additive in the logit, so it cannot select between pair and single.
- So for the order-a optimum the gate is invisible. The best rule without the gate is the best rule built on evidence alone. That this additive argument leaves nothing to gain is verified numerically: order-a G = 0, and the order-a KL equals the mixture KL.
- The order-a class contains the full evidence triple once a ≥ 3. It therefore contains the exact gate-free mixture, and the order-a optimum **is** that mixture. Its value does not depend on a.
- **Numerically,** arity 4 and arity 3 in the 7-position format have:
  - gate value 0.2770182, identical to about 1.2·10⁻¹⁵ (relative);
  - intermediate gain 0.024726, identical;
  - gate accuracy gain 0.10861, identical.
- **At a = 2** the order-2 class cannot represent the three-way evidence mixture. This is the source of XV's small mismatch, 0.2773 against 0.2770 [num].

**Consequence [argument + num].** Within this family, arity 3 and arity 4 can be compared at **exactly** equal decision value, equal intermediate (agreement) benefit, equal accuracy gain, equal Bayes NLL and an identical agreement-cue law. What changes:
- the minimal number of gate sources (3 against 4);
- the minimal decision order (4 against 5);
- the input format, which has one more irrelevant position in the 3-source version.

That is a cleaner complexity contrast than XV's arity 2 against arity 3.

## 3. Feasibility decision (prespecified, FROZEN_PROTOCOL_XVI.md §3)

**Not authorised for neural confirmation**, for two reasons, both measured before any confirmation outcome existed:
- **Compute.** The frozen Workstream A + B queue takes 20.7–23.6 lane-hours, about 10.4–11.8 h of wall time on the two CPUs.
- **Format.** The XIV/XV architecture embeds exactly 6 source slots per question. Arity 4 with the pair-and-single evidence needs 7, which changes the token count (49 against 43) and the source embedding table. A fair arity-3 against arity-4 comparison would then also need arity-3 runs in the 7-source format: at least 8 more runs of 15,000 steps, about 20 lane-hours.

**Exploratory pilot rule.** At most 2 runs, and only if P1 = BROAD and at least 6 h remain after the frozen queue. The outcome of this rule is reported in §5.

## 4. What can and cannot be said about scaling

- Three complexity levels cannot identify a general asymptotic law. With arity-4 neural data absent, **no** growth model (linear, multiplicative, threshold) can be compared at all [frozen design].
- The defensible content is:
  - (i) exact task constructions in which complexity changes while decision value is held exactly fixed [proof];
  - (ii) the arity-2/3 learnability evidence from Workstream A (EXTENDED_RELATIONAL_LEARNING.md), which decides whether an arity-4 experiment would be informative.

## 5. Results and conclusions

**Pilot rule outcome.**
- P1 came out BROAD, so arity-3 learning was reliable in condition C, and an arity-4 experiment **would now be diagnostic**: arity 4 would not merely add censored curves.
- The rule's second condition (at least 6 h of session after the queue) was not met. I ended training after the frozen queue (10 h 55 min), so **no arity-4 neural run was made** [frozen rule].

**Q16 (is arity-4 training diagnostically useful?).**
- **Yes, now.** In condition C, arity 3 is acquired in 7/7 models given enough exposure (1.0–3.9), so an arity-4 run with matched value (Proposition 3) would measure a delay rather than repeat the censoring.
- It needs a 7-source input format and arity-3 baselines retrained in that format: about 20 lane-hours on this machine.

**Q17 (scaling).**
- **Supported [frozen + expl]:**
  - At equal decision value and equal competing cue (ν = 1/3), going from arity 2 to arity 3 multiplies the effective optimisation needed to acquire the gate by about 2–9× (exposure 0.37–0.52 against 1.0–3.4; different schedules, so this is exploratory).
  - At arity 3 the cost of complexity is a **delay with large variation between seeds** in condition C, and a **delay or a persistent finite-budget barrier** in condition D (strong cue, lower value).
- **Not identifiable:**
  - any growth law: polynomial, exponential, threshold;
  - whether the arity-4 cost would be a further constant factor, a larger factor, or a barrier.

  Two neural complexity levels cannot discriminate these, and none was attempted.
