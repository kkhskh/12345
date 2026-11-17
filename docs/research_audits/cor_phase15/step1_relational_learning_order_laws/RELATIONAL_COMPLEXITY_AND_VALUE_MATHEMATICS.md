# Relational complexity and decision value in the XV task family

**Evidence labels.**
- **[proof]**: mathematical argument.
- **[num]**: exact enumeration over all 3⁶ = 729 inputs of one question, or a stated numerical check.

**Sources.**
- Scripts: `scripts/xv_math.py`, `scripts/xv_math_core.py`, `scripts/xv_math_verify.py`, `scripts/xv_noisy_math.py`.
- Logs: `logs/xv_math_dev.json`, `logs/xv_math_verified.json`, `logs/xv_noisy_math.json`.

## 1. Generator (src/xvtask.py)

**Setup.** Each question has a candidate Y ~ U{0, 1, 2} and six sources x₀…x₅. The parameters are gate arity a ∈ {2, 3}, evidence reliability β, and noise-pair agreement ν.

**Gate sources.**
- x₀, …, x_{a−2} are i.i.d. uniform.
- The last gate source x_{a−1} is set to −(x₀ + … + x_{a−2}) mod 3 if g = 1, and to one of the other two values (uniformly) if g = 0. Here g ~ Bernoulli(½).
- So g = [x₀ + … + x_{a−1} ≡ 0 (mod 3)].
- For a = 2, x₂ is an independent uniform "irrelevant" source. This keeps the number of input positions the same across conditions.

**Evidence sources.**
- **g = 1.** x₃ and x₄ independently equal Y with probability β; otherwise each is uniform over the other two values. x₅ is uniform.
- **g = 0.** x₅ equals Y with probability β (otherwise uniform over the others). x₃ is uniform, and x₄ = x₃ with probability ν, otherwise one of the other two values.

**Exact posterior.**
- P(y | x) ∝ L₃₄(y) if g = 1, and ∝ L₅(y) if g = 0.
- Here L_S(y) = ∏_{s∈S} (β if x_s = y, else (1 − β)/2).

**Numerical checks.**
- The sampler's empirical frequencies match the exact joint to within 1.7·10⁻⁴ on 300,000 questions.
- The Bayes formula matches the brute-force posterior to 10⁻¹⁶.
- The gate-free formula matches brute-force marginalisation over g exactly.
- Source: `logs/xv_math_verified.json`.

## 2. Structural complexity

Two different quantities need to be kept apart:
- the **gate arity**, i.e. the number of sources needed to determine trust;
- the **decision interaction order**.

### 2.1 Minimal gate arity

**Proposition 1 [proof + num].** Take any proper subset of the gate sources. Its joint law is the same under g = 0 and g = 1. So the trust relation needs all a gate sources.

*Proof.* The free gate sources x₀…x_{a−2} are i.i.d. uniform whatever g is. Now take a subset that contains x_{a−1} but omits some free source x_j. Given g, x_{a−1} is a fixed function of the sum, or uniform over two values. Averaging over the omitted uniform x_j makes the sum uniform. So x_{a−1} is uniform and independent of the remaining free sources, whatever g is. ∎

The exact mutual information confirms this:
- I(g; x_s) = 0 for every single source.
- For a = 3, I(g; x_s, x_t) = 0 for every pair of gate sources.
- I(g; all a gate sources) = ln 2 = H(g).

So the arity-3 gate is a genuine three-way relation, and no pairwise shortcut exists.

### 2.2 Minimal decision interaction order

**Definition.** A rule has order ≤ k if its logits are a sum of tables w_A(x_A, y) over source subsets with |A| ≤ k. This is the class used in XIV.

**Proposition 2 [proof + num].** The Bayes rule has interaction order exactly a + 1.

*Upper bound.* log P(y | x) = g·log L₃₄(y) + (1 − g)·log L₅(y) − log Z. Each log L term is a sum of single-source terms, and g is a function of the a gate sources. So every term involves at most a + 1 sources. ∎

*Lower bound* [num, with the XIV Proposition 3 argument]. The expected-NLL-optimal order-a rule has gate-flip slope exactly 0, and it stays strictly below Bayes (KL 0.138–0.277).
- The moment-matching argument goes through because every order-≤ a feature falls into one of two cases:
  - it omits at least one gate source, so by Proposition 1 it is independent of (g, Y, x_E);
  - or it contains all a gate sources and no evidence source, so it is a function of g alone, and g is independent of Y.
- The order-(a + 1) optimum reaches KL < 10⁻⁶.
- Minimal decision order (verified): 3 for a = 2 and 4 for a = 3.

**Where the arity counts.** Gate arity a affects the decision order (a + 1). It also affects the order of the relation that must be computed internally: a modular sum of a tokens. It does **not** affect the decision value (§3).

## 3. Decision usefulness

**Definitions.**
- **Gate gain.** Δ_gate = L*(order ≤ a, the best rule without the gate) − L(Bayes), in expected NLL.
- **Intermediate gain.** Δ_int = L*(order 1) − L*(order ≤ a).
  - For a = 2 the intermediate class is order 2 on the evidence.
  - For a = 3 it is order 3 on the evidence, which is the complete gate-free mixture.

**Proposition 3 [proof]. Usefulness is identical across arity.**
- Given g the posterior does not depend on a, and g is identified exactly in both arities. So the Bayes NLL is equal for a = 2 and a = 3.
- The law of the evidence and of g is also identical, so the exact gate-free mixture NLL is equal too.
- The only difference is the evidence-only capacity of the order-a class: order 2 for a = 2 against order 3 for a = 3. It changes Δ_gate by 0.09% (ν = 1/3) and 0.9% (ν = 0.05).

**Values [num]:**

| a | ν | Bayes NLL | order-1 NLL | best order-a without the gate | Δ_gate | Δ_int | accuracy: order 1 / Bayes | I(g; [x₃ = x₄]) |
|---|---|---|---|---|---|---|---|---|
| 2 | 1/3 | 0.2880 | 0.5898 | 0.5653 | 0.2773 | 0.0245 | 0.791 / 0.900 | 0.124 |
| 3 | 1/3 | 0.2880 | 0.5898 | 0.5651 | 0.2770 | 0.0247 | 0.791 / 0.900 | 0.124 |
| 2 | 0.05 | 0.2880 | 0.5035 | 0.4278 | 0.1398 | 0.0757 | 0.872 / 0.900 | 0.345 |
| 3 | 0.05 | 0.2880 | 0.5035 | 0.4265 | 0.1385 | 0.0770 | 0.872 / 0.900 | 0.345 |

**Confound [proof].** With β fixed, Δ_gate + Δ_int = L*(order 1) − L(Bayes). Lowering ν makes the noise pair disagree more often, so pair agreement becomes a stronger trust cue. This lowers Δ_gate and raises Δ_int *together*. The "usefulness" contrast is therefore also an intermediate-cue contrast.

- **Accuracy.** The gate's accuracy gain differs much more than its NLL gain: 0.109 against 0.028.
- **Secondary mismatch.** Order-1 accuracy is 0.791 against 0.872.

**Other knobs considered:**
- **β.** Lowering β to 0.75 lowers both gains (Δ_gate 0.194, Δ_int 0.010; `logs/xv_math_grid2.json`). It also lowers Bayes accuracy to 0.75 and changes the evidence strength for every rule, so its confound is larger.

## 4. The intermediate strategy is diagnostically available

- **Pair agreement is a genuine trust cue.** For ν = 0.05 it carries 0.345 of the 0.693 nats of g, with a gain of 0.076. For ν = 1/3 it carries 0.124 nats, with a gain of 0.025.
- **It cannot be seen in accuracy.** The order-1 and order-a optima have *equal accuracy*: 0.791 / 0.791 and 0.872 / 0.872.
- **It can be seen through A_p** (§5).

## 5. Diagnostics and their exact reference values (src/xvdiag.py)

**Edits.**
- **Gate flip:** the last gate source is redrawn so that g switches.
- **Swap (x₄ ↔ x₅):** the pairing changes, and the multiset of evidence values is preserved.

**Exact reference values on the confirmation stream:**

| rule | G | A_p |
|---|---|---|
| order-1 optimum | 0 | 0 |
| order-a optimum | 0 | 1.00–1.12 |
| gate-free rule | 0 | 1 |
| Bayes | 1 | 0 |

**Proof that order ≤ a rules have G = 0:** Proposition 2, as in XIV.

## 6. Learning-stage predictions do not follow from the mathematics alone

The mathematics fixes the *available* stages and their exact values. It does not fix which stages gradient training visits. The hypotheses (H1, H2, RACE) add assumptions about optimisation, and those assumptions are what the frozen predictions test (TASK_VARIANT_EXPERIMENTAL_DESIGN.md §3).

## 7. Counterexamples and limits [proof by construction]

- **Gate-source law.** If the last gate source were drawn uniformly, so that P(g = 1) = 1/3, single sources would still be independent of g. But the usefulness would change, so the arity contrast would no longer be value-matched. That is why the generator fixes P(g) = ½.
- **Evidence order.** If the evidence were a pair under both regimes, the a = 3 order-a class would no longer contain the complete gate-free mixture. The 0.09%–0.9% mismatch would then grow.
- **Shortcut through x₂ (a = 2).** None. x₂ is independent of everything (MI 0).
- **Noisy gate (Workstream F).** See §8.

## 8. Workstream F: a noisy gate (mathematics only)

**Construction.** Take condition B, but let the evidence regime follow g only with probability γ. Trust then becomes uncertain given the gate. The values are in `logs/xv_noisy_math.json` and are summarised in OPTIMIZATION_AND_MECHANISM_ANALYSIS.md §4.
