# Review C/E: causal identification, circularity, representation identification

Reviewer scope: `CAUSAL_BRIDGE_MAIN_ARGUMENT.md` (MAIN), `proofs/PROOFS.md`, `proofs/bridge_math.py`, `tests/test_bridge.py`, judged against mandate §§1–4, 7–11, 23, 27. I re-ran the tests (`pytest tests -q`: 21 passed, no cache written). I also ran probes from a separate directory (`/tmp/review_C/probe.py`, which imports `bridge_math` read-only). Their numbers are quoted below. No package file other than this one was modified.

---

## Q1. Is the new object a consistency relation?

### FATAL-1: the RB verdict cannot depend on consistency, by its own invariance (MAIN §3, R4; PROOFS RB.2)
RB.2 makes the verdict invariant to "every relabeling of levels". So which levels are "support", "neutral" or "oppose", and whether A and B agree or conflict, has no effect on whether `m` is relation-blind. RB therefore tests one thing: whether two input factors interact non-removably (an ordinal interaction). It is not a test of agreement or conflict. "Relation-sensitive" is a renaming of "non-additive up to a monotone transform". Any generic nonlinear network satisfies that. Probe (d): in 200 random 1-layer models with random tokens as levels, **172/200 3×3 tables are relation-sensitive (171 with a crossover)**, and 89/200 random 2×2 tables have a crossover. Rejecting relation-blindness is the default outcome for arbitrary non-semantic levels. It carries no information about consistency.
*Correction:* drop the claim that RB "gives the consistency question an identification standard". RB is a generic interaction test. Consistency content could only come from a hypothesis that predicts *which* cells cross, keyed to agreement structure, and the package has no such hypothesis (see FATAL-2).

### FATAL-2: the softmax "consensus" crossover comes from key-side salience competition, not from value-side agreement (MAIN R2, R5, §8; PROOFS CI, D)
Once the attention weights are fixed, a head's output is **linear in the values**. Nothing in the head reads a relation among the `v_j`, and the weights are computed from `q·k` alone. The only non-additivity is the normalization `1/Z` acting on the weights. CI.2 shows that the sign is `w'_j(v'_j−M) − w_j(v_j−M)`. The deviation from consensus therefore matters only *through a change in weight*. Probe (a), the Construction D weights with W's salience set equal to N's (`s_W = 0`): the effect of N→W is **+0.327 / +0.010 / +0.0025** for B = N / S / OPP. The crossover is gone, and sub-consensus supporting evidence *raises* the margin.

This also falsifies the headline "distinctive prediction" (MAIN §8: "supportive-but-sub-consensus evidence lowers the margin") under the package's own intervention family. Probe (a2) gives the converse: weak *opposition* under agreeing strong opposition *raises* the margin (+0.125). Agreement versus conflict is not the operative variable. FIB.1 already concedes this. The operative variables are (Δ salience) × (value DLA minus weighted-mean DLA).
*Correction:* restate the result as "attention normalization dilutes: when a substituted token gains weight, the head's DLA moves toward that token's DLA". This is ordinary normalization nonlinearity (mandate H1.8), and it should not be called a reconciliation or consistency mechanism.

### SERIOUS-1: the sign is supplied by the readout, not by the relation (MAIN §2 table, R3; PROOFS NC, CI)
`ũ = P(γ⊙Δu)` is `σ·∂a/∂h`: the readout's downstream sensitivity. `v_j = ũ^T(Z_j W_V W_O)` is the value's direct logit attribution, and `M` is the head's DLA. Source–consensus deviation is therefore *DLA(source) − DLA(head)*, a quantity defined by attribution. No outcome response enters it, so it is not circular in the statistical-leakage sense. But under mandate §8 the directional information comes from the unembedding: the "relation" only compares two readout projections. In NC the package itself says `sign(a)` comes from the readout. The "relation" there supplies only whether the norm rose, and that moves every margin toward `κ`, which is a temperature or confidence effect.
*Correction:* state explicitly in MAIN §2 that every sign comes from `ũ`, and that the relation contributes only an ordering *in readout units*.

### SERIOUS-2: "writer alignment" (C) is a function of the sum, so by the package's own SS it is not a relation (PROOFS NC "Relational reading")
`Σ_{k,l} c̃_k^T c̃_l = ‖P Σ_k c_k‖²`. This is the norm of the residual, a function of the sum. SS(iii) proves that non-sum statistics of same-cut writers are inert. So "alignment moves σ" is a redescription of "the final residual norm moves σ". This is the known confidence-regulation channel (Stolfo et al. 2024), as MAIN §8 concedes.
*Correction:* remove (C) from the list of consistency relations, or reclassify it as the residual norm.

## Q2. Intervention semantics and the explicit-mechanism claim

### OK: the basic intervention semantics are sound (PROOFS §0)
Token substitution and recomputed site replacement are well defined. Nowhere is `do(C)` written for a derived statistic. Edits in `N_C` are concrete residual edits.

### SERIOUS-3: the CI.3 "added source" sign law cannot be realized by the package's intervention family (MAIN R2; PROOFS CI.3)
CI.3 requires `w_j = 0 < w'_j`. Under softmax, `exp(s) > 0`, so a present token never has zero weight. Substitution at a position is CI.2, which carries a weight-change term. At layers ≥ 2, substituting one token changes every later position's keys, values and query, so CI.1 applies with all indices moving, and no single-source sign survives. Construction D uses CI.2, not CI.3. MAIN R2 and §8 state the CI.3 law as if it applied.
*Correction:* state the decision-relevant law as CI.2 (first layer) or CI.1 (later layers), and drop "sign(v'_j − M)" as a standalone prediction.

### FATAL-3: route 2.1 ("explicit mechanism") is not justified (MAIN R2: "the architecture contains exactly the operation mandate §2.1 asks for"; §10, arrow 1)
No component computes `v'_j − M` or responds to it. That term is the algebraic decomposition of the change in a weighted mean (CI is "the algebra of weighted means", MAIN §6). Every softmax head in every model satisfies it, whatever its function, so it cannot discriminate a reconciliation mechanism from a generic output function (mandate H1.9). The mandate says "do not assume attention implements reconciliation", and this claim declares that every head does, by identity. At most, (B) is an exact *descriptor of the direct term*, which is route 2.2 at best. MAIN §8 concedes it is redundant with the attention-weight gradient on sign.
*Correction:* reclassify links 1–2 from (E) to (P), with the (P) predictor reducing to weight-space sensitivity, and delete "Established" from arrow 1.

## Q3. Assumption A4

### FATAL-4: A4 encodes the outcome (MAIN §4 A4, §7, §10 arrow 3; mandate Rule D)
A4 states that `sign(total effect) = sign(direct effect through the aggregating head)`. The total effect *is* the response `R`, and the direct effect is given by an identity (CI). So the decision-level "prediction" is A4 composed with CI: "the response has the sign of a computed quantity, assuming the response has the sign of that computed quantity". The intermediate reasoning (CI) adds no constraint on `R` beyond what A4 asserts. That fails mandate §8, last paragraph, and Rule D. "Testable by path patching" does not rescue it:
- (i) path patching on the evaluation inputs uses the evaluation outcomes;
- (ii) on a disjoint calibration family it needs an additional, unstated transport assumption (family homogeneity of path dominance);
- (iii) "the aggregating head(s)" are not specified, so the analyst can choose, after the fact, the head whose direct sign matches the total.

### SERIOUS-4: even the single-block case needs A4, and "direct effect" is ambiguous (PROOFS CI-5; MAIN §10 arrow 3)
CI-5 gives `Δm = Δa/σ' + a(1/σ' − 1/σ)`. The σ term is a second path even with one attention-only layer. Probe (b), Construction D with B = S: `Δa/σ' = −0.724` against a σ term of **+0.216** (30%, opposite sign), for a total of −0.508. Construction LN shows the σ term alone can create crossovers. `test_D_crossover_locus_is_the_consensus` checks the sign match empirically; nothing guarantees it. A4 does not say whether the direct effect is taken on `a` or on `m`.
*Correction:* define A4's direct effect on `m` with σ held at its clean value, name the head set in advance, and acknowledge that A4 is already needed in Construction D.

## Q4. Identification and frozen choices

### OK: canonical variables and symmetries (MAIN §3)
`w`, `v`, `M`, `a`, `σ`, `m` are invariant under the listed symmetries (residual permutation with γ/β, head and neuron permutation, refactorization of `W_V W_O` and `W_Q W_K^T`). No map is fitted.

### SERIOUS-5: "not fitted" is not the same as "not tunable"; the discrete choice ledger is open (MAIN §3 "underdetermined"; mandate Failure #30)
Choices that remain open:
- candidate pair and read position;
- evidence slots;
- token sets per level, including the neutral baseline token, whose *salience* determines the sign (FATAL-2);
- templates or contexts;
- layer and the "aggregating head(s)" in A4;
- number of levels;
- tie tolerance (`1e-12`);
- the split of RB.6 into training and held-out comparisons;
- aggregation over contexts.

With about 86% of random 3×3 tables relation-sensitive, a search over levels or slots will produce crossovers almost surely. The referenced freezing protocol `OUTCOME_LEAKAGE_AND_FREEZING.md` **does not exist** in the package (nor do `THEOREM_REGISTER.md`, `SOURCE_REGISTER.md` and `RESEARCH_JOURNAL.md`).

### SERIOUS-6: metric-freeness fails once tables are aggregated over a family (MAIN R6, §3; PROOFS RB.2)
RB.2 holds per table. Any family-level claim ("for a family", A4) averages margins, probabilities or log-probabilities across contexts, and a mean is not invariant to monotone transformation of per-item values. Either RB is run per context, which multiplies the tests and needs a pre-stated combination rule, or the claim to be "free of metrics and links" is false at the family level.

### SERIOUS-7: RB.6 is noncircular but uninformative about consistency (MAIN R6, §10 arrow 4)
RB.6 derives forced orders from the *null* (additivity). A violation supports "some non-removable interaction", which is the generic default (FATAL-1). The alternative predicts no direction except through the consensus locus, and that locus fails as stated (FATAL-2, SERIOUS-3) and needs A4 (FATAL-4). Everything that must be frozen for a valid RB.6 test is listed in SERIOUS-5, plus the head set and the CI.2/CI.1 form of the locus. Even with all of that frozen, a confirmation would show attention dilution, not consistency.

### SERIOUS-8: the baseline in §8 is a straw man (MAIN §8; mandate §11)
"Additive attribution predicts `sign(v'_j)`" compares against a per-value attribution. The equal-information baseline (patch the head, read its DLA change, or take the attention-weight gradient) yields exactly the CI quantity, as MAIN's own second row concedes. Nothing distinctive remains at equal information.

### MINOR-1: RB.6 box bound (PROOFS RB.6)
The coordinate bound of 100 is justified only for designs up to 3×3. Larger designs need a proof or an unbounded LP with the gauge fixed.

### MINOR-2: attention sinks (MAIN §8)
In trained models, most weight often sits on a BOS sink, so `M` is mostly the sink's DLA. The consensus is then not over evidence sources at all. This is worth stating as a scope risk.

### SERIOUS-9: procedural gaps (MAIN header, "log in `logs/`"; mandate §24)
`logs/` is empty, yet MAIN claims a test log. `WORK_STATE.md` says the verdict and reviews are pending. Four referenced files are missing (SERIOUS-5). The tests do pass on re-run, but the package claims a log that is not there.

## Q5. Verdict and substitution

### OK: the proven parts are correct
SS, NC.1–6, CI.1–4, RB.1–5, CORR, MIX and FIB are proved as stated. Retiring O/D (MAIN §11) is justified.

### FATAL-5: the package commits the substitution failure (MAIN Summary, §10; mandate §0, Rule A)
The question "does consistency have a noncircular causal role in decisions" has been replaced by three easier ones:
- does softmax normalization make a token's effect depend on the other tokens' DLA (yes, by weighted-mean algebra, in every head);
- does the final-LN norm act as a temperature (known);
- do two input factors interact ordinally (generic).

Consistency enters only as level labels, to which the RB verdict is invariant. MAIN §10 says the chain "does not fail. It is located." In fact the located break (A4) is the decision arrow itself, and arrow 1 (that the relation is a consistency relation computed by the model) also fails (FATAL-1/2/3).

## Recommended verdict

**`STEP1_NECESSARY_CONDITIONS_ONLY`**, with O/D separately judged unsupported. `STEP1_ORIGINAL_CONSISTENCY_HYPOTHESIS_UNSUPPORTED`, scoped to the investigated classes, is an equally defensible primary verdict. Those classes are: same-cut writer relations, softmax value-consensus, final-LN norm alignment, and two-factor ordinal interaction.

Reasons:
- **Not CONDITIONAL_BRIDGE.** The conditional bridge is circular. Its key assumption A4 is the decision-sign claim, and the bridge's relation is not a consistency relation (FATAL-1 to FATAL-4).
- **Not CAUSAL_PREDICTOR_ONLY.** The only predictor, consensus deviation, is not consistency-specific and reduces to the attention-weight gradient or head-patching DLA at equal information. MAIN §8 concedes this.
- **What genuinely survives.** These are necessary conditions and boundaries:
  - SS: same-cut agreement is inert;
  - FIB.1: semantic agreement does not determine the sign;
  - CI: any value-side effect is mediated by weight change, a corrected form of the result;
  - NC.5–6: the flip criterion, and vacuity when `|V| ≥ d`;
  - CORR and MIX.

**Minimum needed to restore CONDITIONAL:**
- a relation that the model computes non-linearly *in the agreement variable*, which value averaging is not;
- a direction prediction keyed to agreement structure that survives the probe with salience held fixed;
- A4 replaced by an assumption that is stated without the response and identified on a frozen calibration family, with the head set named in advance;
- the missing freezing protocol and logs added.
