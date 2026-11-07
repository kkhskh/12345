# Final Step 1 verdict

**Primary verdict: `STEP1_NECESSARY_CONDITIONS_ONLY`.**

**Secondary description, scoped to the investigated classes: `STEP1_ORIGINAL_CONSISTENCY_HYPOTHESIS_UNSUPPORTED`.** The classes are listed in `CAUSAL_BRIDGE_MAIN_ARGUMENT.md` §9.

Preserved, unchanged: `COR_PHASE1_CONDITIONAL_THEORY_REPAIR_REQUIRED`; `OBSTRUCTION_READY = NO`; `PHASE_2 = NO GO`. Step 2 is not opened. No pretrained-model experiment was run. No sealed or historical evaluation data was used. No historical file was modified (`logs/historical_files_unchanged.txt`).

## Reasons in one paragraph

On canonical variables there is no identification problem left: nothing is fitted, and the scalars used are invariant under the architecture's own symmetries. The question is therefore well posed, and the answer can be derived exactly. In a residual Transformer, one evidence source's effect on a decision can depend on other sources only through nonlinear reads, and same-cut relations are provably inert. At the two architecture-native nonlinear reads, the dependence is ordinary normalization:

- **softmax** brings other sources in only through attention reallocation, multiplied by a difference of readout attributions;
- the **final norm** acts as a temperature with a pull toward the bias ranking, and does nothing at all to rankings under RMSNorm.

Neither carries consistency-specific information. Neither beats head patching or the attention-weight gradient at equal information. The one premise that would carry the result to the decision (A4) is close to restating the outcome. What the step adds are exact boundaries that any future consistency mechanism must respect.

## The 48 questions

1. **What precisely is the consistency property?** Three were tested:
   - agreement between independently manipulable sources (task-semantic);
   - deviation of a source from a head's weighted consensus, `v'_j − M` (readout-defined);
   - "writer alignment" in a decision-null space, which turned out to be the norm of a sum.

   The original `O`/`D` was not reused.
2. **Does it correspond to actual computation?** Partly. The consensus `M` is the head's output projected on `ũ`, so it is computed. The deviation is not a computed variable. It enters the increment only multiplied by a weight change (CI.3*). The norm is computed by LN, but it is not a relation (SS).
3. **Does the model explicitly respond to it?** No. Every softmax head satisfies the consensus identity by algebra, whatever its function. There is no dedicated operation (H1 rejected).
4. **Can it still predict a causal response?** For a head's direct numerator term, yes, exactly. But that predictor is head patching (route 2.2 adds nothing distinctive).
5. **Why would the prediction follow?** From weighted-mean algebra (CI) and the LN formula (NC). Both are identities.
6. **What determines response direction?** For the head term: `sign(w(v'−v) + Δw(v'−M))`, own change plus reallocation. For a norm edit: `−sign(a)·sign(Δσ²)`. For the decision: these plus the σ channel and downstream paths. None of these is a consistency relation.
7. **Can the same consistency occur with opposite decisions?** Yes. SS fibers; the agreement indicator's conflict fiber (FIB.1); D′ at the decision level.
8. **When is the ambiguity eliminated?** For the layer-0 head term, by knowing `w, w', v, v', M` (CI.3*). For norm edits, by knowing `a, κ, σ_min` (NC.5). At the decision level, never by the relation alone.
9. **Can those conditions be verified independently?** Yes, from activations and frozen weights, without the response. But they are attribution quantities.
10. **Can a wrong decision be computationally consistent?** Yes. Unanimous wrong evidence (CI.4); CORR.
11. **Can a correct decision be inconsistent?** Yes. In a conflict, the stronger side can be right.
12. **Does exact structural compatibility become vacuous?** Yes (1A Theorem Z).
13. **Does approximate compatibility measure surrogate error?** Yes (1A Theorem App).
14. **How is surrogate error separated from decision conflict?** Here, by not approximating anything. Conflict is a task-semantic relation between sources. What the model does with it is ordinary normalization.
15. **Are the candidate states causally sufficient?** The canonical variables are complete by construction: the input tokens and the full residual. "Candidate states" in the `P`-sense are not used.
16. **Are they identifiable?** Yes. The scalars and centered writes are invariant. Raw writes are not (review B, M3).
17. **Can arbitrary maps manufacture agreement?** Not here, because no maps are fitted. With fitted maps they can (Phase 1 Attack E; Sutter et al. 2025).
18. **Can arbitrary metrics reverse conclusions?** Not for RB on a frozen scalar (RB.2). Switching the scalar can (review B, S1), so the scalar must be frozen.
19. **How is candidate identity defined?** The vocabulary index. Pair margins on the shared logit scale.
20. **Can candidate-specific consistency rank competing logits?** No consistency relation was found that does. NC ranks *within C* under norm edits, toward `κ`, and not at all under RMSNorm. That is the norm channel, not consistency.
21. **What role does downstream readout play?** It supplies every sign: `ũ` defines `v_j` and `M`, and `sign(a)` enters NC.
22. **Are decision labels or logits hidden in the predictor?** No responses are. The readout direction is, by design, as in DLA. The base margin enters NC through `sign(a)`.
23. **Is any repair objective circular?** The mixed objective is, by MIX. None is used.
24. **Are parameters fixed independently of evaluation outcomes?** There are no fitted parameters. The design choices must be frozen (the ledger).
25. **What theorem connects the structural relation to a decision response?** For the direct and final-layer terms: CI, CI.3*, CI-5, NC. For the decision: none.
26. **Which assumptions carry the causal information?** At the decision level, A4. It is rejected as outcome-adjacent.
27. **Are those assumptions independently meaningful?** A1–A3: yes. A4: no.
28. **Does the theorem survive the strongest counterexample?** The identities do. Any decision-level reading does not: D′, the σ channel (591/4000), self-repair.
29. **Does it reduce to ordinary attribution?** Yes. To head-patching DLA and the attention-weight gradient.
30. **Does it reduce to causal abstraction?** "A head is a weighted-mean aggregator" is an exact abstraction instance. Nothing beyond that.
31. **Does it reduce to standard sensitivity theory?** Yes, plus LN self-repair (Rushing and Nanda 2024), entropy neurons (Stolfo et al. 2024), and conjoint measurement.
32. **What distinctive scientific prediction remains?** None that is consistency-specific. Possibly new minor corollaries: the NC flip criterion and `κ` limit, and removability as an interpretability criterion.
33. **What would falsify it?** Nothing to falsify remains for consistency. RB.6 falsifies relation-blindness, and the NC flip criterion is falsifiable for LN-with-bias models. Neither is a consistency claim.
34. **What observations would identify the mechanism in a frozen Transformer?** For any *future* consistency hypothesis:
    - a component whose output is nonlinear in an agreement variable, with salience held fixed (for example, an MLP neuron responding to the product of two evidence features);
    - frozen head sets and a path-patching convention;
    - per-context RB on a frozen scalar;
    - a direction prediction that holds when attention weights are clamped.

    Review D lists the trace quantities (BOS mass, `M`, σ, σ*, σ_min).
35. **Which failures are formally resolved?** F5 and F6 (`RESOLVED_UNDER_IDENTIFIED_ASSUMPTIONS`); F7 and F30 (`RESOLVED_CONDITIONALLY`).
36. **Which remain unresolved?** F1, F2 (decision level), F4, and F29.
37. **Which are impossible under the original unrestricted assumptions?** F2 and F3 (Phase 1 Attacks A and B), and F25 as stated. Here, also same-cut relations (SS), the bare agreement indicator (FIB.1), and RMSNorm norm-channel flips.
38. **Which directions have become unproductive?** Residual and coboundary scores (`O`, `D`, and successors). Same-cut agreement. Readout-defined consensus relations. Norm-channel "alignment". Generic RB violations as evidence of consistency.
39. **Which candidate deserves more investigation?** None of H1–H3 as a consistency mechanism. If the project wants a consistency hypothesis at all, the only untested class is MLP or QK computation that is nonlinear *in the agreement variable*, with a salience-clamped prediction. That is a new hypothesis, and Rule C says it must be proposed separately.
40. **Should the O/D mechanism be abandoned?** Yes (main argument §11).
41. **Does the broader distributed-computation hypothesis remain meaningful?** As ordinary mechanistic interpretability (circuits, self-repair, normalization), yes. As a *consistency* theory, it is unsupported in the investigated classes.
42. **What would justify ending Step 1?** This result: every investigated bridge fails, with the failure located. The recommendation is to end Step 1 with this verdict.
43. **What would justify further theoretical reconstruction?** A concrete, architecture-located operation that is nonlinear in an agreement variable, proposed in advance, with a direction claim that survives salience clamping.
44. **What would justify a prospective empirical test?** Only a hypothesis meeting 43. A test of RB or of the NC flip criterion would be a test of normalization, not of consistency.
45. **Strongest positive conclusion.** In LN-final softmax Transformers, the dependence of a source's effect on other sources is completely characterized at the head and final-layer level: reallocation × attribution deviation, and a temperature with a pull toward `κ`. It can be tested without metric choices.
46. **Strongest negative conclusion.** In the investigated classes, consistency has no causal role in decisions beyond ordinary normalization. The original `O`/`D` theory has no causal interpretation.
47. **What remains unknown.**
    - MLP-computed agreement;
    - QK-composition effects;
    - deep-circuit behavior in trained models;
    - how often normalization effects matter in practice.
48. **What did this investigation accomplish that 1A–1E did not?**
    - It changed the object rather than repairing the residual. Canonical variables leave no identification problem, so F5, F6, F7, and F30 are resolved.
    - It derived **exactly** where relation-dependence can live in a real architecture (SS plus CI.3* plus NC), rather than bounding one synthetic block.
    - It brought in a metric-free criterion (RB) and showed that it is generic.
    - It subjected the claims to four genuinely separate-context reviews, which overturned the initial conditional-bridge verdict. The corrections are in `ADVERSARIAL_CROSS_EXAMINATION.md`.
