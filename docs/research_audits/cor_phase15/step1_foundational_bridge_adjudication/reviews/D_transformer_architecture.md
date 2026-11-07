# Review D: is the Transformer architecture realistic?

Reviewer D works independently. It read `CAUSAL_BRIDGE_MAIN_ARGUMENT.md`, `proofs/PROOFS.md`, `proofs/bridge_math.py` and `tests/test_bridge.py`. `pytest tests -q` passes all 21 tests. Scratch checks are in `/tmp/revD/scratch.py`, which does not change the package. No pretrained weights were loaded.

**Overall.** None of the findings is FATAL. The algebra of SS, NC and CI is correct as stated. The problem is in how far the package carries it. Several prose claims go past what the theorems give for GPT-2, Pythia or LLaMA:

- The NC ranking result is absent in RMSNorm models.
- The relational reading of CI holds exactly only for heads in layer 0.
- The stated sign law is wrong for the substitution design that the package actually uses.
- The "direct effect" sign ignores the change in the final-LN scale σ.
- In trained models, A4 is more fragile than "downstream MLP re-reading" suggests.

Verdict label: `CONDITIONAL…_OPEN` can stand. The scope statements in §§5, 8 and 9 need the corrections below.

---

## Architecture facts used

- **GPT-2.** Pre-norm. `ln_f` has a bias. The unembedding is tied to `wte` and has no bias. `c_attn` has q, k and v biases.
- **Pythia (GPT-NeoX).** `use_parallel_residual: true`. `rotary_pct: 0.25`. Embeddings are untied (`tie_word_embeddings: false`). LN has `eps 1e-5` ([pythia-1b config](https://huggingface.co/EleutherAI/pythia-1b/blob/step25000/config.json)). Final LN has a bias. `embed_out` has no bias.
- **LLaMA.** RMSNorm is `weight * h * rsqrt(mean(h²)+eps)`. There is no centering and no bias. `lm_head = nn.Linear(..., bias=False)` ([modeling_llama copy](https://huggingface.co/spaces/MiguelZoo/VideoChatGPT/blob/main/models/modeling_llama.py)).

---

## SS: same-cut sum screening

**OK: SS holds in every architecture checked.**
- *Location:* PROOFS SS.
- *Argument:*
  - GPT-2 concatenates its heads and applies `c_proj` with a single `b_O`. That is a per-head sum.
  - Pythia's parallel block computes `x + attn(ln1 x) + mlp(ln2 x)`. Attention and MLP therefore sit on one cut.
  - LLaMA is sequential, so the cut is the set of heads in one layer.
- *Correction:* none.

**MINOR: R1 names the wrong operations.**
- *Location:* MAIN §5 R1, "a nonlinear operation reads two variables separately: … across features (normalization), or through an MLP nonlinearity".
- *Argument:* Normalization and the MLP read the *sum* at one position, not the writers separately. That is exactly what SS says. The only operation that reads separately is one that combines different positions (attention), or a write read at a different cut.
- *Correction:* Drop "normalization" and "MLP" from the list. Say that they read nonlinear functions of the sum.

**SERIOUS: relation (C), "writer alignment", is screened by SS, so it is not identified.**
- *Location:* MAIN §2(C) and §5 R3; PROOFS NC "Relational reading".
- *Argument:*
  - The identity `d(σ²−ε) = Σ_{k,l} c̃_kᵀc̃_l` is a quadratic expansion of `‖P·Σc_k‖²`. By SS it is a function of the sum alone.
  - The "alignment" depends on how the final residual is split into writers: heads, layers, neurons, or the embedding and bias terms. That split is an analyst choice. This contradicts the claim in §3 that "no choice is left".
  - The only invariant object is σ at the read position.
- *Correction:* Restate NC as being about the norm of the centered final residual. Call "alignment" a decomposition-relative description, not one of the two relation-sensitive aggregation sites.

**MINOR: the SS(iii) fibers cannot be reached from the input.**
- *Location:* PROOFS SS(iii).
- *Argument:* Zero-sum edits of head outputs cannot be produced by any token substitution. Across inputs, a same-cut statistic can still *predict* decisions, because it covaries with the sum. SS rules out mechanism, not prediction.
- *Correction:* Add one sentence saying this.

---

## NC: the final-LN norm channel

**OK: the algebra is exact for LayerNorm models.**
- *Location:* NC.1–NC.5.
- *Argument:*
  - I re-derived each part:
    - `a(h+v) = (γ⊙Δu)ᵀPv`;
    - the sign law `sign R = −sign(a)·sign(2h̃ᵀv+‖v‖²)`;
    - the `σ_min` formula;
    - the flip criterion `aκ<0 ∧ −a/κ > σ_min`.
  - Restricting `N_C` to `1ᵀv = 0` loses nothing, because components along `1` change neither `a` nor `σ`.
- *Correction:* none.

**OK, with a caveat: κ in GPT-2 and Pythia.**
- *Location:* MAIN §9, Level 7.
- *Argument:*
  - For GPT-2, `κ_{cc'} = (w_c − w_c')ᵀβ_f` exactly, with `w` the rows of the tied `wte`.
  - The value is generically nonzero. It equals the folded `b_U` difference that TransformerLens' `fold_ln` produces.
  - Pythia gives the same with the untied `embed_out`.
  - "Generically nonzero" says nothing about size. Flips need `σ' ≥ σ* = −a/κ`. When `|κ|` is small relative to a typical margin `a/σ`, the threshold `σ*` is many multiples of σ. Plausible upstream writes cannot reach it, and the flip channel is practically null.
- *Correction:* State that relevance depends on the ratio `|κ|/|a/σ|` on the task's candidate pairs, and that this ratio is not observed. It is the first frozen measurement (§Q4).

**SERIOUS: the RMSNorm analogue is immediate, and it removes the ranking result.**
- *Location:* MAIN A2, §7 ("invalidates the P in ũ and the reachable-set formula") and §9 ("would need re-derivation").
- *Argument:*
  - For LLaMA, Mistral, Gemma and Qwen (RMSNorm, no norm bias, no `lm_head` bias), the margin is `m = (g⊙Δu)ᵀh / rms(h)`. NC.1 holds with `P = I` and `κ ≡ 0`.
  - NC.2–NC.4 carry over unchanged. Take `N_C = {v : (g⊙Δu_c)ᵀv = 0}`, drop the `1` constraint, and replace `h̃` with `h`.
  - NC.5 then reduces to "no flip is ever possible", since `aκ = 0`. The limit is a zero margin, not `κ`: the distribution over `C` tends to uniform.
  - The norm channel in these models is therefore pure temperature: it is ranking-preserving within `C`. My scratch check found the within-`C` ranking unchanged in 2000/2000 random null edits.
  - The "signed ranking-level statement" (R3) exists only for LN-with-bias models such as GPT-2, Pythia and OPT.
- *Correction:* Replace "needs re-derivation" with this result. Restrict R3 explicitly to models with a final-norm bias or an unembedding bias.

**SERIOUS: a margin within C is not the model's decision.**
- *Location:* NC.3 and NC.5; MAIN R3, "moves the whole candidate ranking".
- *Argument:*
  - `v ∈ N_C` leaves only the logits *in C* on one line. Tokens outside C have `(γ⊙Δu)ᵀv ≠ 0`, so a large null edit can make a token outside C the argmax.
  - The full-softmax probabilities of the candidates in C are not controlled.
  - As `‖v‖→∞`, logits outside C tend to `κ + √d·(γ⊙Δu)ᵀn`, where `n` is the edit direction. That is not the bias ranking.
- *Correction:* Say "the ranking restricted to C". Add the escape condition for tokens outside C.

**MINOR, approaching SERIOUS: the claim "N_C = {0} for full vocab" is correct, but the effective null space is the wrong object as stated.**
- *Location:* NC.6; MAIN §9.
- *Argument:*
  - When `|C| ≥ d`, the forms `{1, γ⊙Δu}` generically span `R^d`, so `N_C = {0}`. Correct.
  - Stolfo et al. 2024 define the effective null space as the span of the bottom `k` singular vectors of `W_U`. They report `k = 12` for GPT-2 Small (with "a sharp drop… around index 755") and `k = 40` for LLaMA2-7B ([arXiv 2406.16254](https://arxiv.org/html/2406.16254v2)).
  - Two corrections are needed:
    - The relevant matrix is `P·diag(γ_f)·W_U`, not `W_U`.
    - With an approximate null direction, `m = (a + δᵀv)/σ' + κ` with `δ ≠ 0`. Single crossing (NC.3) is no longer guaranteed. The large-norm limit shifts by `√d·δᵀn`, so the error does not vanish.
  - The package also understates NC's scope. For a *pair*, the exact `N_C` has dimension 766 in GPT-2 Small. The effective null space matters only for full-vocabulary claims.
- *Correction:*
  - Add the bound `|m − (aτ+κ)| ≤ ‖Π_N(γ⊙Δu)‖·‖v‖/σ'`.
  - Note that NC is non-vacuous for small `C` even without an effective null space.
  - Note that entropy neurons (Stolfo et al.) are the known real instance of the channel.

**MINOR: outlier dimensions can dominate σ in practice.**
- *Location:* NC in GPT-2.
- *Argument:* GPT-2 has a few high-magnitude residual dimensions, the known outlier or rogue dimensions, which can dominate `σ` at the read position. If so, `σ_min` and the reachable range are set by those dimensions, not by evidence writes.
- *Correction:* Record per-dimension shares of `σ²` at `r` as a frozen observation.

---

## CI: the softmax consensus identity

**OK: the identity holds for GPT-2, Pythia and LLaMA.**
- *Location:* CI.1–CI.4.
- *Argument:*
  - Rotary (including Pythia's 25% partial rotary), GQA and the causal mask change only how scores are computed. CI conditions on the weights.
  - GPT-2's `b_K` cancels in the softmax. GPT-2's `b_V` adds the constant `ũᵀb_V W_O` to every `v_j`, and `v'_j − M` is invariant to that shift.
  - In a causal decoder, every path from position `j < r` to `r` passes through some attention edge. CI therefore sits on every path, which is a real strength.
- *Correction:* Mention the invariance to the `b_V` shift. The code has no `b_V`.

**SERIOUS: the relational reading (CI.2/CI.3) is exact only in layer 0.**
- *Location:* MAIN R2 ("for every head … any layer"); PROOFS "Transformer instance".
- *Argument:*
  - CI.1 holds for any recomputed intervention. Its *relational* content, though, needs only index `j` to change.
  - A token substitution at `j` changes `Z_i` for every `i > j` from layer 1 onward, including the query at `r`. In later layers every `(w_i, v_i)` moves.
  - In my 2-layer random-model check, swapping the token at `j` gave a layer-1 head output change whose sign differed from the single-source CI.2 sign in 103/400 cases.
  - In layers ≥ 1, `Σ_j w'_j(v'_j − M)` is simply the head-output difference. It can be computed only from the intervened run, and it is identical to direct logit attribution (DLA) of the clean-minus-patched head output.
- *Correction:* Restrict "sign = sign(v'_j − M)" to layer-0 heads, or to heads whose upstream inputs at positions `≠ j` are frozen (a path-specific effect). Then state which.

**SERIOUS: the stated sign law is wrong for the substitution design used.**
- *Location:* MAIN R2 and §8, "the sign … is sign(v'_j − M)"; PROOFS D proof.
- *Argument:*
  - A slot is substituted (`N → W`), not added. CI.2 gives the sign as `sign[w_W v_W − w_N v_N − (w_W − w_N)M]`.
  - When the substitution does not change the attention weight (`w_W ≈ w_N`), the consensus term cancels. The sign is then `sign(v_W − v_N)`, the additive prediction.
  - Reversal needs both of the following:
    - the substitution raises salience;
    - `M > (w_W v_W − w_N v_N)/(w_W − w_N)`.
  - Construction D gets away with `sign(v'_j − M)` only because `w_N/w_W = e^{−3}`.
- *Correction:* State the prediction with both weights. The distinctive prediction applies only to cells where salience changes.

**SERIOUS: the "direct effect" omits the change in the final-LN scale σ.**
- *Location:* MAIN R2 and §8; A4 wording.
- *Argument:*
  - By CI-5 itself, a head's write changes `σ` as well as `a`. Its direct effect on `m`, with `ln_f` live, is `Δa/σ' + a(1/σ' − 1/σ)`.
  - In my scratch check of random 1-layer models, the sign of the head's numerator change `Δ(ũᵀc_h)` differed from the sign of its live-LN direct effect in 78/600 cases. Construction LN is this same effect.
  - DLA conventions that freeze the LN scale and conventions that recompute it give different "direct effects".
- *Correction:* Define the direct effect with a fixed convention for σ: frozen at clean, or recomputed. Make A4 refer to that convention.

---

## Q2: realism of the consensus law and A4

**SERIOUS: attention sinks shrink the dilution regime.**
- *Location:* Construction D; MAIN R5 and §8.
- *Argument:*
  - In D, BOS has salience 0 and value 0, and gets weight 0.01. The strong source takes about 97% of the mass.
  - Trained models often put much of a head's attention mass on BOS or another sink. With `M ≈ α_S v_S` and `α_S` small, the threshold `M > v_W` usually fails, and the head acts additively.
  - Scratch check: raising only `s_BOS` in D (keys re-solved the same way) gave these results.

    | `s_BOS` | BOS weight | crossover |
    |---|---|---|
    | 0 | 0.01 | present |
    | 1.5 | 0.49 | present, reduced (−0.15 vs +0.23) |
    | 2 | 0.81 | gone (+0.06 / +0.02) |
    | 3 | 0.99 | gone |

  - The law therefore has teeth only for sharp, sink-free heads dominated by content, such as copy and name-mover heads.
- *Correction:* State the operative condition `α_S·v_S ≳ v_W` with the sink mass included. Make the BOS mass of each candidate head a frozen observation.

**SERIOUS: A4 is under-specified and fragile.**
- *Location:* MAIN A4 and §7 D′.
- *Argument:*
  1. **Which heads?** A substitution at `j` reaches `r` through every head in every layer (144 heads in GPT-2 Small). Multi-head redundancy means several heads aggregate the same source in *different* consensus regimes, so their CI terms can have opposite signs. A4 is not defined until the set of aggregating heads `H*` is frozen.
  2. **Re-reading.** In GPT-2, the MLP of the same layer reads the head's write (the D′ path). In Pythia's parallel blocks it does not; only later layers do. Known re-readers can invert a head's effect: negative name movers and backup heads in IOI, and copy-suppression heads in GPT-2 Small.
  3. **Self-repair.** Rushing and Nanda (2024) find that self-repair after ablation is imperfect and highly variable across prompts. It is attributed to final-LN scaling and to Anti-Erasure neurons, and it sometimes overcorrects beyond the original effect ([arXiv 2402.15390](https://arxiv.org/abs/2402.15390)). Overcorrection is exactly a sign violation of A4 on some prompts. Partial compensation preserves the sign, so A4 can hold on average and still fail per item. The Hydra effect (McGrath et al. 2023) points the same way. Both findings come from ablations, not substitutions, so the rate under substitution is unknown.
  4. **Magnitude.** The CI term of one head for a weak-support swap is small compared with the summed additive terms of other heads. A4 needs the *dilution head(s)* to dominate the total, not merely to be sign-correct.
- *Correction:*
  - Restate A4 as: for a frozen `H*` and a fixed path-patching convention (what is frozen at clean, and how σ is treated), `sign(total) = sign(Σ_{h∈H*} CI_h)` on at least a pre-registered fraction of items.
  - Report the share of total effect carried through `H*`.
  - Expect A4 plausibly only in circuits dominated by a few heads, such as late-layer name movers in IOI-type tasks. It is not a model-wide property.

**MINOR: additive attribution as a baseline is partly a strawman.**
- *Location:* MAIN §8, first row.
- *Argument:* DLA of the head-output *difference* already equals the CI quantity exactly. Only per-source DLA with frozen attention (`α_j v_j`) predicts `sign(v'_j)`.
- *Correction:* Name the per-source frozen-attention baseline explicitly.

---

## Q3: the constructions D, D′ and LN

**OK, as an existence proof: D is valid.**
- *Location:* `dilution_model`.
- *Argument:*
  - LN'd vectors lie in the 7-dimensional space `1^⊥`. Six generic token vectors are linearly independent there, so `lstsq` solves *any* per-token table for salience, value and query exactly.
  - The construction is therefore unconstrained interpolation, not evidence that trained weights realize the regime.
  - What it assumes in trained models is only that salience and evidence are linear functions of the LN'd embedding, which is plausible.
- *Correction:* Replace "in a real block" (R5) with "in a block of the real architecture with designed weights".

**MINOR: D's parameter regime is chosen, not typical.**
- *Location:* D parameters.
- *Argument:*
  - D uses `λ = 3` (sharp attention), `d = 8`, an MLP that is either off or random with a small 0.3-scale `W_in`, and no sink.
  - D′ is a single hand-placed inverting neuron with `W_in = 4·e_out`.
  - LN uses a null component `3n`, six times its content part `0.5e`.
  - These regimes are where the effects are large. Their prevalence in trained models is unknown.
- *Correction:* Report the parameter region in which D's crossover survives: sink mass, `w_N/w_W`, `M/v_W`. The sink sweep above is one slice of it.

---

## Q4: frozen observations needed to identify the mechanism in GPT-2

These are listed for planning only; no experiment was run.

**From weights alone:**
1. `κ_{cc'} = (w_c − w_c')ᵀβ_f` for every pre-registered candidate pair.
2. `‖γ_f⊙Δu‖` for the same pairs.
3. The spectrum of `P·diag(γ_f)·W_U`, to fix the effective null space and its leakage `‖Π_N(γ⊙Δu)‖`.
4. For layer-0 heads: per-token salience `LN(E_t)W_Q`/`W_K` and value `ũᵀLN(E_t)W_V W_O` for the evidence tokens. This gives `v'_j` and the salience change `w_W/w_N` without any forward pass.

**From frozen forward traces** (a future authorized run, with the design fixed beforehand):

5. At `r`, for each head:
   - the weights `(w_j)`, including the BOS/sink mass;
   - the values `(v_j)` on `ũ`;
   - the consensus `M` in the B-only prompt;
   - the dilution-cell indicator `w_W v_W − w_N v_N − (w_W−w_N)M < 0`.

   These are computed *before* margins are read.
6. `σ` at `r`, the per-dimension shares of `σ²`, `a`, `σ*`, and `σ_min`, all for the candidate pairs.
7. Path-patching decomposition of each substitution effect into the following parts, with the convention fixed in advance:
   - the direct terms of `H*`, with σ both frozen and live;
   - paths through downstream heads;
   - paths through MLPs;
   - the LN-scale term.

   Then measure the A4 sign agreement rate and the share of total effect carried by `H*`.
8. The RB LP on the frozen factorial. Its held-out comparisons must be chosen from items 4–5, not from margins.
9. Several token instantiations per level. The reversal depends on the magnitude of `v'_j`, so a single exemplar per level confounds level with token.

**Identification requires that:** the items 5–7 measurements put the predicted dilution cells where the RB violations appear, with A4 holding on those cells under the frozen convention. A CI-sign match alone is not enough, because in layers ≥ 1 it is DLA of the head difference.
