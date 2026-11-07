# Independent mathematical and scientific audit — COR Phase 1.5 Step 1 foundational bridge

**Audit date:** 2026-11-07  
**Submitted archive:** `cor_phase15_step1_foundational_bridge_adjudication.zip`  
**Archive SHA-256:** `239924ce7a395622cd62777b95db7e6124ceae9c9983813cbda66f7506735af6`  
**Primary submitted verdict:** `STEP1_NECESSARY_CONDITIONS_ONLY`  
**Independent decision:** **Accept as a partial, scoped negative investigation and a useful theorem package. Reject its assertion that the mandated foundational investigation is complete, and reject several overbroad claims about identification and available causal mechanisms. Do not close Phase 1.5 Step 1 on this evidence.**

## 1. Executed verification

- ZIP enumerated and extracted safely. 45 ZIP entries, 228,357 uncompressed bytes.
- `python -m pytest tests -q` in a clean extracted package: **26 passed in 2.56 s** (`pytest.log`).
- `python scripts/verify_manifest.py`: **34 entries, 0 mismatches** (`manifest.log`).
- Standalone independent NumPy calculations executed with **no import from the submitted implementation** (`independent_standalone.py` and `.log`).
- Additional reproduction using the submitted executable block with *independently designed* adversarial parameters (`independent_checks.py` and `.log`).
- I did **not** access the underlying GitHub repository, original Phase 1 source files, pretrained model weights, or purported separate-agent execution sessions. The ZIP contains this submission and its own accounts of earlier work. Therefore historical-repository integrity and cited prior test runs cannot be independently certified from this ZIP alone.

**What the tests establish:** The provided finite-dimensional examples and coded identities run as documented. They do **not** establish completeness of the search or absence of an agreement-sensitive mechanism in the unexamined classes.

## 2. Correct findings to retain

1. **Same-cut additivity**, properly restricted to no bypass, means that zero-sum edits to additive writers leave all downstream computations unchanged. A relation among writers cannot have a separate downstream effect when their sum and all bypasses are held fixed.
2. **Softmax weighted-mean identities** (CI) and the corrected substitution formula (CI.3*) are valid and informative for the *direct head contribution*. They are not in themselves novel consistency-specific decision predictors. The submission explicitly avoids extending the direct-head sign to the total model decision without assumption A4.
3. **Final LayerNorm** margin factorization `m=a/σ+κ` and decision-null norm effects (NC) are valid in the stipulated class; the final-LN bias and readout assumptions matter. The isolated norm channel is not a new consistency mechanism.
4. **Ordinal nonremovable interaction** (RB) is a meaningful generic screen, but not specific to agreement or conflict. The submitted comparison to additive conjoint measurement/robust ordinal regression correctly limits novelty.
5. **Mixed objective diagnosis** (MIX): the explicit margin reward can generate improvement irrespective of the consistency term. No positive O/D-to-decision theorem follows.
6. The original O/D unrestricted directional and ranking claims remain unsupported. Nothing in this package changes that historical verdict.

## 3. Major omission: a direct implemented consistency-to-decision circuit was not investigated

The package's `OPEN_OBLIGATIONS.md` O-1 and `UNRESOLVED_SCIENTIFIC_QUESTIONS.md` item 1 explicitly say that **MLP operations nonlinear in an agreement variable and QK composition were unexamined**. This conflicts with the mandate's requirement to deeply investigate H1 (explicit reconciliation), H3 (evidence coordination), and the strongest causal-bridge candidates before recommending closure.

I constructed an explicit fully specified **one-layer, three-token, four-residual-dimensional pre-norm Transformer-like block**:

- Two different source positions carry `a e1` and `b e2` for independently manipulable evidence levels `a,b ∈ {-1,+1}`.
- `e1,e2,e3` are orthonormal and mean-zero in `R^4`.
- One causal attention head has `W_Q=W_K=0`, fixed uniform attention, and a value-output map selected so the read token receives `a e1+b e2`.
- A standard four-hidden-unit **GELU MLP** has preactivations `a+b`, `-(a+b)`, `a-b`, `-(a-b)` and writes

  `q(a,b) e3`, where `q(a,b)=GELU(a+b)+GELU(-a-b)-GELU(a-b)-GELU(-a+b)`.

- For binary inputs `q(a,b)` is positive exactly when `a=b` and negative exactly when `a≠b`. The unmodified residual before the MLP has no `e3` projection. Final positive-epsilon LayerNorm only rescales the signed `e3` readout.
- Set candidate unembedding columns `+e3` and `-e3`, with zero biases. The resulting **candidate margin is +3.214221608 for agreement and -3.214221608 for disagreement**, with fixed attention weights.

**Independent output:**

| a | b | Agreement | Candidate margin |
|--:|--:|--|--:|
| -1 | -1 | yes | +3.214221608 |
| -1 | +1 | no | -3.214221608 |
| +1 | -1 | no | -3.214221608 |
| +1 | +1 | yes | +3.214221608 |

This is a **real causal sensitivity to an independently defined binary agreement feature**, implemented by an ordinary nonlinear MLP in the same block class. Changing one evidence sign changes the decision margin, with sign depending on the other source. No attention-reallocation variation or final-norm-induced sign flip is involved. It is an existence example, not a trained-model discovery. Its explanation is an ordinary GELU conjunction/parity circuit, not a novel COR mechanism, and it does not validate the original O/D score. Nonetheless it decisively shows that the **broader claim that only softmax reallocation and final normalization can supply a relation-sensitive decision operation is false**. The submission's narrower §9 exclusions avoid a universal theorem claim, but the README/final synthesis and the recommendation to end Step 1 sound much broader than the actual coverage warrants.

Note that same-cut screening does not preclude this construction: the sum `a e1+b e2` retains both independent features, so the nonlinear downstream MLP can compute their agreement. Same-sum zero-net edits remain screened as the SS theorem intends.

## 4. Major identification overstatement: per-head consensus is not invariant

The main argument §3 states that per-head `w_j`, `v_j`, `M`, and centered writes are invariant under the architecture's function-preserving symmetries. That is false without additional restrictions.

**Constructive reparameterization:** Take two attention heads with **identical Q, K, V** (therefore the same attention pattern and the same pre-output values). If their output matrices are `W_O^1` and `W_O^2`, replace them with

`W_O^1 → W_O^1+E`, `W_O^2 → W_O^2−E`.

The sum of the two head writes and every downstream state is **identical for every input**, yet each head's write and readout-defined consensus generally changes. With one random independent instance:

- Head 1 scalar consensus changes from `4.518808203590434` to `9.876573941715256`.
- Head 2 scalar consensus changes from `−2.689070753325099` to `−8.04683649144992`.
- Maximum change in *total* residual write: `8.88e−16` (floating-point rounding).
- Individual centered head writes change nontrivially (max about `1.497`).

An analogous test using the submitted full-block implementation changed one head's consensus by approximately **14.963** while the logits remained identical to `1.78e−15`.

Thus **determinacy inside one frozen parameterization is not the same as functional/causal identifiability**. The package has not completely solved F5 for its replacement per-head explanation, though fixing a particular trained checkpoint makes the head quantities directly observable in that parameterization. See also Brunner et al., *On Identifiability in Transformers* (ICLR 2020), for related attention-identifiability issues: https://openreview.net/pdf?id=BJg1f6EFDB .

## 5. A formal gap in the written same-cut theorem

Theorem SS assumes "every downstream variable is a function of `S` **and of variables upstream of `S`**". As written, this permits a downstream bypass that reads one of the writer variables individually. For example:

`S=c1+c2`, `Y=S+c1`.

The edit `(c1,c2) → (c1+3,c2−3)` holds `S` fixed but changes `Y` from `3` to `6` in the explicit example `c1=2,c2=-1`.

This satisfies the literal stated description of a downstream function of `S` and an upstream variable, while contradicting SS.i–ii. **The intended theorem is valid for a pure additive cut with no bypass from the edited writers.** Amend the hypothesis: every downstream variable depends on the edited writers **only through S**, with any other upstream inputs held fixed. Standard Transformer heads into a pure residual cut meet this restriction; this is a formal statement fix, not a disproof of the architecture-specific fact.

## 6. Full-mandate coverage and ten-failure matrix

| Original fatality | Claude's actual result | Independent assessment |
|---|---|---|
| F1 causal bridge | Ordinary softmax direct-term and LN analysis; decision arrow absent | **Unresolved.** One designed MLP agreement mechanism exists, but no identified trained-model bridge |
| F2 lowering D may worsen decision | Correct historical disproof; restricted sign identities | **Unresolved** for candidate decision directional consistency |
| F3 D ranking vs logits | Final norm ranking theorem unrelated to D; old counterexamples preserved | **Unresolved** for consistency ranking |
| F4 distinctive prediction | RB held-out orders reduce to established conjoint measurement; revised head effect equals head patching | **No distinctive prediction** |
| F5 nonidentifiability | Canonical quantities replace fitted P and rho | **Partially addressed, overstated.** Per-head gauge example defeats invariance claim |
| F6 mixed objective | Correct `βJ^TA^{-1}J` decomposition | **Diagnostic resolved**; does not rescue bridge |
| F7 outcome tuning | Proposed preregistration/freeze ledger | **Protocol specified, not implemented/validated** |
| F25 same statistic, opposite decision | Correct fibers and scoped counterexamples | **Unresolved for a new decision predictor** |
| F29 comparative advantage | Correctly finds generic results in studied cases | **No distinctive bridge**; coverage incomplete |
| F30 unfrozen choices | Discloses ~12 design choices | **Protocol specified**, no surviving predictor to freeze |

The package's matrix marks F5/F6 as fully resolved under its substituted assumptions and F7/F30 conditionally resolved. These classifications are acceptable for *some methodology choices* but **do not constitute a full repair of the original ten failures**. In particular, F5's invariant-attribution assertion is false without extra conditions.

## 7. Breadth and review limitations

- Three candidate dossiers are present, but H2 has only brief fiber/SS reasoning; H1 and H3 reduce the architecture largely to softmax and final norm. The nonlinear MLP and QK-mediated cross-token relationships are not analyzed, despite being central to the mandate.
- The package's source register acknowledges reading the Phase 1 core files but using mostly the verdicts for earlier Step 1A–1E reports and consulting other files selectively. This was not a fresh line-by-line audit of all prior proof material.
- Four reports claim separate-context subagents; the package discloses roles A/G/H were done by the author. I cannot verify the session isolation from the ZIP alone. Do not describe these as external peer reviews.
- The conclusion `only generic normalization effects survive in investigated classes` is well supported, but the suggestion that this exhausts architecture-native relations is not.
- No pretrained-model runs: no evidence about prevalence, task-specific decision mediation, or necessity of agreement-sensitive circuits in trained models.

## 8. What this means scientifically

**Preserve:** All Phase 1 fatal counterexamples and the status `OBSTRUCTION_READY=NO`, `PHASE_2=NO GO`. The known identities and the decision-level failure of A4 are useful, sound negative findings under stated hypotheses.

**Do not preserve as universal:** No consistency-related computation is possible or relevant in a Transformer; every relation-sensitive effect is softmax/LN normalization; all meaningful identification problems are solved.

**Positive but narrowly scoped:** A hand-built GELU MLP in a legitimate Transformer block can compute an independently defined agreement relation and make it determine candidate preference. This establishes *architectural possibility*, not mechanistic novelty or real-model identification. It demonstrates why the omitted MLP/QK classes must not be excluded by a broad negative verdict.

**Bottom line:** `STEP1_NECESSARY_CONDITIONS_ONLY` is an honest **result label** for the examined scope. It is **not evidence that the original causal-bridge problem has been exhaustively investigated or that Step 1 should be closed**. The next scientific decision is a substantive scope/identification review, not another automatically chained mathematical exercise. Do not advance to Step 2 or Phase 2.

## 9. Reproducibility

- `pytest.log` and `manifest.log`: independent extraction and execution of the submitted code.
- `independent_standalone.py`, `independent_standalone.log`: fully independent NumPy causal circuit, head gauge and formal bypass counterexamples.
- `independent_checks.py`, `independent_checks.log`: adversarial full-block constructions using the submitted forward engine for cross-checks.

All generated under `/mnt/data/claude_foundation_audit/`.
