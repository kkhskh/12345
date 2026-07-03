# Current Project State T0

## Project

Causal Obstruction–Response Theory of Transformer Decisions.

## Central loop

local evidence
→ compatibility / incompatibility
→ obstruction
→ predicted intervention
→ patch / ablate / steer
→ measured Delta obstruction
→ measured Delta margin / stability
→ repair / flip / stabilization

## IOI status

IOI was Experiment 1 and is now a negative/weakening result.

Result:
- component patching worked
- attribution patching dominated
- obstruction-response collapsed to attribution
- IOI is attribution-positive / obstruction-negative

Interpretation:
IOI should be treated as a useful negative control, not the center of the paper.

## Factual conflict status

Experiment 2 is margin-matched factual conflict.

Goal:
Test whether obstruction detects internal incompatibility when final answer margin is approximately controlled.

Clean example:
The capital of France is

Conflict example:
The capital of France is Rome. However, the actual capital of France is

Candidate contrast:
Paris vs Rome

## Results so far

Experiment 2A: scalar / sequence residual-chain obstruction.

Original single-token scalar sweep:

GPT-2 small:
- matched facts: 60
- AUC(O_conflict > O_clean): 0.6219
- partial corr(O, conflict_label | margin): 0.1793

GPT-2 medium:
- matched facts: 89
- AUC: 0.6095
- partial corr: 0.1008

Pythia-410m:
- matched facts: 61
- AUC: 0.6036
- partial corr: -0.0236

Pythia-1b:
- matched facts: 63
- AUC: 0.6818
- partial corr: 0.2211

Interpretation:
Pythia-1b was the first promising smoke result, but it failed the data-size gate.

After multi-token factual answer scoring, Pythia-1b reached 126 matched facts. Scalar / sequence residual-chain obstruction then failed the margin-controlled gate:

- first-token scalar with multi-token margins:
  - matched facts: 126
  - AUC: 0.6369
  - partial corr: 0.1665
- sequence_mean germ:
  - matched facts: 126
  - AUC: 0.6536
  - partial corr: 0.1528
- sequence_sum germ:
  - matched facts: 126
  - AUC: 0.5437
  - partial corr: 0.0163

Decision:
Scalar / sequence residual obstruction is a no-go for component patching.

## Experiment 2B: low-rank factual obstruction

Experiment 2B tested vector-valued low-rank residual-chain germs on the same margin-matched factual conflict task.

Main non-leaky variant:
clean-PCA basis fit on clean train activations only.

Pythia-1b all-split results:

clean_pca r4:
- matched facts: 126
- AUC: 0.9458
- partial corr: 0.5834
- decision: PASS_STRONG

clean_pca r8:
- matched facts: 126
- AUC: 0.9758
- partial corr: 0.7044
- decision: PASS_STRONG

clean_pca r16:
- matched facts: 126
- AUC: 0.9740
- partial corr: 0.7910
- decision: PASS_STRONG

Answer-augmented clean-PCA also passed strongly:
- r4: AUC 0.9808, partial corr 0.5461
- r8: AUC 0.9760, partial corr 0.7010
- r16: AUC 0.9728, partial corr 0.7868

Delta-PCA diagnostic results were also very strong, but should not be treated as main evidence because the representation uses clean-conflict deltas.

Interpretation:
Low-rank vector-valued residual germs reveal a strong factual conflict obstruction signal that scalar / sequence scalar germs missed.

Decision:
Experiment 2B passes the representation-level gate. Component-level factual obstruction and patching may be considered next as a separate Experiment 2C, but patching has not yet been implemented.

## Immediate next empirical gate

Do not mix patching into Experiment 2B.

Next:
- preserve Experiment 2B artifacts and metrics
- write Experiment 2C as a separate component-level factual obstruction / patching step
- require learned low-rank obstruction to keep beating identity/random/shuffled controls
- use component patching only to test causal response, not to establish the representation result

Only then move to component-level factual obstruction and patching.

## T0 math-map purpose

The T0 artifact should keep the paper broad:
- model-family atlas
- top-100 model atlas seed
- external vs internal model geometry
- 3D model-space / basin design
- computation complex K
- stalks F
- germs s
- transports rho
- obstruction O
- response R

The T0 artifact should not narrow the paper to IOI or factual conflict.