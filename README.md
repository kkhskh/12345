# Causal Obstruction Response

Narrow first empirical block for GPT-2 small + TransformerLens + IOI
clean/corrupt residual-chain obstruction-response repair.

## First Pass

1. Generate IOI examples with `notebooks/00_generate_ioi_data.ipynb`.
2. Fit scalar residual-chain transports on clean train prompts.
3. Run the coherent-swapped control.
4. Run the conflict/malformed IOI obstruction diagnostic.

## Experiment 0: Coherent-Swapped Control

The original clean vs swapped comparison is a control, not a real corruption
test:

```text
Experiment 0: coherent-clean vs coherent-swapped IOI control
```

Conclusion from the first run:

```text
Scalar residual-chain obstruction does not distinguish two coherent but
opposite IOI computations.
```

This is scientifically useful because obstruction should detect incompatibility
or degraded evidence, not merely a different coherent answer.

## Conflict Go/No-Go

The active go/no-go threshold is:

```text
AUC(conflict_or_malformed > coherent) > 0.65
```

where `coherent` includes `clean_coherent` and `swapped_coherent`, and
`conflict_or_malformed` includes prompts with conflicting or degraded local IOI
evidence.

## Negative Control: Scalar Germs

Scalar logit-lens obstruction does not detect IOI conflict or malformed prompts,
even across IOI token positions. This suggests that obstruction is not a trivial
function of the final answer direction.

The next representation is low-rank residual vector germs learned from
clean-vs-swapped residual differences at IOI-relevant token positions.

Current failure ladder:

```text
scalar final-token failed
multi-token scalar failed
naive low-rank passed but was confounded
residualized low-rank failed
whitened low-rank was confounded even harder
```

This should be kept as a negative-control sequence. The next diagnostic is
whitened low-rank edge residual obstruction, which tests whether examples leave
the clean residual-transition manifold after covariance and norm effects are
removed.

## Negative Section: Residual Chain

Residual-chain obstruction is insufficient on IOI. Scalar, low-rank,
residualized, and whitened variants either fail or collapse into
distribution-shift detection. Therefore IOI obstruction must be tested at
component-level sites rather than residual-chain summaries.

## Repair Contrast

The candidate contrast always uses clean-target repair:

```text
repair_answer_a = clean correct
repair_answer_b = clean competitor
margin = z_a - z_b
```

For the corrupt prompt, this deliberately evaluates the clean-target margin, so
successful clean residual patching into the corrupt run should increase that
margin and ideally flip it positive.
