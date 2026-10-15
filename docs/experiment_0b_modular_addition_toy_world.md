# Experiment 0B: Modular Addition Toy World

## Why this experiment exists

The updated theory document makes a controlled toy world a requirement of the minimum viable
paper, not an optional extra. The reasoning is the professor's objection: *you are steering
before defining what is being steered.* A steering theory only becomes scientific once it
defines, detects, and causally validates the internal representation it claims to steer.

Modular addition is the cheapest setting where the latent state is known exactly. A small
Transformer trained on `(a + b) mod p` is known to generalize by constructing cyclic
(Fourier-like) representations, so the ground-truth latent state can be computed from the task
simulator rather than guessed from the model. That makes every stage checkable:

    known latent state -> internal representation -> compatibility / obstruction
        -> causal intervention -> behavioural repair

## Pipeline

| Script | Stage | Output |
| --- | --- | --- |
| `run_07a_train_modular_addition.py` | train with checkpoints | `training_curve.csv`, `checkpoints/` |
| `run_07b_modular_latent_representation.py` | representation detection | `latent_probe_metrics.csv` |
| `run_07c_modular_obstruction_over_checkpoints.py` | monitoring | `obstruction_over_checkpoints.csv` |
| `run_07d_modular_causal_intervention.py` | causal test | `intervention_ablation.csv`, `intervention_repair.csv` |

All artifacts land under `artifacts/toy_modular/`. Shared machinery lives in `src/toy_modular.py`.

## Definitions

**Task and contrast.** Input tokens are `[a, b, '=']`, the label is `s = (a + b) mod p`. The
candidate contrast is `c = (correct sum, hardest incorrect sum)`, and the margin is
`m_c = z_correct - z_hardest_wrong` at the `'='` position.

**Known latent state.** For key frequencies `k in K`,

    y(x) = [ cos(2 pi k s / p), sin(2 pi k s / p) ]_{k in K}

computed from the simulator. Key frequencies are read off the Fourier power spectrum of `W_E`.

**Abstraction strength.** `A_cyc(t)` is the fraction of non-constant Fourier power of the
embedding matrix concentrated in the top-`|K|` frequencies. High values mean the model encodes
inputs through a small set of cyclic frequencies, which is the generalizing algorithm.

**Latent-state germs.** `s_y(v; x) = P^y_v phi_v(x)`, with `P^y_v` fit by ridge regression from
site activations to `y(x)` on **training examples only**, then scored on held-out examples.

**Sites and edges.** An ordered chain at the `'='` position: operand embeddings, `resid_pre`,
each attention head's `z`, `resid_mid`, `mlp_post`, `resid_post`. Consecutive sites form the
edge set.

**Transports and obstruction.** `rho^y_{u->v}` fit by ridge on training examples, then

    O_y(x) = sum_e w_e || rho^y_{u->v} s_y(u; x) - s_y(v; x) ||^2

reported normalized by total germ norm.

## Controls

Representation detection is scored against a shuffled-target probe and a matched-dimension
random-subspace probe, so a high probe score cannot come from probe capacity alone. Monitoring
compares learned transports against identity and random transports. The causal stage compares
the cyclic subspace against matched-dimension random, top-PCA, and weight-spectral subspaces,
and compares the latent-target edit against matched-norm edits in each control subspace.

Two design rules matter and were both bugs before they were fixed:

- **Every data-dependent subspace is fit on training activations only.** Fitting the cyclic
  subspace on the held-out activations it is then evaluated on both leaks and, worse, produces a
  subspace inconsistent with the germ driving the obstruction, so ablating it does not even
  destroy the germ it targets.
- **Edits are capped relative to the activation norm.** The minimum-norm edit that moves a germ
  to a target is unbounded where the germ map is rank-deficient. Uncapped, this produced edits
  with collateral KL above 170,000, far outside any regime where the local response
  approximation of Proposition 6 holds. The cap doubles as the intervention-radius ablation.

## Comparisons are within-site

A subspace is compared only against matched-dimension controls **at the same site**, and an edit
only against matched-norm edits at the same site and radius. Comparing the best site for one
subspace against the best site for another conflates subspace choice with site choice: top-PCA
directions at `resid_post` destroy almost all accuracy simply because that site feeds the
unembedding, which says nothing about whether PCA found a meaningful structure.

## Validation run

A small validation run (`p = 31`, `d_model = 64`, `d_mlp = 256`, train fraction 0.5, 30k steps)
was used to exercise the pipeline. It groks: train accuracy 1.000, test accuracy 0.948,
`A_cyc = 0.907`. These numbers are a code check, not the MVP result; the MVP run is `p = 113`
with `d_model = 128`.

**Representation detection passes cleanly.** At the final checkpoint, held-out latent decoding by
site:

| Site | R^2 (held out) | Decode acc | Shuffled control | Random subspace | Chance |
| --- | --- | --- | --- | --- | --- |
| `L0_mlp_post` | +0.572 | 0.927 | 0.013 | 0.509 | 0.032 |
| `L0_resid_post` | +0.540 | 0.904 | 0.029 | 0.873 | 0.032 |
| `L0_resid_mid` | +0.142 | 0.166 | 0.037 | 0.137 | 0.032 |
| `embed_pos0` | -0.060 | 0.004 | 0.025 | 0.019 | 0.032 |

The MLP output carries the cyclic latent state of the sum; the operand embeddings do not, which
is correct, since the sum has not been computed there yet. Note that the random-subspace control
reaches 0.873 at `resid_post`: a random projection of a strongly structured space retains most of
the decodable signal, so probe accuracy alone is weak evidence.

**Monitoring supports the central hypothesis.** From step 172 onward, abstraction strength rises,
obstruction falls, and test accuracy rises together:

| Step | `A_cyc` | Test acc | `O_y` learned | `O_y` identity | `O_y` random |
| --- | --- | --- | --- | --- | --- |
| 172 | 0.520 | 0.333 | 0.727 | 0.983 | 1.628 |
| 3815 | 0.782 | 0.584 | 0.668 | 0.898 | 1.634 |
| 9000 | 0.870 | 0.948 | 0.581 | 0.818 | 1.494 |
| 29999 | 0.907 | 0.948 | 0.509 | 0.677 | 1.511 |

Learned transports produce much lower obstruction than identity or random transports throughout,
so the transport ablation is clean. The AUC for obstruction flagging the model's own errors sits
around 0.67 to 0.75.

One caveat: at the earliest steps (0 to 60) obstruction is *low* while accuracy is at chance,
because the germs are fitting noise in an unstructured representation. Obstruction is only
interpretable once there is a representation for local evidence to be coherent about, so the
early-training values should not be read as coherence.

**The causal stage fails, and this is the important result.** At both the grokked and the
partially grokked checkpoint, the decision is `NO_GO`: the cyclic subspace wins the necessity test
at only 1 of 10 sites, and the latent-target edit never beats matched-norm controls (0 of 30
site-by-radius cells at the partially grokked checkpoint, where 200 incorrect examples make the
comparison well powered).

The sharpest single measurement: at `L0_mlp_post`, projecting out the germ subspace drives the
germ's linear part to exactly zero and collapses held-out latent decoding from 0.927 to 0.037,
which is chance. Model accuracy falls only from 0.948 to 0.936, with collateral KL of 0.016. The
representation the probe found can be completely destroyed while behaviour is essentially
untouched.

This is precisely Proposition 7 of the theory document, superposition non-uniqueness, showing up
as a measurement: a direction that decodes a concept is not thereby the model's causal direction.
The model's readout uses different, redundant directions. The one site where the cyclic subspace
does win is `L0_resid_pre` (accuracy drop 0.441 at KL 2.00, versus 0.025 for the best control).

## Interpretation

Representation detection and monitoring pass. The causal gate fails. That combination is the
framework's own falsification machinery working as designed, and it is a finding rather than a
setback: it is direct evidence for the ablation the updated plan calls *probe-only representation
versus causally validated representation*, and it means the paper cannot use probe decodability as
evidence of a steerable internal object.

The honest claim available from this experiment is narrow. In a setting where the latent state is
known exactly, ridge-probe germs decode it out of sample, learned transports make the germs
markedly more compatible than identity or random transports, and latent obstruction falls as the
generalizing abstraction forms. But the probe-identified subspace is not, at most sites, the
causally load-bearing one.

## Go / no-go criteria

- **Representation detection.** Held-out decode accuracy must clearly beat the shuffled-target
  and matched-dimension random-subspace controls at a minimum of one site.
- **Monitoring.** `A_cyc(t)` rising must accompany `O_y(t)` falling and test accuracy rising, with
  learned transports beating identity and random transports.
- **Necessity.** The cyclic subspace must cause a larger accuracy drop than every
  matched-dimension control at a majority of sites.
- **Repair.** The latent-target edit must beat every matched-norm control edit at a majority of
  site-by-radius cells, with at least 50 incorrect examples available.

`PASS_STRONG` requires both causal criteria; `PASS_WEAK` requires one; otherwise `NO_GO`. Repair
should be evaluated at a partially grokked checkpoint, since a fully grokked model makes too few
errors to measure repair against.

## Running it

```bash
# MVP scale; the training step wants a GPU
python run_07a_train_modular_addition.py --p 113 --steps 25000 --n-checkpoints 40
python run_07b_modular_latent_representation.py --max-checkpoints 12
python run_07c_modular_obstruction_over_checkpoints.py
# choose a partially grokked checkpoint from training_curve.csv for the repair test
python run_07d_modular_causal_intervention.py --checkpoint artifacts/toy_modular/checkpoints/step_0004000.pt
```

A fast end-to-end check (`p = 31`, `d_model = 64`, `d_mlp = 256`, 30k steps) runs on CPU in a
couple of minutes and groks, which is enough to exercise every code path.
