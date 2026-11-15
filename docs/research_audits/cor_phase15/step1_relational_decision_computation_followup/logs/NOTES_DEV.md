# XIII-B development notes (running log of findings; development models = XIII seed-20 cohort + earlier pools)

## A. Pre-assembled vs mixture (dev: uP160, uP163 pre; uP161, uP165 mix; uP164 borderline)
- locate (relational pairs, edit at q' != q; logs/locate_dev_rel.json): readout-token state entering layer 2 carries 0.44-0.49 of the relational
  change in pre-assembled vs 0.27-0.29 in mixture; L1 attention at RO carries it in pre (0.44-0.49) while L1 attention at QS carries it in mixture (0.61-0.71).
- locate (likelihood pairs, a value at q changed; logs/locate_dev_lik.json): candidate evidence reaches the readout token at LAYER 0 in pre-assembled
  (L0ATT@RO 0.84-0.85) but NOT in mixture models (resid2@RO 0.02-0.03; it is read at layer 2 by mixture heads, L2ATT@RO 0.99).  uP164 0.53 (intermediate).
- path_chain (logs/path_chain_dev.json): relational effect originates in layer 0 at the edited group's tokens (transfer 0.86-0.95).  Through
  L1ATT@RO 55-56% (pre) vs 28-38% (mix); through L1ATT@QS 3-4% (pre) vs 65-70% (mix); L2ATT+MLP@RO 91-100% (all); L2MLP@RO 76-78% (pre) vs 27-34% (mix).
  So the 'pre-assembled' class is NOT complete before layer 2: layer 2 (mainly MLP) still performs the combination at the readout token.
- attn_profile: pre-assembled layer-1 readout heads attend broadly (0.7-0.8 to other-question source tokens) with per-group weights uncorrelated with
  responsibilities (|corr| <= 0.1); layer-0 readout heads read the q values (QS mass 0.8-0.95).  Mixture: layer-2 heads 2/3 with QS mass 0.96, corr(B) 0.72/0.98.
- fit_family (logs/fit_family_dev.json): ALL models are fit by the posterior-MIXTURE family (test R2 0.983-0.995) far better than product-of-experts
  pooling (0.82-0.84, no better than POE fits Bayes itself, 0.83).  Pre-assembled fit slightly worse (0.983-0.986) and with softer exclusion (kappa 1.3 vs 1.46-1.48; Bayes-fit kappa 1.62).
- algorithm_discrimination (logs/algdisc_eval_conf13.json): every model closest to B (E-based) on D1/D2; distance to B 0.24-0.28 pre vs 0.10-0.11 mix;
  to A1 >= 1.0.  Error correlation: pre-pre 0.64-0.72, pre-mix 0.06-0.35, mix-mix 0.18-0.30; incomplete uP162 (mixture class) correlates with pre 0.4-0.67.

## B. Controls
- l2_components (relational families, values at q identical): uP161 Shapley R 0.68, V 0.03, O 0.17, M 0.12; uP165 R 0.81, M 0.17; uP164 R 0.22, O 0.39, M 0.29.
  => with purely relational edits the mixture-row pathway carries 68-81% (XIII's 25-33% was for a design that also changed values at q).
  Decoys (single-component proj): MIX 0.84/0.99; magnitude-matched TOP/MAG 0.20/0.00; LOW 0.00; PERM/MEAN produce large generic changes with low E-fidelity (R2 <= 0.33).
  Restoration: removing route Q (L1 attention at q tokens from base) removes 67-69% of the effect (uP161/165; only 18% in uP164); restoring with the
  INTACT mixture rows gives proj 0.92/0.98, fid_E 0.97/0.98; decoy restorations stay at the removed level (0.31-0.44) or produce wrong-direction changes;
  likelihood-edit effects preserved (slope 0.96/1.01).  E vs A1 (P0 family: perfectness flips, e stays 0): model TE 0.15-0.16 matches E (0.14) not A1 (1.07);
  mixture-row transplant fid_E 0.81-0.83 vs fid_A1 0.02-0.03; restoration fid_E 0.82-0.86 vs 0.03.
- das_bal (balanced E/A1): uP161: E-trained fid_E 0.75 / fid_A1 0.02 (source-free E-subspace patch fid_E 0.01); A1-trained fid_A1 0.72 BUT
  source-free patches (any subspace, incl. RANDE-trained) reach fid_A1 0.86-0.93 => A1 'identification' is reproduced without source information:
  not evidence for an A1 variable.  E specificity survives the source-free null.  E/A1 subspace principal cosines 0.77,0.75,0.45,0.32,0.08,0.01.
- l1_graded uP161: mixture q-mass unchanged (0.96-0.99) under every ablation; alignment with B and the behavioural residual fall together
  (ATT+MLP mean a=1: residual 0.00; resample: -0.06).

## C. Two-colluder
- censored_bayes: beta = 1: LOCALQ (self-agreement bits + joint values at q) EXACTLY sufficient (KL 0); beta = 0.6: LOCALQ 0.758 vs FULL 0.794
  (6.1% decisions differ); PERGROUP 0.50.  Proof sketch for beta = 1 in JOINT doc.
- adapt2 beta 0.6 uP161: readout 0.46, attn 0.70, mlp 0.61, full 0.72, fresh 0.72 -- all with J-pair slope ~0 (they do NOT use cross-question identity);
  early01 (layers 0-1 trained, layer 2 + readout FROZEN): 0.789 (Bayes 0.792), KL 0.016, J-pair slope 0.98 R2 0.89; early01ro 0.748, slope 0.32.
- adapt2 beta 0.6 uP160 (pre-assembled): attn 0.771 with J-pair slope 0.49 (uses J partially!), mlp 0.728 slope 0.03.
- decode_J uP161: n4 (cross-group identity count) decodable from frozen resid1/resid2 source tokens at R2 0.86-0.96 (0.81-0.94 beyond per-group stats);
  raw one-hot 0.0; readout-token resid2 0.54-0.70.
