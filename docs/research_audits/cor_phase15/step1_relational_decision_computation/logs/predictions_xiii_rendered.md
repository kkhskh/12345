Mixture-class rule (frozen): a parent is mixture-class iff some layer-2 head has readout-token attention mass on the target question's six source tokens >= 0.8 AND direct-logit variance share >= 0.2 (l2_scan).

Pass rule: an arm passes if its test holds in >= ceil(0.75 n) units (n >= min_units, default 2; otherwise INCONCLUSIVE). A prediction is CONFIRMED iff all its arms pass. An arm with a null is DIAGNOSTIC only if the null fails on every unit; a passing arm whose null passes anywhere is NON-DIAGNOSTIC.

### P1: Layer-2 attention-mixture (ATT-MIX): inserting a same-question donor's natural mixture-head attention row changes the output as the donor's Bayes responsibilities predict, and better than alternative weightings
Units: complete parents, mixture-class only

Class II natural-activation intervention (donor rows are real activations of another natural episode). Development: R2(Bayes) 0.69-0.98, slope 0.95-0.97; shuffled R2 <= 0.002; on donors whose E and A1 eligibilities differ, R2(Bayes) 0.97-0.99 vs R2(A1) 0.17-0.19.

| arm | statement | test (scorer expression) | null (matched control substituted) |
|---|---|---|---|
| a | fidelity to donor Bayes responsibilities: R2 >= 0.6 and slope in [0.4, 1.25] | `{l2_mix_causal:donor.bayes_resp.R2} >= 0.6 and 0.4 <= {l2_mix_causal:donor.bayes_resp.slope} <= 1.25` | `{l2_mix_causal:donor.shuffled_pred.R2} >= 0.6 and 0.4 <= {l2_mix_causal:donor.shuffled_pred.slope} <= 1.25` |
| b | Bayes responsibilities beat the separable-discount alternative A3 by >= 0.05 R2 | `{l2_mix_causal:donor.bayes_resp.R2} - {l2_mix_causal:donor.A3_resp.R2} >= 0.05` | `{l2_mix_causal:donor.A0_resp.R2} - {l2_mix_causal:donor.A3_resp.R2} >= 0.05` |
| c | on donors with E != A1 eligibility, Bayes(E) responsibilities beat perfect-only (A1) responsibilities by >= 0.3 R2 | `{l2_mix_causal:donor_EneA1.bayes_resp.R2} - {l2_mix_causal:donor_EneA1.A1_resp.R2} >= 0.3` | `{l2_mix_causal:donor_EneA1.A1_resp.R2} - {l2_mix_causal:donor_EneA1.bayes_resp.R2} >= 0.3` |
| d | validity: re-inserting the model's own rows changes nothing (noop <= 1e-4) | `{l2_mix_causal:noop_max_abs_change} <= 1e-4` | none (validity/descriptive arm) |

### P2: Natural mediation through mixture-head attention: in the TRUE factorial the mixture-head rows mediate far more of the eligibility effect than in the CTRL factorial and than size-matched non-mixture heads
Units: complete parents, mixture-class only

Class I input edit (TRUE: eligibility-changing edit; CTRL: matched edit that leaves eligibility unchanged) crossed with class II activation transplant of mixture-head rows. Development: DiD 0.32-0.88 (CI lower bound >= 0.27 in all four), decoy-head/mixture NIE 0.01-0.33, ATT-MIX fidelity of the transplant 0.52-0.98.

| arm | statement | test (scorer expression) | null (matched control substituted) |
|---|---|---|---|
| a | difference-in-differences of |NIE| (TRUE - CTRL): 95% CI lower bound >= 0.2 | `{l2_mediation:DiD_mag_NIE_true_minus_ctrl.lo} >= 0.2` | `{l2_mediation:TRUE.mean_mag_NIE_decoy} - {l2_mediation:CTRL.mean_mag_NIE_decoy} >= 0.2` |
| b | mixture heads' TRUE |NIE| >= 0.2 and decoy (non-mixture) heads mediate <= 0.4 of it | `{l2_mediation:TRUE.mean_mag_NIE} >= 0.2 and {l2_mediation:TRUE.mean_mag_NIE_decoy} <= 0.4 * {l2_mediation:TRUE.mean_mag_NIE}` | none (validity/descriptive arm) |
| c | the transplanted change follows ATT-MIX predictions: R2 >= 0.45 | `{l2_mediation:TRUE.fidelity_ATTMIX.R2} >= 0.45` | none (validity/descriptive arm) |

### P3: The mixture-head score contrast that excludes ineligible groups is carried by the E coordinates at layer 2, not by a random subspace or the complement, but matched-budget decoy subspaces (A1, PERM) are NOT specific under mean patching
Units: complete parents, mixture-class only

Class III synthetic patch (mean patch of a subspace). Development (xP81, yP130, vP155, vP156): E mean patch removes 1.37-1.47 of the S=1 contrast, random 6-dim <= 0.013, complement makes it more negative; matched decoys A1 and PERM also remove 0.88-1.43 (xP81, vP155). Arm c is therefore a predicted NON-specificity (a diagnostic limitation of mean patching).

| arm | statement | test (scorer expression) | null (matched control substituted) |
|---|---|---|---|
| a | E-subspace mean patch removes >= 0.8 of the S=1 score contrast | `{score_contrast:E_mean.frac_S1_removed} >= 0.8` | `{score_contrast:random6_mean.frac_S1_removed} >= 0.8` |
| b | complement (442-dim) mean patch removes <= 0.2 of it | `{score_contrast:complement_mean.frac_S1_removed} <= 0.2` | none (validity/descriptive arm) |
| c | matched-budget decoys are not specific: both A1 and PERM subspace mean patches remove >= 0.5 | `min({score_contrast:decoy_A1_mean.frac_S1_removed}, {score_contrast:decoy_PERM_mean.frac_S1_removed}) >= 0.5` | none (validity/descriptive arm) |

### P4: Layer 2 does not reconstruct eligibility: after layer-1 mean ablation the mixture-head attention loses its alignment with Bayes responsibilities, although eligibility remains linearly decodable outside the E coordinates
Units: complete parents, mixture-class only

Class III (mean ablation) plus decoding. Development (xP81, yP130): intact corr(mass, B) 0.95-0.98, after ablation ~0; e decodable from E and from the complement at 1.0; Bayes decision decodable from the complement 0.81-0.92 but not from E (0.41).

| arm | statement | test (scorer expression) | null (matched control substituted) |
|---|---|---|---|
| a | intact corr(mixture mass, Bayes responsibility) >= 0.8 and after layer-1 mean ablation <= 0.3 | `{l1abl_mix:intact.B} >= 0.8 and {l1abl_mix:layer1_mean_ablated.B} <= 0.3` | `{l1abl_mix:intact.noelig} >= 0.8 and {l1abl_mix:layer1_mean_ablated.noelig} <= 0.3` |
| b | redundancy: e_g0 decodable from the complement of E at >= 0.95 | `{l1abl_mix:decode.e_g0.complement} >= 0.95` | `{l1abl_mix:decode.e_g0.majority} >= 0.95` |
| c | the Bayes decision is better decoded from the complement than from E by >= 0.2 | `{l1abl_mix:decode.bayes_decision.complement} - {l1abl_mix:decode.bayes_decision.E} >= 0.2` | `{l1abl_mix:decode.bayes_decision.majority} - {l1abl_mix:decode.bayes_decision.E} >= 0.2` |

### P5: Two-colluder repair (C=2) is local to layer 2: with layers 0-1 frozen, adapting layer-2 attention alone or layer-2 MLP alone recovers the exact C=2 posterior including the J_H-dependent S=3 slice, whereas readout-only adaptation does not recover it overall -- although the frozen C=1 resid2 already supports most S=3 decisions linearly
Units: complete parents

Selective adaptation on frozen resid2 (adapt_sel.py, 1000 steps, C=2, beta=1). Development (xP81, vP155): native acc 0.33 (Bayes 0.79); attn-only 0.80/S3 0.97; mlp-only 0.79-0.80/S3 0.92-0.97; full 0.80-0.80/S3 0.97, KL 0.003-0.005; readout-only 0.60-0.64, KL 0.45-0.49, but S3 0.79-0.89 (chance 0.34, native 0.33).

| arm | statement | test (scorer expression) | null (matched control substituted) |
|---|---|---|---|
| a | the C=1-trained network fails on C=2: native acc <= Bayes acc - 0.2 | `{adapt:native.acc} <= {adapt:bayes_acc} - 0.2` | `{adapt:full.acc} <= {adapt:bayes_acc} - 0.2` |
| b | layer-2 attention-only AND layer-2 MLP-only adaptation each reach acc >= Bayes - 0.03 and S=3 acc >= Bayes(S=3) - 0.1 | `{adapt:attn.acc} >= {adapt:bayes_acc} - 0.03 and {adapt:mlp.acc} >= {adapt:bayes_acc} - 0.03 and {adapt:attn.acc_S3} >= {adapt:bayes_acc_S3} - 0.1 and {adapt:mlp.acc_S3} >= {adapt:bayes_acc_S3} - 0.1` | `{adapt:readout.acc} >= {adapt:bayes_acc} - 0.03 and {adapt:readout.acc_S3} >= {adapt:bayes_acc_S3} - 0.1` |
| c | readout-only adaptation is insufficient overall: KL >= 0.2 and acc <= Bayes - 0.1 | `{adapt:readout.KL_true} >= 0.2 and {adapt:readout.acc} <= {adapt:bayes_acc} - 0.1` | `{adapt:full.KL_true} >= 0.2 and {adapt:full.acc} <= {adapt:bayes_acc} - 0.1` |
| d | the frozen C=1 resid2 already supports the J_H-dependent slice linearly: readout-only S=3 acc >= 0.7 | `{adapt:readout.acc_S3} >= 0.7` | `{adapt:native.acc_S3} >= 0.7` |
| e | validity: parameters outside each condition are bit-identical after training | `{adapt:readout.frozen_params_unchanged} and {adapt:attn.frozen_params_unchanged} and {adapt:mlp.frozen_params_unchanged}` | none (validity/descriptive arm) |

### P6: EXPLORATORY heterogeneity: parents whose pretraining stayed incomplete after the gate are mixture-class
Units: incomplete parents; status: exploratory (not counted among the primary predictions)

Development: all 5 incomplete parents were mixture-class; complete parents split 6 mixture / 5 pre-assembled. Readiness gating is expected to leave few or no incomplete parents, in which case this is inconclusive.

| arm | statement | test (scorer expression) | null (matched control substituted) |
|---|---|---|---|
| a | mixture-class rule holds | `{cls:mixture}` | none (validity/descriptive arm) |

### P7: In mixture-class parents, the dominant mixture head's per-group attention mass correlates with Bayes responsibilities (>= 0.8) more than with the A3 alternative (by >= 0.08) and with eligibility-count M (by >= 0.2)
Units: complete parents, mixture-class only

Class II correlational (natural episodes). Development: dominant head corr(B) 0.70-0.99 (>= 0.8 in 13/15 mixture-class models), B - A3 0.11-0.14, B - M 0.3-0.4.

| arm | statement | test (scorer expression) | null (matched control substituted) |
|---|---|---|---|
| a | corr(mass, B) >= 0.8 | `{mix:corr_B} >= 0.8` | `{mix:corr_M} >= 0.8` |
| b | corr B - corr A3 >= 0.08 and corr B - corr M >= 0.2 | `{mix:corr_B} - {mix:corr_A3} >= 0.08 and {mix:corr_B} - {mix:corr_M} >= 0.2` | `{mix:corr_A3} - {mix:corr_M} >= 0.3 and {mix:corr_W} - {mix:corr_M} >= 0.2` |

### P8: Matched-budget subspace interchange (DAS) is NON-identifying between E and A1 but rejects the group-permuted rule: on E != A1 pairs each trained subspace follows its own training rule; on all discriminating pairs E- and A1-trained subspaces follow E far better than PERM
Units: complete parents

Class III synthetic interchange of matched subspaces (das_decoy.py: identical procedure and budget). Development (xP81, vP155): E≠A1 stratum: E-subspace R2(E) 0.35-0.45 vs R2(A1) 0.12-0.14; A1-subspace R2(A1) 0.29-0.44 vs R2(E) 0.06-0.15; all pairs: E-subspace R2(E) 0.93-0.94 vs R2(PERM) 0.00-0.01; random <= 0.04.

| arm | statement | test (scorer expression) | null (matched control substituted) |
|---|---|---|---|
| a | E != A1 stratum: E-subspace R2(E) - R2(A1) >= 0.1 | `{decoy_disc:E.EvA1.E.R2} - {decoy_disc:E.EvA1.A1.R2} >= 0.1` | `{decoy_disc:random.EvA1.E.R2} - {decoy_disc:random.EvA1.A1.R2} >= 0.1` |
| b | E != A1 stratum: A1-subspace R2(A1) - R2(E) >= 0.1 (non-identifiability) | `{decoy_disc:A1.EvA1.A1.R2} - {decoy_disc:A1.EvA1.E.R2} >= 0.1` | `{decoy_disc:random.EvA1.A1.R2} - {decoy_disc:random.EvA1.E.R2} >= 0.1` |
| c | all discriminating pairs: E-subspace R2(E) >= 0.8 and R2(E) - R2(PERM) >= 0.5 | `{decoy_disc:E.E.R2} >= 0.8 and {decoy_disc:E.E.R2} - {decoy_disc:E.PERM.R2} >= 0.5` | `{decoy_disc:random.E.R2} >= 0.8 and {decoy_disc:random.E.R2} - {decoy_disc:random.PERM.R2} >= 0.5` |
