# 08R2-FIBER Specificity and Natural-Anchor Audit

Read-only analysis of saved FIBER artifacts. No model rerun. Original files were not overwritten.

Official labels unchanged:

- `08R2_VALIDATION_COMPLETE — NEGATIVE`
- Original 0A: **C / PARTIAL**
- `SECTION_5_3 = PARTIAL`
- `OBSTRUCTION_READY = NO`

## 1. Raw integrity

- Files present: {'08r2_fiber_dependency_manifest.json': {'exists': 1, 'bytes': 1732}, '08r2_fiber_probe_contrast_parity.csv': {'exists': 1, 'bytes': 555}, '08r2_fiber_svd_report.json': {'exists': 1, 'bytes': 3697}, '08r2_fiber_matched_pairs.csv': {'exists': 1, 'bytes': 17492}, '08r2_fiber_factorial_interventions.csv': {'exists': 1, 'bytes': 58070}, '08r2_fiber_full_logits.npz': {'exists': 1, 'bytes': 71262}, '08r2_fiber_probe_probabilities.npz': {'exists': 1, 'bytes': 214056}, '08r2_fiber_natural_anchor.csv': {'exists': 1, 'bytes': 420240}, '08r2_fiber_controls.csv': {'exists': 1, 'bytes': 69462}, '08r2_fiber_bootstrap.csv': {'exists': 1, 'bytes': 1165}, '08r2_fiber_completion_status.json': {'exists': 1, 'bytes': 652}, '08r2_fiber_scientific_report.md': {'exists': 1, 'bytes': 4776}}
- Factorial pairs with H00/H10/H01/H11: 32
- Missing factorial cells: 0
- Logit vectors: {'present': 1, 'n_vectors': 128, 'n_ok': 128, 'n_bad': 0, 'max_abs_sum_error': 4.440892098500626e-16}
- Recomputed JS max |saved-raw| 00-01: 4.380363738398785e-09
- Recomputed JS max |saved-raw| 10-11: 5.6436603723675205e-09
- Y vs became_legal-became_illegal max err: 3.3306690738754696e-16
- Identity max |Y11-Y00-(E_R+E_N+I)|: 2.220446049250313e-16
- Probe-array invariance max |H01-H00|: 7.716050021144838e-15
- Probe-array invariance max |H10-H11|: 1.4094628242311558e-17
- Token-set Y from 61-way logits: `NOT_IDENTIFIABLE_FROM_SAVED_DATA legal square sets were not stored in the fiber CSVs`

## 2. Natural-anchor sensitivity (not pooled across ε)

### ε = 1.0

- Donor-null rows/pairs: 64/32; targets=6
- Mean |ΔY|=0.0003162771864545119 median=0.00027427151917725 CI=n=64 clusters=32 point=0.0003 CI95=[0.0002, 0.0004]
- Mean JS=9.631876254142608e-07 CI=n=64 clusters=32 point=0.0000 CI95=[0.0000, 0.0000]
- Mean achieved L2=1.0
- Invariance failures=0 max err=2.220446049250313e-16
- Centered FD (Y+ − Y−)/(2ε): {'mean_fd': -0.00016539518876300754, 'median_fd': -5.723917126942979e-06, 'fd_ci': {'n': 32, 'n_clusters': 32, 'point': -0.00016539518876300754, 'boot_mean': -0.00016635539623313417, 'ci95_low': -0.0003015980636006516, 'ci95_high': -3.4190103506879543e-05}}
- A: {'slice': 'A', 'n_pairs': 24, 'n_recipient_games': 24, 'n_target_games': 2, 'n_rows': 48, 'mean_abs_dY': 0.00025447203689440333, 'median_abs_dY': 0.00021126831944679998, 'mean_js': 9.378495289533302e-07, 'median_js': 7.965732610585638e-07, 'mean_l2': 1.0, 'invariance_fail': 0, 'max_invariance_err': 2.927345865710862e-18, 'abs_dY_ci': {'n': 48, 'n_clusters': 24, 'point': 0.00025447203689440333, 'boot_mean': 0.00025432097234355256, 'ci95_low': 0.00016045583265914654, 'ci95_high': 0.0003506097963282601}, 'js_ci': {'n': 48, 'n_clusters': 24, 'point': 9.378495289533302e-07, 'boot_mean': 9.359724815418408e-07, 'ci95_low': 7.561158393584933e-07, 'ci95_high': 1.1375062783045024e-06}}
- B: {'slice': 'B', 'n_pairs': 8, 'n_recipient_games': 8, 'n_target_games': 6, 'n_rows': 16, 'mean_abs_dY': 0.0005016926351348375, 'median_abs_dY': 0.00037782435796550004, 'mean_js': 1.0392019147970522e-06, 'median_js': 9.347803003747624e-07, 'mean_l2': 1.0, 'invariance_fail': 0, 'max_invariance_err': 2.220446049250313e-16, 'abs_dY_ci': {'n': 16, 'n_clusters': 8, 'point': 0.0005016926351348375, 'boot_mean': 0.0005032993297388275, 'ci95_low': 0.00029409981693381273, 'ci95_high': 0.0007579991377326188}, 'js_ci': {'n': 16, 'n_clusters': 8, 'point': 1.0392019147970522e-06, 'boot_mean': 1.0450021348607801e-06, 'ci95_low': 6.669656814461607e-07, 'ci95_high': 1.4262185453660318e-06}}
- Paired vs random: {'n_pairs': 32, 'n_random_draws_mean': 3.0, 'mean_abs_dY_donor': 0.0003162771864545119, 'mean_abs_dY_rand': 0.00017336593519576228, 'mean_abs_dY_diff': 0.0001429112512587496, 'median_abs_dY_diff': 7.490177263962133e-05, 'mean_js_diff': 6.055011227311003e-07, 'frac_donor_outperforms': 0.84375, 'abs_dY_diff_ci': {'n': 32, 'n_clusters': 32, 'point': 0.0001429112512587496, 'boot_mean': 0.00014111025322638543, 'ci95_low': 7.006603727976298e-05, 'ci95_high': 0.00021629890860456192}, 'A': {'frac_outperforms': 0.8333333333333334, 'n': 24, 'n_targets': 2}, 'B': {'frac_outperforms': 0.875, 'n': 8, 'n_targets': 6}}

### ε = 2.0

- Donor-null rows/pairs: 64/32; targets=6
- Mean |ΔY|=0.0006580428815188853 median=0.00057265009780845 CI=n=64 clusters=32 point=0.0007 CI95=[0.0005, 0.0009]
- Mean JS=4.61286937232236e-06 CI=n=64 clusters=32 point=0.0000 CI95=[0.0000, 0.0000]
- Mean achieved L2=2.0
- Invariance failures=0 max err=2.220446049250313e-16
- Centered FD (Y+ − Y−)/(2ε): {'mean_fd': -0.00018187550633163327, 'median_fd': -5.90950039480731e-06, 'fd_ci': {'n': 32, 'n_clusters': 32, 'point': -0.00018187550633163327, 'boot_mean': -0.00018268423339259804, 'ci95_low': -0.00032536085423684864, 'ci95_high': -4.74430128908678e-05}}
- A: {'slice': 'A', 'n_pairs': 24, 'n_recipient_games': 24, 'n_target_games': 2, 'n_rows': 48, 'mean_abs_dY': 0.0005176816759415054, 'median_abs_dY': 0.00045919511071260003, 'mean_js': 4.200999414482809e-06, 'median_js': 3.6983023814967982e-06, 'mean_l2': 2.0, 'invariance_fail': 0, 'max_invariance_err': 2.220446049250313e-16, 'abs_dY_ci': {'n': 48, 'n_clusters': 24, 'point': 0.0005176816759415054, 'boot_mean': 0.0005174789720078232, 'ci95_low': 0.00032312825045053574, 'ci95_high': 0.0007116314375973957}, 'js_ci': {'n': 48, 'n_clusters': 24, 'point': 4.200999414482809e-06, 'boot_mean': 4.19276544214003e-06, 'ci95_low': 3.4262451929672015e-06, 'ci95_high': 5.067500553760853e-06}}
- B: {'slice': 'B', 'n_pairs': 8, 'n_recipient_games': 8, 'n_target_games': 6, 'n_rows': 16, 'mean_abs_dY': 0.001079126498251025, 'median_abs_dY': 0.00082586498451745, 'mean_js': 5.848479245841017e-06, 'median_js': 3.890673008617345e-06, 'mean_l2': 2.0, 'invariance_fail': 0, 'max_invariance_err': 2.220446049250313e-16, 'abs_dY_ci': {'n': 16, 'n_clusters': 8, 'point': 0.001079126498251025, 'boot_mean': 0.0010833165227242068, 'ci95_low': 0.0006503402436620925, 'ci95_high': 0.0015936694166472035}, 'js_ci': {'n': 16, 'n_clusters': 8, 'point': 5.848479245841017e-06, 'boot_mean': 5.879221116254103e-06, 'ci95_low': 3.524957828211425e-06, 'ci95_high': 8.323044016477822e-06}}
- Paired vs random: {'n_pairs': 32, 'n_random_draws_mean': 3.0, 'mean_abs_dY_donor': 0.0006580428815188853, 'mean_abs_dY_rand': 0.0003473549517387431, 'mean_abs_dY_diff': 0.00031068792978014223, 'median_abs_dY_diff': 0.00015827677585398338, 'mean_js_diff': 3.1346308036664545e-06, 'frac_donor_outperforms': 0.875, 'abs_dY_diff_ci': {'n': 32, 'n_clusters': 32, 'point': 0.00031068792978014223, 'boot_mean': 0.0003070898566021453, 'ci95_low': 0.00016299681694207028, 'ci95_high': 0.00046080782508501033}, 'A': {'frac_outperforms': 0.875, 'n': 24, 'n_targets': 2}, 'B': {'frac_outperforms': 0.875, 'n': 8, 'n_targets': 6}}

### ε = 4.0

- Donor-null rows/pairs: 64/32; targets=6
- Mean |ΔY|=0.0015120965252168519 median=0.0011184067351863 CI=n=64 clusters=32 point=0.0015 CI95=[0.0011, 0.0020]
- Mean JS=4.759706365060447e-05 CI=n=64 clusters=32 point=0.0000 CI95=[0.0000, 0.0001]
- Mean achieved L2=4.0
- Invariance failures=0 max err=2.220446049250313e-16
- Centered FD (Y+ − Y−)/(2ε): {'mean_fd': -0.00021801268509702337, 'median_fd': -6.23922592168924e-06, 'fd_ci': {'n': 32, 'n_clusters': 32, 'point': -0.00021801268509702337, 'boot_mean': -0.0002182698577339862, 'ci95_low': -0.00037369380271164843, 'ci95_high': -7.334250617215004e-05}}
- A: {'slice': 'A', 'n_pairs': 24, 'n_recipient_games': 24, 'n_target_games': 2, 'n_rows': 48, 'mean_abs_dY': 0.0011311896782543022, 'median_abs_dY': 0.00077995047927835, 'mean_js': 2.855998898775379e-05, 'median_js': 1.8029595949992744e-05, 'mean_l2': 4.0, 'invariance_fail': 0, 'max_invariance_err': 6.938893903907228e-18, 'abs_dY_ci': {'n': 48, 'n_clusters': 24, 'point': 0.0011311896782543022, 'boot_mean': 0.001131021366565804, 'ci95_low': 0.000702600849310592, 'ci95_high': 0.0015673268818392468}, 'js_ci': {'n': 48, 'n_clusters': 24, 'point': 2.855998898775379e-05, 'boot_mean': 2.84978812451853e-05, 'ci95_low': 2.3231411813920918e-05, 'ci95_high': 3.4672907390856534e-05}}
- B: {'slice': 'B', 'n_pairs': 8, 'n_recipient_games': 8, 'n_target_games': 6, 'n_rows': 16, 'mean_abs_dY': 0.0026548170661045, 'median_abs_dY': 0.0023982230009145502, 'mean_js': 0.00010470828763915654, 'median_js': 1.960307135886489e-05, 'mean_l2': 4.0, 'invariance_fail': 0, 'max_invariance_err': 2.220446049250313e-16, 'abs_dY_ci': {'n': 16, 'n_clusters': 8, 'point': 0.0026548170661045, 'boot_mean': 0.002662101945927161, 'ci95_low': 0.0018387632712030715, 'ci95_high': 0.00359755884232654}, 'js_ci': {'n': 16, 'n_clusters': 8, 'point': 0.00010470828763915654, 'boot_mean': 0.00010527088872408075, 'ci95_low': 3.7087896818365615e-05, 'ci95_high': 0.00018584032403192298}}
- Paired vs random: {'n_pairs': 32, 'n_random_draws_mean': 3.0, 'mean_abs_dY_donor': 0.0015120965252168519, 'mean_abs_dY_rand': 0.0006980638206649533, 'mean_abs_dY_diff': 0.0008140327045518985, 'median_abs_dY_diff': 0.00037301777040739165, 'mean_js_diff': 4.085259692777031e-05, 'frac_donor_outperforms': 0.90625, 'abs_dY_diff_ci': {'n': 32, 'n_clusters': 32, 'point': 0.0008140327045518985, 'boot_mean': 0.0008083239737753747, 'ci95_low': 0.0004832724213901486, 'ci95_high': 0.0011465437271481072}, 'A': {'frac_outperforms': 0.875, 'n': 24, 'n_targets': 2}, 'B': {'frac_outperforms': 1.0, 'n': 8, 'n_targets': 6}}

These perturbations sit near a natural donor residual. That is not a proof they are on-manifold.

## 3. Wrong-donor specificity

{'present': 1, 'n': 32, 'n_recipient_games': 32, 'n_target_games': 6, 'mean_wrong_dY': 0.6240248035195239, 'mean_correct_E_N': 1.0936478252310151, 'mean_correct_minus_wrong': 0.46962302171149106, 'median_correct_minus_wrong': 0.4122050169046611, 'mean_wrong_dP_gain': 0.20350093781075648, 'mean_wrong_dP_loss': -0.49991422555457504, 'mean_wrong_kl': 2.4795705023583055, 'mean_wrong_l2': 16.67041100153408, 'mean_correct_l2_N': 15.386063777524564, 'cost_matched': 0, 'wrong_donor_js': 'NOT_IDENTIFIABLE_FROM_SAVED_DATA', 'diff_ci': {'n': 32, 'n_clusters': 32, 'point': 0.46962302171149106, 'boot_mean': 0.47071171036032455, 'ci95_low': 0.38241974605521484, 'ci95_high': 0.5612887472472681}, 'A_n_targets': 2, 'B_n_targets': 6, 'hybrid_random': {'n': 32, 'n_pairs': 32, 'mean_abs_dY': 0.1246633727056522, 'mean_l2': 15.386063777524564}, 'note': 'Wrong-donor rows were scored on the intended target legal set. Full-distribution JS for wrong-donor was not saved.'}

If L2 is not matched, this is not a cost-matched comparison.

## 4. SVD / uniqueness

- Rank 124 nullspace 388 tolerance 1e-08 sensitivity [{'tol': 1e-06, 'rank': 124, 'nullspace_dim': 388}, {'tol': 1e-08, 'rank': 124, 'nullspace_dim': 388}, {'tol': 1e-10, 'rank': 124, 'nullspace_dim': 388}]
- Δ_N is one vector in that nullspace. It is not a uniquely identified board-state coordinate, and the 388-dimensional nullspace is not a new board representation.

## 5. Claims

### A. Continuous-probe insufficiency: **SUPPORTED**

Preserved from the original FIBER run after raw-array verification. Two activations with matching complete probe probabilities produce different F_x.

### B. Local causal sensitivity near natural activations: **NOT_SUPPORTED**

Primary scale is ε=1. Larger ε are reported separately and are not pooled into this verdict.

### C. Target-specific missing mathematical state: **INCONCLUSIVE**

Not inferred from A or B. Requires donor-derived null directions to beat random-null and wrong-donor controls on the intended legal set.

## Limitations

- Hybrid H01/H10 activations can be off-manifold; their large L2 is not evidence of a natural germ.
- Recipient-game CIs on category A use 24 recipients and 2 targets.
- History, prefix length, and policy features can sit in the residual even when the board probe is matched.
- Wrong-donor JS was not stored.
- Rank is a numerical truncation of W, not a unique causal basis.

## Verdicts

- Claim A: `SUPPORTED`
- Claim B: `NOT_SUPPORTED`
- Claim C: `INCONCLUSIVE`
