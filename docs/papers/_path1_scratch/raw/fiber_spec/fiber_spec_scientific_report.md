# 08R2-FIBER-SPEC Scientific Report

L2-matched correct-versus-wrong-donor probe-null interventions on a new disjoint validation pair set.

- `08R2_VALIDATION_COMPLETE — NEGATIVE`
- Original 0A: **C / PARTIAL**
- `SECTION_5_3 = PARTIAL`
- `OBSTRUCTION_READY = NO`

Verdict: `TARGET_SPECIFIC_NULL_EFFECT_SUPPORTED`
Selected pairs: A=32 B=16 hash=2789accfc4b0908168509f7a998f9e95ddb8a4f36f7a661aa1b9d5e4a11143d3
Attempts: {'pairs': 48, 'editor_fail': 0, 'invariance_fail': 0, 'l2_fail': 0, 'valid': 48}
Gate: {'invariance_ok': 1, 'l2_ok': 1, 'mean_delta_specific_positive': 1, 'ci_lower_gt_0': 1, 'mean_delta_ge_floor': 1, 'both_components_correct_sign': 1, 'beats_random': 1, 'integrity_ok': 1}
SVD rank=124 null=388
Pooled: {'n': 48, 'n_recipient_games': 48, 'n_target_games': 48, 'mean_delta_specific': 0.39162660148043377, 'median_delta_specific': 0.3769966303438347, 'mean_delta_random': 0.708583514286317, 'mean_dG': 0.2394166634149233, 'mean_dQ': -0.1522099380655105, 'frac_correct_wins_wrong': 0.9583333333333334, 'frac_correct_wins_random': 1.0, 'delta_specific_ci': {'n': 48, 'n_clusters': 48, 'point': 0.39162660148043377, 'boot_mean': 0.3921046298921325, 'ci95_low': 0.31745051211471276, 'ci95_high': 0.47584000267139553}, 'delta_random_ci': {'n': 48, 'n_clusters': 48, 'point': 0.708583514286317, 'boot_mean': 0.7085342284567859, 'ci95_low': 0.6020622147364137, 'ci95_high': 0.8169696797707171}}
A: {'n': 32, 'n_recipient_games': 32, 'n_target_games': 32, 'mean_delta_specific': 0.4226149611333635, 'median_delta_specific': 0.39377688825252954, 'mean_delta_random': 0.6895603307635643, 'mean_dG': 0.2526348131851674, 'mean_dQ': -0.16998014794819608, 'frac_correct_wins_wrong': 0.9375, 'frac_correct_wins_random': 1.0, 'delta_specific_ci': {'n': 32, 'n_clusters': 32, 'point': 0.4226149611333635, 'boot_mean': 0.4221103636483622, 'ci95_low': 0.3200846309652058, 'ci95_high': 0.5320755303743938}, 'delta_random_ci': {'n': 32, 'n_clusters': 32, 'point': 0.6895603307635643, 'boot_mean': 0.6889344537878987, 'ci95_low': 0.5572312483371643, 'ci95_high': 0.8316069940753238}}
B: {'n': 16, 'n_recipient_games': 16, 'n_target_games': 16, 'mean_delta_specific': 0.32964988217457447, 'median_delta_specific': 0.3244184825247203, 'mean_delta_random': 0.7466298813318228, 'mean_dG': 0.21298036387443514, 'mean_dQ': -0.11666951830013932, 'frac_correct_wins_wrong': 1.0, 'frac_correct_wins_random': 1.0, 'delta_specific_ci': {'n': 16, 'n_clusters': 16, 'point': 0.32964988217457447, 'boot_mean': 0.3307085657323131, 'ci95_low': 0.2502968641824598, 'ci95_high': 0.40870340321926657}, 'delta_random_ci': {'n': 16, 'n_clusters': 16, 'point': 0.7466298813318228, 'boot_mean': 0.7474918257373013, 'ci95_low': 0.5818497989002485, 'ci95_high': 0.9207429937410097}}

A supported gate means the correct-donor null component beat L2-matched wrong donors and random null directions on the intended legal set.
It does not identify a unique game-state coordinate and does not validate obstruction theory.
