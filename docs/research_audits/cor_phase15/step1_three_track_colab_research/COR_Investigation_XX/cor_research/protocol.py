"""Investigation XX protocol, v2 (after the three pre-run agent reviews A/B/C, see reviews/ADJUDICATION_v1.md).  The single
machine-readable source of every design constant (written to protocols/*.json by scripts/make_protocols.py, frozen with hashes by
scripts/freeze.py, verified by the runner before any job).  Nothing in this file depends on any confirmation outcome."""
import copy

# ------------------------------------------------------------------------------------------------------------------------------
# Parents (A1).  Preregistered, ordered list; every listed parent is accounted for in the denominator (valid / invalid / failed
# pretraining / screened / eligible units).  Fresh XX parents first (review A M3).  uP161-165 are PREVIOUSLY OBSERVED (XVIII
# screened them and ran intervention arms on uP163; thresholds were informed by them): every Track A verdict is also reported on
# fresh parents only, and a headline GENERALIZES requires the fresh-only verdict not to be FAILS (scorer).
HIST_PARENTS = ["uP161", "uP162", "uP163", "uP164", "uP165"]; PREVIOUSLY_OBSERVED = list(HIST_PARENTS)
FRESH_PARENTS = [f"nP{3000 + i}" for i in range(1, 13)]
PILOT_PARENTS = ["nP3901"]                                     # pilot material is disjoint from every confirmation parent/data order (review C M8)
DEV_PARENT = "uP160"                                           # XIX development parent: never in a scientific preset
PARENT_VALIDITY = dict(metric="pretraining eval aux_acc (train_tasks.evaluate('pcount', seed=41_000_000))", min_aux_acc=0.95,
                       note="below threshold -> INVALID_PRETRAINING (reported); a failed pretraining job -> FAILED_PRETRAINING (reported). "
                            "Known: uP162 = 0.779 (threshold set knowing this; review A M3).")
PRETRAIN = dict(task="pcount", steps=3000, recipe="train_tasks.train (XIII cohort13_pretrain.py), resumable wrapper cor_research/training/pretrain.py")

# Screening: baseline runs only, BEFORE any intervention; two data orders (review A M2).
SCREEN = dict(seeds=[20, 21, 22, 23, 24, 25, 26, 27], D=901, D2=902, stop=1211, stream=91_000_000, all_seeds_always_run=True,
              unit_rule="unit (parent, base) is ELIGIBLE iff some seed s (L = W = s) gives a competent, causally pure baseline of the base class at step "
                        "1211 on BOTH screening data orders 901 and 902 (base E: EARLY, base M: MIXTURE); the unit seed is the lowest such s. Only screening "
                        "records whose measurement job is registered done with matching checkpoint/code hashes and the frozen stream are used. A screening "
                        "run or measurement that fails 3 attempts makes THAT SEED non-qualifying (recorded as failed); the parent is still screened.")
COMP = dict(KL=0.06, acc=0.80); CLASS = dict(E_hi=0.30, E_lo_M=0.10, M_hi=0.30, M_lo_E=0.15)

CONF_D = [2001, 2002, 2003, 2004, 2005, 2006]; PILOT_D = [2901]
STREAMS = dict(measure_primary=91_100_000, measure_window=91_150_000, measure_secondary=91_200_000)
STEPS = dict(window=100, horizon_hist=1211, primary=4000, stability_end=12000, stability_lr=1e-4, stability_lr_hi=1e-3, early_checks=[150, 300])

# Window (step 0-100) states.  base E: target route T = own0 (layer 0); base M: T = own2 (layer 2).
WINDOW = {
    "B": {},
    "I": dict(FX_INTERV="gof:100", FX_EDGES="{T}", FX_B="1e9"),
    "vsgo": dict(FX_INTERV="vsgo:100", FX_EDGES="{T}"),
    "rsg": dict(FX_INTERV="rsg:100", FX_EDGES="{T}"),
    "lbl": dict(FX_INTERV="lbl:100", FX_LBL_P="0.5", FX_LBL_SEED="{lblseed}"),
}
ALLG = ["L0_LN", "L0_MLP", "L0_OV", "L0_QK", "L1_LN", "L1_MLP", "L1_OV", "L1_QK", "L2_LN", "L2_MLP", "L2_OV", "L2_QK", "WA", "emb", "final_LN"]
QK = "L0_QK,L1_QK,L2_QK"; NONQK = "L0_OV,L0_MLP,L0_LN,L1_OV,L1_MLP,L1_LN,L2_OV,L2_MLP,L2_LN,WA,emb,final_LN"
# Continuation arms (branch from a composed or window step-100 state; train to 4000; snapshots include 150, 300, 1211).
#   transplant src = (base state, donor state); groups taken from the donor; opt = Adam moments (follow = with the parameters).
#   'bases' restricts an arm to one base.  reverse=True marks necessity-type reverse transplants (compatibility flag applies).
ARMS = {
    "fullB": dict(kind="transplant", src=("B", "I"), groups="none", opt="follow", bases="EM", role="anchor (B state re-entered through the composer; bitwise check)"),
    "fullI": dict(kind="transplant", src=("B", "I"), groups="all", opt="follow", bases="EM", role="anchor (I state)"),
    "paramI_optB": dict(kind="transplant", src=("B", "I"), groups="all", opt="B", bases="EM", role="parameters without moments"),
    "paramB_optI": dict(kind="transplant", src=("B", "I"), groups="none", opt="I", bases="EM", role="moments without parameters (optimizer-state-only control)"),
    "BfromI_L0": dict(kind="transplant", src=("B", "I"), groups="L0", opt="follow", bases="EM", role="forward transplant, layer 0"),
    "IfromB_L0": dict(kind="transplant", src=("I", "B"), groups="L0", opt="follow", bases="E", reverse=True, role="reverse transplant (restore B's layer 0 into I)"),
    "BfromI_L0L1": dict(kind="transplant", src=("B", "I"), groups="L0,L1", opt="follow", bases="M", role="component combination"),
    "BfromI_L0L2": dict(kind="transplant", src=("B", "I"), groups="L0,L2", opt="follow", bases="M", role="component combination"),
    "IfromB_L0L2": dict(kind="transplant", src=("I", "B"), groups="L0,L2", opt="follow", bases="M", reverse=True, role="reverse combination"),
    "randnorm": dict(kind="perturb", src=("B", "I"), groups_by_base=dict(E="L0", M="L0,L2"), opt="B", bases="EM",
                     role="parameter-norm control: B + random direction with ||I-B|| per group, groups = the base's sufficiency groups (review A M6)"),
    "BfromVSGO_QK": dict(kind="transplant", src=("B", "vsgo"), groups=QK, opt="follow", bases="M",
                         role="routing-state-only transplant: QK of every layer from the vsgo window state (review A B2 option b)"),
    "BfromVSGO_nonQK": dict(kind="transplant", src=("B", "vsgo"), groups=NONQK, opt="follow", bases="M", role="complement: every non-QK group from the vsgo window state"),
    "BfromVSGO_L2QK": dict(kind="transplant", src=("B", "vsgo"), groups="L2_QK", opt="follow", bases="M", role="layer-2 QK only from the vsgo window state (review A r2 N1)"),
    "vsgo": dict(kind="window", window="vsgo", bases="EM", role="value/output-gradient intervention through T (routing gradient intact)"),
    "rsg": dict(kind="window", window="rsg", bases="EM",
                role="readout-row routing-gradient removal at the target layer (on M = layer-2 QK frozen + upstream gradient change: a replication, review A B2)"),
    "lbl": dict(kind="window", window="lbl", bases="EM", role="matched non-targeted disturbance (validity: step-100 KL excess >= 0.5 x the block's)"),
}
HIGH_EXTRA_ARMS = {
    "IfromB_L0_QK": dict(kind="transplant", src=("I", "B"), groups="L0_QK", opt="follow", bases="E", reverse=True, role="EXPLORATORY: reverse transplant, layer-0 QK only"),
    "IfromB_L0_OV": dict(kind="transplant", src=("I", "B"), groups="L0_OV", opt="follow", bases="E", reverse=True, role="EXPLORATORY: reverse transplant, layer-0 OV only"),
    "half_L0L2": dict(kind="transplant", src=("B", "I"), groups="L0,L2", opt="B", alpha=0.5, bases="EM", role="EXPLORATORY: interpolation 0.5 on L0+L2"),
}
VALIDITY = dict(lbl_excess_ratio_min=0.5, incompatible_kl_excess=0.10, incompatible_sensitivity=[0.05, 0.20],
                incompatible_rule="a composed state is INCOMPATIBLE if its KL at step 100 (composed state) exceeds both window states' KL, or at step 150 / 300 "
                                  "exceeds both anchors' KL, by more than 0.10 (uncalibrated; flag rates and 0.05 / 0.20 sensitivity reported). A supporting outcome "
                                  "of a flagged transplant (any transplant arm) is relabelled INCOMPATIBLE (not evidence)")

# ------------------------------------------------------------------------------------------------------------------------------
# Track A frozen hypotheses and verdict rules (v2, review A B1/M1/M5/M13).
#   phi = 0.5 [phi_E + phi_M], phi_r = [l(nec_r^arm) - l(nec_r^B)] / [l(nec_r^I) - l(nec_r^B)], l(x) = log(max(x,0) + EPS), EPS = 0.01
#   (sensitivity EPS = 0.003 and 0.03 reported).  I_like: phi >= 0.75 and class == I's.  B_like: phi <= 0.25 and class == B's.
#   partial: 0.25 <= phi <= 0.75 and class HYBRID.  Pair CONTRA: I_like -> phi < 0.65 or class == B's;  B_like -> phi > 0.35 or class == I's;
#   partial -> class == B's or I's with phi outside [0.15, 0.85].
#   Parent: SUPPORTS iff #SUPPORTED >= ceil(0.5 n_ev) and #CONTRA <= max(1, floor(0.25 n_ev)); CONTRADICTS iff #CONTRA >= ceil(n_ev / 2) and
#   #CONTRA > #SUPPORTED; else PARTIAL; n_ev < 3 -> INCONCLUSIVE.  Cross-parent GENERALIZES requires every parent at n_ev = 4 and all but one at n_ev >= 5.
#   Cross-parent (n_ev parents): INCONCLUSIVE if n_ev < 4; GENERALIZES iff #SUPPORTS >= max(n_ev - 1, ceil(0.8 n_ev)) and #CONTRADICTS == 0;
#   FAILS_TO_GENERALIZE iff #CONTRADICTS >= max(2, ceil(n_ev / 3)); otherwise NOT_ESTABLISHED.  Operating characteristics: tests/test_verdict_oc.py.
#   Mediation hypotheses are CONDITIONAL on the switch in the same pair; a mediation GENERALIZES is reported as GENERALIZES_CONDITIONAL
#   unless HA1 GENERALIZES on the same base.  Every verdict is reported on all parents and on fresh parents only.
BAND = dict(I_like=(0.75, 9.99), B_like=(-9.99, 0.25), partial=(0.25, 0.75)); DELTA = 0.10; EPS = 0.01; EPS_SENS = [0.003, 0.03]
MIN_PAIRS_PER_PARENT = 3; MIN_PARENTS = 4
PARENT_RULE = dict(support_frac=0.5, max_contra=1, max_contra_frac=0.25, contra_frac=0.5, contra_rule="C >= ceil(n/2) and C > S (review A r2 N4)")   # v2.1: calibrated by tests/test_verdict_oc.py (the 0.75 rule was non-monotone in n_D)
CROSS_RULE = dict(gen_frac=0.8, gen_exceptions=1, fail_min=2, fail_frac=1 / 3)
HYPOTHESES_A = [
    dict(id="HA1", bases="EM", arm="fullI", expect="switch", step=4000, role="primary",
         text="Blocking the base's own route for the first 100 steps switches the organisation: the I anchor is competent and of the opposite causal class at 4000."),
    dict(id="HA2", bases="EM", arm="paramI_optB", expect="I_like", step=4000, role="mediation", text="The step-100 parameters carry the selection (I parameters with B moments -> I-like)."),
    dict(id="HA3", bases="EM", arm="paramB_optI", expect="B_like", step=4000, role="mediation", text="Adam moments alone do not carry it (B parameters with I moments -> B-like)."),
    dict(id="HA4a", bases="E", arm="BfromI_L0", expect="I_like", step=4000, role="replication_anchor", text="E: the I layer-0 state is sufficient (near-tautological; replication anchor, not a finding)."),
    dict(id="HA4b", bases="E", arm="IfromB_L0", expect="B_like", step=4000, role="mediation", text="E: the I layer-0 state is necessary (restoring B's layer 0 prevents the switch); INCOMPATIBLE pairs are not evidence."),
    dict(id="HA5a", bases="M", arm="BfromI_L0L1", expect="I_like", step=4000, role="mediation", text="M: L0+L1 sufficient."),
    dict(id="HA5b", bases="M", arm="BfromI_L0L2", expect="I_like", step=4000, role="mediation", text="M: L0+L2 sufficient."),
    dict(id="HA5c", bases="M", arm="BfromI_L0", expect="not_I_like", step=4000, role="weak_negative", text="M: L0 alone is not sufficient (weak negative claim; not headlined)."),
    dict(id="HA5d", bases="M", arm="IfromB_L0L2", expect="B_like", step=4000, role="mediation", text="M: restoring B's L0+L2 prevents the switch; INCOMPATIBLE pairs are not evidence."),
    dict(id="HA6a", bases="EM", arm="randnorm", expect="B_like", step=4000, role="control", text="A random displacement of the sufficiency groups with matched per-group norm does not reproduce the effect."),
    dict(id="HA6b", bases="EM", arm="lbl", expect="B_like", step=4000, role="control", text="Label-noise loss disturbance (valid strength only) does not reproduce the effect."),
    dict(id="HA7", bases="M", arm="vsgo", expect="I_like", step=4000, role="replication", text="M: removing V+O learning through the blocked route suffices (XIX replication)."),
    dict(id="HA10a", bases="M", arm="BfromVSGO_QK", expect="I_like", step=4000, role="discriminating", condition="vsgo_I_like",
         text="M (new): the QK WEIGHTS (all layers) learned under vsgo suffice. This is a QK-weights vs non-QK-weights test, not a pure routing-vs-value test "
              "(vsgo changes the layer-0/1 QK gradients on M; review A r2 N1). Evaluated only in pairs whose vsgo arm is I-like. Interpretation table: DESIGN §1."),
    dict(id="HA10b", bases="M", arm="BfromVSGO_nonQK", expect="B_like", step=4000, role="discriminating", condition="vsgo_I_like",
         text="M (new): the non-QK weights learned under vsgo do not suffice (complement of HA10a; only in pairs whose vsgo arm is I-like)."),
    dict(id="HA8E", bases="E", arm="rsg", expect="partial", step=4000, co_primary_step=1211, role="low_confidence",
         text="E: removing the routing gradient of the layer-0 readout rows (all edges) gives a PARTIAL, HYBRID-class outcome. Basis: one dev run (uP160, D801, "
              "step 1211). Overlaps XIX E_P5_frzp_partial; SUPPORTED distinguishes little, a CONTRA outcome is the informative one."),
    dict(id="HA9", bases="EM", arm="fullB,fullI", expect="persist", step=4000, role="primary",
         text="Persistence: each anchor's causal class at 1211 equals its class at 4000 and it is competent at 4000 (>= 80% of evaluable pairs per parent)."),
]
EXPLORATORY_A = [
    dict(id="XA1", bases="E", arm="vsgo", text="E: vsgo effect (no directional prediction in XIX; distribution reported)."),
    dict(id="XA2", bases="M", arm="rsg", text="M: rsg = replication of XIX M_P5_frzp_nil plus an upstream-gradient change (review A B2: not discriminating)."),
    dict(id="XA3", bases="EM", arm="*", step=1211, text="Every hypothesis re-evaluated at the historical horizon 1211 (secondary)."),
    dict(id="XA4", bases="EM", arm="fullB,fullI", text="Stability 4000 -> 12000: low-LR (peak 1e-4) and full-LR positive control (peak 1e-3); wording limited to "
                                                     "'class unchanged under this continuation' unless the full-LR control moves some organisation."),
    dict(id="XA5", bases="EM", arm="*", text="Unconditional sensitivity: over every pair with a native B anchor, the arm's class and its concordance with the same pair's fullI class."),
    dict(id="XA6", bases="E", arm="IfromB_L0_QK,IfromB_L0_OV,half_L0L2", text="High-preset extra arms: exploratory, no frozen prediction."),
    dict(id="XA7", bases="M", arm="BfromVSGO_L2QK", text="M: layer-2-QK-only transplant from the vsgo window (localises HA10a; exploratory)."),
]

# ------------------------------------------------------------------------------------------------------------------------------
# Track B (v2, reviews A B3-B5/M8-M11 and B BLK-1/2, MAJ-1..7).
TRACK_B = dict(
    units="confirmation anchors fullB and fullI; class = causal class at the analysed step (competent, pure)",
    matching=dict(selection_stream=91_430_000, n=3000, steps="every saved snapshot in [150, 4000]",
                  rule="per parent, kappa_p = the largest KL at step 4000 among the parent's anchors that are competent and pure at 4000 (floored at 0.015); "
                       "each anchor's matched snapshot is the EARLIEST snapshot whose selection-stream KL <= kappa_p (every such anchor reaches it by 4000: no "
                       "class-dependent censoring by construction); matched step, LR and step difference are reported per class (review A M9)"),
    sets=dict(nat_stream=91_400_000, disc_stream=91_410_000, pair_stream=91_420_000, n_nat=12000, n_disc_pool=40000, n_pairs=3000),
    candidates="bayes; local; noelig; nocount; maprole (full role MAP, tie-averaged); softrole (geometric pooling over roles: listed, calibration-only); "
               "historical nested power-mean with lapse (xviiim.rho_fit_lapse; descriptive); threshold-4 rule: NOT LOCATED in the vendored code, not reconstructed",
    thresholds=dict(delta_dis=0.02, delta_dir=0.03, delta_w=0.03, delta_oqi=0.10, delta_kl=0.005, min_parents_both=4, indicative_below_parents=6, alpha=0.05),
    tests={
        "HB_c": "PRIMARY functional test (decision level, calibration-invariant, on-support tie-aware discriminating set). Per parent: the within-unit contrast "
                "(different data orders in both terms: between = fullB(D1) vs fullI(D2), within = same class, D1 != D2) measures class + intervention history; the "
                "same-D cross-unit contrast (parents with both bases; history-matched) measures class only; the parent statistic is their average when both exist "
                "(history cancels, review A r2 N-B2), else the within-unit contrast with the parent flagged history-confounded. If every parent is confounded, a "
                "difference is reported as FUNCTIONAL_DIFFERENCE (class and intervention history not separable), never as B1.",
        "HB_c_dir": "rule-directed (v2.1, tie-aware): for k in {local, noelig} (nocount / maprole differences are exact ties), on the inputs where k's UNIQUE "
                    "decision differs from Bayes's: per model d_k = P(decision = k's) - P(decision = the third option) (zero in expectation for precision-only "
                    "errors); per parent mean_EARLY - mean_MIXTURE; Holm across the two rules on t-test p; B1 requires the native-vs-native and induced contrasts "
                    "to have the SAME sign (missing -> not satisfied)",
        "HB_a": "DESCRIPTIVE ONLY: rule-mixture weights over the precision family F0' (tau, eps, alpha' free), KL-adjusted (within-class regression to kappa_p)",
        "HB_b": "DESCRIPTIVE ONLY: deviation above the parametric-precision null Bayes(alpha') at the model's KL and above the iid logit-noise null",
        "HB_d": "DESCRIPTIVE LOCUS profile: other-question interchange from the HYBRID run (source-matched; base-run null must reproduce the base), IIA normalised by "
                "the model's own following of Bayes(x_h); EARLY - MIXTURE at the layer-2 input. It shows where cross-question evidence is read, not what is computed.",
        "HB_e": "precision: KL at 4000, mean_p(E - M); expected positive from development (dev dKL ~ 0.015 > delta_kl): a '+B3' flag is not a discovery",
        "equivalence": "TOST at parent level: functional equivalence iff the upper 95% bound of HB_c < delta_dis AND for every rule the 90% CI of HB_c_dir lies within "
                       "+-delta_dir (t-interval on parent means; exact sign-flip p reported)",
    },
    classification=["INSUFFICIENT_UNITS if < 4 parents have >= 1 evaluable model of each class (labels 'indicative' below 6)",
                    "B1-FUNCTIONAL (different decision functions; internal causal mechanism NOT identified by this design): HB_c difference (lower 95% bound > "
                    "delta_dis, sign-flip p reported) AND some rule-directed HB_c_dir difference (Holm) with the same sign in the native-vs-native and induced contrasts",
                    "FUNCTIONAL_DIFFERENCE_RULE_UNIDENTIFIED: HB_c difference without a rule-directed difference",
                    "B2-LOCUS (functionally equivalent; different location of unidentified content): equivalence AND |HB_d| > delta_oqi (sign-consistent)",
                    "B3 (same rule, different precision): equivalence AND not B2-LOCUS AND HB_e; '+B3' flag added to B2-LOCUS when HB_e",
                    "EQUIVALENT_NO_DIFFERENCE_DETECTED: equivalence and neither HB_d nor HB_e",
                    "NON_IDENTIFIABLE (underpowered): neither a functional difference nor equivalence"],
    headline_rule="v2.1 (review A r2 N-B1): the FUNCTIONAL status (difference / equivalence / neither) must agree between the matched and the final levels "
                  "(else NON_IDENTIFIABLE); the locus (HB_d) is read at the matched level and the precision flag (HB_e) at the final level (matching removes the "
                  "precision difference by construction)",
    frozen_prediction="From development (uP160 XIX anchors, n = 2 + 2; development only): predicted B2-LOCUS (+B3 flag): no functional difference at matched "
                      "precision, equivalence achieved if >= 6 parents; HB_d EARLY > MIXTURE at the layer-2 input; HB_e positive (expected, not a discovery).",
)
TRACK_C = dict(code="cor_research/mathematics (od_original, od_repaired, t4_order, run_track_c)", neural_test=None,
               classification="cor_research.mathematics.run_track_c.classify (predeclared; C5 is reviewer-adjudicated with the candidate list)",
               rule="no neural O/D experiment is run: every candidate neural prediction considered (factor-deletion destinations, O-tie sensitivity, soft-gate "
                    "expected-energy pooling, T4 latent-role expected energy, T4 additive complex) coincides with exact Bayes on the declared model, is a standard "
                    "mean-field/log-linear pooling distinction, needs an unstated bridge assumption, or is not representable (review B §3.6, C-MAJ-3)")

# ------------------------------------------------------------------------------------------------------------------------------
ARMSET = ["fullB", "fullI", "paramI_optB", "paramB_optI", "BfromI_L0", "IfromB_L0", "BfromI_L0L1", "BfromI_L0L2", "IfromB_L0L2", "randnorm", "BfromVSGO_QK", "BfromVSGO_nonQK", "BfromVSGO_L2QK", "vsgo", "rsg", "lbl"]
PRESETS = {
    "smoke": dict(label="SMOKE: software test only (CPU-feasible, shortened schedules). NEVER evidence; every verdict is forced to TECHNICAL_ONLY.",
                  parents=[DEV_PARENT], screen_seeds=[21, 23], n_D=1, D_list=[2001], arms=["fullB", "fullI", "paramI_optB", "rsg", "lbl"], stability=True, trackb=True, max_units=2,
                  forced_units=[[DEV_PARENT, "E", 21], [DEV_PARENT, "M", 23]],
                  overrides=dict(STEPS=dict(window=100, horizon_hist=150, primary=200, stability_end=260, stability_lr=1e-4, stability_lr_hi=1e-3, early_checks=[150]),
                                 SCREEN_stop=150, stability_snaps=[220, 240]),
                  claims_can="code paths execute", claims_cannot="anything scientific"),
    "pilot": dict(label="Pilot: end-to-end technical validation on separate material (parent nP3901, data order 2901). NOT confirmation evidence.", parents=PILOT_PARENTS,
                  screen_seeds=[20, 21, 22, 23], n_D=1, D_list=PILOT_D, arms=["fullB", "fullI", "paramI_optB", "rsg", "lbl"], stability=False, trackb=True, max_units=2,
                  claims_can="the pipeline runs end to end on this runtime; measured throughput; storage", claims_cannot="any scientific claim (verdicts forced to TECHNICAL_ONLY)"),
    "standard": dict(label="Standard research", parents=FRESH_PARENTS[:4] + HIST_PARENTS, screen_seeds=SCREEN["seeds"], n_D=4, arms=ARMSET, stability=False, trackb=True,
                     claims_can="parent-level Track A verdicts if >= 4 parents are evaluable per base (fresh-only verdicts need >= 4 fresh parents); Track B indicative (< 6 parents)",
                     claims_cannot="persistence beyond 4000 steps; generalisation to other architectures/tasks"),
    "extended": dict(label="Extended research", parents=FRESH_PARENTS[:8] + HIST_PARENTS, screen_seeds=SCREEN["seeds"], n_D=6, arms=ARMSET, stability=True, trackb=True,
                     claims_can="as standard, with 6 data orders per unit, fresh-only verdicts, and the stability continuations (low-LR + full-LR control) for anchors",
                     claims_cannot="other architectures/tasks"),
    "high": dict(label="High-compute research", parents=FRESH_PARENTS + HIST_PARENTS, screen_seeds=SCREEN["seeds"], n_D=6, arms=ARMSET + list(HIGH_EXTRA_ARMS),
                 stability="all_arms", trackb=True, claims_can="as extended, plus exploratory QK/OV splits and interpolation, stability for every arm",
                 claims_cannot="other architectures/tasks (no validated alternative architecture exists in COR)"),
}


def effective(preset):
    """the protocol as it applies to one preset (only 'smoke' shortens schedules; every scientific preset uses the historical schedule)"""
    P = full(); pre = P["PRESETS"][preset]; o = pre.get("overrides")
    if o:
        P["STEPS"] = dict(o["STEPS"]); P["SCREEN"] = dict(P["SCREEN"], stop=o["SCREEN_stop"]); P["STABILITY_SNAPS"] = o["stability_snaps"]
    else: P["STABILITY_SNAPS"] = [6000, 8000]
    P["D_LIST"] = pre.get("D_list") or P["CONF_D"][: pre["n_D"]]
    P["PRESET"] = preset; P["TECHNICAL_ONLY"] = preset in ("smoke", "pilot"); return P


def full():
    return copy.deepcopy(dict(version="XX-2", HIST_PARENTS=HIST_PARENTS, PREVIOUSLY_OBSERVED=PREVIOUSLY_OBSERVED, FRESH_PARENTS=FRESH_PARENTS, PILOT_PARENTS=PILOT_PARENTS,
                              DEV_PARENT=DEV_PARENT, PARENT_VALIDITY=PARENT_VALIDITY, PRETRAIN=PRETRAIN, SCREEN=SCREEN, COMP=COMP, CLASS=CLASS, CONF_D=CONF_D, PILOT_D=PILOT_D,
                              STREAMS=STREAMS, STEPS=STEPS, WINDOW=WINDOW, ALLG=ALLG, ARMS=ARMS, HIGH_EXTRA_ARMS=HIGH_EXTRA_ARMS, VALIDITY=VALIDITY, BAND=BAND, DELTA=DELTA, EPS=EPS,
                              EPS_SENS=EPS_SENS, MIN_PAIRS_PER_PARENT=MIN_PAIRS_PER_PARENT, MIN_PARENTS=MIN_PARENTS, PARENT_RULE=PARENT_RULE, CROSS_RULE=CROSS_RULE,
                              HYPOTHESES_A=HYPOTHESES_A, EXPLORATORY_A=EXPLORATORY_A, TRACK_B=TRACK_B, TRACK_C=TRACK_C, ARMSET=ARMSET, PRESETS=PRESETS))
