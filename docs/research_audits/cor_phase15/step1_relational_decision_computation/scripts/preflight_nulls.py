"""Preflight of the FROZEN scorer against null scenarios (mandate: 'run null controls against the scorer').
Builds synthetic units whose analysis outputs are what each null scenario implies, injects them into logs/<analysis>_null.json (plus the adaptation dir
logs/adapt_sel_null/), runs score_xiii.py on them, and requires that NO intervention arm passes on any null unit.
Null scenarios (values follow from the scenario by construction, not tuned):
  NOOP     every intervention leaves the output unchanged (observed change == 0 -> fidelity slope 0, R2 0; NIE 0; contrast removal 0; no adaptation gain)
  UNIFORM  the model outputs the uniform distribution (all output changes 0; accuracy = chance 1/3)
  RANDOM   interventions produce output changes unrelated to the predictions (R2 ~ 0, slope ~ 0, DiD ~ 0 with CI straddling 0; random-direction removal)
  ALTMECH  the network implements the PERFECT-ONLY (A1) mixture instead of E: fidelity to Bayes(E) < fidelity to A1 on E != A1 donors; DAS subspaces follow A1
The scenarios keep the real attention structure (mixture class) so that the mixture-only arms are evaluated, i.e. this is the hardest case for the scorer.
P6 (exploratory) and P7 (correlational, not an intervention) are reported but excluded from the 'no arm passes' requirement; P7 is expected to pass on
NOOP/UNIFORM/RANDOM because it does not intervene -- recorded as a limitation (P7 can only be supporting evidence)."""
import json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); L = os.path.join(HERE, "..", "logs")
fz = lambda R2=0.0, slope=0.0: dict(R2=R2, slope=slope, corr=0.0)
def unit(kind):
    zero = kind in ("NOOP", "UNIFORM"); rnd = kind == "RANDOM"; alt = kind == "ALTMECH"
    f0 = fz(0.0, 0.0) if zero else (fz(0.01, 0.03) if rnd else None)
    scan = {"logit_var_share": {"resid2": 0.01, "h2.0": 0.0, "h2.1": 0.6, "h2.2": 0.0, "h2.3": 0.3, "mlp2": 0.1},
            "heads": {"0": {"mass_q_groups": 0.2, "corr": {}}, "2": {"mass_q_groups": 0.2, "corr": {}},
                      "1": {"mass_q_groups": 0.95, "corr": {"B": 0.95, "A1": 0.95, "A3": 0.8, "A0": -0.4, "M": 0.55, "W": -0.4}},
                      "3": {"mass_q_groups": 0.95, "corr": {"B": 0.9, "A1": 0.9, "A3": 0.8, "A0": -0.4, "M": 0.55, "W": -0.4}}}}
    if alt: mc = dict(noop_max_abs_change=0.0, donor=dict(bayes_resp=fz(0.85, 0.9), A3_resp=fz(0.83, 0.85), A0_resp=fz(0.1, 0.3), A1_resp=fz(0.97, 0.97), shuffled_pred=fz(0.0, 0.0)),
                      donor_EneA1=dict(n=90, bayes_resp=fz(0.18, 0.5), A1_resp=fz(0.97, 0.97)))
    else: mc = dict(noop_max_abs_change=0.0, donor=dict(bayes_resp=f0, A3_resp=f0, A0_resp=f0, A1_resp=f0, shuffled_pred=f0), donor_EneA1=dict(n=90, bayes_resp=f0, A1_resp=f0))
    nie = 0.0 if zero else 0.05
    med = dict(TRUE=dict(mean_mag_NIE=1.0 if alt else nie, mean_mag_NIE_decoy=nie, fidelity_ATTMIX=fz(0.2 if alt else 0.0)),
               CTRL=dict(mean_mag_NIE=1.0 if alt else nie, mean_mag_NIE_decoy=nie, fidelity_ATTMIX=fz(0.0)),
               DiD_mag_NIE_true_minus_ctrl=dict(est=0.0, lo=-0.05 if not zero else 0.0, hi=0.05 if not zero else 0.0))   # ALTMECH: eligibility edits move A1 too, but TRUE and CTRL (eligibility-preserving) then mediate equally
    rem = 0.0 if zero else 0.02
    sc = {"E_mean": {"frac_S1_removed": 0.05 if alt else rem}, "complement_mean": {"frac_S1_removed": 1.0 if alt else rem}, "random6_mean": {"frac_S1_removed": rem},
          "decoy_A1_mean": {"frac_S1_removed": 1.2 if alt else rem}, "decoy_PERM_mean": {"frac_S1_removed": rem}}
    l1 = dict(intact=dict(B=0.95 if not zero else 0.95, noelig=-0.5), layer1_mean_ablated=dict(B=0.95 if (zero or rnd) else 0.9, noelig=-0.5),
              decode={"e_g0": {"E": 1.0, "complement": 1.0 if alt else 0.6, "majority": 0.64}, "bayes_decision": {"E": 0.4, "complement": 0.45, "majority": 0.39}})
    d = lambda E, A1, P: {"E": fz(E), "A1": fz(A1), "PERM": fz(P)}
    if alt: dd = {"E": dict(d(0.9, 0.95, 0.0), EvA1=d(0.1, 0.4, 0.1)), "A1": dict(d(0.9, 0.95, 0.0), EvA1=d(0.1, 0.4, 0.1)), "random": dict(d(0.02, 0.02, 0.02), EvA1=d(0.0, 0.0, 0.0))}
    else: v = 0.0 if zero else 0.02; dd = {k: dict(d(v, v, v), EvA1=d(v, v, v)) for k in ("E", "A1", "random")}
    acc = 1 / 3
    ad = dict(bayes_acc=0.79, bayes_acc_S3=0.97, bayes_acc_S12=0.78, native=dict(acc=acc, acc_S3=acc, acc_S12=acc, KL_true=1.5, KL_S3=0.7, KL_S12=1.5))
    for c in ("readout", "attn", "mlp", "full", "fresh"):
        ad[c] = dict(ad["native"]) if (zero or rnd) else dict(acc=0.74, acc_S3=0.34, acc_S12=0.78, KL_true=0.05, KL_S3=0.65, KL_S12=0.0)   # ALTMECH: learns the product rule only
        ad[c]["frozen_params_unchanged"] = True
    return dict(l2_scan=scan, l2_mix_causal=mc, l2_mediation=med, score_contrast=sc, l1abl_mix=l1, decoy_disc=dd), ad
names = {k: f"NULL{k}:full_c1:0:" for k in ("NOOP", "UNIFORM", "RANDOM", "ALTMECH")}
files = {}; os.makedirs(os.path.join(L, "adapt_sel_null"), exist_ok=True)
for k, sp in names.items():
    u, ad = unit(k)
    for a, v in u.items(): files.setdefault(a, {})[sp] = v
    json.dump(dict(model=sp, **ad), open(os.path.join(L, "adapt_sel_null", f"NULL{k}_full_c1_s4000_seed0.json"), "w"))
for a, v in files.items(): json.dump(v, open(os.path.join(L, f"{a}_null.json"), "w"))
open(os.path.join(L, "preflight_null_units.txt"), "w").write("\n".join(names.values()) + "\n")
env = dict(os.environ, SCORE_SFX="_null"); env.update({"SCORE_SPEC": sys.argv[1]} if len(sys.argv) > 1 else {})
r = subprocess.run([sys.executable, os.path.join(HERE, "score_xiii.py"), "--dev", os.path.join(L, "preflight_null_units.txt")], env=env, capture_output=True, text=True); print(r.stdout[-3000:], r.stderr[-2000:])
S = json.load(open(os.path.join(L, "score_xiii_null.json"))); bad = []; arm_passes = []
# Requirement (prediction level): no primary intervention prediction would be CONFIRMED by any single null unit (all its arms true on that unit).
# Arm-level passes on null units are recorded: they mark arms that are NOT discriminating against that null on their own.
for pid, p in S["predictions"].items():
    if pid in ("P6", "P7"): continue
    for u in names.values():
        if all(a["units"].get(u) for a in p["arms"].values()): bad.append((pid, u))
    for aid, a in p["arms"].items():
        for u, v in a["units"].items():
            if v and not a["text"].startswith("validity"): arm_passes.append((pid, aid, u))
json.dump(dict(predictions_confirmed_by_a_null_unit=bad, ok=not bad, arm_level_passes_on_null_units=arm_passes,
               P7_passes_on_null_units=[u for u in names.values() if all(a["units"].get(u) for a in S["predictions"]["P7"]["arms"].values())]),
          open(os.path.join(L, "preflight_nulls_result.json"), "w"), indent=1)
print("PREFLIGHT", "OK: no primary prediction is confirmed by any null unit" if not bad else f"FAIL: {bad}"); print("arm-level passes on nulls:", arm_passes)
