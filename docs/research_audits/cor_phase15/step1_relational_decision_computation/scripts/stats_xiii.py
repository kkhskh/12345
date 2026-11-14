"""Parent-level statistics for Investigation XIII (confirmation cohort, or dev with --dev SFX UNITSFILE).
Per unit: the key metric of every prediction arm; across units: mean, SD, 95% t-interval (n >= 2), min, max -- separately for all units, complete,
mixture-class.  Episode-level CIs where the analysis provides them (DiD bootstrap) are carried through.  -> logs/stats_xiii<SFX>.json and .md"""
import json, os, sys, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); L = os.path.join(HERE, "..", "logs")
import numpy as np
DEV = "--dev" in sys.argv
sfx = sys.argv[sys.argv.index("--dev") + 1] if DEV else "_conf"
S = json.load(open(os.path.join(L, "score_xiii" + (sfx if DEV else "") + ".json")))
def J(n):
    f = os.path.join(L, f"{n}{sfx}.json"); return json.load(open(f)) if os.path.exists(f) else {}
D = {k: J(k) for k in ("l2_scan", "l2_mix_causal", "l2_mediation", "score_contrast", "l1abl_mix", "decoy_disc")}
def ad(sp):
    p, s_, sd, suf = sp.split(":"); f = os.path.join(L, "adapt_sel" + sfx, f"{p}_{s_}_s4000_seed{sd}{suf}.json"); return json.load(open(f)) if os.path.exists(f) else None
def g(o, path):
    for k in path.split("."):
        if not isinstance(o, dict): return None
        o = o.get(k)
    return o
METRICS = [("P1 R2 Bayes", "l2_mix_causal", "donor.bayes_resp.R2"), ("P1 slope", "l2_mix_causal", "donor.bayes_resp.slope"), ("P1 R2 A3", "l2_mix_causal", "donor.A3_resp.R2"),
           ("P1 R2 shuffled", "l2_mix_causal", "donor.shuffled_pred.R2"), ("P1c R2 Bayes (E!=A1)", "l2_mix_causal", "donor_EneA1.bayes_resp.R2"), ("P1c R2 A1 (E!=A1)", "l2_mix_causal", "donor_EneA1.A1_resp.R2"),
           ("P2 TRUE |NIE|", "l2_mediation", "TRUE.mean_mag_NIE"), ("P2 CTRL |NIE|", "l2_mediation", "CTRL.mean_mag_NIE"), ("P2 decoy |NIE|", "l2_mediation", "TRUE.mean_mag_NIE_decoy"),
           ("P2 DiD", "l2_mediation", "DiD_mag_NIE_true_minus_ctrl.est"), ("P2 DiD lo", "l2_mediation", "DiD_mag_NIE_true_minus_ctrl.lo"), ("P2 ATT-MIX R2", "l2_mediation", "TRUE.fidelity_ATTMIX.R2"),
           ("P2 NIE proj on TE", "l2_mediation", "TRUE.frac_TE_mediated"),
           ("P3 E removed", "score_contrast", "E_mean.frac_S1_removed"), ("P3 random removed", "score_contrast", "random6_mean.frac_S1_removed"), ("P3 complement removed", "score_contrast", "complement_mean.frac_S1_removed"),
           ("P3 A1-decoy removed", "score_contrast", "decoy_A1_mean.frac_S1_removed"), ("P3 PERM-decoy removed", "score_contrast", "decoy_PERM_mean.frac_S1_removed"),
           ("P4 corr intact", "l1abl_mix", "intact.B"), ("P4 corr L1-ablated", "l1abl_mix", "layer1_mean_ablated.B"), ("P4 e decode complement", "l1abl_mix", "decode.e_g0.complement"),
           ("P4 decision decode complement", "l1abl_mix", "decode.bayes_decision.complement"), ("P4 decision decode E", "l1abl_mix", "decode.bayes_decision.E"),
           ("P8 E-sub R2(E) E!=A1", "decoy_disc", "E.EvA1.E.R2"), ("P8 E-sub R2(A1) E!=A1", "decoy_disc", "E.EvA1.A1.R2"), ("P8 A1-sub R2(A1) E!=A1", "decoy_disc", "A1.EvA1.A1.R2"),
           ("P8 A1-sub R2(E) E!=A1", "decoy_disc", "A1.EvA1.E.R2"), ("P8 E-sub R2(E) all", "decoy_disc", "E.E.R2"), ("P8 E-sub R2(PERM) all", "decoy_disc", "E.PERM.R2"), ("P8 random R2(E)", "decoy_disc", "random.E.R2"),
           ("resid2 logit share", "l2_scan", "logit_var_share.resid2")]
AD = [("P5 Bayes acc", "bayes_acc"), ("P5 native acc", "native.acc"), ("P5 readout acc", "readout.acc"), ("P5 readout S3 acc", "readout.acc_S3"), ("P5 readout KL", "readout.KL_true"),
      ("P5 attn acc", "attn.acc"), ("P5 attn S3 acc", "attn.acc_S3"), ("P5 mlp acc", "mlp.acc"), ("P5 mlp S3 acc", "mlp.acc_S3"), ("P5 full acc", "full.acc"), ("P5 full KL", "full.KL_true"), ("P5 fresh acc", "fresh.acc")]
units = S["units"]; comp = S["complete"]; mix = [u for u in comp if S["mixture_class"].get(u)]
rows = {}
for u in units:
    r = {nm: g(D[src].get(u), path) for nm, src, path in METRICS}; a = ad(u); r.update({nm: g(a, path) for nm, path in AD}); rows[u] = r
def summ(vals):
    v = np.array([x for x in vals if isinstance(x, (int, float))], float)
    if not len(v): return None
    o = dict(n=int(len(v)), mean=float(v.mean()), min=float(v.min()), max=float(v.max()))
    if len(v) >= 2:
        from scipy import stats as SS
        sd = float(v.std(ddof=1)); h = float(SS.t.ppf(0.975, len(v) - 1) * sd / math.sqrt(len(v))); o.update(sd=sd, ci95=[o["mean"] - h, o["mean"] + h])
    return o
agg = {grp: {nm: summ([rows[u][nm] for u in us]) for nm in rows[units[0]]} if us else {} for grp, us in (("complete", comp), ("mixture_complete", mix), ("all", units))}
out = dict(units=units, complete=comp, mixture_complete=mix, per_unit=rows, aggregate=agg)
json.dump(out, open(os.path.join(L, f"stats_xiii{sfx}.json"), "w"), indent=1)
md = ["| metric | " + " | ".join(u.split(":")[0] for u in units) + " | mean [95% CI] (complete; mixture-class for mechanism rows) |", "|---|" + "---|" * (len(units) + 1)]
for nm in rows[units[0]]:
    grp = "mixture_complete" if not nm.startswith("P5") and not nm.startswith("P8") and nm != "resid2 logit share" else "complete"
    a = agg[grp].get(nm) if agg[grp] else None
    cell = "–" if not a else (f"{a['mean']:.3f}" + (f" [{a['ci95'][0]:.3f}, {a['ci95'][1]:.3f}]" if "ci95" in a else "") + f" (n={a['n']})")
    md.append(f"| {nm} | " + " | ".join("–" if rows[u][nm] is None else f"{rows[u][nm]:.3f}" for u in units) + f" | {cell} |")
open(os.path.join(L, f"stats_xiii{sfx}.md"), "w").write("\n".join(md) + "\n"); print("\n".join(md[:6]))
