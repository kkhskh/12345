"""Frozen uncertainty report for a score19 output (pre-review C10).  Usage: stats19.py SCORE.json OUT.json
Per hypothesis (one base each; never pooled across bases): per-pair value (and, for phi of a single arm, its E- and M-route components),
range, mean, descriptive two-sided 95% t-interval (n-1 df; n <= 3, so wide and descriptive only), k of n pairs inside the band, and the
exact one-sided 95% Clopper-Pearson lower bound on the proportion of matched pairs that behave as predicted (3/3 -> 0.368, 2/2 -> 0.224).
No p-values.  Also the evaluable-pair counts and, for every arm, the per-pair categories."""
import json, sys, math
from scipy import stats
S = json.load(open(sys.argv[1])); out = dict(evaluable_pairs=S.get("evaluable_pairs"), hypotheses={}, categories={},
    measurement_floor=dict(note="phi uses log(nec + 0.01): necessity below ~0.02 nats is treated as route absent; across three independent measurement streams the same model's phi moved by at most 0.008 (pre-review C, logs/scorer_validation19.json); differences between arms smaller than ~0.05 are not interpretable", eps=0.01, stream_phi_max_shift=0.008))
import re
def comps(pn, arm, fam, eps=0.01):
    F = S["pairs"][pn]["families"][fam]; A = F["arms"]; b, i, a = A[F["anchors"]["B"]], A[F["anchors"]["I"]], A[arm]; l = lambda x: math.log(max(x, 0.0) + eps)
    return {r: (l(a[f"nec_{r}"]) - l(b[f"nec_{r}"])) / (l(i[f"nec_{r}"]) - l(b[f"nec_{r}"])) for r in ("E", "M")}
def cp_lower(k, n, a=0.05): return 0.0 if k == 0 else float(stats.beta.ppf(a, k, n - k + 1))
for hid, h in S["hypotheses"].items():
    vals = h["values"]; n = len(vals); r = dict(band=h["band"], n=n, dropped=h["dropped"])
    if h.get("kind", "band") == "band":
        v = [x["v"] for x in vals.values()]; k = sum(h["lo"] <= x["v"] <= h["hi"] and x["class_ok"] for x in vals.values())
        r.update(values={p: x["v"] for p, x in vals.items()}, k_in_band=k, cp_lower95=cp_lower(k, n) if n else None)
        m_ = re.fullmatch(r"phi\['([A-Za-z0-9_+]+)'\]", h.get("expr", ""))
        if m_: r["route_components"] = {p: comps(p, m_.group(1), h.get("family", "T")) for p in vals}
        if n: r.update(min=min(v), max=max(v), mean=sum(v) / n)
        if n >= 2:
            sd = math.sqrt(sum((x - r["mean"]) ** 2 for x in v) / (n - 1)); t = stats.t.ppf(0.975, n - 1); r.update(sd=sd, t95=[r["mean"] - t * sd / math.sqrt(n), r["mean"] + t * sd / math.sqrt(n)])
    else:
        k = sum(x["category"] in h["support"] for x in vals.values()); r.update(values={p: x["category"] for p, x in vals.items()}, k_supported=k, cp_lower95=cp_lower(k, n) if n else None)
    out["hypotheses"][hid] = r
for pn, P in S["pairs"].items():
    for fam, F in P["families"].items():
        for an, A in F["arms"].items(): out["categories"].setdefault(an, {})[pn] = dict(category=A["category"], phi=A["phi"], nec_E=A["nec_E"], nec_M=A["nec_M"], KL=A["KL"], acc=A["acc"], f=A["f"])
json.dump(out, open(sys.argv[2], "w"), indent=1)
for hid, r in out["hypotheses"].items(): print(hid.ljust(28), r["band"].ljust(13), "n", r["n"], "k", r.get("k_in_band", r.get("k_supported")), "CPlb %.2f" % (r["cp_lower95"] or 0), "range", [round(r.get("min", float("nan")), 2), round(r.get("max", float("nan")), 2)])
