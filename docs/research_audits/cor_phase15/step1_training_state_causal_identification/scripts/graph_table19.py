"""Machine-readable graph/gradient table (mandate stage 1) from logs/graph_audit19.json.  Columns per base x state x mode:
forward_changed (max |dlogit| > 1e-5), d loss/d bias on T edges and on complement edges, and for each parameter class whether the
gradient differs from plain training (rel_diff > 1e-4): routing (target-layer WQ/WK), value (target-layer WV), target-layer other,
upstream layers, shared embeddings, downstream (layers above target + readout WA/bf/gf), plus max one-step update and Adam-state rel diffs.
Note: since a softmax is invariant to a constant logit shift, d loss/d bias_complement = -d loss/d bias_T under plain training; a mode that
stops gradient only on T logits therefore leaves the complement-side routing signal intact."""
import json, csv, os, re
H = os.path.dirname(os.path.abspath(__file__)); J = json.load(open(os.path.join(H, "..", "logs", "graph_audit19.json")))
LAY = lambda n: (-1 if n.startswith("emb") else (99 if n in ("WA", "bf", "gf") else (int(n.split("/")[1]) if n.startswith("pre") else 2)))
rows = []
for base, B in J.items():
    tl = B["target_layer"]
    for st, S in B["states"].items():
        for mode, m in S.items():
            ch = {n for n, t in m["tensors"].items() if t["rel_diff"] > 1e-4}
            cls = dict(routing=set(), value=set(), target_other=set(), upstream=set(), shared=set(), downstream=set())
            for n in m["tensors"]:
                L = LAY(n); leaf = n.split("/")[-1]
                k = "shared" if L == -1 else ("downstream" if L > tl else ("upstream" if L < tl else ("routing" if leaf in ("WQ", "WK") else ("value" if leaf == "WV" else "target_other"))))
                cls[k].add(n)
            r = dict(base=base, state=st, mode=mode, forward_changed=m["forward_max_logit_diff"] > 1e-5, forward_max_logit_diff=m["forward_max_logit_diff"],
                     dloss_dbias_T=m["dloss_dbias_T"], dloss_dbias_complement=m["dloss_dbias_complement"], n_tensors_grad_changed=len(ch))
            for k, s in cls.items(): r[f"grad_changed_{k}"] = f"{len(s & ch)}/{len(s)}"
            r["update_rel_diff_max"] = max(m["update_rel_diff"].values()) if "update_rel_diff" in m else None; r["optstate_rel_diff_max"] = m.get("optstate_rel_diff_max")
            for fld in ("step_update_rel_diff", "step_mu_rel_diff", "step_nu_rel_diff"):        # v2: per component class, the ACTUAL training step
                if fld in m:
                    for k, s_ in cls.items(): r[f"{fld[5:]}_{k}"] = max((m[fld][n] for n in s_), default=None)
            if "pnz_noise_norm_over_plain_update" in m: r["pnz_noise_over_update_max"] = max(m["pnz_noise_norm_over_plain_update"].values()); r["pnz_noise_over_update_min"] = min(m["pnz_noise_norm_over_plain_update"].values())
            rows.append(r)
out = os.path.join(H, "..", "logs", "graph_table19.csv")
keys = []
for r in rows: keys += [k for k in r if k not in keys]
with open(out, "w", newline="") as f: w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
json.dump(rows, open(out.replace(".csv", ".json"), "w"), indent=1)
for r in rows:
    if r["state"] == "step0": print(r["base"], r["mode"].ljust(6), "fwd" if r["forward_changed"] else "   ", "dT=%+.4f dC=%+.4f" % (r["dloss_dbias_T"], r["dloss_dbias_complement"]), *[f"{k[13:]}={r[k]}" for k in r if k.startswith("grad_changed")])
