import sys, json, numpy as np
PKG = "/home/claude/12345/docs/research_audits/cor_phase15/step1_relational_operator_reuse"
sys.path.insert(0, PKG + "/src"); sys.path.insert(0, PKG + "/../step1_cross_family_relational_foundations/src")
import gatelib as GL, xviislot as XS
conds = {"A3": "3:0.9:0.9:0.3333333333333333", "A2": "2:0.9:0.9:0.05"}; Ds = {k: XS.SlotDiag(c, n=600) for k, c in conds.items()}
out = {}
for name in ["A3_s804", "A2_s705"]:
    A = json.load(open(f"{PKG}/logs/analysis/{name}.json")); D = Ds[name[:2]]; p = GL.load(f"{PKG}/confirm_outputs/{name}/step{A['selected']}.npz")
    i, j = A["pairs"][0]; b = D.score(p); r = D.score(p, (i, j, 0, 2)); f = D.score(p, (i, j, 0, 3))
    o = {k: dict(base=[round(b[k][i], 3), round(b[k][j], 3)], r02=[round(r[k][i], 3), round(r[k][j], 3)], r03=[round(f[k][i], 3), round(f[k][j], 3)]) for k in ("F_G", "KL", "acc", "conf", "A_p")}
    o["other_slots_FG_base_vs_r02"] = [[round(b["F_G"][q], 2), round(r["F_G"][q], 2)] for q in range(6) if q not in (i, j)]
    out[name] = dict(pair=[i, j], **o); print(name, json.dumps(out[name]))
json.dump(out, open("/tmp/revB17/kl.json", "w"), indent=1)
