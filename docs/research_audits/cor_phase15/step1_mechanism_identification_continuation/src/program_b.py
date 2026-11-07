"""Program B (§7): moving Q/K evidence gates with K = 3 candidates.

Construction qk3 (d = 16, positions A, B, R; causal):
  layer 1: R attends to A (marker), copies A's evidence z_A onto fQ_k (query feature) and onto fU_k
           (A's direct vote, weight wU).
  layer 2: head G: query reads fQ_k at R, key reads fB_k at B, so s_RB = lam <zA~, zB~>/sqrt(dh)
           (zA~, zB~ = LN-scaled evidence; s_RA = s_RR = 0); value reads fB_k and writes kB fU_k.
  optional: competing head H' (key reads fB, value writes -kC fU), rereading MLP on fU, final LN.
  logits_k = fU_k . h_R.
Experiments:
  B1  bilinear impossibility for K>=3 (proof in doc) + best bilinear/linear classifier of argmax
      agreement vs an MLP (sklearn, fresh seeds).
  B2  routing tracks <zA,zB>, not argmax agreement: P(routing suppressed | agree) etc.
  B3  total-margin theorem: sign of the routing effect on the k*-vs-j margin predicted from values.
  B4  causal tests: recomputed source change, Q-only, K-only, V-only patches, routing ablation,
      support-preserving and agreement-changing source changes.
  B5  attacks: competing head, rereading MLP, final LN with bias.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "step1_causal_bridge_deep_investigation" / "src"))
import cbmath  # noqa: E402
from cbmath import Layer, Model  # noqa: E402
from semantics import sample_k, semantic_agreement  # noqa: E402

LOG, OUT = [], {}


def log(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.append(s)


def qk3(lam=3.0, L1=12.0, kappa=1.5, mu=3.0, rho=3.0, wQ=2.0, wU=0.3, kB=1.5, kC=0.0, reread=0.0,
        final_ln=False, bias=0.0, qk_off=False, d=16, eps=1e-5):
    Bs = cbmath.mean_zero_basis(d)
    fA, fB, fQ, fU = Bs[0:3], Bs[3:6], Bs[6:9], Bs[9:12]
    anS, anR, mkA = Bs[12], Bs[13], Bs[14]
    dh = 3
    # layer 1: one head, R -> A
    WQ1 = np.zeros((1, d, dh)); WQ1[0, :, 0] = L1 * anR
    WK1 = np.zeros((1, d, dh)); WK1[0, :, 0] = mkA
    WV1 = np.zeros((1, d, dh)); WO1 = np.zeros((1, dh, d))
    for k in range(3):
        WV1[0, :, k] = fA[k]
        WO1[0, k] = wQ * fQ[k] + wU * fU[k]
    lay1 = Layer(WQ=WQ1, WK=WK1, WV=WV1, WO=WO1, bO=np.zeros(d), Win=np.zeros((d, 1)), bin=np.zeros(1),
                 Wout=np.zeros((1, d)), bout=np.zeros(d), mlp_on=False)
    # layer 2: gate head G (+ optional competing head C)
    H = 2 if kC else 1
    WQ2 = np.zeros((H, d, dh)); WK2 = np.zeros((H, d, dh)); WV2 = np.zeros((H, d, dh)); WO2 = np.zeros((H, dh, d))
    for k in range(3):
        WQ2[0, :, k] = (0.0 if qk_off else lam) * fQ[k]
        WK2[0, :, k] = fB[k]
        WV2[0, :, k] = fB[k]
        WO2[0, k] = kB * fU[k]
        if kC:
            # competing head: uniform routing (no query), reads B's evidence, writes it with opposite sign
            WV2[1, :, k] = fB[k]
            WO2[1, k] = -kC * fU[k]
    s = fU.sum(0) / math.sqrt(3)
    lay2 = Layer(WQ=WQ2, WK=WK2, WV=WV2, WO=WO2, bO=np.zeros(d), act="relu",
                 Win=np.column_stack([fU[k] for k in range(3)]) if reread else np.zeros((d, 1)),
                 bin=-0.3 * np.ones(3) if reread else np.zeros(1),
                 Wout=np.stack([-reread * fU[k] for k in range(3)]) if reread else np.zeros((1, d)),
                 bout=np.zeros(d), mlp_on=bool(reread))
    WU = np.stack([fU[k] for k in range(3)], axis=1)
    bf = bias * fU[0] if bias else np.zeros(d)
    model = Model(layers=[lay1, lay2], WU=WU, bU=np.zeros(3), final_ln=final_ln, bf=bf, eps=eps)

    def embed(zA, zB):
        A = zA @ fA + kappa * anS + mu * mkA
        B = zB @ fB + kappa * anS
        R = rho * anR
        return np.stack([A, B, R])
    return model, embed


def run(model, emb, zA, zB, hook=None):
    return cbmath.forward(model, emb(zA, zB), 2, hook)


def main():
    rng = np.random.default_rng(41)
    bk = sample_k("T5b", 3000, rng)
    zA, zB = bk.zA, bk.zB
    sem = semantic_agreement(zA, zB).astype(bool)

    # B1 best bilinear+linear vs MLP classifier of argmax agreement
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    tr = sample_k("T5b", 20000, np.random.default_rng(42)); te = sample_k("T5b", 20000, np.random.default_rng(43))
    def feats(b):
        return np.concatenate([np.einsum("ni,nj->nij", b.zA, b.zB).reshape(len(b.zA), -1), b.zA, b.zB], 1)
    ytr, yte = semantic_agreement(tr.zA, tr.zB), semantic_agreement(te.zA, te.zB)
    lr = LogisticRegression(C=100.0, max_iter=5000).fit(feats(tr), ytr)
    mlp = MLPClassifier((64, 64), max_iter=600, random_state=0).fit(np.concatenate([tr.zA, tr.zB], 1), ytr)
    acc_bil = float(lr.score(feats(te), yte)); acc_mlp = float(mlp.score(np.concatenate([te.zA, te.zB], 1), yte))
    # binary analogue: z = (t, -t), agreement = sign(tA tB) is bilinear-exact
    tA, tB = np.random.default_rng(44).uniform(-3, 3, (2, 20000))
    acc_bin = float(((tA * tB > 0) == (tA * tB > 0)).mean())
    log(f"B1 argmax-agreement classifiers (train seed 42, test seed 43): best bilinear+linear sign {acc_bil:.4f}; "
        f"MLP(64,64) {acc_mlp:.4f}; base rate {max(yte.mean(), 1 - yte.mean()):.4f}. K=2 bilinear sign(tA tB) exact: {acc_bin}")
    OUT["B1"] = dict(bilinear=acc_bil, mlp=acc_mlp, base=float(max(yte.mean(), 1 - yte.mean())))

    # B2 routing vs relation
    m, emb = qk3()
    aB, dots, sB = [], [], []
    for i in range(len(zA)):
        t = run(m, emb, zA[i], zB[i])
        aB.append(t.pattern[1][0][2, 1]); sB.append(t.scores[1][0][2, 1]); dots.append(zA[i] @ zB[i])
    aB, dots, sB = map(np.array, (aB, dots, sB))
    uni = 1 / 3
    res = dict(sign_score_eq_sign_dot=float((np.sign(sB) == np.sign(dots)).mean()),
               P_routing_suppressed_given_agree=float((aB[sem] < uni).mean()),
               P_routing_enhanced_given_disagree=float((aB[~sem] > uni).mean()))
    log("B2 routing to B in qk3 (3000 T5b draws, seed 41):", json.dumps(res))
    OUT["B2"] = res

    # B3 total-margin theorem: effect of routing (vs uniform-routing ablation) on margin k*-j, k* = argmax zA
    mo, embo = qk3(qk_off=True)
    hit = tot = 0
    for i in range(len(zA)):
        k = int(zA[i].argmax()); j = int(np.argsort(zA[i])[-2])
        t = run(m, emb, zA[i], zB[i]); to = run(mo, embo, zA[i], zB[i])
        dm = (t.logits[k] - t.logits[j]) - (to.logits[k] - to.logits[j])
        # prediction from frozen values: values at A and R are zero on fU; B's value margin is kB(zB~_k - zB~_j)
        VB = t.VO[1][0][1]  # value-output at position B, layer 2, head 0
        vm = float(VB @ (m.WU[:, k] - m.WU[:, j]))
        pred = np.sign(t.pattern[1][0][2, 1] - to.pattern[1][0][2, 1]) * np.sign(vm)
        if abs(dm) > 1e-9:
            tot += 1; hit += pred == np.sign(dm)
    log(f"B3 total-margin theorem (no rereader, no competing writer, no final LN): predicted sign of routing "
        f"effect on margin(k*, runner-up) correct {hit}/{tot}")
    OUT["B3"] = dict(hit=int(hit), n=int(tot))

    # B4 causal tests on a fixed agree/disagree example pair
    za = np.array([2.0, 0.0, 0.0]); zb_ag = np.array([2.0, 0.0, 0.0]); zb_dis = np.array([0.0, 2.0, 0.0])
    def mk(t):
        return float(t.logits[0] - t.logits[1])
    base = run(m, emb, za, zb_dis); cf = run(m, emb, za, zb_ag)
    res = {"recomputed source change (zB dis->ag)": (mk(base), mk(cf))}
    for site in ["Q0", "K0", "V0"]:
        arr = getattr(cf, site[0])[1][0]
        res[f"{site}-only patch (layer 2) from agree run"] = mk(run(m, emb, za, zb_dis, cbmath.replace_site(1, site, arr)))
    res["routing ablation (uniform) on agree run"] = mk(run(mo, embo, za, zb_ag))
    # support-preserving change: scale zB (argmax kept), agreement kept
    res["support-preserving zB x0.25 (agree)"] = mk(run(m, emb, za, 0.25 * zb_ag))
    # agreement-changing at equal raw dot: zB = (0, -2, 0) has <zA,zB> = 0 like zb_dis but different support
    res["agreement-changing equal-dot zB=(0,-2,0)"] = mk(run(m, emb, za, np.array([0.0, -2.0, 0.0])))
    # negative dot but same top candidate
    zA2 = np.array([1.0, -3.0, 0.0]); zB2 = np.array([1.0, 0.0, -3.0])
    t2 = run(m, emb, zA2, zB2)
    res["agree-on-0 with <zA,zB>=+1 : alpha_B"] = float(t2.pattern[1][0][2, 1])
    zA3 = np.array([1.0, -3.0, 0.5]); zB3 = np.array([1.0, 0.5, -3.0])
    t3 = run(m, emb, zA3, zB3)
    res["agree-on-0 with <zA,zB>=-1.75 : alpha_B"] = float(t3.pattern[1][0][2, 1])
    res["  margin(0 vs 2) full vs uniform"] = (float(t3.logits[0] - t3.logits[2]),
                                              float(run(mo, embo, zA3, zB3).logits[0] - run(mo, embo, zA3, zB3).logits[2]))
    for k_, v_ in res.items():
        log(f"B4 {k_}: {v_}")
    OUT["B4"] = res

    # B5 attacks
    log("B5 attacks on the total-margin conclusion (agree pair zA=zB=(2,0,0) vs disagree zB=(0,2,0); margin 0 vs 1)")
    att = {}
    for name, kw in [("baseline", {}), ("competing head kC=3", dict(kC=3.0)), ("rereading MLP 3.0", dict(reread=3.0)),
                     ("final LN, bias 0", dict(final_ln=True)), ("final LN, bias +2 on fU_0", dict(final_ln=True, bias=2.0))]:
        mm, ee = qk3(**kw)
        ag, di = mk(run(mm, ee, za, zb_ag)), mk(run(mm, ee, za, zb_dis))
        mmo, eeo = qk3(qk_off=True, **kw)
        ago = mk(run(mmo, eeo, za, zb_ag))
        att[name] = dict(agree=ag, disagree=di, agree_uniform=ago, routing_effect_on_agree=ag - ago)
        log(f"   {name:28s} m(agree)={ag:+.4f} m(disagree)={di:+.4f}  routing effect on agree run {ag - ago:+.4f}")
    sweep = {}
    for kC in [0.0, 1.0, 2.0, 3.0, 4.0, 6.0]:
        mm, ee = qk3(kC=kC) if kC else qk3()
        sweep[kC] = mk(run(mm, ee, za, zb_ag)) - mk(run(mm, ee, za, zb_dis))
    log("   competing-head sweep, m(agree) - m(disagree) vs kC: " + ", ".join(f"{k}: {v:+.4f}" for k, v in sweep.items()))
    att["kC_sweep"] = sweep
    OUT["B5"] = att

    root = HERE.parent
    (root / "logs" / "program_b.txt").write_text("\n".join(LOG) + "\n")
    (root / "logs" / "program_b.json").write_text(json.dumps(OUT, indent=1, default=float))


if __name__ == "__main__":
    main()
