"""Post-review (R2 FATAL-1, R5 H1, R7 point 1): an identification criterion that separates a RELATION gate
from an additive (per-source) gate with the same decisions and the same relation-blind pathway.

Circuits:
  C'    : cbmath.rereading()  (block-1 MLP computes q = h(c(u+v)) - h(c(u-v)), sign q = sign(uv))
  TWIN  : C' with block-1 MLP replaced by per-source units writing an AND-of-supports gate feature
          (R2's construction: q' = lam/2 (phi(u) + phi(v)) - lam/2, phi = clamp(k t, -1, 1); no unit reads both)
  RF    : relation_free_equivalent.relation_free()  (fully additive margin)
Tests:
  S1  mediator sign table: sign of the gate feature at R in the four sign quadrants (generator-level).
      Relation (D1, candidate-symmetric) iff (+,+)>0, (-,-)>0, (+,-)<0, (-,+)<0.
  S2  decision-level mediator interchange from an agreement-for-c' run into a conflict run:
      patch R's block-1 MLP output from (-a,-b) into (a,-b). Relation gate -> sources routed (follow sources);
      additive AND gate -> prior.  (off-manifold; labelled)
  S3  graded-output single-cancellation test (T-RF'): count row pairs (a,a') whose margin order reverses across b.
      Any margin h(f(a)+g(b)) with h strictly increasing has 0 reversals.
  X4  the matched relation-blind contrast on all three circuits (shows X4 alone does not identify a relation).
"""
import copy, itertools, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1] / "step1_causal_bridge_deep_investigation" / "src"))
import cbmath
from relation_free_equivalent import relation_free

C, D, E = cbmath.rereading()
e1, e2, e3 = D["e1"], D["e2"], D["e3"]
R = 3
LOG, OUT = [], {}


def log(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.append(s)


def additive_twin(k=20.0):
    m = copy.deepcopy(C); L1 = m.layers[0]
    lam = cbmath.forward(C, E(1, 1), R).mlp_out[0][R] @ e3
    L1.Win = np.column_stack([k * e1, k * e1, k * e2, k * e2]); L1.bin = np.array([1., -1., 1., -1.])
    L1.Wout = np.stack([lam * e3, -lam * e3, lam * e3, -lam * e3]) * 0.5
    L1.bout = -1.5 * lam * e3
    L1.act = "relu"
    return m


def gate_feature(m, a, b, emb=E):
    return float(cbmath.forward(m, emb(a, b), R).mlp_out[0][R] @ e3)


def mg(m, a, b, hook=None, emb=E):
    return cbmath.margin(cbmath.forward(m, emb(a, b), R, hook))


def main():
    T = additive_twin()
    RF, embRF = relation_free()
    g = np.concatenate([-np.linspace(3, 0.1, 30), np.linspace(0.1, 3, 30)])
    same = np.mean([np.sign(mg(C, a, b)) == np.sign(mg(T, a, b)) for a in g for b in g])
    log(f"TWIN vs C' decision agreement on 60x60 grid, |a|,|b| in [0.1,3]: {same:.4f}")
    OUT["twin_decision_agreement"] = float(same)

    # S1
    log("S1 mediator sign table (gate feature at R on e3), magnitudes (1.0, 0.7):")
    tab = {}
    for name, m in (("C'", C), ("TWIN", T)):
        row = {q: gate_feature(m, sa * 1.0, sb * 0.7) for q, (sa, sb) in {"++": (1, 1), "--": (-1, -1), "+-": (1, -1), "-+": (-1, 1)}.items()}
        rel = row["++"] > 0 and row["--"] > 0 and row["+-"] < 0 and row["-+"] < 0
        tab[name] = dict(row, relation=rel)
        log(f"   {name:5s} " + "  ".join(f"{k}:{v:+.4f}" for k, v in row.items()) + f"   -> candidate-symmetric relation: {rel}")
    rng = np.random.default_rng(71)
    pts = rng.uniform(0.1, 3, (500, 2))
    for name, m in (("C'", C), ("TWIN", T)):
        ok = np.mean([(gate_feature(m, a, b) > 0) and (gate_feature(m, -a, -b) > 0) and (gate_feature(m, a, -b) < 0)
                      and (gate_feature(m, -a, b) < 0) for a, b in pts])
        tab[name]["fraction_of_500_magnitude_draws_with_relation_pattern"] = float(ok)
        log(f"   {name}: relation sign pattern holds on {ok:.3f} of 500 random magnitude draws")
    OUT["S1"] = tab

    # S2
    log("S2 interchange R's block-1 MLP output from (-a,-b) into (a,-b) [off-manifold]: decision after patch")
    res = {}
    for name, m in (("C'", C), ("TWIN", T)):
        follows = 0; prior = 0; n = 0
        for a, b in pts[:200]:
            src = cbmath.forward(m, E(-a, -b), R)
            base = cbmath.forward(m, E(a, -b), R)
            val = base.mlp_out[0].copy(); val[R] = src.mlp_out[0][R]
            tr = cbmath.forward(m, E(a, -b), R, cbmath.replace_site(0, "mlp_out", val))
            alpha_src = tr.pattern[1][0][R, 1] + tr.pattern[1][0][R, 2]
            n += 1; follows += alpha_src > 0.5; prior += tr.pattern[1][0][R, 0] > 0.5
        res[name] = dict(P_routes_to_sources=follows / n, P_routes_to_prior=prior / n)
        log(f"   {name:5s} routes to sources {follows / n:.3f}, to prior {prior / n:.3f}")
    OUT["S2"] = res

    # S3 single-cancellation
    log("S3 graded-output single-cancellation test on a 30x30 grid in [-3,3] (|.|>=0.1): row pairs with order reversal")
    gg = np.concatenate([-np.linspace(3, .1, 15), np.linspace(.1, 3, 15)])
    s3 = {}
    for name, f in (("C'", lambda a, b: mg(C, a, b)), ("TWIN", lambda a, b: mg(T, a, b)),
                    ("RF", lambda a, b: mg(RF, a, b, emb=embRF))):
        M = np.array([[f(a, b) for b in gg] for a in gg])
        viol = tot = 0
        for i, j in itertools.combinations(range(len(gg)), 2):
            d = M[i] - M[j]; s = np.sign(d[np.abs(d) > 1e-3])
            tot += 1; viol += (s > 0).any() and (s < 0).any()
        s3[name] = (viol, tot)
        log(f"   {name:5s} reversals {viol}/{tot}")
    OUT["S3"] = s3

    # X4 on all three
    log("X4 matched relation-blind contrast (block-1 MLP removed) on each circuit, pair (1,1) vs (3,-0.2381):")
    x4 = {}
    for name, m, emb in (("C'", C, E), ("TWIN", T, E), ("RF", RF, embRF)):
        mb = copy.deepcopy(m); mb.layers[0].mlp_on = False
        b1, b2 = mg(mb, 1, 1, emb=emb), mg(mb, 3, -0.2381, emb=emb)
        f1, f2 = mg(m, 1, 1, emb=emb), mg(m, 3, -0.2381, emb=emb)
        x4[name] = dict(blind=(b1, b2), full=(f1, f2), flip=bool(np.sign(f1) != np.sign(f2)))
        log(f"   {name:5s} blind {b1:+.4f} / {b2:+.4f}   full {f1:+.4f} / {f2:+.4f}   decision flip {x4[name]['flip']}")
    OUT["X4_all"] = x4
    (HERE.parent / "logs" / "mediator_sign_test.txt").write_text("\n".join(LOG) + "\n")
    (HERE.parent / "logs" / "mediator_sign_test.json").write_text(json.dumps(OUT, indent=1, default=float))


if __name__ == "__main__":
    main()
