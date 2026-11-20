"""Track B jobs (v2: parent-level precision matching, review A M9).
  select  RUN_DIR PARENT OUT.json            selection-stream KL of every snapshot in [150, primary] + necessity at the primary step
  match   PARENT OUT.json SEL1.json ...      kappa_p = largest primary-step selection KL among the parent's anchors that are competent and pure
                                             at the primary step (floor 0.015); each anchor's matched step = the EARLIEST snapshot with KL <= kappa_p
  battery RUN_DIR PARENT MATCH.json OUT.json the frozen necessity measurement and the functional battery at the matched step and at the primary
                                             step; decisions on the discriminating set saved to OUT_<level>.npz (for HB_c / HB_c_dir)
The selection stream (91_430_000) is used for nothing else; all tests use their own frozen streams."""
import os, sys, json, re
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from cor_research import env as E
E.configure(); E.legacy_paths()
import numpy as np
from cor_research import protocol as PR
from cor_research.scoring import functional as FN, measure as MS
from cor_research.generators import t4 as T4
from cor_research.models import load as LD


def cfgP(): return PR.effective(os.environ["COR_PRESET"]) if os.environ.get("COR_PRESET") else PR.full() | dict(STEPS=PR.STEPS, PRESET=None)
def quick(P): return P.get("PRESET") == "smoke"


def select(run_dir, parent, out):
    P = cfgP(); mp = P["TRACK_B"]["matching"]; n = 600 if quick(P) else mp["n"]; prim = P["STEPS"]["primary"]
    steps = sorted(int(m.group(1)) for m in (re.match(r"step(\d+)\.npz$", f) for f in os.listdir(run_dir)) if m and 150 <= int(m.group(1)) <= prim)
    X, q, _, _ = T4.sample(n, mp["selection_stream"]); PB = T4.bayes(X, q); I = np.arange(len(q)); import xviiim as XM; sk = {}
    for s in steps: M = LD.load(run_dir, parent, s); sk[s] = FN.kl(PB, XM.RP.sm(LD.forward_fn(M)(X)[I, q]))
    nec = MS.measure(run_dir, parent, prim, P["STREAMS"]["measure_primary"])
    rec = dict(run=os.path.basename(os.path.normpath(run_dir)), parent=parent, selection_KL=sk, primary_step=prim, necessity_primary={k: nec[k] for k in ("KL", "acc", "nec_E", "nec_M", "ckpt_sha256")},
               snapshot_sha256={s: E.sha256(os.path.join(run_dir, f"step{s}.npz")) for s in steps})
    E.atomic_json(rec, out); return rec


def match(parent, out, sels):
    from cor_research.scoring.track_a import cls, comp
    S = [json.load(open(f)) for f in sels]; prim = S[0]["primary_step"]
    ok = [s for s in S if comp(s["necessity_primary"]) and cls(s["necessity_primary"]) != "HYBRID"]
    kappa = max([s["selection_KL"][str(prim)] if str(prim) in s["selection_KL"] else s["selection_KL"][prim] for s in ok] + [0.015]) if ok else None
    res = {}
    for s in S:
        sk = {int(k): v for k, v in s["selection_KL"].items()}
        cand = [st for st in sorted(sk) if kappa is not None and sk[st] <= kappa + 1e-12]
        res[s["run"]] = dict(matched_step=cand[0] if cand else None, KL_at_match=sk[cand[0]] if cand else None, censored=not cand)
    rec = dict(parent=parent, kappa=kappa, n_anchors=len(S), n_competent_pure_at_primary=len(ok), matched=res, rule=PR.TRACK_B["matching"]["rule"])
    E.atomic_json(rec, out); return rec


def battery(run_dir, parent, match_file, out):
    P = cfgP(); cfg = dict(P["TRACK_B"]["sets"]); run = os.path.basename(os.path.normpath(run_dir)); prim = P["STEPS"]["primary"]
    if quick(P): cfg.update(n_nat=1200, n_disc_pool=4000, n_pairs=300)
    mt = json.load(open(match_file))["matched"][run]; rec = dict(run=run, parent=parent, match=mt, analyses={}); os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    for tag, st in (("matched", mt["matched_step"]), ("final", prim)):
        if st is None: continue
        if not os.path.exists(os.path.join(run_dir, f"step{st}.npz")): rec["analyses"][tag] = dict(step=st, missing=True); continue
        nec = MS.measure(run_dir, parent, st, P["STREAMS"]["measure_primary"]); fb = FN.run_checkpoint(run_dir, parent, st, cfg, want_decisions=True); dec = fb.pop("_decisions")
        np.savez_compressed(out[:-5] + f"_{tag}.tmp.npz", dec=dec["dec"], **{f"rule_{k}": v for k, v in dec["rule_dec"].items()}); os.replace(out[:-5] + f"_{tag}.tmp.npz", out[:-5] + f"_{tag}.npz")
        rec["analyses"][tag] = dict(decisions_sha256=E.sha256(out[:-5] + f"_{tag}.npz"), step=st, necessity={k: nec[k] for k in ("KL", "acc", "nec_E", "nec_M", "nec_E_se", "nec_M_se", "ckpt_sha256") if k in nec}, functional=fb)
    rec["code_sha256"] = code_hashes(); E.atomic_json(rec, out); return rec


def code_hashes(): return dict(FN.code_hashes(), **{"trackb_run.py": E.sha256(os.path.abspath(__file__))})


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "select": r = select(sys.argv[2], sys.argv[3], sys.argv[4]); print(json.dumps(dict(run=r["run"], selection_KL={k: round(v, 4) for k, v in r["selection_KL"].items()})))
    elif mode == "match": r = match(sys.argv[2], sys.argv[3], sys.argv[4:]); print(json.dumps(dict(kappa=r["kappa"], matched={k: v["matched_step"] for k, v in r["matched"].items()})))
    elif mode == "battery": r = battery(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]); print(json.dumps(dict(run=r["run"], levels=list(r["analyses"]))))
    else: raise SystemExit(f"unknown mode {mode}")
