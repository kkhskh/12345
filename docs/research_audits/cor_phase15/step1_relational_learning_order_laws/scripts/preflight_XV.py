"""XV preflight of score_XV.py on SYNTHETIC trajectories only (no confirmation output exists or is read).
P0 refuses on the real (empty) state; P1 a race-pattern scenario -> P1 COMPLEXITY_DELAYS, P2 INCONCLUSIVE, P3 RACE;
P2 an H2-pattern scenario -> P3 H2_PAYOFF, P2 BENEFIT_SPEEDS; P3 a no-effect scenario -> P1 NO_EFFECT, P3 H1_COMPLEXITY_FIRST;
P4 a missing trajectory -> refused; P5 a done.json with the wrong seed -> refused.  Writes logs/preflight_XV.json."""
import json, os, sys, subprocess, tempfile, shutil, hashlib
HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(PKG, "..", "step1_foundational_atlas_continuation", "src"))
import numpy as np, jax, train_tasks as TT
D = json.load(open(os.path.join(PKG, "logs", "XV_design.json"))); SN = list(range(25, 2001, 25)) + list(range(2250, 4001, 250))
IH = {s: hashlib.sha256(b"".join(np.asarray(x).tobytes() for x in jax.tree_util.tree_leaves(TT.init("gate", s)))).hexdigest() for s in D["seeds"]}
def run(root):
    o = os.path.join(tempfile.mkdtemp(), "s.json"); p = subprocess.run([sys.executable, os.path.join(HERE, "score_XV.py"), root, o], capture_output=True, text=True)
    return p.returncode, (json.load(open(o)) if os.path.exists(o) else None)
def build(sc):
    root = tempfile.mkdtemp(); shutil.copytree(os.path.join(PKG, "logs"), os.path.join(root, "logs")); shutil.copytree(os.path.join(PKG, "scripts"), os.path.join(root, "scripts")); shutil.copytree(os.path.join(PKG, "src"), os.path.join(root, "src"))
    for sib in ("step1_foundational_atlas_continuation", "step1_cross_family_relational_foundations", "step1_relational_mechanism_deepening", "step1_latent_reliability"):
        t = os.path.join(os.path.dirname(root), sib)
        if not os.path.exists(t): os.symlink(os.path.abspath(os.path.join(PKG, "..", sib)), t)
    for r in D["runs"]:
        d = os.path.join(root, r["dir"]); os.makedirs(d)
        for k in SN: open(os.path.join(d, f"step{k}.npz"), "w").close()
        json.dump(dict(cond=r["spec"], seed=r["seed"], stop=4000, blind=True, init_sha256=IH[r["seed"]]), open(os.path.join(d, "done.json"), "w"))
        tI, tG = sc[r["cond"]]; tr = []
        for k in SN:
            G = 0.0 if (tG is None or k < tG) else 1.0; A = 0.8 if (tI is not None and k >= tI and G <= 0.1) else 0.05
            tr.append(dict(step=k, NLL=0.3, acc=0.9 if G > 0.5 else 0.8, KL_bayes=0.0 if k == 4000 else 0.3, G=G if k < 4000 else (1.0 if tG is not None else 0.0), A_p=A))
        json.dump(dict(cond=r["spec"], stream=D["stream"], n=D["n_episodes"], refs=dict(order1=dict(acc=0.8)), traj=tr), open(os.path.join(root, r["traj"]), "w"))
    return root
res = {}
rc, o = run(PKG); res["P0_refuse_real"] = rc == 2 and o is None
rc, o = run(build(dict(A=(200, 600), B=(None, 425), C=(400, 1700), D=(200, 2000)))); res["P1_race"] = rc == 0 and o["verdicts"] == dict(P1="COMPLEXITY_DELAYS", P2="INCONCLUSIVE", P3="RACE"); res["P1_detail"] = o and o["verdicts"]
rc, o = run(build(dict(A=(200, 900), B=(None, 425), C=(None, 1500), D=(200, 3000)))); res["P2_h2"] = rc == 0 and o["verdicts"]["P3"] == "H2_PAYOFF" and o["verdicts"]["P2"] == "BENEFIT_SPEEDS"; res["P2_detail"] = o and o["verdicts"]
rc, o = run(build(dict(A=(100, 500), B=(100, 500), C=(100, 520), D=(100, 510)))); res["P3_noeffect"] = rc == 0 and o["verdicts"]["P1"] == "NO_EFFECT" and o["verdicts"]["P3"] == "H1_COMPLEXITY_FIRST"; res["P3_detail"] = o and o["verdicts"]
root = build(dict(A=(200, 600), B=(None, 425), C=(400, 1700), D=(200, 2000))); os.remove(os.path.join(root, D["runs"][5]["traj"])); rc, o = run(root); res["P4_missing_traj"] = rc == 2 and o is None
root = build(dict(A=(200, 600), B=(None, 425), C=(400, 1700), D=(200, 2000))); f = os.path.join(root, D["runs"][3]["dir"], "done.json"); j = json.load(open(f)); j["seed"] = 999; json.dump(j, open(f, "w")); rc, o = run(root); res["P5_wrong_seed"] = rc == 2 and o is None
# score_T4: refusal on real state; synthetic pass (source stage then gate) -> P6 CONFIRMED; synthetic no-source-stage -> FALSIFIED
def runT4(root):
    o = os.path.join(tempfile.mkdtemp(), "t.json"); p = subprocess.run([sys.executable, os.path.join(HERE, "score_T4.py"), root, o], capture_output=True, text=True)
    return p.returncode, (json.load(open(o)) if os.path.exists(o) else None)
rc, o = runT4(PKG); res["T4_P0_refuse_real"] = rc == 2 and o is None
sys.path.insert(0, os.path.join(PKG, "..", "step1_latent_reliability", "src")); import nets as NN, bayes as BB
SN4 = list(range(50, 3001, 50)) + list(range(3250, 6001, 250))
def buildT4(src_stage):
    root = build(dict(A=(200, 600), B=(None, 425), C=(400, 1700), D=(200, 2000)))
    for s in (301, 302, 303, 304):
        d = os.path.join(root, "confirm_outputs", f"T4ds_s{s}"); os.makedirs(d, exist_ok=True)
        for k in SN4: open(os.path.join(d, f"step{k}.npz"), "w").close()
        ih = hashlib.sha256(b"".join(np.asarray(x).tobytes() for x in jax.tree_util.tree_leaves(NN.init_base("deepsets", "M", BB.Env("T4", K=3, Q=6), s)))).hexdigest()
        json.dump(dict(seed=s, steps=6000, blind=True, init_sha256=ih), open(os.path.join(d, "done.json"), "w"))
        tr = [dict(step=k, acc=0.7, KL_bayes=0.001 if k > 800 else 0.3, KL_own=(0.01 if (src_stage and k < 800) else 0.2), G4=(0.8 if k >= 800 else 0.0)) for k in SN4]
        json.dump(dict(stream=98_800_000, n=3000, traj=tr), open(os.path.join(root, "confirm_outputs", f"traj_T4ds_s{s}.json"), "w"))
    return root
rc, o = runT4(buildT4(True)); res["T4_P1_confirm"] = rc == 0 and o["verdicts"]["P6"] == "CONFIRMED"
rc, o = runT4(buildT4(False)); res["T4_P2_falsify"] = rc == 0 and o["verdicts"]["P6"] == "FALSIFIED"
json.dump(res, open(os.path.join(PKG, "logs", "preflight_XV.json"), "w"), indent=1); print(res); sys.exit(0 if all(v for k, v in res.items() if not k.endswith("detail")) else 1)
