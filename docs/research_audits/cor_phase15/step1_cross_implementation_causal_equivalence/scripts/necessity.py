"""Necessity without collateral damage, restoration, redundancy (mandate §7).  Usage: necessity.py PHASE (cal | conf)
Same model sets and fit stream as equiv_run.py; evaluation stream EVAL + 50_000.
Per model and decision q (all six), with the model's primary site and encoder (P: readout 36+q, P_full; T: question q's
six source tokens, T_own):
  MARG   on-support marginalisation of the count variable: n* ~ P(n^{(-q)} | X_q) (counts of 8 generator draws of
         X_{-q} given X_q; exact conditional sampler), a' = a + E(n*) - E(n); output = mean of the 8 softmaxes.
         If the count variable carries all relational information the model uses at that site, the output moves to the
         own-question Bayes posterior; drop fraction = (acc - acc_MARG)/(acc - acc_own_bayes).  Values stay within the
         encoder's image of on-support counts (no mean replacement of whole subspaces).
  controls (same 8 draws, same norm budget):
    null_cross   identical procedure with the matched-capacity wrong-variable encoder (cross-pair agreement counts)
    rotated      the same delta vectors rotated by a fixed random orthogonal matrix per position (norm-preserving)
    wrong_site   P: question q's source tokens with a T_own encoder fitted in the P model; T: readout 36+q with P_full
    all_sources  (T only) marginalise every source token's own leave-k counts (redundancy across questions' tokens)
  SPAN-ABLATE / RESTORE: project the site onto the encoder span U_t (rank <= f) and replace that component by its
    per-position natural mean; then RESTORE by adding back E(n) - mean_n E(n) (true counts) or E(n_other) - mean
    (counts of another episode: must not restore).
Reported: accuracy, drop fraction of the relational gain, KL(own Bayes || output), KL(full Bayes || output), and the
fraction of the natural decision changes, for each condition.  Writes logs/PHASE/necessity.json."""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np
import eqv as E

PH = sys.argv[1]
SETS = {"cal": (["P0", "P1", "T0", "T1"], 12_100_000, 12_250_000), "conf": (["P2", "P3", "P4", "T2", "T10", "T11"], 12_300_000, 12_450_000)}
names, FIT, EVAL = SETS[PH]; NF, NE, ND = 4000, int(os.environ.get("NE", 1000)), 8
OUT = os.path.join(HERE, "..", "logs", PH); os.makedirs(OUT, exist_ok=True)
t0 = time.time(); log = lambda s: print(f"[{time.time() - t0:7.1f}s] {s}", flush=True)
Xf = E.sample(NF, FIT)["X"]; De = E.sample(NE, EVAL); X, Y = De["X"], De["Y"]
post = E.G.posterior(E.F4, X); own, full = post["own_y"], post["post_y"]; n0 = E.loo(X); n0c = E.loo(X, E.CROSS)
draws = {q: [E.cond_pairs(X, q, np.random.default_rng(EVAL + 1000 * q + j)) for j in range(ND)] for q in range(E.Q)}
acc = lambda P: float((P.argmax(-1) == Y).mean())
def kl(p, r): return float((p * (np.log(p + 1e-12) - np.log(r + 1e-12))).sum(-1).mean())
R = dict(phase=PH, models=names, acc_own_bayes=acc(own), acc_bayes=acc(full), per_model={})


def rot(d, seed):
    Qm, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(d, d))); return Qm.astype(np.float32)


for m in names:
    M = E.Model(m); k = m[0] if m[0] in "PT" else "P"; d = M.d
    Af = M.acts(Xf)[0]; A0, L0, _, _ = M.acts(X); P0 = E.softmax(L0); a0 = acc(P0)
    prim = "P_full" if k == "P" else "T_own"
    enc = E.fit_encoder(prim, Xf, Af); encn = E.fit_encoder(prim, Xf, Af, E.CROSS)
    encw = E.fit_encoder("T_own" if k == "P" else "P_full", Xf, Af)
    Rot = {t: rot(d, 77 + t) for t in range(E.T)}
    out = {}
    def summarize(nm, Pq):
        out[nm] = dict(acc=acc(Pq), drop_frac=float((a0 - acc(Pq)) / max(a0 - R["acc_own_bayes"], 1e-9)),
                       KL_own_to_out=kl(own, Pq), KL_full_to_out=kl(full, Pq),
                       frac_decisions_changed=float((Pq.argmax(-1) != P0.argmax(-1)).mean()))
    summarize("clean", P0)
    conds = {c: np.zeros_like(P0) for c in ("marg", "null_cross", "rotated", "wrong_site") + (("all_sources",) if k == "T" else ())}
    for q in range(E.Q):
        pos = E.site_positions(k, q); wpos = E.site_positions("T" if k == "P" else "P", q)
        acc_c = {c: 0 for c in conds}
        for Xd in draws[q]:
            nd, ndc = E.loo(Xd), E.loo(Xd, E.CROSS)
            dl = E.encoder_delta(enc, prim, X, n0, nd, d, pos)
            dln = E.encoder_delta(encn, prim, X, n0c, ndc, d, pos, E.CROSS)
            dlr = dl.copy()
            for t in pos: dlr[:, t] = dl[:, t] @ Rot[t]
            dlw = E.encoder_delta(encw, "T_own" if k == "P" else "P_full", X, n0, nd, d, wpos)
            deltas = dict(marg=dl, null_cross=dln, rotated=dlr, wrong_site=dlw)
            if k == "T": deltas["all_sources"] = E.encoder_delta(enc, prim, X, n0, nd, d, list(range(36)))
            for c, dd in deltas.items(): acc_c[c] = acc_c[c] + E.softmax(M.run(X, dd)[:, q]) / ND
        for c in conds: conds[c][:, q] = acc_c[c]
    for c, Pq in conds.items(): summarize(c, Pq)
    # span ablation and restoration
    Psp = np.zeros_like(P0); Pre = np.zeros_like(P0); Pwr = np.zeros_like(P0)
    for q in range(E.Q):
        pos = E.site_positions(k, q); dl_ab = np.zeros_like(A0); dl_re = np.zeros_like(A0); dl_wr = np.zeros_like(A0)
        sh = np.roll(np.arange(NE), 1)
        for t in pos:
            W = enc[t]; U, s, _ = np.linalg.svd(W.T, full_matrices=False); U = U[:, s > 1e-6 * s.max()]; Pu = U @ U.T
            mu = Af[:, t].mean(0)
            dl_ab[:, t] = -((A0[:, t] - mu) @ Pu)
            Fe = E.site_features(prim, X)[t]; Ffit = E.site_features(prim, Xf)[t]
            En = (Fe - Ffit.mean(0)) @ W
            dl_re[:, t] = dl_ab[:, t] + En @ Pu; dl_wr[:, t] = dl_ab[:, t] + En[sh] @ Pu
        Psp[:, q] = E.softmax(M.run(X, dl_ab)[:, q]); Pre[:, q] = E.softmax(M.run(X, dl_re)[:, q]); Pwr[:, q] = E.softmax(M.run(X, dl_wr)[:, q])
    summarize("span_ablate", Psp); summarize("span_ablate_restore_true", Pre); summarize("span_ablate_restore_wrong_episode", Pwr)
    out["restoration_fraction_of_lost_gain"] = float((acc(Pre) - acc(Psp)) / max(a0 - acc(Psp), 1e-9))
    R["per_model"][m] = out; log(f"{m}: " + json.dumps({c: round(v["drop_frac"], 3) for c, v in out.items() if isinstance(v, dict)}))
json.dump(R, open(os.path.join(OUT, "necessity.json"), "w"), indent=1, default=float)
