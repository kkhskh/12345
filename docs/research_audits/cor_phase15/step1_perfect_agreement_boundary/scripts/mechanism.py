"""P6 mechanism analysis (interventions at layer-2 input = output of the frozen, curriculum-pretrained layers 0-1).
Usage: mechanism.py PHASE STREAM MODEL [...]   PHASE dev uses STREAM for development; the frozen P6 run uses stream 16_000_000.
Pairs: natural episodes (STREAM), one decision q each, F45 / F34 / F0 edits built exactly as in boundary_run.py.
Decoder for n^{(-q)} fitted on independent episodes (STREAM + 200_000) at the decision site (eqv.site_positions).
Interventions on the BEFORE episode, adding (A_after - A_before) restricted to:
  all    : every position (sanity: reproduces the full response)
  site   : the decision site positions only
  nonsite: every position except the site
  sub    : at the site, the projection onto the decoder subspace for the edited group (row space of its 6-class logistic weights)
  subc   : at the site, the orthogonal complement of 'sub'
Per family and regime: share = sum(dm_patch * dm_full) / sum(dm_full^2) (regression share of the model's own response), and the
patched excess B_patch = mean(dm_patch - dm_bayes) over oriented pairs.  Redundancy: ablation = the AFTER episode with the site
subspace component replaced by the BEFORE value (share of response removed)."""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np
import boundary as B
R, E, G = B.R, B.E, B.G
PH, STREAM, names = sys.argv[1], int(sys.argv[2]), sys.argv[3:]; NE = int(os.environ.get("NE", 3000)); NFIT = int(os.environ.get("NFIT", 4000))
OUT = os.path.join(HERE, "..", "logs", f"mech_{PH}"); os.makedirs(OUT, exist_ok=True); t0 = time.time()
D = E.sample(NE, STREAM); X, cfg = D["X"], D["cfg"]; q = np.random.default_rng(STREAM + 1).integers(0, 6, NE); idx = np.arange(NE)
fam, meta = {}, {}
for name, frm in (("F45", 4), ("F34", 3)):
    Xp = X.copy(); gs = np.full(NE, -1); ks = gs.copy(); ok = np.zeros(NE, bool)
    for qq in range(6):
        s = q == qq; a, g_, k_, o = B.count_edit(X[s], qq, cfg[s], np.random.default_rng(STREAM + 10 + qq + 100 * frm), frm, frm + 1); Xp[s], gs[s], ks[s], ok[s] = a, g_, k_, o
    vote = np.array([B.group_vote(X[i:i + 1], q[i], max(gs[i], 0))[0] for i in idx]); fam[name] = Xp; meta[name] = dict(g=gs, ok=ok, vote=vote, orient=ok & (vote >= 0))
    if name == "F45": fam["F0"] = B.token_control(X, gs, ks); meta["F0"] = meta["F45"]
n0 = E.loo(X)[idx, q]; Hb = n0 == 5; Vq = np.stack([np.array([B.group_vote(X[i:i + 1], q[i], g)[0] for i in idx]) for g in range(3)], 1)
def regime(f):
    rg = np.full(NE, "", object); gs, vote = meta[f]["g"], meta[f]["vote"]
    for i in np.where(meta[f]["orient"])[0]:
        oth = [h for h in range(3) if h != gs[i]]
        rg[i] = "R1" if any(Hb[i, h] and Vq[i, h] == vote[i] for h in oth) else ("R2" if any((not Hb[i, h]) and Vq[i, h] == vote[i] for h in oth) else "R3")
    return rg
REG = {f: regime(f) for f in fam}; c0B = B.bayes_c(X, q); cB = {f: B.bayes_c(Xf, q) for f, Xf in fam.items()}
Dfit = E.sample(NFIT, STREAM + 200_000)["X"]; res = {}
for m in names:
    M = B.Model(m); kind = M.kind; A0, L0 = M.acts(X)[:2]; c0 = E.clogit(L0[idx, q]); Afit = M.acts(Dfit)[0]
    dec = E.fit_decoder(kind, Dfit, Afit); dfeat = len(dec["mu"]); out = {}
    # decoding accuracy of n_g (held-out = the pair episodes)
    accs = []
    for f in ("F45",):
        Aa = M.acts(fam[f])[0]
        for i_set, (AA, XX) in enumerate(((A0, X), (Aa, fam[f]))):
            preds = np.zeros((NE, 3), int)
            for qq in range(6):
                s = q == qq; preds[s] = E.decode(dec, AA[s], qq)[1]
            accs.append(float((preds == E.loo(XX)[idx, q]).mean()))
    out["decode_acc_counts"] = accs
    for f, Xf in fam.items():
        Aa, La = M.acts(Xf)[:2]; ca = E.clogit(La[idx, q]); dA = Aa - A0; o = meta[f]["orient"]; v = np.maximum(meta[f]["vote"], 0); gs = np.maximum(meta[f]["g"], 0)
        dm_full = B.oriented_margin(ca, v) - B.oriented_margin(c0, v); dm_B = B.oriented_margin(cB[f], v) - B.oriented_margin(c0B, v)
        site = np.zeros((NE, A0.shape[1]), bool)
        for i in idx: site[i, E.site_positions(kind, q[i])] = True
        # subspace projector per example (group gs, decision site): decoder weight rows for group g, mapped back to activations
        npos = len(E.site_positions(kind, 0)); dmod = A0.shape[2]
        Pg = []
        for g in range(3):
            W = dec["clf"][g].coef_ / dec["sd"][None]               # (classes, npos*dmod) in raw feature space
            Qm, _ = np.linalg.qr(W.T); Pg.append(Qm)                 # orthonormal basis (npos*dmod, r)
        def sub_delta(Dl, comp=False):
            Z = np.zeros_like(Dl)
            for i in idx:
                pos = E.site_positions(kind, q[i]); vv = Dl[i, pos].reshape(-1); Qm = Pg[gs[i]]; pr = Qm @ (Qm.T @ vv)
                Z[i, pos] = (vv - pr if comp else pr).reshape(npos, dmod)
            return Z
        deltas = dict(all=dA, site=dA * site[:, :, None], nonsite=dA * (~site)[:, :, None], sub=sub_delta(dA), subc=sub_delta(dA, True))
        fo = {}
        for k, dl in deltas.items():
            cp = E.clogit(M.run(X, dl)[idx, q]); dm_p = B.oriented_margin(cp, v) - B.oriented_margin(c0, v)
            row = dict(share=float((dm_p * dm_full)[o].sum() / max((dm_full ** 2)[o].sum(), 1e-12)), mean_abs=float(np.abs(dm_p[o]).mean()))
            for rg in ("R1", "R2", "R3"):
                s = o & (REG[f] == rg)
                if s.sum() >= 10: row[rg] = dict(n=int(s.sum()), share=float((dm_p * dm_full)[s].sum() / max((dm_full ** 2)[s].sum(), 1e-12)),
                                                 B_patch=float((dm_p - dm_B)[s].mean()), B_full=float((dm_full - dm_B)[s].mean()), mean_dm_patch=float(dm_p[s].mean()), mean_dm_full=float(dm_full[s].mean()))
            fo[k] = row
        # redundancy: AFTER episode with site-subspace change removed
        ca_abl = E.clogit(M.run(Xf, -sub_delta(dA))[idx, q]); dm_abl = B.oriented_margin(ca_abl, v) - B.oriented_margin(c0, v)
        fo["ablate_sub_after"] = dict(remaining_share=float((dm_abl * dm_full)[o].sum() / max((dm_full ** 2)[o].sum(), 1e-12)),
                                      R1_remaining=float((dm_abl * dm_full)[o & (REG[f] == "R1")].sum() / max((dm_full ** 2)[o & (REG[f] == "R1")].sum(), 1e-12)))
        fo["mean_abs_full"] = float(np.abs(dm_full[o]).mean()); out[f] = fo
    res[m] = dict(kind=kind, **out); json.dump(res[m], open(os.path.join(OUT, f"{m.replace('@', '_')}.json"), "w"), indent=1)
    r1 = out["F45"]
    print(f"[{time.time() - t0:.0f}s] {m} {kind} dec {out['decode_acc_counts']}  F45 share all {r1['all']['share']:.2f} site {r1['site']['share']:.2f} nonsite {r1['nonsite']['share']:.2f} sub {r1['sub']['share']:.2f} subc {r1['subc']['share']:.2f} abl-remaining {r1['ablate_sub_after']['remaining_share']:.2f} | R1 B_full {r1['site'].get('R1',{}).get('B_full',np.nan):+.3f} B_site {r1['site'].get('R1',{}).get('B_patch',np.nan):+.3f} B_sub {r1['sub'].get('R1',{}).get('B_patch',np.nan):+.3f} | F0 site |dm| {out['F0']['site']['mean_abs']:.3f}", flush=True)
