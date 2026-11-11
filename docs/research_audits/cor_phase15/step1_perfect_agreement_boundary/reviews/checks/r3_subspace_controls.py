"""R3 reviewer check (read-only on the package): F45 R1 patched responses at the site for
  sub   : decoder row space of the edited group g (as mechanism.py)
  subh  : decoder row space of a different group h=(g+1)%3 (non-target count subspace)
  rand  : random subspace of the same rank, at the site
Also: fraction of ||dA_site||^2 in each subspace; regime counts for F45/F34.  Same streams as the frozen run."""
import os, sys, time
P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(P, "src"))
import numpy as np, boundary as B
E = B.E; G = B.G
STREAM = 16_000_000; NE = int(os.environ.get("NE", 3000)); NFIT = 4000; names = sys.argv[1:]
D = E.sample(NE, STREAM); X = D["X"]; cfg = D["cfg"]; q = np.random.default_rng(STREAM + 1).integers(0, 6, NE); idx = np.arange(NE)
fam = {}; meta = {}
for name, frm in (("F45", 4), ("F34", 3)):
    Xp = X.copy(); gs = np.full(NE, -1); ok = np.zeros(NE, bool)
    for qq in range(6):
        s = q == qq; a, g_, k_, o = B.count_edit(X[s], qq, cfg[s], np.random.default_rng(STREAM + 10 + qq + 100 * frm), frm, frm + 1); Xp[s], gs[s], ok[s] = a, g_, o
    vote = np.array([B.group_vote(X[i:i + 1], q[i], max(gs[i], 0))[0] for i in idx]); fam[name] = Xp; meta[name] = dict(g=gs, vote=vote, orient=ok & (vote >= 0))
n0 = E.loo(X)[idx, q]; Hb = n0 == 5; Vq = np.stack([np.array([B.group_vote(X[i:i + 1], q[i], g)[0] for i in idx]) for g in range(3)], 1)
def regime(f):
    rg = np.full(NE, "", object); gs, vote = meta[f]["g"], meta[f]["vote"]
    for i in np.where(meta[f]["orient"])[0]:
        oth = [h for h in range(3) if h != gs[i]]
        rg[i] = "R1" if any(Hb[i, h] and Vq[i, h] == vote[i] for h in oth) else ("R2" if any((not Hb[i, h]) and Vq[i, h] == vote[i] for h in oth) else "R3")
    return rg
REG = {f: regime(f) for f in fam}
for f in fam:
    print(f, "oriented", int(meta[f]["orient"].sum()), {r: int((REG[f] == r).sum()) for r in ("R1", "R2", "R3")})
    s = REG[f] == "R1"; print("  R1 distinct episodes with F45&F34 both R1:", int(((REG["F45"] == "R1") & (REG["F34"] == "R1")).sum()))
Dfit = E.sample(NFIT, STREAM + 200_000)["X"]; t0 = time.time()
for m in names:
    M = B.Model(m); kind = M.kind; A0, L0 = M.acts(X)[:2]; c0 = E.clogit(L0[idx, q]); dec = E.fit_decoder(kind, Dfit, M.acts(Dfit)[0])
    npos = len(E.site_positions(kind, 0)); dmod = A0.shape[2]; rng = np.random.default_rng(0)
    Pg = [np.linalg.qr((dec["clf"][g].coef_ / dec["sd"][None]).T)[0] for g in range(3)]
    Pr = [np.linalg.qr(rng.standard_normal((npos * dmod, Pg[g].shape[1])))[0] for g in range(3)]
    f = "F45"; Aa, La = M.acts(fam[f])[:2]; ca = E.clogit(La[idx, q]); dA = Aa - A0; o = meta[f]["orient"]; v = np.maximum(meta[f]["vote"], 0); gs = np.maximum(meta[f]["g"], 0)
    # PCA control: top-r principal directions of the site delta itself, fitted on the F34 family (no count labels, other edits)
    dA34 = M.acts(fam["F34"])[0] - A0; S34 = np.stack([dA34[i, E.site_positions(kind, q[i])].reshape(-1) for i in idx if meta["F34"]["orient"][i]])
    Ppca = np.linalg.svd(S34, full_matrices=False)[2][:Pg[0].shape[1]].T
    dm_full = B.oriented_margin(ca, v) - B.oriented_margin(c0, v); r1 = o & (REG[f] == "R1")
    def proj(which):
        Z = np.zeros_like(dA); fr = []
        for i in idx:
            pos = E.site_positions(kind, q[i]); vv = dA[i, pos].reshape(-1)
            Qm = {"sub": Pg[gs[i]], "subh": Pg[(gs[i] + 1) % 3], "rand": Pr[gs[i]], "pca": Ppca, "site": None}[which]
            pr = vv if Qm is None else Qm @ (Qm.T @ vv); Z[i, pos] = pr.reshape(npos, dmod)
            if o[i]: fr.append(float(pr @ pr / max(vv @ vv, 1e-12)))
        return Z, float(np.median(fr))
    line = [m, f"rank {Pg[0].shape[1]} dim {npos*dmod}", f"R1 full {dm_full[r1].mean():+.3f}"]
    for w in ("site", "sub", "subh", "rand", "pca"):
        Z, fr = proj(w); dm = B.oriented_margin(E.clogit(M.run(X, Z)[idx, q]), v) - B.oriented_margin(c0, v)
        sh = (dm * dm_full)[o].sum() / (dm_full ** 2)[o].sum()
        line.append(f"{w}: R1 dm {dm[r1].mean():+.3f} share {sh:.2f} normfrac {fr:.3f}")
    print(f"[{time.time()-t0:.0f}s]", " | ".join(line), flush=True)
