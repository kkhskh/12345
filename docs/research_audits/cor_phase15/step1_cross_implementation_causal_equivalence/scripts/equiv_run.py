"""Main experiment of Investigation IV.  Usage: equiv_run.py PHASE   (PHASE = cal | conf | site)
  cal  : calibration (development) models P0, P1 (P-count transfer seeds 0, 1) and T0, T1 (T1 transfer parents 0, 1);
         fit stream 12_100_000, evaluation stream 12_200_000.  Used to build and debug the analysis and to set the
         frozen predictions; never used for confirmation.
  conf : confirmation models P2, P3, P4 and T2, T10, T11; fit stream 12_300_000, evaluation stream 12_400_000.
  site : training-objective site arm S5, S6, S7 (P-count objective supervised at a source token), with P2..P4 as the
         readout arm; streams as conf.
For every model: encoders (fit stream; generator features only) and decoders; natural counterfactual families at every
decision q (cond, single-group edit, pi-preserving edit, count-preserving relabel); within-model abstract interventions
with floors and ceilings; locality tests; pathway (path-patching) tests; off-manifold statistics.  Across models:
cross-model transport through the abstract count state, and decision-response (Bayes-residual) equivalence.
Writes logs/PHASE/results.json."""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np
import eqv as E

PH = sys.argv[1]
SETS = {"cal": (["P0", "P1", "T0", "T1"], 12_100_000, 12_200_000),
        "conf": (["P2", "P3", "P4", "T2", "T10", "T11"], 12_300_000, 12_400_000),
        "site": (["S5", "S6", "S7", "P2", "P3", "P4"], 12_300_000, 12_400_000)}
names, FIT, EVAL = SETS[PH]
if PH == "site": names = [m for m in names if os.path.exists(E.ckpt_path(m))]
NF, NE = int(os.environ.get("NF", 4000)), int(os.environ.get("NE", 1000))
OUT = os.path.join(HERE, "..", "logs", PH); os.makedirs(OUT, exist_ok=True)
kind_of = lambda m: "P" if m[0] in "PS" and not (PH == "site" and m[0] == "S" and os.environ.get("S_AS_T")) else "T"
t0 = time.time(); log = lambda s: print(f"[{time.time() - t0:7.1f}s] {s}", flush=True)

Df = E.sample(NF, FIT); Xf = Df["X"]
De = E.sample(NE, EVAL); X, Y, cfg = De["X"], De["Y"], De["cfg"]
Je = E.G.joint(E.F4, X); n0 = E.loo(X); bayes0 = E.bayes_from(X, n0, Je)
own_bayes = E.G.posterior(E.F4, X)["own_y"]

# ---------------------------------------------------------------- counterfactual families (generator-defined, support-checked)
fam = {}
for q in range(E.Q):
    r = np.random.default_rng(EVAL + 100 + q)
    Xc = E.cond_pairs(X, q, r); fam[("cond", q)] = dict(X=Xc, ok=np.ones(NE, bool))
    Xs, gs, ok, sw = E.single_group_edit(X, q, r, cfg); fam[("single", q)] = dict(X=Xs, ok=ok, g=gs, same_world=sw)
    Xpp, okp = E.pi_preserving_edit(X, q, r); fam[("pipres", q)] = dict(X=Xpp, ok=okp)
    fam[("relabel", q)] = dict(X=E.relabel_pairs(X, q), ok=np.ones(NE, bool))
support_stats = {f: dict(frac_ok=float(np.mean([fam[(f, q)]["ok"].mean() for q in range(E.Q)])),
                         frac_marginal_support=float(np.mean([E.episode_support(fam[(f, q)]["X"])[fam[(f, q)]["ok"]].mean() for q in range(E.Q)])),
                         frac_same_world=float(np.mean([E.same_world(fam[(f, q)]["X"], cfg)[fam[(f, q)]["ok"]].mean() for q in range(E.Q)])))
                 for f in ("cond", "single", "pipres", "relabel")}
for q in range(E.Q):
    for f in ("cond", "single", "pipres", "relabel"):
        fam[(f, q)]["n1"] = E.loo(fam[(f, q)]["X"])
        fam[(f, q)]["bayes1"] = E.bayes_from(fam[(f, q)]["X"], fam[(f, q)]["n1"])
log(f"families built; support {json.dumps(support_stats)}")
np.savez_compressed(os.path.join(OUT, "episodes.npz"), X=X, Y=Y, cfg=cfg, Xfit=Xf,
                    **{f"{f}_q{q}": fam[(f, q)]["X"] for q in range(E.Q) for f in ("cond", "single", "pipres", "relabel")})

R = dict(phase=PH, models=names, fit_stream=FIT, eval_stream=EVAL, NF=NF, NE=NE, support=support_stats, per_model={})
cache = {}
ENC_KINDS = {"P": ["P_full"], "T": ["T_own", "T_full", "T_wronggroup"]}


def soft_delta(enc, kind, q, P0, P1, d):
    """encoder delta for decision q from soft count features P (N,3,6) (probabilities from a decoder)."""
    N = len(P0); dl = np.zeros((N, E.T, d), np.float32)
    for t in E.site_positions(kind[0], q):
        W = enc[t]
        if kind == "P_full": dl[:, t] = (P1.reshape(N, 18) - P0.reshape(N, 18)) @ W
        else: g = (t % 6) // 2; dl[:, t] = (P1[:, g] - P0[:, g]) @ W
    return dl


def maha_fit(Af, pos):
    out = {}
    for t in pos:
        Z = Af[:, t]; mu = Z.mean(0); Cv = np.cov(Z.T) + 1e-3 * np.eye(Z.shape[1]); out[t] = (mu, np.linalg.inv(Cv))
    return out


def maha(stats, A, pos):
    return np.mean([np.einsum("nd,de,ne->n", A[:, t] - stats[t][0], stats[t][1], A[:, t] - stats[t][0]) for t in pos], 0)


for m in names:
    M = E.Model(m); k = kind_of(m); d = M.d
    Af = M.acts(Xf)[0]; A0, L0, attn0, mlp0 = M.acts(X)
    encs = {ek: E.fit_encoder(ek, Xf, Af) for ek in ENC_KINDS[k]}
    encs["null_cross"] = E.fit_encoder("P_full" if k == "P" else "T_own", Xf, Af, E.CROSS)
    dec = E.fit_decoder(k, Xf, Af)
    allpos = sorted(set(sum([E.site_positions(k, q) for q in range(E.Q)], [])))
    mst = maha_fit(Af, allpos)
    # decoding accuracy on the eval set (held-out stream)
    dacc = np.mean([(E.decode(dec, A0, q)[1] == n0[:, q]).mean() for q in range(E.Q)])
    res = dict(kind=k, acc=float((L0.argmax(-1) == Y).mean()), acc_bayes=float((bayes0.argmax(-1) == Y).mean()),
               acc_own_bayes=float((own_bayes.argmax(-1) == Y).mean()), decoder_acc_per_component=float(dacc), within={}, offmanifold={})
    log(f"{m}: acc {res['acc']:.3f} decoder acc {dacc:.3f}")
    cache[m] = dict(M=M, k=k, encs=encs, dec=dec, A0=A0, L0=L0, nat={}, d=d)
    for f in ("cond", "single", "pipres", "relabel"):
        agg = {}
        for q in range(E.Q):
            Fm = fam[(f, q)]; ok = Fm["ok"]; Xp = Fm["X"]; n1 = Fm["n1"]
            Ap, Lp, _, _ = M.acts(Xp); cache[m]["nat"][(f, q)] = (Ap, Lp)
            pos = E.site_positions(k, q); b, nt = L0[:, q], Lp[:, q]
            conds = {}
            dl = np.zeros_like(A0); dl[:, pos] = Ap[:, pos] - A0[:, pos]; conds["full_site_patch"] = dl
            for ek, enc in encs.items():
                ekk = ek if ek != "null_cross" else ("P_full" if k == "P" else "T_own")
                pr = E.CROSS if ek == "null_cross" else E.G.GROUPS
                nn0, nn1 = (E.loo(X, pr), E.loo(Xp, pr))
                conds[f"enc|{ek}"] = E.encoder_delta(enc, ekk, X, nn0, nn1, d, pos, pr)
                if ek == ENC_KINDS[k][0]:                                   # primary encoder: floors and locality
                    sh = np.roll(np.arange(NE), 1)
                    conds[f"enc|{ek}|floor_mismatched"] = E.encoder_delta(enc, ekk, X, nn0, nn1[sh], d, pos, pr)
                    if f == "single" and k == "T":
                        gpos = [6 * q + 2 * gg + j for gg in range(3) for j in range(2)]
                        dloc = np.zeros_like(A0); done = np.zeros_like(A0)
                        full = conds[f"enc|{ek}"]
                        for i in range(NE):
                            if Fm["g"][i] < 0: continue
                            tt = [6 * q + 2 * Fm["g"][i], 6 * q + 2 * Fm["g"][i] + 1]
                            dloc[i, tt] = full[i, tt]; done[i, tt[0]] = full[i, tt[0]]
                        conds[f"enc|{ek}|local_group_tokens_only"] = dloc
                        conds[f"enc|{ek}|one_token_of_pair_only"] = done
            # transport-independent self decoder (alpha_M then E_M)
            P0d = E.decode(dec, A0, q)[0].reshape(NE, 3, 6); P1d = E.decode(dec, Ap, q)[0].reshape(NE, 3, 6)
            conds["self_decoded_then_encoded"] = soft_delta(encs[ENC_KINDS[k][0]], ENC_KINDS[k][0], q, P0d, P1d, d)
            cache[m]["nat"][(f, q, "dec")] = (P0d, P1d)
            for nm, dl in conds.items():
                o = M.run(X, dl)[:, q]
                agg.setdefault(nm, []).append((b[ok], nt[ok], o[ok]))
                if nm in ("full_site_patch", f"enc|{ENC_KINDS[k][0]}") and f == "cond":
                    Ap2 = A0 + dl
                    res["offmanifold"].setdefault(nm, []).append(float(np.median(maha(mst, Ap2, pos))))
            if f == "cond":
                res["offmanifold"].setdefault("natural_base", []).append(float(np.median(maha(mst, A0, pos))))
                res["offmanifold"].setdefault("natural_cf", []).append(float(np.median(maha(mst, Ap, pos))))
        out = {}
        for nm, lst in agg.items():
            b, nt, o = (np.concatenate([x[i] for x in lst]) for i in range(3))
            out[nm] = E.score(b, nt, o)
        ce, fl = out["full_site_patch"], out.get(f"enc|{ENC_KINDS[k][0]}|floor_mismatched")
        if fl is not None:
            for nm in out:
                if nm.startswith("enc|") or nm.startswith("self"):
                    out[nm]["iia_adjusted"] = E.adjusted(out[nm]["iia"], fl["iia"], ce["iia"])
        res["within"][f] = out
    res["offmanifold"] = {k_: float(np.mean(v)) for k_, v in res["offmanifold"].items()}
    # ------------------------------------------------ pathway test (cond family, primary encoder)
    pw = {}
    for q in range(E.Q):
        Fm = fam[("cond", q)]; pos = E.site_positions(k, q); n1 = Fm["n1"]
        dl = E.encoder_delta(encs[ENC_KINDS[k][0]], ENC_KINDS[k][0], X, n0, n1, d, pos)
        ma = np.zeros(E.T, np.float32); ma[36 + q] = 1; z = np.zeros(E.T, np.float32)
        full = M.run(X, dl)[:, q]
        fa = M.run_frozen(X, dl, attn0, mlp0, ma, z)[:, q]           # readout's layer-2 attention output frozen
        fm = M.run_frozen(X, dl, attn0, mlp0, z, ma)[:, q]           # readout's layer-2 MLP output frozen
        for nm, o in (("full", full), ("attn_frozen", fa), ("mlp_frozen", fm)):
            pw.setdefault(nm, []).append(np.abs(E.clogit(o) - E.clogit(L0[:, q])).sum(-1))
    tot = np.concatenate(pw["full"]).mean()
    res["pathway"] = {nm: float(np.concatenate(v).mean() / tot) for nm, v in pw.items()}
    R["per_model"][m] = res
    log(f"{m}: within done; pathway {res['pathway']}")

# ---------------------------------------------------------------- cross-model transport (Prediction II)
tr = {}
for D in names:
    for Rm in names:
        cd, cr = cache[D], cache[Rm]; kr = cr["k"]; ek = ENC_KINDS[kr][0]
        for f in ("cond", "single"):
            acc = {"donor_decoded": [], "oracle_true_counts": [], "donor_decoded_floor": []}
            for q in range(E.Q):
                Fm = fam[(f, q)]; ok = Fm["ok"]; pos = E.site_positions(kr, q)
                P0d, P1d = cd["nat"][(f, q, "dec")]
                Lp = cr["nat"][(f, q)][1]; b, nt = cr["L0"][:, q], Lp[:, q]
                dl = soft_delta(cr["encs"][ek], ek, q, P0d, P1d, cr["d"])
                sh = np.roll(np.arange(NE), 1)
                dlf = soft_delta(cr["encs"][ek], ek, q, P0d[sh], P1d[sh], cr["d"])
                dlo = E.encoder_delta(cr["encs"][ek], ek, X, n0, Fm["n1"], cr["d"], pos)
                for nm, dd in (("donor_decoded", dl), ("donor_decoded_floor", dlf), ("oracle_true_counts", dlo)):
                    acc[nm].append((b[ok], nt[ok], cr["M"].run(X, dd)[:, q][ok]))
            s = {nm: E.score(*(np.concatenate([x[i] for x in v]) for i in range(3))) for nm, v in acc.items()}
            s["transport_fidelity_iia"] = E.adjusted(s["donor_decoded"]["iia"], s["donor_decoded_floor"]["iia"], s["oracle_true_counts"]["iia"])
            s["transport_fidelity_r"] = E.adjusted(s["donor_decoded"]["r_dlogit"], s["donor_decoded_floor"]["r_dlogit"], s["oracle_true_counts"]["r_dlogit"])
            tr[f"{D}->{Rm}|{f}"] = s
    log(f"transport from {D} done")
R["transport"] = tr

# ---------------------------------------------------------------- decision-response equivalence beyond Bayes (H1 vs H3)
rng = np.random.default_rng(EVAL + 7)
import itertools
SUPP = np.array([v for v in itertools.product(range(6), repeat=3) if 5 in v])
qsel = rng.integers(0, E.Q, NE)                              # one decision per episode
A_q = E.agree(X)[np.arange(NE), qsel]                       # (NE,3) own-question agreements at the scored decision
def nat_ok(v, a):                                             # (X_q, v) has positive generator probability
    return any(v[h] == 5 and a[h] == 1 for h in range(3))
nr = n0.copy(); noff = n0.copy(); okoff = np.zeros(NE, bool)
for i in range(NE):
    nat = [v for v in SUPP if nat_ok(v, A_q[i])]; off = [v for v in SUPP if not nat_ok(v, A_q[i])]
    nr[i, qsel[i]] = nat[rng.integers(len(nat))]
    if off: noff[i, qsel[i]] = off[rng.integers(len(off))]; okoff[i] = True
npp = n0.copy(); okpp = np.zeros(NE, bool)
for i in range(NE):
    v = n0[i, qsel[i]]; H = np.where(v == 5)[0]
    if len(H) != 1: continue
    o = [g for g in range(3) if g != H[0]]; opts = ([1] if all(v[g] <= 3 for g in o) else []) + ([-1] if all(v[g] >= 1 for g in o) else [])
    if not opts: continue
    s_ = opts[rng.integers(len(opts))]; w = v.copy(); w[o] += s_; npp[i, qsel[i]] = w; okpp[i] = True
bayes_r = E.bayes_from(X, nr, Je); idx = np.arange(NE)
dB = E.clogit(np.log(bayes_r[idx, qsel])) - E.clogit(np.log(bayes0[idx, qsel]))
resp = {"random_support": {}, "off_natural_support": {}, "pi_preserving_abstract": {}, "pi_preserving_input": {}}
for m in names:
    c = cache[m]; k = c["k"]; ek = ENC_KINDS[k][0]; d = c["d"]; L = np.zeros((NE, 3)); Lpp = np.zeros((NE, 3)); Lin = np.zeros((NE, 3)); Loff = np.zeros((NE, 3))
    for q in range(E.Q):
        sel = qsel == q; pos = E.site_positions(k, q)
        if not sel.any(): continue
        dl = E.encoder_delta(c["encs"][ek], ek, X[sel], n0[sel], nr[sel], d, pos); L[sel] = c["M"].run(X[sel], dl)[:, q]
        dl = E.encoder_delta(c["encs"][ek], ek, X[sel], n0[sel], npp[sel], d, pos); Lpp[sel] = c["M"].run(X[sel], dl)[:, q]
        dl = E.encoder_delta(c["encs"][ek], ek, X[sel], n0[sel], noff[sel], d, pos); Loff[sel] = c["M"].run(X[sel], dl)[:, q]
        Lin[sel] = c["nat"][("pipres", q)][1][sel][:, q]
    base = c["L0"][idx, qsel]
    resp["random_support"][m] = E.clogit(L) - E.clogit(base)
    resp["pi_preserving_abstract"][m] = E.clogit(Lpp) - E.clogit(base)
    resp["off_natural_support"][m] = E.clogit(Loff) - E.clogit(base)
    resp["pi_preserving_input"][m] = E.clogit(Lin) - E.clogit(base)
okin = np.array([fam[("pipres", qsel[i])]["ok"][i] for i in range(NE)])
masks = {"random_support": np.ones(NE, bool), "off_natural_support": okoff, "pi_preserving_abstract": okpp, "pi_preserving_input": okin}
de = {}
for fam_name, rr in resp.items():
    mk = masks[fam_name]; out = {"n": int(mk.sum())}
    for a in names:
        da = rr[a][mk]
        if fam_name == "random_support":
            dBm = dB[mk]; beta = (da * dBm).sum() / (dBm * dBm).sum(); out[f"{a}|r_with_bayes"] = float(np.corrcoef(da.ravel(), dBm.ravel())[0, 1])
            out[f"{a}|beta"] = float(beta); rr[a] = rr[a] - beta * dB             # residual from Bayes (per-model scale)
        out[f"{a}|mean_abs_resp"] = float(np.abs(rr[a][mk]).mean())
    for a, b2 in itertools.combinations(names, 2):
        x1, x2 = rr[a][mk].ravel(), rr[b2][mk].ravel()
        sh = np.roll(np.arange(mk.sum()), 1)
        out[f"{a}~{b2}|r"] = float(np.corrcoef(x1, x2)[0, 1]); out[f"{a}~{b2}|r_shuffled_null"] = float(np.corrcoef(x1, rr[b2][mk][sh].ravel())[0, 1])
        out[f"{a}~{b2}|sign_agree_argmax"] = float((rr[a][mk].argmax(-1) == rr[b2][mk].argmax(-1)).mean())
    de[fam_name] = out
R["decision_response"] = de
np.savez_compressed(os.path.join(OUT, "responses.npz"), qsel=qsel, dB=dB, **{f"{fn}|{m}": v for fn, rr in resp.items() for m, v in rr.items()},
                    **{f"mask|{fn}": v for fn, v in masks.items()})
json.dump(R, open(os.path.join(OUT, "results.json"), "w"), indent=1, default=float)
log("written " + os.path.join(OUT, "results.json"))
