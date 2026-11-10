"""Reviewer check of Prediction VI statistics (logs/conf/responses.npz): reproduce point estimates; Monte-Carlo error of the
300-replicate percentile bound; per-pair episode-bootstrap SEs; leave-one-model-out; exhaustive curriculum-label permutation
(model-level null); two-way (models-within-curriculum x episodes) bootstrap; T2-excluded contrast."""
import itertools, os, numpy as np
B = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
Z = np.load(os.path.join(B, "logs/conf/responses.npz")); names = ["P2", "P3", "P4", "T2", "T10", "T11"]
rng = np.random.default_rng(12345)
def r(a, b): return np.corrcoef(a.ravel(), b.ravel())[0, 1]
def contrast(Rm, labT):
    Tm = [m for m in names if m in labT]; Pm = [m for m in names if m not in labT]
    wT = np.mean([Rm[a, b] for a, b in itertools.combinations(Tm, 2)]); wP = np.mean([Rm[a, b] for a, b in itertools.combinations(Pm, 2)])
    cr = np.mean([Rm[a, b] for a in Tm for b in Pm]); return cr - wT, cr - wP, wT, wP, cr
for fn in ["pi_preserving_abstract", "pi_preserving_input", "off_natural_support", "random_support"]:
    mk = Z[f"mask|{fn}"]; idx = np.where(mk)[0]; resp = {m: Z[f"{fn}|{m}"].astype(float) for m in names}
    if fn == "random_support":
        dB = Z["dB"]
        for m in names: beta = (resp[m][mk] * dB[mk]).sum() / (dB[mk] ** 2).sum(); resp[m] = resp[m] - beta * dB
    def Rmat(ii): return {(a, b): r(resp[a][ii], resp[b][ii]) for a, b in itertools.permutations(names, 2)}
    R0 = Rmat(idx); x, xP, wT, wP, cr = contrast(R0, {"T2", "T10", "T11"})
    print(f"\n===== {fn} n_ep {len(idx)}: within_P {wP:.3f} within_T {wT:.3f} cross {cr:.3f} cross-wT {x:.3f} cross-wP {xP:.3f}")
    boots = [Rmat(rng.choice(idx, len(idx))) for _ in range(1000)]
    bx = np.array([contrast(b, {"T2", "T10", "T11"})[0] for b in boots])
    print(f"episode bootstrap 1000: cross-wT CI [{np.percentile(bx,2.5):.3f}, {np.percentile(bx,97.5):.3f}], SE {bx.std():.3f}")
    ub = [np.percentile(rng.choice(bx, 300), 97.5) for _ in range(200)]
    print(f"MC sd of the 300-replicate upper bound: {np.std(ub):.4f}")
    pse = {f"{a}~{b}": np.std([bb[a, b] for bb in boots]) for a, b in itertools.combinations(names, 2)}
    print("per-pair r point (episode-boot SE):", {k: f"{R0[tuple(k.split('~'))]:.3f}({v:.3f})" for k, v in pse.items()})
    # leave one model out
    for drop in names:
        keep = [m for m in names if m != drop]; Tm = [m for m in keep if m[0] == "T"]; Pm = [m for m in keep if m[0] == "P"]
        wT_ = np.mean([R0[a, b] for a, b in itertools.combinations(Tm, 2)]); cr_ = np.mean([R0[a, b] for a in Tm for b in Pm])
        wP_ = np.mean([R0[a, b] for a, b in itertools.combinations(Pm, 2)])
        print(f"  drop {drop}: within_T {wT_:.3f} within_P {wP_:.3f} cross {cr_:.3f} cross-wT {cr_-wT_:.3f}")
    # exhaustive label permutation: which 3 models are called "T"
    stats = []
    for lab in itertools.combinations(names, 3):
        c = contrast(R0, set(lab)); stats.append((c[0], lab, min(c[0], c[1])))
    obs = x; k = sum(s[0] <= obs + 1e-12 for s in stats)
    print(f"label permutation (20 splits): #splits with cross-'T' <= observed = {k}/20 -> p = {k/20:.2f};",
          "most extreme splits:", [(round(s[0], 3), s[1]) for s in sorted(stats)[:3]])
    obs2 = min(x, xP); k2 = sum(s[2] <= obs2 + 1e-12 for s in stats)
    print(f"symmetric stat min(cross-wT, cross-wP): obs {obs2:.3f}, p = {k2}/20")
    # two-way bootstrap: resample models within curriculum (with replacement), drop self-pairs, plus episodes
    tw = []
    for bb in boots[:1000]:
        Ps = rng.choice(["P2", "P3", "P4"], 3); Ts = rng.choice(["T2", "T10", "T11"], 3)
        wT_ = [bb[a, b] for a, b in itertools.combinations(Ts, 2) if a != b]
        cr_ = [bb[a, b] for a in Ts for b in Ps]
        if wT_: tw.append(np.mean(cr_) - np.mean(wT_))
    tw = np.array(tw); print(f"two-way bootstrap (models within curriculum x episodes, {len(tw)} valid): cross-wT CI [{np.percentile(tw,2.5):.3f}, {np.percentile(tw,97.5):.3f}]; frac upper<-0.10 criterion-like: P(stat>=-0.10)={np.mean(tw>=-0.10):.3f}")
    # T2 excluded
    Tm = ["T10", "T11"]; cr_ = np.mean([R0[a, b] for a in Tm for b in ["P2", "P3", "P4"]])
    print(f"  T2 excluded: within_T(T10~T11) {R0['T10','T11']:.3f} cross(P x T10,T11) {cr_:.3f}; min within_T pair {min(R0[a,b] for a,b in itertools.combinations(['T2','T10','T11'],2)):.3f} vs max cross pair {max(R0[a,b] for a in ['T2','T10','T11'] for b in ['P2','P3','P4']):.3f}")
