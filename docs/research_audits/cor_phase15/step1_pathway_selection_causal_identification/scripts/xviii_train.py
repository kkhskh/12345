"""XVIII harness = XIV ft_factorial_c.py (byte-identical logic for every XIV mode; bitwise replay checked in logs/replay_check.json) plus
four XVIII training-time interventions on the edge set FX_EDGES (src/maskfwd.edges), active for steps < T:
  gsg:<T>  forward UNCHANGED; stop_gradient on the attention logits of the target edges (routing gradient removed, values still learn)
  fst:<T>  forward BLOCKED (target attention = 0, as -1e9), backward through the UNBLOCKED softmax (straight-through: a = a_full + sg(a_blk - a_full))
  vrs:<T>  attention UNCHANGED; the value vectors read through the target edges are taken from another episode of the batch (roll by 1)
  gof:<T>  finite logit offset -FX_B on the target edges (graded suppression)
  gsv:<T>  forward UNCHANGED; ALL gradient through the target edges removed (logits stop-gradient, and the target term of the head output
           stop-gradient: no routing and no value gradient through T)                               [added after pre-review B]
  sha:<T>  adaptive share clamp (pre-review A, §3.3): a stop-gradient per-head logit offset beta_h(i) on the target edges, solved by 8 Newton
           steps so that the head-mean target share over (episodes, readout rows) equals a schedule A*_h(i); routing gradient flows.
           FX_SHA_SCHED = slog.json of the paired baseline (per-step, per-head target shares); FX_SHA_LO=1 holds A*_h(i) = A*_h(0);
           FX_SHA_CONTENT = real | rolled (rolled = the vrg value substitution on T).
  vrg:<T>  as vrs, with stop_gradient on the rolled values (no cross-episode value training)        [added after pre-review B]
FX_GLOG=1 also records per-parameter-group gradient norms (emb, layer0, layer1, layer2, WA) in the window, and for sha the offsets and achieved shares.
FX_SLOG=<edges> (plain runs): records, for steps < FX_SLOGT, the per-head mean target share of the current parameters on the current batch
(before the update) and the batch loss, to slog.json.  Logging only: it does not change the training computation.
FX_GLOG=1 additionally records the global gradient norm and the batch loss at every step of the window (for the new modes only).

XIV Stage I harness = ft_factorial_b.py (XIII-E) plus: FX_INTERV=msk:<T> with FX_EDGES=<kind> (off-route sham: -1e9 bias on the
src/maskfwd.edges(kind) edge set, same mechanism as the real blocks), FX_INTERV=lnz:<T> with FX_EDGES=<kind> FX_LSIG=<sigma> (location-matched
disturbance: sigma*N(0,1) added to the attention logits on that edge set, fresh per step and element, key PRNGKey(525252) folded with the
step), and FX_EXTRA_SNAPS=a,b,... (additional checkpoint steps).  Everything else is byte-for-byte the XIII-E harness logic.

XIII-C Workstream A1/A2: full_c1 fine-tuning with the randomness sources SEPARATED.

Audit of XIII-B scripts/native_train.py (spec full_c1, target label) -- every use of the fine-tuning seed `s`:
  (1) fresh layer-2 block      : N.init("M", ENV, s + 7777)["layers"][2]
  (2) fresh readout map WA      : N.init("M", ENV, s + 7777)["WA"]           (same init call as (1); separable by calling init with two seeds)
  (3) training data stream      : np.random.default_rng(1_000_003*(s+1) + 7919*4 + 31*3 + 17) -> NL.Bm.sample_fast(ENV, 256, rng) per step
                                  (one rng draws episodes AND their order; there is no separate shuffling -- each step is a fresh sample)
  other randomness: none in the training loop (no dropout, no data augmentation; AdamW deterministic; final LN gf/bf inherited from parent;
  embeddings and layers 0-1 inherited from parent).  Evaluation (ER.run) uses fixed files and no rng.  Floating-point summation order
  is fixed by XLA_FLAGS (single-threaded Eigen, intra_op 1, as in XIII-B lane.sh) -- verified by bitwise reproduction (repro check below).

Usage: ft_factorial.py PARENT LSEED WSEED DSEED [STEPS=4000]
  LSEED: seed of the fresh layer-2 init; WSEED: seed of the fresh WA init; DSEED: seed of the data stream.
  L = W = D = s reproduces XIII-B native_train.py full_c1 seed s exactly (checked against its saved checkpoints).
Optional training-time intervention (Workstream A4/R2):  FX_INTERV="<name>:<T>"  applies an attention mask for steps < T:
  ro0  : layer 0, readout tokens (36..41) may not attend to source tokens (0..35)   [suppresses early candidate readout]
  ro01 : same mask in layers 0 and 1                                               [suppresses early readout + layer-1 readout pooling]
  rq1  : layer 1, readout tokens may not attend to any source token                 [suppresses layer-1 relational pooling into readout]
  l2q  : layer 2 (fresh), readout token 36+q may not attend to the 6 source tokens of its own q   [suppresses mixture-head formation]
Writes checkpoints/ft/<tag>/{step<k>.npz, curve.json, final.json}; dense log-spaced checkpoints for trajectories.
RESUMABLE (added after the 07:23Z container restart killed the first three runs): at every checkpoint the AdamW state and the numpy
bit-generator state are written to resume.pkl; a restarted run continues from the last checkpoint and is bitwise identical to an
uninterrupted run (checked in logs/repro_resume.json).  A run whose final.json exists exits immediately (idempotent)."""
import json, os, sys, time, pickle
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src")); sys.path.insert(0, os.path.join(HERE, "..", "..", "step1_cross_family_relational_foundations", "src"))
import numpy as np, jax, jax.numpy as jnp, optax
import native as NV, maskfwd as MFW
NL, N = NV.NL, NV.N
parent = sys.argv[1]; LS, WS, DS = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]); STEPS = int(sys.argv[5]) if len(sys.argv) > 5 else 4000
INTERV = os.environ.get("FX_INTERV", ""); iname, iT = (INTERV.split(":") + ["0"])[:2] if INTERV else ("", "0"); iT = int(iT)
STOP = int(os.environ.get("FX_STOP", 0))
# XIII-D additions: FX_SAVEOPT=1 saves optimiser + rng state at every checkpoint (opt_step<k>.pkl) so that later runs can BRANCH from it;
# FX_BRANCH=<tag>:<k> starts from that run's step-k parameters, optimiser and rng state (bitwise continuation of the source run);
# FX_IWIN=<a>:<b> applies the FX_INTERV mask for steps a <= i < b (default 0:T).  Branch runs get the tag suffix _br<k>.
BRANCH = os.environ.get("FX_BRANCH", ""); IWIN = os.environ.get("FX_IWIN", "")                                     # repro check: stop after STOP steps (LR schedule unchanged)
tag = f"{parent}_fx_L{LS}_W{WS}_D{DS}" + (f"_s{STEPS}" if STEPS != 4000 else "") + (f"_{iname}{os.environ.get('FX_EDGES', '') if iname in ('msk', 'lnz', 'gsg', 'fst', 'vrs', 'gof', 'gsv', 'sha', 'vrg') else ''}{(('lo' if os.environ.get('FX_SHA_LO') == '1' else 'hi') + os.environ.get('FX_SHA_CONTENT', 'real')) if iname == 'sha' else ''}{('s' + os.environ['FX_LSIG']) if iname == 'lnz' else ''}{('b' + os.environ['FX_B']) if iname == 'gof' else ''}T{iT}" if INTERV else "") + (f"_stop{STOP}" if STOP else "") + (f"_wa{os.environ['FX_WA']}" if os.environ.get("FX_WA") else "") + (f"_br{BRANCH.split(':')[1]}" if BRANCH else "") + (f"_w{IWIN.replace(':', '-')}" if IWIN else "") + os.environ.get("FX_TAGSUF", "")
OUT = os.path.join(HERE, "..", "checkpoints", "ft", tag); os.makedirs(OUT, exist_ok=True)
pf = NV.load_parent(parent); fresh = lambda s: N.init("M", NL.ENV, s)
WA0 = np.asarray(fresh(WS + 7777)["WA"]); WAT = os.environ.get("FX_WA", "")          # XIII-D A4: controlled transforms of the fresh readout map
if WAT.startswith("perm"): WA0 = WA0[:, [int(c) for c in WAT[4:]]]                     # column permutation: same spectrum and column space
elif WAT == "neg": WA0 = -WA0                                                         # sign flip: same spectrum and column space
elif WAT.startswith("scale"): WA0 = WA0 * float(WAT[5:])                              # direction-preserving rescaling
elif WAT.startswith("rot"): WA0 = np.linalg.qr(np.random.default_rng(int(WAT[3:])).normal(size=(64, 64)))[0] @ WA0   # input-side rotation: same spectrum, new column space
elif WAT.startswith("ws"):                                                             # transplant: column space of WA from seed a, singular values of the own seed
    Wo = np.asarray(fresh(int(WAT[2:]) + 7777)["WA"]); U, S_, Vt = np.linalg.svd(WA0, full_matrices=False); Uo, So, Vto = np.linalg.svd(Wo, full_matrices=False); WA0 = Uo @ np.diag(S_) @ Vt
t = dict(gf=pf["gf"], bf=pf["bf"], WA=jnp.asarray(WA0, jnp.float32), new=[fresh(LS + 7777)["layers"][2]])
t["emb"] = {k: pf[k] for k in NL.FROZEN_KEYS}; t["pre"] = [pf["layers"][l] for l in range(2)]
NEG = -1e9
def mask_for(layer):
    m = np.zeros((43, 43), np.float32)
    if iname in ("ro0", "ro01") and (layer == 0 or (iname == "ro01" and layer == 1)): m[36:42, 0:36] = NEG
    if iname == "rq1" and layer == 1: m[36:42, 0:36] = NEG
    if iname == "l2q" and layer == 2:
        for qq in range(6): m[36 + qq, qq * 6:(qq + 1) * 6] = NEG
    if iname == "ro0p" and layer == 0:                                  # XIII-E: partial early-route block, k of the 6 sources of every question
        k = int(os.environ["FX_RO0P"])
        for qq in range(6): m[36:42, qq * 6:qq * 6 + k] = NEG
    if iname == "sham" and layer in (0, 2): return jnp.zeros((43, 43), jnp.float32)   # XIII-E: numerical-path sham (masked code path, zero bias)
    if iname == "msk":                                                  # XIV: off-route sham (same -1e9 masking mechanism, off-route edges)
        el, em = MFW.edges(os.environ["FX_EDGES"])
        if layer == el: m[em] = NEG
    if iname in ("gsg", "fst", "vrs", "gof", "gsv", "sha", "vrg"):     # XVIII: the new modes; zero bias marks the layer, block_m does the rest
        el, em = MFW.edges(os.environ["FX_EDGES"])
        if layer == el: return jnp.zeros((43, 43), jnp.float32)
    if iname == "lnz":                                                  # XIV: location-matched logit noise; zero bias marks the layer
        el, em = MFW.edges(os.environ["FX_EDGES"])
        if layer == el: return jnp.zeros((43, 43), jnp.float32)
    return jnp.asarray(m) if m.any() else None
LNZ = (MFW.edges(os.environ["FX_EDGES"])[1].astype(np.float32), float(os.environ["FX_LSIG"])) if iname == "lnz" else None; LKEY = jax.random.PRNGKey(525252)
XNEW = ("gsg", "fst", "vrs", "gof", "gsv", "sha", "vrg")
XM = jnp.asarray(MFW.edges(os.environ["FX_EDGES"])[1].astype(np.float32)) if iname in XNEW else None; XB = float(os.environ.get("FX_B", 0))
SHA_ROLL = os.environ.get("FX_SHA_CONTENT", "real") == "rolled"
if iname == "sha":                                                       # schedule A*[i, h] from the paired baseline's slog.json
    _sl = json.load(open(os.environ["FX_SHA_SCHED"]))["shares"]; _A = np.array([r[1] for r in _sl], np.float32)
    if os.environ.get("FX_SHA_LO") == "1": _A = np.repeat(_A[:1], len(_A), 0)
    SHA_A = jnp.asarray(_A)
def sha_ell(s):
    """log-odds l[n,h,q] = logsumexp_T(s) - logsumexp_C(s) on the 6 readout rows (each row's T = its own-question sources)."""
    sr = s[:, :, 36:42, :]; m = XM[36:42][None, None]
    lT = jax.nn.logsumexp(jnp.where(m > 0, sr, -1e30), -1); lC = jax.nn.logsumexp(jnp.where(m > 0, -1e30, sr), -1); return lT - lC
def sha_beta(s, step_i):
    l = sha_ell(jax.lax.stop_gradient(s)); tgt = SHA_A[jnp.minimum(step_i, SHA_A.shape[0] - 1)]; b = jnp.zeros(l.shape[1])
    for _ in range(8):
        p = jax.nn.sigmoid(b[None, :, None] + l); f = p.mean((0, 2)) - tgt; d = (p * (1 - p)).mean((0, 2)); b = jnp.clip(b - f / jnp.maximum(d, 1e-6), -20.0, 20.0)
    return b
def block_m(lay, x, bias, step_i=0):
    z = N.ln(x, lay["g1"], lay["b1"])
    q = jnp.einsum("ntd,hde->nhte", z, lay["WQ"]); k = jnp.einsum("ntd,hde->nhte", z, lay["WK"]); v = jnp.einsum("ntd,hde->nhte", z, lay["WV"])
    s = jnp.einsum("nhte,nhse->nhts", q, k) / jnp.sqrt(q.shape[-1])
    if bias is not None: s = s + bias
    if LNZ is not None: s = s + LNZ[1] * jnp.asarray(LNZ[0]) * jax.random.normal(jax.random.fold_in(LKEY, step_i), s.shape)
    if XM is not None:                                                  # XVIII modes (XM = target edge mask, float 0/1, (43,43))
        if iname in ("gsg", "gsv"): s = XM * jax.lax.stop_gradient(s) + (1.0 - XM) * s
        if iname == "gof": s = s - XB * XM
        if iname == "fst":
            af = jax.nn.softmax(s, -1); ab = jax.nn.softmax(s + NEG * XM, -1); a = af + jax.lax.stop_gradient(ab - af)
        elif iname == "sha":
            s = s + jax.lax.stop_gradient(sha_beta(s, step_i))[None, :, None, None] * XM; a = jax.nn.softmax(s, -1)
        else: a = jax.nn.softmax(s, -1)
        if iname in ("vrs", "vrg") or (iname == "sha" and SHA_ROLL):
            vr = jnp.roll(v, 1, axis=0); vr = jax.lax.stop_gradient(vr) if iname in ("vrg", "sha") else vr
            ctx = jnp.einsum("nhts,nhse->nhte", a * (1.0 - XM), v) + jnp.einsum("nhts,nhse->nhte", a * XM, vr)
        elif iname == "gsv":
            ctx = jnp.einsum("nhts,nhse->nhte", a * (1.0 - XM), v) + jax.lax.stop_gradient(jnp.einsum("nhts,nhse->nhte", a * XM, v))
        else: ctx = jnp.einsum("nhts,nhse->nhte", a, v)
        x = x + jnp.einsum("nhte,hed->ntd", ctx, lay["WO"])
        h = jax.nn.gelu(N.ln(x, lay["g2"], lay["b2"]) @ lay["W1"] + lay["c1"], approximate=True); return x + h @ lay["W2"] + lay["c2"]
    a = jax.nn.softmax(s, -1); x = x + jnp.einsum("nhte,hed->ntd", jnp.einsum("nhts,nhse->nhte", a, v), lay["WO"])
    h = jax.nn.gelu(N.ln(x, lay["g2"], lay["b2"]) @ lay["W1"] + lay["c1"], approximate=True); return x + h @ lay["W2"] + lay["c2"]
NOISE = float(os.environ.get("FX_NOISE", 0)); NKEY = jax.random.PRNGKey(424242)
def logits_X(tt, X, masked=False, step_i=0):
    x = N.embed(dict(tt["emb"]), X)
    for l, lay in enumerate(tt["pre"]):
        bias = mask_for(l) if masked else None
        x = NL.block(lay, x, l=l) if bias is None else block_m(lay, x, bias, step_i)
    if masked and iname == "noise": x = x + NOISE * jax.random.normal(jax.random.fold_in(NKEY, step_i), x.shape)   # XIII-E general-disturbance control
    for i, lay in enumerate(tt["new"]):
        bias = mask_for(2) if masked else None
        x = NL.block(lay, x, l=10 + i) if bias is None else block_m(lay, x, bias, step_i)
    return N.ln(x, tt["gf"], tt["bf"])[:, 36:42] @ tt["WA"]
opt = optax.adamw(optax.warmup_cosine_decay_schedule(0.0, 1e-3, min(200, STEPS // 10), STEPS), weight_decay=1e-4); st = opt.init(t)
def lossf(tt, X, Y, masked, step_i=0):
    lp = jax.nn.log_softmax(logits_X(tt, X, masked, step_i), -1); return (-jnp.take_along_axis(lp, Y[..., None], -1)[..., 0]).mean()
def mkstep(masked):
    @jax.jit
    def step(tt, st, X, Y, step_i=0):
        l, g = jax.value_and_grad(lambda a: lossf(a, X, Y, masked, step_i))(tt); up, st = opt.update(g, st, tt); return optax.apply_updates(tt, up), st, l
    return step
SLOG = os.environ.get("FX_SLOG", ""); SLOGT = int(os.environ.get("FX_SLOGT", 100)); slog = []
def _att_at(tt, X, lay_i, masked, step_i):
    """attention logits s (n,h,43,43) at layer lay_i under the run's forward (plain, or the masked forward of this mode)."""
    x = N.embed(dict(tt["emb"]), X); lays = list(tt["pre"]) + list(tt["new"])
    for l in range(lay_i):
        bias = mask_for(l if l < 2 else 2) if masked else None
        x = (NL.block(lays[l], x, l=l if l < 2 else 10) if bias is None else block_m(lays[l], x, bias, step_i))
    lay = lays[lay_i]; z = N.ln(x, lay["g1"], lay["b1"]); q = jnp.einsum("ntd,hde->nhte", z, lay["WQ"]); k = jnp.einsum("ntd,hde->nhte", z, lay["WK"])
    return jnp.einsum("nhte,nhse->nhts", q, k) / jnp.sqrt(q.shape[-1])
if SLOG:
    _SL_l, _SL_m = MFW.edges(SLOG); _SL_m = jnp.asarray(_SL_m.astype(np.float32))
    _slf = jax.jit(lambda tt, X: (jax.nn.softmax(_att_at(tt, X, _SL_l, False, 0), -1)[:, :, 36:42] * _SL_m[None, None, 36:42]).sum(-1).mean((0, 2)))
GLOG = os.environ.get("FX_GLOG") == "1" and iname in XNEW; glog = []
if GLOG and iname == "sha":
    _EL = MFW.edges(os.environ["FX_EDGES"])[0]
    def _shad(tt, X, step_i):
        sraw = _att_at(tt, X, _EL, True, step_i); b = sha_beta(sraw, step_i); a = jax.nn.softmax(sraw + b[None, :, None, None] * XM, -1)
        return jnp.concatenate([b, (a[:, :, 36:42] * XM[None, None, 36:42]).sum(-1).mean((0, 2))])
    _shadj = jax.jit(_shad)
if GLOG:
    def _gnf(tt, X, Y, step_i):
        g = jax.grad(lambda a: lossf(a, X, Y, True, step_i))(tt)
        return jnp.stack([optax.global_norm(g), optax.global_norm(g["emb"]), optax.global_norm(g["pre"][0]), optax.global_norm(g["pre"][1]), optax.global_norm(g["new"]), optax.global_norm(g["WA"])])
    _gn = jax.jit(_gnf)
step_plain, step_mask = mkstep(False), (mkstep(True) if INTERV else None)
fwd = jax.jit(logits_X)
def fn_of(tt): return lambda X: np.concatenate([np.asarray(fwd(tt, jnp.asarray(X[i:i + 1000]))) for i in range(0, len(X), 1000)])
rng = np.random.default_rng(1_000_003 * (DS + 1) + 7919 * 4 + 31 * 3 + 17)                  # identical to native_train.py full_c1 with seed DS
snaps = sorted(set([int(round(x)) for x in np.geomspace(50, STEPS, 12)] + [25, 100, 150, 200, 300, 400, 600, 800, 1000, 1500, 2000, 3000, STEPS] + [int(a) for a in os.environ.get("FX_EXTRA_SNAPS", "").split(",") if a]))
snaps = [s for s in snaps if s <= STEPS]; t0 = time.time(); hist = []
if os.path.exists(os.path.join(OUT, "final.json")) and not STOP: print(f"[{tag}] already complete", flush=True); sys.exit(0)
start = 0; RP_ = os.path.join(OUT, "resume.pkl")
iw0, iw1 = (map(int, IWIN.split(":")) if IWIN else (0, iT))
if BRANCH and not os.path.exists(RP_):
    btag, bk = BRANCH.split(":"); bdir = os.path.join(HERE, "..", "checkpoints", "ft", btag)
    for alt in (os.path.join(NV.X13E, "checkpoints", "ft", btag), os.path.join(NV.X13D, "checkpoints", "ft", btag)):   # XIV: XIV, then XIII-E, then XIII-D
        if not (os.path.isdir(bdir) and os.path.exists(os.path.join(bdir, f"opt_step{BRANCH.split(':')[1]}.pkl"))): bdir = alt
    R = pickle.load(open(os.path.join(bdir, f"opt_step{bk}.pkl"), "rb"))
    start = int(bk); rng.bit_generator.state = R["rng"]; hist = list(R["hist"])
    z = np.load(os.path.join(bdir, f"step{bk}.npz")); t = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(t), [jnp.asarray(z[f"arr_{k}"]) for k in range(len(z.files))])
    st = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(st), [jnp.asarray(a) for a in R["opt"]]); print(f"[{tag}] branched from {btag} step {bk}", flush=True)
if os.path.exists(RP_):   # XIII-E: STOP runs resume too (XIII-D OOM lesson)
    R = pickle.load(open(RP_, "rb")); start = R["step"]; rng.bit_generator.state = R["rng"]; hist = R["hist"]
    z = np.load(os.path.join(OUT, f"step{start}.npz")); t = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(t), [jnp.asarray(z[f"arr_{k}"]) for k in range(len(z.files))])
    st = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(st), [jnp.asarray(a) for a in R["opt"]]); print(f"[{tag}] resumed at step {start}", flush=True)
if STOP and start >= STOP: print(f"[{tag}] already at stop", flush=True); sys.exit(0)
for i in range(start, STEPS):
    d = NL.Bm.sample_fast(NL.ENV, 256, rng); X = jnp.asarray(d["X"]); Y = jnp.asarray(d["Y"])
    if SLOG and i < SLOGT: slog.append((i, [float(x) for x in np.asarray(_slf(t, X))]))
    if GLOG and iw0 <= i < iw1: _g = np.asarray(_gn(t, X, Y, jnp.int32(i))); glog.append((i, float(_g[0])))
    if GLOG and iname == "sha" and iw0 <= i < iw1: _sd = np.asarray(_shadj(t, X, jnp.int32(i)))
    t, st, l = (step_mask(t, st, X, Y, jnp.int32(i)) if (INTERV and iw0 <= i < iw1) else step_plain(t, st, X, Y))
    if GLOG and iw0 <= i < iw1: glog[-1] = glog[-1] + (float(l),) + tuple(float(x) for x in _g[1:])
    if GLOG and iname == "sha" and iw0 <= i < iw1: glog[-1] = glog[-1] + tuple(float(x) for x in _sd)
    if SLOG and i < SLOGT: slog[-1] = slog[-1] + (float(l),)
    if SLOG and i + 1 == SLOGT: json.dump(dict(edges=SLOG, fields="[step, per-head share before update, batch loss]", shares=slog), open(os.path.join(OUT, "slog.json"), "w"))
    if (i + 1) % max(STEPS // 40, 1) == 0: hist.append((i + 1, float(l)))
    if (i + 1) in snaps:
        np.savez(os.path.join(OUT, f"step{i + 1}.npz"), *[np.asarray(x) for x in jax.tree_util.tree_leaves(t)])
        pickle.dump(dict(step=i + 1, rng=rng.bit_generator.state, hist=hist, opt=[np.asarray(a) for a in jax.tree_util.tree_leaves(st)]), open(RP_ + ".tmp", "wb")); os.replace(RP_ + ".tmp", RP_)
        if os.environ.get("FX_SAVEOPT") and (not os.environ.get("FX_SAVEOPT_AT") or str(i + 1) in os.environ["FX_SAVEOPT_AT"].split(",")): pickle.dump(dict(step=i + 1, rng=rng.bit_generator.state, hist=hist, opt=[np.asarray(a) for a in jax.tree_util.tree_leaves(st)]), open(os.path.join(OUT, f"opt_step{i + 1}.pkl"), "wb"))
        print(f"[{tag}] step {i + 1} loss {float(l):.4f} {time.time() - t0:.0f}s", flush=True)
    if GLOG and i + 1 == iw1: json.dump(glog, open(os.path.join(OUT, "window_log.json"), "w"))
    if STOP and i + 1 >= STOP: json.dump(dict(stop=STOP, hist=hist, seconds=time.time() - t0), open(os.path.join(OUT, "stop_done.json"), "w")); sys.exit(0)
    if (i + 1) in snaps and int(os.environ.get("FX_DIE_AFTER", 0)) == i + 1: sys.exit(3)       # resume test only: simulate a crash
ev = NV.load_eval(os.path.join(NV.X13B, "logs", "eval_conf13.npz")); r = NV.ER.run(fn_of(t), ev); s = NV.summarize(r)
json.dump(dict(summary=s, families=r["families"], quality=r["quality"]), open(os.path.join(OUT, "final.json"), "w"), indent=1, default=float)
json.dump(dict(parent=parent, LSEED=LS, WSEED=WS, DSEED=DS, steps=STEPS, interv=INTERV, seconds=time.time() - t0, hist=hist, snaps=snaps, final=s),
          open(os.path.join(OUT, "curve.json"), "w"), indent=1, default=float)
print(f"[{tag}] done acc {s['acc']:.4f} KL {s['KL']:.4f} {time.time() - t0:.0f}s", flush=True)
