"""XIII-C shared helpers: model loading for factorial (ft_factorial) and XIII-B checkpoints at any step, and unit naming.
Unit spec strings:
  'uP161:fx:21:21:22'            -> ft_factorial L=21 W=21 D=22  (this package, checkpoints/ft/)
  'uP161:fx:21:21:22:ro0T800'    -> same with training intervention suffix
  'uP161:s:22'                   -> XIII-B / XIII native_train full_c1 seed 22 (eval_conf13 tag), identical to fx:22:22:22
optional '@<step>' suffix selects a checkpoint step (default 4000)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, jax, jax.numpy as jnp
import native as NV, ftlib as F
NL, N = NV.NL, NV.N
PKG = os.path.join(HERE, "..")


def unit_dir(spec):
    spec = spec.split("@")[0]; p = spec.split(":")
    if p[1] == "fx":
        tag = f"{p[0]}_fx_L{p[2]}_W{p[3]}_D{p[4]}" + (f"_{p[5]}" if len(p) > 5 else "")
        own = os.path.join(PKG, "checkpoints", "ft", tag)
        for d in (own, os.path.join(NV.X13E, "checkpoints", "ft", tag), os.path.join(NV.X13D, "checkpoints", "ft", tag), os.path.join(NV.X13C, "checkpoints", "ft", tag)):   # XIV, then XIII-E, XIII-D, XIII-C
            if os.path.isdir(d): return d
        return own
    if p[1] == "s": return F.model_dir(f"{p[0]}_full_c1_s4000_seed{p[2]}_eval_conf13")
    raise ValueError(spec)


def factors(spec):
    """(parent, L, W, D, interv) for a unit spec; XIII-B seed s -> (s, s, s)."""
    p = spec.split("@")[0].split(":")
    if p[1] == "s": s = int(p[2]); return p[0], s, s, s, ""
    return p[0], int(p[2]), int(p[3]), int(p[4]), (p[5] if len(p) > 5 else "")


def load(spec, step=None):
    if "@" in spec and step is None: step = int(spec.split("@")[1])
    step = step or 4000; parent = spec.split(":")[0]; D = unit_dir(spec); pf = NV.load_parent(parent)
    fresh = lambda s: N.init("M", NL.ENV, s)
    tmpl = dict(gf=pf["gf"], bf=pf["bf"], WA=fresh(7777)["WA"], new=[fresh(7777)["layers"][2]], emb={k: pf[k] for k in NL.FROZEN_KEYS}, pre=[pf["layers"][l] for l in range(2)])
    z = np.load(os.path.join(D, f"step{step}.npz"))
    t = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(tmpl), [jnp.asarray(z[f"arr_{i}"]) for i in range(len(z.files))])
    return dict(tag=os.path.basename(D) + f"@{step}", parent=parent, spec="full_c1", emb=t["emb"], layers=list(t["pre"]) + [t["new"][0]], gf=t["gf"], bf=t["bf"], WA=t["WA"], dir=D, step=step)


def steps_available(spec):
    D = unit_dir(spec); return sorted(int(f[4:-4]) for f in os.listdir(D) if f.startswith("step") and f.endswith(".npz"))
