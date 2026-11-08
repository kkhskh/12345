"""Checkpoint I/O: nested param dicts <-> flat .npz (no pickle needed to load)."""
import numpy as np


def flatten(p, prefix=""):
    out = {}
    for k, v in p.items():
        if k == "layers":
            for i, lay in enumerate(v):
                out.update(flatten(lay, f"{prefix}layers.{i}."))
        elif isinstance(v, dict):
            out.update(flatten(v, f"{prefix}{k}."))
        else:
            out[prefix + k] = np.asarray(v)
    return out


def unflatten(flat):
    p = {}
    layers = {}
    for k, v in flat.items():
        parts = k.split(".")
        if parts[0] == "layers":
            layers.setdefault(int(parts[1]), {})[".".join(parts[2:])] = v
        elif len(parts) == 1:
            p[k] = v
        else:
            p.setdefault(parts[0], {})[".".join(parts[1:])] = v
    if layers:
        p["layers"] = [layers[i] for i in sorted(layers)]
    return p


def save(path, p):
    np.savez(path, **flatten(p))


def load(path):
    with np.load(path) as z:
        return unflatten({k: z[k] for k in z.files})
