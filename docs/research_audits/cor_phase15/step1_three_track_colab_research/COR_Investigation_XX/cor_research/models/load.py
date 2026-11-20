"""Model loading for every track: a fine-tuned run directory + parent name + step -> the historical model dict (xviiim.load_dir:
parent embeddings and layers 0-1 replaced by the run's trained copies, the fresh layer 2, final LN, readout map WA).
Parents: uP16x (vendored historical XIII cohort, read-only) and nP<seed> (XX fresh parents under $COR_PARENT_ROOT)."""
import os
from cor_research import env as E


def load(run_dir, parent, step):
    E.patch_parents(); import xviiim as XM
    if not os.path.exists(os.path.join(run_dir, f"step{step}.npz")): raise FileNotFoundError(f"{run_dir}/step{step}.npz")
    return XM.load_dir(run_dir, parent, step)


def forward_fn(M):
    import jax, xviiim as XM
    f = jax.jit(lambda X_: XM.F.forward(M, X_)); return lambda X: XM.F.batched(f, X)
