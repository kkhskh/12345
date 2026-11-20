"""Investigation XX environment: package paths, run/parent roots, device detection, deterministic XLA flags, legacy import paths.

Import this module FIRST in every process (it sets XLA_FLAGS before JAX is imported).
  COR_RUN_ROOT     where all run outputs go (Drive or local scratch); default <package>/RUNS
  COR_PARENT_ROOT  where fresh XX parents are written/read; default <COR_RUN_ROOT>/parents
  COR_DEVICE       'auto' (default), 'gpu' or 'cpu'
Determinism: CPU uses single-threaded Eigen (bitwise reproducible, the XIII-XIX setting); GPU uses --xla_gpu_deterministic_ops=true and
disables autotuning differences and forces full-fp32 matmuls (JAX_DEFAULT_MATMUL_PRECISION=highest): run-to-run deterministic on the same GPU
model and software stack (verified on the runtime by the preflight resume/determinism test); NOT bitwise identical to CPU.
Run identity: identity() describes the numerical environment; the pipeline records it once per run root (RUN_IDENTITY.json) and refuses
to run any job on a different identity (review C B2)."""
import os, sys, json, hashlib
PKG = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
LEGACY = os.path.join(PKG, "cor_research", "legacy", "cor_phase15")
HARNESS = os.path.join(LEGACY, "step1_training_state_causal_identification", "scripts", "harness_xx.py")
RUN_ROOT = os.path.abspath(os.environ.setdefault("COR_RUN_ROOT", os.path.join(PKG, "RUNS")))
PARENT_ROOT = os.path.abspath(os.environ.setdefault("COR_PARENT_ROOT", os.path.join(RUN_ROOT, "parents")))
CPU_FLAGS = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
GPU_FLAGS = "--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0"


def has_gpu():
    if os.environ.get("COR_DEVICE", "auto") == "cpu": return False
    try:
        import subprocess
        return subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=20).returncode == 0
    except Exception: return False


def configure(device=None):
    """Set XLA flags and JAX platform before JAX is imported.  Returns 'gpu' or 'cpu'."""
    dev = device or os.environ.get("COR_DEVICE", "auto")
    if dev == "auto": dev = "gpu" if has_gpu() else "cpu"
    os.environ["COR_DEVICE"] = dev
    if dev == "cpu":
        os.environ["XLA_FLAGS"] = CPU_FLAGS; os.environ["JAX_PLATFORMS"] = "cpu"
    else:
        os.environ["XLA_FLAGS"] = GPU_FLAGS; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"
        os.environ["JAX_PLATFORMS"] = "cuda"                          # review C r2 N6: never fall back to CPU silently in GPU mode       # review C B2: no TF32 on Ampere+ (fp32 matmuls, as on the historical CPU runs)
        os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    return dev


def legacy_paths():
    for p in ("step1_training_state_causal_identification/src", "step1_pathway_selection_causal_identification/src", "step1_cross_family_relational_foundations/src",
              "step1_relational_acquisition_limits/src", "step1_foundational_atlas_continuation/src", "step1_latent_reliability/src", "step1_relational_computation_atlas/src"):
        q = os.path.join(LEGACY, p)
        if q not in sys.path: sys.path.insert(0, q)


def patch_parents():
    """Make legacy native.parent_path resolve XX fresh parents 'nP<seed>' -> PARENT_ROOT/zP<seed>/pre.npz."""
    legacy_paths(); import native as NV
    if getattr(NV, "_xx_patched", False): return NV
    pp = NV.parent_path
    def parent_path(name):
        if name.startswith("n"): return os.path.join(os.environ["COR_PARENT_ROOT"], "z" + name[1:], "pre.npz")
        return pp(name)
    NV.parent_path = parent_path; NV._xx_patched = True; return NV


def sha256(path, bs=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(bs), b""): h.update(b)
    return h.hexdigest()


def atomic_json(obj, path):
    tmp = path + ".tmp"; os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(tmp, "w") as f: json.dump(obj, f, indent=1, default=float)
    os.replace(tmp, path)


def read_json(path, default=None):
    try:
        with open(path) as f: return json.load(f)
    except Exception: return default


def identity():
    """the numerical environment of this process (compared against RUN_IDENTITY.json before any job runs)"""
    import importlib.metadata as IM, platform, subprocess, shutil
    v = {}
    for n in ("jax", "jaxlib", "optax", "numpy", "scipy", "jax-cuda12-plugin", "jax-cuda12-pjrt"):
        try: v[n] = IM.version(n)
        except Exception: v[n] = None
    for d in IM.distributions():                                          # CUDA runtime wheels (review C r2 N6): part of the identity
        nm = (d.metadata["Name"] or "").lower()
        if nm.startswith("nvidia-"): v[nm] = d.version
    gpu = None
    if os.environ.get("COR_DEVICE") == "gpu" and shutil.which("nvidia-smi"):
        try: gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip().splitlines()[0]   # model, not driver patch
        except Exception: gpu = "unknown"
    fz = read_json(os.path.join(PKG, "protocols", "FREEZE.json")) or {}
    return dict(device=os.environ.get("COR_DEVICE"), gpu=gpu, python=".".join(platform.python_version().split(".")[:2]), packages=v, xla_flags=os.environ.get("XLA_FLAGS"),
                matmul_precision=os.environ.get("JAX_DEFAULT_MATMUL_PRECISION"), freeze_sha256=fz.get("freeze_sha256"))
