"""Generate notebooks/00-04 (thin: configuration + calls into cor_research.pipeline).  python scripts/make_notebooks.py"""
import os, nbformat as nbf
PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")); NB = os.path.join(PKG, "notebooks"); os.makedirs(NB, exist_ok=True)
md = nbf.v4.new_markdown_cell; code = nbf.v4.new_code_cell

CONFIG = '''# ===================== CONFIGURATION (the only cell you need to edit) =====================
import os
PRESET = os.environ.get("COR_NB_PRESET", "pilot")    # "pilot" (technical validation) | "standard" | "extended" | "high"   ("smoke" = CPU software test)
USE_DRIVE = os.environ.get("COR_NB_DRIVE", "1") == "1"   # keep every checkpoint, registry record and result on Google Drive (survives disconnects)
DRIVE_DIR = "/content/drive/MyDrive/COR_XX_RUNS"
PACKAGE_ZIP = "/content/drive/MyDrive/cor_inv20_colab_three_track_research.zip"   # or upload the ZIP to /content
BUDGET_HOURS = None       # stop launching new jobs after this many hours (running jobs finish; nothing is shortened)
CONCURRENCY = "auto"      # "auto" = measured by the benchmark; or an integer
RETRY_FAILED = False      # True: failed jobs get a new attempt (their failed attempts stay in the registry)
DEVICE = os.environ.get("COR_NB_DEVICE", "auto")   # "auto" | "gpu" | "cpu"
ALLOW_IDENTITY_CHANGE = False   # True only if you deliberately continue a run root on a different GPU type / software stack (recorded)
RUN_ROOT = os.environ.get("COR_NB_RUN_ROOT") or None   # None = DRIVE_DIR/<preset> (Drive) or /content/COR_XX_RUNS/<preset>
# (the COR_NB_* environment overrides exist only for the package's automated notebook tests)'''

LOCATE = '''# Locate / extract the package and make it importable (idempotent)
import os, sys, zipfile, glob
CANDIDATES = ["/content/COR_Investigation_XX", os.path.abspath("COR_Investigation_XX"), os.path.abspath("..")]
PKG = next((c for c in CANDIDATES if os.path.exists(os.path.join(c, "cor_research", "pipeline.py"))), None)
if PKG is None:
    if USE_DRIVE and not os.path.isdir("/content/drive/MyDrive"):
        from google.colab import drive; drive.mount("/content/drive")
    zips = [PACKAGE_ZIP] + glob.glob("/content/cor_inv20_colab_three_track_research*.zip")
    z = next((x for x in zips if os.path.exists(x)), None)
    assert z, "package ZIP not found: upload cor_inv20_colab_three_track_research.zip to /content or set PACKAGE_ZIP"
    with zipfile.ZipFile(z) as zf: zf.extractall("/content")
    PKG = "/content/COR_Investigation_XX"
sys.path.insert(0, PKG); os.chdir(PKG); print("package:", PKG)'''

GPU = '''# 1. GPU and memory
import subprocess, shutil
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout if shutil.which("nvidia-smi") else "NO GPU visible (CPU only: Track C and smoke tests only)")
print(subprocess.run(["bash", "-c", "free -g | head -2; nproc"], capture_output=True, text=True).stdout)'''

INSTALL = '''# 2. Exact dependencies (pinned in requirements.txt / requirements-gpu.txt).  If anything had to be installed, the runtime restarts
#    once automatically; then run all cells again from the top (every step is idempotent and resumes).
import importlib.metadata as IM, subprocess, sys, os, shutil
def pins(f): return dict(l.split("#")[0].strip().split("==") for l in open(f) if "==" in l.split("#")[0])
want = pins("requirements.txt"); gpu = shutil.which("nvidia-smi") is not None and DEVICE != "cpu"
if gpu: want.update(pins("requirements-gpu.txt"))
def have(n):
    try: return IM.version(n.split("[")[0])
    except Exception: return None
missing = {n: v for n, v in want.items() if have(n) != v}
if missing:
    print("installing", missing)
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"] + (["-r", "requirements-gpu.txt"] if gpu else []), capture_output=True, text=True)
    print(r.stdout[-2000:], r.stderr[-3000:])
    if r.returncode != 0: raise SystemExit("dependency installation FAILED (see the output above); nothing else will run")
    print("dependencies installed; restarting the runtime: run all cells again"); os.kill(os.getpid(), 9)
else: print("all pinned dependencies present:", want)'''

SETUP = '''# 3. Persistent storage + device (run root on Drive: checkpoints, registry, results)
from cor_research import pipeline as PL
INFO = PL.setup(PRESET, run_root=RUN_ROOT, use_drive=USE_DRIVE, drive_dir=DRIVE_DIR, device=DEVICE, allow_identity_change=ALLOW_IDENTITY_CHANGE); INFO'''

def nb(name, title, cells):
    n = nbf.v4.new_notebook(); n.cells = [md(title), code(CONFIG), code(LOCATE)] + cells
    n.metadata = dict(kernelspec=dict(name="python3", display_name="Python 3", language="python"), accelerator="GPU", colab=dict(provenance=[], gpuType="L4"))
    nbf.write(n, os.path.join(NB, name))

nb("00_RUN_COMPLETE_INVESTIGATION_XX.ipynb", """# COR Investigation XX — complete three-track run
Phase 1.5, Step 1 (≈ 25% provisional).  This notebook runs the whole frozen protocol with the selected preset: Track A (causal
generalisation across independent pretrained parents), Track B (functional algorithm identification), Track C (O/D mathematics).
**Nothing here is a result until the run finishes and is reviewed.**  Re-running any cell resumes from the saved state.
Steps: GPU check → dependencies → package & checkpoint verification → storage → mathematics → freeze verification & preflight
(graph audit, planted scorer tests, measured benchmark, budget) → execution (resumable) → scoring → reports → export.""",
   [code(GPU), code(INSTALL), code(SETUP),
    md("## 4. Verify the package, the vendored checkpoints and the dependency pins"), code("PL.verify_package(PRESET)"),
    md("## 5. Track C mathematics (exact; CPU is enough; independent of all training)"), code("TRACK_C = PL.track_c()"),
    md("## 6. Freeze verification and experimental preflight\nFreeze hashes, intervention graph audit, planted scorer tests, **measured** throughput and the budget of every preset on this runtime."), code("PRE = PL.preflight(preset=PRESET)"),
    md("## 7. Execute the frozen plan (screening → eligibility → confirmation → measurement → Track B)\nSafe to interrupt. Re-run this cell after a disconnect: interrupted jobs resume from their exact saved state."),
    code("SUMMARY = PL.run(PRESET, tracks='ABC', budget_hours=BUDGET_HOURS, concurrency=CONCURRENCY, retry_failed=RETRY_FAILED)"),
    md("## 8. Frozen scorers"), code("PL.score(PRESET)"),
    md("## 9. Reports (RESULTS/) and the 20 integrity checks"), code("REP, INTEGRITY = PL.report(PRESET)"),
    md("## 10. Export a verifiable results archive (+ SHA256 + review bundle)\nReturn this archive for independent analysis. Checkpoints stay in the run root (hashes are in the archive)."), code("ZIP, SHA = PL.export(PRESET)")])

nb("01_TRACK_A_CAUSAL_GENERALIZATION.ipynb", "# Track A — causal generalisation across independent pretrained parents\nSame code as notebook 00 (cor_research.pipeline); Track A jobs only.",
   [code(GPU), code(INSTALL), code(SETUP), code("PL.verify_package(PRESET)"), code("PRE = PL.preflight(preset=PRESET)"),
    code("SUMMARY = PL.run(PRESET, tracks='A', budget_hours=BUDGET_HOURS, concurrency=CONCURRENCY, retry_failed=RETRY_FAILED)"), code("PL.score(PRESET)"),
    code("REP, INTEGRITY = PL.report(PRESET)\nprint(open(os.path.join(INFO['run_root'], 'RESULTS', 'TRACK_A_CAUSAL_GENERALIZATION.md')).read())")])

nb("02_TRACK_B_FUNCTIONAL_ALGORITHMS.ipynb", "# Track B — are EARLY and MIXTURE different algorithms?\nTrack B analyses the confirmation anchors of Track A (they are trained here if not yet present). Same code as notebook 00.",
   [code(GPU), code(INSTALL), code(SETUP), code("PL.verify_package(PRESET)"), code("PRE = PL.preflight(preset=PRESET)"),
    code("SUMMARY = PL.run(PRESET, tracks='AB', budget_hours=BUDGET_HOURS, concurrency=CONCURRENCY, retry_failed=RETRY_FAILED)"), code("PL.score(PRESET)"),
    code("REP, INTEGRITY = PL.report(PRESET)\nprint(open(os.path.join(INFO['run_root'], 'RESULTS', 'TRACK_B_FUNCTIONAL_IDENTIFICATION.md')).read())")])

nb("03_TRACK_C_OD_MATHEMATICS.ipynb", "# Track C — O/D mathematics (exact; runs on CPU; independent of any training)\nOriginal O/D (Phase-1 definitions), the repaired O/D of the project note (independent implementation), T4 interaction-order bound, and the predeclared C1–C5 classification. No neural O/D experiment is run unless a distinctive prediction exists.",
   [code(INSTALL), code(SETUP.replace("device=DEVICE", "device='cpu' if DEVICE == 'auto' else DEVICE")), code("TRACK_C = PL.track_c()"),
    code("import json\nfor k, v in TRACK_C['repaired_gate'].items(): print(k, {x: v[x] for x in ('R1_max_err_signed_identity', 'M1_O_is_function_of_sorted_posterior', 'R3_gate_flip_following_repaired', 'LP_max_margin_by_order')})\nprint(json.dumps(TRACK_C['t4_order']['statement']))")])

nb("04_ANALYZE_AND_VERIFY_RESULTS.ipynb", "# Analyse and verify results\nScores, RESULTS/, the 20 integrity checks, the review bundle and the export archive. Uses the same modules as notebook 00.",
   [code(INSTALL), code(SETUP), code("PL.score(PRESET)"), code("REP, INTEGRITY = PL.report(PRESET)"),
    code("print(open(os.path.join(INFO['run_root'], 'RESULTS', 'MAIN_RESEARCH_SYNTHESIS.md')).read())"), code("ZIP, SHA = PL.export(PRESET)")])
print("notebooks written")
