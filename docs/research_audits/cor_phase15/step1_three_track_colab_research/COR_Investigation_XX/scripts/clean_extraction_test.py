"""Clean-environment pre-delivery test (SOFTWARE VALIDATION ONLY).  python scripts/clean_extraction_test.py ZIP OUT_DIR [--smoke]
Extracts the ZIP into an empty directory and, from the extracted copy only (no access to the build tree): verifies the package
manifest, the vendored checkpoint hashes, the dependency pins and the freeze; validates every notebook (nbformat) and that every
module they import exists; runs the tests (pretraining equivalence + resume, tag rules, planted scorers, verdict OC, Track B OC, Track C
regression); executes notebook 03 (Track C, CPU) with nbclient; optionally (--smoke) runs the complete smoke pipeline.
Writes OUT_DIR/CLEAN_EXTRACTION_REPORT.json.  CPU only: it does not test a GPU."""
import os, sys, json, zipfile, subprocess, time, shutil
zp, out = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]); os.makedirs(out, exist_ok=True); ext = os.path.join(out, "extract"); shutil.rmtree(ext, ignore_errors=True)
R = dict(zip=zp, started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), steps={})
with zipfile.ZipFile(zp) as zf: R["zip_test"] = zf.testzip() is None; zf.extractall(ext)
PKG = os.path.join(ext, "COR_Investigation_XX"); env = dict(os.environ, PYTHONPATH=PKG, COR_DEVICE="cpu", COR_TEST_OUT=os.path.join(out, "test_out"), COR_RUN_ROOT=os.path.join(out, "runs"))
os.makedirs(env["COR_TEST_OUT"], exist_ok=True)
def step(name, args, timeout=3600, cwd=PKG):
    t = time.time(); r = subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
    R["steps"][name] = dict(rc=r.returncode, seconds=round(time.time() - t, 1), tail=(r.stdout[-1500:] + r.stderr[-1500:]) if r.returncode else r.stdout[-400:]); print(name, r.returncode, flush=True)
code = """
import os, json; from cor_research.verification import integrity as IN
r = IN.run('standard'); print(json.dumps({k: r['checks'][k]['status'] for k in ('01_code_version', '02_dependencies', '03_checkpoint_provenance', '05_frozen_protocol_hashes')}))
import sys; sys.exit(0 if all(r['checks'][k]['status'] == 'PASS' for k in ('01_code_version', '02_dependencies', '03_checkpoint_provenance', '05_frozen_protocol_hashes')) else 1)"""
step("package_manifest_checkpoints_deps_freeze", [sys.executable, "-c", code])
nbcode = """
import nbformat, glob, re, importlib, sys, os
bad = []
for f in sorted(glob.glob('notebooks/*.ipynb')):
    nb = nbformat.read(f, as_version=4); nbformat.validate(nb)
    src = '\\n'.join(c.source for c in nb.cells if c.cell_type == 'code')
    for m in set(re.findall(r'from (cor_research[\\w.]*) import', src)) | set(re.findall(r'PL\\.(\\w+)\\(', src)):
        try:
            if m.startswith('cor_research'): importlib.import_module(m)
            else: getattr(importlib.import_module('cor_research.pipeline'), m)
        except Exception as e: bad.append((f, m, str(e)))
print(bad); sys.exit(1 if bad else 0)"""
step("notebooks_validate_and_imports", [sys.executable, "-c", nbcode])
for t in ("test_pretrain.py", "test_tags.py", "test_scorers_planted.py", "test_verdict_oc.py", "test_trackb_oc.py", "test_track_c.py"): step(t, [sys.executable, os.path.join("tests", t)])
exe = """
import nbformat, os, sys; from nbclient import NotebookClient
nb = nbformat.read('notebooks/03_TRACK_C_OD_MATHEMATICS.ipynb', as_version=4)
NotebookClient(nb, timeout=3600, kernel_name='python3', resources={'metadata': {'path': os.getcwd()}}).execute()
nbformat.write(nb, os.path.join(os.environ['COR_TEST_OUT'], '03_executed.ipynb')); print('executed', len(nb.cells), 'cells')"""
env.update(COR_NB_PRESET="standard", COR_NB_DRIVE="0", COR_NB_DEVICE="cpu", COR_NB_RUN_ROOT=os.path.join(out, "runs", "nb03"))
step("notebook_03_executed_nbclient", [sys.executable, "-c", exe], timeout=3600, cwd=os.path.join(PKG, "notebooks"))
if "--smoke" in sys.argv: step("smoke_pipeline", [sys.executable, os.path.join("scripts", "smoke_pipeline.py"), os.path.join(out, "smoke"), "--concurrency", "2"], timeout=7200)
R["all_pass"] = R["zip_test"] and all(v["rc"] == 0 for v in R["steps"].values()); R["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
json.dump(R, open(os.path.join(out, "CLEAN_EXTRACTION_REPORT.json"), "w"), indent=1); print("ALL PASS" if R["all_pass"] else "FAILURES"); sys.exit(0 if R["all_pass"] else 1)
