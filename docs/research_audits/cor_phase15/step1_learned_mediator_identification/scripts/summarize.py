import json, sys, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for tag in sys.argv[1:]:
    f = ROOT / "logs" / f"protocol_{tag}.json"
    if not f.exists(): continue
    d = json.load(open(f)); cnt = collections.defaultdict(collections.Counter)
    for k, v in d.items():
        task, arch = k.split("/")[:2]
        for s, r in v.items():
            cnt[(task, arch, s)][r.get("verdict", "AGG-LIKE(no differing decisions)")] += 1
    print(f"== {tag}")
    for key, c in sorted(cnt.items()):
        print("  ", key, dict(c))
