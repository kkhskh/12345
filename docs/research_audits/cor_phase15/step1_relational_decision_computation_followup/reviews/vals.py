import json,re,sys,runpy
sys.argv=['x','--units','/tmp/review13b_integrity/udev.txt','--sfx','_dev13b']
g=runpy.run_path('/tmp/review13b_integrity/score_copy.py')
SPEC=g['SPEC']; get=g['get']
for u in [l.strip() for l in open('/tmp/review13b_integrity/udev.txt') if l.strip()]:
  print('==',u)
  for P in SPEC['predictions']:
    for A in P['arms']:
      for key in ('test','null'):
        if key in A:
          ms=re.findall(r"\{([^}]+)\}",A[key]); vs={m:get(u,m) for m in ms if not m.startswith('exp')}
          print(P['id'],A['id'],key,{k:(round(v,3) if isinstance(v,float) else v) for k,v in vs.items()})
