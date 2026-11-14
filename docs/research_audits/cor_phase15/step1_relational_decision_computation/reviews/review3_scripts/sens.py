import json, os, sys, math
PK="/home/claude/12345/docs/research_audits/cor_phase15/step1_relational_decision_computation"
HERE=PK+"/scripts"; sys.path.insert(0,HERE)
os.environ["SCORE_SFX"]="_conf"; sys.argv=["score_xiii.py"]
src=open(HERE+"/score_xiii.py").read(); src=src[:src.index("out = dict(units=UNITS")]
g={"__file__":HERE+"/score_xiii.py","__name__":"sc"}; exec(compile(src,"defs","exec"),g)
SPEC=g["SPEC"]; U=lambda p:f"u{p}:full_c1:20:_eval_conf13"
orig_mc=g["mixture_class"]
def score(units_mix, units_all, label, share_th=None):
    if share_th is not None:
        g["SPEC"]["mixture_class_rule"]["share"]=share_th
    print("==",label)
    for P in SPEC["predictions"]:
        if P["id"] in ("P6",): continue
        units = units_mix if P.get("mixture_only") else units_all
        row=[]; verdict="confirmed"
        for A in P["arms"]:
            res={u:g["evaluate_expr"](u,A["test"]) for u in units}
            n=len(units); k=sum(bool(v) for v in res.values())
            st="pass" if k>=math.ceil(0.75*n) else "fail"
            if st=="fail": verdict="falsified"
            row.append(f"{A['id']}:{k}/{n}"+("" if st=="pass" else "F")+" ["+",".join(u[1:5]+("+" if v else "-" if v is not None else "?") for u,v in res.items())+"]")
        print(P["id"],verdict," ".join(row))
base=[U(p) for p in ("P161","P165")]
comp=[U(p) for p in ("P160","P161","P163","P164","P165")]
score(base,comp,"frozen (161,165)")
score(base+[U("P164")],comp,"+uP164 as mixture (n=3)")
# keep top_mix working for 164 by lowering share threshold to 0.19
score(base+[U("P164")],comp,"+uP164, share th 0.19 (so P7 top_mix defined)",share_th=0.19)
g["SPEC"]["mixture_class_rule"]["share"]=0.2
score(base+[U("P162")],comp,"+uP162 (incomplete) as unit (n=3)")
score([U("P162")],comp,"uP162 alone (mechanism arms per-unit)")
score([U("P164")],comp,"uP164 alone, th .2")
g["SPEC"]["mixture_class_rule"]["share"]=0.19
score([U("P164")],comp,"uP164 alone, th .19")
score(base+[U("P162"),U("P164")],comp+[U("P162")],"all 4 mixture-like (161,162,164,165) n=4, th .19")
