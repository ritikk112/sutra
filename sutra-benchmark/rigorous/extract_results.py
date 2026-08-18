#!/usr/bin/env python3
import json, re
# The two workflow returns (later run supersedes; merge, preferring non-null score)
FILES=["/tmp/claude-1000/-home-ritik-Desktop-sutra/f2ce7530-a1a4-4531-9856-7d9aa4558a38/tasks/wo6t2wal0.output",
       "/tmp/claude-1000/-home-ritik-Desktop-sutra/f2ce7530-a1a4-4531-9856-7d9aa4558a38/tasks/wd30emknp.output"]
def load_array(path):
    try: txt=open(path).read()
    except FileNotFoundError: return []
    i=txt.find("[")
    if i<0: return []
    try: return json.loads(txt[i:])
    except Exception as e:
        # try to locate the outermost array
        try: return json.loads(txt[i:txt.rfind("]")+1])
        except Exception: print("parse fail",path,e); return []
merged={}
for path in FILES:
    for r in load_array(path):
        c=r.get("cell")
        if not c: continue
        if c not in merged: merged[c]=r
        elif merged[c].get("score") is None and r.get("score") is not None: merged[c]=r
        elif r.get("score") is not None and merged[c].get("score") is not None:
            pass
# expected 108 cells
ARMS=["SUTRA_ONLY","GREP_ONLY","BOTH"]; TR=[1,2,3]
TIX={"frappe":["FL1","FL2","FL3","FS1","FS2","FS3"],"dify":["DL1","DL2","DL3","DS1","DS2","DS3"]}
allcells=[f"{r}|{t}|{a}|t{n}" for r in TIX for t in TIX[r] for a in ARMS for n in TR]
have=[c for c in allcells if c in merged]
missing_answer=[c for c in allcells if c not in merged or not merged.get(c,{}).get("answer")]
missing_score=[c for c in allcells if merged.get(c,{}).get("score") is None]
print(f"cells present={len(have)}/108  missing_answer={len(missing_answer)}  missing_score={len(missing_score)}")
print("missing_score cells:", sorted(missing_score))
# save answers+scores
json.dump(merged, open("/home/ritik/Desktop/sutra/sutra-benchmark/rigorous/answers_scores.json","w"), indent=1)
print("wrote answers_scores.json")
# constraint violations
d=json.load(open("/home/ritik/Desktop/sutra/sutra-benchmark/rigorous/cost_constraints.json"))
print("\nconstraint-violating cells:")
for cell,row in d.items():
    if row["violations"]>0:
        print(f"  {cell}: {row['violations']}/{row['trials']} trials; sutra.max={row['sutra']['max']} grep.max={row['grep']['max']}")
