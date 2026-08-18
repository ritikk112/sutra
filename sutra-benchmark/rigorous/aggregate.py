#!/usr/bin/env python3
"""Final aggregation: merge 90 base grades (results.json) + 18 DS2/DS3 grades (grade_missing output)
+ 3 clean violator re-runs (new transcript dir). Produce per-cell median+range, per-arm x class x repo
aggregates, pre-registered conclusions, and raw JSON. Cost = Sonnet-5 rates."""
import json, glob, re, statistics as st
from collections import defaultdict

BASE=json.load(open("/home/ritik/Desktop/sutra/sutra-benchmark/rigorous/results.json"))
RERUN_DIR="/home/ritik/.claude-account1/projects/-home-ritik-Desktop-sutra/5d5da943-8e2c-467b-8fbe-07a9b7f411aa/subagents/workflows/wf_f33c6c0e-c9a"
GM_OUT="/tmp/claude-1000/-home-ritik-Desktop-sutra/f2ce7530-a1a4-4531-9856-7d9aa4558a38/tasks/w25200h7v.output"
MISSING=json.load(open("/home/ritik/Desktop/sutra/sutra-benchmark/rigorous/missing_answers.json"))
STD=dict(IN=3.0,OUT=15.0,CR=0.30,CW=3.75)

def load_arr(path):
    txt=open(path).read(); i=txt.find("[")
    obj,_=json.JSONDecoder().raw_decode(txt[i:])
    return obj

# 1) merge 18 DS2/DS3 grades (idx -> score) onto groups
gm=load_arr(GM_OUT)
ds_scores=defaultdict(list)
for r in gm:
    ds_scores[r["group"]].append(r["score"])
for grp,scs in ds_scores.items():
    if grp in BASE:
        BASE[grp]["scores"]=sorted(scs); BASE[grp]["n_scored"]=len(scs)
        BASE[grp]["score_median"]=st.median(scs)

# 2) clean violator re-runs: parse the 3 new solves + grades from RERUN_DIR
def lines(path):
    o=[]
    for ln in open(path):
        ln=ln.strip()
        if ln:
            try:o.append(json.loads(ln))
            except:pass
    return o
def prompt_of(objs):
    for o in objs:
        m=o.get("message") if isinstance(o.get("message"),dict) else o
        if (m or {}).get("role")=="user":
            c=m.get("content")
            if isinstance(c,str):return c
            if isinstance(c,list):return " ".join(b.get("text","") for b in c if isinstance(b,dict) and b.get("type")=="text")
    return ""
def scan(objs):
    it=ot=cr=cw=turns=0;tools=defaultdict(int);struct=None
    for o in objs:
        m=o.get("message") if isinstance(o.get("message"),dict) else o
        u=(m or {}).get("usage") or o.get("usage")
        if isinstance(u,dict):
            it+=u.get("input_tokens",0) or 0;ot+=u.get("output_tokens",0) or 0
            cr+=u.get("cache_read_input_tokens",0) or 0;cw+=u.get("cache_creation_input_tokens",0) or 0
        if (m or {}).get("role")=="assistant" and isinstance(m.get("content"),list):
            turns+=1
            for b in m["content"]:
                if isinstance(b,dict) and b.get("type")=="tool_use":
                    tools[b.get("name","?")]+=1
                    inp=b.get("input") or {}
                    if isinstance(inp,dict) and ("answer" in inp or "score" in inp):struct=inp
    return it,ot,cr,cw,turns,dict(tools),struct
def cost(it,ot,cr,cw): return it/1e6*STD["IN"]+ot/1e6*STD["OUT"]+cr/1e6*STD["CR"]+cw/1e6*STD["CW"]
TP={"frappe":{"FL3":"Trace how Frappe throttles too-frequent","FS3":"One worker process serves many different sites"},"dify":{"DL3":"Trace how a workflow variable reference"}}
rerun={}  # group -> {cost,turns,sutra,grep,viol,answer}
for path in glob.glob(f"{RERUN_DIR}/agent-*.jsonl"):
    objs=lines(path);p=prompt_of(objs)
    if "You are a senior engineer investigating" not in p: continue
    repo="frappe" if "/frappe_src" in p else "dify"
    tid=None
    for k,pre in TP.get(repo,{}).items():
        if pre in p: tid=k; break
    if not tid: continue
    it,ot,cr,cw,turns,tools,struct=scan(objs)
    sutra=sum(v for k,v in tools.items() if "sutra" in k.lower())
    grep=sum(v for k,v in tools.items() if k in ("Bash","Grep","Glob"))
    rerun[f"{repo}|{tid}|SUTRA_ONLY"]=dict(cost=cost(it,ot,cr,cw),turns=turns,sutra=sutra,grep=grep,
        viol=(grep>0),toolcalls=sum(tools.values()),fresh=it+ot,answer=(struct or {}).get("answer"))
print("re-run violator solves parsed:", list(rerun.keys()), "| any still violating:", {k:v["viol"] for k,v in rerun.items()})
json.dump({"rerun":rerun,"ds_scores":ds_scores}, open("/home/ritik/Desktop/sutra/sutra-benchmark/rigorous/_merge_debug.json","w"), indent=1, default=list)
print("scored coverage now:", sum(1 for g in BASE.values() if g.get("n_scored",0)>=3), "/36 groups fully scored")
json.dump(BASE, open("/home/ritik/Desktop/sutra/sutra-benchmark/rigorous/results_merged.json","w"), indent=1)
print("wrote results_merged.json (+ rerun in _merge_debug.json)")
