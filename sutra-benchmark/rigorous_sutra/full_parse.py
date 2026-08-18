#!/usr/bin/env python3
"""Single-repo (sutra) A/B/C parse: per solve -> cell(arm,tid)+answer+cost+tools+constraint;
per grade -> score matched to its solver answer, by candidate-answer prefix. Writes results.json."""
import json, glob, re, statistics as st
from collections import defaultdict

TD="/home/ritik/.claude-account1/projects/-home-ritik-Desktop-sutra/5d5da943-8e2c-467b-8fbe-07a9b7f411aa/subagents/workflows/wf_4a1d98bb-fa4"
OUT="/home/ritik/Desktop/sutra/sutra-benchmark/rigorous_sutra/results.json"
# sonnet-5 rates: std $3/$15, cache read 0.1x=$0.30, cache write 1.25x=$3.75; intro $2/$10, 0.20, 2.50
STD=dict(IN=3.0,OUT=15.0,CR=0.30,CW=3.75); INTRO=dict(IN=2.0,OUT=10.0,CR=0.20,CW=2.50)
TICKET_PREFIX={"sutra":{
 "SL1":"Trace how sutra re-scores its initial search candidates",
 "SL2":"Trace how sutra builds the stable identifier",
 "SL3":"Trace how sutra performs an incremental re-index",
 "SS1":"Sutra runs several independent searches for one query",
 "SS2":"A single sutra server has many repositories indexed",
 "SS3":"Sutra needs to link a function call in one file"}}
NTICK=6; NCELLS=NTICK*3*6  # 108

def lines(path):
    out=[]
    with open(path) as f:
        for ln in f:
            ln=ln.strip()
            if ln:
                try: out.append(json.loads(ln))
                except: pass
    return out

def prompt_of(path):
    for o in lines(path):
        m=o.get("message") if isinstance(o.get("message"),dict) else o
        if (m or {}).get("role")=="user":
            c=m.get("content")
            if isinstance(c,str): return c
            if isinstance(c,list): return " ".join(b.get("text","") for b in c if isinstance(b,dict) and b.get("type")=="text")
    return ""

def scan(path):
    it=ot=cr=cw=turns=0; tools=defaultdict(int); struct=None
    for o in lines(path):
        m=o.get("message") if isinstance(o.get("message"),dict) else o
        u=(m or {}).get("usage") or o.get("usage")
        if isinstance(u,dict):
            it+=u.get("input_tokens",0) or 0; ot+=u.get("output_tokens",0) or 0
            cr+=u.get("cache_read_input_tokens",0) or 0; cw+=u.get("cache_creation_input_tokens",0) or 0
        if (m or {}).get("role")=="assistant" and isinstance(m.get("content"),list):
            turns+=1
            for b in m["content"]:
                if isinstance(b,dict) and b.get("type")=="tool_use":
                    tools[b.get("name","?")]+=1
                    inp=b.get("input") or {}
                    if isinstance(inp,dict) and ("answer" in inp or "score" in inp): struct=inp
    return dict(input=it,output=ot,cache_read=cr,cache_creation=cw,turns=turns,tools=dict(tools),struct=struct)

def cost(u,R): return u["input"]/1e6*R["IN"]+u["output"]/1e6*R["OUT"]+u["cache_read"]/1e6*R["CR"]+u["cache_creation"]/1e6*R["CW"]

def classify(p):
    if "You are a senior engineer investigating" in p: kind="solve"
    elif "You are blind-grading" in p: kind="grade"
    else: return None
    repo="sutra"  # single repo
    arm="?"
    if "ONLY the Sutra MCP tools and the Read tool" in p: arm="SUTRA_ONLY"
    elif "ONLY Bash, Grep, Glob, and Read" in p: arm="GREP_ONLY"
    elif "any of — the Sutra MCP tools" in p: arm="BOTH"
    tid="?"
    for k,pre in TICKET_PREFIX[repo].items():
        if pre in p: tid=k; break
    cand=None
    if kind=="grade":
        m=re.search(r'CANDIDATE ANSWER:\s*"""\s*(.*?)\s*"""', p, re.S)
        cand=(m.group(1)[:160] if m else None)
    return dict(kind=kind,repo=repo,arm=arm,tid=tid,cand=cand)

solves=[]; grades=[]
for path in glob.glob(f"{TD}/agent-*.jsonl"):
    p=prompt_of(path); c=classify(p)
    if not c: continue
    u=scan(path)
    if c["kind"]=="solve":
        if c["arm"]=="?" or c["tid"]=="?": continue
        ans=(u["struct"] or {}).get("answer")
        if not ans: continue
        sutra=sum(v for k,v in u["tools"].items() if "sutra" in k.lower())
        grep=sum(v for k,v in u["tools"].items() if k in ("Bash","Grep","Glob"))
        viol=(c["arm"]=="GREP_ONLY" and sutra>0) or (c["arm"]=="SUTRA_ONLY" and grep>0)
        solves.append(dict(group=(c["repo"],c["tid"],c["arm"]),answer=ans,ans_key=ans[:160],
            cost_std=cost(u,STD),cost_intro=cost(u,INTRO),fresh=u["input"]+u["output"],
            total_ctx=u["input"]+u["output"]+u["cache_read"]+u["cache_creation"],turns=u["turns"],
            toolcalls=sum(u["tools"].values()),sutra=sutra,grep=grep,read=u["tools"].get("Read",0),
            tools=u["tools"],violation=viol,path=path.split("/")[-1]))
    else:
        sc=(u["struct"] or {}).get("score")
        if sc is not None and c["cand"]: grades.append(dict(cand=c["cand"],score=sc,tid=c["tid"]))

# match grades to solves by candidate answer prefix
usedg=set()
for s in solves:
    s["score"]=None
    for i,g in enumerate(grades):
        if i in usedg: continue
        if g["cand"] and (s["answer"].startswith(g["cand"]) or g["cand"].startswith(s["ans_key"][:120])):
            s["score"]=g["score"]; usedg.add(i); break

groups=defaultdict(list)
for s in solves: groups[s["group"]].append(s)
def mmm(xs): xs=sorted(xs); return dict(median=round(st.median(xs),4),min=round(min(xs),4),max=round(max(xs),4),n=len(xs))
out={}; graded=sum(1 for s in solves if s["score"] is not None); viol=sum(1 for s in solves if s["violation"])
for key in sorted(groups):
    repo,tid,arm=key; tr=groups[key]
    scores=[s["score"] for s in tr if s["score"] is not None]
    out[f"{repo}|{tid}|{arm}"]=dict(trials=len(tr),
        cost_std=mmm([s["cost_std"] for s in tr]),cost_intro=mmm([s["cost_intro"] for s in tr]),
        fresh=mmm([s["fresh"] for s in tr]),total_ctx=mmm([s["total_ctx"] for s in tr]),
        turns=mmm([s["turns"] for s in tr]),toolcalls=mmm([s["toolcalls"] for s in tr]),
        sutra=mmm([s["sutra"] for s in tr]),grep=mmm([s["grep"] for s in tr]),read=mmm([s["read"] for s in tr]),
        scores=sorted(scores),score_median=(st.median(scores) if scores else None),
        n_scored=len(scores),violations=sum(s["violation"] for s in tr),
        per_trial=[{"cost_std":round(s["cost_std"],4),"cost_intro":round(s["cost_intro"],4),"score":s["score"],
                    "viol":s["violation"],"sutra":s["sutra"],"grep":s["grep"],"read":s["read"],"turns":s["turns"],
                    "toolcalls":s["toolcalls"],"answer":s["answer"],"path":s["path"]} for s in tr])
json.dump(out,open(OUT,"w"),indent=1)
print(f"solve transcripts used={len(solves)} (expect {NCELLS})  grade results found={len(grades)}  solves matched to a score={graded}/{len(solves)}  constraint violations={viol}")
unscored=[k for k,v in out.items() if v["n_scored"]<v["trials"]]
print(f"groups={len(out)} (expect {NTICK*3});  groups with <6 scores: {len(unscored)} -> {unscored[:24]}")
violcells={k:v["violations"] for k,v in out.items() if v["violations"]>0}
print(f"cells with violations: {violcells}")
print("wrote", OUT)
