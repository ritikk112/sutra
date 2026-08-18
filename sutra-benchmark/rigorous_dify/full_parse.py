#!/usr/bin/env python3
"""Single-repo (dify) A/B/C parse WITH TIME: per solve -> cell(arm,tid)+answer+cost+tools+constraint
+ wall-clock duration (last-first transcript timestamp, seconds); per grade -> score matched by
candidate-answer prefix. Writes results.json.

TIME CAVEAT: duration is per-agent wall-clock from the transcript's first->last timestamp. Agents ran
concurrently (cap ~cpu-2) under shared API rate limits, so ABSOLUTE seconds are inflated vs running one
agent alone. Arm-vs-arm comparison is fair (all arms interleaved under the same load); 'turns' is the
concurrency-independent structural latency proxy."""
import json, glob, re, statistics as st
from datetime import datetime
from collections import defaultdict

TD="/home/ritik/.claude-account1/projects/-home-ritik-Desktop-sutra/5d5da943-8e2c-467b-8fbe-07a9b7f411aa/subagents/workflows/wf_3cd70def-8e4"
OUT="/home/ritik/Desktop/sutra/sutra-benchmark/rigorous_dify/results.json"
STD=dict(IN=3.0,OUT=15.0,CR=0.30,CW=3.75); INTRO=dict(IN=2.0,OUT=10.0,CR=0.20,CW=2.50)
TICKET_PREFIX={"dify":{
 "DL1":"Trace how retrieval reranks candidate",
 "DL2":"Trace how Dify checks a workspace",
 "DL3":"Trace how a workflow variable reference",
 "DS1":"Where does Dify stop a single application from taking",
 "DS2":"A generation that keeps going far too long",
 "DS3":"A document that repeatedly fails to index"}}
NTICK=6; NCELLS=NTICK*3*6

def lines(path):
    out=[]
    with open(path) as f:
        for ln in f:
            ln=ln.strip()
            if ln:
                try: out.append(json.loads(ln))
                except: pass
    return out

def parse_ts(s):
    if not isinstance(s,str): return None
    try: return datetime.fromisoformat(s.replace("Z","+00:00"))
    except: return None

def prompt_of(objs):
    for o in objs:
        m=o.get("message") if isinstance(o.get("message"),dict) else o
        if (m or {}).get("role")=="user":
            c=m.get("content")
            if isinstance(c,str): return c
            if isinstance(c,list): return " ".join(b.get("text","") for b in c if isinstance(b,dict) and b.get("type")=="text")
    return ""

def scan(objs):
    it=ot=cr=cw=turns=0; tools=defaultdict(int); struct=None; tss=[]
    for o in objs:
        ts=parse_ts(o.get("timestamp"))
        if ts: tss.append(ts)
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
    dur=(max(tss)-min(tss)).total_seconds() if len(tss)>=2 else 0.0
    return dict(input=it,output=ot,cache_read=cr,cache_creation=cw,turns=turns,tools=dict(tools),struct=struct,dur=dur)

def cost(u,R): return u["input"]/1e6*R["IN"]+u["output"]/1e6*R["OUT"]+u["cache_read"]/1e6*R["CR"]+u["cache_creation"]/1e6*R["CW"]

def classify(p):
    if "You are a senior engineer investigating" in p: kind="solve"
    elif "You are blind-grading" in p: kind="grade"
    else: return None
    repo="dify"
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
    objs=lines(path); p=prompt_of(objs); c=classify(p)
    if not c: continue
    u=scan(objs)
    if c["kind"]=="solve":
        if c["arm"]=="?" or c["tid"]=="?": continue
        ans=(u["struct"] or {}).get("answer")
        if not ans: continue
        sutra=sum(v for k,v in u["tools"].items() if "sutra" in k.lower())
        grep=sum(v for k,v in u["tools"].items() if k in ("Bash","Grep","Glob"))
        viol=(c["arm"]=="GREP_ONLY" and sutra>0) or (c["arm"]=="SUTRA_ONLY" and grep>0)
        solves.append(dict(group=(c["repo"],c["tid"],c["arm"]),answer=ans,ans_key=ans[:160],
            cost_std=cost(u,STD),cost_intro=cost(u,INTRO),fresh=u["input"]+u["output"],
            total_ctx=u["input"]+u["output"]+u["cache_read"]+u["cache_creation"],turns=u["turns"],dur=u["dur"],
            toolcalls=sum(u["tools"].values()),sutra=sutra,grep=grep,read=u["tools"].get("Read",0),
            tools=u["tools"],violation=viol,path=path.split("/")[-1]))
    else:
        sc=(u["struct"] or {}).get("score")
        if sc is not None and c["cand"]: grades.append(dict(cand=c["cand"],score=sc,tid=c["tid"]))

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
        dur=mmm([s["dur"] for s in tr]),
        fresh=mmm([s["fresh"] for s in tr]),total_ctx=mmm([s["total_ctx"] for s in tr]),
        turns=mmm([s["turns"] for s in tr]),toolcalls=mmm([s["toolcalls"] for s in tr]),
        sutra=mmm([s["sutra"] for s in tr]),grep=mmm([s["grep"] for s in tr]),read=mmm([s["read"] for s in tr]),
        scores=sorted(scores),score_median=(st.median(scores) if scores else None),
        n_scored=len(scores),violations=sum(s["violation"] for s in tr),
        per_trial=[{"cost_std":round(s["cost_std"],4),"cost_intro":round(s["cost_intro"],4),"dur":round(s["dur"],1),
                    "score":s["score"],"viol":s["violation"],"sutra":s["sutra"],"grep":s["grep"],"read":s["read"],
                    "turns":s["turns"],"toolcalls":s["toolcalls"],"answer":s["answer"],"path":s["path"]} for s in tr])
json.dump(out,open(OUT,"w"),indent=1)
print(f"solve transcripts used={len(solves)} (expect {NCELLS})  grades found={len(grades)}  matched-score={graded}/{len(solves)}  violations={viol}")
unscored=[k for k,v in out.items() if v["n_scored"]<v["trials"]]
print(f"groups={len(out)} (expect {NTICK*3}); groups <6 scores: {len(unscored)} -> {unscored[:24]}")
print("cells with violations:", {k:v["violations"] for k,v in out.items() if v["violations"]>0})
# time sanity
alldur=[s["dur"] for s in solves]
print(f"duration(s): median={st.median(alldur):.0f} min={min(alldur):.0f} max={max(alldur):.0f}")
print("wrote", OUT)
