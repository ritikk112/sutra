#!/usr/bin/env python3
"""Parse workflow solver transcripts: derive cell (repo,arm,ticket) from the prompt,
compute per-transcript billed cost + tool-call breakdown + arm-constraint compliance.
Groups the 3 trials of each cell -> median + min/max."""
import json, os, glob, statistics as st
from collections import defaultdict

TD = "/home/ritik/.claude-account1/projects/-home-ritik-Desktop-sutra/5d5da943-8e2c-467b-8fbe-07a9b7f411aa/subagents/workflows/wf_8e7c6cdd-80f"
# Sonnet-5 rates $/1M
STD  = dict(IN=3.0, OUT=15.0, CR=0.30, CW=3.75)
INTRO= dict(IN=2.0, OUT=10.0, CR=0.20, CW=2.50)

TICKET_PREFIX = {
 "frappe":{"FL1":"Trace how Frappe fires recurring background","FL2":"Trace how Frappe decides whether the current user may READ",
           "FL3":"Trace how Frappe throttles too-frequent","FS1":"A user opens a record; before they save",
           "FS2":"A flood of slow, long-running background jobs","FS3":"One worker process serves many different sites"},
 "dify":{"DL1":"Trace how retrieval reranks candidate","DL2":"Trace how Dify checks a workspace",
         "DL3":"Trace how a workflow variable reference","DS1":"Where does Dify stop a single application from taking",
         "DS2":"A generation that keeps going far too long","DS3":"A document that repeatedly fails to index"},
}

def first_user_prompt(path):
    with open(path) as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try: o=json.loads(line)
            except: continue
            m=o.get("message") if isinstance(o.get("message"),dict) else o
            if (m or {}).get("role")=="user":
                c=m.get("content")
                if isinstance(c,str): return c
                if isinstance(c,list):
                    return " ".join(b.get("text","") for b in c if isinstance(b,dict) and b.get("type")=="text")
    return ""

def parse_usage_tools(path):
    it=ot=cr=cw=turns=0; tools=defaultdict(int)
    with open(path) as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try: o=json.loads(line)
            except: continue
            m=o.get("message") if isinstance(o.get("message"),dict) else o
            u=(m or {}).get("usage") or o.get("usage")
            if isinstance(u,dict):
                it+=u.get("input_tokens",0) or 0; ot+=u.get("output_tokens",0) or 0
                cr+=u.get("cache_read_input_tokens",0) or 0; cw+=u.get("cache_creation_input_tokens",0) or 0
            if (m or {}).get("role")=="assistant":
                turns+=1; c=m.get("content")
                if isinstance(c,list):
                    for b in c:
                        if isinstance(b,dict) and b.get("type")=="tool_use": tools[b.get("name","?")]+=1
    return dict(input=it,output=ot,cache_read=cr,cache_creation=cw,turns=turns,tools=dict(tools))

def cost(u,R): return (u["input"]/1e6*R["IN"]+u["output"]/1e6*R["OUT"]+u["cache_read"]/1e6*R["CR"]+u["cache_creation"]/1e6*R["CW"])

def classify(p):
    if "You are a senior engineer investigating" in p: kind="solve"
    elif "You are blind-grading" in p: kind="grade"
    else: return None
    repo = "frappe" if "/frappe_src" in p else ("dify" if "Desktop/dify/dify" in p else "?")
    if "ONLY the Sutra MCP tools and the Read tool" in p: arm="SUTRA_ONLY"
    elif "ONLY Bash, Grep, Glob, and Read" in p: arm="GREP_ONLY"
    elif "any of — the Sutra MCP tools" in p or "any of — the Sutra MCP tools" in p: arm="BOTH"
    else: arm="?"
    tid="?"
    if repo in TICKET_PREFIX:
        for k,pre in TICKET_PREFIX[repo].items():
            if pre in p: tid=k; break
    return dict(kind=kind,repo=repo,arm=arm,tid=tid)

cells=defaultdict(list)   # (repo,tid,arm) -> list of per-trial dicts
grade_n=0; unknown=0
for path in glob.glob(f"{TD}/agent-*.jsonl"):
    p=first_user_prompt(path)
    c=classify(p)
    if not c: unknown+=1; continue
    if c["kind"]=="grade": grade_n+=1; continue
    if c["repo"]=="?" or c["arm"]=="?" or c["tid"]=="?": unknown+=1; continue
    u=parse_usage_tools(path)
    sutra=sum(v for k,v in u["tools"].items() if "sutra" in k.lower())
    grep=sum(v for k,v in u["tools"].items() if k in ("Bash","Grep","Glob"))
    read=u["tools"].get("Read",0)
    viol = (c["arm"]=="GREP_ONLY" and sutra>0) or (c["arm"]=="SUTRA_ONLY" and grep>0)
    cells[(c["repo"],c["tid"],c["arm"])].append(dict(
        cost_std=cost(u,STD),cost_intro=cost(u,INTRO),fresh=u["input"]+u["output"],
        total_ctx=u["input"]+u["output"]+u["cache_read"]+u["cache_creation"],
        turns=u["turns"],toolcalls=sum(u["tools"].values()),sutra=sutra,grep=grep,read=read,
        tools=u["tools"],violation=viol))

def mmm(xs): xs=sorted(xs); return dict(median=round(st.median(xs),4),min=round(min(xs),4),max=round(max(xs),4),n=len(xs))
out={}
print(f"solve cells found={len(cells)} (expect 36)  grade_transcripts={grade_n}  unknown={unknown}")
viol_total=0
for key in sorted(cells):
    repo,tid,arm=key; tr=cells[key]
    for t in tr:
        if t["violation"]: viol_total+=1
    row=dict(trials=len(tr),
        cost_std=mmm([t["cost_std"] for t in tr]), cost_intro=mmm([t["cost_intro"] for t in tr]),
        fresh=mmm([t["fresh"] for t in tr]), total_ctx=mmm([t["total_ctx"] for t in tr]),
        turns=mmm([t["turns"] for t in tr]), toolcalls=mmm([t["toolcalls"] for t in tr]),
        sutra=mmm([t["sutra"] for t in tr]), grep=mmm([t["grep"] for t in tr]),
        violations=sum(t["violation"] for t in tr),
        per_trial_cost_std=[round(t["cost_std"],4) for t in tr])
    out[f"{repo}|{tid}|{arm}"]=row
print(f"total per-trial constraint violations={viol_total}")
json.dump(out, open("/home/ritik/Desktop/sutra/sutra-benchmark/rigorous/cost_constraints.json","w"), indent=2)
# quick view: per (repo,arm) aggregate median cost across its tickets/trials
agg=defaultdict(list)
for key,tr in cells.items():
    repo,tid,arm=key
    for t in tr: agg[(repo,arm)].append(t["cost_std"])
print("\n== median billed $ per (repo,arm), across all its tickets*trials ==")
for k in sorted(agg):
    xs=agg[k]; print(f"  {k[0]:6} {k[1]:10} median=${st.median(xs):.3f}  mean=${sum(xs)/len(xs):.3f}  n={len(xs)}")
print("\nwrote cost_constraints.json")
