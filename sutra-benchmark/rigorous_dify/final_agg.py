#!/usr/bin/env python3
"""Single-repo (dify) aggregation + pre-registered conclusions, WITH TIME.
Drops constraint-violating trials; per-cell + per(class,arm) + overall medians for cost/TIME/turns/score;
secondary semantic read excluding the ambiguous-gold DS2; prints conclusions with overlap flags and a
per-cell sign test on cost & time direction. Writes final_analysis.json + report_data.json."""
import json, statistics as st
from math import comb
from collections import defaultdict, Counter
R=json.load(open("/home/ritik/Desktop/sutra/sutra-benchmark/rigorous_dify/results.json"))

LEX={"DL1","DL2","DL3"}; TIDS=["DL1","DL2","DL3","DS1","DS2","DS3"]; ARMS=["SUTRA_ONLY","GREP_ONLY","BOTH"]
def cls(tid): return "lexical" if tid in LEX else "semantic"
def med(xs): return round(st.median(xs),4) if xs else None
def rng(xs): return [round(min(xs),4),round(max(xs),4)] if xs else None
def overlap(a,b): return None if (not a or not b) else (not (a[1]<b[0] or a[0]>b[1]))
def sign_p(k,n):  return min(1, 2*sum(comb(n,i) for i in range(k,n+1))/2**n)

# clean per-cell trials (drop violators)
cells={}
for key,g in R.items():
    repo,tid,arm=key.split("|")
    kept=[t for t in g["per_trial"] if not t["viol"]]
    dropped=len(g["per_trial"])-len(kept)
    note=""
    if dropped: note=f"{dropped} violating trial(s) dropped; n={len(kept)}"
    if tid=="DL3" and arm=="SUTRA_ONLY" and dropped:
        note+=" — DL3 answer lives in the external graphon pkg (not indexed), so index-only navigation cannot fully confirm it without grep."
    cells[key]={"repo":repo,"tid":tid,"arm":arm,"cls":cls(tid),"trials":kept,"note":note}

def cellstats(tr):
    costs=[t["cost_std"] for t in tr]; durs=[t["dur"] for t in tr]; turns=[t["turns"] for t in tr]
    scores=[t["score"] for t in tr if t["score"] is not None]
    return dict(n=len(tr),cost_median=med(costs),cost_range=rng(costs),
        dur_median=med(durs),dur_range=rng(durs),
        turns_median=med(turns),turns_range=rng(turns),
        score_median=med(scores),scores=sorted(scores),n_scored=len(scores),
        sutra_median=med([t["sutra"] for t in tr]),grep_median=med([t["grep"] for t in tr]),
        any_fabrication=any(s==0 for s in scores))
percell={k:{**dict(repo=c["repo"],tid=c["tid"],arm=c["arm"],cls=c["cls"],note=c["note"]),**cellstats(c["trials"])} for k,c in cells.items()}

# pooled per (class,arm) and overall
def pool(pred):
    p=defaultdict(lambda:{"cost":[],"dur":[],"turns":[],"score":[],"sutra":[],"grep":[]})
    for k,c in cells.items():
        if not pred(c): continue
        for t in c["trials"]:
            d=p[c["arm"]]
            d["cost"].append(t["cost_std"]); d["dur"].append(t["dur"]); d["turns"].append(t["turns"])
            d["sutra"].append(t["sutra"]); d["grep"].append(t["grep"])
            if t["score"] is not None: d["score"].append(t["score"])
    return p
def agg_of(v):
    return dict(cost_median=med(v["cost"]),cost_range=rng(v["cost"]),
        dur_median=med(v["dur"]),dur_range=rng(v["dur"]),
        turns_median=med(v["turns"]),turns_range=rng(v["turns"]),
        score_median=med(v["score"]),score_mean=round(st.mean(v["score"]),2) if v["score"] else None,
        score_dist={s:v["score"].count(s) for s in sorted(set(v["score"]))},
        sutra_median=med(v["sutra"]),grep_median=med(v["grep"]),
        n=len(v["cost"]),n_scored=len(v["score"]),any_fab=any(s==0 for s in v["score"]))
overall={a:agg_of(v) for a,v in pool(lambda c:True).items()}
byclass={f"{cl}|{a}":agg_of(v) for cl in ["lexical","semantic"] for a,v in pool(lambda c:c["cls"]==cl).items()}
sem_noDS2={a:agg_of(v) for a,v in pool(lambda c:c["cls"]=="semantic" and c["tid"]!="DS2").items()}

out=dict(per_cell=percell, per_arm_overall=overall, per_class_arm=byclass, semantic_excluding_DS2=sem_noDS2)
json.dump(out,open("/home/ritik/Desktop/sutra/sutra-benchmark/rigorous_dify/final_analysis.json","w"),indent=1)

def line(n,a):
    return (f"    {n:11} cost=${a['cost_median']:.4f}[{a['cost_range'][0]:.4f}-{a['cost_range'][1]:.4f}]  "
            f"time={a['dur_median']:.0f}s[{a['dur_range'][0]:.0f}-{a['dur_range'][1]:.0f}]  "
            f"turns={a['turns_median']:.0f}  score_med={a['score_median']} mean={a['score_mean']} dist={a['score_dist']}  "
            f"sutra={a['sutra_median']} grep={a['grep_median']} n={a['n']}")
print("="*104); print("PRE-REGISTERED CONCLUSIONS — single repo: dify @ d67123e (cost=sonnet-5 std; time=wall-clock, concurrency-inflated)")
print("\n## OVERALL"); base=overall["GREP_ONLY"]
for a in ARMS: print(line(a,overall[a]))
for a in ARMS:
    if a=="GREP_ONLY": continue
    x=overall[a]
    cg=(x['cost_median']-base['cost_median'])/base['cost_median']*100; co=overlap(x['cost_range'],base['cost_range'])
    tg=(x['dur_median']-base['dur_median'])/base['dur_median']*100; to=overlap(x['dur_range'],base['dur_range'])
    print(f"    -> {a} vs GREP: cost {cg:+.0f}% ({'OVERLAP' if co else 'DISJOINT'})  time {tg:+.0f}% ({'OVERLAP' if to else 'DISJOINT'})")
for cl in ["lexical","semantic"]:
    print(f"\n## {cl.upper()}"); b=byclass[f"{cl}|GREP_ONLY"]
    for a in ARMS: print(line(a,byclass[f"{cl}|{a}"]))
print("\n## SEMANTIC excluding ambiguous-gold DS2")
for a in ARMS:
    s=sem_noDS2.get(a);  print(f"    {a:11} score_med={s['score_median']} mean={s['score_mean']} dist={s['score_dist']} n_scored={s['n_scored']}" if s else f"    {a}: n/a")

# per-cell sign tests on cost & time direction (SUTRA vs GREP), only cells where SUTRA has >=1 clean trial
def cm(tid,arm,k):
    tr=cells[f"dify|{tid}|{arm}"]["trials"]; xs=[t[k] for t in tr]; return st.median(xs) if xs else None
csu=tsu=n=0
print("\n## per-cell SUTRA vs GREP (median)")
for tid in TIDS:
    sc,gc=cm(tid,"SUTRA_ONLY","cost_std"),cm(tid,"GREP_ONLY","cost_std")
    sd,gd=cm(tid,"SUTRA_ONLY","dur"),cm(tid,"GREP_ONLY","dur")
    ncl=len(cells[f"dify|{tid}|SUTRA_ONLY"]["trials"])
    if sc is None: print(f"  {tid}: SUTRA has 0 clean trials (all violated) — excluded"); continue
    n+=1; csu+= sc>gc; tsu+= sd>gd
    print(f"  {tid}: cost SUTRA ${sc:.4f} vs GREP ${gc:.4f} ({'>' if sc>gc else '<='})  time SUTRA {sd:.0f}s vs GREP {gd:.0f}s ({'>' if sd>gd else '<='})  [SUTRA clean n={ncl}]")
print(f"\n  SUTRA costlier than GREP in {csu}/{n} cells -> sign p={sign_p(csu,n):.3f}")
print(f"  SUTRA slower than GREP in {tsu}/{n} cells -> sign p={sign_p(tsu,n):.3f}")

# emit report_data.json for the dashboard
D={"tids":TIDS,"per_cell":{},"per_arm":{},"per_class":{}}
for tid in TIDS:
    for a in ARMS:
        pc=percell[f"dify|{tid}|{a}"]
        D["per_cell"][f"{tid}|{a}"]=dict(cost=pc["cost_median"],cost_rng=pc["cost_range"],dur=pc["dur_median"],dur_rng=pc["dur_range"],
            turns=pc["turns_median"],score=pc["score_median"],scores=pc["scores"],n=pc["n"],sutra=pc["sutra_median"],grep=pc["grep_median"],note=pc["note"])
for a in ARMS:
    o=overall[a]; D["per_arm"][a]=dict(cost_med=o["cost_median"],cost_rng=o["cost_range"],dur_med=o["dur_median"],dur_rng=o["dur_range"],
        turns_med=o["turns_median"],score_mean=o["score_mean"],score_med=o["score_median"],score_dist=o["score_dist"],
        sutra_med=o["sutra_median"],grep_med=o["grep_median"],n=o["n"])
for cl in ["lexical","semantic"]:
    for a in ARMS:
        c=byclass[f"{cl}|{a}"]; D["per_class"][f"{cl}|{a}"]=dict(cost_med=c["cost_median"],cost_rng=c["cost_range"],dur_med=c["dur_median"],dur_rng=c["dur_range"],score_med=c["score_median"],score_dist=c["score_dist"])
json.dump(D,open("/home/ritik/Desktop/sutra/sutra-benchmark/rigorous_dify/report_data.json","w"),indent=1)
print("\nwrote final_analysis.json + report_data.json")
