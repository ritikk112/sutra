#!/usr/bin/env python3
"""Single-repo (sutra) aggregation + pre-registered conclusions.
Drops constraint-violating trials (records how many), computes per-cell median[min-max] for
cost/turns/score/tool-usage, per (class,arm) pooled aggregates, and prints the pre-registered
conclusion tables (any arm-difference whose ranges OVERLAP the grep baseline is flagged 'not
significant'). Writes final_analysis.json."""
import json, statistics as st
from collections import defaultdict
R=json.load(open("/home/ritik/Desktop/sutra/sutra-benchmark/rigorous_sutra/results.json"))

LEX={"SL1","SL2","SL3"}
def cls(tid): return "lexical" if tid in LEX else "semantic"
def med(xs): return round(st.median(xs),4) if xs else None
def rng(xs): return [round(min(xs),4),round(max(xs),4)] if xs else None
def overlap(a,b):  # two [min,max] ranges overlap?
    if not a or not b: return None
    return not (a[1]<b[0] or a[0]>b[1])

# build clean per-cell trials (drop violators)
cells={}
for key,g in R.items():
    repo,tid,arm=key.split("|")
    kept=[t for t in g["per_trial"] if not t["viol"]]
    dropped=len(g["per_trial"])-len(kept)
    cells[key]={"repo":repo,"tid":tid,"arm":arm,"cls":cls(tid),"trials":kept,
                "note":(f"{dropped} violating trial(s) dropped; n={len(kept)}" if dropped else "")}

# per-cell summary
percell={}
for key,c in cells.items():
    tr=c["trials"]
    costs=[t["cost_std"] for t in tr]; turns=[t["turns"] for t in tr]
    scores=[t["score"] for t in tr if t["score"] is not None]
    percell[key]=dict(repo=c["repo"],tid=c["tid"],arm=c["arm"],cls=c["cls"],n=len(tr),
        cost_median=med(costs),cost_range=rng(costs),
        cost_intro_median=med([t["cost_intro"] for t in tr]),
        turns_median=med(turns),turns_range=rng(turns),
        score_median=med(scores),scores=sorted(scores),n_scored=len(scores),
        sutra_median=med([t["sutra"] for t in tr]),grep_median=med([t["grep"] for t in tr]),
        read_median=med([t["read"] for t in tr]),toolcalls_median=med([t["toolcalls"] for t in tr]),
        any_fabrication=any(s==0 for s in scores),note=c["note"])

# per (class,arm) pooled across the 3 tickets' trials; also overall per arm
pool=defaultdict(lambda:{"cost":[],"turns":[],"score":[],"sutra":[],"grep":[],"toolcalls":[]})
poolall=defaultdict(lambda:{"cost":[],"turns":[],"score":[],"sutra":[],"grep":[],"toolcalls":[]})
for key,c in cells.items():
    for t in c["trials"]:
        for pk in [(c["cls"],c["arm"]), ("ALL",c["arm"])]:
            d = pool[pk] if pk[0]!="ALL" else poolall[pk]
            d["cost"].append(t["cost_std"]); d["turns"].append(t["turns"])
            d["sutra"].append(t["sutra"]); d["grep"].append(t["grep"]); d["toolcalls"].append(t["toolcalls"])
            if t["score"] is not None: d["score"].append(t["score"])
def agg_of(v):
    return dict(cost_median=med(v["cost"]),cost_range=rng(v["cost"]),
        turns_median=med(v["turns"]),turns_range=rng(v["turns"]),
        score_median=med(v["score"]),score_dist={s:v["score"].count(s) for s in sorted(set(v["score"]))},
        sutra_median=med(v["sutra"]),grep_median=med(v["grep"]),toolcalls_median=med(v["toolcalls"]),
        n=len(v["cost"]),n_scored=len(v["score"]),any_fab=any(s==0 for s in v["score"]))
agg={f"{cl}|{arm}":agg_of(v) for (cl,arm),v in pool.items()}
aggall={arm:agg_of(v) for (_,arm),v in poolall.items()}

out=dict(per_cell=percell, per_class_arm=agg, per_arm_overall=aggall)
json.dump(out,open("/home/ritik/Desktop/sutra/sutra-benchmark/rigorous_sutra/final_analysis.json","w"),indent=1)

ARMS=["SUTRA_ONLY","GREP_ONLY","BOTH"]
def line(name,a):
    return (f"    {name:11} cost=${a['cost_median']:.4f} [{a['cost_range'][0]:.4f}-{a['cost_range'][1]:.4f}]  "
            f"turns={a['turns_median']:.0f} [{a['turns_range'][0]}-{a['turns_range'][1]}]  "
            f"score_med={a['score_median']} dist={a['score_dist']}  sutra_med={a['sutra_median']} grep_med={a['grep_median']}  n={a['n']}")
print("="*100)
print("PRE-REGISTERED CONCLUSIONS — single repo: sutra  (cost = sonnet-5 std $/run)")
for scope,table in [("OVERALL (all 6 tickets)",aggall)]:
    print(f"\n## {scope}")
    base=table["GREP_ONLY"]
    for arm in ARMS:
        print(line(arm,table[arm]))
    for arm in ARMS:
        if arm=="GREP_ONLY": continue
        a=table[arm]; gap=(a['cost_median']-base['cost_median'])/base['cost_median']*100
        ov=overlap(a['cost_range'],base['cost_range'])
        sgap = a['score_median']-base['score_median'] if (a['score_median'] is not None and base['score_median'] is not None) else None
        print(f"    -> {arm} vs GREP: cost {gap:+.0f}% (ranges {'OVERLAP -> NOT significant' if ov else 'DISJOINT -> significant'}); "
              f"quality delta median score {sgap:+.1f}")
for cl in ["lexical","semantic"]:
    print(f"\n## {cl.upper()} tickets")
    base=agg[f"{cl}|GREP_ONLY"]
    for arm in ARMS:
        print(line(arm,agg[f'{cl}|{arm}']))
    for arm in ARMS:
        if arm=="GREP_ONLY": continue
        a=agg[f"{cl}|{arm}"]; gap=(a['cost_median']-base['cost_median'])/base['cost_median']*100
        ov=overlap(a['cost_range'],base['cost_range'])
        print(f"    -> {arm} vs GREP: cost {gap:+.0f}% (ranges {'OVERLAP -> NOT significant' if ov else 'DISJOINT -> significant'})")

print("\n## per-cell (median[min-max], n after dropping violators)")
for key in sorted(percell):
    c=percell[key]
    print(f"  {key:26} n={c['n']} cost=${c['cost_median']:.4f}{c['cost_range']} turns={c['turns_median']:.0f} "
          f"score={c['score_median']}{c['scores']} sutra={c['sutra_median']} grep={c['grep_median']}"
          + (f"  [{c['note']}]" if c['note'] else ""))

# key diagnostic: does BOTH actually use the index when free to choose?
b=aggall["BOTH"]
print(f"\n## BOTH tool-choice diagnostic: median sutra calls={b['sutra_median']}, median grep calls={b['grep_median']} "
      f"(if sutra~0 -> BOTH≈GREP because the index was barely used, as in the 2-repo run)")
print("\nwrote final_analysis.json")
