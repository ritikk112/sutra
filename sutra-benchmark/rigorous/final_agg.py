#!/usr/bin/env python3
import json, statistics as st
from collections import defaultdict
R=json.load(open("results.json"))
def load_arr(p):
    t=open(p).read(); i=t.find("["); o,_=json.JSONDecoder().raw_decode(t[i:]); return o
gm=load_arr("/tmp/claude-1000/-home-ritik-Desktop-sutra/f2ce7530-a1a4-4531-9856-7d9aa4558a38/tasks/w25200h7v.output")
gm_scores=defaultdict(list)
for r in gm: gm_scores[r["group"]].append(r["score"])
rr=json.load(open("_merge_debug.json"))["rerun"]   # frappe FL3/FS3 clean, dify DL3 still viol

LEX={"FL1","FL2","FL3","DL1","DL2","DL3"}
# build clean per-cell trials
cells={}   # (repo,tid,arm) -> {"trials":[{cost,turns,sutra,grep,score,viol}], "note":...}
for key,g in R.items():
    repo,tid,arm=key.split("|")
    trials=[dict(cost=t["cost_std"],turns=t["turns"],sutra=t["sutra"],grep=t["grep"],
                 score=t.get("score"),viol=t.get("viol",False)) for t in g["per_trial"]]
    # attach DS2/DS3 scores
    if key in gm_scores:
        scs=gm_scores[key][:]
        for t in trials:
            if t["score"] is None and scs: t["score"]=scs.pop(0)
    cells[key]={"trials":trials,"note":""}
# splice clean re-runs for FL3/FS3; drop violators
for key in ["frappe|FL3|SUTRA_ONLY","frappe|FS3|SUTRA_ONLY"]:
    tr=cells[key]["trials"]; tr=[t for t in tr if not t["viol"]]   # drop the 1 violating original
    d=rr[key]; tr.append(dict(cost=d["cost"],turns=d["turns"],sutra=d["sutra"],grep=d["grep"],score=4 if key.endswith("FL3|SUTRA_ONLY") or True else None,viol=d["viol"]))
    # scores for re-run: FL3=4, FS3=4 (from grader); set explicitly below
    cells[key]["trials"]=tr; cells[key]["note"]="1 violating trial dropped; clean re-run added (n=3)"
# fix re-run scores (grader gave 4 for both)
for key,sc in [("frappe|FL3|SUTRA_ONLY",4),("frappe|FS3|SUTRA_ONLY",4)]:
    cells[key]["trials"][-1]["score"]=sc
# dify DL3 SUTRA_ONLY: re-run also violated -> keep only the clean original trials
k="dify|DL3|SUTRA_ONLY"; cells[k]["trials"]=[t for t in cells[k]["trials"] if not t["viol"]]
cells[k]["note"]=f"SUTRA_ONLY could not stay grep-free (answer is in external graphon pkg not indexed); {len(cells[k]['trials'])} clean trial(s) kept, both violating runs dropped"

def cls(tid): return "lexical" if tid in LEX else "semantic"
def med(xs): return round(st.median(xs),4) if xs else None
def rng(xs): return [round(min(xs),4),round(max(xs),4)] if xs else None

# per-cell summary
percell={}
for key,c in cells.items():
    repo,tid,arm=key.split("|"); tr=c["trials"]
    costs=[t["cost"] for t in tr]; turns=[t["turns"] for t in tr]
    scores=[t["score"] for t in tr if t["score"] is not None]
    percell[key]=dict(repo=repo,tid=tid,arm=arm,cls=cls(tid),n=len(tr),
        cost_median=med(costs),cost_range=rng(costs),
        turns_median=med(turns),turns_range=rng(turns),
        score_median=med(scores),scores=sorted(scores),
        sutra_median=med([t["sutra"] for t in tr]),grep_median=med([t["grep"] for t in tr]),
        any_fabrication=any(s==0 for s in scores),note=c["note"])

# per (repo,class,arm) pooled across the 3 tickets' trials
pool=defaultdict(lambda:{"cost":[],"turns":[],"score":[]})
for key,c in cells.items():
    repo,tid,arm=key.split("|")
    for t in c["trials"]:
        pk=(repo,cls(tid),arm)
        pool[pk]["cost"].append(t["cost"]); pool[pk]["turns"].append(t["turns"])
        if t["score"] is not None: pool[pk]["score"].append(t["score"])
agg={}
for pk,v in pool.items():
    repo,cl,arm=pk
    agg[f"{repo}|{cl}|{arm}"]=dict(cost_median=med(v["cost"]),cost_range=rng(v["cost"]),
        turns_median=med(v["turns"]),turns_range=rng(v["turns"]),
        score_median=med(v["score"]),score_dist={s:v["score"].count(s) for s in sorted(set(v["score"]))},
        n=len(v["cost"]),any_fab=any(s==0 for s in v["score"]))

# also semantic-without-DS2 (ambiguous gold)
pool2=defaultdict(list)
for key,c in cells.items():
    repo,tid,arm=key.split("|")
    if cls(tid)=="semantic" and tid!="DS2":
        for t in c["trials"]:
            if t["score"] is not None: pool2[(repo,arm)].append(t["score"])
sem_noDS2={f"{r}|{a}":dict(score_median=med(v),n=len(v),dist={s:v.count(s) for s in sorted(set(v))}) for (r,a),v in pool2.items()}

out=dict(per_cell=percell, per_repo_class_arm=agg, semantic_quality_excluding_DS2=sem_noDS2)
json.dump(out,open("final_analysis.json","w"),indent=1)

# ---- print pre-registered conclusion tables ----
print("="*90)
print("COST (median billed $/run, Sonnet-5 std) + range, by repo x class x arm")
for repo in ["frappe","dify"]:
    for cl in ["lexical","semantic"]:
        print(f"\n  {repo} / {cl}:")
        rows=[(arm,agg[f'{repo}|{cl}|{arm}']) for arm in ["SUTRA_ONLY","GREP_ONLY","BOTH"]]
        for arm,a in rows:
            print(f"    {arm:11} cost=${a['cost_median']:.3f} [{a['cost_range'][0]:.3f}-{a['cost_range'][1]:.3f}]  "
                  f"turns={a['turns_median']:.0f} [{a['turns_range'][0]}-{a['turns_range'][1]}]  "
                  f"score_med={a['score_median']}  dist={a['score_dist']} n={a['n']}")
        # cheapest arm
        cheapest=min(rows,key=lambda r:r[1]['cost_median'])
        base=dict(rows)['GREP_ONLY']
        for arm,a in rows:
            if arm=='GREP_ONLY':continue
            gap=(a['cost_median']-base['cost_median'])/base['cost_median']*100
            overlap = not (a['cost_range'][1]<base['cost_range'][0] or a['cost_range'][0]>base['cost_range'][1])
            print(f"    -> {arm} vs GREP_ONLY cost {gap:+.0f}% (ranges {'OVERLAP' if overlap else 'DISJOINT'})")
print("\n== SEMANTIC quality excluding the ambiguous DS2 gold ==")
for repo in ["frappe","dify"]:
    for arm in ["SUTRA_ONLY","GREP_ONLY","BOTH"]:
        s=sem_noDS2.get(f"{repo}|{arm}")
        if s: print(f"  {repo} {arm:11} score_med={s['score_median']} dist={s['dist']} n={s['n']}")
print("\nwrote final_analysis.json")
