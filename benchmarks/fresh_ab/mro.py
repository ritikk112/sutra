"""How many unresolved edges would an MRO-aware 'self.f()' rule recover?

Uses the graph's own `extends` edges to walk the caller's class hierarchy, so
this is an estimate from real data rather than an assumption.
"""
import json
from collections import defaultdict
from pathlib import Path
ART=Path.home()/".sutra/artifacts"
REPOS=[("requests","psf__requests",321),("flask","pallets__flask",497),
       ("fastapi","fastapi__fastapi",1063),("pydantic","pydantic__pydantic",2799),
       ("celery","celery__celery",3601),("django","django__django",11010),
       ("sqlalchemy","sqlalchemy__sqlalchemy",12830)]
CALLABLE={"function","method","class","constructor"}
print(f"{'repo':<12}{'unres':>7}{'self-only':>11}{'+MRO walk':>11}{'recovered':>11}{'new rate':>10}")
rows=[]
for name,slug,size in REPOS:
    p=ART/slug/"graph.json"
    if not p.exists(): continue
    g=json.loads(p.read_text()); syms=g["symbols"]; rels=g["relationships"]
    by_name=defaultdict(list); owner=defaultdict(list)
    for s in syms:
        if s.get("kind") in CALLABLE and not s.get("is_local"):
            by_name[s["name"]].append(s)
            cid=s.get("enclosing_class_id") or s.get("enclosing_moniker")
            if cid: owner[cid].append(s["name"])
    encl={s["id"]:(s.get("enclosing_class_id") or s.get("enclosing_moniker")) for s in syms}
    # class -> base classes, from resolved extends edges
    bases=defaultdict(list)
    for r in rels:
        if r.get("kind")=="extends" and r.get("is_resolved") and r.get("target_id"):
            bases[r["source_id"]].append(r["target_id"])
    def mro(cid, seen=None, depth=0):
        if seen is None: seen=set()
        if not cid or cid in seen or depth>6: return []
        seen.add(cid)
        out=[cid]
        for b in bases.get(cid,[]): out+=mro(b,seen,depth+1)
        return out
    calls=[r for r in rels if r.get("kind")=="calls"]
    matchable=[r for r in calls if r.get("target_name") in by_name]
    unres=[r for r in matchable if not r.get("is_resolved")]
    self_only=mro_hit=0
    for r in unres:
        cid=encl.get(r.get("source_id")); n=r["target_name"]
        if not cid: continue
        if n in owner.get(cid,[]): self_only+=1; mro_hit+=1; continue
        for c in mro(cid)[1:]:
            if n in owner.get(c,[]): mro_hit+=1; break
    base=len(matchable)-len(unres)
    newrate=(base+mro_hit)/len(matchable)*100 if matchable else 0
    oldrate=base/len(matchable)*100 if matchable else 0
    print(f"{name:<12}{len(unres):>7}{self_only:>10}{mro_hit:>11}{mro_hit/max(len(unres),1)*100:>10.0f}%"
          f"{f'{oldrate:.0f}%->{newrate:.0f}%':>10}")
    rows.append({"repo":name,"defs":size,"unresolved":len(unres),"self_only":self_only,
                 "mro_recoverable":mro_hit,"old_rate":round(oldrate,1),"projected_rate":round(newrate,1)})
json.dump(rows, open("/Users/ritikshukla/Desktop/claude-dir/sutra/benchmarks/fresh_ab/resolver_mro_estimate.json","w"), indent=1)
