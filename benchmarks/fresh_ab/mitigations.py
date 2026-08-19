"""Offline benchmark of the proposed resolver mitigations.

Each is applied to the real artifact graphs using the resolver's own rules, so
the recovery numbers are measured rather than projected.

  M6  dedup candidates by definition site before declaring ambiguity
      (re-exports make one symbol look like several)
  M2  resolve EXTENDS with the same local/unique rules, then walk the MRO
      for self.f() calls
  M1  self-first: caller's own class owns a candidate
"""
import json
from collections import defaultdict
from pathlib import Path

ART = Path.home()/".sutra/artifacts"
REPOS=[("requests","psf__requests",321),("flask","pallets__flask",497),
       ("fastapi","fastapi__fastapi",1063),("pydantic","pydantic__pydantic",2799),
       ("celery","celery__celery",3601),("django","django__django",11010),
       ("sqlalchemy","sqlalchemy__sqlalchemy",12830)]
CALLABLE={"function","method","class","constructor"}
CLASSY={"class"}
rows=[]

for name,slug,size in REPOS:
    p=ART/slug/"graph.json"
    if not p.exists(): continue
    g=json.loads(p.read_text()); syms=g["symbols"]; rels=g["relationships"]

    by_name=defaultdict(list)
    for s in syms:
        if s.get("kind") in CALLABLE and not s.get("is_local"):
            by_name[s["name"]].append(s)
    cls_by_name=defaultdict(list)
    for s in syms:
        if s.get("kind") in CLASSY and not s.get("is_local"):
            cls_by_name[s["name"]].append(s)
    file_of={s["id"]:s.get("file_path") for s in syms}
    encl={s["id"]:(s.get("enclosing_class_id") or s.get("enclosing_moniker")) for s in syms}
    owner=defaultdict(set)
    for s in syms:
        if s.get("kind") in CALLABLE and not s.get("is_local"):
            c=s.get("enclosing_class_id") or s.get("enclosing_moniker")
            if c: owner[c].add(s["name"])

    def site(s):  # identity of a definition, independent of how it's re-exported
        return (s.get("file_path"), s.get("line_start"))

    calls=[r for r in rels if r.get("kind")=="calls"]
    matchable=[r for r in calls if r.get("target_name") in by_name]
    unres=[r for r in matchable if not r.get("is_resolved")]
    base=len(matchable)-len(unres)

    # ---- M6: dedup by definition site -------------------------------------
    m6=sum(1 for r in unres if len({site(c) for c in by_name[r["target_name"]]})==1)

    # ---- M2a: resolve EXTENDS with local-then-unique -----------------------
    ext=[r for r in rels if r.get("kind")=="extends"]
    ext_resolved={}
    for r in ext:
        n=r.get("target_name")
        if not n or n not in cls_by_name: continue
        cands=cls_by_name[n]
        sf=file_of.get(r.get("source_id"))
        same=[c for c in cands if c.get("file_path")==sf]
        pick=None
        if len(same)==1: pick=same[0]
        elif len({site(c) for c in cands})==1: pick=cands[0]
        if pick: ext_resolved[r["source_id"]]=ext_resolved.get(r["source_id"],[])+[pick["id"]]
    ext_matchable=[r for r in ext if r.get("target_name") in cls_by_name]

    # ---- M1 + M2b: self-first, then MRO over the newly resolved hierarchy ---
    def mro(cid, seen=None, d=0):
        if seen is None: seen=set()
        if not cid or cid in seen or d>6: return []
        seen.add(cid); out=[cid]
        for b in ext_resolved.get(cid,[]): out+=mro(b,seen,d+1)
        return out
    m1=m2b=0
    for r in unres:
        cid=encl.get(r.get("source_id")); n=r["target_name"]
        if not cid: continue
        if n in owner.get(cid,()): m1+=1; continue
        chain=mro(cid)[1:]
        if any(n in owner.get(c,()) for c in chain): m2b+=1

    tot=len(matchable) or 1
    # stack them, avoiding double counting: M6 first (pure dedup), then M1, then M2b
    m6_set={id(r) for r in unres if len({site(c) for c in by_name[r["target_name"]]})==1}
    stacked=0
    for r in unres:
        n=r["target_name"]; cid=encl.get(r.get("source_id"))
        if len({site(c) for c in by_name[n]})==1: stacked+=1; continue
        if cid and n in owner.get(cid,()): stacked+=1; continue
        if cid and any(n in owner.get(c,()) for c in mro(cid)[1:]): stacked+=1
    rows.append({"repo":name,"defs":size,"matchable":tot,"unresolved":len(unres),
        "rate":base/tot*100,
        "m6":m6,"m1":m1,"m2b":m2b,"stacked":stacked,
        "ext_matchable":len(ext_matchable),"ext_resolvable":len(ext_resolved),
        "rate_stacked":(base+stacked)/tot*100})

print("MITIGATION RECOVERY, measured on the real artifact graphs")
print(f"{'repo':<12}{'unres':>7}{'M6 dedup':>10}{'M1 self':>9}{'M2 MRO':>9}{'stacked':>9}{'rate now':>10}{'projected':>11}")
for r in rows:
    print(f"{r['repo']:<12}{r['unresolved']:>7}{r['m6']:>10}{r['m1']:>9}{r['m2b']:>9}{r['stacked']:>9}"
          f"{r['rate']:>9.0f}%{r['rate_stacked']:>10.0f}%")

print(f"\nM2 precondition — can EXTENDS be resolved with the same rules?")
print(f"{'repo':<12}{'extends matchable':>19}{'resolvable':>12}{'rate':>8}")
for r in rows:
    em=r["ext_matchable"] or 1
    print(f"{r['repo']:<12}{r['ext_matchable']:>19}{r['ext_resolvable']:>12}{r['ext_resolvable']/em*100:>7.0f}%")
json.dump(rows, open("/Users/ritikshukla/Desktop/claude-dir/sutra/benchmarks/fresh_ab/resolver_mitigations.json","w"), indent=1)
