"""If 100% of misses are ambiguity, WHICH ambiguity? Size each candidate fix."""
import json
from collections import Counter, defaultdict
from pathlib import Path

ART = Path.home() / ".sutra/artifacts"
REPOS = [("requests","psf__requests",321),("flask","pallets__flask",497),
         ("fastapi","fastapi__fastapi",1063),("pydantic","pydantic__pydantic",2799),
         ("celery","celery__celery",3601),("django","django__django",11010),
         ("sqlalchemy","sqlalchemy__sqlalchemy",12830)]
CALLABLE={"function","method","class","constructor"}
rows=[]
for name,slug,size in REPOS:
    p=ART/slug/"graph.json"
    if not p.exists(): continue
    g=json.loads(p.read_text()); syms=g["symbols"]
    by_name=defaultdict(list)
    for s in syms:
        if s.get("kind") in CALLABLE and not s.get("is_local"): by_name[s["name"]].append(s)
    file_of={s["id"]:s.get("file_path") for s in syms}
    encl_of={s["id"]:s.get("enclosing_moniker") or s.get("enclosing_class_id") for s in syms}
    calls=[r for r in g["relationships"] if r.get("kind")=="calls"]
    unres=[r for r in calls if r.get("target_name") in by_name and not r.get("is_resolved")]

    form=Counter(); same_file_cands=Counter(); same_class=0; has_import=0
    for r in unres:
        cands=by_name[r["target_name"]]
        md=r.get("metadata") or {}
        form[md.get("call_form") or "none"]+=1
        if md.get("import_source"): has_import+=1
        sf=file_of.get(r.get("source_id"))
        n_sf=sum(1 for c in cands if c.get("file_path")==sf)
        same_file_cands[min(n_sf,2)]+=1
        # would a class-hierarchy rule help? caller's enclosing class owns a candidate
        cls=encl_of.get(r.get("source_id"))
        if cls and any((c.get("enclosing_moniker") or c.get("enclosing_class_id"))==cls for c in cands):
            same_class+=1
    n=len(unres) or 1
    rows.append({"repo":name,"defs":size,"unresolved":len(unres),
      "method_form_pct":form.get("method",0)/n*100,
      "direct_form_pct":form.get("direct",0)/n*100,
      "has_import_source_pct":has_import/n*100,
      "same_class_candidate_pct":same_class/n*100,
      "zero_same_file_cand_pct":same_file_cands.get(0,0)/n*100,
      "multi_same_file_cand_pct":same_file_cands.get(2,0)/n*100})

print("Unresolved edges — what shape are they? (% of that repo's unresolved)")
print(f"{'repo':<12}{'unres':>7}{'method()':>10}{'direct()':>10}{'has import':>12}{'caller class':>14}{'no same-file':>14}")
for r in rows:
    print(f"{r['repo']:<12}{r['unresolved']:>7}{r['method_form_pct']:>9.0f}%{r['direct_form_pct']:>9.0f}%"
          f"{r['has_import_source_pct']:>11.0f}%{r['same_class_candidate_pct']:>13.0f}%{r['zero_same_file_cand_pct']:>13.0f}%")
print("""
  method()      call was obj.f()/self.f() -- needs a receiver type to resolve
  direct()      call was a bare f() -- needs scope/import reasoning only
  has import    the edge carries an import_source the 'import' rule could not match
  caller class  the CALLER's own class defines a candidate with that name
                -> an MRO/self-first rule would resolve these with no type inference
  no same-file  no candidate in the caller's file, so the 'local' rule never applies""")
json.dump(rows, open("/Users/ritikshukla/Desktop/claude-dir/sutra/benchmarks/fresh_ab/resolver_fixsize.json","w"), indent=1)
