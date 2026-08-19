"""Categorise every unresolved CALLS edge in each indexed repo.

The resolver reports a rate but not a reason. graph.json carries every edge
with is_resolved, target_name and call_form, so the reasons can be counted
instead of guessed.
"""
import json, builtins
from collections import Counter, defaultdict
from pathlib import Path

ART = Path.home() / ".sutra/artifacts"
REPOS = [("requests","psf__requests",321), ("flask","pallets__flask",497),
         ("fastapi","fastapi__fastapi",1063), ("pydantic","pydantic__pydantic",2799),
         ("celery","celery__celery",3601), ("django","django__django",11010),
         ("sqlalchemy","sqlalchemy__sqlalchemy",12830)]
BUILTINS = set(dir(builtins))
out = []

for name, slug, size in REPOS:
    p = ART / slug / "graph.json"
    if not p.exists():
        continue
    g = json.loads(p.read_text())
    syms = g["symbols"]
    # candidate index, mirroring the resolver: non-local callables by name
    CALLABLE = {"function", "method", "class", "constructor"}
    by_name = defaultdict(list)
    for s in syms:
        if s.get("kind") in CALLABLE and not s.get("is_local"):
            by_name[s["name"]].append(s)
    # which file each symbol lives in, to test the same-file hypothesis
    file_of = {s["id"]: s.get("file_path") for s in syms}

    calls = [r for r in g["relationships"] if r.get("kind") == "calls"]
    matchable = [r for r in calls if r.get("target_name") in by_name]
    unres = [r for r in matchable if not r.get("is_resolved")]
    res = [r for r in matchable if r.get("is_resolved")]

    cat = Counter()
    for r in unres:
        n = r["target_name"]
        cands = by_name[n]
        if len(cands) > 1:
            cat["ambiguous: >1 candidate with this name"] += 1
        elif r.get("metadata", {}).get("call_form") == "method":
            cat["single candidate, method call-form"] += 1
        else:
            cat["single candidate, still unresolved"] += 1
    # how many candidates does a typical unresolved edge face?
    ncand = [len(by_name[r["target_name"]]) for r in unres]
    ncand.sort()
    # same-file share among RESOLVED edges (the 'local' rule's territory)
    same_file = 0
    for r in res:
        sf = file_of.get(r.get("source_id"))
        tf = file_of.get(r.get("target_id"))
        if sf and tf and sf == tf:
            same_file += 1

    out.append({
        "repo": name, "defs": size,
        "calls_total": len(calls), "matchable": len(matchable),
        "external_share": (len(calls) - len(matchable)) / len(calls) * 100,
        "resolved": len(res), "unresolved": len(unres),
        "rate": len(res) / len(matchable) * 100 if matchable else 0,
        "same_file_share_of_resolved": same_file / len(res) * 100 if res else 0,
        "median_candidates_faced": ncand[len(ncand)//2] if ncand else 0,
        "p90_candidates_faced": ncand[int(len(ncand)*.9)] if ncand else 0,
        "cats": dict(cat),
    })

print(f"{'repo':<12}{'defs':>7}{'rate':>7}{'ambiguous':>11}{'method-form':>13}{'other':>8}   {'med cand':>9}{'p90 cand':>9}")
for r in out:
    tot = r["unresolved"] or 1
    a = r["cats"].get("ambiguous: >1 candidate with this name", 0)
    m = r["cats"].get("single candidate, method call-form", 0)
    o = r["cats"].get("single candidate, still unresolved", 0)
    print(f"{r['repo']:<12}{r['defs']:>7}{r['rate']:>6.0f}%{a/tot*100:>10.0f}%{m/tot*100:>12.0f}%{o/tot*100:>7.0f}%   "
          f"{r['median_candidates_faced']:>9}{r['p90_candidates_faced']:>9}")

print(f"\n{'repo':<12}{'same-file share of RESOLVED edges':>36}{'external calls (unindexed callee)':>36}")
for r in out:
    print(f"{r['repo']:<12}{r['same_file_share_of_resolved']:>35.1f}%{r['external_share']:>35.1f}%")

json.dump(out, open("/Users/ritikshukla/Desktop/claude-dir/sutra/benchmarks/fresh_ab/resolver_forensics.json","w"), indent=1)
