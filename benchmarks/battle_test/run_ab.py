"""Kind-mode A/B eval (hard/off/soft) over freshly indexed repos.

Usage: .venv/bin/python run_ab.py  (from the sutra repo root)
Reads  eval_new/<name>.json ground-truth files, writes eval_new/ab_results.json
and prints per-repo tables + per-query details.
"""
import json
import sys
from pathlib import Path

SUTRA = "/Users/ritikshukla/Desktop/claude-dir/sutra"
sys.path.insert(0, SUTRA)
SCRATCH = Path(__file__).parent

from sutra.core.artifact.loader import ArtifactLoader          # noqa: E402
from sutra.mcp.registry import EmbedderCache                    # noqa: E402
from sutra.core.retrieval.pipeline import RetrievalPipeline     # noqa: E402

DATASETS = {
    "pallets/flask": ("pallets__flask", "flask.json"),
    "psf/requests": ("psf__requests", "requests.json"),
    "pydantic/pydantic": ("pydantic__pydantic", "pydantic.json"),
}
MODES = ("hard", "off", "soft")
ART = Path.home() / ".sutra" / "artifacts"

cache = EmbedderCache()
out = {}

for repo, (slug, fname) in DATASETS.items():
    data = json.loads((SCRATCH / fname).read_text())
    cases = data["cases"]
    snapshot = ArtifactLoader().load(ART / slug)
    embedder = cache.get(snapshot.embedding_model_id, snapshot.embedding_dims)

    # validate golds exist
    monikers = set(snapshot.symbols)
    bad = [(c["id"], g) for c in cases for g in c["golds"] if g not in monikers]
    if bad:
        print(f"!! {repo}: {len(bad)} golds NOT in index:")
        for cid, g in bad:
            print(f"   {cid}: {g}")

    repo_res = {"cases": [], "modes": {}}
    pipes = {m: RetrievalPipeline(snapshot, embedder, kind_mode=m) for m in MODES}
    analyzer = pipes["soft"]._analyzer

    for c in cases:
        hint = analyzer.parse(c["query"]).kind_hint
        row = {"id": c["id"], "bucket": c["bucket"], "query": c["query"],
               "golds": c["golds"], "kind_hint": sorted(hint) if hint else None,
               "gold_kinds": sorted({snapshot.symbols[g].get("kind") for g in c["golds"] if g in monikers}),
               "ranks": {}}
        for m in MODES:
            results = pipes[m].search(c["query"], top_k=10)
            rank = None
            for i, r in enumerate(results, 1):
                if r.moniker in c["golds"]:
                    rank = i
                    break
            row["ranks"][m] = rank
        repo_res["cases"].append(row)

    n = len(cases)
    for m in MODES:
        ranks = [r["ranks"][m] for r in repo_res["cases"]]
        repo_res["modes"][m] = {
            "recall@5": sum(1 for r in ranks if r and r <= 5) / n,
            "recall@10": sum(1 for r in ranks if r and r <= 10) / n,
            "mrr": sum(1 / r for r in ranks if r) / n,
            "zero_recall": sum(1 for r in ranks if r is None),
        }
    out[repo] = repo_res

    print(f"\n===== {repo} ({n} queries) =====")
    print(f"{'mode':6} {'r@5':>6} {'r@10':>6} {'MRR':>6} {'zero':>5}")
    for m in MODES:
        s = repo_res["modes"][m]
        print(f"{m:6} {s['recall@5']:6.3f} {s['recall@10']:6.3f} {s['mrr']:6.3f} {s['zero_recall']:>5}")
    print("per-query (rank hard/off/soft | hint | gold kinds):")
    for r in repo_res["cases"]:
        rk = r["ranks"]
        f = lambda x: str(x) if x else "-"
        print(f"  {r['id']:6} [{r['bucket']}] {f(rk['hard']):>2}/{f(rk['off']):>2}/{f(rk['soft']):>2}"
              f"  hint={','.join(r['kind_hint']) if r['kind_hint'] else 'none':18}"
              f" gold={','.join(r['gold_kinds']):10} {r['query'][:60]}")

(SCRATCH / "ab_results.json").write_text(json.dumps(out, indent=1))
print("\nsaved ->", SCRATCH / "ab_results.json")
