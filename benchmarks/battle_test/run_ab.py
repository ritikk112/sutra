"""Kind-mode A/B eval (hard/off/soft) over the battle-test ground truth.

Usage (from anywhere; paths resolve relative to the repo root):
    .venv/bin/python benchmarks/battle_test/run_ab.py
        [--modes hard,off,soft] [--no-prefix] [--verb-boost 1.15]
        [--dense-only] [--out ab_results.json] [--quiet-queries]

Reads  benchmarks/battle_test/{flask,requests,pydantic}.json and prints
per-repo tables + per-query ranks; writes the JSON summary next to the
datasets.

--no-prefix    : disable the bge query-instruction prefix (isolates its effect)
--verb-boost X : boost for verb-derived kind hints (default: same as noun boost)
--dense-only   : vector channel only (fusion-layer diagnosis, Task 2)
"""
import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
SCRATCH = Path(__file__).resolve().parent

from sutra.core.artifact.loader import ArtifactLoader           # noqa: E402
from sutra.core.embedder.local import LocalEmbedder              # noqa: E402
from sutra.core.retrieval.channels import VectorChannel          # noqa: E402
from sutra.core.retrieval.pipeline import RetrievalPipeline      # noqa: E402
from sutra.core.vector_store import InMemoryVectorStore          # noqa: E402

DEFAULT_MANIFEST = {
    "pallets/flask": ("pallets__flask", "flask.json"),
    "psf/requests": ("psf__requests", "requests.json"),
    "pydantic/pydantic": ("pydantic__pydantic", "pydantic.json"),
}
ART = Path.home() / ".sutra" / "artifacts"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="hard,off,soft")
    ap.add_argument("--no-prefix", action="store_true")
    ap.add_argument("--verb-boost", type=float, default=None)
    ap.add_argument("--dense-only", action="store_true")
    ap.add_argument("--rrf-k", type=int, default=None)
    ap.add_argument("--vector-weight", type=float, default=None)
    ap.add_argument("--out", default="ab_results.json")
    ap.add_argument("--quiet-queries", action="store_true")
    ap.add_argument(
        "--manifest", default=None,
        help="JSON file {repo: [artifact_slug, dataset_file]}; dataset paths "
             "and --out resolve relative to the manifest's directory. "
             "Default: the battle-test datasets next to this script.",
    )
    ap.add_argument(
        "--artifacts-dir", default=None,
        help="Artifacts root (default ~/.sutra/artifacts).",
    )
    args = ap.parse_args()
    modes = args.modes.split(",")

    global SCRATCH, ART
    if args.manifest:
        mpath = Path(args.manifest).resolve()
        datasets = {
            repo: tuple(v) for repo, v in json.loads(mpath.read_text()).items()
        }
        SCRATCH = mpath.parent
    else:
        datasets = DEFAULT_MANIFEST
    if args.artifacts_dir:
        ART = Path(args.artifacts_dir).expanduser()

    embedders: dict[tuple, LocalEmbedder] = {}

    def get_embedder(model_id: str, dims: int) -> LocalEmbedder:
        name = model_id.removeprefix("sentence-transformers/")
        key = (name, args.no_prefix)
        if key not in embedders:
            embedders[key] = LocalEmbedder(
                model_name=name, dimensions=dims,
                query_instruction=None if args.no_prefix else "auto",
            )
        return embedders[key]

    out = {"config": vars(args)}
    for repo, (slug, fname) in datasets.items():
        cases = json.loads((SCRATCH / fname).read_text())["cases"]
        snapshot = ArtifactLoader().load(ART / slug)
        embedder = get_embedder(snapshot.embedding_model_id, snapshot.embedding_dims)

        monikers = set(snapshot.symbols)
        bad = [(c["id"], g) for c in cases for g in c["golds"] if g not in monikers]
        for cid, g in bad:
            print(f"!! {repo} {cid}: gold NOT in index: {g}")

        channels = (
            (VectorChannel(InMemoryVectorStore(snapshot)),) if args.dense_only else None
        )
        extra = {}
        if args.rrf_k is not None:
            extra["rrf_k"] = args.rrf_k
        if args.vector_weight is not None:
            extra["channel_weights"] = {"vector": args.vector_weight}
        pipes = {
            m: RetrievalPipeline(
                snapshot, embedder, kind_mode=m,
                kind_boost_verb=args.verb_boost, channels=channels, **extra,
            )
            for m in modes
        }
        analyzer = pipes[modes[0]]._analyzer

        repo_res = {"cases": [], "modes": {}}
        for c in cases:
            parsed = analyzer.parse(c["query"], embed=False)
            row = {
                "id": c["id"], "bucket": c["bucket"], "query": c["query"],
                "golds": c["golds"],
                "kind_hint": sorted(parsed.kind_hint) if parsed.kind_hint else None,
                "hint_source": parsed.kind_hint_source,
                "gold_kinds": sorted({
                    snapshot.symbols[g].get("kind") for g in c["golds"] if g in monikers
                }),
                "ranks": {},
            }
            for m in modes:
                results = pipes[m].search(c["query"], top_k=10)
                rank = None
                for i, r in enumerate(results, 1):
                    if r.moniker in c["golds"]:
                        rank = i
                        break
                row["ranks"][m] = rank
            repo_res["cases"].append(row)

        n = len(cases)
        for m in modes:
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
        for m in modes:
            s = repo_res["modes"][m]
            print(f"{m:6} {s['recall@5']:6.3f} {s['recall@10']:6.3f} "
                  f"{s['mrr']:6.3f} {s['zero_recall']:>5}")
        if not args.quiet_queries:
            print(f"per-query (rank {'/'.join(modes)} | hint(src) | gold kinds):")
            for r in repo_res["cases"]:
                f = lambda x: str(x) if x else "-"
                ranks = "/".join(f"{f(r['ranks'][m]):>2}" for m in modes)
                hint = ",".join(r["kind_hint"]) if r["kind_hint"] else "none"
                src = f"({r['hint_source']})" if r["hint_source"] else ""
                print(f"  {r['id']:6} [{r['bucket']}] {ranks}"
                      f"  hint={hint + src:24}"
                      f" gold={','.join(r['gold_kinds']):10} {r['query'][:58]}")

    (SCRATCH / args.out).write_text(json.dumps(out, indent=1))
    print("\nsaved ->", SCRATCH / args.out)


if __name__ == "__main__":
    main()
