"""Efficiency metrics for the sutra-vs-control workflow.

Scans the workflow transcript dir's agent-*.jsonl solver transcripts
(matched via the '[trial marker: <ticket>-<arm>-t<n>]' line embedded in
each solver prompt), and per trial computes:
  - tool-call counts by tool (sutra MCP / Bash / Grep / Glob / Read)
  - calls-to-localization: index of the first tool call whose INPUT or
    RESULT mentions a gold marker for that ticket (proxy for how fast
    the agent touched the right mechanism)
  - output tokens if usage records are present

Usage: python3 analyze_ab_transcripts.py <transcript_dir>
Writes ab_efficiency.json next to this script and prints a summary.
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# gold markers per ticket: substrings that identify the gold mechanism
GOLD_MARKERS = {
    "SL1": ["reranker.py", "def rerank", "rerank("],
    "SL2": ["moniker.py", "MonikerBuilder"],
    "SL3": ["git_differ", "changed_files", "incremental_updater"],
    "SS1": ["rrf_fuse", "fusion.py"],
    "SS2": ["registry.py", "ServingUnit", "SnapshotRegistry"],
    "SS3": ["lsp_resolver", "LspResolver", "pyright-langserver"],
    "FT1": ["run_endpoint_function", "run_in_threadpool"],
    "FT2": ["serialize_response", "ResponseValidationError"],
    "FT3": ["solve_dependencies", "request_params_to_args"],
    "FT4": ["request_validation_exception_handler"],
    "FT5": ["OAuth2PasswordBearer"],
    "FT6": ["get_openapi", "openapi_schema"],
    "FK1": ["open_session", "save_session"],
    "FK2": ["full_dispatch_request", "wsgi_app"],
    "RQ1": ["resolve_redirects"],
    "RQ2": ["get_adapter", "HTTPAdapter"],
    "PY1": ["ModelMetaclass", "complete_model_class", "collect_model_fields"],
    "PY2": ["model_validate", "validate_python", "__pydantic_validator__"],
}
MARKER_RE = re.compile(r"\[trial marker: ([A-Z]{2}\d)-(sutra|control)-t(\d)\]")


def analyze_file(path: Path):
    text_head = path.read_text(errors="replace")
    m = MARKER_RE.search(text_head[:20000])
    if not m:
        return None
    ticket, arm, trial = m.group(1), m.group(2), int(m.group(3))
    golds = GOLD_MARKERS[ticket]
    counts = Counter()
    first_gold_call = None
    call_index = 0
    out_tokens = 0
    cache_write_tokens = 0
    for line in text_head.splitlines():
        try:
            rec = json.loads(line)
        except Exception:
            continue
        msg = rec.get("message") or {}
        usage = msg.get("usage") or {}
        out_tokens += usage.get("output_tokens", 0) or 0
        cache_write_tokens += usage.get("cache_creation_input_tokens", 0) or 0
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for b in content:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "tool_use":
                call_index += 1
                counts[b.get("name")] += 1
                blob = json.dumps(b.get("input", {}))
                if first_gold_call is None and any(g in blob for g in golds):
                    first_gold_call = call_index
            if b.get("type") == "tool_result" and first_gold_call is None:
                blob = json.dumps(b.get("content", ""))[:200000]
                if any(g in blob for g in golds):
                    first_gold_call = call_index
    sutra_calls = sum(v for k, v in counts.items() if k and k.startswith("mcp__sutra__"))
    return {
        "ticket": ticket, "arm": arm, "trial": trial,
        "sutra": sutra_calls,
        "bash": counts.get("Bash", 0), "grep": counts.get("Grep", 0),
        "glob": counts.get("Glob", 0), "read": counts.get("Read", 0),
        "total_tool_calls": sum(counts.values()),
        "first_gold_call": first_gold_call,
        "output_tokens": out_tokens,
        "cache_write_tokens": cache_write_tokens,
    }


def main():
    tdir = Path(sys.argv[1])
    rows = []
    for f in sorted(tdir.glob("agent-*.jsonl")):
        r = analyze_file(f)
        if r:
            rows.append(r)
    out = Path(__file__).parent / "ab_efficiency.json"
    out.write_text(json.dumps(rows, indent=1))
    print(f"{len(rows)} solver transcripts matched -> {out}")

    by_arm = defaultdict(list)
    for r in rows:
        by_arm[r["arm"]].append(r)
    for arm, rs in sorted(by_arm.items()):
        n = len(rs)
        med = lambda xs: sorted(xs)[len(xs) // 2] if xs else None
        loc = [r["first_gold_call"] for r in rs if r["first_gold_call"]]
        print(f"\n{arm}: n={n}")
        print(f"  median tool calls: {med([r['total_tool_calls'] for r in rs])}")
        print(f"  median calls-to-gold: {med(loc)}  (localized in {len(loc)}/{n})")
        print(f"  median output tokens: {med([r['output_tokens'] for r in rs])}")
        print(f"  median cache-write tokens: {med([r['cache_write_tokens'] for r in rs])}")
        print(f"  mean sutra calls: {sum(r['sutra'] for r in rs)/n:.1f}")
    # paired per-ticket medians
    print("\nper-ticket median calls-to-gold (sutra | control):")
    tickets = sorted({r["ticket"] for r in rows})
    med = lambda xs: sorted(xs)[len(xs) // 2] if xs else None
    for t in tickets:
        s = [r["first_gold_call"] for r in rows if r["ticket"] == t and r["arm"] == "sutra" and r["first_gold_call"]]
        c = [r["first_gold_call"] for r in rows if r["ticket"] == t and r["arm"] == "control" and r["first_gold_call"]]
        print(f"  {t}: {med(s)} | {med(c)}")


if __name__ == "__main__":
    main()
