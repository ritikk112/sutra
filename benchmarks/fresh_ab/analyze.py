"""Efficiency + latency metrics for the fresh sutra-vs-control benchmark.

Ticket-agnostic: gold markers are loaded from tickets.json rather than hardcoded,
so the same analyzer serves the pilot and the main run.

Per solver transcript (matched via the '[trial marker: <ID>-<arm>-t<n>]' line
embedded in each solver prompt) it computes:

  navigation   tool-call counts by tool; calls-to-gold (index of the first call
               whose INPUT or RESULT mentions a gold marker)
  latency      wall_clock_s (first->last record), model_time_s, tool_time_s,
               unclassified_s, max_gap_s, and time_to_first_gold_s
  cost         output tokens, cache-write tokens (context growth proxy)
  hygiene      sutra-tool call count (adoption in the sutra arm; contamination
               in the control arm)
  contention   overlap_peak / overlap_mean: how many OTHER solvers were running
               concurrently during this one's span, derived post-hoc from
               timestamps. Wall clock is only comparable between arms if this is
               balanced, so it is measured rather than assumed.

Usage: python3 analyze.py <transcript_dir> [tickets.json] [-o out.json]
"""
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent
MARKER_RE = re.compile(r"\[trial marker: ([A-Z]{2}\d+)-(sutra|control)-t(\d+)\]")


def load_markers(tickets_path: Path) -> dict:
    data = json.loads(tickets_path.read_text())
    tickets = data["tickets"] if isinstance(data, dict) else data
    return {t["id"]: t["gold_markers"] for t in tickets}


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def analyze_file(path: Path, markers_by_ticket: dict):
    text = path.read_text(errors="replace")
    m = MARKER_RE.search(text[:20000])
    if not m:
        return None
    ticket, arm, trial = m.group(1), m.group(2), int(m.group(3))
    golds = markers_by_ticket.get(ticket)
    if golds is None:
        return None

    counts = Counter()
    first_gold_call = None
    first_gold_time = None
    call_index = 0
    out_tokens = cache_write_tokens = 0
    times = []           # every record timestamp, in order
    model_s = tool_s = other_s = 0.0
    max_gap = 0.0
    prev_t = None
    prev_had_tool_use = False

    for line in text.splitlines():
        try:
            rec = json.loads(line)
        except Exception:
            continue
        t = ts(rec["timestamp"]) if rec.get("timestamp") else None
        msg = rec.get("message") or {}
        usage = msg.get("usage") or {}
        out_tokens += usage.get("output_tokens", 0) or 0
        cache_write_tokens += usage.get("cache_creation_input_tokens", 0) or 0

        content = msg.get("content")
        blocks = content if isinstance(content, list) else []
        has_tool_use = any(isinstance(b, dict) and b.get("type") == "tool_use" for b in blocks)
        has_tool_result = any(isinstance(b, dict) and b.get("type") == "tool_result" for b in blocks)

        if t is not None:
            times.append(t)
            if prev_t is not None:
                gap = (t - prev_t).total_seconds()
                max_gap = max(max_gap, gap)
                # gap ending in a tool_result = the tool was executing;
                # gap ending in an assistant turn = the model was generating.
                if has_tool_result and prev_had_tool_use:
                    tool_s += gap
                elif rec.get("type") == "assistant":
                    model_s += gap
                else:
                    other_s += gap
            prev_t = t
            prev_had_tool_use = has_tool_use

        for b in blocks:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "tool_use":
                name = b.get("name")
                # ToolSearch loads MCP schemas; it is a harness artifact, not
                # investigation, and only the sutra arm needs it. Counting it
                # would hand the control arm a free call.
                if name == "ToolSearch":
                    continue
                call_index += 1
                counts[name] += 1
                blob = json.dumps(b.get("input", {}))
                if first_gold_call is None and any(g in blob for g in golds):
                    first_gold_call, first_gold_time = call_index, t
            if b.get("type") == "tool_result" and first_gold_call is None:
                blob = json.dumps(b.get("content", ""))[:200000]
                if any(g in blob for g in golds):
                    first_gold_call, first_gold_time = call_index, t

    if not times:
        return None
    start, end = times[0], times[-1]
    wall = (end - start).total_seconds()
    sutra_calls = sum(v for k, v in counts.items() if k and k.startswith("mcp__sutra__"))

    return {
        "ticket": ticket, "arm": arm, "trial": trial,
        "transcript": path.name,
        # navigation
        "sutra": sutra_calls,
        "bash": counts.get("Bash", 0), "grep": counts.get("Grep", 0),
        "glob": counts.get("Glob", 0), "read": counts.get("Read", 0),
        "total_tool_calls": sum(counts.values()),
        "first_gold_call": first_gold_call,
        # latency
        "wall_clock_s": round(wall, 1),
        "model_time_s": round(model_s, 1),
        "tool_time_s": round(tool_s, 1),
        "unclassified_s": round(other_s, 1),
        "max_gap_s": round(max_gap, 1),
        "time_to_first_gold_s": round((first_gold_time - start).total_seconds(), 1)
                                if first_gold_time else None,
        # cost
        "output_tokens": out_tokens,
        "cache_write_tokens": cache_write_tokens,
        # for contention pass
        "_start": start.isoformat(), "_end": end.isoformat(),
    }


def add_contention(rows):
    """How many OTHER solvers overlapped each row's span. Wall clock is only
    comparable across arms if this is balanced between them."""
    spans = [(ts(r["_start"]), ts(r["_end"]), r) for r in rows]
    for s, e, r in spans:
        overlaps = sum(1 for s2, e2, r2 in spans if r2 is not r and s2 < e and e2 > s)
        r["overlap_count"] = overlaps


def med(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else round((xs[n // 2 - 1] + xs[n // 2]) / 2, 1)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    tdir = Path(args[0])
    tickets_path = Path(args[1]) if len(args) > 1 else HERE / "tickets.json"
    out_path = HERE / (sys.argv[sys.argv.index("-o") + 1] if "-o" in sys.argv else "efficiency.json")

    markers = load_markers(tickets_path)
    rows = []
    for f in sorted(tdir.rglob("agent-*.jsonl")):
        r = analyze_file(f, markers)
        if r:
            rows.append(r)
    add_contention(rows)
    out_path.write_text(json.dumps(rows, indent=1))
    print(f"{len(rows)} solver transcripts matched -> {out_path}")
    if not rows:
        return

    by_arm = defaultdict(list)
    for r in rows:
        by_arm[r["arm"]].append(r)

    METRICS = [
        ("calls-to-gold", "first_gold_call"),
        ("total tool calls", "total_tool_calls"),
        ("wall clock (s)", "wall_clock_s"),
        ("model time (s)", "model_time_s"),
        ("tool time (s)", "tool_time_s"),
        ("time-to-first-gold (s)", "time_to_first_gold_s"),
        ("max gap (s)", "max_gap_s"),
        ("output tokens", "output_tokens"),
        ("cache-write tokens", "cache_write_tokens"),
        ("concurrent solvers", "overlap_count"),
    ]
    for arm in sorted(by_arm):
        rs = by_arm[arm]
        loc = [r for r in rs if r["first_gold_call"]]
        print(f"\n{arm}: n={len(rs)}  (localized in {len(loc)}/{len(rs)})")
        for label, key in METRICS:
            print(f"  median {label:<24} {med([r[key] for r in rs])}")
        print(f"  mean sutra calls          {sum(r['sutra'] for r in rs)/len(rs):.2f}")

    print("\nper-ticket median (sutra | control):")
    print(f"  {'ticket':<7} {'calls-to-gold':<15} {'wall clock s':<14} {'t-to-gold s'}")
    for t in sorted({r["ticket"] for r in rows}):
        cell = lambda arm, key: med([r[key] for r in rows if r["ticket"] == t and r["arm"] == arm])
        print(f"  {t:<7} "
              f"{str(cell('sutra','first_gold_call'))+' | '+str(cell('control','first_gold_call')):<15} "
              f"{str(cell('sutra','wall_clock_s'))+' | '+str(cell('control','wall_clock_s')):<14} "
              f"{str(cell('sutra','time_to_first_gold_s'))+' | '+str(cell('control','time_to_first_gold_s'))}")


if __name__ == "__main__":
    main()
