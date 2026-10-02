"""Sutra's server-level MCP instructions are its only advertising.

Claude Code defers MCP tools: an agent sees six bare `mcp__sutra__*` names plus
these instructions, and no per-tool descriptions until it loads them. The first
version pitched Sutra only for multi-repo and call-graph questions; in the
LocBench adoption pilot (2026-10-02) Claude Code made 0 Sutra calls (and 0
ToolSearch lookups) on 5 single-repo issues. These tests pin the intent of the
text, not its wording.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from sutra.mcp.server import INSTRUCTIONS


def test_instructions_pitch_single_repo_search_by_behaviour():
    text = " ".join(INSTRUCTIONS.split())
    assert "know WHAT the code does but not what it is CALLED" in text
    assert "bug report" in text
    # Honest about when NOT to use it.
    assert "Grep stays the better tool for an exact identifier" in text


def test_instructions_keep_multi_repo_and_graph_guidance():
    text = " ".join(INSTRUCTIONS.split())
    assert "MULTIPLE repositories" in text
    assert "sutra_get_callers" in text and "sutra_expand_neighbors" in text


def test_instructions_do_not_demand_a_list_repos_round_trip_first():
    assert "Start with sutra_list_repos" not in INSTRUCTIONS
    assert "`repo` argument is optional" in INSTRUCTIONS


def test_initialize_serves_the_instructions_over_stdio(tmp_path):
    """A real `sutra serve` hands the text to the client in its initialize result."""
    from sutra.core.embedder.fixture import FixtureEmbedder
    from sutra.core.extractor.adapters.python import PythonAdapter
    from sutra.core.indexer import Indexer
    from sutra.core.output.json_graph_exporter import JsonGraphExporter

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "core.py").write_text("def real():\n    pass\n")
    Indexer(adapters={"python": PythonAdapter()}, exporter=JsonGraphExporter(), embedder=FixtureEmbedder()
            ).index(root=repo, repo_url="https://github.com/t/r", output_dir=tmp_path / "arts" / "t__r")
    exe = Path(sys.executable).parent / ("sutra.exe" if sys.platform == "win32" else "sutra")
    proc = subprocess.Popen(
        [str(exe), "serve", "--artifacts-dir", str(tmp_path / "arts")],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    )
    try:
        proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2025-06-18", "capabilities": {},
            "clientInfo": {"name": "test", "version": "0"}}}) + "\n")
        proc.stdin.flush()
        result = json.loads(proc.stdout.readline())["result"]
    finally:
        proc.kill()
        proc.wait()
    assert result["instructions"] == INSTRUCTIONS
