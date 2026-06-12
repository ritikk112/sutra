"""
The Sutra MCP server — THE product (SUTRA_PHASE2.md, locked decision #3).

An AI agent connects over MCP and queries a team's indexed repos.  The
whole consumer stack is in-memory, built from a directory of artifacts at
boot: zero Postgres, zero infrastructure.  `pip install` + an artifacts
directory is the entire deployment story.

    python -m sutra.mcp --artifacts-dir ~/sutra-artifacts          # stdio
    python -m sutra.mcp --artifacts-dir ~/sutra-artifacts --http   # team server
"""
