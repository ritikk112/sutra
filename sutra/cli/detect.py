"""Read-only environment detection — pure functions, no prompting, no writes.

Every check returns a small :class:`CheckResult` (name, ok, detail, hint) so the
same functions can back both the interactive wizard (``sutra init``) and the
non-interactive ``sutra doctor`` report.  Nothing here prompts, installs, or
mutates state; the heaviest thing a check does is ``shutil.which`` or an
``importlib.util.find_spec`` (which does not import the module).
"""
from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class CheckResult:
    """Outcome of a single detection check.

    ``ok`` is a tri-state via ``critical``: a check can be informational
    (``critical=False``) so ``doctor`` does not fail the run just because an
    *optional* capability (docker, GPU, pyright) is absent.
    """

    name: str
    ok: bool
    detail: str
    hint: str = ""
    critical: bool = False


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def python_version() -> CheckResult:
    v = sys.version_info
    ok = (v.major, v.minor) >= (3, 11)
    detail = f"{v.major}.{v.minor}.{v.micro} ({platform.system()} {platform.machine()})"
    return CheckResult(
        name="Python version",
        ok=ok,
        detail=detail,
        hint="" if ok else "Sutra requires Python >= 3.11.",
        critical=True,
    )


def in_virtualenv() -> CheckResult:
    # sys.prefix != sys.base_prefix is the canonical venv/virtualenv signal.
    active = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    detail = sys.prefix if active else f"system interpreter at {sys.prefix}"
    return CheckResult(
        name="Virtual environment",
        ok=active,
        detail=detail,
        hint="" if active else "Not in a venv — installing packages may touch system Python.",
        critical=False,
    )


def docker_present() -> CheckResult:
    path = shutil.which("docker")
    return CheckResult(
        name="Docker",
        ok=path is not None,
        detail=path or "not found on PATH",
        hint="" if path else "Optional — only needed for the bundled pgvector container.",
        critical=False,
    )


def gpu_present() -> CheckResult:
    # Read-only signal: presence of nvidia-smi on PATH.  We do NOT execute it or
    # import torch (importing torch is multi-second and has side effects).
    path = shutil.which("nvidia-smi")
    return CheckResult(
        name="GPU (NVIDIA)",
        ok=path is not None,
        detail="nvidia-smi found" if path else "no nvidia-smi on PATH",
        hint="" if path else "Optional — local embeddings run on CPU without a GPU.",
        critical=False,
    )


def pyright_langserver_present() -> CheckResult:
    # Mirror frontend/api/main.py: check next to the interpreter then PATH.
    candidate = Path(sys.executable).parent / "pyright-langserver"
    found = candidate.exists() or shutil.which("pyright-langserver") is not None
    where = str(candidate) if candidate.exists() else (shutil.which("pyright-langserver") or "")
    return CheckResult(
        name="pyright-langserver",
        ok=found,
        detail=where or "not found",
        hint="" if found else "Needed only for --resolver lsp and the web UI; `pip install pyright`.",
        critical=False,
    )


def command_present(command: str, *, optional: bool = True) -> CheckResult:
    path = shutil.which(command)
    return CheckResult(
        name=f"`{command}` on PATH",
        ok=path is not None,
        detail=path or "not found",
        hint="" if path else f"`{command}` not found.",
        critical=not optional,
    )


def claude_cli_present() -> CheckResult:
    path = shutil.which("claude")
    return CheckResult(
        name="Claude Code CLI",
        ok=path is not None,
        detail=path or "not found",
        hint="" if path else "Optional — used to auto-register the Sutra MCP server.",
        critical=False,
    )


def env_var_set(name: str) -> CheckResult:
    val = os.environ.get(name, "")
    ok = bool(val.strip())
    return CheckResult(
        name=f"env {name}",
        ok=ok,
        detail="set" if ok else "unset/empty",
        hint="" if ok else f"Export {name} before running.",
        critical=False,
    )


def sentence_transformers_importable() -> CheckResult:
    # find_spec does NOT import the (heavy) package — safe for a detection pass.
    spec = importlib.util.find_spec("sentence_transformers")
    ok = spec is not None
    return CheckResult(
        name="sentence-transformers",
        ok=ok,
        detail="installed" if ok else "not installed",
        hint="" if ok else "Needed for the local embedder; the wizard can install it.",
        critical=False,
    )


def artifacts_dir_exists(path: Path) -> CheckResult:
    p = Path(path).expanduser()
    ok = p.is_dir()
    n = 0
    if ok:
        n = sum(1 for child in p.iterdir() if child.is_dir())
    return CheckResult(
        name="Artifacts directory",
        ok=ok,
        detail=f"{p} ({n} repo artifact(s))" if ok else f"{p} does not exist",
        hint="" if ok else "Created automatically the first time you `sutra index`.",
        critical=False,
    )


# ---------------------------------------------------------------------------
# Existing-config loaders (read-only; used to offer current values as defaults)
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    """Load an existing ``config/sutra.yaml`` (or ``{}`` when absent/empty)."""
    p = Path(path).expanduser()
    if not p.exists():
        return {}
    try:
        with open(p, encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except (OSError, yaml.YAMLError):
        return {}


def load_env(path: Path) -> dict[str, str]:
    """Parse a ``.env`` file into a plain dict (``export`` prefix + quotes handled)."""
    p = Path(path).expanduser()
    result: dict[str, str] = {}
    if not p.exists():
        return result
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        if key:
            result[key] = val
    return result
