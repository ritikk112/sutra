"""Read / merge / write ``config/sutra.yaml`` + ``.env``.

Rules enforced here (the wizard depends on them):

* **Atomic writes** — temp file in the same directory + ``os.replace`` so a
  crash mid-write never leaves a truncated config.
* **Diff before overwrite** — :func:`plan_write` returns the exact current-vs-new
  text; the caller shows it and requires confirmation before :func:`commit_write`.
* **Secrets only in ``.env``** — the YAML embedder block is built from a strict
  whitelist that contains ``api_key_env`` (the *name* of the env var) but never a
  raw key.  Actual secret *values* are written to ``.env`` (which is gitignored).

This module never prompts; it produces plans and commits them.
"""
from __future__ import annotations

import contextlib
import difflib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import yaml

# Keys allowed into the YAML `embedder:` block.  Deliberately excludes anything
# that could hold a secret value (there is no raw-key field here by design).
_YAML_EMBEDDER_KEYS = (
    "provider",
    "model",
    "dimensions",
    "batch_size",
    "api_key_env",
    "base_url",
)

# Env var names whose *values* are redacted in the shown diff.
_SECRET_HINTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PG_URL")

DEFAULT_ARTIFACTS_DIR = Path.home() / ".sutra" / "artifacts"


# ---------------------------------------------------------------------------
# Path defaults
# ---------------------------------------------------------------------------

def default_artifacts_dir() -> Path:
    raw = os.environ.get("SUTRA_ARTIFACTS_DIR", "").strip()
    return Path(raw).expanduser() if raw else DEFAULT_ARTIFACTS_DIR


def default_config_path() -> Path:
    return Path("config/sutra.yaml")


def default_env_path() -> Path:
    return Path(".env")


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def read_yaml(path: Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def read_env(path: Path) -> dict[str, str]:
    p = Path(path)
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


# ---------------------------------------------------------------------------
# Building / merging
# ---------------------------------------------------------------------------

def sanitize_embedder_block(embedder_cfg: dict) -> dict:
    """Project an embedder config dict onto the YAML-safe whitelist.

    Empty/None optional values are dropped so we do not persist ``base_url: ''``.
    ``provider`` is always present.  Any raw-secret-looking key is refused loudly.
    """
    for k in embedder_cfg:
        if k not in _YAML_EMBEDDER_KEYS and k.lower() in {"api_key", "apikey", "key", "token"}:
            raise ValueError(
                f"Refusing to write secret-looking key {k!r} into YAML; "
                "secrets belong in .env only."
            )
    block: dict = {"provider": embedder_cfg.get("provider", "fixture")}
    for k in _YAML_EMBEDDER_KEYS:
        if k == "provider":
            continue
        val = embedder_cfg.get(k)
        if val is None or val == "":
            continue
        block[k] = val
    return block


def merge_config(existing: dict, embedder_block: dict) -> dict:
    """Replace only the ``embedder`` block, preserving any other top-level keys."""
    merged = dict(existing or {})
    merged["embedder"] = embedder_block
    return merged


def render_yaml(config: dict) -> str:
    return yaml.safe_dump(config, sort_keys=False, default_flow_style=False)


def render_env(env_vars: dict[str, str]) -> str:
    lines = [f"{k}={env_vars[k]}" for k in sorted(env_vars)]
    return ("\n".join(lines) + "\n") if lines else ""


def _redact_env_text(text: str) -> str:
    out: list[str] = []
    for line in text.splitlines():
        key = line.split("=", 1)[0].strip().upper()
        if "=" in line and any(h in key for h in _SECRET_HINTS):
            out.append(f"{line.split('=', 1)[0]}=***redacted***")
        else:
            out.append(line)
    return "\n".join(out) + ("\n" if out else "")


def diff_text(old: str, new: str, label: str) -> str:
    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"{label} (current)",
            tofile=f"{label} (new)",
        )
    )


# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------

def atomic_write(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` atomically (temp file + ``os.replace``)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=".sutra-tmp-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
        os.replace(tmp, p)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


# ---------------------------------------------------------------------------
# Plan / commit
# ---------------------------------------------------------------------------

@dataclass
class WritePlan:
    config_path: Path
    env_path: Path
    config_old: str
    config_new: str
    env_old: str
    env_new: str

    @property
    def config_changed(self) -> bool:
        return self.config_old != self.config_new

    @property
    def env_changed(self) -> bool:
        return self.env_old != self.env_new

    @property
    def anything_changed(self) -> bool:
        return self.config_changed or self.env_changed

    def config_diff(self) -> str:
        return diff_text(self.config_old, self.config_new, str(self.config_path))

    def env_diff(self) -> str:
        # Redact secret values so the diff can be printed to the terminal safely.
        return diff_text(
            _redact_env_text(self.env_old),
            _redact_env_text(self.env_new),
            str(self.env_path),
        )


def plan_write(
    config_path: Path,
    env_path: Path,
    embedder_cfg: dict,
    env_updates: dict[str, str],
) -> WritePlan:
    """Compute the exact new file contents without writing anything.

    * The YAML embedder block replaces only ``embedder:`` in the existing config.
    * ``.env`` updates are merged over any existing values (blank values drop the
      key rather than persisting an empty assignment).
    """
    config_path = Path(config_path)
    env_path = Path(env_path)

    config_old = config_path.read_text(encoding="utf-8") if config_path.exists() else ""
    existing_config = read_yaml(config_path)
    block = sanitize_embedder_block(embedder_cfg)
    new_config = merge_config(existing_config, block)
    config_new = render_yaml(new_config)

    env_old_text = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
    merged_env = read_env(env_path)
    for k, v in env_updates.items():
        if v is None or v == "":
            merged_env.pop(k, None)
        else:
            merged_env[k] = v
    env_new = render_env(merged_env)

    return WritePlan(
        config_path=config_path,
        env_path=env_path,
        config_old=config_old,
        config_new=config_new,
        env_old=env_old_text,
        env_new=env_new,
    )


def commit_write(plan: WritePlan) -> list[Path]:
    """Atomically write whichever files actually changed.  Returns written paths."""
    written: list[Path] = []
    if plan.config_changed:
        atomic_write(plan.config_path, plan.config_new)
        written.append(plan.config_path)
    if plan.env_changed:
        atomic_write(plan.env_path, plan.env_new)
        written.append(plan.env_path)
    return written
