"""The "choose an embedder" step and its live validation probe.

Split cleanly into:

* **pure validation** — :func:`probe_embedder` (build via the cross-agent
  ``factory.from_dict`` contract, ``.embed(["probe"])``, read ``.dimensions``) and
  :func:`discover_dimensions` (raw endpoint call to auto-discover the vector
  width of an OpenAI-compatible server).  Both are unit-testable with a real
  ``FixtureEmbedder`` or a real local dummy HTTP server — no mocks.
* **prompting** — :func:`choose_embedder`, the questionary arrow-key menu that
  gathers fields and drives the probe with a "skip validation" escape hatch.

``PromptIO`` (input/output for questionary) and the questionary/prompt_toolkit
compatibility shim also live here and are reused by ``init.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sutra.core.embedder.factory import from_dict


class WizardAborted(Exception):
    """Raised when the user cancels a prompt (Ctrl-C / ESC) mid-wizard."""


# ---------------------------------------------------------------------------
# questionary <-> prompt_toolkit compatibility shim
# ---------------------------------------------------------------------------
# questionary 2.1.0's cosmetic `_fix_unecessary_blank_lines` assumes a
# prompt_toolkit layout structure that changed in prompt_toolkit 3.0.52, so every
# `select` prompt raises AttributeError('VSplit' ...).  The function only tweaks
# window sizing, so wrapping it defensively is safe and keeps arrow-key menus
# working.  Applied once, idempotently, on import.
_QUESTIONARY_PATCHED = False


def _apply_questionary_patch() -> None:
    global _QUESTIONARY_PATCHED
    if _QUESTIONARY_PATCHED:
        return
    try:
        import questionary.prompts.common as _qc

        _orig = _qc._fix_unecessary_blank_lines

        def _safe(ps: Any) -> None:
            try:
                _orig(ps)
            except Exception:
                # Purely cosmetic (window height); ignore layout drift.
                pass

        _qc._fix_unecessary_blank_lines = _safe
    except Exception:
        pass
    _QUESTIONARY_PATCHED = True


@dataclass
class PromptIO:
    """Injectable stdin/stdout for questionary so the flow is scriptable in tests.

    In production both are ``None`` and questionary uses the real terminal.  Tests
    pass a ``prompt_toolkit`` pipe input and a ``DummyOutput`` to feed scripted
    keystrokes.
    """

    input: Any = None
    output: Any = None

    def kwargs(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if self.input is not None:
            d["input"] = self.input
        if self.output is not None:
            d["output"] = self.output
        return d


def _ask(question: Any) -> Any:
    """Run a questionary question; treat cancellation (None) as an abort signal."""
    result = question.ask()
    if result is None:
        raise WizardAborted()
    return result


def ask_text(io: PromptIO, message: str, *, default: str = "") -> str:
    import questionary

    _apply_questionary_patch()
    return _ask(questionary.text(message, default=default, **io.kwargs()))


def ask_password(io: PromptIO, message: str) -> str:
    import questionary

    _apply_questionary_patch()
    return _ask(questionary.password(message, **io.kwargs()))


def ask_confirm(io: PromptIO, message: str, *, default: bool = True) -> bool:
    import questionary

    _apply_questionary_patch()
    return bool(_ask(questionary.confirm(message, default=default, **io.kwargs())))


def ask_select(io: PromptIO, message: str, choices: list[Any], *, default: Any = None) -> str:
    import questionary

    _apply_questionary_patch()
    kwargs = io.kwargs()
    if default is not None:
        kwargs["default"] = default
    return _ask(questionary.select(message, choices=choices, **kwargs))


# ---------------------------------------------------------------------------
# Pure validation
# ---------------------------------------------------------------------------

@dataclass
class ProbeResult:
    ok: bool
    dimensions: int | None
    vector_len: int | None
    model_id: str | None
    error: str | None = None


def probe_embedder(embedder_cfg: dict, *, probe_text: str = "probe") -> ProbeResult:
    """Build a candidate embedder from ``embedder_cfg`` and do one live embed.

    Mirrors exactly what the wizard needs before writing any config: construct via
    the shared ``from_dict`` contract, embed a single probe string, and read back
    the declared ``dimensions`` alongside the real returned vector length.
    Any failure (bad key, unreachable endpoint, dimension mismatch) is captured as
    ``ok=False`` with a human-readable ``error`` rather than raising.
    """
    try:
        embedder = from_dict({"embedder": embedder_cfg})
        vectors = embedder.embed([probe_text])
        vec_len = int(vectors.shape[1]) if getattr(vectors, "ndim", 0) == 2 else len(vectors[0])
        return ProbeResult(
            ok=True,
            dimensions=int(embedder.dimensions),
            vector_len=vec_len,
            model_id=embedder.model_id,
        )
    except Exception as exc:
        return ProbeResult(
            ok=False,
            dimensions=None,
            vector_len=None,
            model_id=None,
            error=f"{type(exc).__name__}: {exc}",
        )


def discover_dimensions(
    base_url: str,
    model: str,
    *,
    api_key: str | None = None,
    probe_text: str = "probe",
) -> int:
    """Auto-discover an OpenAI-compatible endpoint's native embedding width.

    ``OpenAIEmbedder`` requires ``dimensions`` up front when ``base_url`` is set,
    so we cannot build it to discover the width.  Instead we make one raw
    ``/v1/embeddings`` call (no ``dimensions`` param) and read the returned vector
    length.  This is the value the wizard then stores in YAML and re-validates.
    """
    from openai import OpenAI

    client = OpenAI(
        base_url=base_url,
        api_key=api_key or "sutra-no-key-required",
        max_retries=1,
    )
    resp = client.embeddings.create(model=model, input=[probe_text])
    return len(resp.data[0].embedding)


# ---------------------------------------------------------------------------
# Result of the interactive step
# ---------------------------------------------------------------------------

@dataclass
class EmbedderSetupResult:
    embedder_cfg: dict
    env_vars: dict[str, str] = field(default_factory=dict)
    dimensions: int | None = None
    validated: bool = False


# Menu labels -> internal kind
_LOCAL = "Local  (free, offline — sentence-transformers)"
_OPENAI = "OpenAI  (hosted API — needs OPENAI_API_KEY)"
_COMPAT = "OpenAI-compatible endpoint  (Ollama / LM Studio / vLLM / Together / Azure …)"


# ---------------------------------------------------------------------------
# Interactive step
# ---------------------------------------------------------------------------

def choose_embedder(
    io: PromptIO,
    *,
    defaults: dict | None = None,
    emit: Any = None,
) -> EmbedderSetupResult:
    """Interactive embedder selection + optional live probe.

    ``defaults`` is an existing ``embedder`` block (from a prior run) used to
    pre-fill answers.  ``emit`` is an optional ``callable(str)`` for status lines.
    Returns the chosen embedder config, any secret env vars to persist, and the
    discovered dimensions.  Never raises on a *validation* failure — it loops back
    or lets the user skip.
    """
    defaults = defaults or {}

    def say(msg: str) -> None:
        if emit is not None:
            emit(msg)

    prior_provider = defaults.get("provider")
    prior_base_url = str(defaults.get("base_url") or "")
    if prior_base_url:
        default_choice = _COMPAT
    elif prior_provider == "openai":
        default_choice = _OPENAI
    elif prior_provider == "local":
        default_choice = _LOCAL
    else:
        default_choice = _LOCAL

    choice = ask_select(
        io,
        "Which embedder should Sutra use?",
        choices=[_LOCAL, _OPENAI, _COMPAT],
        default=default_choice,
    )

    if choice == _LOCAL:
        return _setup_local(io, defaults, say)
    if choice == _OPENAI:
        return _setup_openai(io, defaults, say)
    return _setup_compatible(io, defaults, say)


def _setup_local(io: PromptIO, defaults: dict, say: Any) -> EmbedderSetupResult:
    from sutra.cli import detect

    model = ask_text(
        io,
        "Local model name:",
        default=str(defaults.get("model") or "all-MiniLM-L6-v2"),
    )
    dims = int(defaults.get("dimensions") or 384)
    batch_size = int(defaults.get("batch_size") or 32)
    cfg = {
        "provider": "local",
        "model": model,
        "dimensions": dims,
        "batch_size": batch_size,
    }

    if not detect.sentence_transformers_importable().ok:
        say("sentence-transformers is not installed.")
        if ask_confirm(
            io,
            "Install it now (CPU-only torch, no CUDA)? This can take a few minutes.",
            default=False,
        ):
            from sutra.cli import provision

            res = provision.install_cpu_torch(on_output=say)
            if not res.ok:
                say(f"Install failed: {res.message}. Continuing; validation will be skipped.")

    return _validate_or_skip(io, cfg, say)


def _setup_openai(io: PromptIO, defaults: dict, say: Any) -> EmbedderSetupResult:
    import os

    model = ask_text(
        io,
        "OpenAI embedding model:",
        default=str(defaults.get("model") or "text-embedding-3-small"),
    )
    api_key_env = ask_text(
        io,
        "Env var holding your API key:",
        default=str(defaults.get("api_key_env") or "OPENAI_API_KEY"),
    ).strip()

    env_vars: dict[str, str] = {}
    if not os.environ.get(api_key_env, "").strip():
        if ask_confirm(io, f"{api_key_env} is not set. Enter the key now?", default=True):
            key = ask_password(io, f"{api_key_env} value (stored only in .env):").strip()
            if key:
                env_vars[api_key_env] = key
                os.environ[api_key_env] = key  # so the immediate probe can use it

    cfg = {
        "provider": "openai",
        "model": model,
        "dimensions": int(defaults.get("dimensions") or 1536),
        "batch_size": int(defaults.get("batch_size") or 100),
        "api_key_env": api_key_env,
    }
    result = _validate_or_skip(io, cfg, say)
    result.env_vars.update(env_vars)
    return result


def _setup_compatible(io: PromptIO, defaults: dict, say: Any) -> EmbedderSetupResult:
    import os

    base_url = ask_text(
        io,
        "Endpoint base URL (…/v1):",
        default=str(defaults.get("base_url") or "http://localhost:11434/v1"),
    ).strip()
    model = ask_text(
        io,
        "Model name:",
        default=str(defaults.get("model") or "nomic-embed-text"),
    )
    api_key_env = ask_text(
        io,
        "Env var holding the API key (blank if none, e.g. Ollama):",
        default=str(defaults.get("api_key_env") or ""),
    ).strip()

    env_vars: dict[str, str] = {}
    api_key: str | None = None
    if api_key_env:
        api_key = os.environ.get(api_key_env, "").strip() or None
        if not api_key and ask_confirm(
            io, f"{api_key_env} is not set. Enter it now?", default=True
        ):
            entered = ask_password(io, f"{api_key_env} value (stored only in .env):").strip()
            if entered:
                api_key = entered
                env_vars[api_key_env] = entered
                os.environ[api_key_env] = entered

    cfg: dict = {
        "provider": "openai",
        "model": model,
        "base_url": base_url,
        "api_key_env": api_key_env,
        "batch_size": int(defaults.get("batch_size") or 100),
    }
    if defaults.get("dimensions"):
        cfg["dimensions"] = int(defaults["dimensions"])

    # Compatible endpoints require dimensions; discover them via a raw probe.
    validate = ask_confirm(io, "Probe the endpoint now to auto-discover dimensions?", default=True)
    if not validate:
        # Cannot proceed without dimensions for a base_url embedder; ask the user.
        dims_txt = ask_text(io, "Vector dimensions for this model:", default=str(cfg.get("dimensions") or "768"))
        cfg["dimensions"] = int(dims_txt)
        return EmbedderSetupResult(embedder_cfg=cfg, env_vars=env_vars, dimensions=cfg["dimensions"], validated=False)

    while True:
        try:
            dims = discover_dimensions(base_url, model, api_key=api_key)
            cfg["dimensions"] = int(dims)
            say(f"Discovered {dims} dimensions from {base_url}.")
        except Exception as exc:
            say(f"Endpoint probe failed: {type(exc).__name__}: {exc}")
            action = ask_select(
                io,
                "How do you want to proceed?",
                choices=["Retry", "Enter dimensions manually (skip validation)", "Abort embedder setup"],
            )
            if action.startswith("Retry"):
                continue
            if action.startswith("Enter"):
                dims_txt = ask_text(io, "Vector dimensions:", default=str(cfg.get("dimensions") or "768"))
                cfg["dimensions"] = int(dims_txt)
                return EmbedderSetupResult(cfg, env_vars, cfg["dimensions"], validated=False)
            raise WizardAborted()

        probe = probe_embedder(cfg)
        if probe.ok:
            say(f"Validated: {probe.model_id} -> {probe.dimensions} dims.")
            result = EmbedderSetupResult(cfg, env_vars, probe.dimensions, validated=True)
            return result
        say(f"Validation failed: {probe.error}")
        if not ask_confirm(io, "Retry the endpoint?", default=True):
            return EmbedderSetupResult(cfg, env_vars, cfg.get("dimensions"), validated=False)


def _validate_or_skip(io: PromptIO, cfg: dict, say: Any) -> EmbedderSetupResult:
    """Shared validate-with-escape-hatch loop for Local / OpenAI."""
    if not ask_confirm(io, "Validate now with a live probe embed? (recommended)", default=True):
        return EmbedderSetupResult(cfg, {}, cfg.get("dimensions"), validated=False)

    while True:
        probe = probe_embedder(cfg)
        if probe.ok:
            say(f"Validated: {probe.model_id} -> {probe.dimensions} dims.")
            return EmbedderSetupResult(cfg, {}, probe.dimensions, validated=True)
        say(f"Validation failed: {probe.error}")
        action = ask_select(
            io,
            "How do you want to proceed?",
            choices=["Retry", "Skip validation and continue", "Abort embedder setup"],
        )
        if action.startswith("Retry"):
            continue
        if action.startswith("Skip"):
            return EmbedderSetupResult(cfg, {}, cfg.get("dimensions"), validated=False)
        raise WizardAborted()
