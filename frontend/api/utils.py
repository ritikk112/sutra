from __future__ import annotations

import re


GIT_URL_RE = re.compile(r"^(https?|git|ssh)://.+")
GIT_SSH_RE = re.compile(r"^git@.+:.+\.git$")
ARTIFACT_ALLOWLIST = {"graph.json", "embeddings.npy", "embeddings_index.json"}


def looks_like_git_url(url: str) -> bool:
    val = url.strip()
    return bool(GIT_URL_RE.match(val) or GIT_SSH_RE.match(val))
