from frontend.api.utils import ARTIFACT_ALLOWLIST, looks_like_git_url


def test_git_url_validation_accepts_expected_patterns() -> None:
    assert looks_like_git_url("https://github.com/gin-gonic/gin")
    assert looks_like_git_url("git://example.com/org/repo.git")
    assert looks_like_git_url("git@github.com:org/repo.git")


def test_git_url_validation_rejects_invalid_values() -> None:
    assert not looks_like_git_url("")
    assert not looks_like_git_url("github.com/org/repo")
    assert not looks_like_git_url("ftp://example.com/repo")


def test_artifact_allowlist_is_literal_expected_set() -> None:
    assert ARTIFACT_ALLOWLIST == {"graph.json", "embeddings.npy", "embeddings_index.json"}
