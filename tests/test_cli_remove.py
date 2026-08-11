"""Tests for `sutra remove` (sutra.cli.remove).

No mocks: a real temp artifacts tree of `<slug>/graph.json` dirs, real path
resolution, real shutil deletes. `run()` is driven with injected confirm/echo
callables (the repo's CLI convention — no CliRunner anywhere).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from sutra.cli import remove as remove_module


def _make_repo(root: Path, slug: str) -> Path:
    d = root / slug
    d.mkdir(parents=True)
    (d / "graph.json").write_text("{}")
    (d / ".ready").write_text("gen-1")
    return d


def _tree(root: Path) -> dict[str, Path]:
    return {
        "frappe__frappe": _make_repo(root, "frappe__frappe"),
        "langgenius__dify": _make_repo(root, "langgenius__dify"),
        "sutra": _make_repo(root, "sutra"),
    }


class TestPlanRemove:
    def test_no_target_plans_all_indexed(self, tmp_path: Path) -> None:
        dirs = _tree(tmp_path)
        plan = remove_module.plan_remove(tmp_path, None)
        assert plan.error is None
        assert set(plan.targets) == set(dirs.values())

    def test_git_url_resolves_to_slug(self, tmp_path: Path) -> None:
        dirs = _tree(tmp_path)
        plan = remove_module.plan_remove(tmp_path, "https://github.com/frappe/frappe.git")
        assert plan.targets == [dirs["frappe__frappe"]]

    def test_owner_repo_name_resolves(self, tmp_path: Path) -> None:
        dirs = _tree(tmp_path)
        plan = remove_module.plan_remove(tmp_path, "langgenius/dify")
        assert plan.targets == [dirs["langgenius__dify"]]

    def test_bare_name_resolves(self, tmp_path: Path) -> None:
        dirs = _tree(tmp_path)
        plan = remove_module.plan_remove(tmp_path, "sutra")
        assert plan.targets == [dirs["sutra"]]

    def test_local_path_resolves_via_local_name(self, tmp_path: Path) -> None:
        """A local dir path indexes as `local/<name>` (== bare `<name>` slug);
        removing by that path must find the same dir."""
        dirs = _tree(tmp_path)
        src = tmp_path / "src" / "sutra"          # a checkout whose name is 'sutra'
        src.mkdir(parents=True)
        plan = remove_module.plan_remove(tmp_path, str(src))
        assert plan.targets == [dirs["sutra"]]

    def test_literal_slug_resolves(self, tmp_path: Path) -> None:
        dirs = _tree(tmp_path)
        plan = remove_module.plan_remove(tmp_path, "frappe__frappe")
        assert plan.targets == [dirs["frappe__frappe"]]

    def test_unknown_target_errors_and_lists(self, tmp_path: Path) -> None:
        _tree(tmp_path)
        plan = remove_module.plan_remove(tmp_path, "nope/missing")
        assert plan.targets == []
        assert plan.error is not None
        assert "nope/missing" in plan.error
        assert "frappe__frappe" in plan.error   # lists what IS indexed


class TestExecuteRemove:
    def test_execute_deletes_target_dirs(self, tmp_path: Path) -> None:
        dirs = _tree(tmp_path)
        plan = remove_module.plan_remove(tmp_path, "sutra")
        remove_module.execute_remove(plan)
        assert not dirs["sutra"].exists()
        assert dirs["frappe__frappe"].exists()   # others untouched


class TestRun:
    def test_yes_removes_specific_repo(self, tmp_path: Path) -> None:
        dirs = _tree(tmp_path)
        out: list[str] = []
        rc = remove_module.run(
            "frappe/frappe", tmp_path, assume_yes=True, echo=out.append
        )
        assert rc == 0
        assert not dirs["frappe__frappe"].exists()
        assert dirs["sutra"].exists()

    def test_decline_removes_nothing(self, tmp_path: Path) -> None:
        dirs = _tree(tmp_path)
        rc = remove_module.run(
            "sutra", tmp_path, confirm=lambda _p: False, echo=lambda _m: None
        )
        assert rc == 1
        assert all(d.exists() for d in dirs.values())

    def test_no_target_with_yes_removes_all(self, tmp_path: Path) -> None:
        dirs = _tree(tmp_path)
        rc = remove_module.run(None, tmp_path, assume_yes=True, echo=lambda _m: None)
        assert rc == 0
        assert all(not d.exists() for d in dirs.values())

    def test_unknown_target_returns_2(self, tmp_path: Path) -> None:
        _tree(tmp_path)
        rc = remove_module.run(
            "ghost", tmp_path, assume_yes=True, echo=lambda _m: None,
            err=lambda _m: None,
        )
        assert rc == 2

    def test_empty_root_reports_nothing_to_remove(self, tmp_path: Path) -> None:
        out: list[str] = []
        rc = remove_module.run(None, tmp_path, assume_yes=True, echo=out.append)
        assert rc == 0
        assert any("othing to remove" in m for m in out)


class TestResolutionSafety:
    def test_bare_name_does_not_match_via_host_strip(self, tmp_path: Path) -> None:
        """`langgenius/dify` must NOT resolve to an unrelated bare `dify` dir
        (repo_name_from_url would strip `langgenius` as a host)."""
        dify = _make_repo(tmp_path, "dify")        # unrelated local checkout
        plan = remove_module.plan_remove(tmp_path, "langgenius/dify")
        assert plan.targets == []
        assert plan.error is not None
        assert dify.exists()                        # untouched

    def test_mixed_case_bare_name_resolves(self, tmp_path: Path) -> None:
        d = _make_repo(tmp_path, "langgenius__dify")
        plan = remove_module.plan_remove(tmp_path, "LangGenius/Dify")
        assert plan.targets == [d]

    def test_multi_level_name_and_url_resolve(self, tmp_path: Path) -> None:
        d = _make_repo(tmp_path, "group__subgroup__svc")
        assert remove_module.plan_remove(tmp_path, "group/subgroup/svc").targets == [d]
        assert remove_module.plan_remove(
            tmp_path, "https://gitlab.com/group/subgroup/svc.git"
        ).targets == [d]

    def test_crafted_traversal_target_does_not_escape(self, tmp_path: Path) -> None:
        _make_repo(tmp_path, "sutra")
        outside = tmp_path.parent / "SHOULD_NOT_DELETE"
        outside.mkdir(exist_ok=True)
        try:
            plan = remove_module.plan_remove(tmp_path, "../SHOULD_NOT_DELETE")
            assert plan.targets == []                # never resolves outside root
            assert outside.exists()
        finally:
            outside.rmdir()


class TestOrphans:
    def test_orphan_missing_graph_json_is_removable(self, tmp_path: Path) -> None:
        """A dir with embeddings but no graph.json (interrupted index) must
        still be reclaimable by name."""
        d = tmp_path / "frappe__frappe"
        d.mkdir()
        (d / "embeddings.npy").write_bytes(b"\x00")
        plan = remove_module.plan_remove(tmp_path, "frappe/frappe")
        assert plan.targets == [d]

    def test_orphan_included_in_remove_all(self, tmp_path: Path) -> None:
        good = _make_repo(tmp_path, "sutra")
        orphan = tmp_path / "half__done"
        orphan.mkdir()
        (orphan / "embeddings.npy").write_bytes(b"\x00")
        plan = remove_module.plan_remove(tmp_path, None)
        assert set(plan.targets) == {good, orphan}


class TestExecuteRobustness:
    def test_symlinked_child_is_unlinked_not_followed(self, tmp_path: Path) -> None:
        real = tmp_path / "real_target"
        real.mkdir()
        (real / "graph.json").write_text("{}")
        (real / "keep.txt").write_text("precious")
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()
        link = artifacts / "linked__repo"
        link.symlink_to(real, target_is_directory=True)

        failures = remove_module.execute_remove(
            remove_module.RemovePlan(targets=[link])
        )
        assert failures == []
        assert not link.exists()                    # the link is gone
        assert (real / "keep.txt").exists()          # its target is untouched

    def test_already_deleted_target_is_tolerated(self, tmp_path: Path) -> None:
        d = _make_repo(tmp_path, "sutra")
        plan = remove_module.plan_remove(tmp_path, "sutra")
        import shutil
        shutil.rmtree(d)                             # vanishes before execute
        assert remove_module.execute_remove(plan) == []   # no crash

    def test_remove_all_continues_past_symlink(self, tmp_path: Path) -> None:
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()
        a = _make_repo(artifacts, "aaa__x")
        b = _make_repo(artifacts, "bbb__y")
        real = tmp_path / "ext"
        real.mkdir()
        (real / "graph.json").write_text("{}")
        (artifacts / "zzz__link").symlink_to(real, target_is_directory=True)

        rc = remove_module.run(None, artifacts, assume_yes=True, echo=lambda _m: None)
        assert rc == 0
        assert not a.exists() and not b.exists()     # real repos removed
        assert (real / "graph.json").exists()        # symlink target survived


class TestCommand:
    def test_remove_command_registered(self) -> None:
        from sutra.cli.main import app

        names = {c.callback.__name__ for c in app.registered_commands}
        assert "remove" in names

    def test_command_delegates_and_deletes(self, tmp_path: Path) -> None:
        import typer

        from sutra.cli import main

        _make_repo(tmp_path, "sutra")
        with pytest.raises(typer.Exit) as exc:
            main.remove(target="sutra", artifacts_dir=tmp_path, yes=True)
        assert exc.value.exit_code == 0
        assert not (tmp_path / "sutra").exists()
