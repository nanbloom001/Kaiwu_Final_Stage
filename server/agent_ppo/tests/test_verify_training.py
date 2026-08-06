from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock

from agent_ppo.tools import verify_training as verify


def make_server(directory: str) -> Path:
    root = Path(directory)
    (root / "agent_ppo/tests").mkdir(parents=True)
    (root / "agent_ppo/workflow").mkdir()
    (root / "conf").mkdir()
    (root / "isaac_env").mkdir()
    (root / "train_test.py").write_text("x = 1\n", encoding="utf-8")
    for name in ("test_p2_core.py", "test_nav_checkpoint.py", "test_nav_contract.py", "test_p3_contract.py"):
        (root / "agent_ppo/tests" / name).write_text("def test_ok(): pass\n", encoding="utf-8")
    return root


def git_result(stdout: str = "", returncode: int = 0, stderr: str = ""):
    return mock.Mock(stdout=stdout, returncode=returncode, stderr=stderr)


def test_profile_choices_and_targeted_selector():
    assert verify.PROFILE_NAMES == ("fast", "container", "release")
    with tempfile.TemporaryDirectory() as directory:
        root = make_server(directory)
        tests, fallback = verify.select_targeted_tests(root, ["agent_ppo/workflow/p2_nav_ppo_workflow.py"])
        assert not fallback
        assert "agent_ppo/tests/test_p2_core.py" in tests
        assert "agent_ppo/tests/test_nav_checkpoint.py" in tests


def test_changed_paths_requests_server_relative_git_diffs():
    with tempfile.TemporaryDirectory() as directory:
        root = make_server(directory)
        calls = []

        def fake_git(_root, args):
            calls.append(args)
            if args[0] == "diff":
                return git_result("agent_ppo/workflow/example.py\0")
            return git_result("")

        with mock.patch.object(verify, "_git", side_effect=fake_git):
            paths, skipped = verify.changed_paths(root)
        assert paths == ["agent_ppo/workflow/example.py"]
        assert not skipped
        diff_calls = [call for call in calls if call[0] == "diff"]
        assert all("--relative" in call for call in diff_calls)
        assert all(call[-2:] == ["--", "."] for call in diff_calls)


def test_unmapped_runtime_change_falls_back_to_agent_ppo_tests():
    with tempfile.TemporaryDirectory() as directory:
        root = make_server(directory)
        tests, fallback = verify.select_targeted_tests(root, ["agent_ppo/model/new_runtime.py"])
        assert fallback
        assert "agent_ppo/tests/test_p3_contract.py" in tests


def test_mixed_mapped_and_unmapped_runtime_changes_still_fall_back():
    with tempfile.TemporaryDirectory() as directory:
        root = make_server(directory)
        (root / "agent_ppo/tests/test_st9_opt3_d2.py").write_text(
            "raise ImportError('retired')\n", encoding="utf-8"
        )
        tests, fallback = verify.select_targeted_tests(
            root,
            [
                "agent_ppo/workflow/p2_nav_ppo_workflow.py",
                "agent_ppo/model/new_runtime.py",
            ],
        )
        assert fallback
        assert "agent_ppo/tests/test_p3_contract.py" in tests
        assert "agent_ppo/tests/test_st9_opt3_d2.py" not in tests


def test_release_paths_exclude_cache_and_include_server_roots():
    with tempfile.TemporaryDirectory() as directory:
        root = make_server(directory)
        (root / "agent_ppo/__pycache__").mkdir()
        (root / "agent_ppo/__pycache__/bad.py").write_text("bad", encoding="utf-8")
        (root / "conf/config.py").write_text("ok = 1", encoding="utf-8")
        (root / "isaac_env/base.py").write_text("ok = 1", encoding="utf-8")
        paths = verify.release_paths(root)
        assert "train_test.py" in paths
        assert "conf/config.py" in paths
        assert "isaac_env/base.py" in paths
        assert not any("__pycache__" in path for path in paths)


def test_release_profile_discovers_nested_root_tests():
    with tempfile.TemporaryDirectory() as directory:
        root = make_server(directory)
        nested = root / "tests/integration/test_nested.py"
        nested.parent.mkdir(parents=True)
        nested.write_text("def test_nested(): pass\n", encoding="utf-8")
        profile = verify.build_profile(root, "release", [])
        assert "tests/integration/test_nested.py" in profile["tests"]
        assert "tests/integration/test_nested.py" in profile["python"]


def test_release_quarantine_is_explicit_and_can_be_disabled():
    tests = [
        "agent_ppo/tests/test_st9_opt3_d2.py",
        "agent_ppo/tests/test_p2_core.py",
        "tests/test_visual_policy_optimization.py",
    ]
    selected, deselected, quarantined = verify.release_test_selection(
        tests, include_quarantined=False
    )
    assert "agent_ppo/tests/test_st9_opt3_d2.py" not in selected
    assert "agent_ppo/tests/test_p2_core.py" in selected
    assert any("test_agent_ppo_does_not_import_base_env" in node for node in deselected)
    assert "agent_ppo/tests/test_st9_opt3_d2.py" in quarantined

    selected_all, deselected_all, quarantined_all = verify.release_test_selection(
        tests, include_quarantined=True
    )
    assert selected_all == sorted(tests)
    assert not deselected_all
    assert not quarantined_all


def test_fingerprint_changes_when_content_changes():
    with tempfile.TemporaryDirectory() as directory:
        root = make_server(directory)
        target = root / "agent_ppo/workflow/example.py"
        target.write_text("value = 1\n", encoding="utf-8")
        with mock.patch.object(verify, "_git", return_value=git_result("head\n")):
            first = verify.source_fingerprint(root, ["agent_ppo/workflow/example.py"], "fast")
            target.write_text("value = 2\n", encoding="utf-8")
            second = verify.source_fingerprint(root, ["agent_ppo/workflow/example.py"], "fast")
        assert first != second


def test_fingerprint_changes_when_execution_contract_changes():
    with tempfile.TemporaryDirectory() as directory:
        root = make_server(directory)
        with mock.patch.object(verify, "_git", return_value=git_result("head\n")):
            without_smoke = verify.source_fingerprint(
                root, [], "container", {"nav_smoke": False}
            )
            with_smoke = verify.source_fingerprint(
                root, [], "container", {"nav_smoke": True, "num_envs": 8}
            )
        assert without_smoke != with_smoke


def test_fingerprint_changes_when_tracked_workspace_diff_changes():
    with tempfile.TemporaryDirectory() as directory:
        root = make_server(directory)
        responses = {
            ("rev-parse", "HEAD"): git_result("head\n"),
            ("rev-parse", "--show-prefix"): git_result(""),
            ("status", "--porcelain=v1", "-z", "--untracked-files=all", "--", "."):
                git_result(""),
        }

        def first_git(_root, args):
            if args == ["diff", "--binary", "HEAD"]:
                return git_result("first")
            return responses[tuple(args)]

        def second_git(_root, args):
            if args == ["diff", "--binary", "HEAD"]:
                return git_result("second")
            return responses[tuple(args)]

        with mock.patch.object(verify, "_git", side_effect=first_git):
            first = verify.source_fingerprint(root, [], "fast")
        with mock.patch.object(verify, "_git", side_effect=second_git):
            second = verify.source_fingerprint(root, [], "fast")
        assert first != second


def test_matching_evidence_reused_and_mismatch_rejected():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "evidence.json"
        path.write_text(json.dumps({"version": verify.VERSION, "profile": "fast", "fingerprint": "abc", "success": True}), encoding="utf-8")
        assert verify.reusable_evidence(path, "fast", "abc")
        assert not verify.reusable_evidence(path, "container", "abc")
        assert not verify.reusable_evidence(path, "fast", "changed")


def test_skipped_tests_cannot_produce_success_evidence():
    passing_steps = [{"returncode": 0}]
    assert verify.verification_success(passing_steps, tests_skipped=False)
    assert not verify.verification_success(passing_steps, tests_skipped=True)
    assert not verify.verification_success([{"returncode": 1}], tests_skipped=False)


def test_compile_sources_never_imports_or_writes_cache():
    with tempfile.TemporaryDirectory() as directory:
        root = make_server(directory)
        (root / "agent_ppo/workflow/check.py").write_text("x = 1\n", encoding="utf-8")
        success, output = verify.compile_sources(root, ["agent_ppo/workflow/check.py"])
        assert success and not output
        assert not list(root.rglob("*.pyc"))


def test_run_command_timeout_decodes_byte_output():
    timeout = subprocess.TimeoutExpired(
        ["slow-command"], 0.01, output=b"partial stdout", stderr=b"partial stderr"
    )
    with mock.patch.object(verify.subprocess, "run", side_effect=timeout):
        result = verify.run_command(["slow-command"], Path.cwd(), timeout=0.01)
    assert result["returncode"] == 124
    assert result["stdout"] == "partial stdout"
    assert "partial stderr" in result["stderr"]
    assert "command timed out" in result["stderr"]


def test_nav_smoke_requires_target_reached_and_always_stops():
    with tempfile.TemporaryDirectory() as directory:
        root = make_server(directory)
        calls = []
        def runner(command, _root, timeout=None):
            calls.append(command)
            if "status" in command:
                return {"command": command, "duration_s": 0, "returncode": 0, "stdout": '{"target_reached": true}', "stderr": ""}
            return {"command": command, "duration_s": 0, "returncode": 0, "stdout": "", "stderr": ""}
        steps = verify.nav_smoke(root, 2, "/tmp/smoke", 1, runner)
        assert all(step["returncode"] == 0 for step in steps)
        assert "agent_ppo.tools.nav_full_smoke" in calls[-1]
        assert "stop" in calls[-1]
        assert calls[-1][-2:] == ["--grace-seconds", "5"]


def test_nav_smoke_start_failure_stops_without_polling():
    with tempfile.TemporaryDirectory() as directory:
        root = make_server(directory)
        calls = []

        def runner(command, _root, timeout=None):
            calls.append(command)
            return {
                "command": command,
                "duration_s": 0,
                "returncode": 2 if "start" in command else 0,
                "stdout": "",
                "stderr": "failed",
            }

        steps = verify.nav_smoke(root, 2, "/tmp/smoke", 1, runner)
        assert [step["name"] for step in steps] == [
            "nav-smoke-start",
            "nav-smoke-target",
            "nav-smoke-stop",
        ]
        assert not any("status" in command for command in calls)


def test_nav_smoke_start_timeout_still_stops():
    with tempfile.TemporaryDirectory() as directory:
        root = make_server(directory)
        calls = []

        def runner(command, _root, timeout=None):
            calls.append(command)
            return {
                "command": command,
                "duration_s": 0,
                "returncode": 124 if "start" in command else 0,
                "stdout": "",
                "stderr": "command timed out",
            }

        steps = verify.nav_smoke(root, 2, "/tmp/smoke", 1, runner)
        assert steps[-1]["name"] == "nav-smoke-stop"
        assert "stop" in calls[-1]


def test_nav_smoke_failure_is_recorded_when_status_never_reaches_target():
    with tempfile.TemporaryDirectory() as directory:
        root = make_server(directory)
        def runner(command, _root, timeout=None):
            output = '{"target_reached": false}' if "status" in command else ""
            return {"command": command, "duration_s": 0, "returncode": 0, "stdout": output, "stderr": ""}
        with mock.patch.object(verify.time, "sleep"):
            steps = verify.nav_smoke(root, 2, "/tmp/smoke", 0.001, runner)
        assert any(step["name"] == "nav-smoke-target" and step["returncode"] == 1 for step in steps)


def test_atomic_evidence_write_leaves_only_destination():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "evidence" / "fast.json"
        verify.atomic_write_evidence(path, {"success": True})
        assert json.loads(path.read_text(encoding="utf-8")) == {"success": True}
        assert not list(path.parent.glob("fast.json.*"))


def test_evidence_destination_rejects_source_and_existing_external_files():
    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory)
        root = make_server(str(base / "server"))
        allowed = root / ".verification/fast.json"
        allowed.parent.mkdir()
        allowed.write_text("{}\n", encoding="utf-8")
        assert verify.evidence_destination(root, str(allowed)) == allowed.resolve()
        try:
            verify.evidence_destination(root, "train_test.py")
        except ValueError as exc:
            assert "must use a .json suffix" in str(exc)
        else:
            raise AssertionError("source evidence destination was accepted")
        tracked_like = root / "result.json"
        tracked_like.write_text("{}\n", encoding="utf-8")
        try:
            verify.evidence_destination(root, str(tracked_like))
        except ValueError as exc:
            assert ".verification" in str(exc)
        else:
            raise AssertionError("server-root evidence destination was accepted")
        external = base / "existing.json"
        external.write_text("{}\n", encoding="utf-8")
        try:
            verify.evidence_destination(root, str(external))
        except ValueError as exc:
            assert "refusing to overwrite" in str(exc)
        else:
            raise AssertionError("existing external evidence destination was accepted")
        fresh_external = base / "fresh.json"
        assert verify.evidence_destination(root, str(fresh_external)) == fresh_external.resolve()


def test_evidence_destination_rejects_sibling_repository_paths():
    with tempfile.TemporaryDirectory() as directory:
        repository = Path(directory) / "repository"
        root = make_server(str(repository / "server"))
        sibling = repository / "deploy/fresh.json"
        with mock.patch.object(
            verify,
            "_git",
            return_value=git_result(str(repository.resolve()) + "\n"),
        ):
            try:
                verify.evidence_destination(root, str(sibling))
            except ValueError as exc:
                assert "Git repository" in str(exc)
            else:
                raise AssertionError("sibling repository evidence path was accepted")


def test_execute_fails_if_workspace_changes_during_verification():
    with tempfile.TemporaryDirectory() as directory:
        root = make_server(directory)
        evidence_path = root / ".verification/evidence.json"
        args = verify.parser().parse_args(
            ["--profile", "fast", "--evidence", str(evidence_path)]
        )
        captured = {}

        def capture_evidence(_path, evidence):
            captured.update(evidence)

        passing_step = {
            "command": ["git"],
            "duration_s": 0,
            "returncode": 0,
            "stdout": "",
            "stderr": "",
        }
        with (
            mock.patch.object(
                verify,
                "changed_paths",
                side_effect=[([], []), (["agent_ppo/model/late.py"], [])],
            ),
            mock.patch.object(
                verify, "source_fingerprint", side_effect=["before", "after"]
            ),
            mock.patch.object(verify, "_git", return_value=git_result("head\n")),
            mock.patch.object(verify, "_step_command", return_value=passing_step),
            mock.patch.object(verify, "atomic_write_evidence", side_effect=capture_evidence),
        ):
            assert verify.execute(args, root) == 1
        assert not captured["success"]
        assert captured["fingerprint"] == "before"
        assert captured["fingerprint_after"] == "after"
        assert any(
            step.get("name") == "workspace-stability" for step in captured["steps"]
        )


def test_documented_dash_b_entrypoint_does_not_create_bytecode_cache():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        package = root / "agent_ppo/tools"
        package.mkdir(parents=True)
        (root / "agent_ppo/__init__.py").write_text("", encoding="utf-8")
        (package / "__init__.py").write_text("", encoding="utf-8")
        (package / "verify_training.py").write_text(
            Path(verify.__file__).read_text(encoding="utf-8"), encoding="utf-8"
        )
        completed = subprocess.run(
            [sys.executable, "-B", "-m", "agent_ppo.tools.verify_training", "--help"],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0
        assert not list(root.rglob("__pycache__"))
        assert not list(root.rglob("*.pyc"))
