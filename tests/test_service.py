from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from agent_loop.benchmarks import build_regression_suite
from agent_loop.models import SuccessCheck, SuccessCheckKind, TaskSpec, ToolPermissions
from agent_loop.service import AgentLoopService
from agent_loop.settings import Settings


class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            body = b'{"ok": true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return


@pytest.fixture()
def local_http_server() -> str:
    server = HTTPServer(("127.0.0.1", 0), _HealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


@pytest.fixture()
def service_env(tmp_path: Path) -> tuple[AgentLoopService, Path]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    settings = Settings(
        home_dir=tmp_path / ".agent_loop",
        db_path=tmp_path / "agent.db",
        default_workspace_root=workspace,
        default_http_domains=["127.0.0.1"],
        default_shell_prefixes=["python", "echo"],
    )
    return AgentLoopService(settings=settings), workspace


def _permissions(workspace: Path, enable_http: bool = False) -> ToolPermissions:
    return ToolPermissions(
        workspace_root=str(workspace),
        enable_file=True,
        enable_shell=True,
        enable_http=enable_http,
        allowed_shell_prefixes=["python", "echo"],
        allowed_http_domains=["127.0.0.1"],
    )


def test_run_task_end_to_end_and_memory_retrieval(service_env: tuple[AgentLoopService, Path]) -> None:
    service, workspace = service_env
    task = TaskSpec(
        objective="Write a sales summary file.",
        context={
            "action_plan": [{"name": "file.write", "params": {"path": "reports/summary.txt", "content": "sales improved"}}],
            "final_output": "summary created",
        },
        success_checks=[
            SuccessCheck(kind=SuccessCheckKind.file_exists, target="reports/summary.txt"),
            SuccessCheck(kind=SuccessCheckKind.file_contains, target="reports/summary.txt", expected="sales improved"),
        ],
        permissions=_permissions(workspace),
    )

    result = service.run_task(task)

    assert result.outcome.success is True
    assert result.evaluation.mode == "rule"
    assert (workspace / "reports" / "summary.txt").read_text(encoding="utf-8") == "sales improved"
    inspection = service.inspect_run(result.trace.run_id)
    assert any(event.type == "tool_call" for event in inspection.trace.events)
    memories = service.store.search_memories("sales summary file", limit=3, threshold=0.0, task_type="general")
    assert memories
    assert any(memory.memory_type.value == "episodic" for memory in memories)


def test_failure_generates_lessons_and_candidate(service_env: tuple[AgentLoopService, Path]) -> None:
    service, workspace = service_env
    task = TaskSpec(
        objective="Choose a safe way to create the output file.",
        context={
            "tool_options": [
                {"name": "shell.run", "params": {"command": "powershell -Command Write-Output blocked"}},
                {"name": "file.write", "params": {"path": "safe/result.txt", "content": "safe-path"}},
            ],
            "final_output": "attempted safe creation",
        },
        success_checks=[SuccessCheck(kind=SuccessCheckKind.file_contains, target="safe/result.txt", expected="safe-path")],
        permissions=_permissions(workspace),
    )

    result = service.run_task(task)

    assert result.outcome.success is False
    assert result.learning is not None
    assert result.learning.lessons
    assert result.learning.candidate_strategy is not None
    assert result.learning.candidate_strategy.tool_weights["shell"] <= 0.1
    assert any(memory.memory_type.value == "semantic" for memory in result.learning.memories)


def test_replay_and_promote_candidate(service_env: tuple[AgentLoopService, Path]) -> None:
    service, workspace = service_env
    historical_task = TaskSpec(
        objective="Prefer the deterministic tool option.",
        context={
            "tool_options": [
                {"name": "shell.run", "params": {"command": "powershell -Command Write-Output blocked"}},
                {"name": "file.write", "params": {"path": "promotion/history.txt", "content": "approved"}},
            ],
            "final_output": "promotion history complete",
        },
        success_checks=[SuccessCheck(kind=SuccessCheckKind.file_contains, target="promotion/history.txt", expected="approved")],
        permissions=_permissions(workspace),
    )
    service.run_task(historical_task)

    task = TaskSpec(
        objective="Prefer the deterministic tool option.",
        context={
            "tool_options": [
                {"name": "shell.run", "params": {"command": "powershell -Command Write-Output blocked"}},
                {"name": "file.write", "params": {"path": "promotion/result.txt", "content": "approved"}},
            ],
            "final_output": "promotion trial complete",
        },
        success_checks=[SuccessCheck(kind=SuccessCheckKind.file_contains, target="promotion/result.txt", expected="approved")],
        permissions=_permissions(workspace),
    )

    result = service.run_task(task)
    candidate = result.learning.candidate_strategy
    assert candidate is not None

    replay = service.replay_strategy(candidate.id, limit=1)
    assert replay.approved is True
    assert replay.candidate_success_rate == 1.0
    assert replay.baseline_success_rate == 0.0
    assert replay.task_ids == [historical_task.id]

    decision = service.promote_strategy(candidate.id, limit=1)
    assert decision.approved is True
    active = service.registry.ensure_active("general")
    assert active.id == candidate.id


def test_regression_suite_covers_file_shell_http(service_env: tuple[AgentLoopService, Path], local_http_server: str) -> None:
    service, workspace = service_env
    suite = build_regression_suite(str(workspace), local_http_server)

    results = [service.run_task(task) for task in suite]

    assert len(results) == 3
    assert all(result.outcome.success for result in results)
    assert any(event.payload.get("metadata", {}).get("status_code") == 200 for event in results[-1].trace.events if event.type == "tool_result")



def test_replay_requires_historical_tasks(service_env: tuple[AgentLoopService, Path]) -> None:
    service, workspace = service_env
    task = TaskSpec(
        objective="Replay should stay candidate without prior history.",
        context={
            "tool_options": [
                {"name": "shell.run", "params": {"command": "powershell -Command Write-Output blocked"}},
                {"name": "file.write", "params": {"path": "replay_guard/result.txt", "content": "approved"}},
            ],
            "final_output": "replay guard complete",
        },
        success_checks=[SuccessCheck(kind=SuccessCheckKind.file_contains, target="replay_guard/result.txt", expected="approved")],
        permissions=_permissions(workspace),
    )

    result = service.run_task(task, auto_promote=True, replay_limit=1)

    assert result.learning is not None
    assert result.learning.candidate_strategy is not None
    assert result.learning.replay_report is not None
    assert result.learning.replay_report.approved is False
    assert result.learning.replay_report.tasks_evaluated == 0
    assert result.learning.promotion_decision is not None
    assert result.learning.promotion_decision.approved is False
    active = service.registry.ensure_active("general")
    assert active.id != result.learning.candidate_strategy.id
def test_auto_replay_and_promote_from_run_task(service_env: tuple[AgentLoopService, Path]) -> None:
    service, workspace = service_env
    historical_task = TaskSpec(
        objective="Auto-promote the safer strategy after task completion.",
        context={
            "tool_options": [
                {"name": "shell.run", "params": {"command": "powershell -Command Write-Output blocked"}},
                {"name": "file.write", "params": {"path": "autopromote/history.txt", "content": "approved"}},
            ],
            "final_output": "autopromote history complete",
        },
        success_checks=[SuccessCheck(kind=SuccessCheckKind.file_contains, target="autopromote/history.txt", expected="approved")],
        permissions=_permissions(workspace),
    )
    service.run_task(historical_task)

    task = TaskSpec(
        objective="Auto-promote the safer strategy after task completion.",
        context={
            "tool_options": [
                {"name": "shell.run", "params": {"command": "powershell -Command Write-Output blocked"}},
                {"name": "file.write", "params": {"path": "autopromote/result.txt", "content": "approved"}},
            ],
            "final_output": "autopromote complete",
        },
        success_checks=[SuccessCheck(kind=SuccessCheckKind.file_contains, target="autopromote/result.txt", expected="approved")],
        permissions=_permissions(workspace),
    )

    result = service.run_task(task, auto_promote=True, replay_limit=1)

    assert result.learning is not None
    assert result.learning.candidate_strategy is not None
    assert result.learning.replay_report is not None
    assert result.learning.replay_report.approved is True
    assert result.learning.replay_report.task_ids == [historical_task.id]
    assert result.learning.promotion_decision is not None
    assert result.learning.promotion_decision.approved is True
    active = service.registry.ensure_active("general")
    assert active.id == result.learning.candidate_strategy.id



