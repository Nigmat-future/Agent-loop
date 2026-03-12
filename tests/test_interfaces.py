from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient
from typer.testing import CliRunner

from agent_loop.api import create_app
from agent_loop.cli import app as cli_app
from agent_loop.models import SuccessCheck, SuccessCheckKind, TaskSpec, ToolPermissions
from agent_loop.settings import Settings


def test_api_and_cli_entrypoints(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    settings = Settings(
        home_dir=tmp_path / ".agent_loop",
        db_path=tmp_path / "api.db",
        default_workspace_root=workspace,
        default_shell_prefixes=["python", "echo"],
    )
    task = TaskSpec(
        objective="Write a file through the API.",
        context={"action_plan": [{"name": "file.write", "params": {"path": "api/task.txt", "content": "api-ok"}}], "final_output": "api complete"},
        success_checks=[SuccessCheck(kind=SuccessCheckKind.file_contains, target="api/task.txt", expected="api-ok")],
        permissions=ToolPermissions(workspace_root=str(workspace), enable_file=True, enable_shell=True, enable_http=False, allowed_shell_prefixes=["python", "echo"], allowed_http_domains=[]),
    )

    client = TestClient(create_app(settings=settings))
    response = client.post("/tasks/run", json=task.model_dump(mode="json"))
    assert response.status_code == 200
    payload = response.json()
    assert payload["outcome"]["success"] is True

    run_id = payload["trace"]["run_id"]
    inspect_response = client.get(f"/runs/{run_id}")
    assert inspect_response.status_code == 200
    assert inspect_response.json()["evaluation"]["mode"] == "rule"

    task_file = tmp_path / "task.json"
    task_file.write_text(task.model_dump_json(indent=2), encoding="utf-8")
    runner = CliRunner()
    cli_result = runner.invoke(cli_app, ["run-task", str(task_file), "--db-path", str(tmp_path / "cli.db")])
    assert cli_result.exit_code == 0, cli_result.stdout
    cli_payload = json.loads(cli_result.stdout)
    assert cli_payload["outcome"]["success"] is True
