from __future__ import annotations

from .models import SuccessCheck, SuccessCheckKind, TaskSpec, ToolPermissions


def build_regression_suite(workspace_root: str, base_url: str) -> list[TaskSpec]:
    permissions = ToolPermissions(
        workspace_root=workspace_root,
        enable_file=True,
        enable_shell=True,
        enable_http=True,
        allowed_shell_prefixes=["python", "echo"],
        allowed_http_domains=["127.0.0.1"],
    )
    return [
        TaskSpec(
            task_type="general",
            objective="Write a file inside the workspace.",
            context={"action_plan": [{"name": "file.write", "params": {"path": "artifacts/file-check.txt", "content": "file-ok"}}], "final_output": "file task complete"},
            success_checks=[
                SuccessCheck(kind=SuccessCheckKind.file_exists, target="artifacts/file-check.txt"),
                SuccessCheck(kind=SuccessCheckKind.file_contains, target="artifacts/file-check.txt", expected="file-ok"),
            ],
            permissions=permissions,
        ),
        TaskSpec(
            task_type="general",
            objective="Use shell to create a file.",
            context={"action_plan": [{"name": "shell.run", "params": {"command": "python -c \"from pathlib import Path; Path('artifacts/shell-check.txt').write_text('shell-ok', encoding='utf-8')\""}}], "final_output": "shell task complete"},
            success_checks=[SuccessCheck(kind=SuccessCheckKind.file_contains, target="artifacts/shell-check.txt", expected="shell-ok")],
            permissions=permissions,
        ),
        TaskSpec(
            task_type="general",
            objective="Call a local HTTP endpoint.",
            context={"action_plan": [{"name": "http.get", "params": {"url": f"{base_url}/health"}}], "final_output": "http task complete"},
            success_checks=[SuccessCheck(kind=SuccessCheckKind.http_status, target=f"{base_url}/health", expected=200)],
            permissions=permissions,
        ),
    ]
