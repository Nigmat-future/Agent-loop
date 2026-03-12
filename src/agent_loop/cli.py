from __future__ import annotations

import json
from pathlib import Path

import typer

from .service import AgentLoopService
from .settings import Settings
from .models import StrategyStatus, TaskSpec

app = typer.Typer(help="Agent Loop CLI")


def _service(db_path: Path | None = None) -> AgentLoopService:
    settings = Settings.from_env()
    if db_path is not None:
        settings.db_path = db_path
    return AgentLoopService(settings=settings)


def _print(value: object) -> None:
    typer.echo(json.dumps(value, ensure_ascii=False, indent=2, default=str))


@app.command("run-task")
def run_task(
    task_file: Path,
    db_path: Path | None = None,
    auto_learn: bool = True,
    auto_promote: bool = True,
    replay_limit: int = 3,
) -> None:
    service = _service(db_path)
    task = TaskSpec.model_validate_json(task_file.read_text(encoding="utf-8"))
    result = service.run_task(
        task,
        auto_learn=auto_learn,
        auto_promote=auto_promote,
        replay_limit=replay_limit,
    )
    _print(result.model_dump(mode="json"))


@app.command("inspect-run")
def inspect_run(run_id: str, db_path: Path | None = None) -> None:
    service = _service(db_path)
    _print(service.inspect_run(run_id).model_dump(mode="json"))


@app.command("learn-from-run")
def learn_from_run(run_id: str, db_path: Path | None = None) -> None:
    service = _service(db_path)
    _print(service.learn_from_run(run_id).model_dump(mode="json"))


@app.command("replay-strategy")
def replay_strategy(strategy_id: str, limit: int = 5, db_path: Path | None = None) -> None:
    service = _service(db_path)
    _print(service.replay_strategy(strategy_id, limit=limit).model_dump(mode="json"))


@app.command("promote-strategy")
def promote_strategy(strategy_id: str, limit: int = 5, db_path: Path | None = None) -> None:
    service = _service(db_path)
    _print(service.promote_strategy(strategy_id, limit=limit).model_dump(mode="json"))


@app.command("list-memories")
def list_memories(memory_type: str | None = None, limit: int = 50, db_path: Path | None = None) -> None:
    service = _service(db_path)
    _print([memory.model_dump(mode="json") for memory in service.list_memories(memory_type=memory_type, limit=limit)])


@app.command("list-strategies")
def list_strategies(task_type: str | None = None, status: str | None = None, db_path: Path | None = None) -> None:
    service = _service(db_path)
    parsed_status = StrategyStatus(status) if status else None
    _print([strategy.model_dump(mode="json") for strategy in service.list_strategies(task_type=task_type, status=parsed_status)])


@app.command("serve")
def serve(host: str = "127.0.0.1", port: int = 8000, db_path: Path | None = None) -> None:
    import uvicorn

    settings = Settings.from_env()
    if db_path is not None:
        settings.db_path = db_path
    uvicorn.run("agent_loop.api:create_app", factory=True, host=host, port=port)

