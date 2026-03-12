from __future__ import annotations

from fastapi import FastAPI, HTTPException

from .models import LearningResult, MemoryRecord, PromotionDecision, ReplayReport, RunInspection, RunResult, StrategyBundle, TaskSpec
from .service import AgentLoopService
from .settings import Settings


def create_app(settings: Settings | None = None) -> FastAPI:
    service = AgentLoopService(settings=settings)
    app = FastAPI(title="Agent Loop", version="0.1.0")

    @app.post("/tasks/run", response_model=RunResult)
    def run_task(
        task: TaskSpec,
        auto_learn: bool = True,
        auto_promote: bool = True,
        replay_limit: int = 3,
    ) -> RunResult:
        return service.run_task(
            task,
            auto_learn=auto_learn,
            auto_promote=auto_promote,
            replay_limit=replay_limit,
        )

    @app.get("/runs/{run_id}", response_model=RunInspection)
    def inspect_run(run_id: str) -> RunInspection:
        try:
            return service.inspect_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/runs/{run_id}/learn", response_model=LearningResult)
    def learn_from_run(run_id: str) -> LearningResult:
        try:
            return service.learn_from_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/strategies/{strategy_id}/replay", response_model=ReplayReport)
    def replay_strategy(strategy_id: str, limit: int = 5) -> ReplayReport:
        try:
            return service.replay_strategy(strategy_id, limit=limit)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/strategies/{strategy_id}/promote", response_model=PromotionDecision)
    def promote_strategy(strategy_id: str, limit: int = 5) -> PromotionDecision:
        try:
            return service.promote_strategy(strategy_id, limit=limit)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/memories", response_model=list[MemoryRecord])
    def list_memories(memory_type: str | None = None, limit: int = 50) -> list[MemoryRecord]:
        return service.list_memories(memory_type=memory_type, limit=limit)

    @app.get("/strategies", response_model=list[StrategyBundle])
    def list_strategies(task_type: str | None = None, status: str | None = None) -> list[StrategyBundle]:
        parsed_status = None
        if status:
            from .models import StrategyStatus

            parsed_status = StrategyStatus(status)
        return service.list_strategies(task_type=task_type, status=parsed_status)

    return app


app = create_app()

