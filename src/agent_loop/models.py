from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(UTC)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


class RunStatus(str, Enum):
    pending = "pending"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    cancelled = "cancelled"


class StrategyStatus(str, Enum):
    active = "active"
    candidate = "candidate"
    archived = "archived"


class MemoryType(str, Enum):
    episodic = "episodic"
    semantic = "semantic"
    procedural = "procedural"


class EventType(str, Enum):
    plan = "plan"
    memory_retrieval = "memory_retrieval"
    tool_call = "tool_call"
    tool_result = "tool_result"
    retry = "retry"
    error = "error"
    reflection = "reflection"
    finish = "finish"
    evaluation = "evaluation"
    learning = "learning"


class SuccessCheckKind(str, Enum):
    file_exists = "file_exists"
    file_contains = "file_contains"
    final_output_contains = "final_output_contains"
    http_status = "http_status"
    no_error_events = "no_error_events"


class Budget(BaseModel):
    max_steps: int = 8
    max_shell_calls: int = 2
    max_http_calls: int = 2
    timeout_seconds: int = 30
    max_errors: int = 3


class ToolPermissions(BaseModel):
    workspace_root: str | None = None
    enable_file: bool = True
    enable_shell: bool = True
    enable_http: bool = False
    allowed_shell_prefixes: list[str] = Field(
        default_factory=lambda: ["python", "echo", "dir", "type"]
    )
    allowed_http_domains: list[str] = Field(default_factory=list)


class SuccessCheck(BaseModel):
    kind: SuccessCheckKind
    target: str | None = None
    expected: str | int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskSpec(BaseModel):
    id: str = Field(default_factory=lambda: new_id("task"))
    task_type: str = "general"
    objective: str
    context: dict[str, Any] = Field(default_factory=dict)
    success_checks: list[SuccessCheck] = Field(default_factory=list)
    budget: Budget = Field(default_factory=Budget)
    permissions: ToolPermissions = Field(default_factory=ToolPermissions)
    created_at: datetime = Field(default_factory=utc_now)


class TraceEvent(BaseModel):
    timestamp: datetime = Field(default_factory=utc_now)
    type: EventType
    payload: dict[str, Any] = Field(default_factory=dict)


class ToolAction(BaseModel):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""


class TaskOutcome(BaseModel):
    status: RunStatus = RunStatus.pending
    success: bool = False
    result_summary: str = ""
    artifact_paths: list[str] = Field(default_factory=list)
    final_output: str = ""
    error: str | None = None


class RunTrace(BaseModel):
    run_id: str
    task_id: str
    strategy_id: str
    events: list[TraceEvent] = Field(default_factory=list)
    total_steps: int = 0
    shell_calls: int = 0
    http_calls: int = 0
    token_usage: int = 0
    estimated_cost: float = 0.0
    started_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = None
    final_output: str = ""


class EvalReport(BaseModel):
    id: str = Field(default_factory=lambda: new_id("eval"))
    run_id: str
    mode: str
    success: bool
    confidence: float
    summary: str
    failure_reason: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class LessonCandidate(BaseModel):
    id: str = Field(default_factory=lambda: new_id("lesson"))
    run_id: str
    lesson_type: str = "heuristic"
    title: str
    description: str
    applicability: str
    confidence: float
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class StrategyBundle(BaseModel):
    id: str = Field(default_factory=lambda: new_id("strategy"))
    version: int = 1
    task_type: str = "general"
    status: StrategyStatus = StrategyStatus.candidate
    source: str = "manual"
    parent_strategy_id: str | None = None
    system_prompt: str = "Act cautiously, prefer deterministic tools, and respect guardrails."
    planning_template: str = "Plan minimal steps, validate constraints, and avoid repeated failures."
    tool_weights: dict[str, float] = Field(
        default_factory=lambda: {"file": 1.0, "shell": 1.0, "http": 0.7}
    )
    retrieval_threshold: float = 0.2
    retry_budget: int = 1
    decomposition_rules: list[str] = Field(
        default_factory=lambda: [
            "Prefer structured actions from task context when available.",
            "Use shell only for commands inside the allowlist.",
            "Stop when success checks are satisfied or budget is exhausted.",
        ]
    )
    metrics: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    activated_at: datetime | None = None


class PromotionDecision(BaseModel):
    id: str = Field(default_factory=lambda: new_id("promotion"))
    candidate_strategy_id: str
    baseline_strategy_id: str
    approved: bool
    summary: str
    metrics: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class MemoryRecord(BaseModel):
    id: str = Field(default_factory=lambda: new_id("memory"))
    memory_type: MemoryType
    run_id: str | None = None
    task_id: str | None = None
    summary: str
    content: str
    embedding: list[float] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class RunInspection(BaseModel):
    task: TaskSpec
    trace: RunTrace
    outcome: TaskOutcome
    evaluation: EvalReport | None = None
    lessons: list[LessonCandidate] = Field(default_factory=list)


class ReplayReport(BaseModel):
    candidate_strategy_id: str
    baseline_strategy_id: str
    tasks_evaluated: int
    candidate_success_rate: float
    baseline_success_rate: float
    candidate_avg_steps: float
    baseline_avg_steps: float
    candidate_avg_cost: float
    baseline_avg_cost: float
    candidate_error_rate: float
    baseline_error_rate: float
    approved: bool
    summary: str
    task_ids: list[str] = Field(default_factory=list)


class LearningResult(BaseModel):
    run_id: str
    memories: list[MemoryRecord] = Field(default_factory=list)
    lessons: list[LessonCandidate] = Field(default_factory=list)
    candidate_strategy: StrategyBundle | None = None
    replay_report: ReplayReport | None = None
    promotion_decision: PromotionDecision | None = None


class RunResult(BaseModel):
    task: TaskSpec
    trace: RunTrace
    outcome: TaskOutcome
    evaluation: EvalReport
    learning: LearningResult | None = None

