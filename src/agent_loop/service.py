from __future__ import annotations

import json
import shlex
import time
import sqlite3
import subprocess
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .models import (
    EvalReport,
    EventType,
    LearningResult,
    LessonCandidate,
    MemoryRecord,
    MemoryType,
    PromotionDecision,
    ReplayReport,
    RunInspection,
    RunResult,
    RunStatus,
    RunTrace,
    StrategyBundle,
    StrategyStatus,
    SuccessCheckKind,
    TaskOutcome,
    TaskSpec,
    ToolAction,
    TraceEvent,
    new_id,
    utc_now,
)
from .settings import Settings
from .vector import cosine_similarity, hash_embedding, pack_embedding, tokenize, unpack_embedding


class GuardrailError(RuntimeError):
    pass


class ToolExecutionError(RuntimeError):
    pass


class ProviderError(RuntimeError):
    pass


@dataclass
class ToolResult:
    success: bool
    output: str
    artifact_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class SQLiteStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _init_db(self) -> None:
        schema = """
        CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY,
            task_type TEXT NOT NULL,
            objective TEXT NOT NULL,
            task_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS runs (
            id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL,
            strategy_id TEXT NOT NULL,
            status TEXT NOT NULL,
            success INTEGER NOT NULL DEFAULT 0,
            started_at TEXT NOT NULL,
            completed_at TEXT,
            total_steps INTEGER NOT NULL DEFAULT 0,
            shell_calls INTEGER NOT NULL DEFAULT 0,
            http_calls INTEGER NOT NULL DEFAULT 0,
            token_usage INTEGER NOT NULL DEFAULT 0,
            estimated_cost REAL NOT NULL DEFAULT 0,
            final_output TEXT,
            error TEXT,
            task_snapshot_json TEXT NOT NULL,
            outcome_json TEXT,
            FOREIGN KEY(task_id) REFERENCES tasks(id)
        );
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            seq INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            type TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            FOREIGN KEY(run_id) REFERENCES runs(id)
        );
        CREATE TABLE IF NOT EXISTS evaluations (
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL UNIQUE,
            report_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(run_id) REFERENCES runs(id)
        );
        CREATE TABLE IF NOT EXISTS lessons (
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            lesson_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(run_id) REFERENCES runs(id)
        );
        CREATE TABLE IF NOT EXISTS strategies (
            id TEXT PRIMARY KEY,
            task_type TEXT NOT NULL,
            version INTEGER NOT NULL,
            status TEXT NOT NULL,
            source TEXT NOT NULL,
            parent_strategy_id TEXT,
            bundle_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            activated_at TEXT
        );
        CREATE TABLE IF NOT EXISTS promotion_decisions (
            id TEXT PRIMARY KEY,
            candidate_strategy_id TEXT NOT NULL,
            baseline_strategy_id TEXT NOT NULL,
            approved INTEGER NOT NULL,
            decision_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            memory_type TEXT NOT NULL,
            run_id TEXT,
            task_id TEXT,
            summary TEXT NOT NULL,
            content TEXT NOT NULL,
            embedding BLOB,
            metadata_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(memory_id UNINDEXED, summary, content);
        CREATE INDEX IF NOT EXISTS idx_runs_task_id ON runs(task_id);
        CREATE INDEX IF NOT EXISTS idx_events_run_id_seq ON events(run_id, seq);
        CREATE INDEX IF NOT EXISTS idx_strategies_task_type_status ON strategies(task_type, status);
        CREATE INDEX IF NOT EXISTS idx_memories_type_run ON memories(memory_type, run_id);
        """
        with self._connect() as connection:
            connection.executescript(schema)

    @staticmethod
    def _dump_json(value: Any) -> str:
        if hasattr(value, "model_dump"):
            value = value.model_dump(mode="json")
        return json.dumps(value, ensure_ascii=False)

    def save_task(self, task: TaskSpec) -> TaskSpec:
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO tasks(id, task_type, objective, task_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (task.id, task.task_type, task.objective, self._dump_json(task), task.created_at.isoformat()),
            )
        return task

    def get_task(self, task_id: str) -> TaskSpec:
        with self._connect() as connection:
            row = connection.execute("SELECT task_json FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown task id: {task_id}")
        return TaskSpec.model_validate_json(row["task_json"])

    def list_tasks(self, task_type: str | None = None, limit: int = 20) -> list[TaskSpec]:
        sql = "SELECT task_json FROM tasks"
        params: list[Any] = []
        if task_type:
            sql += " WHERE task_type = ?"
            params.append(task_type)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [TaskSpec.model_validate_json(row["task_json"]) for row in rows]

    def create_run(self, run_id: str, task: TaskSpec, strategy: StrategyBundle) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO runs(id, task_id, strategy_id, status, success, started_at, task_snapshot_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (run_id, task.id, strategy.id, RunStatus.running.value, 0, utc_now().isoformat(), self._dump_json(task)),
            )

    def append_event(self, run_id: str, seq: int, event: TraceEvent) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO events(run_id, seq, timestamp, type, payload_json) VALUES (?, ?, ?, ?, ?)",
                (run_id, seq, event.timestamp.isoformat(), event.type.value, self._dump_json(event.payload)),
            )

    def finalize_run(self, trace: RunTrace, outcome: TaskOutcome) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE runs SET status = ?, success = ?, completed_at = ?, total_steps = ?, shell_calls = ?, http_calls = ?, token_usage = ?, estimated_cost = ?, final_output = ?, error = ?, outcome_json = ? WHERE id = ?",
                (
                    outcome.status.value,
                    1 if outcome.success else 0,
                    trace.completed_at.isoformat() if trace.completed_at else None,
                    trace.total_steps,
                    trace.shell_calls,
                    trace.http_calls,
                    trace.token_usage,
                    trace.estimated_cost,
                    outcome.final_output,
                    outcome.error,
                    self._dump_json(outcome),
                    trace.run_id,
                ),
            )

    def get_run_trace(self, run_id: str) -> RunTrace:
        with self._connect() as connection:
            run = connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            rows = connection.execute("SELECT * FROM events WHERE run_id = ? ORDER BY seq ASC", (run_id,)).fetchall()
        if run is None:
            raise KeyError(f"Unknown run id: {run_id}")
        return RunTrace(
            run_id=run["id"],
            task_id=run["task_id"],
            strategy_id=run["strategy_id"],
            events=[TraceEvent(timestamp=row["timestamp"], type=row["type"], payload=json.loads(row["payload_json"])) for row in rows],
            total_steps=run["total_steps"],
            shell_calls=run["shell_calls"],
            http_calls=run["http_calls"],
            token_usage=run["token_usage"],
            estimated_cost=run["estimated_cost"],
            started_at=run["started_at"],
            completed_at=run["completed_at"],
            final_output=run["final_output"] or "",
        )

    def get_outcome(self, run_id: str) -> TaskOutcome:
        with self._connect() as connection:
            row = connection.execute("SELECT outcome_json FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None or row["outcome_json"] is None:
            raise KeyError(f"Missing outcome for run id: {run_id}")
        return TaskOutcome.model_validate_json(row["outcome_json"])

    def save_evaluation(self, report: EvalReport) -> EvalReport:
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO evaluations(id, run_id, report_json, created_at) VALUES (?, ?, ?, ?)",
                (report.id, report.run_id, self._dump_json(report), report.created_at.isoformat()),
            )
        return report

    def get_evaluation(self, run_id: str) -> EvalReport | None:
        with self._connect() as connection:
            row = connection.execute("SELECT report_json FROM evaluations WHERE run_id = ?", (run_id,)).fetchone()
        return EvalReport.model_validate_json(row["report_json"]) if row else None
    def save_lessons(self, lessons: list[LessonCandidate]) -> list[LessonCandidate]:
        if not lessons:
            return []
        with self._connect() as connection:
            for lesson in lessons:
                connection.execute(
                    "INSERT OR REPLACE INTO lessons(id, run_id, lesson_json, created_at) VALUES (?, ?, ?, ?)",
                    (lesson.id, lesson.run_id, self._dump_json(lesson), lesson.created_at.isoformat()),
                )
        return lessons

    def list_lessons(self, run_id: str) -> list[LessonCandidate]:
        with self._connect() as connection:
            rows = connection.execute("SELECT lesson_json FROM lessons WHERE run_id = ? ORDER BY created_at ASC", (run_id,)).fetchall()
        return [LessonCandidate.model_validate_json(row["lesson_json"]) for row in rows]

    def save_strategy(self, strategy: StrategyBundle) -> StrategyBundle:
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO strategies(id, task_type, version, status, source, parent_strategy_id, bundle_json, created_at, activated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    strategy.id,
                    strategy.task_type,
                    strategy.version,
                    strategy.status.value,
                    strategy.source,
                    strategy.parent_strategy_id,
                    self._dump_json(strategy),
                    strategy.created_at.isoformat(),
                    strategy.activated_at.isoformat() if strategy.activated_at else None,
                ),
            )
        return strategy

    def get_strategy(self, strategy_id: str) -> StrategyBundle:
        with self._connect() as connection:
            row = connection.execute("SELECT bundle_json FROM strategies WHERE id = ?", (strategy_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown strategy id: {strategy_id}")
        return StrategyBundle.model_validate_json(row["bundle_json"])

    def get_active_strategy(self, task_type: str) -> StrategyBundle | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT bundle_json FROM strategies WHERE task_type = ? AND status = ? ORDER BY version DESC LIMIT 1",
                (task_type, StrategyStatus.active.value),
            ).fetchone()
        return StrategyBundle.model_validate_json(row["bundle_json"]) if row else None

    def get_strategy_by_source(self, source: str, task_type: str | None = None) -> StrategyBundle | None:
        sql = "SELECT bundle_json FROM strategies WHERE source = ?"
        params: list[Any] = [source]
        if task_type:
            sql += " AND task_type = ?"
            params.append(task_type)
        sql += " ORDER BY version DESC LIMIT 1"
        with self._connect() as connection:
            row = connection.execute(sql, params).fetchone()
        return StrategyBundle.model_validate_json(row["bundle_json"]) if row else None

    def list_strategies(self, task_type: str | None = None, status: StrategyStatus | None = None) -> list[StrategyBundle]:
        sql = "SELECT bundle_json FROM strategies"
        params: list[Any] = []
        filters: list[str] = []
        if task_type:
            filters.append("task_type = ?")
            params.append(task_type)
        if status:
            filters.append("status = ?")
            params.append(status.value)
        if filters:
            sql += " WHERE " + " AND ".join(filters)
        sql += " ORDER BY version DESC"
        with self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [StrategyBundle.model_validate_json(row["bundle_json"]) for row in rows]

    def next_strategy_version(self, task_type: str) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COALESCE(MAX(version), 0) AS version FROM strategies WHERE task_type = ?", (task_type,)).fetchone()
        return int(row["version"] or 0) + 1

    def archive_active_strategies(self, task_type: str) -> None:
        active = self.list_strategies(task_type=task_type, status=StrategyStatus.active)
        with self._connect() as connection:
            for strategy in active:
                archived = strategy.model_copy(update={"status": StrategyStatus.archived})
                connection.execute("UPDATE strategies SET status = ?, bundle_json = ? WHERE id = ?", (archived.status.value, self._dump_json(archived), archived.id))

    def save_promotion_decision(self, decision: PromotionDecision) -> PromotionDecision:
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO promotion_decisions(id, candidate_strategy_id, baseline_strategy_id, approved, decision_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    decision.id,
                    decision.candidate_strategy_id,
                    decision.baseline_strategy_id,
                    1 if decision.approved else 0,
                    self._dump_json(decision),
                    decision.created_at.isoformat(),
                ),
            )
        return decision

    def save_memory(self, memory: MemoryRecord) -> MemoryRecord:
        embedding = memory.embedding or hash_embedding(memory.summary + "\n" + memory.content)
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO memories(id, memory_type, run_id, task_id, summary, content, embedding, metadata_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (memory.id, memory.memory_type.value, memory.run_id, memory.task_id, memory.summary, memory.content, pack_embedding(embedding), self._dump_json(memory.metadata), memory.created_at.isoformat()),
            )
            connection.execute("DELETE FROM memory_fts WHERE memory_id = ?", (memory.id,))
            connection.execute("INSERT INTO memory_fts(memory_id, summary, content) VALUES (?, ?, ?)", (memory.id, memory.summary, memory.content))
        return memory.model_copy(update={"embedding": embedding})

    def list_memories(self, memory_type: str | None = None, limit: int = 50) -> list[MemoryRecord]:
        sql = "SELECT * FROM memories"
        params: list[Any] = []
        if memory_type:
            sql += " WHERE memory_type = ?"
            params.append(memory_type)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [self._row_to_memory(row) for row in rows]

    def list_memories_by_run(self, run_id: str) -> list[MemoryRecord]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM memories WHERE run_id = ? ORDER BY created_at ASC", (run_id,)).fetchall()
        return [self._row_to_memory(row) for row in rows]

    def search_memories(self, query: str, limit: int, threshold: float, task_type: str) -> list[MemoryRecord]:
        query_embedding = hash_embedding(query)
        token_query = " OR ".join(tokenize(query)[:6])
        fts_ids: set[str] = set()
        with self._connect() as connection:
            if token_query:
                try:
                    rows = connection.execute("SELECT memory_id FROM memory_fts WHERE memory_fts MATCH ? LIMIT ?", (token_query, max(limit * 5, 10))).fetchall()
                    fts_ids = {row["memory_id"] for row in rows}
                except sqlite3.OperationalError:
                    fts_ids = set()
            rows = connection.execute("SELECT * FROM memories ORDER BY created_at DESC LIMIT ?", (max(limit * 10, 50),)).fetchall()
        scored: list[tuple[float, MemoryRecord]] = []
        for row in rows:
            memory = self._row_to_memory(row)
            applicability = str(memory.metadata.get("applicability", "global"))
            if applicability not in {"global", task_type}:
                continue
            score = cosine_similarity(query_embedding, memory.embedding) + (0.2 if memory.id in fts_ids else 0.0)
            if score >= threshold:
                scored.append((score, memory))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [memory for _, memory in scored[:limit]]

    def _row_to_memory(self, row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(id=row["id"], memory_type=row["memory_type"], run_id=row["run_id"], task_id=row["task_id"], summary=row["summary"], content=row["content"], embedding=unpack_embedding(row["embedding"]), metadata=json.loads(row["metadata_json"]), created_at=row["created_at"])


class Guardrail:
    banned_fragments = [" rm ", " del ", "shutdown", "format", "mkfs", "rmdir /s"]

    def __init__(self, settings: Settings):
        self.settings = settings

    def workspace_root(self, task: TaskSpec) -> Path:
        return Path(task.permissions.workspace_root or self.settings.default_workspace_root).resolve()

    def resolve_path(self, task: TaskSpec, target: str) -> Path:
        workspace = self.workspace_root(task)
        candidate = Path(target)
        if not candidate.is_absolute():
            candidate = workspace / candidate
        candidate = candidate.resolve()
        if candidate != workspace and workspace not in candidate.parents:
            raise GuardrailError(f"Path escapes workspace: {target}")
        return candidate

    def check_shell(self, task: TaskSpec, command: str) -> None:
        if not task.permissions.enable_shell:
            raise GuardrailError("Shell tool is disabled.")
        lowered = f" {command.lower()} "
        if any(fragment in lowered for fragment in self.banned_fragments):
            raise GuardrailError("Blocked dangerous shell command.")
        tokens = shlex.split(command, posix=False)
        if not tokens:
            raise GuardrailError("Empty shell command.")
        binary = tokens[0].lower()
        allowed = {item.lower() for item in (task.permissions.allowed_shell_prefixes or self.settings.default_shell_prefixes)}
        if binary not in allowed:
            raise GuardrailError(f"Shell command not in allowlist: {binary}")

    def check_http(self, task: TaskSpec, url: str) -> None:
        if not task.permissions.enable_http:
            raise GuardrailError("HTTP tool is disabled.")
        hostname = urlparse(url).hostname or ""
        allowed = set(task.permissions.allowed_http_domains or self.settings.default_http_domains)
        if hostname not in allowed:
            raise GuardrailError(f"HTTP domain not allowed: {hostname}")

class ToolRouter:
    def __init__(self, guardrail: Guardrail):
        self.guardrail = guardrail

    def execute(self, task: TaskSpec, action: ToolAction) -> ToolResult:
        if action.name == "file.write":
            if not task.permissions.enable_file:
                raise GuardrailError("File tool is disabled.")
            path = self.guardrail.resolve_path(task, str(action.params["path"]))
            path.parent.mkdir(parents=True, exist_ok=True)
            content = str(action.params.get("content", ""))
            path.write_text(content, encoding="utf-8")
            return ToolResult(True, f"Wrote {len(content)} characters.", str(path))
        if action.name == "file.append":
            if not task.permissions.enable_file:
                raise GuardrailError("File tool is disabled.")
            path = self.guardrail.resolve_path(task, str(action.params["path"]))
            path.parent.mkdir(parents=True, exist_ok=True)
            content = str(action.params.get("content", ""))
            with path.open("a", encoding="utf-8") as handle:
                handle.write(content)
            return ToolResult(True, f"Appended {len(content)} characters.", str(path))
        if action.name == "file.read":
            path = self.guardrail.resolve_path(task, str(action.params["path"]))
            if not path.exists():
                raise ToolExecutionError(f"File not found: {path}")
            return ToolResult(True, path.read_text(encoding="utf-8"), str(path), {"length": path.stat().st_size})
        if action.name == "shell.run":
            command = str(action.params["command"])
            self.guardrail.check_shell(task, command)
            result = subprocess.run(command, shell=True, cwd=str(self.guardrail.workspace_root(task)), capture_output=True, text=True, timeout=task.budget.timeout_seconds)
            output = ((result.stdout or "") + (result.stderr or "")).strip()
            if result.returncode != 0:
                raise ToolExecutionError(f"Shell command failed with exit code {result.returncode}: {output}")
            return ToolResult(True, output, metadata={"exit_code": result.returncode})
        if action.name in {"http.get", "http.request"}:
            url = str(action.params["url"])
            self.guardrail.check_http(task, url)
            request = Request(url=url, method=str(action.params.get("method", "GET")).upper(), headers=action.params.get("headers", {}))
            with urlopen(request, timeout=task.budget.timeout_seconds) as response:
                body = response.read().decode("utf-8")
                status = response.status
            if status >= 400:
                raise ToolExecutionError(f"HTTP request failed with status {status}")
            return ToolResult(True, body, metadata={"status_code": status, "url": url})
        raise ToolExecutionError(f"Unsupported action: {action.name}")


class HeuristicProvider:
    name = "heuristic"

    def plan(self, task: TaskSpec, strategy: StrategyBundle, memories: list[MemoryRecord]) -> list[ToolAction]:
        actions: list[ToolAction] = []
        context = task.context
        if "action_plan" in context:
            actions = [ToolAction.model_validate(item) for item in context["action_plan"]]
        elif "tool_options" in context:
            options = [ToolAction.model_validate(item) for item in context["tool_options"]]
            options.sort(key=lambda item: strategy.tool_weights.get(item.name.split(".", 1)[0], 0.0), reverse=True)
            if options:
                actions = [options[0]]
        else:
            if "write_file" in context:
                actions.append(ToolAction(name="file.write", params=context["write_file"], rationale="structured write"))
            if "append_file" in context:
                actions.append(ToolAction(name="file.append", params=context["append_file"], rationale="structured append"))
            if "read_file" in context:
                actions.append(ToolAction(name="file.read", params=context["read_file"], rationale="structured read"))
            if "shell_command" in context:
                actions.append(ToolAction(name="shell.run", params={"command": context["shell_command"]}, rationale="structured shell"))
            if "http_request" in context:
                request = dict(context["http_request"])
                actions.append(ToolAction(name="http.request", params=request, rationale="structured http"))
        if not actions:
            actions = [ToolAction(name="finish", params={"message": f"No deterministic action available for: {task.objective}"}, rationale="fallback")]
        if actions[-1].name != "finish":
            memory_hint = f" Reused {len(memories)} memories." if memories else ""
            actions.append(ToolAction(name="finish", params={"message": task.context.get("final_output", f"Completed objective: {task.objective}.{memory_hint}".strip())}, rationale="finish"))
        return actions

    def judge(self, trace: RunTrace, outcome: TaskOutcome) -> tuple[bool, float, str]:
        errors = sum(1 for event in trace.events if event.type == EventType.error)
        success = errors == 0 and bool(outcome.final_output.strip())
        return success, (0.55 if success else 0.35), ("Heuristic judge inferred success." if success else "Heuristic judge observed failures or empty output.")


class OpenAICompatibleProvider:
    name = "openai_compatible"

    def __init__(self, base_url: str, api_key: str | None, model: str, site_url: str | None = None, app_name: str = "Agent Loop"):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.site_url = site_url
        self.app_name = app_name

    def plan(self, task: TaskSpec, strategy: StrategyBundle, memories: list[MemoryRecord]) -> list[ToolAction]:
        if not self.api_key:
            raise ProviderError("AGENT_LOOP_OPENAI_API_KEY is required for the OpenAI-compatible provider.")
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": f"{strategy.system_prompt}\n{strategy.planning_template}\nYou are planning tool actions for a local agent. Allowed tools: file.write, file.append, file.read, shell.run, http.get, http.request, finish.\nIf task.context.action_plan exists, treat it as the preferred authoritative workflow unless it violates the objective.\nReturn strict JSON only, with the shape {{'actions': [{{'name': str, 'params': object, 'rationale': str}}]}}. End with a finish action."},
                {"role": "user", "content": json.dumps({"objective": task.objective, "context": task.context, "memories": [memory.summary for memory in memories]}, ensure_ascii=False)},
            ],
        }
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.app_name:
            headers["X-Title"] = self.app_name
        last_error: Exception | None = None
        for attempt in range(3):
            request = Request(url=f"{self.base_url}/chat/completions", data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
            try:
                with urlopen(request, timeout=60) as response:
                    data = json.loads(response.read().decode("utf-8"))
                break
            except URLError as exc:
                last_error = exc
                if attempt == 2:
                    raise ProviderError(f"Planner request failed after 3 attempts: {exc}") from exc
                time.sleep(1.5 * (attempt + 1))
        content = self._extract_json_content(data["choices"][0]["message"]["content"])
        return [ToolAction.model_validate(item) for item in json.loads(content)["actions"]]

    def judge(self, trace: RunTrace, outcome: TaskOutcome) -> tuple[bool, float, str]:
        return HeuristicProvider().judge(trace, outcome)

    @staticmethod
    def _extract_json_content(content: str) -> str:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if len(lines) >= 3:
                cleaned = "\n".join(lines[1:-1]).strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end >= start:
            cleaned = cleaned[start : end + 1]
        return cleaned


def build_provider(settings: Settings) -> HeuristicProvider | OpenAICompatibleProvider:
    if settings.provider == "openai_compatible":
        return OpenAICompatibleProvider(settings.openai_base_url, settings.openai_api_key, settings.model, settings.openai_site_url, settings.openai_app_name)
    return HeuristicProvider()


class RunRecorder:
    def __init__(self, store: SQLiteStore, run_id: str, task_id: str, strategy_id: str):
        self.store = store
        self.trace = RunTrace(run_id=run_id, task_id=task_id, strategy_id=strategy_id)

    def record(self, event_type: EventType, payload: dict[str, Any]) -> TraceEvent:
        event = TraceEvent(type=event_type, payload=payload)
        self.trace.events.append(event)
        self.store.append_event(self.trace.run_id, len(self.trace.events) - 1, event)
        return event


class AgentRuntime:
    def __init__(self, tools: ToolRouter):
        self.tools = tools

    def execute(self, task: TaskSpec, strategy: StrategyBundle, provider: HeuristicProvider | OpenAICompatibleProvider, memories: list[MemoryRecord], recorder: RunRecorder) -> tuple[RunTrace, TaskOutcome]:
        recorder.record(EventType.memory_retrieval, {"memory_ids": [memory.id for memory in memories], "count": len(memories)})
        try:
            actions = provider.plan(task, strategy, memories)
        except ProviderError as exc:
            outcome = TaskOutcome(status=RunStatus.failed, error=str(exc), success=False, final_output=str(exc), result_summary=str(exc))
            recorder.record(EventType.error, {"message": str(exc), "phase": "plan"})
            recorder.trace.completed_at = utc_now()
            return recorder.trace, outcome
        recorder.record(EventType.plan, {"strategy_version": strategy.version, "actions": [action.model_dump(mode="json") for action in actions]})
        outcome = TaskOutcome(status=RunStatus.running)
        artifacts: list[str] = []
        errors = 0
        repeated_errors: dict[str, int] = {}
        observations: list[str] = []
        for step_index, action in enumerate(actions, start=1):
            if step_index > task.budget.max_steps:
                outcome.error = "Step budget exhausted."
                recorder.record(EventType.error, {"message": outcome.error})
                break
            recorder.trace.total_steps += 1
            if action.name == "finish":
                outcome.final_output = str(action.params.get("message", ""))
                recorder.record(EventType.finish, {"message": outcome.final_output})
                break
            if action.name.startswith("shell") and recorder.trace.shell_calls >= task.budget.max_shell_calls:
                outcome.error = "Shell call budget exhausted."
                recorder.record(EventType.error, {"message": outcome.error, "action": action.name})
                break
            if action.name.startswith("http") and recorder.trace.http_calls >= task.budget.max_http_calls:
                outcome.error = "HTTP call budget exhausted."
                recorder.record(EventType.error, {"message": outcome.error, "action": action.name})
                break
            attempts = 0
            while True:
                try:
                    recorder.record(EventType.tool_call, {"action": action.model_dump(mode="json"), "attempt": attempts + 1})
                    result = self.tools.execute(task, action)
                    if action.name.startswith("shell"):
                        recorder.trace.shell_calls += 1
                    if action.name.startswith("http"):
                        recorder.trace.http_calls += 1
                    if result.artifact_path:
                        artifacts.append(result.artifact_path)
                    observations.append(result.output)
                    recorder.record(EventType.tool_result, {"action": action.name, "output": result.output, "artifact_path": result.artifact_path, "metadata": result.metadata})
                    recorder.record(EventType.reflection, {"observation": result.output[:200], "step": step_index})
                    break
                except (GuardrailError, ToolExecutionError, subprocess.TimeoutExpired) as exc:
                    errors += 1
                    message = str(exc)
                    repeated_errors[message] = repeated_errors.get(message, 0) + 1
                    recorder.record(EventType.error, {"action": action.name, "message": message, "attempt": attempts + 1})
                    if attempts < strategy.retry_budget and errors <= task.budget.max_errors and repeated_errors[message] < 2:
                        attempts += 1
                        recorder.record(EventType.retry, {"action": action.name, "attempt": attempts})
                        continue
                    outcome.error = message
                    break
            if outcome.error:
                break
        if not outcome.final_output:
            outcome.final_output = observations[-1] if observations else (outcome.error or "No output generated.")
        outcome.artifact_paths = list(dict.fromkeys(artifacts))
        outcome.result_summary = outcome.final_output[:240]
        outcome.status = RunStatus.failed if outcome.error else RunStatus.succeeded
        outcome.success = outcome.error is None
        recorder.trace.token_usage = sum(len(event.payload.get("output", "")) for event in recorder.trace.events if isinstance(event.payload.get("output", ""), str))
        recorder.trace.estimated_cost = round(0.0002 * recorder.trace.total_steps + 0.00005 * recorder.trace.http_calls + 0.00003 * recorder.trace.shell_calls, 6)
        recorder.trace.final_output = outcome.final_output
        recorder.trace.completed_at = utc_now()
        return recorder.trace, outcome

class EvaluationService:
    def __init__(self, guardrail: Guardrail, provider: HeuristicProvider | OpenAICompatibleProvider):
        self.guardrail = guardrail
        self.provider = provider

    def evaluate(self, task: TaskSpec, trace: RunTrace, outcome: TaskOutcome) -> EvalReport:
        if task.success_checks:
            details: list[dict[str, Any]] = []
            passed = 0
            failure_reason: str | None = None
            for check in task.success_checks:
                success, detail = self._evaluate_check(task, check.kind.value, check.target, check.expected, trace, outcome)
                details.append({"kind": check.kind.value, "target": check.target, "expected": check.expected, "success": success, "detail": detail})
                if success:
                    passed += 1
                elif failure_reason is None:
                    failure_reason = detail
            success = passed == len(task.success_checks)
            return EvalReport(run_id=trace.run_id, mode="rule", success=success, confidence=0.95, summary=f"{passed}/{len(task.success_checks)} rule checks passed.", failure_reason=failure_reason, metrics={"checks": details, "steps": trace.total_steps, "errors": sum(1 for event in trace.events if event.type == EventType.error)})
        success, confidence, summary = self.provider.judge(trace, outcome)
        failure = None if success else outcome.error or "Judge marked the run as unsuccessful."
        return EvalReport(run_id=trace.run_id, mode="heuristic", success=success, confidence=confidence, summary=summary, failure_reason=failure, metrics={"low_confidence": confidence < 0.6, "steps": trace.total_steps})

    def _evaluate_check(self, task: TaskSpec, kind: str, target: str | None, expected: Any, trace: RunTrace, outcome: TaskOutcome) -> tuple[bool, str]:
        if kind == SuccessCheckKind.file_exists.value and target:
            path = self.guardrail.resolve_path(task, target)
            return path.exists(), f"File {'exists' if path.exists() else 'is missing'}: {path}"
        if kind == SuccessCheckKind.file_contains.value and target:
            path = self.guardrail.resolve_path(task, target)
            if not path.exists():
                return False, f"File is missing: {path}"
            content = path.read_text(encoding="utf-8")
            expected_text = str(expected or "")
            return expected_text in content, f"Expected text {'found' if expected_text in content else 'not found'} in {path.name}"
        if kind == SuccessCheckKind.final_output_contains.value:
            expected_text = str(expected or "")
            return expected_text in outcome.final_output, "Final output check evaluated."
        if kind == SuccessCheckKind.http_status.value:
            status = None
            for event in reversed(trace.events):
                if event.type == EventType.tool_result and event.payload.get("metadata", {}).get("url") == target:
                    status = event.payload.get("metadata", {}).get("status_code")
                    break
            return status == expected, f"Observed status {status}, expected {expected}."
        if kind == SuccessCheckKind.no_error_events.value:
            errors = sum(1 for event in trace.events if event.type == EventType.error)
            return errors == 0, f"Observed {errors} error events."
        return False, f"Unsupported success check kind: {kind}"


class StrategyRegistry:
    def __init__(self, store: SQLiteStore):
        self.store = store

    def ensure_active(self, task_type: str) -> StrategyBundle:
        strategy = self.store.get_active_strategy(task_type)
        if strategy:
            return strategy
        strategy = StrategyBundle(version=self.store.next_strategy_version(task_type), task_type=task_type, status=StrategyStatus.active, source="default", activated_at=utc_now())
        self.store.save_strategy(strategy)
        procedural_memory = MemoryRecord(memory_type=MemoryType.procedural, summary=f"Active strategy v{strategy.version} for {task_type}", content=json.dumps({"planning_template": strategy.planning_template, "tool_weights": strategy.tool_weights, "retry_budget": strategy.retry_budget}, ensure_ascii=False), metadata={"applicability": task_type, "strategy_id": strategy.id})
        self.store.save_memory(procedural_memory)
        return strategy

    def create_candidate(self, parent: StrategyBundle, source: str, fields: dict[str, Any]) -> StrategyBundle:
        candidate = parent.model_copy(deep=True)
        candidate.id = new_id("strategy")
        candidate.version = self.store.next_strategy_version(parent.task_type)
        candidate.status = StrategyStatus.candidate
        candidate.source = source
        candidate.parent_strategy_id = parent.id
        candidate.created_at = utc_now()
        candidate.activated_at = None
        for key, value in fields.items():
            setattr(candidate, key, value)
        self.store.save_strategy(candidate)
        procedural_memory = MemoryRecord(memory_type=MemoryType.procedural, summary=f"Candidate strategy v{candidate.version} for {candidate.task_type}", content=json.dumps({"planning_template": candidate.planning_template, "tool_weights": candidate.tool_weights, "retry_budget": candidate.retry_budget}, ensure_ascii=False), metadata={"applicability": candidate.task_type, "strategy_id": candidate.id})
        self.store.save_memory(procedural_memory)
        return candidate

    def promote(self, strategy: StrategyBundle) -> StrategyBundle:
        self.store.archive_active_strategies(strategy.task_type)
        promoted = strategy.model_copy(update={"status": StrategyStatus.active, "activated_at": utc_now()})
        self.store.save_strategy(promoted)
        return promoted


class LearningPipeline:
    def __init__(self, store: SQLiteStore, registry: StrategyRegistry):
        self.store = store
        self.registry = registry

    def process(self, task: TaskSpec, trace: RunTrace, outcome: TaskOutcome, evaluation: EvalReport, strategy: StrategyBundle) -> LearningResult:
        existing_lessons = self.store.list_lessons(trace.run_id)
        existing_candidate = self.store.get_strategy_by_source(f"learning:{trace.run_id}", task.task_type)
        memories = self.store.list_memories_by_run(trace.run_id)
        if not memories:
            memories = self._write_memories(task, trace, outcome, evaluation)
        lessons = existing_lessons or self._derive_lessons(task, trace, evaluation)
        if not existing_lessons:
            self.store.save_lessons(lessons)
        candidate = existing_candidate or self._derive_candidate(task, trace, evaluation, strategy, lessons)
        return LearningResult(run_id=trace.run_id, memories=memories, lessons=lessons, candidate_strategy=candidate)

    def _write_memories(self, task: TaskSpec, trace: RunTrace, outcome: TaskOutcome, evaluation: EvalReport) -> list[MemoryRecord]:
        episodic = MemoryRecord(memory_type=MemoryType.episodic, run_id=trace.run_id, task_id=task.id, summary=f"{task.objective} -> {'success' if evaluation.success else 'failure'}", content=json.dumps({"final_output": outcome.final_output, "failure_reason": evaluation.failure_reason, "steps": trace.total_steps}, ensure_ascii=False), metadata={"applicability": task.task_type})
        memories = [self.store.save_memory(episodic)]
        if evaluation.failure_reason:
            semantic = MemoryRecord(memory_type=MemoryType.semantic, run_id=trace.run_id, task_id=task.id, summary=f"Failure pattern for {task.task_type}", content=evaluation.failure_reason, metadata={"applicability": task.task_type, "kind": "failure_pattern"})
            memories.append(self.store.save_memory(semantic))
        return memories

    def _derive_lessons(self, task: TaskSpec, trace: RunTrace, evaluation: EvalReport) -> list[LessonCandidate]:
        errors = [event.payload.get("message", "") for event in trace.events if event.type == EventType.error]
        lessons: list[LessonCandidate] = []
        if evaluation.failure_reason:
            lessons.append(LessonCandidate(run_id=trace.run_id, title="Avoid the failing action path", description=evaluation.failure_reason, applicability=task.task_type, confidence=0.82, metadata={"error_count": len(errors), "primary_error": errors[0] if errors else evaluation.failure_reason}))
        if trace.total_steps >= max(task.budget.max_steps - 1, 2):
            lessons.append(LessonCandidate(run_id=trace.run_id, title="Prefer shorter decompositions", description="The run consumed most of the step budget; favor simpler plans on similar tasks.", applicability=task.task_type, confidence=0.64, metadata={"steps": trace.total_steps}))
        return lessons

    def _derive_candidate(self, task: TaskSpec, trace: RunTrace, evaluation: EvalReport, strategy: StrategyBundle, lessons: list[LessonCandidate]) -> StrategyBundle | None:
        if not lessons:
            return None
        fields: dict[str, Any] = {
            "planning_template": strategy.planning_template,
            "tool_weights": dict(strategy.tool_weights),
            "retrieval_threshold": strategy.retrieval_threshold,
            "retry_budget": strategy.retry_budget,
            "decomposition_rules": list(strategy.decomposition_rules),
            "system_prompt": strategy.system_prompt,
        }
        primary_error = " ".join([lesson.description + " " + str(lesson.metadata.get('primary_error', '')) for lesson in lessons] + [str(event.payload.get('message', '')) for event in trace.events if event.type == EventType.error]).lower()
        if "shell command not in allowlist" in primary_error or "blocked dangerous shell command" in primary_error:
            fields["tool_weights"]["shell"] = min(fields["tool_weights"].get("shell", 1.0), 0.1)
            fields["planning_template"] += " Verify shell allowlists before planning any command."
        if "path escapes workspace" in primary_error:
            fields["planning_template"] += " Keep all file operations strictly inside the workspace root."
        if evaluation.failure_reason:
            fields["retrieval_threshold"] = max(0.05, strategy.retrieval_threshold - 0.05)
            fields["retry_budget"] = max(0, strategy.retry_budget - 1)
            fields["decomposition_rules"].append("When multiple tool options exist, prefer the safer deterministic route.")
        elif trace.total_steps > 2:
            fields["retry_budget"] = max(0, strategy.retry_budget - 1)
            fields["planning_template"] += " Favor shorter plans when similar memories exist."
        return self.registry.create_candidate(strategy, f"learning:{trace.run_id}", fields)

class AgentLoopService:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.from_env()
        self.store = SQLiteStore(self.settings.resolved_db_path())
        self.guardrail = Guardrail(self.settings)
        self.tools = ToolRouter(self.guardrail)
        self.provider = build_provider(self.settings)
        self.runtime = AgentRuntime(self.tools)
        self.evaluator = EvaluationService(self.guardrail, self.provider)
        self.registry = StrategyRegistry(self.store)
        self.learning = LearningPipeline(self.store, self.registry)

    def run_task(
        self,
        task: TaskSpec,
        auto_learn: bool = True,
        auto_promote: bool = False,
        replay_limit: int = 3,
    ) -> RunResult:
        task = self._normalize_task(task)
        self.store.save_task(task)
        strategy = self.registry.ensure_active(task.task_type)
        query = task.objective + "\n" + json.dumps(task.context, ensure_ascii=False)
        memories = self.store.search_memories(query, limit=5, threshold=strategy.retrieval_threshold, task_type=task.task_type)
        run_id = new_id("run")
        self.store.create_run(run_id, task, strategy)
        recorder = RunRecorder(self.store, run_id, task.id, strategy.id)
        trace, outcome = self.runtime.execute(task, strategy, self.provider, memories, recorder)
        evaluation = self.evaluator.evaluate(task, trace, outcome)
        recorder.record(EventType.evaluation, evaluation.model_dump(mode="json"))
        trace = recorder.trace
        trace.completed_at = utc_now()
        outcome.success = evaluation.success
        outcome.status = RunStatus.succeeded if evaluation.success else RunStatus.failed
        outcome.result_summary = evaluation.summary
        if evaluation.success:
            outcome.error = None
        elif not outcome.error:
            outcome.error = evaluation.failure_reason
        self.store.save_evaluation(evaluation)
        learning = None
        promotion_error = None
        if auto_learn:
            learning = self.learning.process(task, trace, outcome, evaluation, strategy)
            if auto_promote and learning.candidate_strategy is not None:
                try:
                    replay_report, promotion_decision = self._evaluate_promotion_candidate(
                        learning.candidate_strategy.id,
                        limit=replay_limit,
                    )
                    learning.replay_report = replay_report
                    learning.promotion_decision = promotion_decision
                except Exception as exc:
                    promotion_error = str(exc)
                    recorder.record(
                        EventType.error,
                        {
                            "action": "strategy.promote",
                            "candidate_strategy_id": learning.candidate_strategy.id,
                            "message": promotion_error,
                        },
                    )
            recorder.record(
                EventType.learning,
                {
                    "memory_ids": [memory.id for memory in learning.memories],
                    "lesson_ids": [lesson.id for lesson in learning.lessons],
                    "candidate_strategy_id": learning.candidate_strategy.id if learning.candidate_strategy else None,
                    "replay_approved": learning.replay_report.approved if learning.replay_report else None,
                    "promotion_approved": learning.promotion_decision.approved if learning.promotion_decision else None,
                    "promotion_error": promotion_error,
                },
            )
            trace = recorder.trace
            trace.completed_at = utc_now()
        self.store.finalize_run(trace, outcome)
        return RunResult(task=task, trace=trace, outcome=outcome, evaluation=evaluation, learning=learning)

    def inspect_run(self, run_id: str) -> RunInspection:
        trace = self.store.get_run_trace(run_id)
        return RunInspection(task=self.store.get_task(trace.task_id), trace=trace, outcome=self.store.get_outcome(run_id), evaluation=self.store.get_evaluation(run_id), lessons=self.store.list_lessons(run_id))

    def learn_from_run(self, run_id: str) -> LearningResult:
        inspection = self.inspect_run(run_id)
        strategy = self.store.get_strategy(inspection.trace.strategy_id)
        evaluation = inspection.evaluation or self.evaluator.evaluate(inspection.task, inspection.trace, inspection.outcome)
        if inspection.evaluation is None:
            self.store.save_evaluation(evaluation)
        return self.learning.process(inspection.task, inspection.trace, inspection.outcome, evaluation, strategy)

    def replay_strategy(self, strategy_id: str, limit: int = 5) -> ReplayReport:
        candidate = self.store.get_strategy(strategy_id)
        baseline = self.store.get_strategy(candidate.parent_strategy_id) if candidate.parent_strategy_id else self.registry.ensure_active(candidate.task_type)
        tasks = self._historical_replay_tasks(candidate, limit)
        if not tasks:
            return ReplayReport(
                candidate_strategy_id=candidate.id,
                baseline_strategy_id=baseline.id,
                tasks_evaluated=0,
                candidate_success_rate=0.0,
                baseline_success_rate=0.0,
                candidate_avg_steps=0.0,
                baseline_avg_steps=0.0,
                candidate_avg_cost=0.0,
                baseline_avg_cost=0.0,
                candidate_error_rate=0.0,
                baseline_error_rate=0.0,
                approved=False,
                summary="Not enough historical tasks available for replay.",
                task_ids=[],
            )
        baseline_results = [self._execute_ephemeral(task, baseline) for task in tasks]
        candidate_results = [self._execute_ephemeral(task, candidate) for task in tasks]
        candidate_success_rate = sum(1 for result in candidate_results if result.evaluation.success) / len(candidate_results)
        baseline_success_rate = sum(1 for result in baseline_results if result.evaluation.success) / len(baseline_results)
        candidate_avg_steps = sum(result.trace.total_steps for result in candidate_results) / len(candidate_results)
        baseline_avg_steps = sum(result.trace.total_steps for result in baseline_results) / len(baseline_results)
        candidate_avg_cost = sum(result.trace.estimated_cost for result in candidate_results) / len(candidate_results)
        baseline_avg_cost = sum(result.trace.estimated_cost for result in baseline_results) / len(baseline_results)
        candidate_error_rate = sum(1 for result in candidate_results if result.outcome.error) / len(candidate_results)
        baseline_error_rate = sum(1 for result in baseline_results if result.outcome.error) / len(baseline_results)
        success_gain = candidate_success_rate > baseline_success_rate + 1e-9
        step_gain = candidate_avg_steps < baseline_avg_steps - 0.1
        cost_gain = candidate_avg_cost < baseline_avg_cost - 1e-6
        success_not_worse = candidate_success_rate >= baseline_success_rate - 1e-9
        error_not_worse = candidate_error_rate <= baseline_error_rate + 1e-9
        approved = success_not_worse and error_not_worse and (success_gain or step_gain or cost_gain)
        if approved:
            summary = "Candidate meets strict promotion thresholds against historical tasks."
        else:
            summary = "Candidate did not strictly improve over the active baseline on historical replay tasks."
        return ReplayReport(candidate_strategy_id=candidate.id, baseline_strategy_id=baseline.id, tasks_evaluated=len(tasks), candidate_success_rate=candidate_success_rate, baseline_success_rate=baseline_success_rate, candidate_avg_steps=candidate_avg_steps, baseline_avg_steps=baseline_avg_steps, candidate_avg_cost=candidate_avg_cost, baseline_avg_cost=baseline_avg_cost, candidate_error_rate=candidate_error_rate, baseline_error_rate=baseline_error_rate, approved=approved, summary=summary, task_ids=[task.id for task in tasks])

    def promote_strategy(self, strategy_id: str, limit: int = 5) -> PromotionDecision:
        _, decision = self._evaluate_promotion_candidate(strategy_id, limit=limit)
        return decision

    def _historical_replay_tasks(self, candidate: StrategyBundle, limit: int) -> list[TaskSpec]:
        fetch_limit = max(limit * 5, limit + 10)
        tasks = self.store.list_tasks(task_type=candidate.task_type, limit=fetch_limit)
        source_run_id = candidate.source.split(":", 1)[1] if candidate.source.startswith("learning:") else None
        source_task_id = None
        source_created_at = None
        if source_run_id:
            try:
                source_trace = self.store.get_run_trace(source_run_id)
                source_task = self.store.get_task(source_trace.task_id)
                source_task_id = source_task.id
                source_created_at = source_task.created_at
            except KeyError:
                source_task_id = None
                source_created_at = None
        historical = []
        for task in tasks:
            if source_task_id and task.id == source_task_id:
                continue
            if source_created_at and task.created_at >= source_created_at:
                continue
            historical.append(task)
            if len(historical) >= limit:
                break
        historical.reverse()
        return historical
    def _evaluate_promotion_candidate(self, strategy_id: str, limit: int = 5) -> tuple[ReplayReport, PromotionDecision]:
        replay = self.replay_strategy(strategy_id, limit=limit)
        candidate = self.store.get_strategy(strategy_id)
        decision = PromotionDecision(
            candidate_strategy_id=replay.candidate_strategy_id,
            baseline_strategy_id=replay.baseline_strategy_id,
            approved=replay.approved,
            summary=replay.summary,
            metrics=replay.model_dump(mode="json"),
        )
        self.store.save_promotion_decision(decision)
        if replay.approved:
            self.registry.promote(candidate)
        return replay, decision

    def list_memories(self, memory_type: str | None = None, limit: int = 50) -> list[MemoryRecord]:
        return self.store.list_memories(memory_type=memory_type, limit=limit)

    def list_strategies(self, task_type: str | None = None, status: StrategyStatus | None = None) -> list[StrategyBundle]:
        return self.store.list_strategies(task_type=task_type, status=status)

    def _normalize_task(self, task: TaskSpec) -> TaskSpec:
        task = task.model_copy(deep=True)
        if not task.permissions.workspace_root:
            task.permissions.workspace_root = str(self.settings.default_workspace_root)
        if not task.permissions.allowed_shell_prefixes:
            task.permissions.allowed_shell_prefixes = list(self.settings.default_shell_prefixes)
        if not task.permissions.allowed_http_domains:
            task.permissions.allowed_http_domains = list(self.settings.default_http_domains)
        return task

    def _execute_ephemeral(self, task: TaskSpec, strategy: StrategyBundle) -> RunResult:
        task = self._normalize_task(task)
        original_workspace = Path(task.permissions.workspace_root or self.settings.default_workspace_root).resolve()
        replay_parent = self.settings.home_dir / "replays"
        replay_parent.mkdir(parents=True, exist_ok=True)
        replay_workspace = Path(tempfile.mkdtemp(prefix="replay_", dir=str(replay_parent)))
        if original_workspace.exists():
            ignored_names = {self.settings.home_dir.name, "__pycache__", ".pytest_cache"}
            shutil.copytree(
                original_workspace,
                replay_workspace,
                dirs_exist_ok=True,
                ignore=lambda _src, names: [name for name in names if name in ignored_names],
            )
        task = task.model_copy(deep=True)
        task.permissions.workspace_root = str(replay_workspace)
        memories = self.store.search_memories(task.objective, limit=3, threshold=strategy.retrieval_threshold, task_type=task.task_type)
        run_id = new_id("replay")
        self.store.create_run(run_id, task, strategy)
        recorder = RunRecorder(self.store, run_id, task.id, strategy.id)
        trace, outcome = self.runtime.execute(task, strategy, self.provider, memories, recorder)
        evaluation = self.evaluator.evaluate(task, trace, outcome)
        recorder.record(EventType.evaluation, evaluation.model_dump(mode="json"))
        trace = recorder.trace
        trace.completed_at = utc_now()
        outcome.success = evaluation.success
        outcome.status = RunStatus.succeeded if evaluation.success else RunStatus.failed
        outcome.result_summary = evaluation.summary
        if not evaluation.success and not outcome.error:
            outcome.error = evaluation.failure_reason
        self.store.save_evaluation(evaluation)
        self.store.finalize_run(trace, outcome)
        return RunResult(task=task, trace=trace, outcome=outcome, evaluation=evaluation, learning=None)










# Multi-turn planning overrides

def _summarize_event(event: TraceEvent) -> dict[str, Any]:
    payload = dict(event.payload)
    if "output" in payload and isinstance(payload["output"], str):
        payload["output"] = payload["output"][:300]
    if "observation" in payload and isinstance(payload["observation"], str):
        payload["observation"] = payload["observation"][:300]
    return {"type": event.type.value, "payload": payload}


class HeuristicProvider:
    name = "heuristic"

    def plan(
        self,
        task: TaskSpec,
        strategy: StrategyBundle,
        memories: list[MemoryRecord],
        trace: RunTrace,
        outcome: TaskOutcome,
    ) -> list[ToolAction]:
        context = task.context
        executed_steps = sum(1 for event in trace.events if event.type in {EventType.tool_call, EventType.finish})
        if "action_plan" in context:
            plan_actions = [ToolAction.model_validate(item) for item in context["action_plan"]]
            if executed_steps < len(plan_actions):
                return [plan_actions[executed_steps]]
            return [ToolAction(name="finish", params={"message": outcome.final_output or task.context.get("final_output", f"Completed objective: {task.objective}")}, rationale="complete planned workflow")]
        if "tool_options" in context:
            if any(event.type == EventType.tool_call for event in trace.events):
                return [ToolAction(name="finish", params={"message": task.context.get("final_output", f"Completed objective: {task.objective}")}, rationale="complete after selecting the preferred option")]
            options = [ToolAction.model_validate(item) for item in context["tool_options"]]
            options.sort(key=lambda item: strategy.tool_weights.get(item.name.split(".", 1)[0], 0.0), reverse=True)
            if options:
                return [options[0]]
            return [ToolAction(name="finish", params={"message": task.context.get("final_output", f"Completed objective: {task.objective}")}, rationale="no options available")]
        derived_actions: list[ToolAction] = []
        if "write_file" in context:
            derived_actions.append(ToolAction(name="file.write", params=context["write_file"], rationale="structured write"))
        if "append_file" in context:
            derived_actions.append(ToolAction(name="file.append", params=context["append_file"], rationale="structured append"))
        if "read_file" in context:
            derived_actions.append(ToolAction(name="file.read", params=context["read_file"], rationale="structured read"))
        if "shell_command" in context:
            derived_actions.append(ToolAction(name="shell.run", params={"command": context["shell_command"]}, rationale="structured shell"))
        if "http_request" in context:
            request = dict(context["http_request"])
            derived_actions.append(ToolAction(name="http.request", params=request, rationale="structured http"))
        if derived_actions:
            if executed_steps < len(derived_actions):
                return [derived_actions[executed_steps]]
            return [ToolAction(name="finish", params={"message": task.context.get("final_output", f"Completed objective: {task.objective}")}, rationale="structured completion")]
        return [ToolAction(name="finish", params={"message": f"No deterministic action available for: {task.objective}"}, rationale="fallback")]

    def judge(self, trace: RunTrace, outcome: TaskOutcome) -> tuple[bool, float, str]:
        errors = sum(1 for event in trace.events if event.type == EventType.error)
        success = errors == 0 and bool(outcome.final_output.strip())
        return success, (0.55 if success else 0.35), ("Heuristic judge inferred success." if success else "Heuristic judge observed failures or empty output.")


class OpenAICompatibleProvider:
    name = "openai_compatible"

    def __init__(self, base_url: str, api_key: str | None, model: str, site_url: str | None = None, app_name: str = "Agent Loop"):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.site_url = site_url
        self.app_name = app_name

    def plan(
        self,
        task: TaskSpec,
        strategy: StrategyBundle,
        memories: list[MemoryRecord],
        trace: RunTrace,
        outcome: TaskOutcome,
    ) -> list[ToolAction]:
        if not self.api_key:
            raise ProviderError("AGENT_LOOP_OPENAI_API_KEY is required for the OpenAI-compatible provider.")
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"{strategy.system_prompt}\n"
                        f"{strategy.planning_template}\n"
                        "You are controlling a local autonomous agent in a multi-turn loop. "
                        "On each turn you must choose exactly one next action. "
                        "Allowed tools: file.write, file.append, file.read, shell.run, http.get, http.request, finish.\n"
                        "Prefer reading or inspecting before writing when the output depends on unknown project content.\n"
                        "If task.context.action_plan exists, treat it as a preferred workflow, but you may still choose the single next step only.\n"
                        "Do not repeat the same failed action with the same parameters after an identical error.\n"
                        "When the deliverable is complete or the success checks are likely satisfied, return finish.\n"
                        "Return strict JSON only with exactly one action: {'actions': [{'name': str, 'params': object, 'rationale': str}]}."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "objective": task.objective,
                            "context": task.context,
                            "success_checks": [check.model_dump(mode="json") for check in task.success_checks],
                            "memories": [memory.summary for memory in memories],
                            "recent_events": [_summarize_event(event) for event in trace.events[-8:]],
                            "current_outcome": outcome.model_dump(mode="json"),
                            "remaining_budget": {
                                "remaining_steps": max(task.budget.max_steps - trace.total_steps, 0),
                                "remaining_shell_calls": max(task.budget.max_shell_calls - trace.shell_calls, 0),
                                "remaining_http_calls": max(task.budget.max_http_calls - trace.http_calls, 0),
                            },
                            "workspace_root": task.permissions.workspace_root,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.app_name:
            headers["X-Title"] = self.app_name
        last_error: Exception | None = None
        for attempt in range(3):
            request = Request(url=f"{self.base_url}/chat/completions", data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
            try:
                with urlopen(request, timeout=60) as response:
                    data = json.loads(response.read().decode("utf-8"))
                break
            except URLError as exc:
                last_error = exc
                if attempt == 2:
                    raise ProviderError(f"Planner request failed after 3 attempts: {exc}") from exc
                time.sleep(1.5 * (attempt + 1))
        content = self._extract_json_content(data["choices"][0]["message"]["content"])
        parsed = json.loads(content)
        actions = [ToolAction.model_validate(item) for item in parsed["actions"]]
        if not actions:
            raise ProviderError("Planner returned no actions.")
        return [actions[0]]

    def judge(self, trace: RunTrace, outcome: TaskOutcome) -> tuple[bool, float, str]:
        return HeuristicProvider().judge(trace, outcome)

    @staticmethod
    def _extract_json_content(content: str) -> str:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if len(lines) >= 3:
                cleaned = "\n".join(lines[1:-1]).strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end >= start:
            cleaned = cleaned[start : end + 1]
        return cleaned


def build_provider(settings: Settings) -> HeuristicProvider | OpenAICompatibleProvider:
    if settings.provider == "openai_compatible":
        return OpenAICompatibleProvider(settings.openai_base_url, settings.openai_api_key, settings.model, settings.openai_site_url, settings.openai_app_name)
    return HeuristicProvider()


class AgentRuntime:
    def __init__(self, tools: ToolRouter):
        self.tools = tools

    def execute(
        self,
        task: TaskSpec,
        strategy: StrategyBundle,
        provider: HeuristicProvider | OpenAICompatibleProvider,
        memories: list[MemoryRecord],
        recorder: RunRecorder,
    ) -> tuple[RunTrace, TaskOutcome]:
        recorder.record(EventType.memory_retrieval, {"memory_ids": [memory.id for memory in memories], "count": len(memories)})
        outcome = TaskOutcome(status=RunStatus.running)
        artifacts: list[str] = []
        errors = 0
        repeated_errors: dict[str, int] = {}
        observations: list[str] = []
        planning_turn = 0
        while recorder.trace.total_steps < task.budget.max_steps:
            planning_turn += 1
            try:
                actions = provider.plan(task, strategy, memories, recorder.trace, outcome)
            except ProviderError as exc:
                outcome.error = str(exc)
                recorder.record(EventType.error, {"message": outcome.error, "turn": planning_turn, "phase": "plan"})
                break
            recorder.record(
                EventType.plan,
                {
                    "strategy_version": strategy.version,
                    "turn": planning_turn,
                    "actions": [action.model_dump(mode="json") for action in actions],
                },
            )
            if not actions:
                outcome.error = "Planner returned no actions."
                recorder.record(EventType.error, {"message": outcome.error, "turn": planning_turn})
                break
            action = actions[0]
            if action.name == "finish":
                outcome.final_output = str(action.params.get("message", ""))
                recorder.record(EventType.finish, {"message": outcome.final_output, "turn": planning_turn})
                break
            if action.name.startswith("shell") and recorder.trace.shell_calls >= task.budget.max_shell_calls:
                outcome.error = "Shell call budget exhausted."
                recorder.record(EventType.error, {"message": outcome.error, "action": action.name, "turn": planning_turn})
                break
            if action.name.startswith("http") and recorder.trace.http_calls >= task.budget.max_http_calls:
                outcome.error = "HTTP call budget exhausted."
                recorder.record(EventType.error, {"message": outcome.error, "action": action.name, "turn": planning_turn})
                break
            recorder.trace.total_steps += 1
            attempts = 0
            while True:
                try:
                    recorder.record(EventType.tool_call, {"action": action.model_dump(mode="json"), "attempt": attempts + 1, "turn": planning_turn})
                    result = self.tools.execute(task, action)
                    if action.name.startswith("shell"):
                        recorder.trace.shell_calls += 1
                    if action.name.startswith("http"):
                        recorder.trace.http_calls += 1
                    if result.artifact_path:
                        artifacts.append(result.artifact_path)
                    observations.append(result.output)
                    recorder.record(EventType.tool_result, {"action": action.name, "output": result.output, "artifact_path": result.artifact_path, "metadata": result.metadata, "turn": planning_turn})
                    recorder.record(EventType.reflection, {"observation": result.output[:200], "step": recorder.trace.total_steps, "turn": planning_turn})
                    break
                except (GuardrailError, ToolExecutionError, subprocess.TimeoutExpired) as exc:
                    errors += 1
                    message = str(exc)
                    repeated_errors[message] = repeated_errors.get(message, 0) + 1
                    recorder.record(EventType.error, {"action": action.name, "message": message, "attempt": attempts + 1, "turn": planning_turn})
                    if attempts < strategy.retry_budget and errors <= task.budget.max_errors and repeated_errors[message] < 2:
                        attempts += 1
                        recorder.record(EventType.retry, {"action": action.name, "attempt": attempts, "turn": planning_turn})
                        continue
                    if errors >= task.budget.max_errors:
                        outcome.error = f"Guardrail stop after {errors} errors: {message}"
                    break
            if outcome.error:
                break
        if not outcome.final_output:
            outcome.final_output = observations[-1] if observations else (outcome.error or "No output generated.")
        if recorder.trace.total_steps >= task.budget.max_steps and not any(event.type == EventType.finish for event in recorder.trace.events):
            outcome.error = outcome.error or "Step budget exhausted before finish."
        outcome.artifact_paths = list(dict.fromkeys(artifacts))
        outcome.result_summary = outcome.final_output[:240]
        outcome.status = RunStatus.failed if outcome.error else RunStatus.succeeded
        outcome.success = outcome.error is None
        recorder.trace.token_usage = sum(
            len(json.dumps(event.payload, ensure_ascii=False))
            for event in recorder.trace.events
        )
        recorder.trace.estimated_cost = round(0.0002 * recorder.trace.total_steps + 0.00005 * recorder.trace.http_calls + 0.00003 * recorder.trace.shell_calls, 6)
        recorder.trace.final_output = outcome.final_output
        recorder.trace.completed_at = utc_now()
        return recorder.trace, outcome












