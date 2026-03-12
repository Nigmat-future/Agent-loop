# Agent Loop - Project Overview

**Version:** 0.1.0  
**Status:** Active Development  
**Language:** Python 3.11+  
**Framework:** FastAPI + Typer CLI  

---

## Table of Contents

1. [Project Mission](#project-mission)
2. [Architecture Overview](#architecture-overview)
3. [Core Concepts](#core-concepts)
4. [Key Components](#key-components)
5. [Data Model](#data-model)
6. [Workflow & Lifecycle](#workflow--lifecycle)
7. [API & CLI Reference](#api--cli-reference)
8. [Configuration](#configuration)
9. [Example Usage](#example-usage)

---

## Project Mission

**Agent Loop** is a **local, single-machine AI agent self-learning framework** that enables agents to autonomously:

- Execute tasks with runtime tracing
- Automatically evaluate performance
- Distill experiences into lessons
- Generate and test strategy candidates
- Promote successful strategies with version control and rollback

The framework provides both **REST API** (FastAPI) and **CLI** (Typer) interfaces for running tasks, managing strategies, and accessing memory.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Agent Loop Service                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Task Execution Pipeline                 │  │
│  │  ┌─────────────────────────────────────────────────┐ │  │
│  │  │ 1. Runtime Execution (AgentRuntime)            │ │  │
│  │  │    - Plan actions via Provider                 │ │  │
│  │  │    - Execute tools (file, shell, http)         │ │  │
│  │  │    - Record trace events                       │ │  │
│  │  └─────────────────────────────────────────────────┘ │  │
│  │  ┌─────────────────────────────────────────────────┐ │  │
│  │  │ 2. Evaluation (EvaluationService)              │ │  │
│  │  │    - Rule-based checks (file_exists, etc.)     │ │  │
│  │  │    - Heuristic judgment                        │ │  │
│  │  └─────────────────────────────────────────────────┘ │  │
│  │  ┌─────────────────────────────────────────────────┐ │  │
│  │  │ 3. Learning (LearningPipeline)                 │ │  │
│  │  │    - Extract memories (episodic, semantic)     │ │  │
│  │  │    - Derive lessons                            │ │  │
│  │  │    - Generate candidate strategies             │ │  │
│  │  └─────────────────────────────────────────────────┘ │  │
│  │  ┌─────────────────────────────────────────────────┐ │  │
│  │  │ 4. Promotion (Auto-Replay & Versioning)        │ │  │
│  │  │    - Shadow replay on historical tasks         │ │  │
│  │  │    - Threshold-based promotion                 │ │  │
│  │  │    - Version control & archival                │ │  │
│  │  └─────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │             Persistence & Retrieval                 │  │
│  │  - SQLite + FTS5 (full-text search)                 │  │
│  │  - Vector embeddings (hash-based, 32-dim)          │  │
│  │  - Event tracing                                    │  │
│  │  - Memory management (episodic/semantic/procedural) │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Core Concepts

### 1. **Task Specification (TaskSpec)**
A complete description of what the agent needs to accomplish:

```python
{
  "id": "task_abc123...",           # Unique identifier
  "task_type": "general",            # Categorizes task for strategy selection
  "objective": "Write a demo file",  # What to do
  "context": {                       # Structured hints & data
    "action_plan": [...],            # Optional: preferred action sequence
    "write_file": {...},             # Optional: structured tool inputs
    "final_output": "..."            # Optional: expected output hint
  },
  "success_checks": [                # Evaluation criteria
    {"kind": "file_exists", "target": "path/to/file"},
    {"kind": "file_contains", "target": "path", "expected": "text"}
  ],
  "budget": {                        # Execution limits
    "max_steps": 8,
    "max_shell_calls": 2,
    "max_http_calls": 2,
    "timeout_seconds": 30,
    "max_errors": 3
  },
  "permissions": {                   # Security guardrails
    "workspace_root": ".",           # Confine file operations
    "enable_file": true,
    "enable_shell": true,
    "enable_http": false,
    "allowed_shell_prefixes": ["python", "echo"],
    "allowed_http_domains": []
  }
}
```

### 2. **Strategy Bundle (StrategyBundle)**
A versioned configuration that guides how an agent approaches tasks of a given type:

```python
{
  "id": "strategy_xyz...",
  "version": 1,                      # Incremented on promotion
  "task_type": "general",
  "status": "active|candidate|archived",
  "source": "manual|default|learning:{run_id}",
  "system_prompt": "Act cautiously...",
  "planning_template": "Plan minimal steps...",
  "tool_weights": {                  # Influence planning
    "file": 1.0,
    "shell": 1.0,
    "http": 0.7
  },
  "retrieval_threshold": 0.2,        # Memory relevance cutoff
  "retry_budget": 1,                 # Attempts per failed action
  "decomposition_rules": [...]       # Heuristics for planning
}
```

**Strategy Lifecycle:**
- **Active**: Currently deployed; all new tasks use this
- **Candidate**: Generated by learning; awaits promotion decision
- **Archived**: Previous active strategies; kept for rollback

### 3. **Memory System (MemoryRecord)**
Persistent learning across runs using three types:

| Type | Use Case | Example |
|------|----------|---------|
| **Episodic** | Run outcomes & results | "Task X succeeded with N steps" |
| **Semantic** | Failure patterns & insights | "Shell commands must be in allowlist" |
| **Procedural** | Strategy configurations | Active strategy parameters |

Memories are:
- **Searchable** via vector embedding (32-dim cosine similarity) + FTS5 full-text search
- **Applicability-tagged** ("global" or specific task_type)
- **Relevant** when retrieved before planning (via `retrieval_threshold`)

### 4. **Run Trace (RunTrace)**
Complete execution log of a single task execution:

```python
{
  "run_id": "run_123...",
  "task_id": "task_abc...",
  "strategy_id": "strategy_xyz...",
  "events": [                        # Chronological trace
    {
      "type": "plan|tool_call|tool_result|error|reflection|...",
      "payload": {...}               # Event-specific data
    }
  ],
  "total_steps": 5,
  "shell_calls": 2,
  "http_calls": 0,
  "token_usage": 1024,
  "estimated_cost": 0.0015
}
```

**Event Types:**
- `plan` – Planning output
- `memory_retrieval` – Memories used
- `tool_call` – Action execution start
- `tool_result` – Action output
- `error` – Guardrail/execution failure
- `reflection` – Observation after tool use
- `retry` – Attempt count increment
- `evaluation` – Success/failure judgment
- `learning` – Lessons & strategy candidates
- `finish` – Task completion

### 5. **Evaluation Report (EvalReport)**
Judgment of task success:

```python
{
  "id": "eval_...",
  "run_id": "run_...",
  "mode": "rule|heuristic",
  "success": true/false,
  "confidence": 0.95,
  "summary": "3/3 rule checks passed",
  "failure_reason": "...",          # Why it failed (if failed)
  "metrics": {...}
}
```

**Evaluation Modes:**
1. **Rule-based** (if `task.success_checks` defined):
   - `file_exists` – Path exists check
   - `file_contains` – Content substring search
   - `final_output_contains` – Output text match
   - `http_status` – HTTP response status
   - `no_error_events` – No error events in trace
   - Success = all checks pass

2. **Heuristic** (fallback):
   - No errors AND non-empty output → success
   - Otherwise → failure

### 6. **Learning Result (LearningResult)**
Output of learning pipeline after task execution:

```python
{
  "run_id": "run_...",
  "memories": [...],                 # Extracted episodic/semantic memories
  "lessons": [...],                  # Actionable lessons derived
  "candidate_strategy": {...},       # New strategy to test (if applicable)
  "replay_report": {...},            # Shadow replay comparison (if promoted)
  "promotion_decision": {...}        # Approve/reject candidate
}
```

---

## Key Components

### SQLiteStore
**File:** `src/agent_loop/service.py:SQLiteStore`

Persistence layer using SQLite + FTS5:

**Tables:**
- `tasks` – Task specifications
- `runs` – Execution metadata
- `events` – Trace events
- `evaluations` – Success/failure reports
- `lessons` – Learned heuristics
- `strategies` – Strategy versions
- `promotion_decisions` – Strategy approval history
- `memories` – Episodic/semantic/procedural knowledge
- `memory_fts` – Full-text search index

**Key Methods:**
- `save_task()`, `get_task()`, `list_tasks()`
- `create_run()`, `append_event()`, `finalize_run()`
- `get_run_trace()`, `get_outcome()`
- `save_strategy()`, `get_active_strategy()`, `list_strategies()`
- `search_memories()` – Vector + FTS hybrid search
- `save_memory()`, `list_memories_by_run()`

### Guardrail (Security)
**File:** `src/agent_loop/service.py:Guardrail`

Enforces workspace isolation and command restrictions:

**Checks:**
- **Path escaping**: Ensures all file operations stay within `workspace_root`
- **Shell allowlist**: Only whitelisted binaries (e.g., "python", "echo", "dir", "type")
- **Banned fragments**: Blocks dangerous commands ("rm", "del", "format", "rmdir /s", "shutdown")
- **HTTP domains**: Blocks non-whitelisted hosts

**Methods:**
- `check_shell()` – Validates shell commands
- `check_http()` – Validates HTTP URLs
- `resolve_path()` – Ensures path stays in workspace

### ToolRouter (Execution)
**File:** `src/agent_loop/service.py:ToolRouter`

Executes agent actions with guardrail enforcement:

**Supported Tools:**
1. **file.write** – Write content to file
2. **file.append** – Append to existing file
3. **file.read** – Read file contents
4. **shell.run** – Execute shell command
5. **http.get / http.request** – HTTP requests
6. **finish** – Mark task complete

Returns `ToolResult` with success flag, output string, optional artifact path.

### Provider (Planning)
**File:** `src/agent_loop/service.py:HeuristicProvider & OpenAICompatibleProvider`

Generates action plans given task & memories:

**Heuristic Provider:**
- Uses structured `context.action_plan` if available
- Falls back to structured tool hints
- Deterministic, no LLM calls

**OpenAI-Compatible Provider:**
- Calls LLM (OpenRouter, OpenAI, etc.) via HTTP
- Follows system prompt + planning template
- Returns JSON with action array
- Includes memory context in prompt

### AgentRuntime (Execution)
**File:** `src/agent_loop/service.py:AgentRuntime`

Core execution loop:

1. Retrieve relevant memories
2. Generate action plan via Provider
3. For each action:
   - Execute via ToolRouter
   - Record event
   - Retry on error (up to `strategy.retry_budget`)
4. Check termination:
   - Finish action invoked
   - Step budget exhausted
   - Error budget exhausted
   - Shell/HTTP call limit hit
5. Set outcome (success/failed) based on errors
6. Return trace + outcome

### EvaluationService
**File:** `src/agent_loop/service.py:EvaluationService`

Judges task success:

- Runs rule checks if `task.success_checks` defined
- Falls back to heuristic judgment
- Returns `EvalReport` with confidence & reason

### LearningPipeline
**File:** `src/agent_loop/service.py:LearningPipeline`

Extracts insights and generates candidate strategies:

**Steps:**
1. **Write memories**: Extract episodic (outcome) + semantic (failure patterns)
2. **Derive lessons**: Generic insights ("avoid failing action path", "prefer shorter plans")
3. **Generate candidate**: Modify strategy based on lessons
   - Lower tool weights if dangerous commands failed
   - Increase retrieval threshold if no memories helped
   - Adjust retry budget based on error patterns
   - Add decomposition rules

### StrategyRegistry
**File:** `src/agent_loop/service.py:StrategyRegistry`

Manages strategy versioning and promotion:

- `ensure_active()` – Get/create active strategy for task type
- `create_candidate()` – Branch strategy from parent
- `promote()` – Activate candidate, archive previous active
- `next_strategy_version()` – Increment version number

### AgentLoopService (Main Orchestrator)
**File:** `src/agent_loop/service.py:AgentLoopService`

High-level API orchestrating all components:

```python
service = AgentLoopService(settings)

# Run a task
result = service.run_task(task, auto_learn=True, auto_promote=False)

# Replay strategy on history
replay_report = service.replay_strategy(strategy_id, limit=5)

# Promote a candidate
promotion = service.promote_strategy(strategy_id, limit=5)

# Access memory
memories = service.list_memories(memory_type="episodic", limit=50)
memories = service.search_memories(query="...", limit=5)

# List strategies
strategies = service.list_strategies(task_type="general", status="active")
```

**Key Methods:**
- `run_task()` – Full execution pipeline
- `inspect_run()` – Get run details
- `learn_from_run()` – Re-evaluate & learn from past run
- `replay_strategy()` – Test candidate on historical tasks
- `promote_strategy()` – Promote after successful replay
- `list_memories()`, `search_memories()`
- `list_strategies()`

---

## Data Model

### Enums

```python
# Task Execution
class RunStatus: pending, running, succeeded, failed, cancelled

# Strategy Versioning
class StrategyStatus: active, candidate, archived

# Memory Types
class MemoryType: episodic, semantic, procedural

# Trace Events
class EventType: plan, memory_retrieval, tool_call, tool_result, retry,
                 error, reflection, finish, evaluation, learning

# Success Check Kinds
class SuccessCheckKind: file_exists, file_contains, final_output_contains,
                        http_status, no_error_events
```

### Pydantic Models

All models use **Pydantic v2** with `BaseModel`:

- `TaskSpec` – Task definition
- `StrategyBundle` – Strategy version
- `RunTrace` – Execution log
- `TaskOutcome` – Success/failure status
- `EvalReport` – Evaluation result
- `LessonCandidate` – Learned heuristic
- `MemoryRecord` – Knowledge item
- `RunResult` – Complete result (task + trace + outcome + eval + learning)
- `RunInspection` – Past run summary
- `ReplayReport` – Strategy comparison
- `PromotionDecision` – Approval record

---

## Workflow & Lifecycle

### Single Task Execution

```
1. Load task spec
   ↓
2. Ensure active strategy for task.task_type
   ↓
3. Search memories (query: objective + context, threshold: strategy.retrieval_threshold)
   ↓
4. RUNTIME EXECUTION
   ├─ Provider.plan(task, strategy, memories) → actions
   ├─ For each action:
   │  ├─ Guardrail.check_*() – Security validation
   │  ├─ ToolRouter.execute() – Run action
   │  ├─ Recorder.record() – Log event
   │  └─ Retry on error (up to strategy.retry_budget)
   └─ Build outcome (success=no errors)
   ↓
5. EVALUATION
   ├─ If task.success_checks: Rule-based checks
   └─ Else: Heuristic judgment
   ↓
6. LEARNING (if auto_learn=True)
   ├─ Write memories (episodic outcome, semantic failures)
   ├─ Derive lessons
   ├─ Create candidate strategy
   └─ Record learning event
   ↓
7. PROMOTION (if auto_promote=True & candidate exists)
   ├─ replay_strategy(candidate_id, limit=replay_limit)
   │  └─ For each recent task of same type:
   │     ├─ Re-run with candidate strategy
   │     └─ Compare success rate, cost, steps
   ├─ If candidate improves: Promote to active, archive old
   └─ Record decision
   ↓
8. Return RunResult
```

### Memory Retrieval

```
Query = task.objective + "\n" + json.dumps(task.context)
Threshold = strategy.retrieval_threshold (default 0.2)

Search process:
1. Hash query into 32-dim vector
2. FTS search on tokenized query (6 tokens max, OR'd)
3. For each candidate memory:
   ├─ Check applicability (must be "global" or task.task_type)
   ├─ Compute cosine_similarity(query_embedding, memory.embedding)
   ├─ Add +0.2 bonus if FTS matched
   └─ Keep if score >= threshold
4. Sort by score descending
5. Return top N memories (default 5)
```

### Strategy Promotion

```
Candidate generated by learning if:
- Lessons extracted from failed run, OR
- Run consumed most of step budget

Promotion requires shadow replay:
1. Get all tasks of candidate.task_type
2. Filter: created before candidate's created_at
3. Limit: most recent N (default 3) successful runs
4. Re-execute each with candidate vs. active:
   ├─ Success rate comparison
   ├─ Average steps comparison
   ├─ Average cost comparison
   └─ Error rate comparison
5. Promote if:
   - candidate success_rate >= baseline, AND
   - candidate_steps <= baseline, AND
   - candidate_cost <= baseline
6. Archive active, set candidate.status = "active"
```

---

## API & CLI Reference

### FastAPI Endpoints

**Base URL:** `http://localhost:8000` (default)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/tasks/run` | Execute task (auto-learn, auto-promote) |
| GET | `/runs/{run_id}` | Inspect run |
| POST | `/runs/{run_id}/learn` | Re-evaluate & learn from run |
| POST | `/strategies/{strategy_id}/replay` | Shadow replay on history |
| POST | `/strategies/{strategy_id}/promote` | Promote after successful replay |
| GET | `/memories` | List memories |
| GET | `/strategies` | List strategies |

### CLI Commands

```bash
# Run a task
agent-loop run-task examples/file_task.json \
  [--auto-learn] [--auto-promote] [--replay-limit 3] [--db-path PATH]

# Inspect a run
agent-loop inspect-run run_abc123... [--db-path PATH]

# Re-evaluate & learn
agent-loop learn-from-run run_abc123... [--db-path PATH]

# Test candidate strategy
agent-loop replay-strategy strategy_xyz... [--limit 5] [--db-path PATH]

# Promote candidate to active
agent-loop promote-strategy strategy_xyz... [--limit 5] [--db-path PATH]

# List memories
agent-loop list-memories [--memory-type episodic|semantic|procedural] [--limit 50]

# List strategies
agent-loop list-strategies [--task-type general] [--status active|candidate|archived]

# Start API server
agent-loop serve [--host 127.0.0.1] [--port 8000] [--db-path PATH]
```

---

## Configuration

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `AGENT_LOOP_HOME` | `./.agent_loop` | Home directory (stores DB, config) |
| `AGENT_LOOP_DB_PATH` | `.agent_loop/agent_loop.db` | Custom SQLite path |
| `AGENT_LOOP_PROVIDER` | `heuristic` | Planning provider (`heuristic` or `openai_compatible`) |
| `AGENT_LOOP_MODEL` | `gpt-4o-mini` | Model name (for OpenAI provider) |
| `AGENT_LOOP_OPENAI_BASE_URL` | `https://api.openai.com/v1` | API endpoint |
| `AGENT_LOOP_OPENAI_API_KEY` | _(required for LLM)_ | API key |
| `AGENT_LOOP_OPENAI_SITE_URL` | `null` | Referer header (e.g., for OpenRouter) |
| `AGENT_LOOP_OPENAI_APP_NAME` | `Agent Loop` | App name header |
| `AGENT_LOOP_WORKSPACE_ROOT` | `.` | Default workspace for file operations |
| `AGENT_LOOP_ALLOWED_HTTP_DOMAINS` | `` | Comma-separated HTTP domains |
| `AGENT_LOOP_ALLOWED_SHELL_PREFIXES` | `python,echo,dir,type` | Comma-separated shell commands |

### Settings (Python)

```python
from agent_loop.settings import Settings

settings = Settings(
    home_dir=Path(".agent_loop"),
    db_path=None,                    # Auto-detect if None
    provider="openai_compatible",    # or "heuristic"
    model="openai/gpt-4-turbo",
    openai_base_url="https://openrouter.ai/api/v1",
    openai_api_key="sk-...",
    openai_site_url="https://openrouter.ai",
    default_workspace_root=Path("."),
    default_http_domains=["api.example.com"],
    default_shell_prefixes=["python", "bash"]
)

service = AgentLoopService(settings)
```

---

## Example Usage

### 1. Simple File Write Task (Heuristic Provider)

**File:** `examples/file_task.json`

```json
{
  "task_type": "general",
  "objective": "Write a demo file into the workspace.",
  "context": {
    "action_plan": [
      {
        "name": "file.write",
        "params": {"path": "artifacts/demo.txt", "content": "hello from agent loop"}
      }
    ],
    "final_output": "demo file created"
  },
  "success_checks": [
    {"kind": "file_exists", "target": "artifacts/demo.txt"},
    {"kind": "file_contains", "target": "artifacts/demo.txt", "expected": "hello from agent loop"}
  ],
  "permissions": {
    "workspace_root": ".",
    "enable_file": true,
    "enable_shell": true,
    "enable_http": false,
    "allowed_shell_prefixes": ["python", "echo"],
    "allowed_http_domains": []
  }
}
```

**Execution:**
```bash
agent-loop run-task examples/file_task.json --auto-learn --auto-promote
```

**Result:**
- Heuristic provider extracts `action_plan` from context
- ToolRouter executes `file.write` action
- Guardrail allows write (in workspace)
- Evaluation checks: file exists ✓, contains expected text ✓
- Learning: episodic memory saved (success), no lessons (succeeded quickly)
- Promotion: N/A (success = no candidates generated)

### 2. Shell Task with OpenAI Provider

**Setup:**
```bash
$env:AGENT_LOOP_PROVIDER="openai_compatible"
$env:AGENT_LOOP_MODEL="openai/gpt-4-turbo"
$env:AGENT_LOOP_OPENAI_BASE_URL="https://openrouter.ai/api/v1"
$env:AGENT_LOOP_OPENAI_API_KEY="sk-..."
```

**File:** `examples/real_task_smoke_test.json`

```json
{
  "task_type": "general",
  "objective": "Run the project's regression tests as a smoke check.",
  "context": {
    "action_plan": [
      {"name": "shell.run", "params": {"command": "python -m pytest -q"}}
    ],
    "final_output": "project smoke test completed"
  },
  "success_checks": [
    {"kind": "no_error_events"},
    {"kind": "final_output_contains", "expected": "project smoke test completed"}
  ]
}
```

**Execution:**
```bash
agent-loop run-task examples/real_task_smoke_test.json --auto-learn --auto-promote
```

### 3. Task Without Structured Hints

```json
{
  "task_type": "report_gen",
  "objective": "Generate a performance report by analyzing logs and writing to reports/perf.md",
  "context": {
    "data_file": "logs/metrics.csv",
    "output_file": "reports/perf.md"
  },
  "success_checks": [
    {"kind": "file_exists", "target": "reports/perf.md"},
    {"kind": "file_contains", "target": "reports/perf.md", "expected": "Performance"}
  ]
}
```

**Execution with LLM Planning:**
1. Provider receives task (no action_plan in context)
2. LLM reads objective + context
3. LLM generates:
   - Read logs
   - Process metrics
   - Write report
   - Finish
4. Each action executed with tracing
5. Success evaluated via file checks
6. Learning + promotion as usual

### 4. Inspecting & Learning from Runs

```bash
# Inspect a run
agent-loop inspect-run run_abc123

# Re-learn from run (useful if evaluation changed)
agent-loop learn-from-run run_abc123

# List memories extracted
agent-loop list-memories --memory-type episodic

# List all strategies
agent-loop list-strategies

# Replay candidate on recent tasks
agent-loop replay-strategy strategy_xyz --limit 5

# Promote if confident
agent-loop promote-strategy strategy_xyz --limit 5
```

---

## Key Implementation Details

### Vector Embeddings

**File:** `src/agent_loop/vector.py`

- **Dimension:** 32
- **Method:** Hash-based (SHA256)
- **Similarity:** Cosine distance
- **Normalization:** Unit vector

```python
embedding = hash_embedding(text)  # 32-dim vector
score = cosine_similarity(query_vec, memory_vec)  # [0.0, 1.0]
```

### Error Handling

**Guardrail Errors:**
- Path escapes workspace → `GuardrailError`
- Shell command not whitelisted → `GuardrailError`
- HTTP domain blocked → `GuardrailError`

**Execution Errors:**
- File not found → `ToolExecutionError`
- Shell exit code != 0 → `ToolExecutionError`
- HTTP status >= 400 → `ToolExecutionError`

**Provider Errors:**
- LLM API unreachable → `ProviderError` (retried 3x)
- Invalid JSON response → `ProviderError`

All errors are:
1. Recorded in trace as `EventType.error`
2. Retried (up to `strategy.retry_budget`)
3. Counted against `budget.max_errors`
4. Factored into final outcome

### Cost Estimation

```python
estimated_cost = 0.0002 * total_steps + 0.00005 * http_calls + 0.00003 * shell_calls
```

Simple approximation for budget tracking (not actual API cost).

### Task Normalization

Before execution, `AgentLoopService.run_task()` normalizes:
- Ensures task has valid `id` (generates if missing)
- Ensures `created_at` is set
- Validates budget constraints

---

## Project Structure

```
agent-loop/
├── src/agent_loop/
│   ├── __init__.py
│   ├── models.py           # Pydantic data models
│   ├── service.py          # Core logic (850+ lines)
│   ├── settings.py         # Configuration
│   ├── vector.py           # Embeddings
│   ├── cli.py              # Typer CLI
│   ├── api.py              # FastAPI routes
│   └── benchmarks.py       # (Benchmarking utilities)
├── tests/
│   ├── test_interfaces.py
│   └── test_service.py
├── examples/
│   ├── file_task.json
│   ├── real_task_smoke_test.json
│   ├── real_task_smoke_report.json
│   └── (other examples)
├── pyproject.toml          # Package config
├── README.md               # Quick start
└── .env.openrouter.example  # Config template
```

---

## Quick Start

### Installation

```bash
# From project root
python -m pip install -e .[dev]
```

### Run a Simple Task

```bash
agent-loop run-task examples/file_task.json
```

### Start the API

```bash
uvicorn agent_loop.api:app --reload
```

Then visit `http://localhost:8000/docs` for interactive API docs.

### Run Tests

```bash
pytest -xvs tests/
```

---

## Design Decisions

### Why Heuristic Provider?
- No LLM dependency required
- Deterministic, fast
- Useful for structured tasks with clear action plans
- Falls back to context hints

### Why Hash-Based Embeddings?
- Lightweight (no ML model needed)
- Works offline
- Sufficient for lexical relevance
- FTS5 + vector hybrid search is robust

### Why SQLite + FTS5?
- Single file, no server
- Full-text search out-of-box
- Sufficient for local agent
- Easy to backup/version

### Why Guardrails?
- Prevents accidental filesystem damage
- Blocks dangerous commands
- Localizes HTTP access
- Makes agent loop safe for automation

### Why Shadow Replay?
- Validates improvement before promotion
- Prevents regressive strategies
- Uses only historical tasks (no retraining data loss)
- Strict improvement threshold (candidate must win)

---

## Known Limitations & Future Work

1. **Embeddings**: Hash-based is lightweight but less semantic than ML embeddings
2. **Strategy Generation**: Heuristic rules are hand-crafted; could be more sophisticated
3. **Multi-turn Tasks**: Current design is single-execution; multi-round tasks not directly supported
4. **Parallelism**: Sequential execution; no concurrent task runs
5. **Scalability**: SQLite suitable for local agent; would need migration for multi-agent scenario

---

## Glossary

| Term | Definition |
|------|-----------|
| **TaskSpec** | Complete task definition (objective + budget + permissions + checks) |
| **StrategyBundle** | Versioned agent configuration (prompts, weights, rules) |
| **RunTrace** | Chronological log of all events during task execution |
| **TaskOutcome** | Result of execution (success, final output, errors) |
| **EvalReport** | Success judgment based on rules or heuristics |
| **MemoryRecord** | Single knowledge item (episodic, semantic, or procedural) |
| **RunRecorder** | In-memory event logger during execution |
| **Guardrail** | Security boundary (path, command, HTTP constraints) |
| **Provider** | Planner backend (Heuristic or LLM-based) |
| **AgentRuntime** | Core execution loop |
| **LearningPipeline** | Post-execution analysis & strategy generation |
| **StrategyRegistry** | Version control & promotion logic |

---

## Document Metadata

- **Last Updated:** 2026-03-12
- **Scope:** Project overview for AI comprehension
- **Audience:** Developers, AI agents, architects
- **Depth:** Complete architecture + implementation details
