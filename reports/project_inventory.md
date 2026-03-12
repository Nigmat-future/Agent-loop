# Project Inventory

## Purpose
一个本地单机的 Agent 自学习框架，提供：
- FastAPI API
- Typer CLI
- SQLite + FTS5 持久化
- 任务运行时 trace
- 自动评估、经验提炼、策略候选生成
- 影子评估后晋升与版本化回滚

## Dependencies
- setuptools >= 68
- wheel
- Python >= 3 (exact version requirement truncated in pyproject.toml read)

## Entrypoints
- CLI entrypoint: `agent-loop` (as seen in README usage examples)
- API entrypoint: FastAPI server (e.g., via `uvicorn agent_loop:app`)

