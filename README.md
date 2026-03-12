# Agent Loop

一个本地单机的 Agent 自学习框架，提供：

- `FastAPI` API
- `Typer` CLI
- `SQLite + FTS5` 持久化
- 任务运行时 trace
- 自动评估、经验提炼、策略候选生成
- 影子评估后晋升与版本化回滚

## 快速开始

```bash
python -m pip install -e .[dev]
agent-loop run-task examples/file_task.json
agent-loop list-strategies
```

启动 API：

```bash
uvicorn agent_loop.api:app --reload
```

默认会在工作目录下创建 `.agent_loop/agent_loop.db`。

## OpenRouter

使用 OpenRouter 作为 OpenAI-compatible provider：

```bash
$env:AGENT_LOOP_PROVIDER="openai_compatible"
$env:AGENT_LOOP_MODEL="openai/gpt-4.1-mini"
$env:AGENT_LOOP_OPENAI_BASE_URL="https://openrouter.ai/api/v1"
$env:AGENT_LOOP_OPENAI_API_KEY="<your-openrouter-key>"
$env:AGENT_LOOP_OPENAI_SITE_URL="https://openrouter.ai"
$env:AGENT_LOOP_OPENAI_APP_NAME="Agent Loop"
```

然后执行：

```bash
agent-loop run-task examples/real_task_smoke_report.json
```

默认 `agent-loop run-task` 会在产生候选策略后自动做 shadow replay，并在阈值满足时自动 promote。回放只使用候选策略出现之前的历史任务，而且必须相对当前 active 策略出现严格改进才会晋升；如果历史样本不足，策略会保持 `candidate`。可以用 `--no-auto-promote` 关闭。

