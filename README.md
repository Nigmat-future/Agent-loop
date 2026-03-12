# 🤖 Agent Loop

<p align="center">
  <b>本地单机 · 自主学习 · 策略进化 · 零云依赖</b><br/>
  一个让 AI Agent 能够从每次任务执行中持续学习、自我优化的本地自学习框架
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11%2B-blue?logo=python"/>
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.115%2B-009688?logo=fastapi"/>
  <img alt="SQLite" src="https://img.shields.io/badge/Storage-SQLite%20%2B%20FTS5-003B57?logo=sqlite"/>
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green"/>
  <img alt="Version" src="https://img.shields.io/badge/Version-0.1.0-orange"/>
</p>

---

## ✨ 为什么选择 Agent Loop？

大多数 AI Agent 框架存在一个根本性缺陷——**它们不会从经验中学习**。每次执行都从零开始，相同的错误反复发生，没有持续改进的机制。

**Agent Loop 彻底改变了这一现状**，它构建了一套完整的「执行 → 评估 → 学习 → 进化」闭环：

| 特性 | 传统 Agent 框架 | **Agent Loop** |
|------|---------------|----------------|
| 经验积累 | ❌ 无持久记忆 | ✅ 三类记忆系统（情节/语义/程序性） |
| 策略优化 | ❌ 固定提示词 | ✅ 自动生成候选策略并量化验证 |
| 策略安全 | ❌ 直接替换 | ✅ 影子回放对比，严格优于才晋升 |
| 版本控制 | ❌ 无法回滚 | ✅ 完整版本化，随时回滚 |
| 可观测性 | ❌ 黑盒执行 | ✅ 全链路 Trace，每步有据可查 |
| 云依赖 | ⚠️ 通常需要 | ✅ 完全本地运行，零云依赖 |
| 接口 | 单一 | ✅ REST API + CLI 双接口 |
| 安全防护 | ❌ 无沙箱 | ✅ 工作空间隔离 + 命令白名单 |

---

## 🏗️ 架构总览

```
┌──────────────────────────────────────────────────────────────────┐
│                        Agent Loop Service                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│   ┌──────────────────────────────────────────────────────────┐   │
│   │                   任务执行流水线                           │   │
│   │                                                            │   │
│   │  ① AgentRuntime（执行）                                   │   │
│   │     ├─ Provider 规划动作序列（支持 LLM / 启发式）         │   │
│   │     ├─ ToolRouter 执行工具（file / shell / http）         │   │
│   │     ├─ Guardrail 安全卫士（路径/命令/域名拦截）           │   │
│   │     └─ 完整 Trace 记录每个步骤                           │   │
│   │                                                            │   │
│   │  ② EvaluationService（评估）                              │   │
│   │     ├─ 规则评估：file_exists / file_contains / http_status│   │
│   │     └─ 启发式评估：兜底判断                               │   │
│   │                                                            │   │
│   │  ③ LearningPipeline（学习）                               │   │
│   │     ├─ 写入情节记忆（本次执行结果）                       │   │
│   │     ├─ 提炼语义记忆（失败模式、规律洞察）                 │   │
│   │     ├─ 导出经验教训（Lessons）                            │   │
│   │     └─ 生成候选策略（Candidate Strategy）                 │   │
│   │                                                            │   │
│   │  ④ StrategyRegistry（进化）                               │   │
│   │     ├─ 影子回放：在历史任务上对比候选 vs 现行策略         │   │
│   │     ├─ 严格门槛：成功率↑ & 步数↓ & 成本↓ 才晋升          │   │
│   │     ├─ 自动晋升：archive 旧策略，activate 新策略          │   │
│   │     └─ 完整版本链，支持随时回滚                           │   │
│   └──────────────────────────────────────────────────────────┘   │
│                                                                    │
│   ┌──────────────────────────────────────────────────────────┐   │
│   │               持久化与检索（SQLite + FTS5）               │   │
│   │   向量嵌入（32维余弦相似度）+ 全文搜索混合检索            │   │
│   │   tables: tasks / runs / events / evaluations /           │   │
│   │           lessons / strategies / memories / promotions    │   │
│   └──────────────────────────────────────────────────────────┘   │
│                                                                    │
│             REST API (FastAPI)  ⟷  CLI (Typer)                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🧠 核心概念

### 1. 自进化策略系统（Strategy Bundle）

策略不是静态的提示词，而是一个**可版本化、可量化对比的完整配置包**：

```json
{
  "id": "strategy_xyz...",
  "version": 3,
  "task_type": "general",
  "status": "active",
  "system_prompt": "Act cautiously, prefer deterministic tools...",
  "planning_template": "Plan minimal steps, validate constraints...",
  "tool_weights": { "file": 1.0, "shell": 0.8, "http": 0.5 },
  "retrieval_threshold": 0.25,
  "retry_budget": 2,
  "decomposition_rules": [
    "Prefer structured actions from task context when available.",
    "Use shell only for commands inside the allowlist."
  ]
}
```

策略状态机：`candidate` → `active` → `archived`（可回滚至任意历史版本）

### 2. 三层记忆体系（Memory System）

| 记忆类型 | 存储内容 | 检索方式 |
|---------|---------|---------|
| **情节记忆** (Episodic) | 每次执行的结果、成本、步骤数 | 向量 + 全文 |
| **语义记忆** (Semantic) | 失败模式、规律性洞察 | 向量 + 全文 |
| **程序性记忆** (Procedural) | 当前生效的策略配置参数 | 向量 + 全文 |

检索采用**混合搜索**：32维哈希向量余弦相似度 + FTS5 全文搜索，FTS 命中额外 +0.2 加权，确保精准召回。

### 3. 影子回放机制（Shadow Replay）

候选策略在晋升前，必须通过**历史任务回放对比**：

```
候选策略晋升条件（同时满足）：
  ① 成功率 ≥ 现行策略
  ② 平均步数 ≤ 现行策略
  ③ 平均成本 ≤ 现行策略

注意：
  - 只使用候选策略生成时间之前的历史任务（防数据泄露）
  - 历史样本不足时，策略保持 candidate 状态，不强制晋升
  - 整个过程全自动，无需人工干预
```

### 4. 全链路可观测性（Run Trace）

每次任务执行产生完整 Trace，包含：

| 事件类型 | 说明 |
|---------|-----|
| `plan` | 规划输出 |
| `memory_retrieval` | 检索到的相关记忆 |
| `tool_call` / `tool_result` | 工具调用及其输出 |
| `retry` | 重试记录 |
| `error` | 拦截/执行错误 |
| `reflection` | 工具执行后的观察 |
| `evaluation` | 成功/失败判定 |
| `learning` | 提炼的教训与候选策略 |
| `finish` | 任务完成 |

### 5. 安全防护（Guardrail）

- **路径沙箱**：所有文件操作严格限制在 `workspace_root` 内，防止路径逃逸
- **命令白名单**：只允许执行预设前缀的 shell 命令（默认：`python`, `echo`, `dir`, `type`）
- **危险命令拦截**：自动屏蔽 `rm`, `del`, `format`, `rmdir /s`, `shutdown` 等破坏性操作
- **HTTP 域名白名单**：只允许访问预授权域名

---

## 🚀 快速开始

### 安装

```bash
# 克隆并安装（开发模式）
git clone https://github.com/Nigmat-future/Agent-loop.git
cd Agent-loop
python -m pip install -e .[dev]
```

### 运行第一个任务（无需 LLM，开箱即用）

```bash
# 使用内置示例任务（启发式 Provider，无需 API Key）
agent-loop run-task examples/file_task.json

# 查看已有策略
agent-loop list-strategies

# 查看记忆库
agent-loop list-memories
```

### 启动 REST API 服务

```bash
# 方式一：uvicorn 直接启动
uvicorn agent_loop.api:app --reload

# 方式二：CLI 启动（支持自定义参数）
agent-loop serve --host 0.0.0.0 --port 8000
```

访问 `http://localhost:8000/docs` 即可查看 Swagger 交互文档。

> 默认数据库路径：`.agent_loop/agent_loop.db`（可通过 `--db-path` 或环境变量覆盖）

---

## 🔌 接入 LLM（OpenRouter / OpenAI）

### Linux / macOS

```bash
export AGENT_LOOP_PROVIDER="openai_compatible"
export AGENT_LOOP_MODEL="openai/gpt-4.1-mini"
export AGENT_LOOP_OPENAI_BASE_URL="https://openrouter.ai/api/v1"
export AGENT_LOOP_OPENAI_API_KEY="<your-openrouter-key>"
export AGENT_LOOP_OPENAI_SITE_URL="https://openrouter.ai"
export AGENT_LOOP_OPENAI_APP_NAME="Agent Loop"
```

### Windows (PowerShell)

```powershell
$env:AGENT_LOOP_PROVIDER="openai_compatible"
$env:AGENT_LOOP_MODEL="openai/gpt-4.1-mini"
$env:AGENT_LOOP_OPENAI_BASE_URL="https://openrouter.ai/api/v1"
$env:AGENT_LOOP_OPENAI_API_KEY="<your-openrouter-key>"
$env:AGENT_LOOP_OPENAI_SITE_URL="https://openrouter.ai"
$env:AGENT_LOOP_OPENAI_APP_NAME="Agent Loop"
```

设置完成后运行真实 LLM 任务：

```bash
agent-loop run-task examples/real_task_smoke_report.json
```

> **提示**：不配置 LLM 时，框架自动使用内置启发式 Provider，适合测试和确定性场景。

---

## 📋 任务规格（TaskSpec）

任务以 JSON 文件描述，支持精细的预算控制和安全权限配置：

```json
{
  "task_type": "general",
  "objective": "生成项目 smoke test 报告",
  "context": {
    "action_plan": [
      {
        "name": "shell.run",
        "params": { "command": "python -m pytest -q" }
      }
    ],
    "final_output": "测试报告已生成"
  },
  "success_checks": [
    { "kind": "file_exists",    "target": "reports/smoke_report.md" },
    { "kind": "file_contains",  "target": "reports/smoke_report.md", "expected": "Status: PASS" }
  ],
  "budget": {
    "max_steps": 8,
    "max_shell_calls": 3,
    "max_http_calls": 0,
    "timeout_seconds": 60,
    "max_errors": 2
  },
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

**支持的成功检查类型：**

| 检查类型 | 说明 |
|---------|-----|
| `file_exists` | 指定路径文件存在 |
| `file_contains` | 文件内容包含指定字符串 |
| `final_output_contains` | 最终输出包含指定文本 |
| `http_status` | HTTP 响应状态码匹配 |
| `no_error_events` | Trace 中无错误事件 |

---

## 🖥️ CLI 命令参考

```bash
# ── 任务执行 ──────────────────────────────────────────────────
agent-loop run-task <task.json>          # 执行任务（默认开启自动学习 + 自动晋升）
  [--no-auto-learn]                       # 关闭自动学习
  [--no-auto-promote]                     # 关闭自动晋升（保持候选状态）
  [--replay-limit 3]                      # 影子回放使用的历史任务数量
  [--db-path PATH]                        # 自定义数据库路径

# ── 运行检查 ──────────────────────────────────────────────────
agent-loop inspect-run <run_id>          # 查看某次运行的完整详情
agent-loop learn-from-run <run_id>       # 对历史运行重新触发学习流程

# ── 策略管理 ──────────────────────────────────────────────────
agent-loop list-strategies               # 列出所有策略
  [--task-type general]                   # 按任务类型过滤
  [--status active|candidate|archived]    # 按状态过滤
agent-loop replay-strategy <strategy_id> # 对历史任务回放候选策略
agent-loop promote-strategy <strategy_id># 手动晋升候选策略

# ── 记忆管理 ──────────────────────────────────────────────────
agent-loop list-memories                 # 列出所有记忆
  [--memory-type episodic|semantic|procedural]
  [--limit 50]

# ── 服务启动 ──────────────────────────────────────────────────
agent-loop serve                         # 启动 REST API 服务
  [--host 127.0.0.1] [--port 8000]
```

---

## 🌐 REST API 参考

启动服务后访问 `http://localhost:8000/docs` 查看完整交互文档。

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/tasks/run` | 执行任务（支持 auto_learn / auto_promote 参数） |
| `GET`  | `/runs/{run_id}` | 查看运行详情（含 Trace、评估、教训） |
| `POST` | `/runs/{run_id}/learn` | 对历史运行重新触发学习 |
| `POST` | `/strategies/{id}/replay` | 影子回放（对比候选 vs 现行策略） |
| `POST` | `/strategies/{id}/promote` | 晋升候选策略为活跃策略 |
| `GET`  | `/memories` | 获取记忆列表（支持类型过滤） |
| `GET`  | `/strategies` | 获取策略列表（支持类型/状态过滤） |

**调用示例：**

```bash
# 通过 API 执行任务
curl -X POST http://localhost:8000/tasks/run \
  -H "Content-Type: application/json" \
  -d @examples/file_task.json

# 查询记忆库
curl "http://localhost:8000/memories?memory_type=semantic&limit=10"

# 列出候选策略
curl "http://localhost:8000/strategies?status=candidate"
```

---

## ⚙️ 配置参考

所有配置均可通过环境变量设置：

| 环境变量 | 默认值 | 说明 |
|---------|-------|------|
| `AGENT_LOOP_HOME` | `./.agent_loop` | 数据目录（存放 DB 等） |
| `AGENT_LOOP_DB_PATH` | `.agent_loop/agent_loop.db` | SQLite 数据库路径 |
| `AGENT_LOOP_PROVIDER` | `heuristic` | 规划 Provider：`heuristic` 或 `openai_compatible` |
| `AGENT_LOOP_MODEL` | `gpt-4o-mini` | 使用的 LLM 模型名称 |
| `AGENT_LOOP_OPENAI_BASE_URL` | `https://api.openai.com/v1` | API 端点（兼容 OpenRouter 等） |
| `AGENT_LOOP_OPENAI_API_KEY` | _(LLM 必填)_ | API 密钥 |
| `AGENT_LOOP_OPENAI_SITE_URL` | `null` | Referer 请求头（OpenRouter 场景） |
| `AGENT_LOOP_OPENAI_APP_NAME` | `Agent Loop` | 应用名称请求头 |
| `AGENT_LOOP_WORKSPACE_ROOT` | `.` | 文件操作工作空间根目录 |
| `AGENT_LOOP_ALLOWED_HTTP_DOMAINS` | _(空)_ | 允许访问的 HTTP 域名（逗号分隔） |
| `AGENT_LOOP_ALLOWED_SHELL_PREFIXES` | `python,echo,dir,type` | 允许执行的 shell 命令前缀 |

---

## 🔄 完整执行生命周期

```
① 加载任务规格（TaskSpec）
   ↓
② 获取/创建当前活跃策略（StrategyBundle）
   ↓
③ 检索相关记忆（向量 + FTS5 混合搜索）
   ↓
④ 规划动作序列（LLM Provider 或启发式 Provider）
   ↓
⑤ 逐步执行动作
   ├─ Guardrail 安全检查（路径/命令/域名）
   ├─ ToolRouter 执行（file / shell / http）
   ├─ 记录 Trace 事件
   └─ 错误重试（受 retry_budget 限制）
   ↓
⑥ 评估结果（规则检查 → 启发式兜底）
   ↓
⑦ 学习提炼（auto_learn=True 时）
   ├─ 写入情节记忆（执行结果）
   ├─ 提炼语义记忆（失败模式）
   ├─ 导出经验教训（Lessons）
   └─ 生成候选策略（Candidate）
   ↓
⑧ 策略晋升（auto_promote=True 且有候选时）
   ├─ 在历史任务上影子回放（candidate vs active）
   ├─ 量化对比（成功率 / 步数 / 成本 / 错误率）
   ├─ 严格优于才晋升，否则保留 candidate
   └─ 记录晋升决策
   ↓
⑨ 返回 RunResult（含 trace / outcome / eval / learning）
```

---

## 🗂️ 项目结构

```
agent-loop/
├── src/agent_loop/
│   ├── models.py        # Pydantic 数据模型（TaskSpec、StrategyBundle、MemoryRecord 等）
│   ├── service.py       # 核心服务（Runtime、Evaluation、Learning、Registry）
│   ├── settings.py      # 配置管理（环境变量映射）
│   ├── api.py           # FastAPI REST 接口
│   ├── cli.py           # Typer CLI 命令
│   ├── vector.py        # 向量嵌入与余弦相似度
│   └── benchmarks.py    # 性能基准测试
├── examples/            # 示例任务 JSON 文件
├── tests/               # 测试套件
├── reports/             # 运行报告输出目录
├── pyproject.toml       # 项目配置与依赖
└── PROJECT_OVERVIEW.md  # 详细技术文档
```

---

## 🧪 运行测试

```bash
# 安装开发依赖
python -m pip install -e .[dev]

# 运行测试套件
python -m pytest -q

# 运行并生成 smoke 报告
agent-loop run-task examples/real_task_smoke_report.json
```

---

## 📚 更多文档

详细的技术文档、数据模型说明、工作流描述请参阅 [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)。

---

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！请确保：

1. 新增功能附带对应测试
2. 所有现有测试通过（`python -m pytest -q`）
3. 遵循现有代码风格（Pydantic v2 模型 + 类型注解）

---

## 📄 许可证

MIT License — 详见 [LICENSE](LICENSE) 文件。

