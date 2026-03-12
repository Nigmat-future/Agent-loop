from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field


class Settings(BaseModel):
    home_dir: Path = Field(default_factory=lambda: Path.cwd() / ".agent_loop")
    db_path: Path | None = None
    provider: str = "heuristic"
    model: str = "gpt-4o-mini"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str | None = None
    openai_site_url: str | None = None
    openai_app_name: str = "Agent Loop"
    default_workspace_root: Path = Field(default_factory=Path.cwd)
    default_http_domains: list[str] = Field(default_factory=list)
    default_shell_prefixes: list[str] = Field(
        default_factory=lambda: ["python", "echo", "dir", "type"]
    )

    @classmethod
    def from_env(cls) -> "Settings":
        home_dir = Path(os.getenv("AGENT_LOOP_HOME", Path.cwd() / ".agent_loop"))
        db_override = os.getenv("AGENT_LOOP_DB_PATH")
        domains = os.getenv("AGENT_LOOP_ALLOWED_HTTP_DOMAINS", "")
        shell_prefixes = os.getenv("AGENT_LOOP_ALLOWED_SHELL_PREFIXES", "")
        return cls(
            home_dir=home_dir,
            db_path=Path(db_override) if db_override else None,
            provider=os.getenv("AGENT_LOOP_PROVIDER", "heuristic"),
            model=os.getenv("AGENT_LOOP_MODEL", "gpt-4o-mini"),
            openai_base_url=os.getenv(
                "AGENT_LOOP_OPENAI_BASE_URL", "https://api.openai.com/v1"
            ),
            openai_api_key=os.getenv("AGENT_LOOP_OPENAI_API_KEY"),
            openai_site_url=os.getenv("AGENT_LOOP_OPENAI_SITE_URL"),
            openai_app_name=os.getenv("AGENT_LOOP_OPENAI_APP_NAME", "Agent Loop"),
            default_workspace_root=Path(
                os.getenv("AGENT_LOOP_WORKSPACE_ROOT", str(Path.cwd()))
            ),
            default_http_domains=[item for item in domains.split(",") if item],
            default_shell_prefixes=[
                item for item in shell_prefixes.split(",") if item
            ]
            or ["python", "echo", "dir", "type"],
        )

    def resolved_db_path(self) -> Path:
        self.home_dir.mkdir(parents=True, exist_ok=True)
        return self.db_path or self.home_dir / "agent_loop.db"
