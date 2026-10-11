from typing import Literal, Protocol

from pydantic import BaseModel, Field

from .filter_response import StepLog


class Finding(BaseModel):
    fact: str
    price_inr: float | None = None
    url: str = ""           # captured from the page by code, never typed by the LLM
    verified: bool = True   # the price digits were present on the page the LLM just saw


class WebState(BaseModel):
    task_id: str
    task: str
    log: StepLog = Field(default_factory=StepLog)
    findings: list[Finding] = Field(default_factory=list)
    current_url: str = ""
    last_page: str = ""              # latest browser observation; prices are checked against it
    pending: dict | None = None   # tool call the LLM just made, waiting for the executor
    steps_taken: int = 0
    since_finding: int = 0
    status: Literal["running", "done", "incomplete", "failed"] = "running"
    summary: str = ""
    error: str | None = None


class WebResult(BaseModel):
    """What leaves the web agent. The parent graph turns this into its own output model."""
    task_id: str
    status: Literal["done", "incomplete", "failed"]
    summary: str = ""
    findings: list[Finding] = Field(default_factory=list)
    error: str | None = None
    steps_used: int = 0


class BrowserTools(Protocol):
    async def specs(self, query: str | None = None) -> list[dict]: ...                              # OpenAI-format tool specs
    async def call(self, name: str, args: dict) -> tuple[str, bool]: ...  # (raw_text, is_error)