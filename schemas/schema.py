"""Structured-output models: the ONLY models the LLM is asked to fill.

Rules:
- No runtime fields here (status, retries, errors) -- the model must not see them.
- Field descriptions and class docstrings are sent to the model, so write them
  like instructions.
- Never put these directly into graph state; convert them in the node.

Flow: user speech -> VoiceResponse -> (mode == "delegate") -> Plan -> workers
"""
import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# ─── Shared types ─────────────────────────────────────────────
WorkerName = Literal["browser", "search", "aggregator"]  # <- your real workers

# ─── Helpers ──────────────────────────────────────────────────
_URL = re.compile(r"https?://\S+")
_MD = re.compile(r"[*_`#>~|]")


# ─── Voice agent output ───────────────────────────────────────
class VoiceResponse(BaseModel):
    """How JARVIS answers a spoken request. The speech field is read aloud by TTS."""

    mode: Literal["reply", "delegate"] = Field(
        description="'reply' if you can answer from the conversation alone. "
        "'delegate' if it needs tools: web, PC control, messages, or memory."
    )
    speech: str = Field(
        description="What to say aloud. Plain spoken English, 1-3 short sentences. "
        "No markdown, lists, emojis, or URLs. For 'delegate', a short "
        "acknowledgement like 'On it.'"
    )
    task: str = Field(
        default="",
        description="Only for 'delegate': the request rewritten as a clear, standalone "
        "instruction with pronouns resolved (e.g. 'Email Rahul that I will be late'). "
        "Empty for 'reply'.",
    )
    expects_followup: bool = Field(
        default=False,
        description="True only if you asked the user a question and are waiting for the answer.",
    )

    @field_validator("speech")
    @classmethod
    def _speakable(cls, v: str) -> str:
        v = _URL.sub("", v)  # TTS would read URLs letter by letter
        v = _MD.sub("", v)  # or say "asterisk asterisk"
        return re.sub(r"\s+", " ", v).strip()

    @model_validator(mode="after")
    def _consistent(self) -> "VoiceResponse":
        if not self.speech:
            raise ValueError("speech must not be empty")
        if self.mode == "delegate" and not self.task.strip():
            raise ValueError("delegate requires a task")
        return self


# ─── Planner output ───────────────────────────────────────────
class PlannedTask(BaseModel):
    id: str = Field(description="Short unique id, e.g. t1, t2")
    worker: WorkerName = Field(description="Worker best suited to perform this task")
    description: str = Field(description="What the worker should do for this task")


class Plan(BaseModel):
    """Decomposition of the user request into ordered tasks."""

    tasks: list[PlannedTask] = Field(
        description="Ordered list of tasks needed to fulfil the request. "
        "A separate worker is created for each task."
    )

    @model_validator(mode="after")
    def _unique_ids(self) -> "Plan":
        ids = [t.id for t in self.tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("Task ids must be unique")
        return self