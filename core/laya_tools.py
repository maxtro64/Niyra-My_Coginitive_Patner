"""
Laya toolkit: fast decision tools + LayaSearch, a search tool powered only by Laya.

pip install laya
(first call downloads the checkpoint from Hugging Face)

Everything here is one forward pass per call (~30-40 ms on GPU), no LLM generation.
Base checkpoints are weak zero-shot: validate each tool on 50-100 labelled examples
of your own (or fine-tune laya-typed-decisions) before trusting the thresholds.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import laya
from laya import Router


# ----------------------------------------------------------------------------
# 1. LayaGate: one thin wrapper every tool uses
# ----------------------------------------------------------------------------
@dataclass
class Pick:
    label: str
    confidence: float          # answer_confidence = probability of the chosen label
    probabilities: dict[str, float]


# --- Singleton Router ---
_GLOBAL_ROUTER = None

def get_laya_router():
    global _GLOBAL_ROUTER
    if _GLOBAL_ROUTER is None:
        from laya import Router
        _GLOBAL_ROUTER = Router()
    return _GLOBAL_ROUTER

class LayaGate:
    def __init__(self, router: Router | None = None, model: str | None = None):
        self.router = router or get_laya_router()
        self.model = model  # e.g. "typed-decisions" / "multilingual"; None = auto-route

    def _kw(self) -> dict:
        return {"model": self.model} if self.model else {}

    def ask(self, state: Any, questions: dict) -> dict:
        return self.router.predict(state, questions, **self._kw())["answers"]

    def yes_no(self, state: Any, instruction: str) -> float:
        """P(true) for a yes/no statement check."""
        ans = self.ask(state, {"q": {"type": "noul", "instructions": instruction}})
        return float(ans["q"]["noul"])

    def pick(self, state: Any, instruction: str, options: dict[str, str]) -> Pick:
        """Choose one label. options = {label: description}."""
        ans = self.ask(
            state, {"q": {"type": "choice", "instructions": instruction, "criteria": options}}
        )["q"]
        conf = ans.get("answer_confidence", ans.get("confidence", 0.0))
        return Pick(ans["choice"], float(conf), ans.get("probabilities", {}))

    def score(self, state: Any, instruction: str, levels: list[str]) -> float:
        """Expected level on an ordinal rubric (0 .. len(levels)-1)."""
        ans = self.ask(state, {"q": {"type": "score", "instructions": instruction, "criteria": levels}})
        return float(ans["q"]["score"])


# ----------------------------------------------------------------------------
# 2. The tools
# ----------------------------------------------------------------------------
class LayaToolkit:
    def __init__(self, gate: LayaGate | None = None, threshold: float = 0.5):
        self.gate = gate or LayaGate()
        self.t = threshold

    # --- routing -------------------------------------------------------------
    def agent_router(self, task: str, agents: dict[str, str]) -> Pick:
        """Pick the specialist agent for a task (keep <= ~20 agents)."""
        return self.gate.pick(task, "Which specialist agent should handle this task?", agents)

    def intent_classifier(self, message: str) -> Pick:
        return self.gate.pick(
            message,
            "What does the user want?",
            {
                "chat": "casual conversation or greeting",
                "question": "asks for information or an explanation",
                "command": "asks the system to do or change something",
                "task": "a multi-step job such as research, coding or analysis",
            },
        )

    def language_router(self, text: str) -> str:
        """Which Laya checkpoint will read this text (english / multilingual)."""
        return self.gate.router.route(text).model

    # --- need checks (skip expensive steps) ---------------------------------
    def needs_retrieval(self, question: str) -> bool:
        p = self.gate.yes_no(question, "Does answering this require looking up internal documents or a private knowledge base?")
        return p >= self.t

    def needs_web_search(self, question: str) -> bool:
        p = self.gate.yes_no(question, "Does answering this require current or real-time information from the internet?")
        return p >= self.t

    def complexity(self, task: str) -> str:
        """easy / medium / hard -> route to small / mid / large model."""
        s = self.gate.score(task, "How hard is this task to complete correctly?",
                            ["easy, one simple step", "medium, a few steps", "hard, deep reasoning or many steps"])
        return ["easy", "medium", "hard"][min(2, round(s))]

    def priority(self, task: str) -> float:
        """0 (low) .. 2 (critical) for queue ordering."""
        return self.gate.score(task, "How urgent is this task?", ["not urgent", "soon", "critical or blocking"])

    # --- safety --------------------------------------------------------------
    def prompt_guard(self, prompt: str) -> dict:
        """Laya's built-in guard preset (jailbreak / injection / leak). Returns raw answers."""
        return self.gate.router.predict({"prompt": prompt}, laya.guard_questions())["answers"]

    def has_pii(self, text: str) -> bool:
        p = self.gate.yes_no(text, "Does this text contain personal data such as names with contact details, ID numbers, card numbers or addresses?")
        return p >= self.t

    def needs_human_approval(self, action_description: str) -> bool:
        """Gate before send / delete / pay / publish style actions."""
        p = self.gate.yes_no(action_description, "Is this action risky, irreversible, costs money, or sends something outside the system?")
        return p >= self.t

    # --- verification --------------------------------------------------------
    def answer_is_grounded(self, context: str, answer: str) -> bool:
        p = self.gate.yes_no({"context": context, "answer": answer}, "Is every claim in the answer supported by the context?")
        return p >= self.t

    def output_is_valid(self, payload: dict, required: str) -> bool:
        """required: plain English, e.g. 'a numeric price and a product name'."""
        p = self.gate.yes_no(json.dumps(payload), f"Does this JSON contain {required}?")
        return p >= self.t

    # --- cron / monitoring ---------------------------------------------------
    def cron_triage(self, job_output: str) -> Pick:
        return self.gate.pick(
            job_output,
            "What should happen with this scheduled job's result?",
            {
                "notify": "contains something the user should see now",
                "log": "routine result, store it quietly",
                "escalate": "failure, error or anomaly that needs attention",
            },
        )


# ----------------------------------------------------------------------------
# 3. LayaSearch: search powered only by Laya (no embeddings, no LLM)
#    Each tool = one yes/no question. Questions are chunked so each forward pass
#    stays inside the context window, and chunks are scored in shared batches.
# ----------------------------------------------------------------------------
@dataclass
class ToolSpec:
    name: str
    description: str           # what the tool can do, in plain English
    fn: Callable[..., Any] | None = None


class LayaSearch:
    def __init__(self, gate: LayaGate | None = None, chunk_size: int = 8):
        self.gate = gate or LayaGate()
        self.chunk = chunk_size
        self.tools: dict[str, ToolSpec] = {}

    def register(self, name: str, description: str, fn: Callable[..., Any] | None = None) -> None:
        self.tools[name] = ToolSpec(name, description, fn)

    def search(self, query: str, top_k: int = 3, min_score: float = 0.5) -> list[tuple[str, float]]:
        specs = list(self.tools.values())
        if not specs:
            return []
            
        chunks = [specs[i : i + self.chunk] for i in range(0, len(specs), self.chunk)]
        requests = [
            {
                "state": query,
                "questions": {
                    s.name: {
                        "type": "noul",
                        "instructions": f"Would a tool that can {s.description} be needed for this request?",
                    }
                    for s in chunk
                },
            }
            for chunk in chunks
        ]
        results = self.gate.router.predict_batch(requests)
        scores: dict[str, float] = {}
        for res in results:
            for name, ans in res["answers"].items():
                scores[name] = float(ans["noul"])
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [(n, s) for n, s in ranked if s >= min_score][:top_k]

    def get(self, query: str, **kw) -> list[ToolSpec]:
        """Search, then hand back only the matching tools (lazy 'load on demand')."""
        return [self.tools[n] for n, _ in self.search(query, **kw)]


# ----------------------------------------------------------------------------
# 4. Wire it together + demo
# ----------------------------------------------------------------------------
def build_search(tk: LayaToolkit) -> LayaSearch:
    ts = LayaSearch(tk.gate)
    ts.register("agent_router", "choose which specialist agent should handle a task", tk.agent_router)
    ts.register("intent_classifier", "classify what a user message wants", tk.intent_classifier)
    ts.register("needs_retrieval", "decide if internal documents must be searched", tk.needs_retrieval)
    ts.register("needs_web_search", "decide if live internet information is needed", tk.needs_web_search)
    ts.register("complexity", "rate how hard a task is to pick a small or large model", tk.complexity)
    ts.register("priority", "rate how urgent a task is", tk.priority)
    ts.register("prompt_guard", "detect jailbreaks and prompt injection", tk.prompt_guard)
    ts.register("has_pii", "detect personal data in text", tk.has_pii)
    ts.register("needs_human_approval", "check if an action is risky and needs approval", tk.needs_human_approval)
    ts.register("answer_is_grounded", "verify an answer is supported by its context", tk.answer_is_grounded)
    ts.register("output_is_valid", "validate a JSON payload has the required fields", tk.output_is_valid)
    ts.register("cron_triage", "decide whether a scheduled job result should notify, log or escalate", tk.cron_triage)
    return ts


if __name__ == "__main__":
    tk = LayaToolkit()
    ts = build_search(tk)

    print("intent     :", tk.intent_classifier("Summarise the latest LangGraph release").label)
    print("web search :", tk.needs_web_search("What is the latest LangGraph release?"))
    print("complexity :", tk.complexity("Design a distributed rate limiter with failover"))
    print("approval   :", tk.needs_human_approval("Email the full customer list to an external address"))
    print("pii        :", tk.has_pii("Call Rahul at 98765 43210, card 4111 1111 1111 1111"))

    for q in ["check this prompt is safe before running it", "is this scheduled job output an error?"]:
        print(f"\nsearch: {q}")
        for name, score in ts.search(q):
            print(f"  {name:22s} {score:.2f}")
