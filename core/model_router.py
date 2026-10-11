"""
Standalone LLM router: picks the best specialist agent for a task.

pip install langchain langchain-anthropic pydantic
export ANTHROPIC_API_KEY=...
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field

from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

FALLBACK_AGENT = "general_agent"
FINISH = "FINISH"


@dataclass
class AgentSpec:
    name: str
    description: str
    not_for: str = ""
    examples: list[str] = field(default_factory=list)


AGENTS: dict[str, AgentSpec] = {
    "rag_agent": AgentSpec(
        "rag_agent",
        "Answers questions from the internal knowledge base / uploaded documents.",
        "General world knowledge, live web info, code execution.",
        ["What does our refund policy say?", "Summarise section 3 of the PDF"],
    ),
    "web_search_agent": AgentSpec(
        "web_search_agent",
        "Finds current / real-time information on the internet.",
        "Questions answerable from the internal documents.",
        ["Latest LangGraph release notes"],
    ),
    "code_agent": AgentSpec(
        "code_agent",
        "Writes, debugs, explains and runs code.",
        "Pure conceptual Q&A with no code involved.",
        ["Fix this FastAPI bug", "Write a SQL query for monthly revenue"],
    ),
    "data_analyst_agent": AgentSpec(
        "data_analyst_agent",
        "Analyses tabular data, computes statistics, builds charts.",
        "Writing application code.",
        ["Plot sales by region from this CSV"],
    ),
    "web_agent": AgentSpec(
        "web_agent",
        "Controls a web browser to navigate websites, click buttons, type text, and interact with web applications (e.g. YouTube, forms).",
        "Just answering questions, or writing code.",
        ["Open YouTube and play a video", "Fill out this Google form"],
    ),
    "general_agent": AgentSpec(
        "general_agent",
        "Fallback for chit-chat, writing, brainstorming and anything unclear.",
    ),
}


class RoutingDecision(BaseModel):
    reasoning: str = Field(description="1-2 sentences on why this agent (or FINISH).")
    next_agent: str = Field(
        description=f"Exact agent name from the registry, or '{FINISH}' if the request is fully answered."
    )
    task_for_agent: str = Field(
        default="", description="Self-contained instruction for the chosen agent."
    )
    confidence: float = Field(ge=0.0, le=1.0, description="0-1 confidence.")
    needs_clarification: bool = Field(default=False)
    clarifying_question: str | None = Field(default=None)


ROUTER_PROMPT = """You are the routing brain of a multi-agent system.
Your ONLY job is to decide which specialist agent should act next.

Available agents:
{registry}

Rules:
1. Pick exactly one agent, using the exact name.
2. If agents have already fully answered the user's request, choose {finish}.
3. Do NOT repeat an agent that just returned a satisfactory result.
4. If several skills are needed, pick the one that must run FIRST.
5. If the request is genuinely ambiguous, set needs_clarification=true.
6. Be honest with confidence: below 0.5 means you are guessing.
"""


def _render_registry(agents: dict[str, AgentSpec]) -> str:
    out = []
    for a in agents.values():
        s = f"- {a.name}: {a.description}"
        if a.not_for:
            s += f"\n    NOT for: {a.not_for}"
        if a.examples:
            s += "\n    Examples: " + " | ".join(a.examples)
        out.append(s)
    return "\n".join(out)


class LLMRouter:
    def __init__(
        self,
        agents: dict[str, AgentSpec] = AGENTS,
        model: str | None = None,
        llm_instance = None,
        confidence_threshold: float = 0.75,
        max_history: int = 12,
    ):
        self.agents = agents
        self.threshold = confidence_threshold
        self.max_history = max_history
        
        if llm_instance == "laya":
            import sys
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from core.laya_tools import LayaGate
            
            gate = LayaGate()
            
            class LayaWrapper:
                def __init__(self, agents_dict, threshold):
                    self.agents_dict = agents_dict
                    self.threshold = threshold

                def invoke(self, convo):
                    last_msg = convo[-1].content if convo else ""
                    text_lower = str(last_msg).lower()

                    # Check if there is prior agent work in this conversation
                    has_prior_agent_work = any(
                        getattr(m, "type", "") in ("ai", "assistant", "tool")
                        or (isinstance(m, dict) and m.get("role") in ("assistant", "tool"))
                        for m in convo[:-1]
                    ) if len(convo) > 1 else False
                    
                    # Direct intent detection for distinct agent domains
                    coding_triggers = [
                        "python", "script", "code", "function", "debug", "algorithm",
                        "write a program", "parse", "calculate", "sql query", "bug"
                    ]
                    if any(kw in text_lower for kw in coding_triggers) and not any(w in text_lower for w in ["browse", "youtube"]):
                        return RoutingDecision(
                            reasoning="Direct intent detected for code generation/execution.",
                            next_agent="code_agent",
                            task_for_agent=str(last_msg),
                            confidence=0.95,
                        )

                    browser_triggers = ["open youtube", "open url", "open website", "navigate to", "click button", "fill form", "play song"]
                    if any(kw in text_lower for kw in browser_triggers):
                        return RoutingDecision(
                            reasoning="Direct intent detected for browser automation.",
                            next_agent="web_agent",
                            task_for_agent=str(last_msg),
                            confidence=0.95,
                        )

                    search_triggers = ["search the web", "web search", "search online", "look up on the internet", "google for", "tavily"]
                    if any(kw in text_lower for kw in search_triggers):
                        return RoutingDecision(
                            reasoning="Direct intent detected for live web search.",
                            next_agent="web_search_agent",
                            task_for_agent=str(last_msg),
                            confidence=0.95,
                        )

                    rag_triggers = ["documentation", "knowledge base", "internal doc", "uploaded file", "project readme"]
                    if any(kw in text_lower for kw in rag_triggers):
                        return RoutingDecision(
                            reasoning="Direct intent detected for RAG / internal document retrieval.",
                            next_agent="rag_agent",
                            task_for_agent=str(last_msg),
                            confidence=0.95,
                        )

                    # Build criteria: ONLY include FINISH if an agent has already produced work
                    criteria = {k: v.description for k, v in self.agents_dict.items()}
                    if has_prior_agent_work:
                        criteria[FINISH] = "The user request is fully answered and no further action is needed."

                    pick = gate.pick(
                        last_msg,
                        "Which agent should handle this request?",
                        criteria
                    )
                    
                    next_agent = pick.label
                    confidence = float(pick.confidence)

                    # If classification confidence is below the calibrated threshold, fall back safely
                    if confidence < self.threshold or next_agent not in criteria:
                        return RoutingDecision(
                            reasoning=f"Confidence {confidence:.2f} below threshold {self.threshold}; routing to safe fallback.",
                            next_agent=FALLBACK_AGENT,
                            task_for_agent=str(last_msg),
                            confidence=confidence,
                            needs_clarification=(confidence < 0.35),
                            clarifying_question="Could you please provide more details on what you would like me to do?" if confidence < 0.35 else None
                        )

                    return RoutingDecision(
                        reasoning=f"Routed via Laya System 1 (confidence {confidence:.2f}).",
                        next_agent=next_agent,
                        task_for_agent=str(last_msg),
                        confidence=confidence,
                    )
            
            self.llm = LayaWrapper(agents, self.threshold)
            self.system = ROUTER_PROMPT.format(
                registry=_render_registry(agents), finish=FINISH
            )
        else:
            if llm_instance:
                llm = llm_instance
            else:
                llm = init_chat_model(
                    model or os.getenv("ROUTER_MODEL", "anthropic:claude-sonnet-5-5"),
                    temperature=0,
                )
                
            self.llm = llm.with_structured_output(RoutingDecision)
            self.system = ROUTER_PROMPT.format(
                registry=_render_registry(agents), finish=FINISH
            )

    def route(self, messages: Sequence[BaseMessage]) -> RoutingDecision:
        convo = [SystemMessage(content=self.system), *messages[-self.max_history :]]
        try:
            decision = self.llm.invoke(convo)
        except Exception as e:
            return RoutingDecision(
                reasoning=f"Router error ({type(e).__name__}); using fallback.",
                next_agent=FALLBACK_AGENT,
                task_for_agent=str(messages[-1].content) if messages else "",
                confidence=0.0,
            )
        return self._validate(decision, messages)

    def _validate(self, d: RoutingDecision, messages: Sequence[BaseMessage]) -> RoutingDecision:
        has_prior_agent_work = any(
            getattr(m, "type", "") in ("ai", "assistant", "tool")
            or (isinstance(m, dict) and m.get("role") in ("assistant", "tool"))
            for m in messages[:-1]
        ) if len(messages) > 1 else False

        # Fresh request or conversation without agent output cannot be FINISH
        if d.next_agent == FINISH and (not has_prior_agent_work or len(messages) <= 1):
            d.reasoning += " [unanswered request cannot be FINISH -> fallback general_agent]"
            d.next_agent = FALLBACK_AGENT
            d.confidence = max(d.confidence, 0.5)

        # Premature FINISH guard with low confidence
        if d.next_agent == FINISH and d.confidence < self.threshold:
            d.reasoning += " [uncertain FINISH -> fallback general_agent]"
            d.next_agent = FALLBACK_AGENT

        if d.next_agent != FINISH and d.next_agent not in self.agents:
            d.reasoning += f" [invalid agent '{d.next_agent}' -> fallback]"
            d.next_agent = FALLBACK_AGENT
            d.confidence = min(d.confidence, 0.3)

        if (
            d.next_agent not in (FINISH, FALLBACK_AGENT)
            and d.confidence < self.threshold
            and not d.needs_clarification
        ):
            d.reasoning += f" [low confidence {d.confidence:.2f} < {self.threshold} -> fallback general_agent]"
            d.next_agent = FALLBACK_AGENT

        if d.next_agent != FINISH and not d.task_for_agent and messages:
            d.task_for_agent = str(messages[0].content)
        return d


if __name__ == "__main__":
    router = LLMRouter()
    d = router.route([HumanMessage(content="Fix the bug in my FastAPI login route")])
    print(d.model_dump_json(indent=2))