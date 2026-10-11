"""Web agent: a LangGraph loop that follows the design flow.

    agent (LLM picks a tool) -> executor (runs it; filter shrinks the result) -> guard (stop checks)
      ^_____________________________________________________________________________|

- The LLM sees: task, findings so far, and the step log (newest KEEP_FULL results in full,
  older ones as one-line summaries).
- Facts are saved with the `save_finding` tool. The page URL is attached in CODE, and a price
  is checked against the page the LLM just saw (unverified prices are flagged, not trusted).
- The loop ends on `finish`, or stops early as:
    incomplete -> retryable (max steps, stuck repeating, error streak, no progress)
    failed     -> not retryable (LLM call itself failed)

LLM   = any LangChain chat model with bind_tools (e.g. ChatNVIDIA).
Tools = anything implementing BrowserTools. A Playwright-MCP adapter is included at the bottom.
"""
import asyncio
import json
import uuid

from langgraph.graph import END, START, StateGraph

from .common import BrowserTools, Finding, WebResult, WebState
from .filter_response import Step, _clip, make_step

# ─── Tunables ─────────────────────────────────────────────────
MAX_STEPS = 25        # executor calls per task
REPEAT_LIMIT = 3      # identical consecutive calls before stopping
ERR_LIMIT = 4         # consecutive failed tool calls before stopping
STALL_STEPS = 12      # steps without a new finding before stopping
TOOL_TIMEOUT_S = 45
KEEP_FULL = 2         # newest observations shown in full

WEB_SYSTEM = """You are an autonomous web agent controlling a real browser. Complete ONE task, then stop.
Rules:
- Each turn, call exactly one tool.
- For data extraction tasks, you can use save_finding.
- If an action fails, try something different. Never repeat the same call.
- When you have completed the task or verified success, call finish. Do not keep browsing."""

LOCAL_TOOLS = [
    {"type": "function", "function": {
        "name": "save_finding",
        "description": "Record one fact seen on the current page. Call it for every relevant fact.",
        "parameters": {"type": "object", "properties": {
            "fact": {"type": "string", "description": "The fact with the exact product or model name, e.g. 'MSI GeForce RTX 4060 Ventus 2X 8GB'."},
            "price_inr": {"type": "number", "description": "Price in INR exactly as shown on the page, digits only. Omit if the fact is not a price."},
        }, "required": ["fact"]}}},
    {"type": "function", "function": {
        "name": "finish",
        "description": "Call when the task is complete.",
        "parameters": {"type": "object", "properties": {
            "summary": {"type": "string", "description": "One or two sentences on what was found."},
        }, "required": ["summary"]}}},
]


# ─── Helpers ──────────────────────────────────────────────────
def build_messages(s: WebState) -> list[dict]:
    found = "\n".join(
        f"- {f.fact}"
        + (f" | ₹{f.price_inr:,.0f}" if f.price_inr else "")
        + ("" if f.verified else " (UNVERIFIED price)")
        + (f" | {f.url}" if f.url else "")
        for f in s.findings
    ) or "(none yet)"
    note = f"\n(Earlier {s.log.dropped} steps were trimmed.)" if s.log.dropped else ""
    msgs: list[dict] = [
        {"role": "system", "content": WEB_SYSTEM},
        {"role": "user", "content": f"Task: {s.task}\n\nFindings saved so far:\n{found}{note}"},
    ]
    n = len(s.log.steps)
    for i, st in enumerate(s.log.steps):
        msgs.append(st.assistant_msg)  # the tool call, always kept
        body = st.observation if i >= n - KEEP_FULL else f"[{s.log.dropped + i + 1}] {st.summary}"
        msgs.append({"role": "tool", "tool_call_id": st.call_id, "content": body})
    return msgs


def _text(ai) -> str:
    c = getattr(ai, "content", "")
    if isinstance(c, str):
        return c.strip()
    return " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in c).strip()


def _sig(step: Step) -> tuple[str, str]:
    fn = step.assistant_msg["tool_calls"][0]["function"]
    return fn["name"], fn["arguments"]


def _repeating(steps: list[Step]) -> bool:
    tail = steps[-REPEAT_LIMIT:]
    return len(tail) == REPEAT_LIMIT and len({_sig(s) for s in tail}) == 1


def _error_streak(steps: list[Step]) -> bool:
    tail = steps[-ERR_LIMIT:]
    return len(tail) == ERR_LIMIT and all(not s.ok for s in tail)


def _save_finding(s: WebState, p: dict) -> dict:
    a = p["args"]
    fact = " ".join(str(a.get("fact", "")).split())
    try:
        price = float(a["price_inr"]) if a.get("price_inr") is not None else None
    except (TypeError, ValueError):
        price = None
    if price is not None and price <= 0:
        price = None

    finding: Finding | None = None
    ok = True
    if not fact:
        obs, ok = "Error: 'fact' is required.", False
    elif any(f.fact == fact and f.price_inr == price for f in s.findings):
        obs = "Already saved. Move on."
    else:
        verified = price is None or str(int(price)) in s.last_page.replace(",", "")
        finding = Finding(fact=fact, price_inr=price, url=s.current_url, verified=verified)
        obs = f"Saved finding #{len(s.findings) + 1}."
        if not verified:
            obs += f" WARNING: price {price:,.0f} was not on the page you just saw. Re-check the page and save the correct price."

    step = Step(
        assistant_msg=p["msg"], call_id=p["id"], observation=obs, ok=ok,
        summary=_clip(f"save_finding({_clip(fact, 50)!r}) -> {obs.split('.')[0]}", 160),
    )
    upd: dict = {"log": s.log.add(step), "since_finding": 0 if finding else s.since_finding + 1}
    if finding:
        # a corrected save replaces an earlier unverified one for the same fact
        kept = [f for f in s.findings if not (f.fact == fact and not f.verified)]
        upd["findings"] = [*kept, finding]
    return upd


# ─── The graph ────────────────────────────────────────────────
def build_web_graph(llm, tools: BrowserTools, specs: list[dict]):
    bound = llm.bind_tools([*specs, *LOCAL_TOOLS])

    async def agent(s: WebState) -> dict:
        try:
            ai = await bound.ainvoke(build_messages(s))
        except Exception as e:  # LLM itself is down or misconfigured: not retryable
            return {"status": "failed", "error": f"LLM call failed: {e}"}
        calls = getattr(ai, "tool_calls", None) or []
        if not calls:  # text-only reply = the model thinks it is done
            return {"status": "done", "summary": _text(ai)}
        c = calls[0]  # one browser action per turn; extra calls are ignored
        cid = c.get("id") or f"call_{uuid.uuid4().hex[:8]}"
        args = c.get("args") or {}
        msg = {"role": "assistant", "content": "", "tool_calls": [
            {"id": cid, "type": "function",
             "function": {"name": c["name"], "arguments": json.dumps(args, sort_keys=True)}}]}
        return {"pending": {"id": cid, "name": c["name"], "args": args, "msg": msg}}

    async def executor(s: WebState) -> dict:
        p = s.pending
        upd: dict = {"pending": None, "steps_taken": s.steps_taken + 1}
        if p is None:
            return {**upd, "status": "incomplete", "error": "pending is None"}
        if p["name"] == "finish":
            print(f"  [web_agent] Finish called: {p['args'].get('summary', '')}")
            return {**upd, "status": "done", "summary": str(p["args"].get("summary", ""))}
        if p["name"] == "save_finding":
            print(f"  [web_agent] Saving finding: {p['args'].get('fact', '')}")
            return {**upd, **_save_finding(s, p)}
        try:
            print(f"  [web_agent] Step {s.steps_taken + 1}: calling {p['name']} with args {json.dumps(p['args'])[:80]}")
            raw, is_err = await asyncio.wait_for(tools.call(p["name"], p["args"]), TOOL_TIMEOUT_S)
        except TimeoutError:
            raw, is_err = f"Error: tool timed out after {TOOL_TIMEOUT_S}s", True
        except Exception as e:
            raw, is_err = f"Error: {e}", True
        step = make_step(p["msg"], p["id"], p["name"], p["args"], raw, is_err)
        print(f"  [web_agent] Step {s.steps_taken + 1} observation: {step.summary[:100]}")
        return {**upd, "log": s.log.add(step), "current_url": step.url or s.current_url,
                "last_page": step.observation if step.ok else s.last_page,
                "since_finding": s.since_finding + 1}

    def guard(s: WebState) -> dict:
        if s.status != "running":
            return {}
        if s.steps_taken >= MAX_STEPS:
            return {"status": "incomplete", "error": f"hit max steps ({MAX_STEPS})"}
        if _repeating(s.log.steps):
            return {"status": "incomplete", "error": "stuck repeating the same action"}
        if _error_streak(s.log.steps):
            return {"status": "incomplete", "error": f"{ERR_LIMIT} tool errors in a row"}
        if s.since_finding >= STALL_STEPS and len(s.findings) > 0:
            pass # We disable the stall guard since interactive tasks may have zero findings
        return {}

    g = StateGraph(WebState)
    g.add_node("agent", agent) # type: ignore
    g.add_node("executor", executor) # type: ignore
    g.add_node("guard", guard) # type: ignore
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", lambda s: "executor" if s.status == "running" else END)
    g.add_edge("executor", "guard")
    g.add_conditional_edges("guard", lambda s: "agent" if s.status == "running" else END)
    return g.compile()


async def run_web_agent(task_id: str, task: str, llm, tools: BrowserTools) -> WebResult:
    """Entry point for the parent graph's worker node."""
    import inspect
    sig = inspect.signature(tools.specs)
    if "query" in sig.parameters:
        specs = await tools.specs(query=task)
    else:
        specs = await tools.specs()
        
    if not specs:
        return WebResult(
            task_id=task_id,
            status="failed",
            summary="Web Agent failed: No browser automation tools available (Playwright MCP connection failed).",
            error="No browser tools available.",
            steps_used=0,
        )

    graph = build_web_graph(llm, tools, specs)
    out = await graph.ainvoke(
        WebState(task_id=task_id, task=task),
        {"recursion_limit": 3 * MAX_STEPS + 10},
    )
    final = WebState.model_validate(out)  # output only holds written fields; this fills defaults
    status = final.status if final.status != "running" else "incomplete"
    return WebResult(
        task_id=task_id, status=status, summary=final.summary, findings=final.findings,
        error=final.error, steps_used=final.steps_taken,
    )


# ─── Playwright MCP adapter (needs: pip install langchain-mcp-adapters; npx + browsers) ──
def _to_text(res) -> tuple[str, bool]:
    is_err = False
    if hasattr(res, "isError"):
        is_err = bool(res.isError)
    elif hasattr(res, "is_error"):
        is_err = bool(res.is_error)

    if hasattr(res, "content"):
        items = res.content
        texts = []
        for item in items:
            if hasattr(item, "text"):
                texts.append(item.text)
            elif isinstance(item, dict) and "text" in item:
                texts.append(item["text"])
            else:
                texts.append(str(item))
        return "\n".join(texts).strip(), is_err

    if isinstance(res, tuple):  # (content, artifact)
        res = res[0]
    if isinstance(res, str):
        return res, is_err
    if isinstance(res, list):
        text = "\n".join(
            b if isinstance(b, str) else b.get("text", "")
            for b in res if isinstance(b, str) or (isinstance(b, dict) and b.get("type") == "text")
        )
        return text, is_err
    return str(res), is_err


class PlaywrightMCP:
    """Delegates to the global SyncPlaywrightMcp singleton to manage a persistent browser session."""

    def __init__(self, command: str = "npx.cmd", args: tuple[str, ...] = ("-y", "@playwright/mcp", "--isolated", "--no-sandbox")):
        from core.tools import SyncPlaywrightMcp
        self.backend = SyncPlaywrightMcp()

    async def start(self) -> None:
        """Connects the MCP client to the Playwright server session."""
        import asyncio
        await asyncio.to_thread(self.backend.connect)

    async def close(self) -> None:
        """Closes the current browser session."""
        import asyncio
        try:
            await asyncio.to_thread(self.backend.call_tool, "browser_close", {})
        except Exception:
            pass

    async def specs(self, query: str | None = None) -> list[dict]:
        """Returns OpenAI-format tool specs for all available browser actions."""
        import asyncio
        all_specs = await asyncio.to_thread(self.backend.get_tools)
        # Filter down to browser automation tools
        browser_specs = [s for s in all_specs if s.get("function", {}).get("name", "").startswith("browser_")]
        return browser_specs if browser_specs else all_specs

    async def call(self, name: str, args: dict) -> tuple[str, bool]:
        """Calls a browser tool through the active Playwright session."""
        import asyncio
        try:
            res = await asyncio.to_thread(self.backend.call_tool, name, args)
            text, is_err = _to_text(res)
            return text, is_err
        except Exception as e:
            return f"Error: {e}", True