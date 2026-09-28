"""
Ollama LLM — Local Language Model
Plan-Execute-Verify with persistent browser session, step retries, and task retries.

Key design:
- ONE shared `messages` list per task run — all steps share the same conversation
  thread, so each step sees the previous step's tool results (page content, URLs, etc.)
- Verifier calls `browser_snapshot` directly to confirm page state, not just spoken text
- Planner is tool-aware: knows which browser tools exist so it writes executable steps
- Step retry: up to 3 attempts per step with escalating prompt
- Task retry: up to 2 full replans if too many steps fail
"""

import json
import ollama
from config import (
    LLM_MODEL,
    LLM_SYSTEM_PROMPT,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
)
import time
import inspect
import asyncio
from typing import Generator, List, Optional, Callable


# ─── Retry config ────────────────────────────────────────────────────────────
MAX_STEP_RETRIES = 3
MAX_TASK_RETRIES = 2
STEP_RETRY_WAIT  = 0.8   # seconds between retries

# ─── Planner prompt ──────────────────────────────────────────────────────────
# Tool-aware: lists actual Playwright tools so the planner writes executable steps
PLANNER_PROMPT = """\
You are a precise mission planner for an autonomous AI agent that controls a REAL browser.

The agent has these tools available:
  browser_navigate(url)            — navigate to a URL
  browser_snapshot()               — read current page accessibility tree (use to verify you're on the right page)
  browser_take_screenshot()        — take a screenshot
  browser_find(text)               — find element by text/regex on current page
  browser_click(element)           — click an element (describe it exactly as it appears)
  browser_type(element, text)      — type text into a field
  browser_press_key(key)           — press a keyboard key (Enter, Tab, Escape, etc.)
  browser_fill_form(fields)        — fill multiple form fields at once
  browser_wait_for(text)           — wait until text appears on page
  browser_tabs(action)             — manage tabs (list/create/select/close)
  browser_evaluate(js)             — run JavaScript on the page
  tavily_web_search(query)         — search the internet for real-time data

TASK: Decompose the user's request into a MINIMAL ordered list of ATOMIC steps.

Rules:
- ONE action per step. Never bundle multiple actions.
- Use the EXACT tool names above. Write steps as executable instructions, not descriptions.
- Include all needed details: full URLs, exact button text, exact field names, exact search terms.
- After any navigation or click, include a browser_snapshot step to verify the page changed correctly.
- For simple conversation or knowledge questions needing NO tools, output: ["respond directly"]
- Maximum 12 steps. Be minimal but complete.

GOOD example for "search youtube for Bohemian Rhapsody and play it":
[
  "browser_navigate to https://www.youtube.com",
  "browser_snapshot to verify YouTube home page loaded",
  "browser_find the search box and browser_type 'Bohemian Rhapsody Queen'",
  "browser_press_key Enter to submit search",
  "browser_wait_for 'Bohemian Rhapsody' in search results",
  "browser_snapshot to verify search results page",
  "browser_find 'Bohemian Rhapsody' video by Queen and browser_click it",
  "browser_wait_for the video player to appear",
  "browser_snapshot to verify video is playing"
]

BAD example: ["Search YouTube for Bohemian Rhapsody and click on it and play it"]

Output ONLY a raw JSON array of strings. Zero commentary. Zero markdown fences.

User request: {user_text}
"""

# ─── Step verifier prompt ─────────────────────────────────────────────────────
VERIFIER_PROMPT = """\
You are a strict step-completion auditor for an autonomous browser agent.

Step goal: {step}

Agent's spoken response: {spoken}

Browser page snapshot (accessibility tree excerpt):
{snapshot}

Did the agent successfully complete this step?
Answer with ONLY one word: SUCCESS or FAILURE

SUCCESS if:
- A navigation step: the snapshot shows the expected page/URL content
- A click step: the snapshot shows the page changed or the element was activated
- A type/fill step: the snapshot shows the text is in the field
- A search step: the snapshot contains relevant results
- A verify/snapshot step: always SUCCESS if snapshot was obtained
- A conversational step: the spoken response is relevant and non-empty

FAILURE if:
- Snapshot is empty or shows an error page
- The step required navigating to X but snapshot shows a different/wrong page
- The step required finding/clicking an element that doesn't appear in snapshot
- The spoken response says it could not complete the step
- Tool call returned an error and no recovery is evident
"""

# ─── Triggers ─────────────────────────────────────────────────────────────────
COMPLEX_TASK_TRIGGERS = [
    "go to", "navigate to", "open ", "search for", "find and", "click",
    "fill in", "fill out", "log in", "sign in", "download", "book", "buy",
    "order", "compare", "research", "summarize", "extract", "scrape",
    "check and", " then ", "after that", "step by step", "multiple",
    "lookup", "look up", "browse to", "type in", "play ", "watch ",
    "show me", "get the", "find me",
]


class LanguageModel:
    """Local LLM with browser-session-aware plan-execute-verify loop."""

    def __init__(self):
        self._start_ollama()
        self.history: List[dict] = []
        self._max_history = 20

    # ─── Ollama lifecycle ─────────────────────────────────────────────────────

    def _start_ollama(self):
        try:
            ollama.list()
            return
        except Exception:
            pass
        import subprocess, sys
        flags = 0x08000000 if sys.platform == "win32" else 0
        for exe in ["ollama", r"C:\Users\91639\AppData\Local\Programs\Ollama\ollama.exe"]:
            try:
                subprocess.Popen([exe, "serve"],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                 creationflags=flags)
                break
            except Exception:
                continue
        time.sleep(3)
        try:
            models = ollama.list()
            model_names = [m.model for m in models.models]
            if not any(LLM_MODEL.split(":")[0] in n for n in model_names):
                ollama.pull(LLM_MODEL)
        except Exception:
            pass

    # ─── Non-streaming LLM call ───────────────────────────────────────────────

    def _query_once(self, prompt: str, max_tokens: int = 512, temp: float = 0.1) -> str:
        try:
            resp = ollama.chat(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                options={"temperature": temp, "num_predict": max_tokens,
                         "num_ctx": 4096, "num_gpu": 99},
            )
            return resp.message.content.strip()
        except Exception as e:
            return f"ERROR: {e}"

    # ─── Tool call helper ─────────────────────────────────────────────────────

    def _call_tool_sync(self, mcp_client, tool_name: str, arguments: dict) -> str:
        """Execute a tool synchronously, return string result."""
        try:
            result = mcp_client.call_tool(tool_name, arguments)
            if inspect.iscoroutine(result):
                try:
                    result = asyncio.run(result)
                except Exception:
                    pass
            if hasattr(result, "content"):
                return "\n".join(c.text for c in result.content if hasattr(c, "text"))
            return str(result)
        except Exception as e:
            return f"Tool error: {e}"

    # ─── Planner ──────────────────────────────────────────────────────────────

    def _is_complex_task(self, user_text: str) -> bool:
        lower = user_text.lower()
        return any(t in lower for t in COMPLEX_TASK_TRIGGERS)

    def _plan_task(self, user_text: str) -> Optional[List[str]]:
        raw = self._query_once(
            PLANNER_PROMPT.format(user_text=user_text),
            max_tokens=800, temp=0.15
        )
        start, end = raw.find("["), raw.rfind("]")
        if start == -1 or end == -1:
            return None
        try:
            steps = json.loads(raw[start:end + 1])
            if isinstance(steps, list) and steps:
                return [str(s).strip() for s in steps if s]
        except Exception:
            pass
        return None

    # ─── Browser snapshot for verification ───────────────────────────────────

    def _get_page_snapshot(self, mcp_client) -> str:
        """Get current browser accessibility tree. Returns truncated string."""
        if not mcp_client:
            return "(no browser)"
        snap = self._call_tool_sync(mcp_client, "browser_snapshot", {})
        # Truncate to avoid bloating verifier context
        return snap[:3000] if snap else "(empty snapshot)"

    # ─── Step verifier ────────────────────────────────────────────────────────

    def _verify_step(self, step: str, spoken: str, mcp_client) -> bool:
        """
        Verify step completion using BOTH spoken output AND actual page snapshot.
        Returns True = SUCCESS.
        """
        if not spoken and not mcp_client:
            return False

        # Get live browser state
        snapshot = self._get_page_snapshot(mcp_client) if mcp_client else "(no browser)"

        verdict = self._query_once(
            VERIFIER_PROMPT.format(step=step, spoken=spoken or "(silent)", snapshot=snapshot),
            max_tokens=8, temp=0.0
        )
        return verdict.strip().upper().startswith("SUCCESS")

    # ─── Streaming step executor ──────────────────────────────────────────────

    def _execute_step_stream(
        self,
        shared_messages: List[dict],   # THE shared thread — mutated in place
        step_prompt: str,
        llm_tools,
        interrupt_check: Callable,
        on_tool_call: Callable,
        on_tool_result: Callable,
        mcp_client,
    ) -> Generator[str, None, None]:
        """
        Append step_prompt to shared_messages, stream LLM response,
        handle tool calls, and append ALL results back to shared_messages.

        This keeps the browser session alive across steps: each step
        sees the previous step's tool results (page content, URLs, etc.)
        """
        shared_messages.append({"role": "user", "content": step_prompt})

        full_text = ""
        sentence_buffer = ""
        sentence_terminators = {'.', '!', '?'}
        soft_terminators    = {',', ';', ':', '—', '–'}

        MAX_TOOL_ROUNDS = 5  # guard against infinite tool loops
        tool_round = 0

        while tool_round <= MAX_TOOL_ROUNDS:
            tool_calls_this_round = []
            round_text = ""

            try:
                stream = ollama.chat(
                    model=LLM_MODEL,
                    messages=shared_messages,
                    stream=True,
                    tools=llm_tools,
                    options={
                        "temperature": LLM_TEMPERATURE,
                        "num_predict": LLM_MAX_TOKENS,
                        "num_ctx": 8192,
                        "num_gpu": 99,
                    },
                )

                for chunk in stream:
                    if interrupt_check and interrupt_check():
                        return

                    if chunk.message.tool_calls:
                        tool_calls_this_round.extend(chunk.message.tool_calls)

                    token = chunk.message.content
                    if not token:
                        continue

                    sentence_buffer += token
                    round_text += token
                    full_text += token

                    stripped = sentence_buffer.strip()
                    if not stripped or stripped.startswith("{"):
                        continue
                    last = stripped[-1]
                    if last in sentence_terminators and len(stripped) > 8:
                        yield stripped
                        sentence_buffer = ""
                    elif last in soft_terminators and len(stripped) > 25:
                        yield stripped
                        sentence_buffer = ""

            except Exception as e:
                err = f"LLM error: {e}"
                yield err
                full_text += err
                break

            # Flush sentence buffer
            remaining = sentence_buffer.strip()
            if remaining and not remaining.startswith("{"):
                yield remaining
                full_text += remaining
                sentence_buffer = ""

            if not tool_calls_this_round:
                # No tool calls → LLM is done with this step
                break

            # ── Process tool calls and append results to shared thread ───────
            # Append assistant message with tool_calls to shared thread
            shared_messages.append({
                "role": "assistant",
                "content": round_text.strip(),
                "tool_calls": [tc.model_dump() for tc in tool_calls_this_round]
                if hasattr(tool_calls_this_round[0], 'model_dump')
                else tool_calls_this_round
            })

            for tc in tool_calls_this_round:
                tool_name = tc.function.name
                tool_args = tc.function.arguments

                if on_tool_call:
                    try:
                        on_tool_call(tool_name, tool_args)
                    except Exception:
                        pass

                content_str = self._call_tool_sync(mcp_client, tool_name, tool_args)

                if on_tool_result:
                    try:
                        on_tool_result(tool_name, len(content_str))
                    except Exception:
                        pass

                # ★ KEY: append tool result to the SHARED messages list
                # This means the next step's LLM call sees what was on the page
                shared_messages.append({
                    "role": "tool",
                    "name": tool_name,
                    "content": content_str
                })

            tool_round += 1
            # Loop back → LLM continues with the tool results in context

        # Append final assistant response to shared thread
        if full_text.strip():
            shared_messages.append({"role": "assistant", "content": full_text.strip()})

    # ─── Public API ───────────────────────────────────────────────────────────

    def generate_streaming(
        self,
        user_text: str,
        mcp_client=None,
        interrupt_check=None,
        on_tool_call=None,
        on_tool_result=None,
        on_plan=None,
        on_step=None,
        on_step_done=None,
        on_step_retry=None,
        on_step_failed=None,
        on_task_retry=None,
    ) -> Generator[str, None, None]:
        """
        Main entry.

        SESSION CONTINUITY:
          - `shared_messages` is a single growing list for the entire task.
          - Every step's user prompt, LLM response, tool calls, and tool results
            are appended to it. So step 3 can see what step 1 navigated to.
          - The Playwright MCP server keeps the same browser process alive across
            all tool calls (one persistent session per agent run).
        """
        start_time = time.perf_counter()

        # Tool gating
        enable_tools = False
        if mcp_client:
            user_lower = user_text.lower()
            tool_triggers = [
                "search", "lookup", "look up", "find", "google", "web", "internet",
                "browse", "browser", "navigate", "open page", "go to", "url", "website",
                "weather", "forecast", "stock", "stocks", "price", "who won", "score",
                "click", "screenshot", "scrape", "download", "current time",
                "today's news", "latest news", "breaking news",
                "play", "watch", "show me", "get the", "find me",
            ]
            enable_tools = any(k in user_lower for k in tool_triggers)

        llm_tools = mcp_client.convert_tools_for_llm() if (mcp_client and enable_tools) else None
        full_response_text = ""

        # ── Decide: plan or direct ─────────────────────────────────────────────
        needs_plan = self._is_complex_task(user_text)
        steps = None
        if needs_plan:
            steps = self._plan_task(user_text)
            if not steps or steps == ["respond directly"]:
                steps = None

        if not steps:
            # ── DIRECT RESPONSE ────────────────────────────────────────────────
            shared_messages = [{"role": "system", "content": LLM_SYSTEM_PROMPT}]
            shared_messages.extend(self.history[-self._max_history:])

            for sentence in self._execute_step_stream(
                shared_messages, user_text, llm_tools,
                interrupt_check, on_tool_call, on_tool_result, mcp_client
            ):
                yield sentence
                full_response_text += sentence + " "

        else:
            # ── PLAN-EXECUTE-VERIFY ────────────────────────────────────────────
            task_attempt = 0
            task_success = False

            while task_attempt <= MAX_TASK_RETRIES and not task_success:
                if interrupt_check and interrupt_check():
                    break

                if task_attempt > 0:
                    if on_task_retry:
                        try:
                            on_task_retry(task_attempt)
                        except Exception:
                            pass
                    steps = self._plan_task(
                        f"{user_text}\n\nPrevious attempt failed on some steps. "
                        f"Re-plan with more specific, atomic steps using exact URLs and element descriptions."
                    ) or steps

                if on_plan:
                    try:
                        on_plan(steps)
                    except Exception:
                        pass

                # ── ONE shared message thread for the ENTIRE task ──────────────
                # This is what gives the browser session continuity.
                shared_messages = [{"role": "system", "content": LLM_SYSTEM_PROMPT}]
                shared_messages.extend(self.history[-self._max_history:])

                # Inject full plan as context
                plan_context = (
                    f"The Citizen has requested: {user_text}\n\n"
                    f"Mission plan ({len(steps)} steps — execute in order):\n"
                    + "\n".join(f"  {i+1}. {s}" for i, s in enumerate(steps))
                    + "\n\nIMPORTANT: Each step builds on the previous one. "
                    + "Browser state (current page, session, cookies) is preserved between steps."
                )
                shared_messages.append({"role": "system", "content": plan_context})

                failed_steps = 0

                for idx, step in enumerate(steps):
                    if interrupt_check and interrupt_check():
                        break

                    if on_step:
                        try:
                            on_step(idx, step)
                        except Exception:
                            pass

                    # ── Per-step retry loop ────────────────────────────────────
                    step_succeeded = False
                    step_text_accumulated = ""
                    msgs_checkpoint = len(shared_messages)  # save point for retry revert

                    for attempt in range(MAX_STEP_RETRIES + 1):
                        if interrupt_check and interrupt_check():
                            break

                        if attempt == 0:
                            prompt = (
                                f"Execute step {idx + 1} of {len(steps)}: {step}\n"
                                f"Use the appropriate browser tool if needed. "
                                f"Speak your result in 1-2 short sentences. "
                                f"The browser session is persistent — you are on whatever page the previous step left."
                            )
                        else:
                            # Revert shared_messages to checkpoint to avoid polluting
                            # thread with the failed attempt's tool calls
                            del shared_messages[msgs_checkpoint:]
                            prompt = (
                                f"RETRY {attempt}/{MAX_STEP_RETRIES} — step {idx + 1} verification failed.\n"
                                f"Step goal: {step}\n"
                                f"Previous attempt did not complete this step. Try a different approach:\n"
                                f"- Use browser_snapshot first to see current page state\n"
                                f"- Use more specific element selectors\n"
                                f"- Try browser_find before browser_click\n"
                                f"Speak your result in 1-2 sentences."
                            )
                            if on_step_retry:
                                try:
                                    on_step_retry(idx, attempt, step)
                                except Exception:
                                    pass
                            time.sleep(STEP_RETRY_WAIT)

                        # Reset per-attempt accumulation
                        step_text_accumulated = ""
                        attempt_start = len(shared_messages)

                        for sentence in self._execute_step_stream(
                            shared_messages, prompt, llm_tools,
                            interrupt_check, on_tool_call, on_tool_result, mcp_client
                        ):
                            if interrupt_check and interrupt_check():
                                break
                            yield sentence
                            step_text_accumulated += sentence + " "
                            full_response_text += sentence + " "

                        # ── Verify ─────────────────────────────────────────────
                        step_succeeded = self._verify_step(
                            step,
                            step_text_accumulated.strip(),
                            mcp_client
                        )

                        if step_succeeded:
                            break

                        # If last attempt, mark failed
                        if attempt == MAX_STEP_RETRIES:
                            if on_step_failed:
                                try:
                                    on_step_failed(idx, step)
                                except Exception:
                                    pass
                            failed_steps += 1

                    if on_step_done:
                        try:
                            on_step_done(idx, step_succeeded)
                        except Exception:
                            pass

                # Task success if fewer than half steps failed
                task_success = failed_steps < max(1, len(steps) // 2)
                task_attempt += 1

        # ── History update ─────────────────────────────────────────────────────
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": full_response_text.strip()})
        if len(self.history) > self._max_history * 2:
            self.history = self.history[-self._max_history * 2:]

        elapsed = (time.perf_counter() - start_time) * 1000
        print(f"  [LLM] Done ({elapsed:.0f}ms) | steps={len(steps) if steps else 'direct'} | history={len(self.history)}")

    def clear_history(self):
        """Clear conversation history."""
        self.history.clear()
