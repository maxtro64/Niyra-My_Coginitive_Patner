"""Filter between raw Playwright tool output and the LLM. Pure code, no LLM calls.

For every tool call it produces BOTH things the agent loop needs:
  observation -> compact useful result, shown to the LLM while the step is recent
  summary     -> one line, replaces the observation once the step is old

Assumes Playwright-MCP-style snapshots (lines like `- button "Buy now" [ref=e12]`).
If your tools print a different format, only parse_page() needs changing.
"""
import os
import re
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

MAX_STEPS_KEPT = 100  # hard cap on stored steps (each step = tool call + its result)
MAX_OBS_CHARS = 3000
MAX_ELEMENTS = 40
MAX_HEADINGS = 6
MAX_TEXTS = 20  # plain text lines kept; price lines are always kept first
MAX_TEXT_CHARS = 2000
MAX_SUMMARY_CHARS = 160

_ROLES = ("link", "button", "textbox", "searchbox", "combobox", "checkbox", "radio", "menuitem", "tab", "option")
_ELEM = re.compile(r'^\s*-\s*(' + "|".join(_ROLES) + r')\s+"([^"]*)"(.*)$')
_REF = re.compile(r"\[ref=([\w-]+)\]")
_HEADING = re.compile(r'^\s*-\s*heading\s+"([^"]*)"')
_HREF = re.compile(r"^\s*-\s*/url:\s*(\S+)")
_PAGE_URL = re.compile(r"Page URL:\s*(\S+)")
_PAGE_TITLE = re.compile(r"Page Title:\s*(.+)")
_TEXT = re.compile(r"^\s*-\s*text:\s*(.+)$")
_PRICE = re.compile(r"[₹$€£]\s?\d|\bRs\.?\s?\d")
_ARG_KEYS = ("url", "element", "ref", "selector", "text", "query", "key", "value")


# ─── Output models ────────────────────────────────────────────
class Filtered(BaseModel):
    observation: str
    summary: str
    url: str | None = None
    ok: bool = True


class Step(BaseModel):
    """One tool call, stored once, shown in two sizes."""

    assistant_msg: dict  # the LLM's tool-call message
    call_id: str
    observation: str
    summary: str
    url: str | None = None
    ok: bool = True


class StepLog(BaseModel):
    """Capped history. Drops the oldest WHOLE steps, so a tool call never loses its result."""

    steps: list[Step] = Field(default_factory=list)
    dropped: int = 0  # how many old steps were trimmed away

    def add(self, step: Step, keep: int = MAX_STEPS_KEPT) -> "StepLog":
        steps = [*self.steps, step]
        extra = max(0, len(steps) - keep)
        return StepLog(steps=steps[extra:], dropped=self.dropped + extra)


# ─── Small helpers ────────────────────────────────────────────
def _clip(s: object, n: int) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _trim(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + f"\n…[trimmed {len(s) - n} chars]"


def _squash(raw: str) -> str:
    raw = re.sub(r"[ \t]+", " ", raw)
    return re.sub(r"\n{3,}", "\n\n", raw).strip()


def _tool_name(tool: str) -> str:
    return tool.split("__")[-1].removeprefix("browser_")


def _fmt_args(args: dict) -> str:
    return ", ".join(f"{k}={_clip(args[k], 40)!r}" for k in _ARG_KEYS if k in args)


def _looks_like_error(raw: str) -> bool:
    return raw.lstrip().lower().startswith(("error", "### error"))


def _first_line(raw: str) -> str:
    for line in raw.splitlines():
        line = line.strip()
        if line and not line.startswith("###"):
            return _clip(line, 300)
    return ""


# ─── Page parsing ─────────────────────────────────────────────
@dataclass
class Element:
    role: str
    name: str
    ref: str = ""
    href: str = ""


@dataclass
class Page:
    url: str = ""
    title: str = ""
    headings: list[str] = field(default_factory=list)
    elements: list[Element] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return bool(self.url or self.title or self.headings or self.elements or self.texts)


def parse_page(raw: str) -> Page:
    page = Page()
    if m := _PAGE_URL.search(raw):
        page.url = m.group(1)
    if m := _PAGE_TITLE.search(raw):
        page.title = m.group(1).strip()
        
    # Extract YAML snapshot content if referenced
    snap_match = re.search(r'\[Snapshot\]\((.*?\.ya?ml)\)', raw, re.IGNORECASE)
    if snap_match:
        snap_path = snap_match.group(1)
        if os.path.exists(snap_path):
            with open(snap_path, 'r', encoding='utf-8') as f:
                raw += "\n" + f.read()

    seen: set[tuple[str, str]] = set()
    last: Element | None = None
    for line in raw.splitlines():
        if m := _ELEM.match(line):
            role, name, rest = m.groups()
            if not name.strip():
                last = None
                continue
            ref_match = _REF.search(rest)
            ref_id = ref_match.group(1) if ref_match else ""
            
            dedup_key = ref_id if ref_id else (role, name)
            if dedup_key in seen:
                last = None
                continue
            seen.add(dedup_key)
            
            last = Element(role, _clip(name, 80), ref_id)
            page.elements.append(last)
        elif (m := _HREF.match(line)) and last and last.role == "link" and not last.href:
            last.href = _clip(m.group(1), 90)
        elif (m := _HEADING.match(line)) and len(page.headings) < MAX_HEADINGS:
            page.headings.append(_clip(m.group(1), 80))
        elif m := _TEXT.match(line):
            t = _clip(m.group(1), 100)
            if t not in page.texts:
                page.texts.append(t)
    return page


def _pick_texts(texts: list[str]) -> list[str]:
    """Keep price lines first, fill the rest in page order."""
    keep = {i for i, t in enumerate(texts) if _PRICE.search(t)}
    keep = set(sorted(keep)[: MAX_TEXTS // 2])
    for i in range(len(texts)):
        if len(keep) >= MAX_TEXTS:
            break
        keep.add(i)
    return [texts[i] for i in sorted(keep)]


def format_page(page: Page) -> str:
    lines: list[str] = []
    head = " ".join(x for x in (page.title, f"({page.url})" if page.url else "") if x)
    if head:
        lines.append(f"Page: {head}")
    if page.headings:
        lines.append("Headings: " + " | ".join(page.headings))
    shown = page.elements[:MAX_ELEMENTS]
    if shown:
        lines.append(f"Elements ({len(shown)} of {len(page.elements)}):")
        for e in shown:
            ref = f"[{e.ref}] " if e.ref else ""
            href = f" -> {e.href}" if e.href else ""
            lines.append(f'{ref}{e.role} "{e.name}"{href}')
        if len(page.elements) > len(shown):
            lines.append(f"(+{len(page.elements) - len(shown)} more not shown)")
    if page.texts:
        lines.append("Text:")
        lines.extend(f"- {t}" for t in _pick_texts(page.texts))
    return "\n".join(lines)


# ─── The filter ───────────────────────────────────────────────
def filter_result(tool: str, args: dict | None, raw: str | None, is_error: bool = False) -> Filtered:
    """Turn one raw tool result into (observation, summary)."""
    raw = raw or ""
    call = f"{_tool_name(tool)}({_fmt_args(args or {})})"

    if is_error or _looks_like_error(raw):
        why = _first_line(raw) or "unknown error"
        return Filtered(
            observation=f"ERROR in {call}: {why}",
            summary=_clip(f"{call} -> FAILED: {why}", MAX_SUMMARY_CHARS),
            ok=False,
        )

    page = parse_page(raw)
    if page.found:
        obs = format_page(page)
        detail = f"on '{_clip(page.title or page.url, 50)}', {len(page.elements)} elements"
    else:
        text = _squash(raw)
        obs = _trim(text, MAX_TEXT_CHARS) or "(empty result)"
        detail = f"{len(text)} chars" if text else "empty result"

    return Filtered(
        observation=_trim(obs, MAX_OBS_CHARS),
        summary=_clip(f"{call} -> ok, {detail}", MAX_SUMMARY_CHARS),
        url=page.url if page.found else None,
    )


def make_step(assistant_msg: dict, call_id: str, tool: str, args: dict | None,
              raw: str | None, is_error: bool = False) -> Step:
    """Call this in the executor right after each tool runs."""
    f = filter_result(tool, args, raw, is_error)
    return Step(assistant_msg=assistant_msg, call_id=call_id, **f.model_dump())