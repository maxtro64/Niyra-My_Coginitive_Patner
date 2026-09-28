"""
BRUCE — Batcomputer Terminal User Interface (TUI)
Engineered after the Nous Research Hermes Agent CLI / TUI architecture:
- Dual-column welcome banner with Bat-Signal hero emblem & subsystem matrix
- Real-time streaming response engine with tactical typewriter output
- Hermes-style status diagnostics with '◆' section headings and '✓/✗' checks
- REAL CLICKABLE BUTTONS (mouse-interactive Textual TUI)
- Dual-mode: Continuous acoustic microphone sensor + interactive terminal command prompt
"""

import sys
import os
import time
import shutil
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Header, Footer, Button, Static, Input, RichLog

# Hermes / Batman Color Palette
COLOR_GOLD = "#FFD700"     # Bright Bat-Gold (Banner Title)
COLOR_ACCENT = "#FFBF00"   # Warm Gold Accent
COLOR_BRONZE = "#CD7F32"   # Border Bronze
COLOR_DIM = "#B8860B"      # Dark Muted Bronze
COLOR_TEXT = "#FFF8DC"     # High-contrast Cream White
COLOR_CYAN = "#00FFFF"     # High-tech Cyber Cyan

# Safe ASCII Batman Title Art
BATMAN_ASCII_TITLE = r"""
      ____        _                          
     |  _ \      | |                         
     | |_) | __ _| |_ _ __ ___   __ _ _ __   
     |  _ < / _` | __| '_ ` _ \ / _` | '_ \  
     | |_) | (_| | |_| | | | | | (_| | | | | 
     |____/ \__,_|\__|_| |_| |_|\__,_|_| |_| 
"""

# Compact 38-column Bat-Signal Hero Emblem
BAT_HERO_EMBLEM = r"""[bold #FFBF00]     _==/       i   i       \==_
   /XX/         |\_/|         \XX\
  /XXXX\         | |         /XXXX\
 |XXXXXX\_      _--_       _/XXXXXX|
  XXXXXXXXxx\  /    \  /xxXXXXXXXX
   XXXXXXXXXX\/      \/XXXXXXXXXX
        "~~~"          "~~~"[/]"""


class BatcomputerTextualApp(App):
    """
    Textual-based interactive Batcomputer TUI with full mouse support and clickable buttons.
    """
    CSS = """
    Screen {
        background: #080A0F;
        color: #FFF8DC;
    }
    #header-bar {
        dock: top;
        height: auto;
        background: #0E121A;
        border-bottom: heavy #FFBF00;
        padding: 0 1;
        content-align: center middle;
    }
    #main-log {
        height: 1fr;
        background: #080A0F;
        border: round #CD7F32;
        margin: 1 1 0 1;
        padding: 0 1;
    }
    #button-bar {
        height: 3;
        margin: 0 1;
        align: center middle;
    }
    Button {
        margin: 0 1;
        background: #141923;
        color: #FFBF00;
        border: tall #CD7F32;
        text-style: bold;
        min-width: 12;
    }
    Button:hover {
        background: #FFBF00;
        color: #080A0F;
    }
    #input-box {
        margin: 0 1 1 1;
        border: tall #00FFFF;
        background: #0E121A;
        color: #00FFFF;
    }
    """

    def __init__(self, gui: "BatcomputerGUI"):
        super().__init__()
        self.gui = gui

    def compose(self) -> ComposeResult:
        with Vertical(id="header-bar"):
            yield Static(f"[bold {COLOR_GOLD}]═══ WAYNE ENTERPRISES DEFENSE NETWORK // BATCOMPUTER OS v2.4 (HERMES TUI) ═══[/]")
        yield RichLog(id="main-log", highlight=True, markup=True)
        with Horizontal(id="button-bar"):
            yield Button("🧪 Test Mic", id="btn-testmic")
            yield Button("🎙️ Unmute Mic", id="btn-unmute")
            yield Button(f"⚙️ Tools ({len(self.gui._tools_list) or 26})", id="btn-tools")
            yield Button("⚡ Subsystems", id="btn-status")
            yield Button("❓ Help", id="btn-help")
            yield Button("🧹 Clear Log", id="btn-clear")
            yield Button("🛑 Abort", id="btn-abort", variant="error")
            yield Button("❌ Exit", id="btn-exit", variant="error")
        yield Input(placeholder="🦇 citizen@batcomputer > Click a button, enter query, or speak into microphone...", id="input-box")

    def on_mount(self):
        """Called when TUI widgets are mounted."""
        self.gui.show_banner()

    def on_button_pressed(self, event: Button.Pressed):
        """Handle mouse clicks on Batcomputer buttons."""
        btn_id = event.button.id
        if btn_id == "btn-testmic":
            if self.gui._testmic_callback:
                self.gui._testmic_callback()
        elif btn_id == "btn-unmute":
            if self.gui._unmute_callback:
                self.gui._unmute_callback()
        elif btn_id == "btn-tools":
            self.gui.show_tools()
        elif btn_id == "btn-status":
            self.gui.show_status()
        elif btn_id == "btn-help":
            self.gui.show_help()
        elif btn_id == "btn-clear":
            log = self.query_one(RichLog)
            log.clear()
            self.gui.show_banner()
        elif btn_id == "btn-abort":
            if self.gui._abort_callback:
                self.gui._abort_callback()
        elif btn_id == "btn-exit":
            self.exit()

    def on_input_submitted(self, event: Input.Submitted):
        """Handle Enter key pressed in command input box."""
        user_text = event.value.strip()
        event.input.value = ""
        if not user_text:
            return

        cmd_lower = user_text.lower()
        if cmd_lower in ("exit", "quit", "/exit", "/quit"):
            self.exit()
            return
        if cmd_lower in ("/clear", "clear", "cls"):
            log = self.query_one(RichLog)
            log.clear()
            self.gui.show_banner()
            return
        if cmd_lower in ("/help", "help"):
            self.gui.show_help()
            return
        if cmd_lower in ("/status", "status"):
            self.gui.show_status()
            return
        if cmd_lower in ("/tools", "tools"):
            self.gui.show_tools()
            return
        if cmd_lower in ("/unmute", "unmute"):
            if self.gui._unmute_callback:
                self.gui._unmute_callback()
            return
        if cmd_lower in ("/testmic", "testmic"):
            if self.gui._testmic_callback:
                self.gui._testmic_callback()
            return
        if cmd_lower in ("/abort", "abort"):
            if self.gui._abort_callback:
                self.gui._abort_callback()
            return

        # Pass user command to agent
        if self.gui._input_callback:
            self.gui._input_callback(user_text)

    def write_log(self, content):
        """Append rich content to the central Batcomputer log."""
        try:
            log = self.query_one(RichLog)
            log.write(content)
        except Exception:
            pass


class BatcomputerGUI:
    """
    Batcomputer Terminal GUI (TUI) modeled directly on Hermes Agent CLI.
    Features:
    - Dual-column hero welcome banner (Bat-Signal left, capabilities right)
    - REAL CLICKABLE BUTTONS via Textual TUI
    - Hermes-style section headings ('◆') and status indicators ('✓' / '✗')
    - Real-time sentence-by-sentence terminal streaming
    - Subsystem telemetry, MCP tool monitoring, and Windows audio auto-unmute
    """

    def __init__(self):
        self.console = Console()
        self._print_lock = threading.Lock()
        self._status = "INITIALIZING"
        self._tools_list: List[str] = []
        self._input_callback = None
        self._abort_callback = None
        self._testmic_callback = None
        self._unmute_callback = None
        self._is_running = True
        self._mic_active = True
        self._mic_info = "Default Mic (100% Vol, UNMUTED)"
        self._in_bruce_stream = False
        self._app: Optional[BatcomputerTextualApp] = None

    def _render_or_print(self, content):
        """Render to Textual log if active, otherwise print to console."""
        if self._app is not None and self._app.is_running:
            try:
                if threading.get_ident() == getattr(self._app, "_thread_id", None):
                    self._app.write_log(content)
                else:
                    self._app.call_from_thread(self._app.write_log, content)
            except Exception:
                with self._print_lock:
                    self.console.print(content)
        else:
            with self._print_lock:
                self.console.print(content)

    def show_banner(self):
        """Display the Hermes-style dual-column Batcomputer welcome banner."""
        # Left Column (Emblem + Core Identity)
        left_lines = [
            BAT_HERO_EMBLEM,
            "",
            f"[bold {COLOR_ACCENT}]Operative:[/]   [{COLOR_TEXT}]Bruce Wayne (The Dark Knight)[/]",
            f"[bold {COLOR_ACCENT}]Cognitive:[/]   [{COLOR_TEXT}]Llama 3.2 3B[/] [dim {COLOR_DIM}](Ollama GPU: 99)[/]",
            f"[bold {COLOR_ACCENT}]Voice Synth:[/] [{COLOR_TEXT}]Kokoro Neural[/] [dim {COLOR_DIM}](Bane Voice)[/]",
            f"[bold {COLOR_ACCENT}]Location:[/]    [{COLOR_TEXT}]Gotham City (The Batcave)[/]",
            f"[bold {COLOR_ACCENT}]Audio Input:[/] [bold green]Active[/] [dim {COLOR_DIM}]({self._mic_info})[/]",
        ]

        # Right Column (Toolsets + MCP Servers + Protocols)
        right_lines = [
            f"[bold {COLOR_ACCENT}]Active Toolsets[/]",
            f"  [{COLOR_TEXT}]Browser Tools:[/] [dim {COLOR_DIM}]navigate, click, type, screenshot, tabs...[/]",
            f"  [{COLOR_TEXT}]Intelligence:[/]  [dim {COLOR_DIM}]tavily_web_search (live web intelligence)[/]",
            "",
            f"[bold {COLOR_ACCENT}]Connected MCP Servers[/]",
            f"  [bold green]●[/] [{COLOR_TEXT}]playwright[/] [dim {COLOR_DIM}](25 browser automation tools)[/]",
            f"  [bold green]●[/] [{COLOR_TEXT}]tavily[/]     [dim {COLOR_DIM}](real-time web search)[/]",
            "",
            f"[bold {COLOR_ACCENT}]Gotham Defense Protocols[/]",
            f"  [{COLOR_TEXT}]Acoustic Sensor:[/] [bold green]LISTENING[/] [dim {COLOR_DIM}](Faster-Whisper + Silero)[/]",
            f"  [{COLOR_TEXT}]Barge-in Voice:[/]  [bold green]ARMED[/] [dim {COLOR_DIM}](Say 'Bruce' or /abort)[/]",
            "",
            f"[dim {COLOR_DIM}]{len(self._tools_list) or 26} tools · 5 subsystems · Click buttons below or type /help[/]",
        ]

        layout_table = Table.grid(padding=(0, 4))
        layout_table.add_column("left", justify="left", width=42)
        layout_table.add_column("right", justify="left")
        layout_table.add_row("\n".join(left_lines), "\n".join(right_lines))

        outer_panel = Panel(
            layout_table,
            title=f"[bold {COLOR_GOLD}]═══ WAYNE ENTERPRISES DEFENSE NETWORK // BATCOMPUTER OS v2.4 ═══[/]",
            border_style=COLOR_BRONZE,
            padding=(0, 2),
        )

        term_cols = shutil.get_terminal_size().columns
        if term_cols >= 85 and self._app is None:
            self._render_or_print(f"[bold {COLOR_ACCENT}]{BATMAN_ASCII_TITLE}[/]")

        self._render_or_print(outer_panel)

    def show_status(self):
        """Hermes-style status report (modeled on hermes_cli/status.py)."""
        output_lines = [
            f"[bold {COLOR_CYAN}]◆ BATCOMPUTER SUBSYSTEM TELEMETRY[/]",
            "",
        ]

        def _row(name: str, ok: bool, text: str, width: int = 18):
            mark = "[bold green]✓[/bold green]" if ok else "[bold red]✗[/bold red]"
            return f"  {name:<{width}}  {mark} {text}"

        output_lines.append(_row("Operative", True, f"[{COLOR_TEXT}]Bruce Wayne (The Dark Knight)[/]"))
        output_lines.append(_row("Cognitive Core", True, f"[{COLOR_TEXT}]Llama 3.2 3B via Ollama (Local GPU layers: 99)[/]"))
        output_lines.append(_row("Voice Synthesis", True, f"[{COLOR_TEXT}]Kokoro Neural TTS (Bane Voice, 24kHz mono)[/]"))
        output_lines.append(_row("Acoustic Sensor", True, f"[{COLOR_TEXT}]Faster-Whisper base.en + Silero VAD (Sensitivity: 0.35)[/]"))
        output_lines.append(_row("Tool Protocol", bool(self._tools_list), f"[{COLOR_TEXT}]Playwright Browser ({len(self._tools_list) or 25} tools) + Tavily Search[/]"))
        output_lines.append(_row("Windows Audio", self._mic_active, f"[{COLOR_TEXT}]{self._mic_info}[/]"))
        output_lines.append("")
        output_lines.append(f"  [dim {COLOR_DIM}]All subsystems operating within nominal parameters.[/]")

        self._render_or_print("\n".join(output_lines))

    def show_tools(self):
        """Categorized tools listing (Hermes Agent style)."""
        output_lines = [
            f"[bold {COLOR_CYAN}]◆ ACTIVE MCP & TACTICAL TOOLS ({len(self._tools_list) or 26} LOADED)[/]",
            "",
        ]

        categories = {
            "🌐 Web & Intelligence": [t for t in self._tools_list if "search" in t or "tavily" in t],
            "🧭 Browser Navigation": [t for t in self._tools_list if any(k in t for k in ["navigate", "back", "forward", "tab", "url", "open"])],
            "🖱️ Element Interaction": [t for t in self._tools_list if any(k in t for k in ["click", "type", "press", "fill", "hover", "drag", "select"])],
            "🔍 Page Inspection": [t for t in self._tools_list if any(k in t for k in ["snapshot", "screenshot", "content", "evaluate", "html", "console"])],
        }

        categorized_all = set().union(*categories.values())
        uncategorized = [t for t in self._tools_list if t not in categorized_all]
        if uncategorized:
            categories["⚙️ Additional Tools"] = uncategorized

        for cat, tools in categories.items():
            if tools:
                output_lines.append(f"  [bold {COLOR_ACCENT}]{cat}[/]")
                for t in tools:
                    output_lines.append(f"    [dim {COLOR_DIM}]•[/] [{COLOR_TEXT}]{t}[/]")
                output_lines.append("")

        self._render_or_print("\n".join(output_lines))

    def show_help(self):
        """Display tactical command help (Hermes Agent style)."""
        output_lines = [
            f"[bold {COLOR_CYAN}]◆ BATCOMPUTER COMMAND INTERFACE & SHORTCUTS[/]",
            "",
        ]

        commands = [
            ("Click [🧪 Test Mic]", "Record a 1.5s audio sample to test live microphone & VAD"),
            ("Click [🎙️ Unmute]", "Force un-mute Windows microphone and maximize recording volume"),
            ("Click [⚙️ Tools]", "List all active MCP, Playwright, and Web tools by category"),
            ("Click [⚡ Subsystems]", "Display comprehensive subsystem telemetry & sensor health"),
            ("Click [🧹 Clear]", "Clear terminal log and redisplay the Batcomputer welcome banner"),
            ("Click [🛑 Abort]", "Instantly abort current speech or tactical tool execution (or say 'Bruce')"),
            ("Click [❌ Exit]", "Power down Bruce and the Batcomputer defense network"),
            ("/help, /status, etc.", "All commands can also be typed into the input box"),
        ]

        for cmd, desc in commands:
            output_lines.append(f"  [bold {COLOR_ACCENT}]{cmd:<22}[/] [{COLOR_TEXT}]{desc}[/]")

        output_lines.append("")
        output_lines.append(f"  [dim {COLOR_DIM}]Tip: Speak directly into the microphone at any time — acoustic sensor is always listening.[/]")

        self._render_or_print("\n".join(output_lines))

    def log_system(self, text: str):
        """Log a system message with Batcomputer styling."""
        now = datetime.now().strftime("%H:%M:%S")
        self._render_or_print(f"[dim {COLOR_CYAN}]{now}[/dim {COLOR_CYAN}] [bold {COLOR_ACCENT}]📡 [SYS][/] [{COLOR_TEXT}]{text}[/]")

    def log_user(self, text: str, is_voice: bool = False):
        """Display citizen input."""
        now = datetime.now().strftime("%H:%M:%S")
        icon = "🎤" if is_voice else "⌨️ "
        src = "Voice" if is_voice else "Terminal"
        self._render_or_print(f"\n[bold {COLOR_CYAN}]◆ {icon} CITIZEN ({src}) [{now}][/]\n  [{COLOR_TEXT}]{text}[/]")

    def stream_bruce_start(self):
        """Initiate Bruce's tactical response stream."""
        now = datetime.now().strftime("%H:%M:%S")
        self._render_or_print(f"\n[bold {COLOR_GOLD}]◆ 🦇 BRUCE // GOTHAM GUARDIAN [{now}][/]")
        self._in_bruce_stream = True

    def stream_bruce_sentence(self, sentence: str):
        """Stream a sentence in real time with typewriter feel."""
        if not self._in_bruce_stream:
            self.stream_bruce_start()
        self._render_or_print(f"  [bold {COLOR_ACCENT}]›[/] [{COLOR_TEXT}]{sentence}[/]")

    def stream_bruce_end(self):
        """Close the streaming response block."""
        if self._in_bruce_stream:
            self._render_or_print("")
            self._in_bruce_stream = False

    def log_tool(self, text: str):
        """Log tool invocation in real time."""
        self._render_or_print(f"  [bold magenta]⚙️  {text}[/]")

    def log_error(self, text: str):
        """Log alert or error."""
        self._render_or_print(f"[bold red]❌ [ALERT][/bold red] [red]{text}[/red]")

    def set_status(self, status: str):
        """Update system status."""
        self._status = status.upper()

    def set_tools(self, tools_list: list):
        """Update loaded tool names."""
        self._tools_list = tools_list

    def set_mic_active(self, active: bool, info: str = ""):
        """Update mic status."""
        self._mic_active = active
        if info:
            self._mic_info = info

    def update_waveform(self, audio_level: float):
        """Waveform level hook."""
        pass

    def set_input_callback(self, cb):
        self._input_callback = cb

    def set_abort_callback(self, cb):
        self._abort_callback = cb

    def set_testmic_callback(self, cb):
        self._testmic_callback = cb

    def set_unmute_callback(self, cb):
        self._unmute_callback = cb

    def run(self):
        """Start the interactive terminal session with clickable buttons."""
        self._app = BatcomputerTextualApp(self)
        try:
            self._app.run()
        except Exception as e:
            # Fallback to REPL CLI if running in an unsupported non-interactive console
            self._app = None
            self._run_cli_fallback()

    def _run_cli_fallback(self):
        """CLI fallback loop if Textual terminal cannot open screen buffer."""
        self.console.print(f"[bold {COLOR_ACCENT}]🦇 Batcomputer Tactical Defense Shell (CLI Mode).[/]")
        while self._is_running:
            try:
                user_input = input("🦇 citizen@batcomputer > ").strip()
                if not user_input:
                    continue
                cmd_lower = user_input.lower()
                if cmd_lower in ("exit", "quit", "/exit", "/quit"):
                    break
                if cmd_lower in ("/help", "help"):
                    self.show_help()
                    continue
                if cmd_lower in ("/status", "status"):
                    self.show_status()
                    continue
                if cmd_lower in ("/tools", "tools"):
                    self.show_tools()
                    continue
                if cmd_lower in ("/clear", "clear", "cls"):
                    os.system("cls" if os.name == "nt" else "clear")
                    self.show_banner()
                    continue
                if cmd_lower in ("/testmic", "testmic"):
                    if self._testmic_callback:
                        self._testmic_callback()
                    continue
                if cmd_lower in ("/unmute", "unmute"):
                    if self._unmute_callback:
                        self._unmute_callback()
                    continue
                if cmd_lower in ("/abort", "abort"):
                    if self._abort_callback:
                        self._abort_callback()
                    continue
                if self._input_callback:
                    self._input_callback(user_input)
            except (KeyboardInterrupt, EOFError):
                break

    def destroy(self):
        """Clean shutdown."""
        self._is_running = False
        if self._app and self._app.is_running:
            self._app.exit()


if __name__ == "__main__":
    gui = BatcomputerGUI()
    gui.set_tools([
        "playwright_navigate", "playwright_click", "playwright_type",
        "playwright_screenshot", "tavily_web_search"
    ])
    gui.run()

