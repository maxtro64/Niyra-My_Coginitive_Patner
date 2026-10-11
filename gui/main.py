"""
BRUCE — Batcomputer Terminal User Interface (TUI)
Engineered after the Nous Research Hermes Agent CLI / TUI architecture:
- Dual-column welcome banner with Bat-Signal hero emblem & subsystem matrix
- Real-time streaming response engine with tactical typewriter output
- Hermes-style status diagnostics with '◆' section headings and '✓/✗' checks
- REAL CLICKABLE BUTTONS (mouse-interactive Textual TUI)
- Dual-mode: Continuous acoustic microphone sensor + interactive terminal command prompt

Safety notes:
- Every piece of dynamic text (LLM output, user text, errors) goes through rich.markup.escape,
  so a stray "[/x]" in a response can no longer raise MarkupError and silently drop the line.
- print() output from stt / tts / llm is routed into the log while the TUI is running.
"""

import logging
import os
import shutil
import sys
import threading
import traceback
from datetime import datetime

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8") # type: ignore
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8") # type: ignore
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Input, RichLog, Static

from core.config import (
    AUDIO_SAMPLE_RATE,
    LLM_MODEL,
    STT_MODEL_SIZE,
    TTS_ENGINE,
    TTS_VOICE,
    VAD_THRESHOLD,
)


class BatcomputerLogHandler(logging.Handler):
    """Routes standard Python logging output cleanly into the Batcomputer RichLog."""
    def __init__(self, gui: "BatcomputerGUI"):
        super().__init__()
        self.gui = gui

    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)
            if not msg or not msg.strip():
                return
            if record.levelno >= logging.ERROR:
                self.gui.log_error(f"[{record.name}] {msg}")
            elif record.levelno >= logging.WARNING:
                self.gui.log_system(f"⚠️ [{record.name}] {msg}")
            else:
                self.gui.log_system(f"[{record.name}] {msg}")
        except Exception:
            pass

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


def _voice_label() -> str:
    return os.path.basename(str(TTS_VOICE))


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
    #button-container {
        height: auto;
        margin: 0 1;
        align: center middle;
    }
    .button-row {
        height: 3;
        align: center middle;
    }
    #status-line {
        height: 1;
        margin: 0 2;
        color: #B8860B;
    }
    Button {
        margin: 0 1;
        background: #141923;
        color: #FFBF00;
        border: tall #CD7F32;
        text-style: bold;
        min-width: 14;
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
        with Vertical(id="button-container"):
            with Horizontal(classes="button-row"):
                yield Button("🧪 Test Mic", id="btn-testmic")
                yield Button("🎙️ Unmute Mic", id="btn-unmute")
                yield Button(f"⚙️ Tools ({len(self.gui._tools_list)})", id="btn-tools")
                yield Button("⚡ Subsystems", id="btn-status")
            with Horizontal(classes="button-row"):
                yield Button("❓ Help", id="btn-help")
                yield Button("🧹 Clear Log", id="btn-clear")
                yield Button("🛑 Abort", id="btn-abort", variant="error")
                yield Button("❌ Exit", id="btn-exit", variant="error")
        yield Static("", id="status-line")
        yield Input(placeholder="🦇 citizen@batcomputer > Click a button, enter query, or speak into microphone...", id="input-box")

    def on_mount(self):
        """Called when TUI widgets are mounted."""
        self.gui._app = self
        try:
            self.begin_capture_print(self)
        except Exception:
            pass
        self.gui.show_banner()
        self.set_status_text(self.gui._status)
        self.update_mic_button(self.gui._mic_active)
        self.update_tools_button(len(self.gui._tools_list))

    def on_print(self, event: events.Print) -> None:
        """Captured print() output with ANSI color parsing."""
        text = event.text.rstrip()
        if text:
            try:
                self.write_log(Text.from_ansi(text))
            except Exception:
                self.write_log(Text(text))

    def on_button_pressed(self, event: Button.Pressed):
        """Handle mouse clicks on Batcomputer buttons."""
        btn_id = event.button.id
        if btn_id == "btn-testmic":
            if self.gui._testmic_callback:
                threading.Thread(target=self.gui._testmic_callback, daemon=True).start()
            else:
                threading.Thread(target=self.gui.default_test_mic, daemon=True).start()
        elif btn_id == "btn-unmute":
            if self.gui._unmute_callback:
                self.gui._unmute_callback()
            else:
                self.gui.default_toggle_mute()
        elif btn_id == "btn-tools":
            self.gui.show_tools()
        elif btn_id == "btn-status":
            self.gui.show_status()
        elif btn_id == "btn-help":
            self.gui.show_help()
        elif btn_id == "btn-clear":
            try:
                log = self.query_one(RichLog)
                log.clear()
            except Exception:
                pass
            self.gui.show_banner()
        elif btn_id == "btn-abort":
            if self.gui._abort_callback:
                self.gui._abort_callback()
            else:
                self.gui.log_system("ABORT requested — terminating active actions.")
                self.gui.set_status("IDLE")
        elif btn_id == "btn-exit":
            self.gui.destroy()
            self.exit()

    def on_input_submitted(self, event: Input.Submitted):
        """Handle Enter key pressed in command input box."""
        user_text = event.value.strip()
        event.input.value = ""
        if not user_text:
            return

        cmd_lower = user_text.lower()
        if cmd_lower in ("exit", "quit", "/exit", "/quit"):
            self.gui.destroy()
            self.exit()
            return
        if cmd_lower in ("/clear", "clear", "cls"):
            try:
                log = self.query_one(RichLog)
                log.clear()
            except Exception:
                pass
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
        if cmd_lower in ("/unmute", "unmute", "/mute", "mute"):
            if self.gui._unmute_callback:
                self.gui._unmute_callback()
            else:
                self.gui.default_toggle_mute()
            return
        if cmd_lower in ("/testmic", "testmic"):
            if self.gui._testmic_callback:
                threading.Thread(target=self.gui._testmic_callback, daemon=True).start()
            else:
                threading.Thread(target=self.gui.default_test_mic, daemon=True).start()
            return
        if cmd_lower in ("/abort", "abort"):
            if self.gui._abort_callback:
                self.gui._abort_callback()
            else:
                self.gui.log_system("ABORT requested.")
                self.gui.set_status("IDLE")
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

    def set_status_text(self, status: str):
        try:
            color = "#00FFFF" if status in ("LISTENING", "IDLE") else ("#FFD700" if status in ("SPEAKING", "THINKING") else "#FFBF00")
            self.query_one("#status-line", Static).update(f"[bold {color}]● STATUS: {status}[/]")
        except Exception:
            pass

    def update_mic_button(self, is_active: bool):
        try:
            btn = self.query_one("#btn-unmute", Button)
            btn.label = "🔇 Mute Mic" if is_active else "🎙️ Unmute Mic"
        except Exception:
            pass

    def update_tools_button(self, count: int):
        try:
            btn = self.query_one("#btn-tools", Button)
            btn.label = f"⚙️ Tools ({count})"
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
        self._tools_list: list[str] = []
        self._input_callback = None
        self._abort_callback = None
        self._testmic_callback = None
        self._unmute_callback = None
        self._is_running = True
        self._mic_active = True
        self._mic_info = "Default Mic (100% Vol, UNMUTED)"
        self._in_bruce_stream = False
        self._app: BatcomputerTextualApp | None = None

    def _on_app_thread(self) -> bool:
        if self._app is None:
            return False
        app_thread_id = getattr(self._app, "_thread_id", None)
        return app_thread_id is not None and threading.get_ident() == app_thread_id

    def _render_or_print(self, content):
        """Render to Textual log if active, otherwise print to console."""
        if self._app is not None:
            try:
                if self._on_app_thread():
                    self._app.write_log(content)
                else:
                    self._app.call_from_thread(self._app.write_log, content)
                return
            except Exception:
                pass
        with self._print_lock:
            try:
                self.console.print(content)
            except Exception:
                self.console.print(Text(str(content)))

    def show_banner(self):
        """Display the Hermes-style dual-column Batcomputer welcome banner."""
        n_tools = len(self._tools_list)
        left_lines = [
            BAT_HERO_EMBLEM,
            "",
            f"[bold {COLOR_ACCENT}]Operative:[/]   [{COLOR_TEXT}]Bruce Wayne (The Dark Knight)[/]",
            f"[bold {COLOR_ACCENT}]Cognitive:[/]   [{COLOR_TEXT}]{escape(LLM_MODEL)}[/] [dim {COLOR_DIM}](Ollama, local)[/]",
            f"[bold {COLOR_ACCENT}]Voice Synth:[/] [{COLOR_TEXT}]{escape(TTS_ENGINE.title())}[/] [dim {COLOR_DIM}]({escape(_voice_label())})[/]",
            f"[bold {COLOR_ACCENT}]Location:[/]    [{COLOR_TEXT}]Gotham City (The Batcave)[/]",
            f"[bold {COLOR_ACCENT}]Audio Input:[/] [bold green]Active[/] [dim {COLOR_DIM}]({escape(self._mic_info)})[/]",
        ]

        right_lines = [
            f"[bold {COLOR_ACCENT}]Active Toolsets[/]",
            f"  [{COLOR_TEXT}]Browser Tools:[/] [dim {COLOR_DIM}]navigate, click, type, snapshot, tabs...[/]",
            f"  [{COLOR_TEXT}]Intelligence:[/]  [dim {COLOR_DIM}]tavily_web_search (live web intelligence)[/]",
            "",
            f"[bold {COLOR_ACCENT}]Connected MCP Servers[/]",
            f"  [bold green]●[/] [{COLOR_TEXT}]playwright[/] [dim {COLOR_DIM}](browser automation tools)[/]",
            f"  [bold green]●[/] [{COLOR_TEXT}]tavily[/]     [dim {COLOR_DIM}](real-time web search)[/]",
            "",
            f"[bold {COLOR_ACCENT}]Gotham Defense Protocols[/]",
            f"  [{COLOR_TEXT}]Acoustic Sensor:[/] [bold green]LISTENING[/] [dim {COLOR_DIM}](Faster-Whisper + Silero)[/]",
            f"  [{COLOR_TEXT}]Voice abort:[/]     [bold green]ARMED[/] [dim {COLOR_DIM}](say 'stop' / 'cancel', or /abort)[/]",
            "",
            f"[dim {COLOR_DIM}]{n_tools} tools · 5 subsystems · Click buttons below or type /help[/]",
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
        output_lines.append(_row("Cognitive Core", True, f"[{COLOR_TEXT}]{escape(LLM_MODEL)} via Ollama[/]"))
        output_lines.append(_row("Voice Synthesis", True, f"[{COLOR_TEXT}]{escape(TTS_ENGINE.title())} TTS ({escape(_voice_label())})[/]"))
        output_lines.append(_row("Acoustic Sensor", True, f"[{COLOR_TEXT}]Faster-Whisper {escape(STT_MODEL_SIZE)} + Silero VAD (threshold {VAD_THRESHOLD:.2f})[/]"))
        output_lines.append(_row("Tool Protocol", bool(self._tools_list), f"[{COLOR_TEXT}]{len(self._tools_list)} tools loaded[/]"))
        output_lines.append(_row("Windows Audio", self._mic_active, f"[{COLOR_TEXT}]{escape(self._mic_info)}[/]"))
        output_lines.append("")
        output_lines.append(f"  [dim {COLOR_DIM}]All subsystems operating within nominal parameters.[/]")

        self._render_or_print("\n".join(output_lines))

    def show_tools(self):
        """Categorized tools listing (Hermes Agent style)."""
        output_lines = [
            f"[bold {COLOR_CYAN}]◆ ACTIVE AGENTS & MCP TACTICAL TOOLS ({len(self._tools_list)} TOOLS LOADED)[/]",
            "",
            f"  [bold {COLOR_ACCENT}]🤖 Specialized Worker Agents (Supervisor Map)[/]",
            f"    [dim {COLOR_DIM}]•[/] [{COLOR_TEXT}]web_agent[/]          [dim {COLOR_DIM}]— Playwright browser automation (navigation, clicks, extraction)[/]",
            f"    [dim {COLOR_DIM}]•[/] [{COLOR_TEXT}]code_agent[/]         [dim {COLOR_DIM}]— Sandboxed CodeAct Python execution engine (NVIDIA LLM)[/]",
            f"    [dim {COLOR_DIM}]•[/] [{COLOR_TEXT}]web_search_agent[/]   [dim {COLOR_DIM}]— Real-time internet intelligence & factual synthesis (Tavily)[/]",
            f"    [dim {COLOR_DIM}]•[/] [{COLOR_TEXT}]rag_agent[/]          [dim {COLOR_DIM}]— Local knowledge retrieval & documentation memory[/]",
            f"    [dim {COLOR_DIM}]•[/] [{COLOR_TEXT}]data_analyst_agent[/]  [dim {COLOR_DIM}]— Numerical, statistical, and tabular analysis in sandbox[/]",
            f"    [dim {COLOR_DIM}]•[/] [{COLOR_TEXT}]general_agent[/]      [dim {COLOR_DIM}]— Fast local streaming conversational cognitive core (Ollama)[/]",
            "",
        ]

        if self._tools_list:
            output_lines.append(f"  [bold {COLOR_ACCENT}]🧭 Browser & MCP Tools Loaded ({len(self._tools_list)})[/]")
            for t in self._tools_list:
                output_lines.append(f"    [dim {COLOR_DIM}]•[/] [{COLOR_TEXT}]{escape(t)}[/]")
            output_lines.append("")

        self._render_or_print("\n".join(output_lines))

    def show_help(self):
        """Display tactical command help (Hermes Agent style)."""
        output_lines = [
            f"[bold {COLOR_CYAN}]◆ BATCOMPUTER COMMAND INTERFACE & SHORTCUTS[/]",
            "",
        ]

        commands = [
            ("Click Test Mic", "Record a 1.5s audio sample to test live microphone & VAD"),
            ("Click Unmute", "Force un-mute Windows microphone and maximize recording volume"),
            ("Click Tools", "List all active MCP, Playwright, and Web tools by category"),
            ("Click Subsystems", "Display comprehensive subsystem telemetry & sensor health"),
            ("Click Clear", "Clear terminal log and redisplay the Batcomputer welcome banner"),
            ("Click Abort", "Instantly abort current speech or tactical tool execution (or say 'stop')"),
            ("Click Exit", "Power down Bruce and the Batcomputer defense network"),
            ("/help, /status, etc.", "All commands can also be typed into the input box"),
        ]

        for cmd, desc in commands:
            output_lines.append(f"  [bold {COLOR_ACCENT}]{escape(cmd):<22}[/] [{COLOR_TEXT}]{escape(desc)}[/]")

        output_lines.append("")
        output_lines.append(f"  [dim {COLOR_DIM}]Tip: Speak directly into the microphone at any time — the acoustic sensor is always listening.[/]")

        self._render_or_print("\n".join(output_lines))

    def log_system(self, text: str):
        """Log a system message with Batcomputer styling."""
        now = datetime.now().strftime("%H:%M:%S")
        self._render_or_print(f"[dim {COLOR_CYAN}]{now}[/dim {COLOR_CYAN}] [bold {COLOR_ACCENT}]📡 \\[SYS][/] [{COLOR_TEXT}]{escape(str(text))}[/]")

    def log_user(self, text: str, is_voice: bool = False):
        """Display citizen input."""
        now = datetime.now().strftime("%H:%M:%S")
        icon = "🎤" if is_voice else "⌨️ "
        src = "Voice" if is_voice else "Terminal"
        self._render_or_print(f"\n[bold {COLOR_CYAN}]◆ {icon} CITIZEN ({src}) \\[{now}][/]\n  [{COLOR_TEXT}]{escape(str(text))}[/]")

    def stream_bruce_start(self):
        """Initiate Bruce's tactical response stream."""
        now = datetime.now().strftime("%H:%M:%S")
        self._render_or_print(f"\n[bold {COLOR_GOLD}]◆ 🦇 BRUCE // GOTHAM GUARDIAN \\[{now}][/]")
        self._in_bruce_stream = True

    def stream_bruce_sentence(self, sentence: str):
        """Stream a sentence in real time with typewriter feel."""
        if not self._in_bruce_stream:
            self.stream_bruce_start()
        self._render_or_print(f"  [bold {COLOR_ACCENT}]›[/] [{COLOR_TEXT}]{escape(str(sentence))}[/]")

    def stream_bruce_end(self):
        """Close the streaming response block."""
        if self._in_bruce_stream:
            self._render_or_print("")
            self._in_bruce_stream = False

    def log_tool(self, text: str):
        """Log tool invocation in real time."""
        self._render_or_print(f"  [bold magenta]⚙️  {escape(str(text))}[/]")

    def log_error(self, text: str):
        """Log alert or error."""
        self._render_or_print(f"[bold red]❌ \\[ALERT][/bold red] [red]{escape(str(text))}[/red]")

    def set_status(self, status: str):
        """Update system status (shown in the status line)."""
        self._status = status.upper()
        if self._app is not None:
            try:
                if self._on_app_thread():
                    self._app.set_status_text(self._status)
                else:
                    self._app.call_from_thread(self._app.set_status_text, self._status)
            except Exception:
                pass

    def set_tools(self, tools_list: list):
        """Update loaded tool names."""
        self._tools_list = list(tools_list)
        if self._app is not None:
            try:
                if self._on_app_thread():
                    self._app.update_tools_button(len(self._tools_list))
                else:
                    self._app.call_from_thread(self._app.update_tools_button, len(self._tools_list))
            except Exception:
                pass

    def set_mic_active(self, active: bool, info: str = ""):
        """Update mic status."""
        self._mic_active = active
        if info:
            self._mic_info = info
        if self._app is not None:
            try:
                if self._on_app_thread():
                    self._app.update_mic_button(active)
                else:
                    self._app.call_from_thread(self._app.update_mic_button, active)
            except Exception:
                pass

    def default_test_mic(self):
        """Fallback mic hardware diagnostic."""
        self.log_system("MIC DIAGNOSTIC — Recording 1.5 seconds from default microphone...")
        try:
            import numpy as np
            import sounddevice as sd
            rec = sd.rec(int(AUDIO_SAMPLE_RATE * 1.5), samplerate=AUDIO_SAMPLE_RATE, channels=1, dtype="float32")
            sd.wait()
            max_val = float(np.max(np.abs(rec)))
            rms = float(np.sqrt(np.mean(rec ** 2)))
            if max_val < 0.0001:
                self.log_error("MIC DIAGNOSTIC: Near-zero signal. Check microphone mute button (Fn+F4) or Windows Audio Settings.")
            else:
                self.log_system(f"MIC DIAGNOSTIC: Signal OK | Peak={max_val:.3f} | RMS={rms:.3f}")
        except Exception as e:
            self.log_error(f"Mic diagnostic failed: {e}")

    def default_toggle_mute(self):
        """Fallback unmute/mute status toggle."""
        self._mic_active = not self._mic_active
        if not self._mic_active:
            self.set_mic_active(False, "Default Mic (MUTED by operator)")
            self.log_system("Microphone sensor MUTED. Type queries into prompt below.")
            self.set_status("MIC MUTED")
        else:
            try:
                from voice_agent.bruce import check_and_unmute_windows_mic
                mic_ok, mic_desc = check_and_unmute_windows_mic()
            except Exception:
                mic_ok, mic_desc = True, "Default Mic (100% Vol, UNMUTED)"
            self.set_mic_active(mic_ok, mic_desc)
            self.log_system(f"Microphone sensor UNMUTED & ACTIVE: {mic_desc}")
            self.set_status("LISTENING")

    def update_waveform(self, audio_level: float):
        """Waveform level hook."""

    def set_input_callback(self, cb):
        self._input_callback = cb

    def set_abort_callback(self, cb):
        self._abort_callback = cb

    def set_testmic_callback(self, cb):
        self._testmic_callback = cb

    def set_unmute_callback(self, cb):
        self._unmute_callback = cb

    def setup_logging_bridge(self):
        """Attach custom logging handler to stream python logging into the TUI."""
        try:
            handler = BatcomputerLogHandler(self)
            formatter = logging.Formatter("%(message)s")
            handler.setFormatter(formatter)
            logging.getLogger().addHandler(handler)
        except Exception:
            pass

    def run(self):
        """Start the interactive terminal session with clickable buttons."""
        self.setup_logging_bridge()
        self._app = BatcomputerTextualApp(self)
        try:
            self._app.run()
        except Exception:
            traceback.print_exc()
        finally:
            if self._is_running:
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
    from voice_agent.bruce import BruceAgent
    gui = BatcomputerGUI()
    agent = None
    try:
        agent = BruceAgent(gui)
        agent_thread = threading.Thread(target=agent.run, daemon=True)
        agent_thread.start()
        gui.run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        traceback.print_exc()
        gui.log_error(f"Fatal startup error: {e}")
    finally:
        if agent is not None:
            agent.shutdown()
