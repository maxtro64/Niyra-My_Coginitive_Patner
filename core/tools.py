import asyncio
import concurrent.futures
import os
import shutil
import subprocess
import threading
from pathlib import Path

import yaml
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None

from dotenv import load_dotenv

from core.config import (
    BASE_DIR,
    MCP_CONNECT_TIMEOUT_S,
    TOOL_CALL_TIMEOUT_S,
    TOOL_RESULT_MAX_CHARS,
)

load_dotenv(Path(BASE_DIR) / ".env")


def _describe_exc(e: BaseException) -> str:
    """Flatten ExceptionGroups (anyio task groups) so the REAL cause is printed, not 'unhandled errors in a TaskGroup'."""
    subs = getattr(e, "exceptions", None)
    if subs:
        return "; ".join(_describe_exc(x) for x in subs)
    return f"{type(e).__name__}: {e}"


def bring_browser_to_front():
    """Bring the active Playwright Chromium or Edge window to the foreground on Windows."""
    if os.name != "nt":
        return
    try:
        import ctypes
        from ctypes import wintypes
        import psutil

        user32 = ctypes.windll.user32
        target_hwnds = []

        EnumWindows = user32.EnumWindows
        EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
        IsWindowVisible = user32.IsWindowVisible
        GetWindowTextLength = user32.GetWindowTextLengthW
        GetWindowText = user32.GetWindowTextW
        GetWindowThreadProcessId = user32.GetWindowThreadProcessId

        def callback(hwnd, extra):
            if IsWindowVisible(hwnd):
                length = GetWindowTextLength(hwnd)
                if length > 0:
                    buff = ctypes.create_unicode_buffer(length + 1)
                    GetWindowText(hwnd, buff, length + 1)
                    title = buff.value
                    pid = wintypes.DWORD()
                    GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                    try:
                        p = psutil.Process(pid.value)
                        name = p.name().lower()
                        cmd = " ".join(p.cmdline()).lower()
                    except Exception:
                        name = ""
                        cmd = ""

                    if "chrome" in name or "msedge" in name:
                        if "playwright" in cmd or "remote-debugging" in cmd or title.endswith("- Google Chrome") or "google" in title.lower() or "youtube" in title.lower():
                            target_hwnds.append(hwnd)
            return True

        EnumWindows(EnumWindowsProc(callback), 0)

        for hwnd in target_hwnds:
            user32.ShowWindow(hwnd, 9)  # SW_RESTORE
            user32.keybd_event(0x12, 0, 0, 0)
            user32.SetForegroundWindow(hwnd)
            user32.keybd_event(0x12, 0, 2, 0)
    except Exception:
        pass


class ToolManager:
    """
    Native tools (Tavily) + MCP servers.

    Every MCP server lives inside ONE long-running task (`_run_server`). anyio cancel scopes
    used by stdio_client / ClientSession must be entered and exited in the same task, which the
    old AsyncExitStack + wait_for + separate close() coroutine did not guarantee.
    """

    def __init__(self, registry_path=None):
        self.registry_path = Path(registry_path) if registry_path else Path(BASE_DIR) / "mcp_registry.yaml"
        self.clients = {}             # server_name -> client session
        self.mcp_tools = {}           # tool_name -> server_name
        self.native_tools = {}        # tool_name -> async python function
        self.all_tool_schemas = []
        self._server_tool_names = {}  # server_name -> [tool names]
        self._server_tasks = {}       # server_name -> asyncio.Task
        self._stop_events = {}        # server_name -> asyncio.Event

        self._init_native_tools()

    # ─── Native tools ─────────────────────────────────────────────────────────

    def _init_native_tools(self):
        tavily_key = os.environ.get("TAVILY_API_KEY", "")
        self.tavily_client = TavilyClient(api_key=tavily_key) if (TavilyClient and tavily_key) else None

        self.native_tools["tavily_web_search"] = self._tavily_search
        self.all_tool_schemas.append({
            "type": "function",
            "function": {
                "name": "tavily_web_search",
                "description": "Search the web using Tavily for real-time accurate information. Use this to find current events, facts, or any internet data.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query."}
                    },
                    "required": ["query"]
                }
            }
        })
        
        self.native_tools["save_file"] = self._save_file
        self.all_tool_schemas.append({
            "type": "function",
            "function": {
                "name": "save_file",
                "description": "Save text content to a local file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "The absolute or relative file path to save to. For desktop, use something like 'C:/Users/username/Desktop/filename.md' or 'Desktop/filename.md'."},
                        "content": {"type": "string", "description": "The full text content to write to the file."}
                    },
                    "required": ["path", "content"]
                }
            }
        })
        
        self.native_tools["store_in_memory"] = self._store_in_memory
        self.all_tool_schemas.append({
            "type": "function",
            "function": {
                "name": "store_in_memory",
                "description": "Store important facts, user preferences, or conclusions in long-term memory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "fact": {"type": "string", "description": "The specific fact or insight to remember."}
                    },
                    "required": ["fact"]
                }
            }
        })

    async def _tavily_search(self, arguments):
        if TavilyClient is None:
            return "The tavily-python package is not installed. Web search is unavailable."
        if not self.tavily_client:
            return "TAVILY_API_KEY is not set in the environment variables. Web search is unavailable."

        query = ((arguments or {}).get("query") or "").strip()
        if not query:
            return "No query provided."
        try:
            # TavilyClient is blocking — run it in a worker thread so the event loop
            # (which also serves every MCP call) is never stalled.
            response = await asyncio.to_thread(self.tavily_client.search, query, max_results=3)
        except Exception as e:
            return f"Search failed: {e}"

        lines = []
        if isinstance(response, dict):
            if response.get("answer"):
                lines.append(f"Answer: {response['answer']}")
            for r in response.get("results", [])[:3]:
                lines.append(f"- {r.get('title', '')} ({r.get('url', '')}): {str(r.get('content', ''))[:400]}")
        text = "\n".join(lines) if lines else str(response)
        return text[:TOOL_RESULT_MAX_CHARS]
        
    async def _save_file(self, arguments):
        path = (arguments or {}).get("path")
        content = (arguments or {}).get("content")
        if not path or not content:
            return "Missing path or content."
        try:
            expanded_path = os.path.expanduser(path)
            # if relative Desktop path, prefix with actual Desktop path
            if expanded_path.startswith("Desktop/") or expanded_path.startswith("Desktop\\"):
                expanded_path = os.path.join(os.path.expanduser("~"), expanded_path)
            
            with open(expanded_path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Successfully saved to {expanded_path}"
        except Exception as e:
            return f"Failed to save file: {e}"
            
    async def _store_in_memory(self, arguments):
        fact = (arguments or {}).get("fact")
        if not fact:
            return "No fact provided."
        # Stub for long-term memory until Naiyra's memory module is built
        try:
            mem_path = Path(BASE_DIR) / "long_term_memory.txt"
            with open(mem_path, "a", encoding="utf-8") as f:
                f.write(f"- {fact}\n")
            return f"Stored in memory: {fact}"
        except Exception as e:
            return f"Failed to store in memory: {e}"

    # ─── MCP servers ──────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_command(command):
        """'npx' must resolve to npx.cmd on Windows — shutil.which honours PATHEXT."""
        return shutil.which(command) or command

    async def connect(self):
        try:
            registry = yaml.safe_load(self.registry_path.read_text(encoding="utf-8")) or {}
        except FileNotFoundError:
            print(f"No MCP registry found at {self.registry_path}.")
            return

        servers = registry.get("servers") or {}
        loop = asyncio.get_running_loop()

        for name, config in servers.items():
            command = self._resolve_command(config.get("command"))
            args = config.get("args", [])

            env = os.environ.copy()
            env["PLAYWRIGHT_MCP_HEADLESS"] = os.environ.get("PLAYWRIGHT_MCP_HEADLESS", "false")
            env["HEADLESS"] = os.environ.get("HEADLESS", "false")

            params = StdioServerParameters(command=command, args=args, env=env)
            ready = loop.create_future()
            stop = asyncio.Event()
            task = asyncio.create_task(self._run_server(name, params, ready, stop))
            self._server_tasks[name] = task
            self._stop_events[name] = stop

            try:
                count = await asyncio.wait_for(ready, timeout=MCP_CONNECT_TIMEOUT_S + 10)
                print(f"Connected to MCP: {name} (Loaded {count} tools)")
            except Exception as e:
                print(f"Failed to connect to MCP {name}: {_describe_exc(e)}")
                stop.set()
                task.cancel()

    async def _run_server(self, name, params, ready, stop):
        try:
            # Use subprocess.DEVNULL for errlog so captured or invalid sys.stderr handles in GUI/threads
            # do not trigger OSError [Errno 9] Bad file descriptor on Windows.
            async with stdio_client(params, errlog=subprocess.DEVNULL) as (read, write):
                async with ClientSession(read, write) as client:
                    async with asyncio.timeout(MCP_CONNECT_TIMEOUT_S):
                        await client.initialize()
                        result = await client.list_tools()

                    self.clients[name] = client
                    names = []
                    for tool in result.tools:
                        self.mcp_tools[tool.name] = name
                        names.append(tool.name)
                        self.all_tool_schemas.append({
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "description": tool.description or "",
                                "parameters": (getattr(tool, "inputSchema", None)
                                           or getattr(tool, "input_schema", None)
                                           or {"type": "object", "properties": {}})
                            }
                        })
                    self._server_tool_names[name] = names

                    if not ready.done():
                        ready.set_result(len(result.tools))
                    await stop.wait()          # stay alive (same task) until close()
        except Exception as e:
            if not ready.done():
                ready.set_exception(e)
            else:
                print(f"MCP server '{name}' stopped: {_describe_exc(e)}")
        finally:
            self._forget_server(name)

    def _forget_server(self, name):
        self.clients.pop(name, None)
        gone = set(self._server_tool_names.pop(name, []))
        for tool_name in gone:
            self.mcp_tools.pop(tool_name, None)
        if gone:
            self.all_tool_schemas = [s for s in self.all_tool_schemas
                                     if s["function"]["name"] not in gone]

    async def get_tools(self):
        return self.all_tool_schemas

    def convert_tools_for_llm(self, allowed=None):
        """All tool schemas, or only those whose name is in `allowed`."""
        if allowed is None:
            return list(self.all_tool_schemas)
        allowed = set(allowed)
        return [s for s in self.all_tool_schemas if s["function"]["name"] in allowed]

    def search_tools(self, query: str, top_k: int = 3, min_score: float = 0.5):
        """Use LayaSearch to find the most relevant tools for a given query."""
        try:
            from core.laya_tools import LayaSearch
        except ImportError:
            return [s["function"]["name"] for s in self.all_tool_schemas][:top_k]

        laya_search = LayaSearch()
        
        # Register all available tools dynamically
        for schema in self.all_tool_schemas:
            func = schema.get("function", {})
            name = func.get("name")
            desc = func.get("description")
            if name and desc:
                laya_search.register(name, desc)
                
        # Perform the search
        results = laya_search.search(query, top_k=top_k, min_score=min_score)
        return [name for name, score in results]

    async def call_tool(self, tool_name, arguments):
        arguments = arguments or {}
        if tool_name in self.native_tools:
            return await self.native_tools[tool_name](arguments)

        server_name = self.mcp_tools.get(tool_name)
        if server_name is None or server_name not in self.clients:
            raise ValueError(f"Tool {tool_name} not found.")

        async with asyncio.timeout(TOOL_CALL_TIMEOUT_S):
            res = await self.clients[server_name].call_tool(tool_name, arguments)
            if tool_name.startswith("browser_"):
                asyncio.create_task(asyncio.to_thread(bring_browser_to_front))
            return res

    async def close(self):
        for stop in self._stop_events.values():
            stop.set()
        tasks = [t for t in self._server_tasks.values() if not t.done()]
        if tasks:
            _, pending = await asyncio.wait(tasks, timeout=10)
            for t in pending:
                t.cancel()
        self._server_tasks.clear()
        self._stop_events.clear()


class SyncPlaywrightMcp:
    """Synchronous wrapper for ToolManager using a background event loop."""
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(SyncPlaywrightMcp, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, registry_path=None):
        if self._initialized: return
        self.mcp = ToolManager(registry_path)
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
        self._initialized = True
        self._connected = False

    def connect(self, timeout=None):
        if self._connected: return
        self._connected = True
        # Must be longer than the per-server timeout (a first `npx` run downloads the package)
        timeout = timeout or (MCP_CONNECT_TIMEOUT_S * 2 + 30)
        future = asyncio.run_coroutine_threadsafe(self.mcp.connect(), self.loop)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise TimeoutError(f"MCP connect did not finish within {timeout}s")

    def get_tools(self):
        future = asyncio.run_coroutine_threadsafe(self.mcp.get_tools(), self.loop)
        return future.result(timeout=10)

    def convert_tools_for_llm(self, allowed=None):
        return self.mcp.convert_tools_for_llm(allowed=allowed)

    def search_tools(self, query: str, top_k: int = 3, min_score: float = 0.5):
        """Synchronous wrapper for LayaSearch tool matching."""
        # search_tools is currently a synchronous function in ToolManager, 
        # but if it relies on async resources we'd use run_coroutine_threadsafe.
        # Since we implemented it synchronously in ToolManager, we can call it directly:
        return self.mcp.search_tools(query, top_k, min_score)

    def call_tool(self, tool_name, arguments):
        future = asyncio.run_coroutine_threadsafe(self.mcp.call_tool(tool_name, arguments), self.loop)
        try:
            # Backstop for the in-loop timeout: a stuck page must never hang the agent
            return future.result(timeout=TOOL_CALL_TIMEOUT_S + 15)
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise TimeoutError(f"Tool '{tool_name}' timed out after {TOOL_CALL_TIMEOUT_S}s")

    def close(self):
        future = asyncio.run_coroutine_threadsafe(self.mcp.close(), self.loop)
        try:
            future.result(timeout=20)
        except Exception:
            pass
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(timeout=1.0)
            self._connected = False
            self._initialized = False
            SyncPlaywrightMcp._instance = None
