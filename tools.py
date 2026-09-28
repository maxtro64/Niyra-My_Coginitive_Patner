import asyncio
import threading
import yaml
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack
try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None
from dotenv import load_dotenv

load_dotenv()

class ToolManager:
    def __init__(self):
        self.clients = {}  # server_name -> client session
        self.exit_stacks = {} # server_name -> exit stack
        self.mcp_tools = {} # tool_name -> server_name
        self.native_tools = {} # tool_name -> python function
        self.all_tool_schemas = []
        
        # Initialize native tools
        self._init_native_tools()

    def _init_native_tools(self):
        # Tavily Search Tool
        tavily_key = os.environ.get("TAVILY_API_KEY", "")
        # We can add it regardless, or check if key exists. Let's add it regardless so the agent knows it has it, 
        # and if it fails, it returns a message saying API key is missing.
        self.tavily_client = TavilyClient(api_key=tavily_key) if tavily_key else None
        
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

    async def _tavily_search(self, arguments):
        if not self.tavily_client:
            return "TAVILY_API_KEY is not set in the environment variables. Web search is unavailable."
        
        query = arguments.get("query", "")
        if not query:
            return "No query provided."
        try:
            response = self.tavily_client.search(query)
            return str(response)
        except Exception as e:
            return f"Search failed: {e}"

    async def connect(self):
        try:
            with open("mcp_registry.yaml", "r") as f:
                registry = yaml.safe_load(f)
        except FileNotFoundError:
            print("No mcp_registry.yaml found.")
            return

        servers = registry.get("servers", {})
        for name, config in servers.items():
            command = config.get("command")
            args = config.get("args", [])
            
            env = os.environ.copy()
            env["PLAYWRIGHT_MCP_HEADLESS"] = "0"
            env["HEADLESS"] = "0"

            server_params = StdioServerParameters(command=command, args=args, env=env)
            stack = AsyncExitStack()
            try:
                read, write = await asyncio.wait_for(stack.enter_async_context(stdio_client(server_params)), timeout=25.0)
                client = await asyncio.wait_for(stack.enter_async_context(ClientSession(read, write)), timeout=25.0)
                await asyncio.wait_for(client.initialize(), timeout=25.0)
                
                self.clients[name] = client
                self.exit_stacks[name] = stack
                
                # Fetch tools
                result = await client.list_tools()
                for tool in result.tools:
                    self.mcp_tools[tool.name] = name
                    self.all_tool_schemas.append({
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description or "",
                            "parameters": tool.inputSchema
                        }
                    })
                print(f"Connected to MCP: {name} (Loaded {len(result.tools)} tools)")
            except Exception as e:
                print(f"Failed to connect to MCP {name}: {e}")

    async def get_tools(self):
        return self.all_tool_schemas

    def convert_tools_for_llm(self):
        return self.all_tool_schemas

    async def call_tool(self, tool_name, arguments):
        if tool_name in self.native_tools:
            return await self.native_tools[tool_name](arguments)
            
        if tool_name in self.mcp_tools:
            server_name = self.mcp_tools[tool_name]
            client = self.clients[server_name]
            result = await client.call_tool(tool_name, arguments)
            return result

        raise ValueError(f"Tool {tool_name} not found.")

    async def close(self):
        for stack in self.exit_stacks.values():
            try:
                await stack.aclose()
            except RuntimeError:
                pass


class SyncPlaywrightMcp:
    """Synchronous wrapper for ToolManager using a background event loop."""
    def __init__(self):
        self.mcp = ToolManager()
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()

    def connect(self, timeout=15):
        future = asyncio.run_coroutine_threadsafe(self.mcp.connect(), self.loop)
        return future.result(timeout=timeout)

    def get_tools(self):
        future = asyncio.run_coroutine_threadsafe(self.mcp.get_tools(), self.loop)
        return future.result()

    def convert_tools_for_llm(self):
        return self.mcp.convert_tools_for_llm()

    def call_tool(self, tool_name, arguments):
        future = asyncio.run_coroutine_threadsafe(self.mcp.call_tool(tool_name, arguments), self.loop)
        return future.result()

    def close(self):
        future = asyncio.run_coroutine_threadsafe(self.mcp.close(), self.loop)
        try:
            future.result()
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(timeout=1.0)
