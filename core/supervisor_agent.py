import asyncio
import os
from pathlib import Path
from typing import Callable, Dict

from langchain_core.messages import HumanMessage, SystemMessage

from core.agent_config import get_llm_for_agent
from core.model_router import AGENTS, LLMRouter
from web_agent.web_agent import PlaywrightMCP, run_web_agent


class SupervisorAgent:
    """
    Supervisor Agent coordinates user tasks, routing them to specialized
    worker agents via an explicit dispatch map.
    """

    def __init__(self):
        # 1. Get the local LLM designed specifically for the supervisor router
        self.llm = get_llm_for_agent("supervisor")
        
        # 2. Inject it into the router
        self.router = LLMRouter(llm_instance=self.llm, agents=AGENTS)
        
        # 3. Explicit dispatch map for all specialized worker agents
        self.dispatch_map: Dict[str, Callable] = {
            "code_agent": self._handle_code_agent,
            "web_agent": self._handle_web_agent,
            "web_search_agent": self._handle_web_search_agent,
            "rag_agent": self._handle_rag_agent,
            "data_analyst_agent": self._handle_data_analyst_agent,
            "general_agent": self._handle_general_agent,
        }
        print("[Supervisor] Initialized with explicit dispatch map and local LLM router.")

    async def _handle_code_agent(self, task: str) -> str:
        """Executes coding tasks using the sandboxed CodeAct agent."""
        print("[code_agent] Launching Coding Agent (NVIDIA LLM + Sandbox)...")
        from coding_agent import get_default_sandbox, makeLLM, run_agent
        
        workspace_dir = Path(__file__).resolve().parent.parent / "coding_agent" / "workspace"
        sandbox = get_default_sandbox(workspace=str(workspace_dir), timeout=25)
        code_llm = makeLLM(provider="nvidia")
        
        answer = await asyncio.to_thread(
            run_agent,
            task=task,
            llm=code_llm,
            sandbox=sandbox,
            max_steps=8,
            verbose=True,
        )
        print(f"\n[code_agent] --- Task Finished ---\nAnswer: {answer}")
        return answer

    async def _handle_web_agent(self, task: str) -> str:
        """Executes browser automation tasks via Playwright MCP."""
        print("[web_agent] Launching Playwright Web Agent...")
        specialized_llm = get_llm_for_agent("web_agent")
        tools = PlaywrightMCP()
        await tools.start()
        try:
            result = await run_web_agent(
                task_id="web_task_1",
                task=task,
                llm=specialized_llm,
                tools=tools,
            )
            print(f"\n[web_agent] --- Task Finished ---")
            print(f"[web_agent] Status: {result.status}")
            print(f"[web_agent] Summary: {result.summary}")
            print(f"[web_agent] Steps used: {result.steps_used}")
            if result.error:
                print(f"[web_agent] Error: {result.error}")
                return f"Web Agent Error: {result.error}"
            return f"Web Agent Summary:\n{result.summary}"
        finally:
            keep_open = os.environ.get("KEEP_BROWSER_OPEN", "1") == "1" or "play" in task.lower()
            if not keep_open:
                await tools.close()
                print("[web_agent] Browser closed.")
            else:
                print("[web_agent] Browser left open on screen for viewing.")

    async def _handle_web_search_agent(self, task: str) -> str:
        """Searches the web via Tavily and synthesizes a grounded answer."""
        print("[web_search_agent] Performing live web search...")
        from core.tools import ToolManager
        tm = ToolManager()
        search_res = await tm._tavily_search({"query": task})

        specialized_llm = get_llm_for_agent("web_search_agent")
        prompt = [
            SystemMessage(content=(
                "You are a web search assistant. Answer the user's question clearly and accurately "
                "using the search results below. Cite relevant facts."
            )),
            HumanMessage(content=f"User Query: {task}\n\nSearch Results:\n{search_res}"),
        ]
        res = await asyncio.to_thread(specialized_llm.invoke, prompt)
        answer = getattr(res, "content", str(res))
        print(f"\n[web_search_agent] --- Task Finished ---")
        return answer

    async def _handle_rag_agent(self, task: str) -> str:
        """Retrieves internal knowledge base context and generates an answer."""
        print("[rag_agent] Consulting internal knowledge base...")
        base_dir = Path(__file__).resolve().parent.parent
        context_docs = []

        # Check long term memory or doc assets
        mem_file = base_dir / "long_term_memory.txt"
        if mem_file.exists():
            context_docs.append(f"Long-term memory:\n{mem_file.read_text(encoding='utf-8')[:2000]}")
        readme_file = base_dir / "README.md"
        if readme_file.exists():
            context_docs.append(f"Project documentation:\n{readme_file.read_text(encoding='utf-8')[:3000]}")

        context_str = "\n\n".join(context_docs) or "No internal documents currently indexed."
        specialized_llm = get_llm_for_agent("rag_agent")
        prompt = [
            SystemMessage(content="You are a RAG assistant. Answer based on the provided internal context."),
            HumanMessage(content=f"Context:\n{context_str}\n\nQuestion: {task}"),
        ]
        res = await asyncio.to_thread(specialized_llm.invoke, prompt)
        answer = getattr(res, "content", str(res))
        print(f"\n[rag_agent] --- Task Finished ---")
        return answer

    async def _handle_data_analyst_agent(self, task: str) -> str:
        """Analyzes data and executes numerical/tabular code in the sandbox."""
        print("[data_analyst_agent] Launching Data Analyst Agent...")
        from coding_agent import get_default_sandbox, makeLLM, run_agent
        
        workspace_dir = Path(__file__).resolve().parent.parent / "coding_agent" / "workspace"
        sandbox = get_default_sandbox(workspace=str(workspace_dir), timeout=30)
        code_llm = makeLLM(provider="nvidia")
        
        analyst_task = (
            f"You are a data analyst. Compute the requested analysis, statistics, or metrics:\n{task}"
        )
        answer = await asyncio.to_thread(
            run_agent,
            task=analyst_task,
            llm=code_llm,
            sandbox=sandbox,
            max_steps=8,
            verbose=True,
        )
        print(f"\n[data_analyst_agent] --- Task Finished ---\nAnswer: {answer}")
        return answer

    async def _handle_general_agent(self, task: str) -> str:
        """Handles general inquiries, conversation, and fallback Q&A."""
        print("[general_agent] Processing with General Agent...")
        specialized_llm = get_llm_for_agent("general_agent")
        prompt = [
            SystemMessage(content="You are Naiyra, a helpful and capable cognitive partner."),
            HumanMessage(content=task),
        ]
        res = await asyncio.to_thread(specialized_llm.invoke, prompt)
        answer = getattr(res, "content", str(res))
        print(f"\n[general_agent] --- Task Finished ---")
        return answer

    async def run_task(self, task_description: str) -> str:
        print(f"\n{'='*50}\n[Supervisor] Received Task: '{task_description}'\n{'='*50}")
        
        # Route the task to find out which specialized agent should handle it
        messages = [HumanMessage(content=task_description)]
        decision = self.router.route(messages)
        
        print("[Supervisor] Routing Decision:")
        print(f"  - Next Agent: {decision.next_agent}")
        print(f"  - Confidence: {decision.confidence}")
        print(f"  - Reasoning: {decision.reasoning}")
        print(f"  - Task for Agent: {decision.task_for_agent}")
        
        if decision.next_agent == "FINISH":
            print("[Supervisor] Task is fully complete.")
            return "Task is fully complete."

        # Dispatch via the explicit handler map
        handler = self.dispatch_map.get(decision.next_agent, self._handle_general_agent)
        print(f"\n[Supervisor] Dispatching task via map -> {decision.next_agent}...")
        
        target_task = decision.task_for_agent or task_description
        return await handler(target_task)


async def main():
    supervisor = SupervisorAgent()
    task = "open youtube and play eminem first song it see"
    await supervisor.run_task(task)

if __name__ == "__main__":
    asyncio.run(main())
