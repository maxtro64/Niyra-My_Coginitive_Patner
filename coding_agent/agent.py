"""CodeAct agent : the LLM writes Python, we run it in a sandbox, it reads the output
   and keeps going until it answers with Final"""

import argparse
import os
import sys
from pathlib import Path

# Ensure root workspace and package are accessible
_current_dir = Path(__file__).resolve().parent
_root_dir = _current_dir.parent
if str(_root_dir) not in sys.path:
    sys.path.insert(0, str(_root_dir))
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

try:
    from coding_agent.core import run_agent
    from coding_agent.llm import PROVIDERS, makeLLM, check_ollama_status
    from coding_agent.sandbox import DockerSandbox, SubprocessSandbox, get_default_sandbox
except ImportError:
    from core import run_agent
    from llm import PROVIDERS, makeLLM, check_ollama_status
    from sandbox import DockerSandbox, SubprocessSandbox, get_default_sandbox


def main():
    ap = argparse.ArgumentParser(description="CodeAct agent with a sandboxed Python executor")
    ap.add_argument("task")
    ap.add_argument("--provider", default="nvidia", choices=list(PROVIDERS.keys()))
    ap.add_argument("--model")
    ap.add_argument("--base-url")
    ap.add_argument("--workspace", default=str(_current_dir / "workspace"))
    ap.add_argument("--image", default="python:3.12-slim",
                    help="Docker image; build your own with pandas/numpy baked in")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=20)
    ap.add_argument("--unsafe-subprocess", action="store_true",
                    help="force subprocess sandbox even if Docker is available")
    a = ap.parse_args()
 
    if a.unsafe_subprocess or not DockerSandbox.available():
        sandbox = SubprocessSandbox(a.workspace, timeout=a.timeout)
        if not DockerSandbox.available():
            print("[sandbox: subprocess - Docker unavailable, using host subprocess sandbox]")
        else:
            print("[sandbox: plain subprocess - forced by --unsafe-subprocess]")
    else:
        sandbox = DockerSandbox(a.workspace, image=a.image, timeout=a.timeout)
        print("[sandbox: docker]")
 
    answer = run_agent(a.task, makeLLM(a.provider, a.model, a.base_url), sandbox, a.steps)
    print(f"\n=== ANSWER ===\n{answer}")
 
if __name__ == "__main__":
    main()