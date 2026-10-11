import re

SYSTEM = """You are a coding agent that solves tasks by writing and running Python.
 
Environment:
- Python 3.12 inside a sandbox. No internet, no pip install. Standard library only
  unless the user says otherwise.
- The current directory is /work. Files you write there persist between steps.
- Each run is a fresh process: variables do NOT persist, files do.
 
Protocol:
- To act, reply with exactly ONE ```python code block (a short sentence before it is fine).
- Always print() the values you need to see. You will receive stdout, stderr and the exit code.
- If you get an error, read it, fix the code and try again.
- When the task is done, reply with a line starting with FINAL: followed by the answer,
  and NO code block."""

CODE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)

def extract_code(text):
    m = CODE_RE.search(text)
    return m.group(1).strip() if m else None

def observation(res):
    parts = [f"exit code: {res.returncode}"]
    if res.timed_out:
        parts.append("The run was killed for exceeding the time limit.")
    parts.append(f"stdout:\n{res.stdout or '(empty)'}")
    if res.stderr:
        parts.append(f"stderr:\n{res.stderr}")
    return "\n".join(parts)

def run_agent(task, llm, sandbox, max_steps=8, verbose=True):
    messages = [{"role": "user", "content": task}]
    nudged = False
    for step in range(1, max_steps + 1):
        reply = llm(SYSTEM, messages)
        messages.append({"role": "assistant", "content": reply})
        code = extract_code(reply)
 
        if code is None:
            if "FINAL:" in reply:
                return reply.split("FINAL:", 1)[1].strip()
            if nudged:
                return reply  # model keeps rambling; return what it said
            nudged = True
            messages.append({"role": "user", "content":
                             "Reply with ONE ```python block, or a line starting with FINAL: if you are done."})
            continue
 
        if verbose:
            print(f"\n--- step {step}: running code ---\n{code}\n")
        res = sandbox.run(code)
        obs = observation(res)
        if verbose:
            print(f"--- result ---\n{obs}\n")
        messages.append({"role": "user", "content": obs})
 
    return f"[stopped: reached {max_steps} steps without FINAL]"
