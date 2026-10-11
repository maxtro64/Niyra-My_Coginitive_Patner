import os
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Result:
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool = False

def _clip(text, limit=4000):
    if len(text) <= limit:
        return text

    half = limit // 2
    return text[:half] + f"\n...[{len(text) - limit} chars cut]...\n" + text[-half:]

class DockerSandbox:

    def __init__(self, workspace, image="python:3.12-slim", timeout=20, memory="512", cpus="1"):
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.image, self.timeout, self.memory, self.cpus = image, timeout, memory, cpus
    
    @staticmethod
    def available():
        if not shutil.which("docker"):
            return False
        return subprocess.run(["docker", "info"], capture_output=True).returncode == 0
    
    def run(self, code):
        (self.workspace / "_run.py").write_text(code, encoding="utf-8")
        name = f"codeact-{uuid.uuid4().hex[:8]}"
        cmd = [
            "docker", "run", "--rm", "--name", name,
            "--network", "none",
            "--memory", self.memory, "--cpus", self.cpus, "--pids-limit", "128",
            "--read-only", "--tmpfs", "/tmp:rw,size=64m",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "PYTHONUNBUFFERED=1",
            "-v", f"{self.workspace}:/work", "-w", "/work",
        ]
        if hasattr(os, "getuid"):  # keep files in workspace owned by you, not root
            cmd += ["--user", f"{getattr(os, 'getuid')()}:{getattr(os, 'getgid')()}"]
        cmd += [self.image, "timeout", "-s", "KILL", str(self.timeout), "python", "_run.py"]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout + 15)
            timed_out = r.returncode in (124, 137)  # `timeout -s KILL` -> 137; OOM kill is also 137
            return Result(_clip(r.stdout), _clip(r.stderr), r.returncode, timed_out)
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "kill", name], capture_output=True)
            return Result("", "host-side timeout", -1, True)
 
class SubprocessSandbox:
    """DEV ONLY. Same kernel and filesystem as you, just resource-limited.
    Do not use with untrusted prompts."""
 
    def __init__(self, workspace, timeout=20, mem_mb=512):
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.timeout, self.mem_mb = timeout, mem_mb
 
    def _limits(self):
        try:
            import resource # type: ignore
            mb = self.mem_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (mb, mb)) # type: ignore
            resource.setrlimit(resource.RLIMIT_CPU, (self.timeout, self.timeout)) # type: ignore
            resource.setrlimit(resource.RLIMIT_FSIZE, (50 * 1024 * 1024,) * 2) # type: ignore
            resource.setrlimit(resource.RLIMIT_NPROC, (64, 64)) # type: ignore
        except ImportError:
            pass
        if hasattr(os, "setsid"):
            os.setsid()  # own process group so we can kill children (POSIX only)
 
    def run(self, code):
        (self.workspace / "_run.py").write_text(code, encoding="utf-8")
        from typing import Any
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONDONTWRITEBYTECODE": "1"}
        if os.name == "nt":
            for k in ["SystemRoot", "SystemDrive", "TEMP", "TMP"]:
                if k in os.environ:
                    env[k] = os.environ[k]
        kwargs: dict[str, Any] = dict(cwd=str(self.workspace), capture_output=True, text=True, timeout=self.timeout, env=env)
        if os.name == "posix":
            kwargs["preexec_fn"] = self._limits
        elif os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        try:
            r = subprocess.run([sys.executable, "-I", "_run.py"], **kwargs)
            return Result(_clip(r.stdout), _clip(r.stderr), r.returncode)
        except subprocess.TimeoutExpired:
            return Result("", f"timed out after {self.timeout}s", -1, True)


def get_default_sandbox(workspace="./workspace", timeout=20):
    """
    Returns DockerSandbox if Docker daemon is available and responsive;
    otherwise falls back cleanly to SubprocessSandbox.
    """
    if DockerSandbox.available():
        return DockerSandbox(workspace, timeout=timeout)
    return SubprocessSandbox(workspace, timeout=timeout)

