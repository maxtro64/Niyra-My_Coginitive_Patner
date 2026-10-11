import asyncio
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Dict

# Add parent directory to path so we can from core import supervisor_agent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from core.supervisor_agent import SupervisorAgent
from message_gateway.adapters.base import BaseAdapter

JOBS_FILE = Path(__file__).parent / "cron_jobs.json"

class MessageGateway:
    def __init__(self):
        self.supervisor = SupervisorAgent()
        self.adapters: Dict[str, BaseAdapter] = {}
        self.scheduler = AsyncIOScheduler()
        self._load_jobs_on_start()

    def register_adapter(self, adapter: BaseAdapter):
        adapter.gateway = self
        self.adapters[adapter.name] = adapter
        print(f"[Gateway] Registered adapter: {adapter.name}")

    async def handle_message(self, platform: str, user_id: str, text: str) -> str:
        """Process incoming messages from any platform through the supervisor."""
        print(f"[{platform}] Message from {user_id}: {text}")
        
        # Intercept /cron commands so they work on ALL platforms
        if text.startswith("/cron "):
            return self._handle_cron_cmd(platform, user_id, text)
        if text.startswith("/crons"):
            return self._handle_crons_cmd(platform, user_id)
        if text.startswith("/uncron "):
            return self._handle_uncron_cmd(platform, user_id, text)

        # Standard agent routing
        result = await self.supervisor.run_task(text)
        return result or "Task completed."

    async def send_message(self, platform: str, user_id: str, text: str):
        if platform in self.adapters:
            await self.adapters[platform].send_message(user_id, text)
        else:
            print(f"[Gateway] Error: No adapter found for platform '{platform}'")

    async def start_all(self):
        """Start all adapters concurrently."""
        self.scheduler.start()
        print("[Gateway] Starting all adapters...")
        await asyncio.gather(*(adapter.start() for adapter in self.adapters.values()))

    # --- Unified Cron Logic ---
    def _load_jobs(self) -> dict:
        return json.loads(JOBS_FILE.read_text()) if JOBS_FILE.exists() else {}

    def _save_jobs(self, jobs: dict):
        JOBS_FILE.write_text(json.dumps(jobs, indent=2))

    def _load_jobs_on_start(self):
        jobs = self._load_jobs()
        for job_id, job in jobs.items():
            self._schedule_job_internal(job_id, job)

    async def _fire_job(self, job_id: str):
        job = self._load_jobs().get(job_id)
        if not job:
            return
        print(f"[Gateway] Firing cron job {job_id} for {job['platform']}")
        result = await self.supervisor.run_task(job["prompt"])
        await self.send_message(job["platform"], job["user_id"], result or "Cron completed.")

    def _schedule_job_internal(self, job_id: str, job: dict):
        self.scheduler.add_job(
            self._fire_job,
            CronTrigger.from_crontab(job["cron"]),
            args=[job_id],
            id=job_id,
            replace_existing=True,
        )

    def _handle_cron_cmd(self, platform: str, user_id: str, text: str) -> str:
        parts = text.split(" ")
        if len(parts) < 7:
            return "Usage: /cron <m> <h> <dom> <mon> <dow> <prompt>"
        expr, prompt = " ".join(parts[1:6]), " ".join(parts[6:])
        try:
            CronTrigger.from_crontab(expr)
        except ValueError:
            return f"Bad cron expression: {expr}"
        job_id = uuid.uuid4().hex[:8]
        job = {"cron": expr, "prompt": prompt, "platform": platform, "user_id": user_id}
        jobs = self._load_jobs()
        jobs[job_id] = job
        self._save_jobs(jobs)
        self._schedule_job_internal(job_id, job)
        return f"Scheduled {job_id}: {expr}"

    def _handle_crons_cmd(self, platform: str, user_id: str) -> str:
        jobs = self._load_jobs()
        user_jobs = {k: v for k, v in jobs.items() if v["platform"] == platform and v["user_id"] == user_id}
        text = "\n".join(f"{i}: {j['cron']} -> {j['prompt']}" for i, j in user_jobs.items())
        return text or "No jobs."

    def _handle_uncron_cmd(self, platform: str, user_id: str, text: str) -> str:
        parts = text.split(" ")
        if len(parts) < 2:
            return "Usage: /uncron <job_id>"
        job_id = parts[1]
        jobs = self._load_jobs()
        job = jobs.get(job_id)
        if job and job["platform"] == platform and job["user_id"] == user_id:
            del jobs[job_id]
            self._save_jobs(jobs)
            if self.scheduler.get_job(job_id):
                self.scheduler.remove_job(job_id)
            return "Removed."
        return "No such job."
