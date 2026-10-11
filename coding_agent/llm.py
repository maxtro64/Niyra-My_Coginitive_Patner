import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

try:
    from openai import OpenAI
except ImportError:
    # Fallback if openai is not installed, though it's used in makeLLM
    pass

from dotenv import load_dotenv
load_dotenv()

# Configurable settings through environment or sensible defaults
OLLAMA_SERVER_RAW = os.environ.get("OLLAMA_BASE_URL", os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
OLLAMA_SERVER_BASE = OLLAMA_SERVER_RAW[:-3] if OLLAMA_SERVER_RAW.endswith("/v1") else OLLAMA_SERVER_RAW
OLLAMA_API_V1 = f"{OLLAMA_SERVER_BASE}/v1"
OLLAMA_MODEL_DEFAULT = os.environ.get("OLLAMA_MODEL", os.environ.get("LLM_MODEL", "llama3.2:latest"))

NVIDIA_BASE_URL_DEFAULT = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_MODEL_DEFAULT = os.environ.get("NVIDIA_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")

PROVIDERS = {
    "nvidia": dict(
        base_url=NVIDIA_BASE_URL_DEFAULT, 
        key_env="NVIDIA_API_KEY", 
        model=NVIDIA_MODEL_DEFAULT
    ),
    "ollama": dict(
        base_url=OLLAMA_API_V1,
        key_env=None,
        model=OLLAMA_MODEL_DEFAULT
    ),
}


def check_ollama_status(server_base: str = OLLAMA_SERVER_BASE, model_name: str = OLLAMA_MODEL_DEFAULT) -> dict:
    """
    Verifies that the Ollama server is running and reachable,
    and checks if the configured model exists locally.
    Returns: {"connected": bool, "models": list[str], "model_exists": bool, "error": str | None}
    """
    url = f"{server_base}/api/tags"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "NaiyraCodingAgent"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            installed = [m.get("name", "") for m in data.get("models", [])]
            base_target = model_name.split(":")[0]
            exists = any(m == model_name or m.startswith(f"{base_target}:") for m in installed)
            return {
                "connected": True,
                "models": installed,
                "model_exists": exists,
                "error": None,
            }
    except Exception as e:
        return {
            "connected": False,
            "models": [],
            "model_exists": False,
            "error": str(e),
        }


def makeLLM(provider="nvidia", model=None, base_url=None):
    provider = provider or "nvidia"
    p = PROVIDERS.get(provider, {})
        
    base_url = base_url or p.get("base_url")
    model = model or p.get("model")
        
    key_env = p.get("key_env")
    if key_env:
        key = os.environ.get(key_env, "")
        if not key:
            sys.exit(f"Set the {key_env} environment variable.")
    else:
        key = "ollama"  # OpenAI client requires a non-empty key string for local endpoints
        
    if provider == "ollama":
        # Verify local Ollama server status and model presence
        server_base = base_url.replace("/v1", "").rstrip("/")
        status = check_ollama_status(server_base=server_base, model_name=model)
        if not status["connected"]:
            print(f"⚠️ Warning: Unable to connect to Ollama server at {server_base} ({status['error']})")
        elif not status["model_exists"]:
            print(f"⚠️ Warning: Configured model '{model}' not found in Ollama. Available: {status['models']}")
            if status["models"]:
                print(f"  → Defaulting to available local model: {status['models'][0]}")
                model = status["models"][0]

    client = OpenAI(base_url=base_url, api_key=key)

    def llm(system, message):
        for attempt in range(4):
            try:
                r = client.chat.completions.create(
                    model=model,
                    temperature=0.2,
                    messages=[{'role': "system", "content": system}] + message,
                )
                text = r.choices[0].message.content or ""
                return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
                
            except Exception as e:  # rate limit / transient error -> back off
                if attempt == 3:
                    raise
                print(f"  [llm retry {attempt + 1}: {str(e)[:80]}]")
                time.sleep(2 ** attempt * 2)
 
    return llm

