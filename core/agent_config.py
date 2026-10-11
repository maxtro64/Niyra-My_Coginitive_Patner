"""
Configuration for mapping specialized agents to specific LLM models/providers.
"""
import os

# Define which provider and model each agent should use
AGENT_LLM_CONFIG = {
    # The main supervisor that routes tasks
    "supervisor": {
        "provider": "laya",
        "model": "laya",
        "temperature": 0.0
    },
    
    # Specialized agents
    "code_agent": {
        "provider": "nvidia",
        "model": "nvidia/nemotron-3-ultra-550b-a55b",
        "temperature": 0.2
    },
    "rag_agent": {
        "provider": "ollama",
        "model": "llama3.2:latest",
        "temperature": 0.1
    },
    "web_search_agent": {
        "provider": "nvidia",
        "model": "nvidia/nemotron-3-ultra-550b-a55b",
        "temperature": 0.5
    },
    "web_agent": {
        "provider": "nvidia",
        "model": "nvidia/nemotron-3-super-120b-a12b",
        "temperature": 0.5
    },
    "data_analyst_agent": {
        "provider": "nvidia",
        "model": "nvidia/nemotron-3-ultra-550b-a55b",
        "temperature": 0.1
    },
    "general_agent": {
        "provider": "ollama",
        "model": "llama3.2:latest",
        "temperature": 0.7
    }
}

def get_llm_for_agent(agent_name: str):
    """
    Factory function to initialize and return the correct LangChain LLM 
    instance based on the agent config mapping.
    """
    # Fallback to general agent config if agent name not found
    config = AGENT_LLM_CONFIG.get(agent_name, AGENT_LLM_CONFIG["general_agent"])
    provider = config.get("provider", "ollama")
    model = config.get("model", "llama3")
    temperature = config.get("temperature", 0.0)

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        # Note: ChatOllama needs ollama running locally on port 11434
        return ChatOllama(model=model, temperature=temperature)
        
    if provider == "laya":
        return "laya"
        
    elif provider == "nvidia":
        from langchain_openai import ChatOpenAI
        api_key = os.environ.get("NVIDIA_API_KEY", "")
        return ChatOpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key,  # type: ignore[arg-type]
            model=model,
            temperature=temperature
        )
        
    else:
        raise ValueError(f"Unknown provider: {provider}")
