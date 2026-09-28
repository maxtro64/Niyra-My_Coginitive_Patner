import ollama

def test_tools():
    llm_tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    }]

    print("Sending request...")
    stream = ollama.chat(
        model="qwen2.5:0.5b",
        messages=[{"role": "user", "content": "What's the weather in London?"}],
        stream=True,
        tools=llm_tools
    )

    for chunk in stream:
        print(chunk)
        if chunk.message.tool_calls:
            print("GOT TOOL CALLS:", chunk.message.tool_calls)

if __name__ == "__main__":
    test_tools()
