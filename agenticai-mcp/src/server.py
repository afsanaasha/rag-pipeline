"""
Simple MCP-style tool server (educational demo).
This is NOT a full MCP implementation, but shows the core ideas:
- registering tools
- sending context
- returning structured responses
"""

from typing import Dict, Any, Callable

TOOLS = {}

def tool(name: str):
    def decorator(func: Callable):
        TOOLS[name] = func
        return func
    return decorator

@tool("summarize_text")
def summarize_text(context: Dict[str, Any]):
    text = context.get("text", "")
    words = text.split()
    short = " ".join(words[:20])
    return {
        "summary": short + ("..." if len(words) > 20 else "")
    }

@tool("word_count")
def word_count(context: Dict[str, Any]):
    text = context.get("text", "")
    return {
        "words": len(text.split())
    }

def handle_request(tool_name: str, context: Dict[str, Any]):
    if tool_name not in TOOLS:
        return {"error": "Tool not found"}
    return TOOLS[tool_name](context)


if __name__ == "__main__":
    sample_context = {
        "text": "Model Context Protocol allows AI systems to securely connect to external tools and data sources."
    }

    print("Available tools:", list(TOOLS.keys()))
    result = handle_request("summarize_text", sample_context)
    print("Result:", result)
