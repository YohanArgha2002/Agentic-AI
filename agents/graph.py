import os
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from dotenv import load_dotenv
load_dotenv(override=True)

def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

model = ChatOpenAI(
    model=os.getenv("OLLAMA_MODEL", "llama3"),
    base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1"),
    api_key=os.getenv("OPENAI_API_KEY", "ollama"),
    temperature=0,
    model_kwargs={"response_format": {"type": "json_object"}},
)
tools = [add, multiply]
agent = create_react_agent(
    # disable parallel tool calls
    model=model.bind_tools(tools, parallel_tool_calls=False),
    tools=tools
)
