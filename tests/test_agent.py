"""Verification test for Tara LangGraph Assistant and Tools integration."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config.config import configs
from src.core.llm import get_chat_model
from src.tools import get_all_tools
from src.services.assistant import GroqAssistant


def test_agent_initialization():
    print("--- 1. Testing Configs & LLM Factory ---")
    print(f"Model: {configs.GROQ_MODEL}")
    print(f"Temperature: {configs.GROQ_TEMPERATURE}")
    model = get_chat_model()
    print(f"✔ LLM successfully instantiated: {model.__class__.__name__}")

    print("\n--- 2. Testing Tools Registration ---")
    tools = get_all_tools()
    print(f"✔ Registered tools count: {len(tools)}")
    for t in tools:
        print(f"  - Tool: {t.name}")

    print("\n--- 3. Testing GroqAssistant Graph Compilation ---")
    assistant = GroqAssistant()
    print("✔ Assistant initialized successfully with LangGraph!")
    print(f"Active Thread ID: {assistant.thread_id}")
    print(f"Initial Message Count: {assistant.message_count}")
    assert assistant.graph is not None, "Graph compilation failed"
    print("✔ LangGraph compiled StateGraph verified!")


if __name__ == "__main__":
    print("========================================")
    print("Running Tara End-to-End Integration Test")
    print("========================================")
    test_agent_initialization()
    print("\n========================================")
    print("All integration checks passed!")
    print("========================================")
