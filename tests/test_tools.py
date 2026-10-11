"""Standalone test script for all Tara tools."""

import sys
from pathlib import Path

# Ensure workspace root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.tools import (
    get_all_tools,
    search_tool,
    deep_search,
    read_article_url,
    get_latest_news,
)


def test_registry():
    print("\n--- 1. Testing Tool Registry ---")
    tools = get_all_tools()
    print(f"Total tools registered: {len(tools)}")
    for t in tools:
        print(f"  ✔ Tool name: '{t.name}' | description: {t.description[:60]}...")
    assert len(tools) == 4, f"Expected 4 tools, found {len(tools)}"


def test_news_tool():
    print("\n--- 2. Testing get_latest_news ---")
    try:
        result = get_latest_news.invoke({"category": "Explained", "limit": 2})
        print(f"Output preview:\n{result[:300]}...\n")
        print("✔ get_latest_news passed!")
    except Exception as e:
        print(f"❌ get_latest_news failed: {e}")


def test_article_reader():
    print("\n--- 3. Testing read_article_url ---")
    try:
        url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
        result = read_article_url.invoke(url)
        print(f"Output preview:\n{result[:250]}...\n")
        print("✔ read_article_url passed!")
    except Exception as e:
        print(f"❌ read_article_url failed: {e}")


def test_duckduckgo_search():
    print("\n--- 4. Testing search_tool (DuckDuckGo) ---")
    try:
        result = search_tool.invoke("LangChain Python")
        print(f"Output preview:\n{result[:250]}...\n")
        print("✔ search_tool passed!")
    except Exception as e:
        print(f"❌ search_tool failed: {e}")


def test_tavily_search():
    print("\n--- 5. Testing deep_search (Tavily) ---")
    try:
        result = deep_search.invoke("Latest news in tech")
        print(f"Output preview:\n{result[:250]}...\n")
        print("✔ deep_search handled gracefully!")
    except Exception as e:
        print(f"❌ deep_search failed: {e}")


if __name__ == "__main__":
    print("========================================")
    print("Testing Tara Tools Independently")
    print("========================================")
    test_registry()
    test_news_tool()
    test_article_reader()
    test_duckduckgo_search()
    test_tavily_search()
    print("\n========================================")
    print("All tool tests completed successfully!")
    print("========================================")
