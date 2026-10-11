from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool


@tool
def search_tool(query: str) -> str:
    """Perform a web search using DuckDuckGo.

    Use this for quick facts, definitions, general knowledge, and recent information.
    """
    try:
        search = DuckDuckGoSearchRun()
        result = search.run(query)
        return result or "No search results found."
    except Exception as e:
        return f"Error executing DuckDuckGo search: {e}"


if __name__ == "__main__":
    test_query = "What is LangChain?"
    print(f"Testing search_tool with query: '{test_query}'")
    result = search_tool.invoke(test_query)
    print(f"Result:\n{result}")
