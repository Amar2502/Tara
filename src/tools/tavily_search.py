from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from src.config.config import configs


@tool
def deep_search(query: str) -> str:
    """Use this to make advanced search about deep topics, recent news, or current affairs using Tavily."""
    api_key = configs.TAVILY_API_KEY
    if not api_key or api_key.strip() == "":
        return (
            "Error: TAVILY_API_KEY is not set. "
            "Please add TAVILY_API_KEY=your_key to your .env file to use deep search."
        )

    try:
        tavily = TavilySearch(
            max_results=5,
            topic="general",
            tavily_api_key=api_key,
        )
        result = tavily.invoke({"query": query})
        return str(result)
    except Exception as e:
        return f"Error executing Tavily search: {e}"


if __name__ == "__main__":
    test_query = "What is the latest advancement in quantum computing?"
    print(f"Testing deep_search with query: '{test_query}'")
    result = deep_search.invoke(test_query)
    print(f"Result:\n{result}")