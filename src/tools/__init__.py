"""Tools package providing standalone search, news, and web extraction capabilities."""

from src.tools.search_tool import search_tool
from src.tools.tavily_search import deep_search
from src.tools.get_article_by_url import read_article_url
from src.tools.get_news import get_latest_news

__all__ = [
    "search_tool",
    "deep_search",
    "read_article_url",
    "get_latest_news",
    "get_all_tools",
]


def get_all_tools() -> list:
    """Return all standalone tools available in Tara."""
    return [
        search_tool,
        deep_search,
        read_article_url,
        get_latest_news,
    ]
