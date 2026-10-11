import trafilatura
from langchain_core.tools import tool


@tool
def read_article_url(url: str) -> str:
    """Fetch and extract the full readable text content of a news article or webpage URL using Trafilatura.

    Use this to read, browse, and quote the actual body of an article when a headline or summary is not enough.
    """
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return f"Error: Could not download webpage content from {url}."

        text = trafilatura.extract(downloaded)
        if not text:
            return f"Error: Could not extract readable article text from {url}."

        max_chars = 3500
        if len(text) > max_chars:
            return f"{text[:max_chars]}\n\n[Content truncated. Showing first {max_chars} characters...]"
        return text
    except Exception as e:
        return f"Error reading article from {url}: {e}"


if __name__ == "__main__":
    test_url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    print(f"Testing read_article_url with: {test_url}")
    result = read_article_url.invoke(test_url)
    print(f"Result preview (first 300 chars):\n{result[:300]}...")