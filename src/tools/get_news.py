"""Tool for fetching and parsing the latest news from RSS feeds."""

import html
import re
import xml.etree.ElementTree as ET
from typing import List, Dict, Any

import httpx
from langchain_core.tools import tool

from src.utils.rss_feeds import NEWS_URLS

# Browser User-Agent to avoid HTTP 403 blocks on RSS endpoints
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/130.0 Safari/537.36"
    )
}


def clean_html(raw_html: str) -> str:
    """Strip HTML tags and unescape HTML entities from feed descriptions."""
    if not raw_html:
        return ""
    # Remove HTML tags
    cleaned = re.sub(r"<[^>]+>", " ", raw_html)
    # Unescape HTML entities like &nbsp;, &amp;, &quot;
    cleaned = html.unescape(cleaned)
    # Normalize extra whitespace
    return " ".join(cleaned.split())


def parse_rss_xml(xml_content: str, max_items: int = 5) -> List[Dict[str, str]]:
    """Parse RSS 2.0 XML and return a list of article dictionaries."""
    articles = []
    try:
        root = ET.fromstring(xml_content)
        # Find all <item> tags in the RSS feed
        items = root.findall(".//item")
        for item in items[:max_items]:
            title = item.findtext("title", default="").strip()
            link = item.findtext("link", default="").strip()
            pub_date = item.findtext("pubDate", default="").strip()
            desc = clean_html(item.findtext("description", default=""))

            if title:
                articles.append(
                    {
                        "title": title,
                        "link": link,
                        "published": pub_date,
                        "summary": desc[:250] + ("..." if len(desc) > 250 else ""),
                    }
                )
    except Exception as e:
        # If standard XML parse fails, attempt regex fallback for resilient parsing
        item_matches = re.findall(r"<item>(.*?)</item>", xml_content, re.DOTALL)
        for raw_item in item_matches[:max_items]:
            title_m = re.search(r"<title>(.*?)</title>", raw_item, re.DOTALL)
            link_m = re.search(r"<link>(.*?)</link>", raw_item, re.DOTALL)
            pub_m = re.search(r"<pubDate>(.*?)</pubDate>", raw_item, re.DOTALL)
            desc_m = re.search(r"<description>(.*?)</description>", raw_item, re.DOTALL)

            title = clean_html(title_m.group(1)) if title_m else ""
            link = clean_html(link_m.group(1)) if link_m else ""
            pub_date = clean_html(pub_m.group(1)) if pub_m else ""
            desc = clean_html(desc_m.group(1)) if desc_m else ""

            if title:
                articles.append(
                    {
                        "title": title,
                        "link": link,
                        "published": pub_date,
                        "summary": desc[:250] + ("..." if len(desc) > 250 else ""),
                    }
                )
    return articles


def fetch_category_feed(url: str, limit: int = 3) -> List[Dict[str, str]]:
    """Fetch and parse feed for a given RSS URL."""
    try:
        with httpx.Client(headers=HEADERS, timeout=12.0, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()
            return parse_rss_xml(response.text, max_items=limit)
    except Exception as e:
        return []


def _fetch_news_content(category: str = "all", limit: int = 3) -> str:
    category_lower = category.strip().lower()

    # Determine which feeds to query
    if category_lower in ("all", "", "any"):
        target_feeds = NEWS_URLS
    else:
        target_feeds = [
            f for f in NEWS_URLS if category_lower in f["category"].lower()
        ]
        if not target_feeds:
            avail_categories = ", ".join(f["category"] for f in NEWS_URLS)
            fallback = _fetch_news_content("all", limit=2)
            return (
                f"Category '{category}' not found. Available categories are: {avail_categories}.\n"
                f"Showing top news from all categories instead:\n\n{fallback}"
            )

    output_lines = [f"## Latest News (Filter: {category.title()})\n"]
    total_found = 0

    for feed_info in target_feeds:
        feed_category = feed_info["category"]
        feed_url = feed_info["url"]

        articles = fetch_category_feed(feed_url, limit=limit)
        if not articles:
            continue

        output_lines.append(f"### Category: {feed_category}")
        for idx, art in enumerate(articles, start=1):
            total_found += 1
            output_lines.append(f"{idx}. **{art['title']}**")
            if art["published"]:
                output_lines.append(f"   - *Published:* {art['published']}")
            if art["summary"]:
                output_lines.append(f"   - *Summary:* {art['summary']}")
            if art["link"]:
                output_lines.append(f"   - *Link:* {art['link']}")
            output_lines.append("")

    if total_found == 0:
        return "Could not retrieve news articles at this moment. Please check your internet connection."

    return "\n".join(output_lines)


@tool
def get_latest_news(category: str = "all", limit: int = 3) -> str:
    """Fetch the latest headlines and summaries from curated RSS news feeds.

    Args:
        category: Filter by category (e.g., 'Politics', 'India', 'Explained', 'Deep Politics', 'World', or 'all').
        limit: Number of top articles to fetch per category (default: 3).

    Returns:
        Structured markdown summary of the latest news articles with titles, dates, links, and summaries.
    """
    return _fetch_news_content(category=category, limit=limit)


if __name__ == "__main__":
    print("Testing get_latest_news with category='Explained', limit=2:")
    result = get_latest_news.invoke({"category": "Explained", "limit": 2})
    print(result)