# Tara - Groq & LangGraph Conversational Assistant

A clean, modular conversational assistant built with **LangGraph**, **LangChain**, **ChatGroq**, and **Rich**, featuring standalone tools, memory checkpointing, and live terminal streaming.

---

## Project Structure

```text
Tara/
│
├── main.py                   # Entry point: Rich terminal chat loop & live Markdown
├── .env                      # Secret API keys & configs
├── requirements.txt          # Python dependencies
├── .gitignore                # Git ignore rules
│
├── src/
│   ├── cli/
│   │   ├── __init__.py
│   │   └── chat.py           # Presentation layer (Rich Live & ANSI fallback)
│   ├── config/
│   │   ├── __init__.py
│   │   └── config.py         # Configs dataclass loaded from .env
│   ├── core/
│   │   ├── __init__.py
│   │   └── llm.py            # ChatGroq model factory
│   ├── services/
│   │   ├── __init__.py
│   │   └── assistant.py      # Core agent: LangGraph + Tools + Memory
│   ├── tools/
│   │   ├── __init__.py       # Tools registry (get_all_tools)
│   │   ├── get_article_by_url.py # Webpage extraction via Trafilatura
│   │   ├── get_news.py       # RSS news fetcher from curated feeds
│   │   ├── search_tool.py    # Fast web search via DuckDuckGo
│   │   └── tavily_search.py  # Deep web search via Tavily
│   └── utils/
│       ├── __init__.py
│       ├── colors.py         # ANSI color formatting helpers
│       ├── prompts.py        # Default system prompt
│       └── rss_feeds.py      # RSS feed sources
│
└── tests/
    ├── test_agent.py         # End-to-end agent integration test
    └── test_tools.py         # Standalone tool tests
```

---

## Available Tools

1. **`search_tool`**: Fast web search via DuckDuckGo (definitions, quick facts, current info).
2. **`deep_search`**: Comprehensive web search via Tavily.
3. **`read_article_url`**: Downloads and extracts clean, readable text from any news or webpage URL using Trafilatura.
4. **`get_latest_news`**: Parses live RSS feeds from Indian Express across categories (`Politics`, `India`, `Explained`, `Deep Politics`, `World`).

---

## Configuration (`.env`)

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_TEMPERATURE=0.7
TAVILY_API_KEY=your_optional_tavily_api_key_here
```

---

## Running the Application

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the assistant:
```bash
python main.py
```

### Commands inside chat:
- `/clear` or `/reset`: Clears conversation history and generates a fresh session.
- `/status`: Shows active model and message count.
- `/help`: Displays available commands.
- `exit` or `quit`: Exits the session.

### Testing Tools & Agent Independently:
```bash
python -m tests.test_tools
python -m tests.test_agent
```
