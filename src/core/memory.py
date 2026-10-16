import sqlite3
import threading
from pathlib import Path
from typing import List, Optional
from langchain_core.messages import BaseMessage, RemoveMessage, SystemMessage, ToolMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import MessagesState

# Absolute path to SQLite database to avoid working-directory resolution bugs
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DB_PATH = ROOT_DIR / "chatbot.db"

_db_lock = threading.RLock()
conn = sqlite3.connect(database=str(DB_PATH), check_same_thread=False, timeout=30.0)

# Configure WAL mode and create metadata table to map thread_id (UUID) -> thread_name
with _db_lock, conn:
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS thread_metadata (
            thread_id TEXT PRIMARY KEY,
            thread_name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

def get_checkpointer():
    """Return persistent SQLite Checkpointer"""
    return SqliteSaver(conn=conn)

# 2. Thread Name Storage Helpers
def save_thread_name(thread_id: str, thread_name: str) -> None:
    """Save or update the human-readable thread name for a thread UUID (thread-safe)."""
    with _db_lock, conn:
        conn.execute(
            """
            INSERT INTO thread_metadata (thread_id, thread_name)
            VALUES (?, ?)
            ON CONFLICT(thread_id) DO UPDATE SET thread_name=excluded.thread_name
            """,
            (thread_id, thread_name),
        )

def get_thread_name(thread_id: str) -> Optional[str]:
    """Retrieve the human-readable thread name for a thread UUID."""
    with _db_lock:
        cursor = conn.cursor()
        cursor.execute("SELECT thread_name FROM thread_metadata WHERE thread_id = ?", (thread_id,))
        row = cursor.fetchone()
        return row[0] if row else None


def get_all_threads() -> List[dict]:
    """Retrieve all stored threads with thread_id, thread_name, and created_at."""
    with _db_lock:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT thread_id, thread_name, created_at FROM thread_metadata ORDER BY created_at DESC"
        )
        rows = cursor.fetchall()
        return [
            {"thread_id": r[0], "thread_name": r[1], "created_at": r[2]}
            for r in rows
        ]

# 3. Extended State to Hold Running Summary and Active System Prompt
class ConversationState(MessagesState):
    """Extended state holding message history, running summary, and persisted system prompt."""
    summary: str
    system_prompt: str

# 4. Sliding Window & Summarization Logic
def summarize_messages(
    messages: List[BaseMessage],
    existing_summary: str,
    llm,
    max_recent_messages: int = 8,
) -> dict:
    """
    Summarize older messages if history exceeds `max_recent_messages`.
    Safely keeps tool call pairs intact and removes older messages using RemoveMessage.
    """
    if len(messages) <= max_recent_messages:
        return {}

    # Find safe cut point: never sever an AIMessage with tool_calls from its ToolMessage
    cutoff = len(messages) - max_recent_messages
    if cutoff <= 0:
        return {}

    # If the cutoff lands directly on a ToolMessage, back up so it stays with its AIMessage
    while cutoff > 0 and isinstance(messages[cutoff], ToolMessage):
        cutoff -= 1

    # If messages[cutoff - 1] is an AIMessage with tool_calls, back up so the pair is not split
    if cutoff > 0:
        prev_msg = messages[cutoff - 1]
        if getattr(prev_msg, "tool_calls", None):
            cutoff -= 1

    if cutoff <= 0:
        return {}

    to_summarize = messages[:cutoff]
    if not to_summarize:
        return {}

    # Prompt the LLM to update the summary
    summary_body = (
        f"Extend the current summary of the conversation with these older messages.\n"
        f"Current Summary: {existing_summary or 'None'}\n\n"
        f"Older Messages to absorb:\n"
    )
    for msg in to_summarize:
        if msg.type == "human":
            role = "User"
        elif msg.type == "tool":
            tool_name = getattr(msg, "name", "tool")
            role = f"Tool Result ({tool_name})"
        elif msg.type == "system":
            role = "System"
        else:
            role = "Assistant"

        content_str = str(msg.content) if msg.content is not None else ""
        if len(content_str) > 1000:
            content_str = content_str[:1000] + "... [trimmed]"
        summary_body += f"{role}: {content_str}\n"

    summary_body += "\nWrite a concise summary capturing key facts and decisions without losing context."

    try:
        # Groq Jinja templates require at least one HumanMessage in the messages array
        response = llm.invoke([
            SystemMessage(
                content="You are a concise conversation summarizer. Summarize key facts, context, and outcomes accurately."
            ),
            HumanMessage(content=summary_body),
        ])
        new_summary = str(response.content).strip()
        delete_actions = [RemoveMessage(id=msg.id) for msg in to_summarize if getattr(msg, "id", None)]
        return {
            "summary": new_summary,
            "messages": delete_actions,
        }
    except Exception:
        # Fall back gracefully without crashing the assistant chat loop
        return {}


# 5. Direct Tool Response Trimming (No Summarization)
def trim_tool_message(msg: ToolMessage, max_chars: int = 6000) -> ToolMessage:
    """Trim oversized ToolMessage content directly without summarization to fit within Groq context/rate limits."""
    content = msg.content
    content_str = str(content) if content is not None else ""
    if len(content_str) > max_chars:
        trimmed_content = (
            content_str[:max_chars]
            + f"\n\n... [Tool output trimmed: exceeded {max_chars} character limit to prevent context window overflow]"
        )
        return ToolMessage(
            content=trimmed_content,
            tool_call_id=msg.tool_call_id,
            name=getattr(msg, "name", None),
            id=getattr(msg, "id", None),
            status=getattr(msg, "status", "success"),
        )
    return msg


def trim_tool_messages(messages: List[BaseMessage], max_chars: int = 6000) -> List[BaseMessage]:
    """Inspect and truncate any oversized ToolMessage in a message sequence."""
    return [
        trim_tool_message(m, max_chars) if isinstance(m, ToolMessage) else m
        for m in messages
    ]
