import sqlite3
from langchain_core.messages import BaseMessage, RemoveMessage, SystemMessage, ToolMessage
from typing import List, Optional
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import MessagesState

conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)

# Create metadata table to map thread_id (UUID) -> thread_name
with conn:
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
    """Return persistant SQLite Checkpointer"""
    return SqliteSaver(conn=conn)

# 2. Thread Name Storage Helpers
def save_thread_name(thread_id: str, thread_name: str) -> None:
    """Save or update the human-readable thread name for a thread UUID."""
    with conn:
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
    cursor = conn.cursor()
    cursor.execute("SELECT thread_name FROM thread_metadata WHERE thread_id = ?", (thread_id,))
    row = cursor.fetchone()
    return row[0] if row else None


def get_all_threads() -> List[dict]:
    """Retrieve all stored threads with thread_id, thread_name, and created_at."""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT thread_id, thread_name, created_at FROM thread_metadata ORDER BY created_at DESC"
    )
    rows = cursor.fetchall()
    return [
        {"thread_id": r[0], "thread_name": r[1], "created_at": r[2]}
        for r in rows
    ]

# 3. Extended State to Hold the Running Summary
class ConversationState(MessagesState):
    """Extended state holding both message history and running summary."""
    summary: str

# 4. Sliding Window & Summarization Logic
def summarize_messages(
    messages: List[BaseMessage],
    existing_summary: str,
    llm,
    max_recent_messages: int = 6,
) -> dict:
    """
    Summarize older messages if history exceeds `max_recent_messages`.
    Keeps the most recent messages and removes older ones using RemoveMessage.
    """
    if len(messages) <= max_recent_messages:
        return {}
    # Messages to be summarized (everything older than the last N messages)
    to_summarize = messages[:-max_recent_messages]
    
    # Prompt the LLM to update the summary
    summary_prompt = (
        f"Extend the current summary of the conversation with these older messages.\n"
        f"Current Summary: {existing_summary or 'None'}\n\n"
        f"Older Messages to absorb:\n"
    )
    for msg in to_summarize:
        role = "User" if msg.type == "human" else "Assistant"
        summary_prompt += f"{role}: {msg.content}\n"
    summary_prompt += "\nWrite a concise summary capturing key details and facts."
    
    response = llm.invoke([SystemMessage(content=summary_prompt)])
    new_summary = response.content
    # Mark older messages for deletion from the LangGraph state
    delete_actions = [RemoveMessage(id=msg.id) for msg in to_summarize if msg.id]
    return {
        "summary": new_summary,
        "messages": delete_actions,
    }


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