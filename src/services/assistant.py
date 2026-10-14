"""Assistant service managing conversation flow, LangGraph tool-calling agent, and memory."""

import uuid
from typing import Generator, List

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import MessagesState, START, StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition

from src.config.config import configs
from src.core.llm import get_chat_model
from src.core.memory import (
    ConversationState,
    get_all_threads,
    get_checkpointer,
    get_thread_name,
    save_thread_name,
    summarize_messages,
    trim_tool_messages,
)
from src.tools import get_all_tools 
from src.utils.prompts import DEFAULT_SYSTEM_PROMPT


class GroqAssistant:
    """Conversational Assistant powered by LangGraph, ChatGroq, and standalone tools."""

    def __init__(
        self,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        model: str | None = None,
        temperature: float | None = None,
        thread_id: str | None = None,
    ):
        self.system_prompt = system_prompt
        self.model_name = model or configs.GROQ_MODEL
        self.temperature = temperature if temperature is not None else configs.GROQ_TEMPERATURE

        # Initialize LLM and bind available tools
        self.llm = get_chat_model(model=self.model_name, temperature=self.temperature)
        self.tools = get_all_tools()
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # Thread / Session ID for persistent memory (lazy start on first message)
        self.thread_id: str | None = thread_id
        self.thread_config = (
            {"configurable": {"thread_id": self.thread_id}} if self.thread_id else {}
        )

        # SQLite checkpointer for conversation memory
        self.checkpointer = get_checkpointer()

        # Build the LangGraph tool-calling agent
        self.graph = self._build_graph()

    def _build_graph(self):
        """Construct and compile the LangGraph workflow."""

        def chat_node(state: ConversationState):
            """Prepend system prompt + summary and trim oversized tool responses before invoking LLM."""
            # Trim any oversized tool responses (no summarization)
            messages = trim_tool_messages(list(state["messages"]), max_chars=configs.MAX_TOOL_CHARS)
            summary = state.get("summary", "")
            # Compose system prompt with running summary if available
            system_content = self.system_prompt
            if summary:
                system_content += f"\n\n[Summary of earlier conversation]:\n{summary}"
            # Ensure system prompt is first
            if not messages or not isinstance(messages[0], SystemMessage):
                messages = [SystemMessage(content=system_content)] + messages
            response = self.llm_with_tools.invoke(messages)
            return {"messages": [response]}
        
        def summarize_node(state: ConversationState):
            """Summarize old messages if message count exceeds threshold."""
            return summarize_messages(
                messages=state["messages"],
                existing_summary=state.get("summary", ""),
                llm=self.llm,
                max_recent_messages=6,  # keep the 6 most recent messages
            )

        def route_after_chat(state: ConversationState):
            """Route to tools if requested, or summarize if threshold reached, else END."""
            if tools_condition(state) == "tools":
                return "tools"
            if len(state.get("messages", [])) > 6:
                return "summarize_node"
            return END

        base_tool_node = ToolNode(self.tools)

        def tool_node(state: ConversationState):
            """Execute tools and trim oversized responses before saving to state/checkpoint."""
            output = base_tool_node.invoke(state)
            output_messages = output.get("messages", [])
            trimmed_messages = trim_tool_messages(output_messages, max_chars=configs.MAX_TOOL_CHARS)
            return {"messages": trimmed_messages}

        workflow = StateGraph(ConversationState)
        workflow.add_node("chat_node", chat_node)
        workflow.add_node("tools", tool_node)
        workflow.add_node("summarize_node", summarize_node)

        workflow.add_edge(START, "chat_node")
        workflow.add_conditional_edges(
            "chat_node",
            route_after_chat,
            {"tools": "tools", "summarize_node": "summarize_node", END: END},
        )
        workflow.add_edge("tools", "chat_node")
        workflow.add_edge("summarize_node", END)

        return workflow.compile(checkpointer=self.checkpointer)

    @property
    def has_active_session(self) -> bool:
        """Return True if a conversation thread is currently active."""
        return self.thread_id is not None

    @property
    def thread_name(self) -> str:
        """Return the friendly name of the active thread."""
        if not self.thread_id:
            return "No Active Session"
        name = get_thread_name(self.thread_id)
        return name or "New Chat"

    @property
    def history(self) -> List[BaseMessage]:
        """Return the current conversation message history."""
        if not self.thread_id:
            return []
        try:
            state = self.graph.get_state(self.thread_config)
            return list(state.values.get("messages", []))
        except Exception:
            return []

    @property
    def message_count(self) -> int:
        """Return total number of messages in the current conversation."""
        return len(self.history)

    def _auto_generate_title(self, first_message: str) -> None:
        """Generate a concise title from the first message and store it."""
        try:
            messages = [
                SystemMessage(
                    content="You are a title generator. Respond ONLY with a concise 3 to 5 word topic title summarizing the user message. Do not include quotes, markdown, or punctuation."
                ),
                HumanMessage(
                    content=f"Generate a title for a chat that begins with: {first_message}"
                ),
            ]
            resp = self.llm.invoke(messages)
            title = str(resp.content).strip().strip('"\'*#')
            if not title or len(title) > 60:
                title = " ".join(first_message.split()[:5])
            save_thread_name(self.thread_id, title)
        except Exception:
            # Fallback to first few words of the user's prompt
            fallback_title = " ".join(first_message.split()[:5]) or "Conversation"
            save_thread_name(self.thread_id, fallback_title)

    def stream_chat(self, user_input: str) -> Generator[str, None, None]:
        """Stream response and lazily start/name new chat on first prompt."""
        # Start new thread on-demand if system is in standby
        is_first_message = False
        if not self.thread_id:
            self.thread_id = str(uuid.uuid4())
            self.thread_config = {"configurable": {"thread_id": self.thread_id}}
            is_first_message = True
        elif not self.history or len(self.history) == 0:
            is_first_message = True

        # Auto-title on the first message
        if is_first_message:
            self._auto_generate_title(user_input)

        for chunk, metadata in self.graph.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=self.thread_config,
            stream_mode="messages",
        ):
            if isinstance(chunk, ToolMessage):
                tool_name = getattr(chunk, "name", "tool")
                yield f"\n\n> *Tool `{tool_name}` finished processing.* \n\n"
            elif metadata.get("langgraph_node") == "chat_node":
                content = getattr(chunk, "content", "")
                if content and isinstance(content, str):
                    yield content

    def clear_history(self) -> None:
        """Reset conversation session to standby (new chat will begin on next message)."""
        self.thread_id = None
        self.thread_config = {}

    def switch_thread(self, thread_id: str) -> None:
        """Switch active conversation to a specific thread UUID."""
        self.thread_id = thread_id
        self.thread_config = {"configurable": {"thread_id": self.thread_id}}

    def list_threads(self) -> List[dict]:
        """Return all saved threads with their ID, name, and timestamp."""
        return get_all_threads()