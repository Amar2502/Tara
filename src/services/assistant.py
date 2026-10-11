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
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src.config.config import configs
from src.core.llm import get_chat_model
from src.tools import get_all_tools
from src.utils.prompts import DEFAULT_SYSTEM_PROMPT


class GroqAssistant:
    """Conversational Assistant powered by LangGraph, ChatGroq, and standalone tools."""

    def __init__(
        self,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        model: str | None = None,
        temperature: float | None = None,
    ):
        self.system_prompt = system_prompt
        self.model_name = model or configs.GROQ_MODEL
        self.temperature = temperature if temperature is not None else configs.GROQ_TEMPERATURE

        # Initialize LLM and bind available tools
        self.llm = get_chat_model(model=self.model_name, temperature=self.temperature)
        self.tools = get_all_tools()
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # Thread / Session ID for persistent memory
        self.thread_id = str(uuid.uuid4())
        self.thread_config = {"configurable": {"thread_id": self.thread_id}}

        # In-memory checkpointer for conversation memory
        self.checkpointer = MemorySaver()

        # Build the LangGraph tool-calling agent
        self.graph = self._build_graph()

    def _build_graph(self):
        """Construct and compile the LangGraph workflow."""

        def chat_node(state: MessagesState):
            """Process messages with the LLM and system prompt."""
            messages = list(state["messages"])
            # Ensure the system prompt is always injected at the start
            if not messages or not isinstance(messages[0], SystemMessage):
                messages = [SystemMessage(content=self.system_prompt)] + messages

            response = self.llm_with_tools.invoke(messages)
            return {"messages": [response]}

        tool_node = ToolNode(self.tools)

        workflow = StateGraph(MessagesState)
        workflow.add_node("chat_node", chat_node)
        workflow.add_node("tools", tool_node)

        workflow.add_edge(START, "chat_node")
        workflow.add_conditional_edges("chat_node", tools_condition)
        workflow.add_edge("tools", "chat_node")

        return workflow.compile(checkpointer=self.checkpointer)

    @property
    def history(self) -> List[BaseMessage]:
        """Return the current conversation message history."""
        try:
            state = self.graph.get_state(self.thread_config)
            return list(state.values.get("messages", []))
        except Exception:
            return []

    @property
    def message_count(self) -> int:
        """Return total number of messages in the current conversation."""
        return len(self.history)

    def ask(self, user_input: str) -> str:
        """Send a message and return the assistant response synchronously."""
        final_state = self.graph.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=self.thread_config,
        )
        messages = final_state.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                return str(msg.content)
        return ""

    def stream_chat(self, user_input: str) -> Generator[str, None, None]:
        """Stream the assistant response token-by-token in real-time.

        Yields:
            str: Streamed token chunks and tool status notifications.
        """
        for chunk, metadata in self.graph.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=self.thread_config,
            stream_mode="messages",
        ):
            # If a tool just completed, notify the user
            if isinstance(chunk, ToolMessage):
                tool_name = getattr(chunk, "name", "tool")
                yield f"\n\n> *Tool `{tool_name}` finished processing.* \n\n"

            # Yield conversational content from the LLM node
            elif metadata.get("langgraph_node") == "chat_node":
                content = getattr(chunk, "content", "")
                if content and isinstance(content, str):
                    yield content

    def clear_history(self) -> None:
        """Reset conversation memory by generating a new thread session."""
        self.thread_id = str(uuid.uuid4())
        self.thread_config = {"configurable": {"thread_id": self.thread_id}}
