"""Assistant service managing conversation flow, LangGraph tool-calling agent, and memory."""

import asyncio
import threading
import uuid
from typing import AsyncGenerator, Generator, List, Optional

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import START, StateGraph, END
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
    """Conversational Assistant powered by LangGraph, ChatGroq, and standalone tools.
    
    Supports both synchronous and asynchronous streaming, non-blocking background
    title generation, live tool execution feedback, and stateful system prompts.
    """

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
        """Construct and compile the LangGraph workflow with stateful system prompt and memory."""

        def chat_node(state: ConversationState):
            """Prepend active system prompt + summary and trim oversized tool responses before invoking LLM."""
            # Trim any oversized tool responses (no summarization)
            messages = trim_tool_messages(list(state["messages"]), max_chars=configs.MAX_TOOL_CHARS)
            summary = state.get("summary", "")

            # Retrieve persisted system prompt from state or fall back to instance default
            active_system_prompt = state.get("system_prompt") or self.system_prompt

            # Compose system prompt with running summary if available
            system_content = active_system_prompt
            if summary:
                system_content += f"\n\n[Summary of earlier conversation]:\n{summary}"

            # Filter out any pre-existing SystemMessage to prevent duplicates in prompt
            prompt_messages = [SystemMessage(content=system_content)]
            for m in messages:
                if not isinstance(m, SystemMessage):
                    prompt_messages.append(m)

            # Ensure at least one HumanMessage exists to satisfy Groq/Jinja chat template requirements
            if not any(isinstance(m, HumanMessage) for m in prompt_messages):
                prompt_messages.insert(1, HumanMessage(content="Please assist based on the conversation context."))

            response = self.llm_with_tools.invoke(prompt_messages)

            # Persist system prompt and message response into state checkpoint
            node_output = {"messages": [response]}
            if not state.get("system_prompt"):
                node_output["system_prompt"] = active_system_prompt
            return node_output

        def summarize_node(state: ConversationState):
            """Summarize old messages if message count exceeds threshold."""
            return summarize_messages(
                messages=state["messages"],
                existing_summary=state.get("summary", ""),
                llm=self.llm,
                max_recent_messages=8,  # retain the 8 most recent messages
            )

        def route_after_chat(state: ConversationState):
            """Route to tools if requested, or summarize if threshold reached, else END."""
            if tools_condition(state) == "tools":
                return "tools"
            if len(state.get("messages", [])) > 14:
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

    def update_system_prompt(self, new_prompt: str) -> None:
        """Dynamically update active system prompt and persist it in the current session checkpoint."""
        self.system_prompt = new_prompt
        if self.thread_id and self.thread_config:
            try:
                self.graph.update_state(self.thread_config, {"system_prompt": new_prompt})
            except Exception:
                pass

    def _spawn_title_generation(self, thread_id: str, first_message: str) -> None:
        """Generate title in a background daemon thread without blocking first-token streaming."""
        def _worker():
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
                if title and len(title) <= 60:
                    save_thread_name(thread_id, title)
            except Exception:
                pass

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

    async def _aauto_generate_title(self, thread_id: str, first_message: str) -> None:
        """Asynchronously generate a concise title using LLM ainvoke."""
        try:
            messages = [
                SystemMessage(
                    content="You are a title generator. Respond ONLY with a concise 3 to 5 word topic title summarizing the user message. Do not include quotes, markdown, or punctuation."
                ),
                HumanMessage(
                    content=f"Generate a title for a chat that begins with: {first_message}"
                ),
            ]
            resp = await self.llm.ainvoke(messages)
            title = str(resp.content).strip().strip('"\'*#')
            if title and len(title) <= 60:
                save_thread_name(thread_id, title)
        except Exception:
            pass

    async def _aspawn_title_generation(self, thread_id: str, first_message: str) -> None:
        """Schedule asynchronous title generation on the active asyncio event loop."""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._aauto_generate_title(thread_id, first_message))
        except RuntimeError:
            self._spawn_title_generation(thread_id, first_message)

    def stream_chat(self, user_input: str) -> Generator[str, None, None]:
        """Stream response synchronously with live tool progress notifications and instant first-token delivery."""
        is_first_message = False
        if not self.thread_id:
            self.thread_id = str(uuid.uuid4())
            self.thread_config = {"configurable": {"thread_id": self.thread_id}}
            is_first_message = True
        elif not self.history or len(self.history) == 0:
            is_first_message = True

        # Non-blocking auto-title: Save fast provisional title and refine in background
        if is_first_message:
            provisional_title = " ".join(user_input.split()[:5]) or "New Session"
            save_thread_name(self.thread_id, provisional_title)
            self._spawn_title_generation(self.thread_id, user_input)

        notified_tool_calls: set = set()

        for chunk, metadata in self.graph.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=self.thread_config,
            stream_mode="messages",
        ):
            # 1. Immediate feedback when tool execution is requested by LLM
            for tc in getattr(chunk, "tool_call_chunks", []):
                call_id = tc.get("id") or tc.get("index", 0)
                name = tc.get("name")
                if name and call_id not in notified_tool_calls:
                    notified_tool_calls.add(call_id)
                    yield f"\n\n> ⚙️ *Executing tool `{name}`...*\n\n"

            for tc in getattr(chunk, "tool_calls", []):
                call_id = tc.get("id") or tc.get("name")
                name = tc.get("name")
                if name and call_id not in notified_tool_calls:
                    notified_tool_calls.add(call_id)
                    yield f"\n\n> ⚙️ *Executing tool `{name}`...*\n\n"

            # 2. Feedback when tool finishes processing
            if isinstance(chunk, ToolMessage):
                tool_name = getattr(chunk, "name", "tool")
                yield f"> ✔️ *Tool `{tool_name}` finished processing.* \n\n"

            # 3. Stream generated response tokens
            elif metadata.get("langgraph_node") == "chat_node":
                content = getattr(chunk, "content", "")
                if content and isinstance(content, str):
                    yield content

    async def astream_chat(self, user_input: str) -> AsyncGenerator[str, None]:
        """Stream response asynchronously with live tool progress notifications and instant first-token delivery."""
        is_first_message = False
        if not self.thread_id:
            self.thread_id = str(uuid.uuid4())
            self.thread_config = {"configurable": {"thread_id": self.thread_id}}
            is_first_message = True
        elif not self.history or len(self.history) == 0:
            is_first_message = True

        # Non-blocking auto-title: Save fast provisional title and refine concurrently
        if is_first_message:
            provisional_title = " ".join(user_input.split()[:5]) or "New Session"
            save_thread_name(self.thread_id, provisional_title)
            await self._aspawn_title_generation(self.thread_id, user_input)

        notified_tool_calls: set = set()

        async for chunk, metadata in self.graph.astream(
            {"messages": [HumanMessage(content=user_input)]},
            config=self.thread_config,
            stream_mode="messages",
        ):
            # 1. Immediate feedback when tool execution is requested by LLM
            for tc in getattr(chunk, "tool_call_chunks", []):
                call_id = tc.get("id") or tc.get("index", 0)
                name = tc.get("name")
                if name and call_id not in notified_tool_calls:
                    notified_tool_calls.add(call_id)
                    yield f"\n\n> ⚙️ *Executing tool `{name}`...*\n\n"

            for tc in getattr(chunk, "tool_calls", []):
                call_id = tc.get("id") or tc.get("name")
                name = tc.get("name")
                if name and call_id not in notified_tool_calls:
                    notified_tool_calls.add(call_id)
                    yield f"\n\n> ⚙️ *Executing tool `{name}`...*\n\n"

            # 2. Feedback when tool finishes processing
            if isinstance(chunk, ToolMessage):
                tool_name = getattr(chunk, "name", "tool")
                yield f"> ✔️ *Tool `{tool_name}` finished processing.* \n\n"

            # 3. Stream generated response tokens
            elif metadata.get("langgraph_node") == "chat_node":
                content = getattr(chunk, "content", "")
                if content and isinstance(content, str):
                    yield content

    def invoke(self, user_input: str) -> str:
        """Synchronously execute a full conversation turn and return the final string response."""
        full_response = []
        for chunk in self.stream_chat(user_input):
            full_response.append(chunk)
        return "".join(full_response)

    async def ainvoke(self, user_input: str) -> str:
        """Asynchronously execute a full conversation turn and return the final string response."""
        full_response = []
        async for chunk in self.astream_chat(user_input):
            full_response.append(chunk)
        return "".join(full_response)

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