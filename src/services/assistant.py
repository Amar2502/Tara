"""Assistant service managing conversation flow, LCEL chains, and message history."""

from typing import Generator
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser

from src.core.llm import get_chat_model
from src.utils.prompts import DEFAULT_SYSTEM_PROMPT
from src.config.config import configs


class GroqAssistant:
    """Conversational Assistant that manages chat chains and multi-turn message history."""

    def __init__(
        self,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        model: str | None = None,
        temperature: float | None = None,
    ):
        self.system_prompt = system_prompt
        self.model_name = model or configs.GROQ_MODEL
        self.temperature = temperature if temperature is not None else configs.GROQ_TEMPERATURE
        self.llm = get_chat_model(model=self.model_name, temperature=self.temperature)
        self._history: list[BaseMessage] = []

        # Construct LangChain Expression Language (LCEL) chain
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "{system_prompt}"),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ]
        )
        self.chain = self.prompt_template | self.llm | StrOutputParser()

    @property
    def history(self) -> list[BaseMessage]:
        """Return a copy of the current conversation history."""
        return list(self._history)

    @property
    def message_count(self) -> int:
        """Return total number of messages in the current conversation."""
        return len(self._history)

    def ask(self, user_input: str) -> str:
        """Send a message and return the full assistant response synchronously.

        Args:
            user_input: The message from the user.

        Returns:
            str: Full response from the assistant.
        """
        response: str = self.chain.invoke(
            {
                "system_prompt": self.system_prompt,
                "history": self._history,
                "input": user_input,
            }
        )
        self._history.append(HumanMessage(content=user_input))
        self._history.append(AIMessage(content=response))
        return response

    def stream_chat(self, user_input: str) -> Generator[str, None, None]:
        """Stream the assistant response token-by-token in real-time.

        Args:
            user_input: The message from the user.

        Yields:
            str: Individual streamed tokens/chunks.
        """
        full_response = ""
        for chunk in self.chain.stream(
            {
                "system_prompt": self.system_prompt,
                "history": self._history,
                "input": user_input,
            }
        ):
            full_response += chunk
            yield chunk

        # Persist conversation turn to history upon stream completion
        self._history.append(HumanMessage(content=user_input))
        self._history.append(AIMessage(content=full_response))

    def clear_history(self) -> None:
        """Reset the conversation memory."""
        self._history.clear()
