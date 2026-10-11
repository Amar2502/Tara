"""LLM provider and model factory for LangChain and Groq."""

from langchain_groq import ChatGroq
from src.config.config import configs


def get_chat_model(
    model: str | None = None,
    temperature: float | None = None,
    streaming: bool | None = None,
) -> ChatGroq:
    """Factory function to instantiate and return a configured ChatGroq model.

    Args:
        model: The Groq model name (defaults to configs.GROQ_MODEL).
        temperature: Model sampling temperature (defaults to configs.GROQ_TEMPERATURE).
        streaming: Whether to stream tokens (defaults to configs.STREAMING).

    Returns:
        ChatGroq: Configured LangChain ChatGroq instance.

    Raises:
        ValueError: If GROQ_API_KEY is not configured.
    """
    configs.validate()

    return ChatGroq(
        model=model or configs.GROQ_MODEL,
        temperature=temperature if temperature is not None else configs.GROQ_TEMPERATURE,
        api_key=configs.GROQ_API_KEY,
        streaming=streaming if streaming is not None else configs.STREAMING,
    )
