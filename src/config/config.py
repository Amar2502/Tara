import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# Locate root directory and load .env
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(dotenv_path=ROOT_DIR / ".env")


@dataclass(frozen=True)
class Configs:
    """Application configuration configs loaded from environment variables."""

    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL")
    GROQ_TEMPERATURE: float = float(os.getenv("GROQ_TEMPERATURE"))
    STREAMING: bool = os.getenv("STREAMING", "true").lower() in ("true", "1", "yes")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

    def validate(self) -> None:
        """Validate critical configs."""
        if not self.GROQ_API_KEY or self.GROQ_API_KEY.strip() == "":
            raise ValueError(
                "GROQ_API_KEY is not set. Please add it to your .env file."
            )


# Global configs instance
configs = Configs()
