"""Entry point for the Tara Conversational Assistant with optional Voice Mode."""

import argparse
from src.cli import run_interactive_chat


def main() -> None:
    """Launch the interactive terminal chat interface."""
    parser = argparse.ArgumentParser(
        description="Tara - Conversational Assistant powered by Groq & LangChain"
    )
    parser.add_argument(
        "--voice",
        "-v",
        action="store_true",
        help="Start Tara in voice conversation mode (speech-to-text and text-to-speech)",
    )
    args = parser.parse_args()

    run_interactive_chat(voice_mode=args.voice)


if __name__ == "__main__":
    main()
