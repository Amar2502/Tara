import os

# Enable virtual terminal processing on Windows CMD / PowerShell
if os.name == "nt":
    os.system("")


class Colors:
    """ANSI color codes for terminal formatting."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Standard colors
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    ORANGE = "\033[38;5;208m"
    RED = "\033[91m"
    GRAY = "\033[90m"

    @classmethod
    def user(cls, text: str) -> str:
        """Format text as User prompt (Bold Blue)."""
        return f"{cls.BOLD}{cls.BLUE}{text}{cls.RESET}"

    @classmethod
    def assistant(cls, text: str) -> str:
        """Format text as Assistant label (Bold Green)."""
        return f"{cls.BOLD}{cls.GREEN}{text}{cls.RESET}"

    @classmethod
    def error(cls, text: str) -> str:
        """Format text as Error message (Bold Orange tag + Red text)."""
        return f"{cls.BOLD}{cls.ORANGE}[Error] {cls.RED}{text}{cls.RESET}"

    @classmethod
    def warning(cls, text: str) -> str:
        """Format text as Warning (Orange)."""
        return f"{cls.ORANGE}{text}{cls.RESET}"

    @classmethod
    def notice(cls, text: str) -> str:
        """Format text as Notification / Notice (Yellow)."""
        return f"{cls.YELLOW}{text}{cls.RESET}"

    @classmethod
    def info(cls, text: str) -> str:
        """Format text as Informational (Cyan)."""
        return f"{cls.CYAN}{text}{cls.RESET}"

    @classmethod
    def subtle(cls, text: str) -> str:
        """Format text as Dim / Gray."""
        return f"{cls.GRAY}{text}{cls.RESET}"
