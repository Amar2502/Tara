"""Interactive Terminal Chat for Tara Assistant using Rich."""

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.live import Live

from src.services.assistant import GroqAssistant
from src.config import configs

console = Console()


def show_welcome(assistant: GroqAssistant) -> None:
    """Displays a clean welcome banner in the terminal."""
    info_text = (
        f"[bold cyan]🤖 Tara - Groq & LangChain Assistant[/bold cyan]\n"
        f"[dim]Model:[/dim] [yellow]{assistant.model_name}[/yellow] | "
        f"[dim]Temperature:[/dim] [yellow]{configs.GROQ_TEMPERATURE}[/yellow]\n\n"
        f"[dim]Commands:[/dim] "
        f"[cyan]/clear[/cyan] to reset memory | [cyan]/help[/cyan] for commands | [cyan]exit[/cyan] to quit"
    )
    console.print(Panel(info_text, border_style="cyan", padding=(1, 2)))


def show_help() -> None:
    """Displays available commands."""
    help_text = (
        "[bold cyan]Commands:[/bold cyan]\n"
        "  [cyan]/clear[/cyan] : Reset conversation history\n"
        "  [cyan]/status[/cyan]: Show active model & message count\n"
        "  [cyan]/help[/cyan]  : Show this help message\n"
        "  [cyan]exit[/cyan]   : Exit the assistant"
    )
    console.print(Panel(help_text, border_style="cyan"))


def main() -> None:
    # Initialize the assistant
    try:
        assistant = GroqAssistant()
    except Exception as e:
        console.print(f"[bold red]Initialization failed: {e}[/bold red]")
        return

    # Show the startup banner
    show_welcome(assistant)

    # Main chat loop
    while True:
        try:
            # 1. Ask the user for input
            user_input = console.input("\n[bold blue]You:[/bold blue] ").strip()

            # Skip empty entries
            if not user_input:
                continue

            # 2. Handle commands
            cmd = user_input.lower()
            if cmd in ("exit", "quit", "q"):
                console.print("\n[bold yellow]Goodbye! Have a great day! 👋[/bold yellow]")
                break

            if cmd in ("/clear", "/reset"):
                assistant.clear_history()
                console.print("\n[yellow]✔ Conversation history cleared.[/yellow]")
                continue

            if cmd == "/status":
                console.print(
                    f"\n[cyan]Model:[/cyan] [yellow]{assistant.model_name}[/yellow] | "
                    f"[cyan]Messages in memory:[/cyan] [yellow]{assistant.message_count}[/yellow]"
                )
                continue

            if cmd == "/help":
                show_help()
                continue

            # 3. Stream the assistant's response live as rendered Markdown
            console.print("\n[bold green]Assistant:[/bold green]")
            full_response = ""

            with Live(Markdown(""), refresh_per_second=12, console=console) as live:
                for chunk in assistant.stream_chat(user_input):
                    full_response += chunk
                    live.update(Markdown(full_response))

        except (KeyboardInterrupt, EOFError):
            console.print("\n\n[bold yellow]Session ended. Goodbye![/bold yellow]")
            break
        except Exception as e:
            console.print(f"\n[bold red][Error] {e}[/bold red]")


if __name__ == "__main__":
    main()
