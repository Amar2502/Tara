"""Interactive CLI presentation layer using Rich for Markdown rendering."""

import sys
from src.services.assistant import GroqAssistant
from src.utils.colors import Colors
from src.config.config import configs

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.live import Live
    from rich.table import Table

    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None


def print_rich_banner(assistant: GroqAssistant) -> None:
    """Render welcome banner using Rich Panel."""
    content = (
        f"[bold cyan]🤖 Tara - Groq & LangChain Assistant[/bold cyan]\n"
        f"[dim]Model:[/dim] [yellow]{assistant.model_name}[/yellow]  |  "
        f"[dim]Temperature:[/dim] [yellow]{assistant.temperature}[/yellow]\n\n"
        f"[dim]Commands:[/dim] "
        f"[bold cyan]/clear[/bold cyan] reset  |  "
        f"[bold cyan]/status[/bold cyan] status  |  "
        f"[bold cyan]/help[/bold cyan] help  |  "
        f"[bold cyan]exit[/bold cyan] quit"
    )
    console.print(
        Panel(
            content,
            title="[bold green]Chat Session[/bold green]",
            border_style="cyan",
            padding=(1, 2),
        )
    )


def print_rich_help() -> None:
    """Print available commands as a Rich table."""
    table = Table(title="Available Commands", border_style="cyan")
    table.add_column("Command", style="bold cyan")
    table.add_column("Description", style="white")
    table.add_row("/clear, /reset", "Clear conversation memory")
    table.add_row("/status", "View active model and message count")
    table.add_row("/help", "Show this help table")
    table.add_row("exit, quit, q", "Exit the assistant session")
    console.print(table)


def run_rich_chat(assistant: GroqAssistant) -> None:
    """Interactive loop utilizing Rich Markdown and Live streaming."""
    print_rich_banner(assistant)

    while True:
        try:
            user_input = console.input("\n[bold blue]You:[/bold blue] ").strip()

            if not user_input:
                continue

            cmd = user_input.lower()
            if cmd in ("exit", "quit", "q"):
                console.print("\n[bold yellow]Goodbye! Have a great day! 👋[/bold yellow]")
                break

            if cmd in ("/clear", "/reset"):
                assistant.clear_history()
                console.print("\n[yellow]✔ Conversation memory cleared.[/yellow]")
                continue

            if cmd == "/status":
                console.print(
                    Panel(
                        f"[dim]Model:[/dim] [bold yellow]{assistant.model_name}[/bold yellow]\n"
                        f"[dim]Messages in memory:[/dim] [bold yellow]{assistant.message_count}[/bold yellow]",
                        title="[bold cyan]Assistant Status[/bold cyan]",
                        border_style="cyan",
                    )
                )
                continue

            if cmd == "/help":
                print_rich_help()
                continue

            # Stream markdown response
            console.print("\n[bold green]Assistant:[/bold green]")
            full_response = ""
            with Live(Markdown(""), refresh_per_second=12, console=console) as live:
                for chunk in assistant.stream_chat(user_input):
                    full_response += chunk
                    live.update(Markdown(full_response))

        except (KeyboardInterrupt, EOFError):
            console.print("\n\n[bold yellow]Session interrupted. Goodbye![/bold yellow]")
            break
        except Exception as e:
            console.print(f"\n[bold orange3][Error][/bold orange3] [bold red]{e}[/bold red]")


def run_ansi_chat(assistant: GroqAssistant) -> None:
    """Fallback interactive loop using standard ANSI colors."""
    border = f"{Colors.CYAN}{'=' * 60}{Colors.RESET}"
    print(border)
    print(f"{Colors.BOLD}{Colors.CYAN}🤖 Tara - Groq & LangChain Assistant{Colors.RESET}")
    print(
        f"{Colors.subtle('Model:')} {Colors.notice(assistant.model_name)} | "
        f"{Colors.subtle('Temperature:')} {Colors.notice(str(assistant.temperature))}"
    )
    print(
        f"{Colors.warning('[Tip] Install rich (`pip install rich`) for Markdown terminal formatting.')}"
    )
    print(border)

    while True:
        try:
            user_input = input(f"\n{Colors.user('You:')} ").strip()
            if not user_input:
                continue

            cmd = user_input.lower()
            if cmd in ("exit", "quit", "q"):
                print(f"\n{Colors.notice('Goodbye! Have a great day! 👋')}")
                break

            if cmd in ("/clear", "/reset"):
                assistant.clear_history()
                print(f"\n{Colors.notice('[Conversation memory cleared.]')}")
                continue

            print(f"\n{Colors.assistant('Assistant:')} {Colors.GREEN}", end="", flush=True)
            for chunk in assistant.stream_chat(user_input):
                sys.stdout.write(chunk)
                sys.stdout.flush()
            print(Colors.RESET)

        except (KeyboardInterrupt, EOFError):
            print(f"\n\n{Colors.notice('Session interrupted. Goodbye!')}")
            break
        except Exception as e:
            print(f"\n{Colors.error(str(e))}")


def run_interactive_chat() -> None:
    """Run the main interactive chat session."""
    try:
        assistant = GroqAssistant()
    except Exception as e:
        if RICH_AVAILABLE:
            console.print(f"\n[bold orange3][Error][/bold orange3] [bold red]Initialization failed: {e}[/bold red]")
        else:
            print(f"\n{Colors.error(f'Initialization failed: {e}')}")
        return

    if RICH_AVAILABLE:
        run_rich_chat(assistant)
    else:
        run_ansi_chat(assistant)
