"""Interactive Terminal Chat for Tara Assistant using Rich."""

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.live import Live
from rich.table import Table

from src.services.assistant import GroqAssistant
from src.config import configs

console = Console()


def show_welcome(assistant: GroqAssistant) -> None:
    """Displays a clean welcome banner in the terminal."""
    if assistant.has_active_session:
        session_info = (
            f"[dim]Active Session:[/dim] [bold white]{assistant.thread_name}[/bold white] "
            f"([dim cyan]{assistant.thread_id[:8]}...[/dim cyan])"
        )
    else:
        session_info = (
            f"[dim]Status:[/dim] [italic yellow]System Ready (Type a message to start a new chat)[/italic yellow]"
        )

    info_text = (
        f"[bold cyan]🤖 Tara - Groq & LangChain Assistant[/bold cyan]\n"
        f"[dim]Model:[/dim] [yellow]{assistant.model_name}[/yellow] | "
        f"[dim]Temperature:[/dim] [yellow]{configs.GROQ_TEMPERATURE}[/yellow]\n"
        f"{session_info}\n\n"
        f"[dim]Commands:[/dim] "
        f"[cyan]/history[/cyan] view chats | [cyan]/switch <id>[/cyan] resume chat | "
        f"[cyan]/clear[/cyan] reset | [cyan]/help[/cyan] commands | [cyan]exit[/cyan] quit"
    )
    console.print(Panel(info_text, border_style="cyan", padding=(1, 2)))


def show_help() -> None:
    """Displays available commands."""
    help_text = (
        "[bold cyan]Available Commands:[/bold cyan]\n"
        "  [cyan]/history[/cyan]         : List all past chat sessions with Thread IDs and Titles\n"
        "  [cyan]/switch <id>[/cyan]     : Switch/resume a past chat session by Thread ID\n"
        "  [cyan]/clear, /reset[/cyan]  : Reset session to standby (new chat begins on next prompt)\n"
        "  [cyan]/status[/cyan]          : Show current session name, ID, and memory stats\n"
        "  [cyan]/help[/cyan]            : Show this help message\n"
        "  [cyan]exit, quit, q[/cyan]    : Exit Tara"
    )
    console.print(Panel(help_text, border_style="cyan"))


def show_history(assistant: GroqAssistant) -> None:
    """Displays all stored threads in a formatted Rich table."""
    threads = assistant.list_threads()
    if not threads:
        console.print("\n[yellow]No chat history found yet. Start chatting to create one![/yellow]")
        return

    table = Table(title="💬 All Chat Sessions", border_style="cyan", show_lines=True)
    table.add_column("Status", style="bold green", justify="center", width=10)
    table.add_column("Session Title", style="bold white", min_width=25)
    table.add_column("Thread ID", style="dim cyan", min_width=38)
    table.add_column("Created At", style="dim", min_width=20)

    for t in threads:
        is_active = assistant.has_active_session and (t["thread_id"] == assistant.thread_id)
        status_label = "● ACTIVE" if is_active else ""
        table.add_row(
            status_label,
            t.get("thread_name", "Untitled"),
            t["thread_id"],
            str(t.get("created_at", "")),
        )

    console.print("\n")
    console.print(table)
    console.print("[dim]💡 Tip: Use [cyan]/switch <thread_id>[/cyan] to switch to any past session.[/dim]\n")


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
                console.print(
                    "\n[yellow]✔ Conversation reset to standby. A new chat will start on your next message.[/yellow]"
                )
                continue

            if cmd in ("/history", "/chats", "/sessions"):
                show_history(assistant)
                continue

            if cmd.startswith("/switch"):
                parts = user_input.split(maxsplit=1)
                if len(parts) < 2 or not parts[1].strip():
                    console.print("[yellow]Usage: /switch <thread_id>[/yellow]")
                    continue
                target_id = parts[1].strip()
                assistant.switch_thread(target_id)
                console.print(
                    f"\n[green]✔ Switched to session:[/green] [bold cyan]{assistant.thread_name}[/bold cyan] "
                    f"([dim]{assistant.thread_id}[/dim])"
                )
                console.print(f"[dim]Messages currently in memory: {assistant.message_count}[/dim]")
                continue

            if cmd == "/status":
                if assistant.has_active_session:
                    console.print(
                        f"\n[cyan]Session Name:[/cyan] [bold yellow]{assistant.thread_name}[/bold yellow]\n"
                        f"[cyan]Thread ID:[/cyan] [dim]{assistant.thread_id}[/dim]\n"
                        f"[cyan]Model:[/cyan] [yellow]{assistant.model_name}[/yellow] | "
                        f"[cyan]Messages in memory:[/cyan] [yellow]{assistant.message_count}[/yellow]"
                    )
                else:
                    console.print(
                        f"\n[yellow]Status:[/yellow] Standby (No active session).\n"
                        f"[cyan]Model:[/cyan] [yellow]{assistant.model_name}[/yellow]\n"
                        f"[dim]Type any message to start a new chat, or use /switch <id> to resume an existing one.[/dim]"
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
