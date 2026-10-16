"""Interactive CLI presentation layer for Tara Assistant with real-time Voice STT/TTS support."""

import re
import sys
import threading
import time
from typing import Optional
from src.services.assistant import GroqAssistant
from src.utils.colors import Colors
from src.config.config import configs
from src.voice import VoiceEngine, is_voice_available

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


def show_welcome(assistant: GroqAssistant, voice_mode: bool = False) -> None:
    """Display a clean welcome banner in the terminal."""
    if assistant.has_active_session:
        session_info = (
            f"[dim]Active Session:[/dim] [bold white]{assistant.thread_name}[/bold white] "
            f"([dim cyan]{assistant.thread_id[:8]}...[/dim cyan])"
        )
    else:
        session_info = (
            f"[dim]Status:[/dim] [italic yellow]System Ready (Type a message or speak to start a chat)[/italic yellow]"
        )

    voice_status = (
        "[bold green]ON (Tara Voice: Qwen Cloning - Elizabeth Olsen)[/bold green]"
        if voice_mode
        else "[dim]OFF (type [cyan]/voice[/cyan] to toggle)[/dim]"
    )

    info_text = (
        f"[bold cyan]🤖 Tara - Groq & LangChain Assistant[/bold cyan]\n"
        f"[dim]Model:[/dim] [yellow]{assistant.model_name}[/yellow] | "
        f"[dim]Temperature:[/dim] [yellow]{configs.GROQ_TEMPERATURE}[/yellow] | "
        f"[dim]Voice Mode:[/dim] {voice_status}\n"
        f"{session_info}\n\n"
        f"[dim]Commands:[/dim] "
        f"[cyan]/voice[/cyan] toggle audio | [cyan]/history[/cyan] view chats | [cyan]/switch <id>[/cyan] resume | "
        f"[cyan]/clear[/cyan] reset | [cyan]/help[/cyan] commands | [cyan]exit[/cyan] quit"
    )
    console.print(Panel(info_text, border_style="cyan", padding=(1, 2)))


def show_help() -> None:
    """Display available commands."""
    help_text = (
        "[bold cyan]Available Commands:[/bold cyan]\n"
        "  [cyan]/voice[/cyan]            : Toggle voice mode ON/OFF mid-session (speech STT + TTS)\n"
        "  [cyan]/text[/cyan]             : Explicitly switch to text-only mode\n"
        "  [cyan]/history[/cyan]         : List all past chat sessions with Thread IDs and Titles\n"
        "  [cyan]/switch <id>[/cyan]     : Switch/resume a past chat session by Thread ID\n"
        "  [cyan]/clear, /reset[/cyan]  : Reset session to standby (new chat begins on next prompt)\n"
        "  [cyan]/status[/cyan]          : Show current session name, ID, and memory stats\n"
        "  [cyan]/help[/cyan]            : Show this help message\n"
        "  [cyan]exit, quit, q[/cyan]    : Exit Tara"
    )
    console.print(Panel(help_text, border_style="cyan"))


def show_history(assistant: GroqAssistant) -> None:
    """Display all stored threads in a formatted Rich table."""
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


def run_rich_chat(assistant: GroqAssistant, initial_voice_mode: bool = False) -> None:
    """Interactive loop utilizing Rich Markdown, Live streaming, and seamless voice integration."""
    voice_mode = initial_voice_mode
    voice_engine: Optional[VoiceEngine] = None

    # Check and initialize voice engine if requested on startup
    if voice_mode:
        available, error_msg = is_voice_available()
        if not available:
            console.print(f"\n[bold yellow]⚠️ Voice Mode Warning:[/bold yellow] {error_msg}")
            console.print("[dim]Defaulting to standard text chat mode.[/dim]\n")
            voice_mode = False
        else:
            console.print("[dim cyan]⏳ Initializing Voice Engine (Faster-Whisper & Qwen-TTS)...[/dim cyan]")
            try:
                voice_engine = VoiceEngine()
                voice_engine.warmup()
                console.print("[bold green]✔ Voice Engine fully operational (Qwen Voice Cloning Active on RTX 4060).[/bold green]\n")
                voice_engine.speak_sentence("Hello! Tara voice systems are online and ready.")
                voice_engine.wait_until_done()
            except Exception as e:
                console.print(f"[bold red]Failed to initialize voice engine: {e}[/bold red]")
                voice_mode = False

    show_welcome(assistant, voice_mode=voice_mode)
    sentence_splitter = re.compile(r'([.?!:;]+(?:\s+|$))')

    pending_input: Optional[str] = None
    interrupted_context = {
        "original_query": "",
        "full_response": "",
        "was_interrupted": False,
    }
    RESUME_KEYWORDS = (
        "resume", "continue", "go on", "what were you saying",
        "carry on", "keep going", "pick up where you left off",
        "what were you talking about", "what were you explaining",
        "continue previous", "what was that", "back to what you were saying"
    )

    while True:
        try:
            user_input = ""

            # 1. Capture User Input (From previous barge-in, or fresh Voice/Keyboard)
            if pending_input:
                user_input = pending_input
                pending_input = None
            elif voice_mode and voice_engine:
                console.print("\n[dim cyan]🎙️ Tara is listening... (speak now, or press Enter to type)[/dim cyan]")
                spoken_text = voice_engine.listen(silence_duration=1.2)
                if spoken_text:
                    user_input = spoken_text.strip()
                    console.print(f"[bold blue]You (Voice):[/bold blue] [bold white]{user_input}[/bold white]")
                else:
                    # Fallback to keyboard input if no voice was detected
                    user_input = console.input("[bold blue]You (or press Enter to speak again):[/bold blue] ").strip()
                    if not user_input:
                        continue
            else:
                user_input = console.input("\n[bold blue]You:[/bold blue] ").strip()
                if not user_input:
                    continue

            # 2. Handle voice toggle command spoken aloud
            cmd = user_input.lower()
            if voice_mode and cmd in ("switch to text", "text mode", "disable voice", "turn off voice"):
                voice_mode = False
                if voice_engine:
                    voice_engine.stop_speaking()
                    voice_engine.stop_barge_in_monitor()
                console.print("\n[yellow]✔ Voice mode disabled. Switched to text-only mode.[/yellow]")
                continue

            # 3. Handle CLI Commands
            if cmd in ("exit", "quit", "q"):
                if voice_mode and voice_engine:
                    voice_engine.stop_barge_in_monitor()
                    voice_engine.speak_sentence("Powering down systems. Goodbye!")
                    voice_engine.wait_until_done()
                console.print("\n[bold yellow]Goodbye! Have a great day! 👋[/bold yellow]")
                break

            if cmd in ("/voice", "/audio"):
                if voice_mode:
                    voice_mode = False
                    if voice_engine:
                        voice_engine.stop_speaking()
                        voice_engine.stop_barge_in_monitor()
                    console.print("\n[yellow]✔ Voice mode disabled. Switched to text-only mode.[/yellow]")
                else:
                    available, error_msg = is_voice_available()
                    if not available:
                        console.print(f"\n[bold red]⚠️ Cannot activate voice:[/bold red] {error_msg}")
                        continue
                    if voice_engine is None:
                        console.print("\n[dim cyan]⏳ Loading Voice Engine (Faster-Whisper & Qwen-TTS)...[/dim cyan]")
                        voice_engine = VoiceEngine()
                        voice_engine.warmup()
                    voice_mode = True
                    console.print("\n[bold green]✔ Voice mode activated (Qwen Voice Cloning on RTX 4060)! Speak into your microphone, or type anytime.[/bold green]")
                    voice_engine.speak_sentence("Tara voice mode active.")
                    voice_engine.wait_until_done()
                continue

            if cmd in ("/text",):
                voice_mode = False
                if voice_engine:
                    voice_engine.stop_speaking()
                    voice_engine.stop_barge_in_monitor()
                console.print("\n[yellow]✔ Voice mode disabled. Switched to text-only mode.[/yellow]")
                continue

            if cmd in ("/clear", "/reset"):
                assistant.clear_history()
                interrupted_context["was_interrupted"] = False
                if voice_mode and voice_engine:
                    voice_engine.stop_speaking()
                    voice_engine.stop_barge_in_monitor()
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
                interrupted_context["was_interrupted"] = False
                console.print(
                    f"\n[green]✔ Switched to session:[/green] [bold cyan]{assistant.thread_name}[/bold cyan] "
                    f"([dim]{assistant.thread_id}[/dim])"
                )
                console.print(f"[dim]Messages currently in memory: {assistant.message_count}[/dim]")
                continue

            if cmd == "/status":
                v_stat = "[green]Enabled[/green]" if voice_mode else "[dim]Disabled[/dim]"
                if assistant.has_active_session:
                    console.print(
                        f"\n[cyan]Session Name:[/cyan] [bold yellow]{assistant.thread_name}[/bold yellow]\n"
                        f"[cyan]Thread ID:[/cyan] [dim]{assistant.thread_id}[/dim]\n"
                        f"[cyan]Model:[/cyan] [yellow]{assistant.model_name}[/yellow] | "
                        f"[cyan]Messages in memory:[/cyan] [yellow]{assistant.message_count}[/yellow] | "
                        f"[cyan]Voice Mode:[/cyan] {v_stat}"
                    )
                else:
                    console.print(
                        f"\n[yellow]Status:[/yellow] Standby (No active session).\n"
                        f"[cyan]Model:[/cyan] [yellow]{assistant.model_name}[/yellow] | "
                        f"[cyan]Voice Mode:[/cyan] {v_stat}\n"
                        f"[dim]Type any message to start a new chat, or use /switch <id> to resume an existing one.[/dim]"
                    )
                continue

            if cmd == "/help":
                show_help()
                continue

            # 4. Check if user is asking to resume a previously interrupted explanation
            clean_input = user_input.lower().strip()
            is_resume_request = any(kw in clean_input for kw in RESUME_KEYWORDS)

            if is_resume_request and interrupted_context["was_interrupted"]:
                console.print(f"\n[bold cyan]🔄 Resuming interrupted topic:[/bold cyan] [dim]{interrupted_context['original_query']}[/dim]")
                query_to_send = (
                    f"[System Instruction: The user interrupted your previous explanation of '{interrupted_context['original_query']}' "
                    f"to ask a clarification question. They have now said: '{user_input}'. "
                    f"Seamlessly pick up and continue your explanation from where you were interrupted, without redundantly repeating what you already explained.]"
                )
                interrupted_context["was_interrupted"] = False
            else:
                query_to_send = user_input

            # 5. Stream Response: Live Markdown Rendering in Terminal + Real-Time Voice Synthesis
            console.print("\n[bold green]Tara:[/bold green]")
            full_response = ""
            text_buffer = ""
            stream_interrupted = False

            # Start active background barge-in monitor BEFORE streaming so user can interrupt at any moment!
            if voice_mode and voice_engine:
                voice_engine.start_barge_in_monitor(silence_duration=1.0)

            with Live(Markdown(""), refresh_per_second=12, console=console) as live:
                for chunk in assistant.stream_chat(query_to_send):
                    # Check if user spoke into mic to interrupt
                    if voice_mode and voice_engine and voice_engine.is_barge_in_triggered():
                        stream_interrupted = True
                        break

                    full_response += chunk
                    live.update(Markdown(full_response))

                    # Non-blocking check for keypress tap during streaming
                    try:
                        import msvcrt
                        if msvcrt.kbhit():
                            k = msvcrt.getch()
                            if k == b"\x03":
                                raise KeyboardInterrupt
                            if voice_mode and voice_engine:
                                voice_engine.stop_speaking()
                                voice_engine.stop_barge_in_monitor()
                            interrupted_context = {
                                "original_query": user_input,
                                "full_response": full_response,
                                "was_interrupted": True,
                            }
                            console.print("\n[bold yellow]⚡ Speech interrupted by keypress.[/bold yellow]")
                            stream_interrupted = True
                            break
                    except (ImportError, KeyboardInterrupt):
                        raise
                    except Exception:
                        pass

                    # If voice mode is on, synthesize speech sentence by sentence
                    if voice_mode and voice_engine:
                        # Skip tool notifications from being vocalized
                        if not (chunk.startswith("\n\n> ") or "finished processing" in chunk or "Executing tool" in chunk):
                            text_buffer += chunk
                            parts = sentence_splitter.split(text_buffer)
                            if len(parts) > 1:
                                completed_sentence = parts[0] + parts[1]
                                voice_engine.speak_sentence(completed_sentence)
                                text_buffer = "".join(parts[2:])

            # Speak any leftover text if stream was not interrupted
            if not stream_interrupted and voice_mode and voice_engine and text_buffer.strip():
                voice_engine.speak_sentence(text_buffer.strip())

            # 6. Audio Playback & Barge-In Coordination (Thread 1 waits while Thread 2 monitors)
            if voice_mode and voice_engine:
                if stream_interrupted:
                    # User interrupted mid-stream; Thread 2 is already capturing user speech
                    console.print("[dim italic cyan]⚡ (Listening to your question...)[/dim italic cyan]")
                    interrupted_text = voice_engine.wait_for_barge_in_text()
                    was_interrupted = True
                else:
                    if voice_engine.is_speaking():
                        console.print("[dim italic cyan]⚡ (Tara speaking... speak or press Enter to interrupt)[/dim italic cyan]")

                    # Thread 1 waits while Tara is playing audio, constantly checking if Thread 2 detects voice
                    was_interrupted = False
                    interrupted_text = None
                    while voice_engine.is_speaking():
                        if voice_engine.is_barge_in_triggered():
                            was_interrupted = True
                            break
                        # Non-blocking keypress check
                        try:
                            import msvcrt
                            if msvcrt.kbhit():
                                k = msvcrt.getch()
                                if k == b"\x03":
                                    raise KeyboardInterrupt
                                voice_engine.stop_speaking()
                                voice_engine.stop_barge_in_monitor()
                                was_interrupted = True
                                interrupted_text = None
                                console.print("\n[bold yellow]⚡ Speech interrupted by keypress.[/bold yellow]")
                                break
                        except (ImportError, KeyboardInterrupt):
                            raise
                        except Exception:
                            pass
                        time.sleep(0.04)

                    if was_interrupted and voice_engine.is_barge_in_triggered():
                        console.print("[dim italic cyan]⚡ (Listening to your question...)[/dim italic cyan]")
                        interrupted_text = voice_engine.wait_for_barge_in_text()
                    else:
                        # Tara finished speaking naturally without any voice interruption
                        voice_engine.stop_barge_in_monitor()

                if was_interrupted:
                    interrupted_context = {
                        "original_query": user_input,
                        "full_response": full_response,
                        "was_interrupted": True,
                    }
                    if interrupted_text:
                        console.print(f"\n[bold yellow]⚡ Interrupted by voice:[/bold yellow] [bold white]{interrupted_text}[/bold white]")
                        pending_input = interrupted_text
                    else:
                        console.print("\n[bold yellow]⚡ Speech interrupted.[/bold yellow]")

        except (KeyboardInterrupt, EOFError):
            if voice_engine:
                voice_engine.stop_speaking()
                voice_engine.stop_barge_in_monitor()
            console.print("\n\n[bold yellow]Session ended. Goodbye![/bold yellow]")
            break
        except Exception as e:
            if voice_engine:
                voice_engine.stop_speaking()
                voice_engine.stop_barge_in_monitor()
            console.print(f"\n[bold red][Error] {e}[/bold red]")

    if voice_engine:
        voice_engine.shutdown()


def run_ansi_chat(assistant: GroqAssistant) -> None:
    """Fallback interactive loop using standard ANSI colors when Rich is unavailable."""
    border = f"{Colors.CYAN}{'=' * 60}{Colors.RESET}"
    print(border)
    print(f"{Colors.BOLD}{Colors.CYAN}🤖 Tara - Groq & LangChain Assistant{Colors.RESET}")
    print(
        f"{Colors.subtle('Model:')} {Colors.notice(assistant.model_name)} | "
        f"{Colors.subtle('Temperature:')} {Colors.notice(str(configs.GROQ_TEMPERATURE))}"
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


def run_interactive_chat(voice_mode: bool = False) -> None:
    """Entry point to launch the interactive CLI chat session with optional voice mode."""
    try:
        assistant = GroqAssistant()
    except Exception as e:
        if RICH_AVAILABLE:
            console.print(f"\n[bold red]Initialization failed: {e}[/bold red]")
        else:
            print(f"\n{Colors.error(f'Initialization failed: {e}')}")
        return

    if RICH_AVAILABLE:
        run_rich_chat(assistant, initial_voice_mode=voice_mode)
    else:
        run_ansi_chat(assistant)
