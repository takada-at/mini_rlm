from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from mini_rlm.chat_session import (
    ChatSessionState,
    ChatTurnResult,
    RunSummary,
    add_attachment,
    execute_chat_turn,
    reset_chat_session,
)
from mini_rlm.cli.convert import parse_chat_input
from mini_rlm.cli.data_model import ChatCLIInputType


def _render_welcome(console: Console) -> None:
    console.print(
        Panel.fit(
            "Type /help for commands.",
            title="mini-rlm chat",
            border_style="cyan",
        )
    )


def _render_help(console: Console) -> None:
    table = Table(box=box.SIMPLE, header_style="bold cyan")
    table.add_column("Command", style="bold")
    table.add_column("Description")
    table.add_row("/help", "Show available commands.")
    table.add_row("/files", "List attached files.")
    table.add_row("/add <path>", "Attach a file to the current chat session.")
    table.add_row("/reset", "Clear chat turns while keeping attachments.")
    table.add_row("/run <prompt>", "Force an agent run for the prompt.")
    table.add_row("/exit", "Exit the chat session.")
    console.print(table)


def _render_attachments(console: Console, state: ChatSessionState) -> None:
    if not state.attachments:
        console.print("[dim](no attached files)[/dim]")
        return
    table = Table(box=box.SIMPLE, header_style="bold cyan")
    table.add_column("Name", style="bold")
    table.add_column("Kind", style="cyan")
    table.add_column("Path", style="dim")
    for attachment in state.attachments:
        table.add_row(
            attachment.name,
            attachment.kind.value,
            str(attachment.path),
        )
    console.print(table)


def _render_run_summary(console: Console, summary: RunSummary) -> None:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold")
    table.add_column()
    table.add_row("termination", summary.termination_reason)
    table.add_row("iterations", str(summary.total_iterations))
    table.add_row("tokens", str(summary.total_tokens))
    table.add_row("elapsed", f"{summary.total_time_seconds:.2f}s")
    console.print(
        Panel.fit(
            table,
            title="Run Summary",
            border_style="magenta",
        )
    )


def _render_assistant_message(console: Console, message: str) -> None:
    renderable = Markdown(message) if message.strip() != "" else Text("")
    console.print(
        Panel(
            renderable,
            title="Assistant",
            border_style="green",
            expand=True,
        )
    )


def _render_error(console: Console, message: str) -> None:
    console.print(
        Panel(
            message,
            title="Error",
            border_style="red",
            expand=True,
        )
    )


def _render_notice(console: Console, message: str) -> None:
    console.print(f"[cyan]{message}[/cyan]")


def execute_rich_chat_turn(
    console: Console,
    state: ChatSessionState,
    user_text: str,
    force_run: bool = False,
    verbose: bool = False,
) -> ChatTurnResult:
    with console.status("[bold cyan]Thinking...[/bold cyan]", spinner="dots") as status:
        turn_result = execute_chat_turn(
            state,
            user_text,
            force_run=force_run,
            on_run_start=status.update,
        )
    if verbose and turn_result.turn.run_summary is not None:
        _render_run_summary(console, turn_result.turn.run_summary)
    _render_assistant_message(console, turn_result.turn.assistant_text)
    return turn_result


def run_rich_chat_session(
    state: ChatSessionState,
    verbose: bool = False,
    console: Console | None = None,
) -> ChatSessionState:
    resolved_console = console or Console()
    current_state = state
    _render_welcome(resolved_console)

    while True:
        try:
            user_text = Prompt.ask(
                "[bold cyan]mini-rlm[/bold cyan]", console=resolved_console
            )
        except EOFError:
            resolved_console.print()
            return current_state
        except KeyboardInterrupt:
            resolved_console.print()
            continue

        command = parse_chat_input(user_text)
        if command.type == ChatCLIInputType.EMPTY:
            continue
        if command.type == ChatCLIInputType.EXIT:
            return current_state
        if command.type == ChatCLIInputType.HELP:
            _render_help(resolved_console)
            continue
        if command.type == ChatCLIInputType.FILES:
            _render_attachments(resolved_console, current_state)
            continue
        if command.type == ChatCLIInputType.INVALID:
            _render_error(
                resolved_console,
                command.error_message or "Invalid command.",
            )
            continue
        if command.type == ChatCLIInputType.ADD_FILE:
            if command.file_path is None:
                _render_error(resolved_console, "Usage: /add <path>")
                continue
            try:
                current_state = add_attachment(current_state, command.file_path)
            except Exception as error:
                _render_error(resolved_console, f"Failed to attach file: {error}")
                continue
            _render_notice(resolved_console, f"Attached: {command.file_path.name}")
            continue
        if command.type == ChatCLIInputType.RESET:
            current_state = reset_chat_session(current_state)
            _render_notice(resolved_console, "Chat session reset.")
            continue
        if command.type != ChatCLIInputType.SEND_MESSAGE or command.message is None:
            _render_error(resolved_console, "A prompt is required.")
            continue
        try:
            turn_result = execute_rich_chat_turn(
                resolved_console,
                current_state,
                command.message,
                force_run=command.force_run,
                verbose=verbose,
            )
        except Exception as error:
            _render_error(resolved_console, f"Failed to execute chat turn: {error}")
            continue
        current_state = turn_result.state
