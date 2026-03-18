import argparse
import sys

from mini_rlm.cli.convert import (
    API_KEY_ENV,
    ENDPOINT_ENV,
    MODEL_ENV,
    require_env,
    resolve_file_paths,
)
from mini_rlm.cli.data_model import ChatCLIConfig, RunCLIConfig, RunMode
from mini_rlm.cli.run import run_chat_command, run_run_command


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="mini_rlm user-facing agent CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    chat_parser = subparsers.add_parser("chat", help="Start a chat session.")
    chat_parser.add_argument(
        "--endpoint", help=f"LLM endpoint. Defaults to ${ENDPOINT_ENV}."
    )
    chat_parser.add_argument(
        "--api-key", help=f"LLM API key. Defaults to ${API_KEY_ENV}."
    )
    chat_parser.add_argument("--model", help=f"LLM model. Defaults to ${MODEL_ENV}.")
    chat_parser.add_argument(
        "--file",
        action="append",
        default=[],
        help="Attach a file to the chat session. Repeatable.",
    )
    chat_parser.add_argument(
        "--prompt",
        help="Optional initial prompt before entering interactive mode.",
    )
    chat_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print run summaries for agent executions.",
    )

    run_parser = subparsers.add_parser("run", help="Run a single agent execution.")
    run_parser.add_argument(
        "--endpoint", help=f"LLM endpoint. Defaults to ${ENDPOINT_ENV}."
    )
    run_parser.add_argument(
        "--api-key", help=f"LLM API key. Defaults to ${API_KEY_ENV}."
    )
    run_parser.add_argument("--model", help=f"LLM model. Defaults to ${MODEL_ENV}.")
    run_parser.add_argument(
        "--file",
        action="append",
        default=[],
        help="Attach a file to the run. Repeatable.",
    )
    run_parser.add_argument(
        "--prompt",
        help="Prompt for the single agent execution. If omitted, stdin is used.",
    )
    run_parser.add_argument(
        "--mode",
        choices=[mode.value for mode in RunMode],
        default=RunMode.AUTO.value,
        help="Function/tool preset for the run.",
    )
    run_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Reserved for future detailed output.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    endpoint_url = require_env(ENDPOINT_ENV, args.endpoint)
    api_key = require_env(API_KEY_ENV, args.api_key)
    model = require_env(MODEL_ENV, args.model)
    files = resolve_file_paths(args.file)

    if args.command == "chat":
        return run_chat_command(
            ChatCLIConfig(
                endpoint_url=endpoint_url,
                api_key=api_key,
                model=model,
                files=files,
                initial_prompt=args.prompt,
                verbose=args.verbose,
            )
        )

    prompt = args.prompt
    if prompt is None and not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()
    if prompt is None or prompt == "":
        raise RuntimeError("run requires --prompt or non-empty stdin.")

    return run_run_command(
        RunCLIConfig(
            endpoint_url=endpoint_url,
            api_key=api_key,
            model=model,
            files=files,
            prompt=prompt,
            mode=RunMode(args.mode),
            verbose=args.verbose,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
