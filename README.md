# mini_rlm

An implementation of the Recursive Language Model (RLM) capable of handling both PDF documents and images.
Unlike conventional text-focused RLMs, this implementation is designed to work with OpenAI-compatible multimodal APIs, aiming to solve problems by combining PDF and image processing within a REPL environment while dividing tasks into manageable subproblems.

The current status is under development. The public API is not yet stable.

## Features

- Directly pass images to the LLM for querying
- Process PDF pages by either converting them to images or extracting text
- Maintain persistent REPL sessions with temporary directories for continuous execution of multi-turn code
- Extract subproblems through recursive execution of child REPLs using `rlm_query()`
- Separate state transitions and side effects using the reducer + executor pattern
## System Overview

The main packages are as follows:

- `mini_rlm/llm`: Handles requests to OpenAI-compatible APIs, message construction, and token usage extraction
- `mini_rlm/repl`: Manages Python REPL state and code execution
- `mini_rlm/repl_session`: Controls session management for repeated LLM calls and REPL execution
- `mini_rlm/recursive_query`: Implements recursive execution using child REPLs
- `mini_rlm/pdf`: Provides functionality for obtaining PDF page counts, converting images, and extracting text
- `mini_rlm/image`: Reads image files and converts them into `ImageData` format
- `mini_rlm/custom_functions`: Collection of functions to be injected into the REPL

The REPL can be provided with function collections tailored to specific use cases.

- `minimal_function_collection()`
- `image_function_collection()`
- `pdf_function_collection()`

Other components:

- `scripts`: Collection of tools for external distribution.
    - `scripts/pdf_chapter_split.py`: Extracts chapters from PDF files.
- `dev_scripts`: Development scripts.
- `manual_tests`: Comprehensive tests for manual execution

## Design Philosophy

This repository organizes packages by domain, with data structures defined in each domain's `data_model.py` file.

  - Only use classes for non-data models, implementing functions in a functional programming style
  - Encapsulate complex state transitions within reducers
  - Concentrate side effects in executors while maintaining reducers as pure functions
  - Limit testing primarily to "complex logic sections without side effects"

Both `repl_session` and `llm` are implemented using the reducer pattern.

## Setup

Prerequisites:

- Python 3.12+
- `uv`

Install the dependencies:
  ```bash
  uv sync --dev
  ```

## Environment Variables

The manual execution scripts use these environment variables:

```bash
export MINI_RLM_LLM_ENDPOINT="https://your-host/v1/chat/completions"
export MINI_RLM_LLM_API_KEY="..."
export MINI_RLM_LLM_MODEL="gpt-4.1-mini"
```

For PDF REPL sessions, you may also use configuration settings for submodels as needed:

```bash
export MINI_RLM_LLM_SUB_ENDPOINT="https://your-host/v1/chat/completions"
export MINI_RLM_LLM_SUB_API_KEY="..."
export MINI_RLM_LLM_SUB_MODEL="gpt-4.1-mini"
```

`MINI_RLM_LLM_MODEL` is used for the outer chat / RLM session. `MINI_RLM_LLM_SUB_MODEL` is used for subqueries executed through REPL helpers and recursive calls. If submodel settings are omitted, the main model settings are reused.

This implementation assumes an OpenAI-compatible endpoint that accepts `model` and `messages` in a JSON body and processes them via `POST`.

## Quick Start

### Start the user-facing chat CLI

```bash
export API_ENDPOINT="https://your-host/v1/chat/completions"
export API_KEY="..."
uv run mini-rlm chat --file /path/to/book.pdf
```

Inside the chat session, you can use `/help`, `/files`, `/add <path>`, `/run <prompt>`, and `/exit`.
`mini-rlm chat` uses `openai/gpt-5.3-codex` as the main model and `qwen/qwen3.5-35b-a3b` as the submodel by default. You can override them with `--model` and `--sub_model`.

### Run a single agent execution

```bash
export API_ENDPOINT="https://your-host/v1/chat/completions"
export API_KEY="..."
uv run mini-rlm run --file /path/to/book.pdf "Find the page where Chapter 2 begins."
```

### Extract chapters from a PDF

```bash
export API_ENDPOINT="https://your-host/v1/chat/completions"
export API_KEY="..."
uv run python scripts/pdf_chapter_split.py <pdf_path> <chapter number (1-based)>
```

### Run an image-enabled REPL session

```bash
uv run python manual_tests/repl_describe_image.py
```

## Using as a Library

Minimal example.

```python
from pathlib import Path

from mini_rlm import (
    ReplExecutionRequest,
    ReplSetupRequest,
    create_request_context,
    execute_repl_session,
    pdf_function_collection,
)

request_context = create_request_context(
    endpoint_url="https://your-host/v1/chat/completions",
    api_key="...",
    model="gpt-4.1-mini",
)

result = execute_repl_session(
    ReplExecutionRequest(
        prompt="Find the page where Chapter 2 begins.",
        setup=ReplSetupRequest(
            request_context=request_context,
            file_paths=[Path("/path/to/book.pdf")],
            context_payload={"pdf_path": "book.pdf"},
            functions=pdf_function_collection(),
        ),
        session_request_context=request_context,
    )
)

print(result.termination_reason)
print(result.final_answer)
```

`execute_repl_session()` handles the entire process from REPL setup to session execution and cleanup.

## Development Commands

```bash
make format
make lint
make typecheck
make test
```

## Testing Policy

- Use `pytest` for testing
- Place tests by domain in `tests/<domain>/`
- Use behavior-based testing with `give / when / then` comments
- Avoid excessive unit testing for code with simple side effects or straightforward glue code
## Directory Structure Example

```text
mini_rlm/
  image/
    data_model.py
    convert.py
  pdf/
    data_model.py
    convert.py
  repl/
  repl_session/
  recursive_query/
  llm/
tests/
  image/
  pdf/
  repl/
  repl_session/
```
