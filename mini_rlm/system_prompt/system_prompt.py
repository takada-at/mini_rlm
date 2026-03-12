from pathlib import Path

from mini_rlm.custom_functions import (
    FunctionCollection,
    convert_function_collection_to_string,
)


def create_system_prompt(collection: FunctionCollection | None = None) -> str:
    prompt_path = Path(__file__).parent.parent / "prompts" / "rlm_system_prompt.txt"
    with prompt_path.open("r", encoding="utf-8") as fp:
        content = fp.read()
    tool_doc = convert_function_collection_to_string(collection) if collection else ""
    return content.format(custom_tools_section=tool_doc)
