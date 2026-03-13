from mini_rlm.custom_functions.convert import (
    convert_function_to_string,
    filter_function_collection_for_runtime,
)
from mini_rlm.custom_functions.data_model import FunctionBase, FunctionCollection
from mini_rlm.recursive_query.data_model import RecursiveQueryRuntime

# =============================================================================
# convert_function_to_string
# =============================================================================


def test_basic_function_with_args_and_return_type():
    # given: 名前・引数・戻り値・説明を持つ Function オブジェクト
    func = FunctionBase(
        name="add",
        description="Add two integers.",
        arguments=[
            {"name": "x", "description": "", "type": int},
            {"name": "y", "description": "", "type": int},
        ],
        return_type=int,
    )
    # when: convert_function_to_string を呼ぶ
    result = convert_function_to_string(func)
    # then: シグネチャ・戻り値型・説明が正しく含まれる
    assert "add(x: int, y: int) -> int" in result
    assert "Add two integers." in result


def test_no_arguments():
    # given: 引数を持たない Function オブジェクト
    func = FunctionBase(
        name="get_value",
        description="Return a fixed value.",
        arguments=[],
        return_type=str,
    )
    # when: convert_function_to_string を呼ぶ
    result = convert_function_to_string(func)
    # then: 引数部分が空の括弧になる
    assert "get_value() -> str" in result


def test_no_return_type():
    # given: return_type が None の Function オブジェクト
    func = FunctionBase(
        name="print_value",
        description="Print something.",
        arguments=[{"name": "val", "description": "", "type": str}],
        return_type=None,
    )
    # when: convert_function_to_string を呼ぶ
    result = convert_function_to_string(func)
    # then: 戻り値型が "None" として出力される
    assert "-> None" in result


def test_multiple_arguments_are_comma_separated():
    # given: 3つの引数を持つ Function オブジェクト
    func = FunctionBase(
        name="func",
        description="A function.",
        arguments=[
            {"name": "a", "description": "", "type": int},
            {"name": "b", "description": "", "type": float},
            {"name": "c", "description": "", "type": str},
        ],
        return_type=bool,
    )
    # when: convert_function_to_string を呼ぶ
    result = convert_function_to_string(func)
    # then: 引数がカンマ区切りで並ぶ
    assert "func(a: int, b: float, c: str) -> bool" in result


def test_multiline_description_each_line_indented():
    # given: 複数行の説明を持つ Function オブジェクト
    func = FunctionBase(
        name="compute",
        description="First line.\nSecond line.\nThird line.",
        arguments=[],
        return_type=int,
    )
    # when: convert_function_to_string を呼ぶ
    result = convert_function_to_string(func)
    # then: 各行が4スペースでインデントされている
    assert "    First line." in result
    assert "    Second line." in result
    assert "    Third line." in result


def test_description_with_leading_whitespace_is_dedented():
    # given: 各行に共通の先頭インデントを持つ説明
    description = """\
        Dedented first line.
        Dedented second line.
    """
    func = FunctionBase(
        name="dedented_func",
        description=description,
        arguments=[],
        return_type=None,
    )
    # when: convert_function_to_string を呼ぶ
    result = convert_function_to_string(func)
    # then: 共通インデントが除去された上で4スペース付きで出力される
    assert "    Dedented first line." in result
    assert "    Dedented second line." in result


def test_argument_descriptions_are_included_in_args_section():
    # given: 引数 description を持つ Function オブジェクト
    func = FunctionBase(
        name="search",
        description="Search documents.",
        arguments=[
            {"name": "query", "description": "検索語", "type": str},
            {"name": "limit", "description": "取得件数", "type": int},
        ],
        return_type=list,
    )
    # when: convert_function_to_string を呼ぶ
    result = convert_function_to_string(func)
    # then: docstring の Args セクションに引数 description が含まれる
    assert "Args:" in result
    assert "query (str): 検索語" in result
    assert "limit (int): 取得件数" in result


def test_filter_function_collection_for_runtime_excludes_rlm_query_at_depth_zero():
    # given: rlm_query を含む FunctionCollection と remaining_depth 0 の runtime がある
    collection = FunctionCollection(
        functions=[
            FunctionBase(name="llm_query", description="", arguments=[]),
            FunctionBase(name="rlm_query", description="", arguments=[]),
            FunctionBase(name="llm_image_query", description="", arguments=[]),
        ]
    )
    runtime = RecursiveQueryRuntime(remaining_depth=0)
    # when: runtime に応じて function collection をフィルタする
    filtered = filter_function_collection_for_runtime(collection, runtime)
    # then: child では rlm_query だけが除外される
    assert [function.name for function in filtered.functions] == [
        "llm_query",
        "llm_image_query",
    ]
    assert [function.name for function in collection.functions] == [
        "llm_query",
        "rlm_query",
        "llm_image_query",
    ]


def test_filter_function_collection_for_runtime_keeps_rlm_query_when_depth_remains():
    # given: rlm_query を含む FunctionCollection と remaining_depth 1 の runtime がある
    collection = FunctionCollection(
        functions=[
            FunctionBase(name="llm_query", description="", arguments=[]),
            FunctionBase(name="rlm_query", description="", arguments=[]),
        ]
    )
    runtime = RecursiveQueryRuntime(remaining_depth=1)
    # when: runtime に応じて function collection をフィルタする
    filtered = filter_function_collection_for_runtime(collection, runtime)
    # then: rlm_query は残る
    assert [function.name for function in filtered.functions] == [
        "llm_query",
        "rlm_query",
    ]
