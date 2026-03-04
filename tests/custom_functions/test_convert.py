from mini_rlm.custom_functions.convert import convert_function_to_string
from mini_rlm.custom_functions.data_model import Function

# =============================================================================
# convert_function_to_string
# =============================================================================


def test_basic_function_with_args_and_return_type():
    # given: 名前・引数・戻り値・説明を持つ Function オブジェクト
    func = Function(
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
    func = Function(
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
    func = Function(
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
    func = Function(
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
    func = Function(
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
    func = Function(
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
