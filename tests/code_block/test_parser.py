from mini_rlm.code_block import find_final_answer
from mini_rlm.repl import cleanup, create_repl


def test_find_final_answer_uses_repl_to_decode_quoted_string_literal() -> None:
    # テストしたいふるまい: find_final_answer が REPL state 付きで quoted string の FINAL(...) を読むとき、
    # 実際に FINAL(...) を実行した結果を最終回答として返すこと

    # give: 改行を含む quoted string literal の FINAL(...) がある
    text = 'FINAL("失礼しました。\\n対応としては、PDFの物理ページで 1ページ目=238、29ページ目=266 です。")'
    repl_state = create_repl()
    # when: final answer を抽出する
    try:
        answer = find_final_answer(text, repl_state=repl_state)
    finally:
        cleanup(repl_state)
    # then: 外側の引用符は消え、改行は実際の改行として復元される
    assert answer == (
        "失礼しました。\n"
        "対応としては、PDFの物理ページで 1ページ目=238、29ページ目=266 です。"
    )


def test_find_final_answer_keeps_raw_literal_without_repl_state() -> None:
    # テストしたいふるまい: find_final_answer が REPL state なしで quoted string の FINAL(...) を読むとき、
    # Python の escape 展開をせずに生の literal を保つこと

    # give: Windows path 風の quoted string literal を含む FINAL(...) がある
    text = 'FINAL("C:\\temp\\new")'
    # when: final answer を抽出する
    answer = find_final_answer(text)
    # then: Python 文字列として評価せず、生の literal を返す
    assert answer == '"C:\\temp\\new"'
