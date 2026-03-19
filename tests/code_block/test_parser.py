from mini_rlm.code_block import find_final_answer


def test_find_final_answer_decodes_quoted_string_literal() -> None:
    # テストしたいふるまい: find_final_answer が quoted string の FINAL(...) を読むとき、
    # 外側の引用符とエスケープを解釈して自然な最終回答へ戻すこと

    # give: 改行を含む quoted string literal の FINAL(...) がある
    text = 'FINAL("失礼しました。\\n")'
    # when: final answer を抽出する
    answer = find_final_answer(text)
    # then: 外側の引用符は消え、改行は実際の改行として復元される
    assert answer == ("失礼しました。\n")


def test_find_final_answer_keeps_non_string_literal_as_is() -> None:
    # テストしたいふるまい: find_final_answer が quoted string ではない FINAL(...) を読むとき、
    # 既存どおり内容をそのまま返すこと

    # give: 数値 literal を含む FINAL(...) がある
    text = "FINAL(238)"
    # when: final answer を抽出する
    answer = find_final_answer(text)
    # then: 非文字列 literal はそのまま返る
    assert answer == "238"
