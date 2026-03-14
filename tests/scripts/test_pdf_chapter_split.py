import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def load_pdf_chapter_split_module() -> ModuleType:
    script_path = Path(__file__).resolve().parents[2] / "scripts/pdf_chapter_split.py"
    spec = importlib.util.spec_from_file_location("pdf_chapter_split", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_page_number_accepts_csv_format() -> None:
    module = load_pdf_chapter_split_module()
    # given: LLM が指定どおり csv 形式でページ範囲を返す
    final_answer = "12,20"
    # when: ページ範囲を parse する
    start_page, end_page = module.parse_page_number(final_answer)
    # then: start/end のページ番号をそのまま取得できる
    assert (start_page, end_page) == (12, 20)


def test_parse_page_number_accepts_labeled_format() -> None:
    module = load_pdf_chapter_split_module()
    # given: LLM が start/end ラベル付きでページ範囲を返す
    final_answer = "start: 12, end: 20"
    # when: ページ範囲を parse する
    start_page, end_page = module.parse_page_number(final_answer)
    # then: ラベル付きの応答でもページ範囲を取得できる
    assert (start_page, end_page) == (12, 20)


def test_parse_page_number_accepts_explanatory_text() -> None:
    module = load_pdf_chapter_split_module()
    # given: LLM が説明文を含めて開始ページと終了ページを返す
    final_answer = "Chapter 3 starts at page 12 and ends at page 20."
    # when: ページ範囲を parse する
    start_page, end_page = module.parse_page_number(final_answer)
    # then: 説明文込みでも start/end を抽出できる
    assert (start_page, end_page) == (12, 20)


def test_parse_page_number_raises_for_unparseable_response() -> None:
    module = load_pdf_chapter_split_module()
    # given: LLM 応答にページ範囲が含まれていない
    final_answer = "I could not determine the page range."
    # when: ページ範囲を parse する
    # then: parse 不能として ValueError になる
    with pytest.raises(ValueError):
        module.parse_page_number(final_answer)
