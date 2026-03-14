import importlib.util
from pathlib import Path
from textwrap import dedent
from types import ModuleType


def load_checker_module() -> ModuleType:
    script_path = (
        Path(__file__).resolve().parents[2] / "dev_scripts/check_import_rules.py"
    )
    spec = importlib.util.spec_from_file_location("check_import_rules", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(content).lstrip(), encoding="utf-8")


def test_discover_target_packages_ignores_non_packages(tmp_path: Path) -> None:
    checker = load_checker_module()
    # given: mini_rlm 配下に package と非 package が混在している
    write_file(tmp_path / "mini_rlm/repl/__init__.py", "__all__ = []\n")
    write_file(tmp_path / "mini_rlm/llm/__init__.py", "__all__ = []\n")
    write_file(tmp_path / "mini_rlm/prompts/template.txt", "prompt")
    # when: 検出対象 package を自動 discovery する
    packages = checker.discover_target_packages(tmp_path)
    # then: __init__.py を持つ即時サブパッケージだけが対象になる
    assert packages == ["mini_rlm.llm", "mini_rlm.repl"]


def test_check_import_rules_reports_external_submodule_import(tmp_path: Path) -> None:
    checker = load_checker_module()
    # given: 外部モジュールが対象 package の下位モジュールを直接 import している
    write_file(
        tmp_path / "mini_rlm/repl/__init__.py",
        """
        from mini_rlm.repl.data_model import ReplState

        __all__ = ["ReplState"]
        """,
    )
    write_file(tmp_path / "mini_rlm/repl/data_model.py", "ReplState = object\n")
    write_file(
        tmp_path / "mini_rlm/llm/context_factory.py",
        "from mini_rlm.repl.data_model import ReplState\n",
    )
    # when: import rule を検査する
    violations = checker.check_import_rules(
        repo_root=tmp_path,
        scan_paths=[tmp_path / "mini_rlm"],
        packages=["mini_rlm.repl"],
    )
    # then: package root 経由で import すべき違反として報告される
    assert len(violations) == 1
    assert "via the package root" in violations[0][3]


def test_check_import_rules_allows_internal_submodule_import(tmp_path: Path) -> None:
    checker = load_checker_module()
    # given: 同一 package 内のモジュールが下位モジュールを参照している
    write_file(
        tmp_path / "mini_rlm/repl/__init__.py",
        """
        from mini_rlm.repl.data_model import ReplState

        __all__ = ["ReplState"]
        """,
    )
    write_file(tmp_path / "mini_rlm/repl/data_model.py", "ReplState = object\n")
    write_file(
        tmp_path / "mini_rlm/repl/repl.py",
        "from mini_rlm.repl.data_model import ReplState\n",
    )
    # when: import rule を検査する
    violations = checker.check_import_rules(
        repo_root=tmp_path,
        scan_paths=[tmp_path / "mini_rlm"],
        packages=["mini_rlm.repl"],
    )
    # then: 同一 package 内の参照は許可される
    assert violations == []


def test_check_import_rules_reports_non_exported_root_import(tmp_path: Path) -> None:
    checker = load_checker_module()
    # given: package root から未公開シンボルを import している
    write_file(
        tmp_path / "mini_rlm/repl/__init__.py",
        """
        from mini_rlm.repl.data_model import ReplState

        __all__ = ["ReplState"]
        """,
    )
    write_file(tmp_path / "mini_rlm/repl/data_model.py", "ReplState = object\n")
    write_file(
        tmp_path / "mini_rlm/llm/context_factory.py",
        "from mini_rlm.repl import load_context\n",
    )
    # when: import rule を検査する
    violations = checker.check_import_rules(
        repo_root=tmp_path,
        scan_paths=[tmp_path / "mini_rlm"],
        packages=["mini_rlm.repl"],
    )
    # then: __init__.py 未公開として報告される
    assert len(violations) == 1
    assert "is not exported by mini_rlm.repl.__init__.py" in violations[0][3]


def test_check_import_rules_reports_non_top_level_import(tmp_path: Path) -> None:
    checker = load_checker_module()
    # given: 関数内 import を持つ Python ファイルがある
    write_file(
        tmp_path / "scripts/task.py",
        """
        def run() -> None:
            import os

            print(os.getcwd())
        """,
    )
    # when: import rule を検査する
    violations = checker.check_import_rules(
        repo_root=tmp_path,
        scan_paths=[tmp_path / "scripts"],
        packages=[],
    )
    # then: モジュールトップレベル以外の import として報告される
    assert len(violations) == 1
    assert violations[0][3] == "import statements must appear at module top level"
