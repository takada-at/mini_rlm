from __future__ import annotations

import argparse
import ast
from pathlib import Path

DEFAULT_SCAN_PATHS = ("mini_rlm", "scripts", "manual_tests", "main.py")
IGNORED_DIR_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
}
Violation = tuple[Path, int, int, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check first-party import boundaries and detect non-top-level imports."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=list(DEFAULT_SCAN_PATHS),
        help=(
            "Files or directories to scan. "
            "Defaults to mini_rlm, scripts, manual_tests, and main.py."
        ),
    )
    parser.add_argument(
        "--package",
        action="append",
        dest="packages",
        default=None,
        help=(
            "First-party package to enforce, e.g. mini_rlm.repl. "
            "When omitted, immediate subpackages under mini_rlm are discovered."
        ),
    )
    return parser.parse_args()


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    args = parse_args()
    packages = args.packages or discover_target_packages(repo_root)
    scan_paths = resolve_scan_paths(repo_root, args.paths)
    violations = check_import_rules(
        repo_root=repo_root,
        scan_paths=scan_paths,
        packages=packages,
    )
    if not violations:
        print("No import rule violations found.")
        return 0

    for violation in violations:
        print(format_violation(repo_root, violation))
    print(f"Found {len(violations)} violation(s).")
    return 1


def discover_target_packages(repo_root: Path) -> list[str]:
    package_root = repo_root / "mini_rlm"
    if not package_root.is_dir():
        return []

    packages: list[str] = []
    for child in sorted(package_root.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith(".") or child.name in IGNORED_DIR_NAMES:
            continue
        if (child / "__init__.py").is_file():
            packages.append(f"mini_rlm.{child.name}")
    return packages


def resolve_scan_paths(repo_root: Path, raw_paths: list[str]) -> list[Path]:
    resolved_paths: list[Path] = []
    for raw_path in raw_paths:
        path = Path(raw_path)
        if not path.is_absolute():
            path = repo_root / path
        resolved_paths.append(path.resolve())
    return resolved_paths


def check_import_rules(
    repo_root: Path,
    scan_paths: list[Path],
    packages: list[str],
) -> list[Violation]:
    exports_by_package = {
        package_name: load_public_exports(repo_root, package_name)
        for package_name in packages
    }

    violations: list[Violation] = []
    for file_path in iter_python_files(scan_paths):
        tree, parse_error = parse_python_file(file_path)
        if parse_error is not None:
            violations.append(parse_error)
            continue

        assert tree is not None
        module_name = module_name_for_path(repo_root, file_path)
        violations.extend(find_non_top_level_imports(file_path, tree))
        for package_name, exports in exports_by_package.items():
            violations.extend(
                find_package_boundary_violations(
                    file_path=file_path,
                    module_name=module_name,
                    tree=tree,
                    package_name=package_name,
                    exports=exports,
                )
            )

    return sorted(violations, key=sort_violation)


def iter_python_files(scan_paths: list[Path]) -> list[Path]:
    files: set[Path] = set()
    for scan_path in scan_paths:
        if not scan_path.exists():
            continue
        if scan_path.is_file():
            if scan_path.suffix == ".py":
                files.add(scan_path)
            continue
        for path in scan_path.rglob("*.py"):
            if should_skip_path(path):
                continue
            files.add(path.resolve())
    return sorted(files)


def should_skip_path(path: Path) -> bool:
    return any(part.startswith(".") or part in IGNORED_DIR_NAMES for part in path.parts)


def parse_python_file(file_path: Path) -> tuple[ast.Module | None, Violation | None]:
    source = file_path.read_text(encoding="utf-8")
    try:
        return ast.parse(source, filename=str(file_path)), None
    except SyntaxError as error:
        line = error.lineno or 1
        column = error.offset or 1
        return None, (
            file_path,
            line,
            column,
            f"failed to parse Python file: {error.msg}",
        )


def module_name_for_path(repo_root: Path, file_path: Path) -> str:
    relative_path = file_path.resolve().relative_to(repo_root.resolve())
    module_path = relative_path.with_suffix("")
    parts = list(module_path.parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def find_non_top_level_imports(file_path: Path, tree: ast.Module) -> list[Violation]:
    violations: list[Violation] = []
    for statement in tree.body:
        violations.extend(
            find_non_top_level_imports_in_node(file_path, statement, True)
        )
    return violations


def find_non_top_level_imports_in_node(
    file_path: Path,
    node: ast.AST,
    is_top_level: bool,
) -> list[Violation]:
    violations: list[Violation] = []
    if isinstance(node, (ast.Import, ast.ImportFrom)) and not is_top_level:
        violations.append(
            (
                file_path,
                node.lineno,
                node.col_offset + 1,
                "import statements must appear at module top level",
            )
        )
        return violations

    for child in ast.iter_child_nodes(node):
        violations.extend(find_non_top_level_imports_in_node(file_path, child, False))
    return violations


def load_public_exports(repo_root: Path, package_name: str) -> set[str]:
    init_path = repo_root.joinpath(*package_name.split("."), "__init__.py")
    tree, parse_error = parse_python_file(init_path)
    if parse_error is not None:
        raise RuntimeError(format_violation(repo_root, parse_error))

    assert tree is not None
    static_all = extract_static_all(tree)
    if static_all is not None:
        return static_all
    return collect_public_names(tree)


def extract_static_all(tree: ast.Module) -> set[str] | None:
    for statement in tree.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        if not assigns_name(statement, "__all__"):
            continue
        if statement.value is None:
            return None
        try:
            value = ast.literal_eval(statement.value)
        except (ValueError, TypeError):
            return None
        if not isinstance(value, (list, tuple, set)):
            return None
        if not all(isinstance(item, str) for item in value):
            return None
        return set(value)
    return None


def assigns_name(statement: ast.Assign | ast.AnnAssign, name: str) -> bool:
    if isinstance(statement, ast.Assign):
        return any(is_name_target(target, name) for target in statement.targets)
    return is_name_target(statement.target, name)


def is_name_target(target: ast.expr, name: str) -> bool:
    return isinstance(target, ast.Name) and target.id == name


def collect_public_names(tree: ast.Module) -> set[str]:
    public_names: set[str] = set()
    for statement in tree.body:
        if isinstance(statement, ast.Import):
            for alias in statement.names:
                imported_name = alias.asname or alias.name.split(".")[0]
                if not imported_name.startswith("_"):
                    public_names.add(imported_name)
            continue
        if isinstance(statement, ast.ImportFrom):
            for alias in statement.names:
                if alias.name == "*":
                    continue
                imported_name = alias.asname or alias.name
                if not imported_name.startswith("_"):
                    public_names.add(imported_name)
            continue
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not statement.name.startswith("_"):
                public_names.add(statement.name)
            continue
        if isinstance(statement, (ast.Assign, ast.AnnAssign)):
            public_names.update(extract_public_target_names(statement))
    public_names.discard("__all__")
    return public_names


def extract_public_target_names(statement: ast.Assign | ast.AnnAssign) -> set[str]:
    target_names: set[str] = set()
    targets = (
        statement.targets if isinstance(statement, ast.Assign) else [statement.target]
    )
    for target in targets:
        target_names.update(extract_target_names(target))
    return {name for name in target_names if not name.startswith("_")}


def extract_target_names(target: ast.expr) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        names: set[str] = set()
        for element in target.elts:
            names.update(extract_target_names(element))
        return names
    return set()


def find_package_boundary_violations(
    file_path: Path,
    module_name: str,
    tree: ast.Module,
    package_name: str,
    exports: set[str],
) -> list[Violation]:
    if module_name == package_name or module_name.startswith(f"{package_name}."):
        return []

    violations: list[Violation] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            violations.extend(
                find_import_statement_violations(
                    file_path=file_path,
                    node=node,
                    package_name=package_name,
                )
            )
            continue

        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level != 0 or node.module is None:
            continue
        violations.extend(
            find_import_from_violations(
                file_path=file_path,
                node=node,
                package_name=package_name,
                exports=exports,
            )
        )

    return violations


def find_import_statement_violations(
    file_path: Path,
    node: ast.Import,
    package_name: str,
) -> list[Violation]:
    violations: list[Violation] = []
    for alias in node.names:
        if not alias.name.startswith(f"{package_name}."):
            continue
        violations.append(
            (
                file_path,
                node.lineno,
                node.col_offset + 1,
                (
                    f"external modules must import {package_name} via the package root, "
                    f"not {alias.name}"
                ),
            )
        )
    return violations


def find_import_from_violations(
    file_path: Path,
    node: ast.ImportFrom,
    package_name: str,
    exports: set[str],
) -> list[Violation]:
    if node.module is None:
        return []

    if node.module.startswith(f"{package_name}."):
        return [
            (
                file_path,
                node.lineno,
                node.col_offset + 1,
                (
                    f"external modules must import {package_name} via the package root, "
                    f"not {node.module}"
                ),
            )
        ]

    if node.module != package_name:
        return []

    violations: list[Violation] = []
    for alias in node.names:
        if alias.name == "*":
            violations.append(
                (
                    file_path,
                    node.lineno,
                    node.col_offset + 1,
                    f"star imports from {package_name} are not allowed",
                )
            )
            continue
        if alias.name in exports:
            continue
        violations.append(
            (
                file_path,
                node.lineno,
                node.col_offset + 1,
                f"{alias.name} is not exported by {package_name}.__init__.py",
            )
        )
    return violations


def sort_violation(violation: Violation) -> tuple[str, int, int, str]:
    file_path, line, column, message = violation
    return (str(file_path), line, column, message)


def format_violation(repo_root: Path, violation: Violation) -> str:
    file_path, line, column, message = violation
    try:
        display_path = file_path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        display_path = file_path.resolve()
    return f"{display_path}:{line}:{column}: {message}"


if __name__ == "__main__":
    raise SystemExit(main())
