format:
	uv run ruff format
	uv run ruff check --fix --extend-select I

typecheck:
	uv run mypy --ignore-missing-imports .

lint:
	uv run ruff check .
	uv run python scripts/check_import_rules.py

test:
	uv run pytest -v --tb=short
