format:
	uv run ruff format
	uv run ruff check --fix --extend-select I

typecheck:
	uv run mypy --ignore-missing-imports .

test:
	uv run pytest -v --tb=short
