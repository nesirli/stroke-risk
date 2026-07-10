install:
	uv sync --all-extras

train:
	uv run python -m stroke-risk-api.models.train

tune:
	uv run python -m stroke-risk-api.models.tune

evaluate:
	uv run python -m stroke-risk-api.models.evaluate

predict:
	uv run python -m stroke-risk-api.models.predict

test:
	uv run pytest tests/

lint:
	uv run ruff check src tests