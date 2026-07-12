all:
	install validate_data test tune train evaluate promote inference

install:
	uv sync --all-extras

train:
	uv run python -m src.stroke_risk.models.train

tune:
	uv run python -m src.stroke_risk.models.tune

evaluate:
	uv run python -m src.stroke_risk.models.evaluate

promote:
	uv run python -m src.stroke_risk.models.promote

inference:
	uv run python -m src.stroke_risk.serving.inference

test:
	uv run pytest tests/

validate_data:
	uv run python -m src.stroke_risk.utils.validate_data

interpret:
	uv run python -m src.stroke_risk.models.interpret

serve-api:
	uv run uvicorn src.stroke_risk.app.main:app --reload

lint:
	uv run ruff check src tests