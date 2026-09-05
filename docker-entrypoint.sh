#!/bin/sh
set -e

# /app/data is a mounted volume (often empty on first run on Railway/Coolify),
# which shadows anything baked into the image at that path. Seed it from
# the image on first run without clobbering an already-trained volume.
mkdir -p /app/data/raw
[ -f /app/data/raw/stroke.csv ] || cp /app/seed/raw/stroke.csv /app/data/raw/stroke.csv
[ -f /app/data/best_params.json ] || cp /app/seed/best_params.json /app/data/best_params.json

# If no champion model is registered yet (fresh volume or ephemeral Railway
# deploy), train and promote one so /predict works out of the box.
if ! uv run python - <<'PY'
import mlflow
from stroke_risk.config import settings

mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
try:
    mlflow.MlflowClient().get_model_version_by_alias(
        settings.mlflow_model_name, settings.mlflow_model_alias
    )
except Exception:
    raise SystemExit(1)
PY
then
    echo "No champion model found - training and promoting one"
    uv run python -m src.stroke_risk.models.train
    uv run python -m src.stroke_risk.models.promote
fi

exec "$@"
