FROM python:3.12-slim AS base

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock README.md /app/
RUN uv sync --frozen

COPY src /app/src

# raw dataset + tuned hyperparameters, needed to run train/promote standalone
# in any environment. Kept outside /app/data since that path gets shadowed by
# the volume mount below; docker-entrypoint.sh seeds it in from here.
COPY data/raw/stroke.csv /app/seed/raw/stroke.csv
COPY data/best_params.json /app/seed/best_params.json
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

# mount the host data/ dir here (contains mlflow.db + mlruns/) so the
# registered model promoted via `make promote` is available to serve
VOLUME ["/app/data"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD ["uv", "run", "python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=2).status==200 else 1)"]

# ROOT_PATH lets Coolify serve this behind a subpath (e.g. /portfolio/stroke-predictor)
# so FastAPI/Gradio generate correctly prefixed asset and websocket URLs.
# Leave unset to serve from the domain root.
ENV ROOT_PATH=""

ENTRYPOINT ["/app/docker-entrypoint.sh"]

CMD ["sh", "-c", "uv run uvicorn src.stroke_risk.app.main:app --host 0.0.0.0 --port 8000 --proxy-headers --forwarded-allow-ips='*' --root-path \"$ROOT_PATH\""]