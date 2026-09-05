FROM python:3.12-slim AS base

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock README.md /app/
RUN uv sync --frozen

COPY src /app/src

# raw dataset + tuned hyperparameters, needed to run train/promote standalone
# in any environment. Kept outside /app/data since that path gets shadowed by
# an attached Railway Volume; docker-entrypoint.sh seeds it in from here.
COPY data/raw/stroke.csv /app/seed/raw/stroke.csv
COPY data/best_params.json /app/seed/best_params.json
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

# /app/data holds mlflow.db + mlruns/ at runtime. On Railway, attach a volume
# there (railway volume add -m /app/data) to persist the promoted model across
# redeploys; without it the entrypoint retrains on every deploy.

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python3 -c 'import os,urllib.request,sys; p=os.environ.get("PORT","8000"); sys.exit(0 if urllib.request.urlopen("http://localhost:"+p+"/health",timeout=5).status==200 else 1)'

# ROOT_PATH lets a reverse proxy serve this behind a subpath (e.g. /portfolio/stroke-risk)
# so FastAPI/Gradio generate correctly prefixed asset and websocket URLs.
# Leave unset to serve from the domain root (the default on Railway).
ENV ROOT_PATH=""

ENTRYPOINT ["/app/docker-entrypoint.sh"]

CMD ["sh", "-c", "uv run uvicorn src.stroke_risk.app.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*' --root-path \"$ROOT_PATH\""]