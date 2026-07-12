# Stroke Risk Prediction

A small machine learning API that predicts stroke risk from patient health data. It has a REST API, a web UI, and a full MLflow tracking pipeline.

[![CI](https://github.com/nesirli/stroke-risk-api/actions/workflows/ci.yml/badge.svg)](https://github.com/nesirli/stroke-risk-api/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-FF7C00?logo=gradio&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?logo=mlflow&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)

**Quick links:**
[Live demo](https://nasirnesirli.com/portfolio/stroke-risk/gradio/) ·
[API docs](https://nasirnesirli.com/portfolio/stroke-risk/docs) ·
[Docker image](https://hub.docker.com/r/nasirnesirli/stroke-risk) ·
[CI runs](https://github.com/nesirli/stroke-risk-api/actions)

## What it does

You send patient data (age, glucose level, smoking status, and more). The API returns a stroke risk prediction: high risk or low risk.

The project has three parts:

1. A training pipeline that tunes, trains, evaluates, and tracks a logistic regression model with MLflow.
2. A FastAPI service that serves predictions from the best tracked model.
3. A Gradio UI mounted on the same service, for people who prefer a form over JSON.

## How it works

```
raw data (data/raw/stroke.csv)
    │
    ▼
tune.py      → searches hyperparameters with Optuna, saves best_params.json
    │
    ▼
train.py     → trains a logistic regression pipeline, logs it to MLflow
    │
    ▼
evaluate.py  → checks the latest run on a held-out test set
    │
    ▼
promote.py   → finds the run with the best AUC, registers it, and
               tags it as the "champion" model in the MLflow registry
    │
    ▼
FastAPI app  → loads the champion model at startup, serves /predict
    │
    ▼
Gradio UI    → mounted at /gradio on the same app, calls the model directly
```

## Design decisions

This section explains the "why" behind the setup. Not every choice is obvious from the code alone.

**Logistic regression, not a bigger model.** The dataset is small and the classes are imbalanced. A simple, well-tuned linear model is easier to explain to a non-technical reader and works well here. SHAP plots (`make interpret`) show exactly which features drive each prediction.

**MLflow model registry with a "champion" alias, not just "latest run".** Early on, the serving code just loaded the most recent training run. That is a bug: a newer run is not always a better run. `promote.py` picks the run with the highest AUC and registers it. The API always loads by alias (`stroke-risk@champion`), so it always serves the best known model, not the newest one.

**FastAPI and Gradio in one process.** The UI calls the model directly in Python instead of making an HTTP call to the API. This avoids a second network hop and a second service to deploy. The trade off is that a UI crash can affect the API process, but for a small single-container app this is a fair trade.

**Great Expectations for data validation.** Before training, `validate_data.py` checks column names, value ranges, and category values. This catches bad input data early, before it silently breaks a model.

**Docker Hub, not a cloud registry.** For a project this size, a full cloud setup was not needed. The deployment target is Coolify, a self-hosted platform that pulls images straight from Docker Hub. Less infrastructure, less cost, less to maintain.

**A seed step in the Docker image.** The dataset and tuned hyperparameters live outside `/app/data` in the image (`/app/seed`). `docker-entrypoint.sh` copies them into `/app/data` on first start, only if they are missing. This matters because `/app/data` is a mounted volume in production. A volume mount can hide files baked into the image at that same path. Without this step, training would fail on a fresh deployment with no dataset in sight.

**`ROOT_PATH` support.** The app is deployed behind a reverse proxy at a subpath (`/portfolio/stroke-risk`), not at a domain root. Without telling FastAPI and Gradio about that prefix, their generated links and websocket URLs would point to the wrong place. `ROOT_PATH` is an environment variable, so the same image works at a subpath or at a domain root, depending on where it runs.

**A Makefile as the single entry point.** The project has many steps: install, validate data, test, tune, train, evaluate, promote, serve. A Makefile documents the exact order and the exact command for each step, so nobody has to guess or dig through scripts. `make all` runs the full pipeline from a clean checkout to a served model.

## Project structure

```
src/stroke_risk/
├── app/            FastAPI app, Gradio UI, request/response schemas
├── config.py       All settings, one place, overridable by env vars
├── features/       Feature preprocessing (scaling, encoding)
├── ingest/         Loads and cleans the raw CSV
├── models/         Tune, train, evaluate, promote, interpret
├── serving/        Model loading and inference logic
└── utils/          Data validation with Great Expectations

tests/              Unit tests for every module above
data/               Raw data, tuned params, local MLflow store (mostly gitignored)
notebooks/          Exploratory data analysis and modelling notebooks
```

## Getting started

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/nesirli/stroke-risk-api.git
cd stroke-risk-api
make install
```

Then run the full pipeline once, to get a trained and promoted model:

```bash
make validate_data
make tune       # optional, best_params.json is already committed
make train
make evaluate
make promote
```

Start the API locally:

```bash
make serve-api
```

Visit `http://localhost:8000/docs` for the API, or `http://localhost:8000/gradio` for the UI.

## Makefile commands

| Command | What it does |
|---|---|
| `make install` | Installs all dependencies with uv |
| `make validate_data` | Checks the raw dataset against expected schema and ranges |
| `make tune` | Searches hyperparameters with Optuna, writes `best_params.json` |
| `make train` | Trains the model, logs it to MLflow |
| `make evaluate` | Scores the latest run on a held-out test set |
| `make promote` | Registers the best run as the serving champion |
| `make inference` | Runs a one-off prediction from the command line |
| `make interpret` | Generates SHAP plots for the trained model |
| `make serve-api` | Runs the FastAPI app locally, with auto reload |
| `make test` | Runs the test suite |
| `make lint` | Runs ruff checks |
| `make all` | Runs the full pipeline in order, start to finish |

## Running with Docker

```bash
docker build -t stroke-risk .
docker run -p 8000:8000 -v "$(pwd)/data:/app/data" stroke-risk
```

The container needs `/app/data` mounted so the trained model and MLflow store persist across restarts. On first run, the container seeds the dataset and tuned hyperparameters into that volume automatically. To train inside the running container:

```bash
docker exec <container-name> uv run python -m stroke_risk.models.train
docker exec <container-name> uv run python -m stroke_risk.models.promote
```

To deploy behind a reverse proxy at a subpath, set the `ROOT_PATH` environment variable to that subpath, and make sure the proxy strips the prefix before forwarding the request to the container.

## API reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check, returns `{"status": "ok"}` |
| `POST` | `/predict` | Scores one patient, returns a risk prediction |
| `GET` | `/docs` | Interactive Swagger UI |
| `GET` | `/gradio` | Web form UI |

Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "id": 1,
    "age": 67,
    "hypertension": 0,
    "heart_disease": 1,
    "avg_glucose_level": 228.69,
    "bmi": 36.6,
    "gender": "male",
    "ever_married": "yes",
    "work_type": "private",
    "residence_type": "urban",
    "smoking_status": "formerly_smoked"
  }'
```

Response:

```json
{"patient_id": 1, "prediction": "High Risk of Stroke"}
```

If no model has been promoted yet, `/predict` returns a `503` with a clear message, instead of crashing the app.

## Testing

```bash
make test
```

The suite covers feature building, the model pipeline, request schemas, inference logic, the API endpoints, data validation, and the promote step. External services (MLflow registry calls, the trained model) are mocked or stubbed out, so tests run fast and do not need real infrastructure.

## CI/CD

GitHub Actions runs on every push and pull request to `main`:

1. Lint with ruff.
2. Check formatting with ruff.
3. Run the test suite.
4. If all of that passes, and the push is to `main`, build the Docker image and push it to Docker Hub.

This order matters. The image only gets built and published if the code actually passes its checks first.

## License

No license file yet. All rights reserved by default until one is added.
