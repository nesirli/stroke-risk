import mlflow

from stroke_risk.config import settings


def promote_best_model() -> None:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    runs = mlflow.search_runs(
        experiment_names=[settings.mlflow_experiment],
        order_by=["metrics.auc DESC"],
        max_results=1,
    )
    if runs.empty:
        raise ValueError(f"No runs found for experiment '{settings.mlflow_experiment}'")

    run_id = runs.iloc[0]["run_id"]
    auc = runs.iloc[0]["metrics.auc"]

    model_version = mlflow.register_model(
        model_uri=f"runs:/{run_id}/model",
        name=settings.mlflow_model_name,
    )

    client = mlflow.MlflowClient()
    client.set_registered_model_alias(
        name=settings.mlflow_model_name,
        alias=settings.mlflow_model_alias,
        version=model_version.version,
    )

    print(
        f"Promoted run {run_id} (AUC {auc:.4f}) to "
        f"{settings.mlflow_model_name}@{settings.mlflow_model_alias} (v{model_version.version})"
    )


if __name__ == "__main__":
    promote_best_model()
