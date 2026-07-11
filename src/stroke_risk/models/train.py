import json
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from stroke_risk.models.model import build_log_reg_pipeline
from stroke_risk.ingest.load_data import load_stroke_data
from stroke_risk.config import settings

trusted_types = ["numpy.dtype"]

def train_model(X: pd.DataFrame, y: pd.Series) -> None:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    best_params = json.loads(settings.best_params_path.read_text())
    pipeline = build_log_reg_pipeline(**best_params)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)

    with mlflow.start_run():

        pipeline.fit(X_train, y_train)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)

        mlflow.log_params(best_params)
        mlflow.log_metric('auc', auc)
        mlflow.sklearn.log_model(pipeline, name='model', skops_trusted_types=trusted_types)

        print(f"Model trained. AUC score: {auc:.4f}")

if __name__ == '__main__':
    X, y = load_stroke_data()
    train_model(X, y)