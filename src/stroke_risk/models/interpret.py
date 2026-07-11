import shap
import mlflow
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from stroke_risk.config import settings
from stroke_risk.ingest.load_data import load_stroke_data
from stroke_risk.features.build_features import NUM_COLS


def interpret_model(X, y):

    X_train, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    runs = mlflow.search_runs(
        experiment_names=[settings.mlflow_experiment],
        order_by=["start_time DESC"],
        max_results=1,
    )

    run_id = runs.iloc[0]["run_id"]
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

    preprocessor = model.named_steps["preprocessor"]
    log_reg = model.named_steps["model"]

    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    explainer = shap.LinearExplainer(log_reg, X_train_transformed)
    shap_values = explainer(X_test_transformed)
    feature_names = [re.sub(r"^(num__|cat__)", "", f) for f in preprocessor.get_feature_names_out()]
    shap_values.feature_names = feature_names
    shap_values.data[:, : len(NUM_COLS)] = X_test[NUM_COLS].to_numpy()

    plot_dir = settings.plot_dir
    plot_dir.mkdir(exist_ok=True)

    shap.plots.beeswarm(shap_values, show=False)
    plt.savefig(settings.plot_dir / "log_reg_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()

    sample_ind = 0
    shap.plots.waterfall(shap_values[sample_ind], show=False)
    plt.savefig(plot_dir / "log_reg_shap_waterfall.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"SHAP plots are saved in {plot_dir}.")

if __name__ == '__main__':
    X, y = load_stroke_data()
    interpret_model(X, y)