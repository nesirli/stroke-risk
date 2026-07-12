from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

ROOT_PATH = Path(__file__).parent.parent.parent

class Settings(BaseSettings):
    mlflow_tracking_uri: str = f"sqlite:///{ROOT_PATH / 'data' / 'mlflow.db'}"
    mlflow_experiment: str = "stroke-risk"
    mlflow_model_name: str = "stroke-risk"
    mlflow_model_alias: str = "champion"
    data_dir: Path = ROOT_PATH / "data"
    best_params_path: Path = ROOT_PATH / "data" / "best_params.json"
    plot_dir: Path = ROOT_PATH / "plots"

    model_config = SettingsConfigDict(env_file=ROOT_PATH / ".env")

settings = Settings()