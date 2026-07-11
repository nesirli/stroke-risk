import mlflow
import mlflow.sklearn
import pandas as pd

from stroke_risk.config import settings

def predict(patient: dict) -> dict:
    patient_id = patient.pop('id')


    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    runs = mlflow.search_runs(
        experiment_names=[settings.mlflow_experiment],
        order_by=["start_time DESC"],
        max_results=1,
    )
    run_id = runs.iloc[0]["run_id"]
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

    prediction = model.predict(pd.DataFrame([patient]))[0]
    prediction_interpret = 'High Risk of Stroke' if prediction == 1 else 'Low Risk of Stroke'

    result = {
        'patient_id': patient_id,
        'prediction': prediction_interpret
    }

    return result

if __name__ == '__main__':
    patient = {
        "id": 1,
        "age": 67,
        "hypertension": 0,
        "heart_disease": 1,
        "avg_glucose_level": 228.69,
        "bmi": 36.6,
        "gender": "Male",
        "ever_married": "Yes",
        "work_type": "Private",
        "residence_type": "Urban",
        "smoking_status": "formerly smoked",
    }
    print(predict(patient))