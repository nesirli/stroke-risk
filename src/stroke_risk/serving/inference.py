import mlflow
import mlflow.sklearn
import pandas as pd

from stroke_risk.config import settings

def load_model():
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    return mlflow.sklearn.load_model(
        f"models:/{settings.mlflow_model_name}@{settings.mlflow_model_alias}"
    )

def predict(patient: dict, model) -> dict:
    patient_id = patient.pop('id')

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
        "gender": "male",
        "ever_married": "yes",
        "work_type": "private",
        "residence_type": "urban",
        "smoking_status": "formerly_smoked",
    }
    model = load_model()
    print(predict(patient, model))