
from pathlib import Path

import joblib
import pandas as pd

from src.model_features import extract_model_family


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "car_price_standardized_model.joblib"

model = joblib.load(MODEL_PATH)


def predict_car_price(
    name: str,
    company: str,
    year: int,
    kms_driven: float,
    fuel_type: str,
) -> dict:
    """Predict a used-car asking price using the standardized model."""

    model_family = extract_model_family({
        "name": name,
        "company": company,
    })

    if model_family == "UNKNOWN":
        raise ValueError(
            f"Could not map '{name}' to a standardized model family "
            f"for manufacturer '{company}'."
        )

    car = pd.DataFrame([{
        "Manufacturer": company.strip(),
        "Standardized car model": model_family,
        "manufacturing_year": year,
        "kms_driven": kms_driven,
        "fuel_type": fuel_type,
    }])

    raw_prediction = float(model.predict(car)[0])
    was_negative = raw_prediction < 0
    displayed_prediction = max(0.0, raw_prediction)

    return {
        "predicted_price": round(displayed_prediction, 2),
        "raw_prediction": round(raw_prediction, 2),
        "requires_review": was_negative,
    }
