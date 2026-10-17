from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import predict_car_price

app = FastAPI(
    title="Car Price Prediction API",
    description="Estimate used-car asking prices from vehicle details.",
    version="1.1.0",
)


class CarInput(BaseModel):
    name: str = Field(min_length=1)
    company: str = Field(min_length=1)
    year: int = Field(ge=1990, le=2026)
    kms_driven: float = Field(ge=0)
    fuel_type: str = Field(min_length=1)


@app.get("/")
def home():
    return {"message": "Car Price Prediction API is running"}


@app.post("/predict")
def predict(car: CarInput):
    try:
        return predict_car_price(
            name=car.name,
            company=car.company,
            year=car.year,
            kms_driven=car.kms_driven,
            fuel_type=car.fuel_type,
        )
    except ValueError as exc:
        # Unmapped model families are input issues, not server crashes.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        # Keep the client response safe, but log the underlying exception locally.
        print(f"Prediction error: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the FastAPI server logs.",
        ) from exc
