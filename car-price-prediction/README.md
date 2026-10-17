# Car Price Prediction

An end-to-end machine learning project that estimates used-car listing prices in Indian rupees (INR). It combines a Streamlit interface, a scikit-learn prediction pipeline, and a FastAPI backend.

## Features

- Predict estimated used-car asking prices using vehicle attributes.
- Select a manufacturer and standardized car model.
- Explore model evaluation and prediction-error analysis.
- Access predictions through a FastAPI service.
- View model limitations and evaluation results.

## Tech Stack

- Python
- Pandas
- Scikit-learn
- Streamlit
- FastAPI
- Joblib
- Plotly

## Model Evaluation

The current model candidate is a Random Forest Regressor evaluated using five-fold GroupKFold, with folds grouped by standardized car model.

| Metric | Random Forest | Linear Regression |
|---|---:|---:|
| Mean Absolute Error (MAE) | ₹176,217 | ₹193,686 |
| Root Mean Squared Error (RMSE) | ₹313,442 | ₹291,602 |
| R² | 0.251 | 0.343 |

Random Forest has a lower MAE, while Linear Regression performs better on RMSE and R². The current model selection prioritizes MAE, so this is a trade-off rather than a win across every metric.

## Important Limitations

- Predictions estimate historical listing asking prices, not verified transaction prices.
- Results are estimates, not guaranteed market valuations.
- Errors may be large for premium cars and vehicle models with limited observations.
- GroupKFold evaluates generalization to model families excluded from individual training folds. It is not a future-time evaluation.
- The serialized model's scikit-learn compatibility needs verification because its original training library version is unknown.

## Project Structure

- `app.py` — Streamlit application
- `src/` — prediction API and feature-processing code
- `models/` — serialized model and metadata
- `data/` — datasets and vehicle catalog
- `notebooks/` — exploratory notebook
- `reports/` — evaluation and data-quality analysis
- `static/` — application assets

## Local Setup

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Start the Streamlit application:

```powershell
streamlit run app.py
```

Start the FastAPI backend in a separate terminal:

```powershell
uvicorn src.api:app --reload
```

The API command assumes the FastAPI application object is named `app` in `src/api.py`.

## Data and Attribution

The project uses used-car datasets and a manufacturer/model catalog. Dataset sources and redistribution permissions must be verified and documented before public redistribution.

## Status

Model evaluation and deployment compatibility checks are ongoing.
