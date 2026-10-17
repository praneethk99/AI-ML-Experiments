# Random Forest model update — deployment notes

## What changed
- Confirmed `models/car_price_standardized_model.joblib` contains a `RandomForestRegressor` pipeline (400 trees, `min_samples_leaf=2`, `random_state=42`).
- Updated `models/model_metadata.json` with the GroupKFold results and the model-selection trade-off.
- Added `data/manufacturer_model_catalog.csv` and changed both the Streamlit app and API feature helper to load it from the project itself. This removes the prior dependency on a sibling `car_price_standardized_training_bundle` directory.
- Kept the Streamlit layout and API request/response shape unchanged.

## Evaluation used for the selection
Five-fold GroupKFold grouped by `Standardized car model`:
- Random Forest: mean MAE ₹176,217; mean RMSE ₹313,442; mean R² 0.251.
- Linear Regression: mean MAE ₹193,686; mean RMSE ₹291,602; mean R² 0.343.

Random Forest was selected to prioritize MAE. It does not win on RMSE or R², and the metrics are not a guarantee for an individual prediction.

## Important training limitation
The uploaded ZIP did not include `car_price_training_data.csv` or the original standardized training bundle. Therefore, this update does **not** retrain the model. It uses the Random Forest artifact already present in the ZIP and documents the existing evaluation. Before calling this a freshly retrained production model, include the finalized standardized training dataset, complete the duplicate-aware review, and retrain/validate from that data.

## Run locally (PowerShell)
From the project directory, with the project's virtual environment activated:

```powershell
python -m uvicorn src.api:app --reload
```

In a second terminal, activate the same environment, then run:

```powershell
python -m streamlit run app.py
```

The API should be available at `http://127.0.0.1:8000`; its interactive docs are at `http://127.0.0.1:8000/docs`.

## Checks performed on this ZIP
- Python compile check passed.
- FastAPI `GET /` returned HTTP 200.
- FastAPI `POST /predict` returned HTTP 200 for a sample Audi A4 request.
- Direct prediction checks passed for sample Audi and Maruti requests.

## Model compatibility
The saved estimator emitted a scikit-learn version mismatch warning when loaded in the validation environment. It worked in the checks above, but use the same scikit-learn version as the training environment where possible; if that version is unknown, retrain from source data in the target environment and rerun evaluation before deployment.

## Product disclaimer
The model predicts historical used-car **asking prices**, not verified transaction prices or a guaranteed market valuation.


## Public deployment (two services)

This project needs two services because the Streamlit UI calls the FastAPI model API.

### Render API service
- If the GitHub repository contains this project in a subdirectory, set Root Directory to `car-price-prediction`; otherwise leave it blank.
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn src.api:app --host 0.0.0.0 --port $PORT`
- Health check path: `/`
- After deploy, test the service root and `/docs`.

### Streamlit Community Cloud UI
- Select your GitHub repository.
- Main file path: `car-price-prediction/app.py` if project is in a subdirectory; otherwise `app.py`.
- Choose Python 3.13 if available.
- In Streamlit app Secrets, add TOML: `API_BASE_URL = "https://YOUR-API.onrender.com"` (replace with the real Render URL; do not append `/predict`).

The app reads `API_BASE_URL` from Streamlit secrets, then from an environment variable, and defaults to localhost for local development. Keep the model and `data/manufacturer_model_catalog.csv` in the repository. Do not commit credentials or `.streamlit/secrets.toml`.

The saved model is pinned to scikit-learn 1.8.0 for serialization compatibility. The model was not retrained in this deployment bundle; the included evaluation reflects prior GroupKFold experiments.
