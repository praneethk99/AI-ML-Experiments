
from pathlib import Path
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "data" / "manufacturer_model_catalog.csv"


def normalize_text(value: str) -> str:
    """Normalize text for matching manufacturer and model names."""
    return re.sub(r"\s+", " ", str(value).lower().strip())


def load_model_catalog() -> pd.DataFrame:
    """Load the standardized manufacturer/model catalogue."""
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(
            f"Model catalogue not found: {CATALOG_PATH}"
        )

    catalog = pd.read_csv(CATALOG_PATH)

    required_columns = {"Manufacturer", "Standardized car model"}
    if not required_columns.issubset(catalog.columns):
        raise ValueError(
            "The catalogue must contain Manufacturer and "
            "Standardized car model columns."
        )

    catalog = catalog.dropna(
        subset=["Manufacturer", "Standardized car model"]
    ).copy()

    catalog["Manufacturer"] = catalog["Manufacturer"].astype(str).str.strip()
    catalog["Standardized car model"] = (
        catalog["Standardized car model"].astype(str).str.strip()
    )

    return catalog.drop_duplicates().reset_index(drop=True)


def extract_model_family(row: dict) -> str:
    """Return a standardized model family for a listing."""
    name = normalize_text(row["name"])
    company = normalize_text(row["company"])

    if name.startswith(company):
        name = name[len(company):].strip()

    if company == "maruti" and name.startswith("suzuki "):
        name = name[len("suzuki "):].strip()

    catalog = load_model_catalog()

    manufacturer_matches = catalog[
        catalog["Manufacturer"].map(normalize_text) == company
    ]

    # Prefer the longest matching model name to avoid matching
    # 'i10' before 'Grand i10', or 'Swift' before 'Swift Dzire'.
    candidates = sorted(
        manufacturer_matches["Standardized car model"].tolist(),
        key=len,
        reverse=True,
    )

    for family in candidates:
        pattern = r"(?<![a-z0-9])" + re.escape(
            normalize_text(family)
        ) + r"(?![a-z0-9])"

        if re.search(pattern, name):
            return family

    return "UNKNOWN"
