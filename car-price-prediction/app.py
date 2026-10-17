from pathlib import Path

import os

import html
import textwrap

import pandas as pd

import requests

import streamlit as st
import streamlit.components.v1 as components


# =========================================================

# 1. CONFIGURATION

# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_PATH = PROJECT_ROOT / "data" / "quikr_car.csv"

CSS_PATH = PROJECT_ROOT / "styles.css"

def get_api_base_url():

    """Read the deployed API URL from Streamlit secrets or environment."""

    try:

        configured_url = st.secrets.get("API_BASE_URL", "")

    except Exception:

        configured_url = ""

    return (configured_url or os.getenv("API_BASE_URL") or "http://127.0.0.1:8000").rstrip("/")


API_URL = f"{get_api_base_url()}/predict"

HERO_VIDEO_URL = "/app/static/assets/car-hero.mp4"


def render_cinematic_table(frame: pd.DataFrame, table_label: str = "Data table") -> None:

    """Render a compact HTML table that matches the dark cinematic dashboard."""

    safe_columns = [html.escape(str(column)) for column in frame.columns]

    header = "".join(f"<th scope=\"col\">{column}</th>" for column in safe_columns)

    body_rows = []

    for row in frame.itertuples(index=False, name=None):

        cells = "".join(

            f"<td>{html.escape(str(value))}</td>"

            for value in row

        )

        body_rows.append(f"<tr>{cells}</tr>")

    table_html = (

        f'<div class="cinematic-table-wrap" role="region" aria-label="{html.escape(table_label)}" tabindex="0">'

        f'<table class="cinematic-table"><thead><tr>{header}</tr></thead>'

        f'<tbody>{"".join(body_rows)}</tbody></table></div>'

    )

    st.markdown(table_html, unsafe_allow_html=True)

st.set_page_config(

    page_title="Your Car's Story | Car Price Predictor",

    page_icon="🚘",

    layout="wide",

    initial_sidebar_state="collapsed",

)

# =========================================================

# 2. LOAD CSS

# =========================================================

if CSS_PATH.exists():

    css = CSS_PATH.read_text(encoding="utf-8")

    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

else:

    st.warning("styles.css was not found beside app.py.")

# =========================================================

# 3. LOAD STANDARDIZED MANUFACTURER / MODEL CATALOGUE

# =========================================================


CATALOG_PATH = PROJECT_ROOT / "data" / "manufacturer_model_catalog.csv"


@st.cache_data

def load_car_options():

    if not CATALOG_PATH.exists():

        raise FileNotFoundError(

            f"Standardized model catalogue not found: {CATALOG_PATH}"

        )


    catalog = pd.read_csv(CATALOG_PATH)

    required_columns = {"Manufacturer", "Standardized car model"}

    if not required_columns.issubset(catalog.columns):

        raise ValueError(

            "Catalogue must contain 'Manufacturer' and "

            "'Standardized car model'."

        )


    catalog = catalog.dropna(

        subset=["Manufacturer", "Standardized car model"]

    ).copy()

    catalog["Manufacturer"] = catalog["Manufacturer"].astype(str).str.strip()

    catalog["Standardized car model"] = (

        catalog["Standardized car model"].astype(str).str.strip()

    )

    catalog = catalog[

        (catalog["Manufacturer"] != "")

        & (catalog["Standardized car model"] != "")

    ].drop_duplicates().reset_index(drop=True)


    manufacturers = sorted(catalog["Manufacturer"].unique().tolist())

    return catalog, manufacturers


try:

    cars_df, companies = load_car_options()

except Exception as exc:

    st.error(f"Could not load the standardized model catalogue: {exc}")

    st.stop()


# 4. SESSION STATE

# =========================================================

defaults = {

    "selected_company": None,

    "selected_model": None,

    "prediction_result": None,

    "prediction_details": None,

    "prediction_error": None,

}

for key, default_value in defaults.items():

    if key not in st.session_state:

        st.session_state[key] = default_value

def manufacturer_changed():

    st.session_state.selected_model = None

    st.session_state.prediction_result = None

    st.session_state.prediction_details = None

    st.session_state.prediction_error = None

def model_changed():

    st.session_state.prediction_result = None

    st.session_state.prediction_details = None

    st.session_state.prediction_error = None

# =========================================================

# 5. STORY SECTION HELPER

# UPDATED: LESS HEIGHT AND LESS PADDING

# =========================================================

def render_story_section(

    chapter,

    eyebrow,

    title,

    description,

    element_id=None,

):

    safe_id = f'id="{element_id}"' if element_id else ""

    components.html(

        f"""

        <!DOCTYPE html>

        <html>

        <head>

            <style>

                * {{

                    box-sizing: border-box;

                }}

                html, body {{

                    margin: 0;

                    padding: 0;

                    background: transparent;

                    font-family: Inter, -apple-system,

                        BlinkMacSystemFont, "Segoe UI", sans-serif;

                    color: #f7f8fc;

                }}

                .story {{

                    padding: 16px 8px 8px;

                    background: transparent;

                }}

                .eyebrow {{

                    color: #a9baff;

                    font-size: 11px;

                    font-weight: 700;

                    letter-spacing: 2.5px;

                    line-height: 1.7;

                    text-transform: uppercase;

                    margin-bottom: 14px;

                }}

                h2 {{

                    margin: 0 0 14px;

                    color: #f7f8fc;

                    font-size: clamp(28px, 4vw, 44px);

                    line-height: 1.15;

                    font-weight: 650;

                    letter-spacing: -1.2px;

                }}

                p {{

                    margin: 0;

                    max-width: 760px;

                    color: #b7c0d0;

                    font-size: 15px;

                    line-height: 1.85;

                }}

            </style>

        </head>

        <body>

            <section class="story" {safe_id}>

                <div class="eyebrow">

                    {chapter} / {eyebrow}

                </div>

                <h2>{title}</h2>

                <p>{description}</p>

            </section>

        </body>

        </html>

        """,

        height=140,

        scrolling=False,

    )

# =========================================================

# 6. CHAPTER 01 — CINEMATIC HERO

# =========================================================

components.html(

    f"""

    <!DOCTYPE html>

    <html>

    <head>

        <meta name="viewport" content="width=device-width, initial-scale=1">

        <style>

            * {{

                box-sizing: border-box;

            }}

            html, body {{

                margin: 0;

                padding: 0;

                background: transparent;

                font-family: Inter, -apple-system,

                    BlinkMacSystemFont, "Segoe UI", sans-serif;

            }}

            .hero {{

                position: relative;

                min-height: 540px;

                height: min(72vh, 700px);

                overflow: hidden;

                border-radius: 20px;

                background: #101722;

                color: #f7f8fc;

                isolation: isolate;

            }}

            .hero-video {{

                position: absolute;

                inset: 0;

                width: 100%;

                height: 100%;

                object-fit: cover;

                z-index: -2;

            }}

            .hero-overlay {{

                position: absolute;

                inset: 0;

                z-index: -1;

                background:

                    linear-gradient(

                        90deg,

                        rgba(5, 9, 15, .92) 0%,

                        rgba(5, 9, 15, .68) 48%,

                        rgba(5, 9, 15, .16) 100%

                    ),

                    linear-gradient(

                        0deg,

                        rgba(5, 9, 15, .58),

                        transparent 55%

                    );

            }}

            .hero-content {{

                position: absolute;

                left: clamp(25px, 7vw, 90px);

                right: 24px;

                top: 50%;

                transform: translateY(-50%);

                max-width: 720px;

            }}

            .hero-eyebrow {{

                color: #b8c6ff;

                font-size: 11px;

                font-weight: 700;

                letter-spacing: 3px;

                margin-bottom: 24px;

            }}

            h1 {{

                font-size: clamp(42px, 6.5vw, 82px);

                line-height: 1.02;

                letter-spacing: -3px;

                margin: 0 0 22px;

                font-weight: 700;

                color: #ffffff;

            }}

            .hero-description {{

                max-width: 490px;

                color: #d2d8e3;

                font-size: clamp(15px, 1.7vw, 18px);

                line-height: 1.8;

                margin: 0 0 30px;

            }}

            .hero-cta {{

                display: inline-flex;

                align-items: center;

                justify-content: center;

                gap: 14px;

                padding: 16px 23px;

                border: 1px solid rgba(220, 227, 255, .55);

                border-radius: 5px;

                background: #a9baff;

                color: #101522;

                font-size: 12px;

                font-weight: 800;

                letter-spacing: 1.3px;

                cursor: pointer;

                transition: background .2s ease,

                    transform .2s ease;

            }}

            .hero-cta:hover {{

                background: #c4ceff;

                transform: translateY(-2px);

            }}

            .hero-footnote {{

                margin-top: 20px;

                color: #bfc7d5;

                font-size: 11px;

                letter-spacing: .5px;

            }}

            @media (max-width: 600px) {{

                .hero {{

                    min-height: 510px;

                    height: 68vh;

                    border-radius: 14px;

                }}

                h1 {{

                    letter-spacing: -1.8px;

                }}

                .hero-overlay {{

                    background: linear-gradient(

                        90deg,

                        rgba(5, 9, 15, .88),

                        rgba(5, 9, 15, .42)

                    );

                }}

            }}

            @media (prefers-reduced-motion: reduce) {{

                .hero-cta {{

                    transition: none;

                }}

            }}

        </style>

    </head>

    <body>

        <section class="hero">

            <video

                class="hero-video"

                autoplay

                muted

                loop

                playsinline

                preload="auto"

            >

                <source src="{HERO_VIDEO_URL}" type="video/mp4">

            </video>

            <div class="hero-overlay"></div>

            <div class="hero-content">

                <div class="hero-eyebrow">

                    THE AUTOMOTIVE EXPERIENCE

                </div>

                <h1>Every journey<br>leaves a story.</h1>

                <p class="hero-description">

                    Every mile changes a car's story.

                    Discover what your car might be worth today

                    with a machine-learning powered estimate.

                </p>

                <button class="hero-cta" id="begin-journey" type="button">

                    BEGIN YOUR JOURNEY

                    <span aria-hidden="true">↘</span>

                </button>

                <div class="hero-footnote">

                    A MACHINE-LEARNING POWERED ESTIMATE

                </div>

            </div>

        </section>

        <script>

            document

                .getElementById("begin-journey")

                .addEventListener("click", function () {{

                    try {{

                        const parentDocument = window.parent.document;

                        const target = parentDocument.getElementById(

                            "car-form-anchor"

                        );

                        if (target) {{

                            target.scrollIntoView({{

                                behavior: "smooth",

                                block: "start"

                            }});

                        }} else {{

                            console.warn("Car form anchor was not found.");

                        }}

                    }} catch (error) {{

                        console.error(

                            "Could not navigate to car form:",

                            error

                        );

                    }}

                }});

        </script>

    </body>

    </html>

    """,

    height=570,

    scrolling=False,


)

# =========================================================

# 7. CHAPTER 02 — MEET YOUR CAR

# =========================================================

render_story_section(

    "CHAPTER 02",

    "THE BEGINNING",

    "First, let's meet your car.",

    (

        "Every car is different. Choose its manufacturer and standardized model family, "

        "then we'll estimate its historical asking price using our machine-learning model."

    ),

)

# =========================================================

# 8. CAR DETAILS FORM

# =========================================================

st.markdown(

    '<div id="car-form-anchor"></div>',

    unsafe_allow_html=True,

)

with st.container(border=True):

    st.markdown("### Your car's identity")

    st.caption("Choose a manufacturer to see its standardized model families.")

    company = st.selectbox(

        "Manufacturer",

        options=companies,

        index=None,

        placeholder="Choose a manufacturer",

        key="selected_company",

        on_change=manufacturer_changed,

    )

    if company:

        model_options = sorted(

            cars_df.loc[

                cars_df["Manufacturer"].eq(company),

                "Standardized car model",

            ].unique().tolist()

        )

    else:

        model_options = []

    if st.session_state.selected_model not in model_options:

        st.session_state.selected_model = None

    name = st.selectbox(

        "Car model",

        options=model_options,

        index=None,

        placeholder=(

            "Choose a manufacturer first"

            if not company

            else "Choose a car model"

        ),

        disabled=not bool(company),

        key="selected_model",

        on_change=model_changed,

    )

    st.markdown("### The journey so far")

    col1, col2 = st.columns(2)

    with col1:

        year = st.number_input(

            "Manufacturing year",

            min_value=1990,

            max_value=2026,

            value=2016,

            step=1,

            key="manufacturing_year",

        )

    with col2:

        kms_driven = st.number_input(

            "Kilometres driven",

            min_value=0,

            max_value=1_000_000,

            value=65000,

            step=1000,

            key="kilometres_driven",

        )

    fuel_type = st.selectbox(

        "Fuel type",

        options=["Petrol", "Diesel", "CNG", "LPG", "Electric"],

        key="fuel_type",

    )

    st.caption(

        "This estimate may differ from current local market prices."

    )

    discover_clicked = st.button(

        "Discover my car's value →",

        type="primary",

        width="stretch",

        key="discover_price",

    )

# =========================================================

# 9. CALL FASTAPI

# =========================================================

if discover_clicked:

    st.session_state.prediction_result = None

    st.session_state.prediction_details = None

    st.session_state.prediction_error = None

    if not company or not name:

        st.session_state.prediction_error = (

            "Please select both a manufacturer and a car model."

        )

    else:

        payload = {

            # The selected name is a standardized family from the catalogue.

            # Prefix it with the manufacturer for the current feature mapper.

            "name": f"{company} {name}",

            "company": company,

            "year": int(year),

            "kms_driven": float(kms_driven),

            "fuel_type": fuel_type,

        }

        try:

            with st.spinner(

                "Calculating your car's estimated value..."

            ):

                response = requests.post(

                    API_URL,

                    json=payload,

                    timeout=15,

                )

            if response.ok:

                result = response.json()

                if "predicted_price" not in result:

                    st.session_state.prediction_error = (

                        "The API response is missing predicted_price. "

                        "Please inspect the FastAPI response."

                    )

                else:

                    st.session_state.prediction_result = result

                    st.session_state.prediction_details = payload

            else:

                st.session_state.prediction_error = (

                    f"Prediction failed (HTTP {response.status_code}). "

                    f"API response: {response.text[:1000]}"

                )

        except requests.exceptions.ConnectionError:

            st.session_state.prediction_error = (

                "Cannot connect to FastAPI. Start the API server "

                "in a separate terminal."

            )

        except requests.exceptions.Timeout:

            st.session_state.prediction_error = (

                "The request timed out. Please try again."

            )

        except requests.exceptions.RequestException as exc:

            st.session_state.prediction_error = f"Request error: {exc}"

        except ValueError:

            st.session_state.prediction_error = (

                "The API returned an invalid response. "

                "Check the API server logs."

            )

if st.session_state.prediction_error:

    st.error(st.session_state.prediction_error)

# =========================================================

# 10. CHAPTER 03 — PRICE REVEAL

# =========================================================

render_story_section(

    "CHAPTER 03",

    "THE REVEAL",

    "And now, the moment of truth.",

    (

        "Your car details are processed by our trained model "

        "to generate an estimated resale value."

    ),

    element_id="price-reveal",

)

result = st.session_state.prediction_result

details = st.session_state.prediction_details

if result is not None and details is not None:

    components.html(

        """

        <script>

            const target =

                window.parent.document.getElementById("price-reveal");

            if (target) {

                target.scrollIntoView({

                    behavior: "smooth",

                    block: "start"

                });

            }

        </script>

        """,

        height=0,

    
    )

    try:

        predicted_price = float(result["predicted_price"])

        if not pd.notna(predicted_price):

            raise ValueError("Price is not a valid number.")

        st.markdown(

            '<div class="result-eyebrow">'

            'YOUR ESTIMATED RESALE VALUE'

            '</div>',

            unsafe_allow_html=True,

        )

        st.metric(

            "Estimated car price",

            f"₹{predicted_price:,.0f}",

        )

    except (ValueError, TypeError):

        st.error(

            "The model returned an invalid price. "

            "Please check the API response."

        )

    else:

        if result.get("requires_review", False):

            st.warning(

                "This estimate needs manual review. "

                "Treat it as an uncertain model output."

            )

        st.caption(

            "This is a machine-learning estimate, not a guaranteed "

            "market price or a verified valuation."

        )

        with st.expander("See the details used for this estimate"):

            st.write(f"Manufacturer: {details['company']}")

            st.write(f"Model: {details['name']}")

            st.write(f"Manufacturing year: {details['year']}")

            st.write(

                f"Kilometres driven: {details['kms_driven']:,.0f}"

            )

            st.write(f"Fuel type: {details['fuel_type']}")

with st.container(key="chapter4_dashboard"):

    # =========================================================

    # 11. CHAPTER 04 — BEHIND THE ESTIMATE

    # =========================================================

    render_story_section(

        "CHAPTER 04",

        "BEHIND THE ESTIMATE",

        "A story powered by data.",

        (

            "Explore model performance, prediction errors and patterns "

            "in the dataset as the project develops."

        ),

        element_id="behind-the-estimate",

    )


    # =========================================================

    # 12. TRANSPARENT MODEL-INSIGHTS DASHBOARD

    # These charts use saved out-of-fold evaluation records, not invented data.

    # =========================================================

    GROUP_CV_PATH = PROJECT_ROOT / "reports" / "group_cross_validation_results.csv"

    GROUP_ERROR_PATH = PROJECT_ROOT / "reports" / "group_cv_error_predictions.csv"

    PRICE_BAND_PATH = PROJECT_ROOT / "reports" / "group_cv_errors_by_price_band.csv"


    def format_inr_lakh(value):

        """Format rupee values compactly for dashboard labels."""

        return f"₹{value / 100000:.2f}L"


    @st.cache_data

    def load_model_insights():

        """Load the precomputed grouped cross-validation reports."""

        required_paths = [GROUP_CV_PATH, GROUP_ERROR_PATH, PRICE_BAND_PATH]

        missing = [path.name for path in required_paths if not path.exists()]

        if missing:

            return None, None, None, missing


        metrics = pd.read_csv(GROUP_CV_PATH)

        predictions = pd.read_csv(GROUP_ERROR_PATH)

        price_bands = pd.read_csv(PRICE_BAND_PATH)

        return metrics, predictions, price_bands, []


    metrics_df, oof_predictions, band_errors, missing_reports = load_model_insights()


    if missing_reports:

        st.info(

            "Model-insights charts are unavailable because these evaluation "

            f"reports are missing: {', '.join(missing_reports)}. "

            "The car-price predictor remains available."

        )

    else:

        rf_metrics = metrics_df.loc[

            metrics_df["model"].eq("Random Forest")

        ]

        lr_metrics = metrics_df.loc[

            metrics_df["model"].eq("Linear Regression")

        ]


        if not rf_metrics.empty and not lr_metrics.empty:

            rf = rf_metrics.iloc[0]

            lr = lr_metrics.iloc[0]


            st.markdown("### What the evaluation says")

            st.caption(

                "Five-fold GroupKFold evaluation, grouped by standardized car model. "

                "Each validation fold holds out model families not seen in its training fold. "

                "These are evaluation averages, not a guarantee for an individual car."

            )


            metric_cols = st.columns(3)

            metric_cols[0].metric(

                "Random Forest · MAE",

                format_inr_lakh(float(rf["MAE_mean"])),

                help="Mean absolute error: average absolute difference between the asking price and prediction, averaged across folds.",

            )

            metric_cols[1].metric(

                "Random Forest · RMSE",

                format_inr_lakh(float(rf["RMSE_mean"])),

                help="Root mean squared error penalizes large misses more heavily than MAE.",

            )

            metric_cols[2].metric(

                "Random Forest · R²",

                f"{float(rf['R2_mean']):.3f}",

                help="R² describes fit relative to predicting the fold's mean price; it can be negative and is not an accuracy percentage.",

            )


            st.markdown("#### Model choice involves a trade-off")

            comparison = pd.DataFrame(

                {

                    "Metric": ["MAE", "RMSE", "R²"],

                    "Random Forest": [

                        float(rf["MAE_mean"]),

                        float(rf["RMSE_mean"]),

                        float(rf["R2_mean"]),

                    ],

                    "Linear Regression": [

                        float(lr["MAE_mean"]),

                        float(lr["RMSE_mean"]),

                        float(lr["R2_mean"]),

                    ],

                }

            )

            comparison_display = comparison.assign(

                **{

                    "Random Forest": comparison["Random Forest"].map(

                        lambda value: f"{value:.3f}" if abs(value) < 1 else format_inr_lakh(value)

                    ),

                    "Linear Regression": comparison["Linear Regression"].map(

                        lambda value: f"{value:.3f}" if abs(value) < 1 else format_inr_lakh(value)

                    ),

                }

            )

            render_cinematic_table(comparison_display, "Model evaluation comparison")

            st.caption(

                "Random Forest was selected for its lower mean MAE. Linear Regression "

                "has lower mean RMSE and higher mean R² in this evaluation, so Random "

                "Forest is not the winner on every metric."

            )


            rf_oof = oof_predictions.loc[

                oof_predictions["model"].eq("Random Forest")

            ].copy()

            rf_oof = rf_oof.dropna(subset=["actual_price", "predicted_price"])


            st.markdown("#### Predicted asking price vs. observed asking price")

            st.write(

                "Each point represents an out-of-fold Random Forest prediction. "

                "Points closer to the diagonal relationship between actual and predicted "

                "prices indicate smaller errors; points far away indicate larger misses."

            )

            if not rf_oof.empty:

                import plotly.graph_objects as go


                # Cinematic scatter plot: dark transparent canvas, soft grid,

                # restrained electric-blue points, and a reference diagonal.

                actual_max = float(rf_oof["actual_price"].max())

                predicted_max = float(rf_oof["predicted_price"].max())

                axis_max = max(actual_max, predicted_max) * 1.04


                scatter_fig = go.Figure()

                scatter_fig.add_trace(go.Scatter(

                    x=[0, axis_max],

                    y=[0, axis_max],

                    mode="lines",

                    name="Perfect estimate",

                    line=dict(color="rgba(205, 190, 255, 0.78)", width=1.5, dash="dash"),

                    hoverinfo="skip",

                ))

                scatter_fig.add_trace(go.Scatter(

                    x=rf_oof["actual_price"],

                    y=rf_oof["predicted_price"],

                    mode="markers",

                    name="Out-of-fold predictions",

                    marker=dict(

                        size=7,

                        color="#63B3FF",

                        opacity=0.58,

                        line=dict(width=0.4, color="rgba(190,220,255,0.45)"),

                    ),

                    customdata=rf_oof[["actual_price", "predicted_price"]].to_numpy(),

                    hovertemplate=(

                        "Observed: ₹%{x:,.0f}<br>"

                        "Predicted: ₹%{y:,.0f}<extra></extra>"

                    ),

                ))

                scatter_fig.update_layout(

                    template="plotly_dark",

                    height=420,

                    margin=dict(l=18, r=18, t=24, b=20),

                    paper_bgcolor="rgba(0,0,0,0)",

                    plot_bgcolor="rgba(7,16,31,0.72)",

                    font=dict(family="Inter, sans-serif", color="#DCE7F8", size=12),

                    xaxis=dict(

                        title="Observed asking price (₹)",

                        range=[0, axis_max],

                        gridcolor="rgba(137,164,204,0.14)",

                        zerolinecolor="rgba(137,164,204,0.18)",

                        tickformat=",.0f",

                    ),

                    yaxis=dict(

                        title="Predicted asking price (₹)",

                        range=[0, axis_max],

                        gridcolor="rgba(137,164,204,0.14)",

                        zerolinecolor="rgba(137,164,204,0.18)",

                        tickformat=",.0f",

                        scaleanchor="x",

                        scaleratio=1,

                    ),

                    legend=dict(

                        orientation="h",

                        yanchor="bottom",

                        y=1.01,

                        xanchor="right",

                        x=1,

                        bgcolor="rgba(0,0,0,0)",

                    ),

                    hoverlabel=dict(

                        bgcolor="#111B2D",

                        bordercolor="#6F7FFF",

                        font_color="#F5F7FF",

                    ),

                )

                st.plotly_chart(

                    scatter_fig,

                    width="stretch",

                    config={"displayModeBar": False, "responsive": True},

                )

                st.caption(

                    f"{len(rf_oof):,} out-of-fold predictions shown. Extreme-price listings "

                    "can stretch the axes and make the majority of points appear clustered."

                )


            st.markdown("#### Error changes across asking-price bands")

            st.write(

                "Mean absolute error (MAE) is shown by the listing's actual price band. "

                "This helps reveal where the model tends to miss by more rupees."

            )

            rf_bands = band_errors.loc[

                band_errors["model"].eq("Random Forest")

            ].copy()

            lr_bands = band_errors.loc[

                band_errors["model"].eq("Linear Regression")

            ].copy()

            band_order = [

                "Under ₹1 lakh",

                "₹1–3 lakh",

                "₹3–6 lakh",

                "₹6–10 lakh",

                "Above ₹10 lakh",

            ]

            if not rf_bands.empty and not lr_bands.empty:

                band_comparison = rf_bands[

                    ["actual_price_band", "rows", "MAE"]

                ].rename(columns={"rows": "Listings evaluated", "MAE": "Random Forest"})

                band_comparison = band_comparison.merge(

                    lr_bands[["actual_price_band", "MAE"]].rename(

                        columns={"MAE": "Linear Regression"}

                    ),

                    on="actual_price_band",

                    how="left",

                )

                band_comparison["_order"] = band_comparison["actual_price_band"].map(

                    {name: index for index, name in enumerate(band_order)}

                )

                band_comparison = band_comparison.sort_values("_order")

                import plotly.graph_objects as go


                bar_fig = go.Figure()

                bar_fig.add_trace(go.Bar(

                    x=band_comparison["actual_price_band"],

                    y=band_comparison["Random Forest"],

                    name="Random Forest",

                    marker=dict(

                        color="#7C83FF",

                        line=dict(color="rgba(180,184,255,0.45)", width=0.7),

                    ),

                    hovertemplate="%{x}<br>Random Forest MAE: ₹%{y:,.0f}<extra></extra>",

                ))

                bar_fig.add_trace(go.Bar(

                    x=band_comparison["actual_price_band"],

                    y=band_comparison["Linear Regression"],

                    name="Linear Regression",

                    marker=dict(

                        color="#42C6D9",

                        line=dict(color="rgba(142,235,244,0.4)", width=0.7),

                    ),

                    hovertemplate="%{x}<br>Linear Regression MAE: ₹%{y:,.0f}<extra></extra>",

                ))

                bar_fig.update_layout(

                    template="plotly_dark",

                    barmode="group",

                    height=390,

                    margin=dict(l=18, r=18, t=24, b=28),

                    paper_bgcolor="rgba(0,0,0,0)",

                    plot_bgcolor="rgba(7,16,31,0.72)",

                    font=dict(family="Inter, sans-serif", color="#DCE7F8", size=12),

                    yaxis=dict(

                        title="Mean absolute error (₹)",

                        gridcolor="rgba(137,164,204,0.14)",

                        zerolinecolor="rgba(137,164,204,0.18)",

                        tickformat=",.0s",

                    ),

                    xaxis=dict(

                        title="Actual asking-price band",

                        gridcolor="rgba(0,0,0,0)",

                        tickfont=dict(size=11),

                    ),

                    legend=dict(

                        orientation="h",

                        yanchor="bottom",

                        y=1.01,

                        xanchor="right",

                        x=1,

                        bgcolor="rgba(0,0,0,0)",

                    ),

                    bargap=0.24,

                    bargroupgap=0.10,

                    hoverlabel=dict(

                        bgcolor="#111B2D",

                        bordercolor="#6F7FFF",

                        font_color="#F5F7FF",

                    ),

                )

                st.plotly_chart(

                    bar_fig,

                    width="stretch",

                    config={"displayModeBar": False, "responsive": True},

                )

                band_table_display = band_comparison.drop(columns="_order").rename(

                    columns={"actual_price_band": "Actual asking-price band"}

                ).assign(

                    **{

                        "Random Forest MAE": band_comparison["Random Forest"].map(format_inr_lakh),

                        "Linear Regression MAE": band_comparison["Linear Regression"].map(format_inr_lakh),

                    }

                )[[

                    "Actual asking-price band", "Listings evaluated",

                    "Random Forest MAE", "Linear Regression MAE"

                ]]

                render_cinematic_table(band_table_display, "Price band error comparison")

                st.caption(

                    "The premium-car band contains fewer examples than the middle bands. "

                    "Read those comparisons cautiously; the source contains historical asking prices, not verified sale prices."

                )


        with st.expander("What this dashboard cannot tell you"):

            st.markdown("\n".join([

                "- The source contains historical listing asking prices, not confirmed completed-sale prices.",

                "- A low average error does not guarantee an accurate estimate for a particular car.",

                "- Premium vehicles and model families with few examples can have large errors.",

                "- Repeated similar listings were found; exact duplicate identity has not been established for every repeated listing.",

                "- GroupKFold tests generalization to model families held out from each fold; it is not a future-time market test.",

                "- The displayed metrics come from saved evaluation reports. Re-run the evaluation after the training data and duplicate review are finalized.",

            ]))



# =========================================================
# PORTFOLIO IDENTITY FOOTER
# =========================================================
# Keep this HTML left-aligned before sending it to Streamlit.
# Markdown treats lines indented by four spaces as code blocks,
# which is why the footer tags were appearing as visible text.
footer_html = """
<footer class="minimal-footer">
    <h3>Praneeth Koushik Sripada</h3>

    <p class="minimal-footer-role">
        Analyst <span>|</span>
        Aspiring Product Manager <span>|</span>
        AI Enthusiast
    </p>

    <nav class="minimal-footer-socials" aria-label="Social links">
       
        <a href="https://github.com/praneethk99"
           target="_blank" rel="noopener noreferrer"
           aria-label="GitHub" title="GitHub">
            <svg viewBox="0 0 24 24" aria-hidden="true">
                <path fill="currentColor"
                      d="M12 .9a11.1 11.1 0 0 0-3.51 21.63
                      c.55.1.76-.24.76-.54v-2.1c-3.1.68-3.76-1.32
                      -3.76-1.32-.5-1.3-1.24-1.65-1.24-1.65
                      -1.01-.69.08-.68.08-.68 1.12.08 1.71 1.15
                      1.71 1.15.99 1.7 2.6 1.21 3.23.93
                      .1-.72.39-1.21.7-1.49-2.48-.28-5.09-1.24
                      -5.09-5.52 0-1.22.44-2.22 1.15-3
                      -.12-.28-.5-1.42.11-2.96 0 0 .94-.3
                      3.05 1.15a10.6 10.6 0 0 1 5.55 0
                      c2.11-1.45 3.05-1.15 3.05-1.15
                      .61 1.54.23 2.68.11 2.96.72.78 1.15 1.78
                      1.15 3 0 4.29-2.61 5.23-5.1 5.51
                      .4.35.75 1.03.75 2.08v3.09c0 .3.2.65
                      .77.54A11.1 11.1 0 0 0 12 .9Z"/>
            </svg>
        </a>

        <a href="https://www.linkedin.com/in/praneeth-koushik-sripada-92665a21b/"
           target="_blank" rel="noopener noreferrer"
           aria-label="LinkedIn" title="LinkedIn">
            <svg viewBox="0 0 24 24" aria-hidden="true">
                <path fill="currentColor"
                      d="M20.45 2H3.55C2.69 2 2 2.68 2 3.52v16.96
                      c0 .84.69 1.52 1.55 1.52h16.9
                      c.86 0 1.55-.68 1.55-1.52V3.52
                      c0-.84-.69-1.52-1.55-1.52ZM7.93 18.45
                      H4.97V9h2.96v9.45ZM6.45 7.71
                      a1.71 1.71 0 1 1 0-3.42 1.71 1.71
                      0 0 1 0 3.42ZM19 18.45h-2.96v-4.6
                      c0-1.1-.02-2.52-1.54-2.52-1.55 0-1.79 1.2
                      -1.79 2.44v4.68H9.75V9h2.84v1.29h.04
                      c.4-.74 1.36-1.52 2.8-1.52 2.99 0 3.54
                      1.97 3.54 4.53v5.15Z"/>
            </svg>
        </a>

        <a href="https://medium.com/@praneethk9912"
           target="_blank" rel="noopener noreferrer"
           aria-label="Medium" title="Medium">
            <svg viewBox="0 0 24 24" aria-hidden="true">
                <path fill="currentColor"
                      d="M2 6.5c0-.3-.1-.5-.4-.7L.2 4.1V3.8h6.1
                      l4.7 10.3 4.1-10.3H21v.3l-1.2 1.2
                      c-.1.1-.2.2-.2.4v13.1c0 .2.1.3.2.4l1.2
                      1.2v.3h-6.7v-.3l1.3-1.2c.1-.1.1-.2.1-.4V7.2
                      l-5 12.6h-.7L4.2 7.2v8.9c0 .5.2.9.5 1.3
                      l1.7 2v.3H1.5v-.3l1.7-2c.3-.4.4-.8.4-1.3V6.5Z
                      M22.2 7.1c1.1 0 1.8.9 1.8 2v8.5c0 1.1-.7 2-1.8 2
                      s-1.8-.9-1.8-2V9.1c0-1.1.7-2 1.8-2Z"/>
            </svg>
        </a>
    </nav>

   
</footer>
"""

# Remove leading indentation from every HTML line so Markdown cannot
# interpret any part of the footer as a code block.
footer_html = "\n".join(line.lstrip() for line in footer_html.splitlines()).strip()
st.markdown("""<style>

/* Align all social icon buttons */
.minimal-footer-socials {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 10px !important;
}

.minimal-footer-socials a {
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 30px !important;
    height: 30px !important;
    padding: 0 !important;
    margin: 0 !important;
    line-height: 1 !important;
    vertical-align: middle !important;
}

.minimal-footer-socials a svg {
    display: block !important;
    width: 16px !important;
    height: 16px !important;
    margin: 0 !important;
    flex-shrink: 0;
}

/* Optical correction for the outlined envelope */
.minimal-footer-socials a[aria-label="Email"] svg {
    width: 17px !important;
    height: 17px !important;
    transform: translateY(0.5px);
}

</style>""", unsafe_allow_html=True)

st.markdown(footer_html, unsafe_allow_html=True)
