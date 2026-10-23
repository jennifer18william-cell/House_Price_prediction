import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# HOUSE PRICE PREDICTION
# Uses the uploaded, already-trained model and metrics.
#
# Keep these files in the same folder as app.py:
#   case_study_house_model.pkl
#   model_metrics.json
#
# Recommended scikit-learn version: 1.6.1
# ============================================================

st.set_page_config(
    page_title="HomeValue | House Price Prediction",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "case_study_house_model.pkl"
METRICS_PATH = BASE_DIR / "model_metrics.json"

# The uploaded model was trained with these feature names.
FEATURE_DEFAULTS = {
    "OverallQual": {
        "label": "Overall quality (1–10)",
        "min": 1.0,
        "max": 10.0,
        "default": 6.0,
        "step": 1.0,
        "help": "Overall material and finish quality.",
    },
    "GrLivArea": {
        "label": "Above-ground living area (sq ft)",
        "min": 100.0,
        "max": 10000.0,
        "default": 1500.0,
        "step": 25.0,
        "help": "Above-ground living area in square feet.",
    },
    "GarageCars": {
        "label": "Garage capacity (cars)",
        "min": 0.0,
        "max": 5.0,
        "default": 2.0,
        "step": 1.0,
        "help": "Number of cars the garage can hold.",
    },
    "TotalBsmtSF": {
        "label": "Total basement area (sq ft)",
        "min": 0.0,
        "max": 5000.0,
        "default": 800.0,
        "step": 25.0,
        "help": "Total basement area in square feet.",
    },
    "YearBuilt": {
        "label": "Year built",
        "min": 1800.0,
        "max": 2026.0,
        "default": 2000.0,
        "step": 1.0,
        "help": "Original construction year.",
    },
    "FullBath": {
        "label": "Full bathrooms",
        "min": 0.0,
        "max": 6.0,
        "default": 2.0,
        "step": 1.0,
        "help": "Number of full bathrooms.",
    },
    "BedroomAbvGr": {
        "label": "Bedrooms above ground",
        "min": 0.0,
        "max": 10.0,
        "default": 3.0,
        "step": 1.0,
        "help": "Bedrooms above ground level.",
    },
    "LotArea": {
        "label": "Lot area (sq ft)",
        "min": 500.0,
