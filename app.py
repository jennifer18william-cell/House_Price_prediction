import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏡",
    layout="wide",
)

# Paths are resolved relative to this app.py file.
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "case_study_house_model.pkl"

# Load the saved, trained model
@st.cache_resource
def load_model(model_path):
    with open(model_path, "rb") as file:
        return pickle.load(file)

if not MODEL_PATH.exists():
    st.error(
        "Model file not found: case_study_house_model.pkl. "
        "Add it to the same folder as app.py in your GitHub repository."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error(
        "Could not load the saved model. Check that the model file is valid "
        "and that the scikit-learn version matches the version used to train it."
    )
    with st.expander("Technical details"):
        st.code(str(exc))
    st.stop()

# Header
st.title("🏡 House Price Prediction")
st.caption("Machine Learning Case Study | 300 Housing Records")
st.write("Enter the property details below to estimate its sale price.")

st.divider()

# Property inputs
st.subheader("🏠 Enter Property Details")
col1, col2 = st.columns(2)

with col1:
    quality = st.slider(
        "Overall Quality (1–10)",
        min_value=1,
        max_value=10,
        value=7,
    )
    living_area = st.number_input(
        "Above-Ground Living Area (sq ft)",
        min_value=100,
        max_value=10000,
        value=2000,
        step=50,
    )
    garage_cars = st.number_input(
        "Garage Capacity (cars)",
        min_value=0,
        max_value=5,
        value=2,
        step=1,
    )
    basement_area = st.number_input(
        "Total Basement Area (sq ft)",
        min_value=0,
        max_value=5000,
        value=1000,
        step=50,
    )

with col2:
    year_built = st.number_input(
        "Year Built",
        min_value=1800,
        max_value=2026,
        value=2005,
        step=1,
    )
    full_bath = st.number_input(
        "Full Bathrooms",
        min_value=0,
        max_value=5,
        value=2,
        step=1,
    )
    bedrooms = st.number_input(
        "Bedrooms Above Ground",
        min_value=0,
        max_value=10,
        value=3,
        step=1,
    )
    lot_area = st.number_input(
        "Lot Area (sq ft)",
        min_value=100,
        max_value=100000,
        value=8000,
        step=100,
    )

if st.button(
    "💰 Predict House Price",
    type="primary",
    use_container_width=True,
):
    # Keep feature names consistent with the saved model.
    features = pd.DataFrame([{
        "OverallQual": quality,
        "GrLivArea": living_area,
        "GarageCars": garage_cars,
        "TotalBsmtSF": basement_area,
        "YearBuilt": year_built,
        "FullBath": full_bath,
        "BedroomAbvGr": bedrooms,
        "LotArea": lot_area,
    }])

    # If the model records feature names, align the inputs to them.
    if hasattr(model, "feature_names_in_"):
        expected_features = list(model.feature_names_in_)
        missing_features = [name for name in expected_features if name not in features.columns]
        if missing_features:
            st.error(
                "The saved model expects feature(s) not provided by this form: "
                + ", ".join(missing_features)
            )
            st.stop()
        features = features[expected_features]

    try:
        prediction = float(model.predict(features)[0])
        st.success("Prediction completed!")
        st.metric(
            label="Estimated House Sale Price",
            value=f"${prediction:,.2f}",
        )
        st.caption(
            "This is a machine learning estimate, not a guaranteed market valuation."
        )

        prediction_record = features.copy()
        prediction_record["PredictedSalePrice"] = prediction
        st.download_button(
            "Download prediction as CSV",
            data=prediction_record.to_csv(index=False).encode("utf-8"),
            file_name="house_price_prediction.csv",
            mime="text/csv",
        )
    except Exception as exc:
        st.error(
            "Prediction failed. Confirm that the saved model was trained with "
            "the same eight features and compatible preprocessing."
        )
        with st.expander("Technical details"):
            st.code(str(exc))

st.divider()

with st.expander("About this case study"):
    st.write("""
    **Objective:** Predict house sale prices using machine learning.

    **Dataset:** 300 housing records.

    **Model used:** Saved trained regression model.

    Enter the property characteristics and click **Predict House Price** to view
    the estimated sale price.
    """)
