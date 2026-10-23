import streamlit as st
import pandas as pd
import pickle
import json

st.set_page_config(
    page_title="House Price Prediction | Case Study",
    page_icon="🏡",
    layout="wide"
)

# Load trained model and metrics
@st.cache_resource
def load_model():
    with open("case_study_house_model.pkl", "rb") as file:
        return pickle.load(file)

@st.cache_data
def load_metrics():
    with open("model_metrics.json", "r") as file:
        return json.load(file)

model = load_model()
metrics = load_metrics()

# Header
st.title("🏡 House Price Prediction")
st.caption("Machine Learning Case Study | 300 Housing Records")
st.markdown(
    "Estimate a property's sale price using your trained "
    "Linear Regression model."
)

st.divider()

# Model performance
st.subheader("📊 Model Performance")

linear = metrics["Linear Regression"]
forest = metrics["Random Forest"]

c1, c2, c3 = st.columns(3)
c1.metric("Linear Regression R²", f'{linear["R2 Score"]:.2f}')
c2.metric("Linear Regression MAE", f'${linear["MAE"]:,.2f}')
c3.metric("Random Forest R²", f'{forest["R2 Score"]:.2f}')

st.info(
    "Linear Regression performed best on the current test split. "
    "Results may vary with different data splits."
)

st.divider()

# Property inputs
st.subheader("🏠 Enter Property Details")

col1, col2 = st.columns(2)

with col1:
    quality = st.slider("Overall Quality (1–10)", 1, 10, 7)
    living_area = st.number_input(
        "Above-Ground Living Area (sq ft)",
        min_value=100, max_value=10000, value=2000
    )
    garage_cars = st.number_input(
        "Garage Capacity (cars)",
        min_value=0, max_value=5, value=2
    )
    basement_area = st.number_input(
        "Total Basement Area (sq ft)",
        min_value=0, max_value=5000, value=1000
    )

with col2:
    year_built = st.number_input(
        "Year Built", min_value=1800, max_value=2026, value=2005
    )
    full_bath = st.number_input(
        "Full Bathrooms", min_value=0, max_value=5, value=2
    )
    bedrooms = st.number_input(
        "Bedrooms Above Ground", min_value=0, max_value=10, value=3
    )
    lot_area = st.number_input(
        "Lot Area (sq ft)",
        min_value=100, max_value=100000, value=8000
    )

if st.button("💰 Predict House Price", type="primary", use_container_width=True):
    features = pd.DataFrame([{
        "OverallQual": quality,
        "GrLivArea": living_area,
        "GarageCars": garage_cars,
        "TotalBsmtSF": basement_area,
        "YearBuilt": year_built,
        "FullBath": full_bath,
        "BedroomAbvGr": bedrooms,
        "LotArea": lot_area
    }])

    prediction = model.predict(features)[0]

    st.success("Prediction completed!")
    st.metric("Estimated House Sale Price", f"${prediction:,.2f}")
    st.caption(
        "This is a machine learning estimate, not a guaranteed "
        "market valuation."
    )

st.divider()

with st.expander("About this case study"):
    st.write("""
    **Objective:** Predict house sale prices using machine learning.

    **Dataset:** 300 housing records.

    **Algorithms evaluated:** Linear Regression and Random Forest Regression.

    **Evaluation metrics:** Mean Absolute Error (MAE), Root Mean Squared
    Error (RMSE), and R² score.
    """)
