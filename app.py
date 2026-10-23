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

# Predict sale price
if st.button(
    "💰 Predict House Price",
    type="primary",
    use_container_width=True,
):
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
    except Exception as exc:
        st.error(
            "Prediction failed. Confirm that the input feature names and order "
            "match those used to train the saved model."
        )
        st.code(str(exc))

st.divider()

with st.expander("About this case study"):
    st.write("""
    **Objective:** Predict house sale prices using machine learning.

    **Dataset:** 300 housing records.

    **Model used:** Saved trained Linear Regression model.

    Enter the property characteristics and click **Predict House Price** to view
    the estimated sale price.
    """)
