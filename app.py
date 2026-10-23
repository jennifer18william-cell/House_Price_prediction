num_df = df.select_dtypes(include=np.number)
if num_df.shape[1] >= 2:
    st.subheader("Numeric feature correlation")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(num_df.corr(numeric_only=True), cmap="vlag", center=0, ax=ax)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ----------------------- MODEL COMPARISON -------------------
st.header("3 · Compare regression models")
st.caption("Lower MAE/RMSE are better; R² closer to 1 is generally better.")
st.dataframe(
    metrics.style.format({"MAE": "{:,.2f}", "RMSE": "{:,.2f}", "R²": "{:.3f}"}),
    use_container_width=True,
)
fig, ax = plt.subplots(figsize=(8, 4))
metrics.set_index("Model")[["MAE", "RMSE"]].plot(kind="bar", ax=ax)
ax.set_ylabel("Error (price units)")
ax.tick_params(axis="x", rotation=0)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.info(f"Model selected for prediction: **{best_name}** (based on lower test-set MAE).")
st.subheader("Actual vs predicted prices")
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(y_test, best_res["predictions"], alpha=0.75)
low = min(float(y_test.min()), float(np.min(best_res["predictions"])))
high = max(float(y_test.max()), float(np.max(best_res["predictions"])))
if low == high:
    low -= 1
    high += 1
ax.plot([low, high], [low, high], "--")
ax.set_xlabel("Actual sale price")
ax.set_ylabel("Predicted sale price")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# ----------------------- FEATURE IMPORTANCE -----------------
st.header("4 · Feature importance")
if best_name == "Random Forest Regression":
    try:
        names = best_model.named_steps["preprocessor"].get_feature_names_out()
        importances = best_model.named_steps["regressor"].feature_importances_
        imp_df = pd.DataFrame({"Feature": names, "Importance": importances})
        imp_df = imp_df.sort_values("Importance", ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(9, 5))
        imp_df.sort_values("Importance").plot(
            kind="barh", x="Feature", y="Importance", legend=False, ax=ax
        )
        ax.set_xlabel("Importance")
        ax.set_ylabel("")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    except Exception:
        st.info("Feature importance is unavailable for this dataset.")
else:
    st.caption("Random Forest was not selected by MAE; Linear Regression was selected.")

# ----------------------- PRICE PREDICTION -------------------
st.header("5 · Predict a house price")
st.caption("Enter property details. Input fields are generated from your dataset columns.")
with st.form("prediction_form"):
    values = {}
    input_cols = st.columns(2)
    for i, feature in enumerate(X.columns):
        with input_cols[i % 2]:
            if feature in numeric_features:
                series = pd.to_numeric(X[feature], errors="coerce").dropna()
                if series.empty:
                    low, high, default = 0.0, 1.0, 0.0
                else:
                    low, high, default = float(series.min()), float(series.max()), float(series.median())
                    if low == high:
                        low -= 1.0
                        high += 1.0
                values[feature] = st.number_input(
                    str(feature), min_value=low, max_value=high,
                    value=min(max(default, low), high), key=f"pred_{feature}"
                )
            else:
                options = X[feature].dropna().astype(str).value_counts().index.tolist()
                if not options:
                    options = ["Unknown"]
                values[feature] = st.selectbox(str(feature), options, key=f"pred_{feature}")
    predict_clicked = st.form_submit_button("🏠 Estimate sale price", use_container_width=True)

if predict_clicked:
    new_house = pd.DataFrame([values], columns=X.columns)
    for col in numeric_features:
        new_house[col] = pd.to_numeric(new_house[col], errors="coerce")
    try:
        predicted_price = float(best_model.predict(new_house)[0])
        st.success(f"Estimated sale price: {predicted_price:,.2f}")
        st.caption(f"Estimate produced by {best_name}; not a professional valuation.")
    except Exception as exc:
        st.error("Prediction failed. Check the input data and model.")
        st.exception(exc)

# ----------------------- EXPORT + NOTES ---------------------
st.header("6 · Export and case-study notes")
st.download_button(
    "Download model comparison CSV",
    data=metrics.to_csv(index=False).encode("utf-8"),
    file_name="house_price_model_metrics.csv",
    mime="text/csv",
)
with st.expander("Case study summary and limitations"):
    st.markdown("""
    **Objective:** Predict `SalePrice` from housing characteristics.

    **Workflow:** Data cleaning → EDA → missing-value imputation and categorical
    encoding → train/test split → Linear Regression and Random Forest Regression
    → MAE, RMSE and R² comparison → house-price prediction.

    **Limitation:** A small synthetic dataset of around 300 records is useful for an
    academic case study but does not guarantee reliable real-world valuations.
    Validate the model before practical use.
    """)

st.markdown(
    '<div class="footer">HomeValue · House Price Prediction · Linear Regression & Random Forest Regression</div>',
    unsafe_allow_html=True,
)
