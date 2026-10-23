                "max": 100000.0,
                "default": 0.0,
                "step": 1.0,
                "help": f"Enter a value for {feature}.",
            },
        )
        with (col1 if index % 2 == 0 else col2):
            input_values[feature] = st.number_input(
                config["label"],
                min_value=float(config["min"]),
                max_value=float(config["max"]),
                value=float(config["default"]),
                step=float(config["step"]),
                help=config["help"],
                key=f"feature_{feature}",
            )

    predict_clicked = st.form_submit_button(
        "🏠 Predict house sale price",
        use_container_width=True,
    )

# --------------------------- PREDICTION ----------------------
if predict_clicked:
    input_df = pd.DataFrame(
        [[input_values[feature] for feature in feature_names]],
        columns=feature_names,
    )

    try:
        predicted_price = float(model.predict(input_df)[0])

        st.markdown(
            f'<div class="result-panel">'
            f'<div class="result-label">Estimated sale price</div>'
            f'<div class="result-price">{predicted_price:,.2f}</div>'
            f'<div style="color:#64748b">Prediction from your saved Linear Regression model</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        with st.expander("Review input values"):
            st.dataframe(input_df.T.rename(columns={0: "Entered value"}), use_container_width=True)

        st.download_button(
            "Download this prediction as CSV",
            data=pd.DataFrame([{
                **input_values,
                "PredictedSalePrice": predicted_price,
            }]).to_csv(index=False).encode("utf-8"),
            file_name="house_price_prediction.csv",
            mime="text/csv",
        )

    except Exception as exc:
        st.error(
            "Prediction failed. Check that the model and input features match, "
            "and that the scikit-learn version is compatible."
        )
        with st.expander("Technical error"):
            st.code(str(exc))

# --------------------------- OPTIONAL DATASET EXPLORER -------
st.header("3 · Optional dataset exploration")
st.caption(
    "Upload your housing CSV/Excel file if you also want to inspect its records. "
    "The saved model prediction works without uploading the training dataset."
)

uploaded_dataset = st.file_uploader(
    "Upload housing dataset (optional)",
    type=["csv", "xlsx", "xls"],
    key="optional_dataset",
)

if uploaded_dataset is not None:
    try:
        if uploaded_dataset.name.lower().endswith((".xlsx", ".xls")):
            dataset = pd.read_excel(uploaded_dataset)
        else:
            dataset = pd.read_csv(uploaded_dataset)

        st.write(f"Rows: {dataset.shape[0]:,} · Columns: {dataset.shape[1]:,}")
        st.dataframe(dataset.head(20), use_container_width=True)

        numeric_columns = dataset.select_dtypes(include=np.number).columns.tolist()
        if numeric_columns:
            selected_column = st.selectbox(
                "Choose a numeric column to visualize",
                numeric_columns,
            )
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(dataset[selected_column].dropna(), kde=True, ax=ax)
            ax.set_title(f"Distribution of {selected_column}")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
    except Exception as exc:
        st.error(f"Could not read the uploaded dataset: {exc}")

# --------------------------- PROJECT NOTES ------------------
with st.expander("Case study summary and limitations"):
    st.markdown(
        """
        **Objective:** Estimate house sale price using property characteristics.

        **Saved model:** `case_study_house_model.pkl` (Linear Regression pipeline).

        **Evaluation metrics:** Loaded from `model_metrics.json`; the displayed scores
        are the values recorded in that file and are not recalculated by this app.

        **Important:** A model prediction is an estimate. Accuracy depends on the
        training data, feature definitions, and model validation. This app does not
        establish a professional property valuation.
        """
    )

st.markdown(
    '<div class="footer">HomeValue · Saved Linear Regression Model · House Price Prediction</div>',
    unsafe_allow_html=True,
)
