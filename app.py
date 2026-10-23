from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.20,random_state=42)

@st.cache_resource(show_spinner='Training regression models...')
def fit_models(X_train, y_train, numeric_features, categorical_features):
    transformers = []
    if numeric_features:
        transformers.append(('numeric',Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())]),numeric_features))
    if categorical_features:
        transformers.append(('categorical',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('encoder',OneHotEncoder(handle_unknown='ignore'))]),categorical_features))
    prep = ColumnTransformer(transformers=transformers,remainder='drop')
    linear = Pipeline([('preprocessor',prep),('regressor',LinearRegression())])
    forest = Pipeline([('preprocessor',prep),('regressor',RandomForestRegressor(n_estimators=250,random_state=42,n_jobs=-1))])
    linear.fit(X_train,y_train); forest.fit(X_train,y_train)
    return linear, forest
try:
    linear_model, forest_model = fit_models(X_train,y_train,numeric_features,categorical_features)
except Exception as exc:
    st.error('Models could not be trained. Check dataset values and column types.'); st.exception(exc); st.stop()

def evaluate(model):
    pred = model.predict(X_test)
    return {'MAE':mean_absolute_error(y_test,pred),'RMSE':np.sqrt(mean_squared_error(y_test,pred)),'R²':r2_score(y_test,pred),'predictions':pred}
linear_res, forest_res = evaluate(linear_model), evaluate(forest_model)
metrics = pd.DataFrame([{'Model':'Linear Regression','MAE':linear_res['MAE'],'RMSE':linear_res['RMSE'],'R²':linear_res['R²']},{'Model':'Random Forest Regression','MAE':forest_res['MAE'],'RMSE':forest_res['RMSE'],'R²':forest_res['R²']}])
best_name = 'Random Forest Regression' if forest_res['MAE'] <= linear_res['MAE'] else 'Linear Regression'
best_model, best_res = (forest_model,forest_res) if best_name.startswith('Random') else (linear_model,linear_res)

st.header('1 · Dataset overview')
cols = st.columns(4)
summary = [('Housing records',f'{len(df):,}','Rows with valid sale prices'),('Input features',str(X.shape[1]),'Properties used by model'),('Average sale price',f'{y.mean():,.2f}',f'Median: {y.median():,.2f}'),('Best holdout model',best_name,'Lowest MAE on test data')]
for col,(label,value,note) in zip(cols,summary):
    with col: st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',unsafe_allow_html=True)
with st.expander('Preview dataset and data quality'):
    st.dataframe(df.head(20),use_container_width=True)
    st.dataframe(pd.DataFrame({'Column':df.columns,'Type':[str(df[c].dtype) for c in df.columns],'Missing':[int(df[c].isna().sum()) for c in df.columns],'Unique':[int(df[c].nunique(dropna=True)) for c in df.columns]}),use_container_width=True)

st.header('2 · Exploratory data analysis')
a,b = st.columns(2)
with a:
    st.subheader('Sale-price distribution')
    fig,ax=plt.subplots(figsize=(7,4)); sns.histplot(y,kde=True,ax=ax); ax.set_xlabel(target); ax.set_ylabel('Number of houses'); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
with b:
    st.subheader('Missing values by feature')
    miss=df.isna().sum().sort_values(ascending=False); miss=miss[miss>0].head(15)
    if len(miss):
        fig,ax=plt.subplots(figsize=(7,4)); miss.sort_values().plot(kind='barh',ax=ax); ax.set_xlabel('Missing values'); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    else: st.success('No missing values in the original dataset.')
num_df=df.select_dtypes(include=np.number)
if num_df.shape[1]>=2:
    st.subheader('Numeric feature correlation')
    fig,ax=plt.subplots(figsize=(10,6)); sns.heatmap(num_df.corr(numeric_only=True),cmap='vlag',center=0,ax=ax); fig.tight_layout(); st.pyplot(fig); plt.close(fig)

st.header('3 · Compare regression models')
st.caption('Both models use the same holdout set. Lower MAE/RMSE are better; R² closer to 1 is generally better.')
st.dataframe(metrics.style.format({'MAE':'{:,.2f}','RMSE':'{:,.2f}','R²':'{:.3f}'}),use_container_width=True)
fig,ax=plt.subplots(figsize=(8,4)); metrics.set_index('Model')[['MAE','RMSE']].plot(kind='bar',ax=ax); ax.set_ylabel('Error (price units)'); ax.tick_params(axis='x',rotation=0); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
st.info(f'Model selected for prediction: **{best_name}** (based on lower test-set MAE).')
st.subheader('Actual vs predicted prices')
fig,ax=plt.subplots(figsize=(7,5)); ax.scatter(y_test,best_res['predictions'],alpha=.75); low=min(float(y_test.min()),float(np.min(best_res['predictions']))); high=max(float(y_test.max()),float(np.max(best_res['predictions']))); ax.plot([low,high],[low,high],'--'); ax.set_xlabel('Actual sale price'); ax.set_ylabel('Predicted sale price'); fig.tight_layout(); st.pyplot(fig); plt.close(fig)

st.header('4 · Feature importance')
if best_name.startswith('Random'):
    try:
        names=best_model.named_steps['preprocessor'].get_feature_names_out()
        imps=best_model.named_steps['regressor'].feature_importances_
        imp_df=pd.DataFrame({'Feature':names,'Importance':imps}).sort_values('Importance',ascending=False).head(15)
        fig,ax=plt.subplots(figsize=(9,5)); imp_df.sort_values('Importance').plot(kind='barh',x='Feature',y='Importance',legend=False,ax=ax); ax.set_xlabel('Importance'); ax.set_ylabel(''); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    except Exception: st.info('Feature importance is unavailable for this dataset.')
else: st.caption('Random Forest was not selected by MAE, so the selected model is Linear Regression.')

st.header('5 · Predict a house price')
st.caption('Enter property details. Input fields are generated from the dataset columns.')
with st.form('prediction_form'):
    values={}; input_cols=st.columns(2)
    for i,feature in enumerate(X.columns):
        with input_cols[i%2]:
            if feature in numeric_features:
                vals=pd.to_numeric(X[feature],errors='coerce').dropna()
                if vals.empty: lo,hi,default=0.0,1.0,0.0
                else:
                    lo,hi,default=float(vals.min()),float(vals.max()),float(vals.median())
                    if lo==hi: lo-=1; hi+=1
                values[feature]=st.number_input(str(feature),min_value=lo,max_value=hi,value=min(max(default,lo),hi),key=f'pred_{feature}')
            else:
                opts=X[feature].dropna().astype(str).value_counts().index.tolist() or ['Unknown']
                values[feature]=st.selectbox(str(feature),opts,key=f'pred_{feature}')
    go=st.form_submit_button('🏠 Estimate sale price',use_container_width=True)
if go:
    new_house=pd.DataFrame([values],columns=X.columns)
    for col in numeric_features: new_house[col]=pd.to_numeric(new_house[col],errors='coerce')
    try:
        price=float(best_model.predict(new_house)[0]); st.success(f'Estimated sale price: {price:,.2f}')
        st.caption(f'Estimate produced by {best_name}; it is not a professional property valuation.')
    except Exception as exc: st.error(f'Prediction failed: {exc}')

st.header('6 · Export and case-study notes')
st.download_button('Download model comparison CSV',metrics.to_csv(index=False).encode('utf-8'),'house_price_model_metrics.csv','text/csv')
with st.expander('Case study summary and limitations'):
    st.markdown('''**Objective:** Predict `SalePrice` from housing characteristics.\n\n**Workflow:** Data cleaning → EDA → missing-value imputation and categorical encoding → train/test split → Linear Regression and Random Forest Regression → MAE, RMSE and R² comparison → price prediction.\n\n**Limitation:** A small synthetic dataset of around 300 records is useful for an academic case study but does not guarantee reliable real-world valuations. Validate the model before practical use.''')
st.markdown('<div class="footer">HomeValue · House Price Prediction · Linear Regression & Random Forest Regression</div>',unsafe_allow_html=True)
