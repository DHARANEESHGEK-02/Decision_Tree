import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import plotly.express as px

st.set_page_config(layout="wide", page_icon="💰")
st.title("💼 Salary Predictor")

uploaded_file = st.file_uploader("📁 Upload salaries.csv", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Loaded {df.shape[0]} rows")
    
    # DYNAMIC COLUMN DETECTION - NO HARDCODING
    st.write("**Exact columns:**", df.columns.tolist())
    
    # Find salary column automatically (handles any name variation)
    salary_col = next(col for col in df.columns if 'salary' in col.lower())
    st.write(f"**Using salary column:** `{salary_col}`")
    
    st.dataframe(df.head(10), use_container_width=True)
    
    # SAFE Charts - uses detected column name
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x="company", color=salary_col)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        job_stats = df.groupby("job")[salary_col].mean().reset_index()
        fig = px.bar(job_stats, x="job", y=salary_col)
        st.plotly_chart(fig, use_container_width=True)
    
    # Model - uses detected column
    inputs = df.drop(salary_col, axis=1)
    target = df[salary_col]
    
    le_company = LabelEncoder()
    le_job = LabelEncoder()
    le_degree = LabelEncoder()
    
    inputs["companyn"] = le_company.fit_transform(inputs["company"])
    inputs["jobn"] = le_job.fit_transform(inputs["job"])
    inputs["degreen"] = le_degree.fit_transform(inputs["degree"])
    
    inputsn = inputs[["companyn", "jobn", "degreen"]]
    
    model = tree.DecisionTreeClassifier()
    model.fit(inputsn, target)
    
    st.metric("🎯 Model Accuracy", f"{model.score(inputsn, target):.1%}")
    
    # Prediction
    st.header("🔮 Predict Salary >100k")
    col1, col2, col3 = st.columns(3)
    company = col1.selectbox("🏢 Company", df["company"].unique())
    job = col2.selectbox("💼 Job", df["job"].unique())
    degree = col3.selectbox("🎓 Degree", df["degree"].unique())
    
    if st.button("🚀 Predict", type="primary"):
        test = np.array([[le_company.transform([company])[0],
                          le_job.transform([job])[0],
                          le_degree.transform([degree])[0]]])
        pred = model.predict(test)[0]
        prob = model.predict_proba(test)[0][1]
        
        if pred == 1:
            st.success(f"**💰 YES >$100k** (Conf: {prob:.0%})")
        else:
            st.error(f"**📉 NO ≤$100k** (Conf: {1-prob:.0%})")

else:
    st.info("👆 Upload file to see data & charts")
    st.stop()
