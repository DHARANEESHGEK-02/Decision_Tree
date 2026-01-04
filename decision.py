import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import plotly.express as px

st.set_page_config(layout="wide", page_icon="💰")
st.title("💼 Salary Predictor")

# File Upload - NO STOP
uploaded_file = st.file_uploader("📁 Upload salaries.csv", type="csv")

df = None
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Loaded {df.shape[0]} rows")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Charts immediately
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x="company", color="salarymorethen100k")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(df.groupby("job")["salarymorethen100k"].mean().reset_index())
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("👆 Upload file to see data & charts")
    st.stop()

# Model - exact column names from your CSV
inputs = df.drop("salarymorethen100k", axis=1)
target = df["salarymorethen100k"]

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

st.caption("✅ Tested with your exact CSV[file:1]")
