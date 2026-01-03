import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import plotly.express as px

st.set_page_config(page_title="💼 Salary Predictor", layout="wide", page_icon="💰")
st.markdown("## 💼 Salary >100k Predictor ✨")

# File Uploader with persistence
if 'df' not in st.session_state:
    st.session_state.df = None

uploaded_file = st.file_uploader("📁 Upload salaries.csv", type="csv")
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.success(f"✅ Uploaded! Shape: {df.shape} | Columns: {list(df.columns)}")
        st.write("Preview:")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"❌ Error: {e}")

if st.session_state.df is None:
    st.warning("👆 Please upload CSV first")
    st.stop()

df = st.session_state.df.copy()

# Auto-detect target column
salary_cols = [col for col in df.columns if 'salary' in col.lower()]
if not salary_cols:
    st.error("❌ No salary column found!")
    st.stop()
target_col = salary_cols[0]
st.info(f"📊 Using target: **{target_col}**")

# Sidebar Stats
st.sidebar.header("📈 Stats")
st.sidebar.metric("Records", len(df))
st.sidebar.metric("High Salary %", f"{df[target_col].mean()*100:.1f}%")

# Charts
col1, col2 = st.columns(2)
with col1:
    fig1 = px.histogram(df, x='company', color=target_col, title="By Company")
    st.plotly_chart(fig1, use_container_width=True)
with col2:
    fig2 = px.bar(df.groupby(['job', 'degree'])[target_col].mean().reset_index(),
                  x='job', y=target_col, color='degree', title="Avg by Job/Degree")
    st.plotly_chart(fig2, use_container_width=True)

# Model Training
@st.cache_data
def train_model(_df):
    inputs = _df.drop(target_col, axis=1)
    target = _df[target_col]
    
    encoders = {}
    for col in ['company', 'job', 'degree']:
        if col in inputs.columns:
            le = LabelEncoder()
            inputs[f'{col}n'] = le.fit_transform(inputs[col])
            encoders[col] = le
    
    inputsn = inputs.drop([col for col in ['company', 'job', 'degree'] if col in inputs.columns], axis=1)
    
    model = tree.DecisionTreeClassifier(random_state=42)
    model.fit(inputsn, target)
    return model, encoders, inputsn, target

if 'model' not in st.session_state:
    with st.spinner("Training model..."):
        st.session_state.model, st.session_state.encoders, st.session_state.inputsn, st.session_state.target = train_model(df)

model = st.session_state.model
encoders = st.session_state.encoders

st.metric("✅ Accuracy", f"{model.score(st.session_state.inputsn, st.session_state.target):.3f}")

# Prediction
st.markdown("### 🎯 Predict")
col1, col2, col3 = st.columns(3)
company = col1.selectbox("🏢 Company", options=sorted(df['company'].unique()))
job = col2.selectbox("💼 Job", options=sorted(df['job'].unique()))
degree = col3.selectbox("🎓 Degree", options=sorted(df['degree'].unique()))

if st.button("🔮 Predict", type="primary", use_container_width=True):
    test_features = {}
    for col in ['company', 'job', 'degree']:
        if col in encoders:
            test_features[f'{col}n'] = encoders[col].transform([locals()[col]])[0]
    
    test = np.array([list(test_features.values())])
    pred = model.predict(test)[0]
    prob = model.predict_proba(test)[0][1]
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Prediction", "Yes >100k 💰" if pred == 1 else "No ≤100k 📉")
    with col_b:
        st.metric("Probability", f"{prob:.1%}")

st.caption("🔧 Fixed for exact CSV columns[file:1][file:2]")
