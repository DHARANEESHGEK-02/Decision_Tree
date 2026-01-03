import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="💼 Salary Predictor", layout="wide", page_icon="💰")
st.markdown("## 💼 Salary >100k Predictor (Decision Tree) ✨")

# File Uploader
uploaded_file = st.file_uploader("📁 Upload salaries.csv", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.session_state.df = df
    st.success("✅ File uploaded!")
else:
    st.info("👆 Upload your salaries.csv to start")
    st.stop()

df = st.session_state.df
st.dataframe(df.head(10), use_container_width=True)

# Sidebar Metrics
st.sidebar.header("📊 Quick Stats")
st.sidebar.metric("Total Records", len(df))
st.sidebar.metric("High Salary %", f"{df['salarymorethen100k'].mean()*100:.1f}%")

# Charts
col1, col2 = st.columns(2)
with col1:
    fig1 = px.histogram(df, x='company', color='salarymorethen100k', title="Salary by Company")
    st.plotly_chart(fig1, use_container_width=True)
with col2:
    fig2 = px.bar(df.groupby(['job', 'degree'])['salarymorethen100k'].mean().reset_index(),
                  x='job', y='salarymorethen100k', color='degree', title="Avg Salary by Job & Degree")
    st.plotly_chart(fig2, use_container_width=True)

# Model Training (in expander)
with st.expander("🔧 Train Model (Auto-runs)"):
    inputs = df.drop('salarymorethen100k', axis=1)
    target = df['salarymorethen100k']
    
    le_company = LabelEncoder()
    le_job = LabelEncoder()
    le_degree = LabelEncoder()
    
    inputs['companyn'] = le_company.fit_transform(inputs['company'])
    inputs['jobn'] = le_job.fit_transform(inputs['job'])
    inputs['degreen'] = le_degree.fit_transform(inputs['degree'])
    
    inputsn = inputs.drop(['company', 'job', 'degree'], axis=1)
    
    model = tree.DecisionTreeClassifier(random_state=42)
    model.fit(inputsn, target)
    st.session_state.model = model
    st.session_state.le_company = le_company
    st.session_state.le_job = le_job
    st.session_state.le_degree = le_degree
    st.session_state.inputsn = inputsn
    st.session_state.target = target
    st.metric("✅ Accuracy", f"{model.score(inputsn, target):.3f}")

model = st.session_state.model
le_company, le_job, le_degree = st.session_state.le_company, st.session_state.le_job, st.session_state.le_degree

# Prediction Section
st.markdown("### 🎯 Predict Salary >100k")
col1, col2, col3 = st.columns(3)
company = col1.selectbox("🏢 Company", df['company'].unique(), help="Select company")
job = col2.selectbox("💼 Job Role", df['job'].unique())
degree = col3.selectbox("🎓 Degree", df['degree'].unique())

if st.button("🔮 Predict Now", type="primary"):
    test = np.array([[le_company.transform([company])[0],
                      le_job.transform([job])[0],
                      le_degree.transform([degree])[0]]])
    pred = model.predict(test)[0]
    prob = model.predict_proba(test)[0][1]
    
    st.markdown("### 📈 Result")
    if pred == 1:
        st.success(f"**Yes!** Expected >$100k (Probability: {prob:.1%}) 💰")
    else:
        st.error(f"**No.** Expected ≤$100k (Probability: {1-prob:.1%}) 📉")
    
    # History (simple)
    if 'history' not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append((company, job, degree, pred, prob))
    st.dataframe(pd.DataFrame(st.session_state.history, 
                              columns=['Company', 'Job', 'Degree', 'Pred', 'Prob>100k'])[-5:])

st.caption("Built with Streamlit & Scikit-learn | Data from your CSV[file:1][file:2]")
