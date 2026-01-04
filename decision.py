import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(page_title="Salary Prediction", layout="wide")
st.title("💼 Salary Prediction using Decision Tree")

# -------------------------------------------------
# Upload CSV
# -------------------------------------------------
uploaded_file = st.file_uploader("📂 Upload Salary CSV", type=["csv"])

if uploaded_file is None:
    st.info("👆 Please upload a CSV file to continue")
    st.stop()

df = pd.read_csv(uploaded_file)

# -------------------------------------------------
# Normalize column names
# -------------------------------------------------
df.columns = df.columns.str.strip().str.lower()

# -------------------------------------------------
# Detect target column safely
# -------------------------------------------------
possible_targets = [
    "salary_more_then_100k",
    "salarymorethen100k",
    "salary_more_then100k",
    "salary_more_than_100k"
]

target_col = None
for col in possible_targets:
    if col in df.columns:
        target_col = col
        break

if target_col is None:
    st.error("❌ Could not find salary target column in CSV")
    st.write("Detected columns:", list(df.columns))
    st.stop()

# -------------------------------------------------
# Validate required feature columns
# -------------------------------------------------
required_features = {"company", "job", "degree"}
if not required_features.issubset(df.columns):
    st.error("❌ CSV must contain: company, job, degree")
    st.write("Detected columns:", list(df.columns))
    st.stop()

# -------------------------------------------------
# Dataset Preview
# -------------------------------------------------
st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

# -------------------------------------------------
# Charts (SAFE)
# -------------------------------------------------
st.subheader("📊 Data Insights")

col1, col2 = st.columns(2)

with col1:
    fig1 = px.histogram(
        df,
        x="company",
        color=target_col,
        barmode="group",
        title="Company vs Salary Distribution"
    )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    job_salary = (
        df.groupby("job", as_index=False)[target_col]
        .mean()
    )

    fig2 = px.bar(
        job_salary,
        x="job",
        y=target_col,
        title="Average Salary >100K by Job"
    )
    st.plotly_chart(fig2, use_container_width=True)

# -------------------------------------------------
# Prepare Data for Model
# -------------------------------------------------
X = df[["company", "job", "degree"]]
y = df[target_col]

le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()

X["company_n"] = le_company.fit_transform(X["company"])
X["job_n"] = le_job.fit_transform(X["job"])
X["degree_n"] = le_degree.fit_transform(X["degree"])

X_encoded = X[["company_n", "job_n", "degree_n"]]

# -------------------------------------------------
# Train Model
# -------------------------------------------------
model = DecisionTreeClassifier()
model.fit(X_encoded, y)

accuracy = model.score(X_encoded, y)
st.success(f"✅ Model Accuracy: {accuracy:.2f}")

# -------------------------------------------------
# Prediction Section
# -------------------------------------------------
st.subheader("🔮 Predict Salary")

company = st.selectbox("Company", sorted(df["company"].unique()))
job = st.selectbox("Job Role", sorted(df["job"].unique()))
degree = st.selectbox("Degree", sorted(df["degree"].unique()))

input_data = [[
    le_company.transform([company])[0],
    le_job.transform([job])[0],
    le_degree.transform([degree])[0]
]]

if st.button("Predict Salary"):
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("💰 Salary is MORE than 100K")
    else:
        st.warning("💼 Salary is LESS than or equal to 100K")
