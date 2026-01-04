import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(page_title="Salary Prediction", layout="wide")

st.title("💼 Salary Prediction using Decision Tree")
st.write("Upload a CSV file and predict whether salary is greater than 100K")

# -------------------------------------------------
# File Upload
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "📂 Upload Salary CSV file",
    type=["csv"]
)

if uploaded_file is None:
    st.info("👆 Please upload a CSV file to continue.")
    st.stop()

# -------------------------------------------------
# Load CSV
# -------------------------------------------------
df = pd.read_csv(uploaded_file)

# -------------------------------------------------
# Normalize column names (CRITICAL FIX)
# -------------------------------------------------
df.columns = df.columns.str.strip().str.lower()

# Fix known target column variations
if "salarymorethen100k" in df.columns:
    df = df.rename(columns={"salarymorethen100k": "salary_more_then_100k"})
elif "salary_more_than_100k" in df.columns:
    df = df.rename(columns={"salary_more_than_100k": "salary_more_then_100k"})

# -------------------------------------------------
# Validate required columns
# -------------------------------------------------
required_cols = {"company", "job", "degree", "salary_more_then_100k"}

if not required_cols.issubset(df.columns):
    st.error(f"❌ CSV must contain columns: {required_cols}")
    st.write("Detected columns:", list(df.columns))
    st.stop()

# -------------------------------------------------
# Preview Data
# -------------------------------------------------
st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

# -------------------------------------------------
# Charts
# -------------------------------------------------
st.subheader("📊 Data Insights")

col1, col2 = st.columns(2)

with col1:
    fig1 = px.histogram(
        df,
        x="company",
        color="salary_more_then_100k",
        barmode="group",
        title="Company vs Salary Distribution"
    )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    job_salary = (
        df.groupby("job", as_index=False)["salary_more_then_100k"]
        .mean()
    )

    fig2 = px.bar(
        job_salary,
        x="job",
        y="salary_more_then_100k",
        title="Average Salary >100K by Job"
    )
    st.plotly_chart(fig2, use_container_width=True)

# -------------------------------------------------
# Prepare Data for Model
# -------------------------------------------------
inputs = df.drop("salary_more_then_100k", axis=1)
target = df["salary_more_then_100k"]

le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()

inputs["company_n"] = le_company.fit_transform(inputs["company"])
inputs["job_n"] = le_job.fit_transform(inputs["job"])
inputs["degree_n"] = le_degree.fit_transform(inputs["degree"])

inputs_n = inputs.drop(["company", "job", "degree"], axis=1)

# -------------------------------------------------
# Train Model
# -------------------------------------------------
model = DecisionTreeClassifier()
model.fit(inputs_n, target)

accuracy = model.score(inputs_n, target)
st.success(f"✅ Model Accuracy: {accuracy:.2f}")

# -------------------------------------------------
# Prediction Section
# -------------------------------------------------
st.subheader("🔮 Predict Salary")

company = st.selectbox("Company", sorted(df["company"].unique()))
job = st.selectbox("Job Role", sorted(df["job"].unique()))
degree = st.selectbox("Degree", sorted(df["degree"].unique()))

company_n = le_company.transform([company])[0]
job_n = le_job.transform([job])[0]
degree_n = le_degree.transform([degree])[0]

if st.button("Predict Salary"):
    prediction = model.predict([[company_n, job_n, degree_n]])

    if prediction[0] == 1:
        st.success("💰 Salary is MORE than 100K")
    else:
        st.warning("💼 Salary is LESS than or equal to 100K")
