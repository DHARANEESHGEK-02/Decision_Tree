import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# -----------------------------
# Title
# -----------------------------
st.title("Salary Prediction using Decision Tree")
st.write("Upload a CSV file to train and predict salary")

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload salaries CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    # Read CSV
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # -----------------------------
    # Validate Required Columns
    # -----------------------------
    required_cols = ['company', 'job', 'degree', 'salary_more_then_100k']
    if not all(col in df.columns for col in required_cols):
        st.error(
            "CSV must contain columns: company, job, degree, salary_more_then_100k"
        )
        st.stop()

    # -----------------------------
    # Prepare Data
    # -----------------------------
    inputs = df.drop('salary_more_then_100k', axis='columns')
    target = df['salary_more_then_100k']

    le_company = LabelEncoder()
    le_job = LabelEncoder()
    le_degree = LabelEncoder()

    inputs['company_n'] = le_company.fit_transform(inputs['company'])
    inputs['job_n'] = le_job.fit_transform(inputs['job'])
    inputs['degree_n'] = le_degree.fit_transform(inputs['degree'])

    inputs_n = inputs.drop(['company', 'job', 'degree'], axis='columns')

    # -----------------------------
    # Train Model
    # -----------------------------
    model = DecisionTreeClassifier()
    model.fit(inputs_n, target)

    accuracy = model.score(inputs_n, target)
    st.success(f"Model Accuracy: {accuracy:.2f}")

    # -----------------------------
    # User Input Section
    # -----------------------------
    st.subheader("Enter Details for Prediction")

    company = st.selectbox("Company", df['company'].unique())
    job = st.selectbox("Job Role", df['job'].unique())
    degree = st.selectbox("Degree", df['degree'].unique())

    # Encode inputs
    company_n = le_company.transform([company])[0]
    job_n = le_job.transform([job])[0]
    degree_n = le_degree.transform([degree])[0]

    # -----------------------------
    # Prediction
    # -----------------------------
    if st.button("Predict Salary"):
        prediction = model.predict([[company_n, job_n, degree_n]])

        if prediction[0] == 1:
            st.success("💰 Salary is likely MORE than 100K")
        else:
            st.warning("💼 Salary is likely LESS than or equal to 100K")

else:
    st.info("Please upload a CSV file to continue.")

