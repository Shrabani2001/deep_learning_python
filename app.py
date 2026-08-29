import streamlit as st
import pandas as pd
import pickle

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide"
)

# -------------------------------------------------
# Custom CSS
# -------------------------------------------------
st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.block-container {
    padding-top: 2rem;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #6b7280;
    font-size: 18px;
    margin-bottom: 30px;
}

.card {
    background-color: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.result-approved {
    background-color: #d1fae5;
    border-left: 8px solid #10b981;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
}

.result-rejected {
    background-color: #fee2e2;
    border-left: 8px solid #ef4444;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
}

.result-approved h2 {
    color: #047857;
}

.result-rejected h2 {
    color: #b91c1c;
}

.stButton > button {
    width: 100%;
    height: 50px;
    border-radius: 10px;
    font-size: 18px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)


# -------------------------------------------------
# Load Model
# -------------------------------------------------
try:
    with open("loan_model.pkl", "rb") as file:
        model = pickle.load(file)

except FileNotFoundError:
    st.error("loan_model.pkl not found. Please run main.py first.")
    st.stop()


# -------------------------------------------------
# Header
# -------------------------------------------------
st.markdown(
    '<div class="title">🏦 Loan Approval Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Machine Learning based Loan Application Prediction</div>',
    unsafe_allow_html=True
)


# -------------------------------------------------
# Main Form
# -------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("📋 Applicant Information")

col1, col2, col3 = st.columns(3)

with col1:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    married = st.selectbox(
        "Married",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["0", "1", "2", "3+"]
    )

    education = st.selectbox(
        "Education",
        ["Graduate", "Not Graduate"]
    )


with col2:

    self_employed = st.selectbox(
        "Self Employed",
        ["Yes", "No"]
    )

    applicant_income = st.number_input(
        "Applicant Income",
        min_value=0,
        value=5000,
        step=500
    )

    coapplicant_income = st.number_input(
        "Coapplicant Income",
        min_value=0,
        value=1000,
        step=500
    )

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=0,
        value=150,
        step=10
    )


with col3:

    loan_term = st.selectbox(
        "Loan Amount Term",
        [120, 180, 240, 300, 360, 480],
        index=4
    )

    credit_history = st.selectbox(
        "Credit History",
        ["Good", "Bad"]
    )

    property_area = st.selectbox(
        "Property Area",
        ["Urban", "Semiurban", "Rural"]
    )


st.markdown("</div>", unsafe_allow_html=True)


# -------------------------------------------------
# Convert UI values to model values
# -------------------------------------------------

gender_value = 1 if gender == "Male" else 0

married_value = 1 if married == "Yes" else 0

dependents_mapping = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3+": 3
}

education_value = 0 if education == "Graduate" else 1

self_employed_value = 1 if self_employed == "Yes" else 0

credit_history_value = 1 if credit_history == "Good" else 0

property_area_mapping = {
    "Urban": 2,
    "Semiurban": 1,
    "Rural": 0
}


# -------------------------------------------------
# Prediction Button
# -------------------------------------------------

st.markdown("<br>", unsafe_allow_html=True)

if st.button("🔍 Predict Loan Status"):

    new_applicant = pd.DataFrame({
        "Gender": [gender_value],
        "Married": [married_value],
        "Dependents": [dependents_mapping[dependents]],
        "Education": [education_value],
        "Self_Employed": [self_employed_value],
        "ApplicantIncome": [applicant_income],
        "CoapplicantIncome": [coapplicant_income],
        "LoanAmount": [loan_amount],
        "Loan_Amount_Term": [loan_term],
        "Credit_History": [credit_history_value],
        "Property_Area": [property_area_mapping[property_area]]
    })

    prediction = model.predict(new_applicant)

    # -------------------------------------------------
    # Result
    # -------------------------------------------------

    st.markdown("<br>", unsafe_allow_html=True)

    if prediction[0] == 1:

        st.markdown("""
        <div class="result-approved">
            <h2>✅ LOAN APPROVED</h2>
            <p>The applicant is predicted to be eligible for the loan.</p>
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div class="result-rejected">
            <h2>❌ LOAN REJECTED</h2>
            <p>The applicant is predicted not to be eligible for the loan.</p>
        </div>
        """, unsafe_allow_html=True)


# -------------------------------------------------
# Footer
# -------------------------------------------------

st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown(
    """
    <div style="text-align:center; color:#6b7280;">
        🤖 Powered by Logistic Regression &nbsp;|&nbsp;
        📊 Machine Learning Classification
    </div>
    """,
    unsafe_allow_html=True
)