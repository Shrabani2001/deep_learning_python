
import streamlit as st
import numpy as np
import joblib
import tensorflow as tf


# --------------------------------------------------
# 1. Load Trained Model and Scaler
# --------------------------------------------------

model = tf.keras.models.load_model("student_pass_fail_ann.keras")
scaler = joblib.load("student_scaler.pkl")


# --------------------------------------------------
# 2. Streamlit Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Student Result Prediction",
    page_icon="🎓",
    layout="centered"
)


# --------------------------------------------------
# 3. Application Title
# --------------------------------------------------

st.title("🎓 Student Result Prediction")
st.write("Predict whether a student will **Pass or Fail** based on study hours and attendance.")


# --------------------------------------------------
# 4. Input Section
# --------------------------------------------------

st.subheader("Enter Student Details")

study_hours = st.number_input(
    "📚 Study Hours",
    min_value=0.0,
    max_value=24.0,
    value=5.0,
    step=0.5
)

attendance = st.number_input(
    "📅 Attendance (%)",
    min_value=0.0,
    max_value=100.0,
    value=75.0,
    step=1.0
)


# --------------------------------------------------
# 5. Prediction Button
# --------------------------------------------------

if st.button("🔮 Predict Result"):

    # Create input array
    input_data = np.array([[study_hours, attendance]])

    # Scale input using the same scaler used during training
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled, verbose=0)

    # Get probability
    probability = prediction[0][0]

    # Convert probability to Pass/Fail
    if probability >= 0.5:
        result = "PASS"
    else:
        result = "FAIL"


    # --------------------------------------------------
    # 6. Display Result
    # --------------------------------------------------

    st.subheader("Prediction Result")

    if result == "PASS":
        st.success("🎉 Student is likely to PASS!")
    else:
        st.error("❌ Student is likely to FAIL.")

    st.write(f"**Pass Probability:** {probability * 100:.2f}%")
    st.write(f"**Fail Probability:** {(1 - probability) * 100:.2f}%")