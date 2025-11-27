import streamlit as st
import requests
import json

# -----------------------------
# Streamlit UI Configuration
# -----------------------------
st.set_page_config(
    page_title="Teacher Allocation RL System",
    layout="centered"
)

st.title("📘 Teacher Allocation Reinforcement Learning System")
st.write("This interface interacts with your RL model API running at **http://127.0.0.1:8000/predict**")

API_URL = "http://127.0.0.1:8000/predict"

# -----------------------------
# Input Section
# -----------------------------
st.header("📊 Input School Data")

n_classes = 6  # must match your environment

# Input fields
students = []
prev_coverage = []

st.subheader("Students per Class")
for i in range(n_classes):
    students.append(
        st.number_input(
            f"Class {i+1} students",
            min_value=0,
            max_value=2000,
            value=40,
            step=1
        )
    )

st.subheader("Previous Coverage (0–1)")
for i in range(n_classes):
    prev_coverage.append(
        st.number_input(
            f"Class {i+1} coverage",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.01,
            format="%.2f"
        )
    )

available_teachers = st.number_input(
    "Available Teachers (e.g. 10)",
    min_value=0,
    max_value=30,
    value=10
)

# -----------------------------
# Submit Button
# -----------------------------
if st.button("Predict Allocation"):
    payload = {
        "students": students,
        "prev_coverage": prev_coverage,
        "available_teachers": available_teachers
    }

    st.write("📨 Sending data to RL model...")
    try:
        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            result = response.json()

            st.success("Prediction Success!")

            st.subheader("🧠 RL Model Output")
            st.json(result)

            st.subheader("📌 Teacher Allocation Pattern")
            alloc = result["allocation"]

            for i, a in enumerate(alloc):
                st.write(f"➡️ Class {i+1}: **{a} teacher(s)**")

        else:
            st.error(f"API Error: {response.status_code}")
            st.write(response.text)

    except Exception as e:
        st.error("Could not connect to the API. Make sure it is running!")
        st.write(str(e))
