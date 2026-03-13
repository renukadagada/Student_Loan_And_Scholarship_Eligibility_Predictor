import streamlit as st
import numpy as np
import pickle

st.title("Student Loan & Scholarship Eligibility Predictor")

# Load model and scaler
model = pickle.load(open("student_loan_model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))

# User inputs
age = st.number_input("Age",18,25, step=1)
gender = st.selectbox("Gender (0 = Female, 1 = Male)",[0,1])
gpa = st.number_input("GPA",5.0,10.0, step=0.1)
attendance = st.number_input("Attendance %",50,100, step=1)
extra = st.number_input("Extracurricular Activities",0,10, step=1)
income = st.number_input("Family Income",0,100000, step=1000)
parent_edu = st.selectbox("Parent Education (0=HS,1=Bachelor,2=Master,3=PhD)",[0,1,2,3])
scholar = st.selectbox("Scholarship History",[0,1])
loan = st.selectbox("Loan History",[0,1])
community = st.number_input("Community Service Hours",0,100, step=1)
awards = st.number_input("Academic Awards",0,10, step=1)

if st.button("Predict"):

    # Build 2D numpy array with correct dtype
    input_data = np.array([[age, gender, gpa, attendance, extra, income,
                            parent_edu, scholar, loan, community, awards]], dtype=float)

    # Scale input
    try:
        input_scaled = scaler.transform(input_data)
    except ValueError as e:
        st.error(f"Input error: {e}")
    else:
        prediction = model.predict(input_scaled)

        if prediction[0] == 1:
            st.success("Eligible for Loan / Scholarship")
        else:
            st.error("Not Eligible")
