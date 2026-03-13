import streamlit as st
import numpy as np
import pickle

st.title("Student Loan & Scholarship Eligibility Predictor")

model = pickle.load(open("student_loan_model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))

age = st.number_input("Age", 18, 25)
gender = st.selectbox("Gender (0=Female,1=Male)", [0,1])
gpa = st.number_input("GPA", 5.0, 10.0, step=0.01)
attendance = st.number_input("Attendance (%)", 50, 100)
extracurriculars = st.number_input("Extracurricular Activities", 0, 10)
family_income = st.number_input("Family Income ($)", 0, 100000, step=500)
parent_education = st.selectbox("Parent Education (0=HS,1=BA,2=MA,3=PhD)", [0,1,2,3])
scholarship_history = st.selectbox("Scholarship History (0=No,1=Yes)", [0,1])
loan_history = st.selectbox("Loan History (0=No,1=Yes)", [0,1])
community_hours = st.number_input("Community Service Hours", 0, 50)
study_hours = st.number_input("Study Hours per Week", 0, 50)
internship = st.selectbox("Internship Experience (0=No,1=Yes)", [0,1])
academic_awards = st.number_input("Academic Awards", 0, 10)
research_pub = st.selectbox("Research Publications (0=No,1=Yes)", [0,1])
sports = st.selectbox("Sports Activities (0=No,1=Yes)", [0,1])

if st.button("Predict Eligibility"):
    data = np.array([[age, gender, gpa, attendance, extracurriculars, family_income,
                      parent_education, scholarship_history, loan_history,
                      community_hours, study_hours, internship, academic_awards,
                      research_pub, sports]])
    data_scaled = scaler.transform(data)
    pred = model.predict(data_scaled)[0]
    
    if pred == 1:
        st.success("Congratulations! You are likely eligible for a loan or scholarship.")
    else:
        st.error("Unfortunately, you are not eligible based on the provided information.")