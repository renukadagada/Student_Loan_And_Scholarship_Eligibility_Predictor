import pandas as pd
import numpy as np

np.random.seed(42)
n = 800  # small, GitHub-friendly

data = pd.DataFrame({
    "Age": np.random.randint(18,25,n),
    "Gender": np.random.randint(0,2,n),
    "GPA": np.round(np.random.uniform(5,10,n),2),
    "Attendance": np.random.randint(50,100,n),
    "Extracurriculars": np.random.randint(0,5,n),
    "Family_Income": np.random.randint(5000,50000,n),
    "Parent_Education": np.random.randint(0,4,n),
    "Scholarship_History": np.random.randint(0,2,n),
    "Loan_History": np.random.randint(0,2,n),
    "Community_Service_Hours": np.random.randint(0,50,n),
    "Study_Hours_Per_Week": np.random.randint(5,40,n),
    "Internship_Experience": np.random.randint(0,2,n),
    "Academic_Awards": np.random.randint(0,5,n),
    "Research_Publications": np.random.randint(0,2,n),
    "Sports_Activities": np.random.randint(0,2,n)
})

def eligibility(row):
    score = 0
    if row["GPA"] >= 7.5: score += 3
    elif row["GPA"] >= 6.5: score += 2
    if row["Attendance"] >= 75: score += 2
    if row["Extracurriculars"] >= 2: score += 1
    if row["Family_Income"] < 20000: score += 2
    if row["Scholarship_History"] == 0: score += 1
    if row["Loan_History"] == 0: score += 1
    if row["Community_Service_Hours"] >= 10: score += 1
    if row["Study_Hours_Per_Week"] >= 20: score += 1
    if row["Internship_Experience"] == 1: score += 1
    if row["Academic_Awards"] >= 1: score += 1
    if row["Research_Publications"] == 1: score += 2
    if row["Sports_Activities"] == 1: score += 1
    return 1 if score >= 10 else 0

data["Eligibility"] = data.apply(eligibility, axis=1)
data.to_csv("student_loan_data.csv", index=False)
print("Dataset created locally: student_loan_data.csv")