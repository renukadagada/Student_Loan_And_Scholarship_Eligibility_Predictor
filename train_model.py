import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

data = pd.read_csv("student_loan_data.csv")

X = data.drop("Eligibility", axis=1)
y = data["Eligibility"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=200)

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

print("Model Accuracy:", accuracy)

pickle.dump(model, open("student_loan_model.pkl","wb"))
pickle.dump(scaler, open("scaler.pkl","wb"))

print("Model and scaler saved!")
