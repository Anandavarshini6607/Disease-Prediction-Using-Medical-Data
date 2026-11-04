
import pandas as pd
import joblib

# Load trained model
model = joblib.load("heart_model.pkl")

# Enter patient information
patient_data = {
    "age": 45,
    "sex": 1,
    "cp": 0,
    "trestbps": 130,
    "chol": 220,
    "fbs": 0,
    "restecg": 1,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.0,
    "slope": 1,
    "ca": 0,
    "thal": 2
}

# Convert into DataFrame
input_data = pd.DataFrame([patient_data])

# Predict
prediction = model.predict(input_data)[0]

# Display result
if prediction == 1:
    print("Dataset prediction: Heart disease indicated.")
else:
    print("Dataset prediction: No heart disease indicated.")

print("This is an educational model output, not a medical diagnosis.")
