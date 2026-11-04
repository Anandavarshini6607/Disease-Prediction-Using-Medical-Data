
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Load dataset
data = pd.read_csv("heart.csv")

print("Dataset loaded successfully!")
print(data.head())

# 2. Check dataset information
print(data.info())
print(data.isnull().sum())

# 3. Separate input and output
X = data.drop("target", axis=1)
y = data["target"]

# 4. Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. Create Logistic Regression model
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])

# 6. Train the model
print("Training started...")
model.fit(X_train, y_train)

# 7. Test the model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy * 100, "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 8. Display confusion matrix
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.title("Heart Disease Prediction")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 9. Compare with Random Forest
rf_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print(
    "Random Forest Accuracy:",
    accuracy_score(y_test, rf_pred) * 100,
    "%"
)

# 10. Save the Logistic Regression model
joblib.dump(model, "heart_model.pkl")

print("Model saved successfully!")


