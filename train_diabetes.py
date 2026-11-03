import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import os

# Load dataset
df = pd.read_csv("../datasets/diabetes.csv")

# REMOVE Pregnancies
X = df.drop(["Outcome", "Pregnancies"], axis=1)
y = df["Outcome"]

model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000))
])

model.fit(X, y)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "multi_disease_project", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "diabetes_model.joblib")
joblib.dump(model, MODEL_PATH)

print("✅ Diabetes model trained ")
