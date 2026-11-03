import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# ---------------- LOAD DATASET ----------------
df = pd.read_csv("../datasets/dataset.csv")

X = df.drop("target", axis=1)
y = df["target"]

# ---------------- MODEL ----------------
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y)

# ---------------- SAVE PATH ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "multi_disease_project", "models")


os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "heart_model.joblib")

joblib.dump(model, MODEL_PATH)

print("✅ Heart Disease model trained & saved at:", MODEL_PATH)
