# src/predict.py
import warnings
warnings.filterwarnings("ignore")

import os
import joblib
import pandas as pd


# ---------------------------------
# Resolve project root safely
# ---------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


def predict_disease(disease, user_input):
    """
    disease: diabetes / heart / asthma
    user_input: dict
    returns: (risk_percent: float, accuracy_percent: float | None)
    """

    # -------------------------------
    # LOAD MODEL + PREPROCESSOR
    # -------------------------------
    model_path = os.path.join(MODELS_DIR, f"{disease}_model.pkl")
    preprocessor_path = os.path.join(MODELS_DIR, f"{disease}_preprocessor.pkl")
    accuracy_path = os.path.join(MODELS_DIR, f"{disease}_accuracy.pkl")

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    if os.path.exists(accuracy_path):
        accuracy = joblib.load(accuracy_path)
        accuracy_percent = float(round(accuracy * 100, 2))
    else:
        accuracy_percent = None

    imputer = preprocessor["imputer"]
    scaler = preprocessor["scaler"]
    features = preprocessor["features"]

    # -------------------------------
    # ALIGN INPUT FEATURES
    # -------------------------------
    X = pd.DataFrame([user_input])

    for f in features:
        if f not in X.columns:
            X[f] = 0

    X = X[features]

    # sklearn transformers return NumPy arrays
    X = imputer.transform(X)
    X = scaler.transform(X)

    # -------------------------------
    # PREDICTION
    # -------------------------------
    prob = model.predict_proba(X)[0][1]

    # ⬇️ CRITICAL FIX: force Python float
    risk_percent = float(round(prob * 100, 2))

    return risk_percent, accuracy_percent
