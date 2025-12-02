import os
import gradio as gr
import pandas as pd
import numpy as np
import joblib
import sys
import logging

# Basic logging so Render logs show useful info
logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------
# Load your trained churn model (pipeline)
# -------------------------------------------------
MODEL_PATH = "churn_model.joblib"

try:
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.exception(f"Failed to load model at {MODEL_PATH}: {e}")
    raise RuntimeError(f"Failed to load model at {MODEL_PATH}: {e}")

# -------------------------------------------------
# Feature names used by the model
# -------------------------------------------------
FEATURES = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember"
]

# -------------------------------------------------
# Build input DataFrame for prediction
# -------------------------------------------------
def build_df(CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember):
    data = {
        "CreditScore": CreditScore,
        "Age": Age,
        "Tenure": Tenure,
        "Balance": Balance,
        "NumOfProducts": NumOfProducts,
        "HasCrCard": HasCrCard,
        "IsActiveMember": IsActiveMember
    }

    df = pd.DataFrame([data], columns=FEATURES)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

    logger.debug("Final DF: %s", df.to_dict())
    return df

# -------------------------------------------------
# Prediction Function
# -------------------------------------------------
def predict_churn(CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember):

    X = build_df(CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember)

    # prediction
    try:
        pred = int(model.predict(X)[0])
    except Exception as e:
        logger.exception("Model prediction error")
        return f"Model prediction failed: {e}"

    # probability
    prob = None
    try:
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(X)[0][1])
    except Exception:
        prob = None

    label = "🔴 Customer Will Churn" if pred == 1 else "🟢 Customer Will NOT Churn"

    if prob is not None:
        label += f"\n\nChurn Probability: {prob:.3f}"

    return label

# -------------------------------------------------
# Gradio Interface
# -------------------------------------------------
inputs = [
    gr.Number(label="CreditScore", value=650),
    gr.Number(label="Age", value=35),
    gr.Number(label="Tenure", value=3),
    gr.Number(label="Balance", value=50000.0),
    gr.Number(label="NumOfProducts", value=1),
    gr.Dropdown([0, 1], label="HasCrCard (0=No, 1=Yes)", value=1),
    gr.Dropdown([0, 1], label="IsActiveMember (0=No, 1=Yes)", value=1),
]

examples = [
    [650, 35, 3, 50000, 1, 1, 1],
    [500, 42, 7, 120000, 2, 0, 0],
]

interface = gr.Interface(
    fn=predict_churn,
    inputs=inputs,
    outputs="text",
    title="Customer Churn Prediction Model",
    description="This model predicts whether a customer will churn using 7 numeric features.",
    examples=examples
)

# -------------------------------------------------
# Launch (works locally and on Render)
# -------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))  # Render supplies PORT env var
    logger.info("Starting Gradio on 0.0.0.0:%s", port)
    interface.launch(server_name="0.0.0.0", server_port=port, share=False)

