import time
import joblib
import pandas as pd
import streamlit as st
from log_utils import log_prediction

st.set_page_config(page_title="Profitability Prediction App", layout="centered")
st.title("Transaction Profitability Prediction with Monitoring")

@st.cache_resource
def load_models():
    old_model = joblib.load("model_v1.pkl")  # trained on Quantity only
    new_model = joblib.load("model_v2.pkl")  # trained on all features
    return old_model, new_model

old_model, new_model = load_models()

if "pred_ready" not in st.session_state:
    st.session_state["pred_ready"] = False

# ----------------- INPUT -----------------
st.sidebar.header("Input Parameters")

details_df = pd.read_csv("Details.csv")  # read once

quantity = st.sidebar.number_input("Quantity", min_value=1, value=1)
amount = st.sidebar.number_input("Amount", min_value=0.0, value=100.0)
category = st.sidebar.selectbox("Category", sorted(details_df["Category"].unique()))
sub_category = st.sidebar.selectbox("Sub-Category", sorted(details_df["Sub-Category"].unique()))
payment_mode = st.sidebar.selectbox("Payment Mode", sorted(details_df["PaymentMode"].unique()))

input_df = pd.DataFrame([{
    "Quantity": quantity,
    "Amount": amount,
    "Category": category,
    "Sub-Category": sub_category,
    "PaymentMode": payment_mode
}])

st.subheader("Input Summary")
st.write(input_df)

# ----------------- PREDICTION -----------------
if st.button("Run Prediction"):
    start = time.time()

    # Baseline model: only the features it was trained on
    old_pred = old_model.predict(input_df[["Quantity"]])[0]

    # Improved model: all features
    new_pred = new_model.predict(input_df)[0]

    latency_ms = (time.time() - start) * 1000

    st.session_state.update({
        "pred_ready": True,
        "old_pred": old_pred,
        "new_pred": new_pred,
        "latency_ms": latency_ms,
        "summary": f"Qty={quantity}, Amt={amount}, Cat={category}, Sub={sub_category}, Pay={payment_mode}"
    })

# ----------------- SHOW RESULTS -----------------
if st.session_state.get("pred_ready", False):
    st.subheader("Predictions")
    st.write(f"Baseline v1 Profitability (0=Loss,1=Profit): {st.session_state['old_pred']}")
    st.write(f"Improved v2 Profitability (0=Loss,1=Profit): {st.session_state['new_pred']}")
    st.write(f"Latency: {st.session_state['latency_ms']:.1f} ms")

    # Feedback
    feedback_score = st.slider("Feedback Score (1‑5)", 1, 5, 4)
    feedback_text = st.text_area("Comments (optional)")

    if st.button("Submit Feedback"):
        log_prediction(
            "v1", "baseline", st.session_state["summary"],
            st.session_state["old_pred"], st.session_state["latency_ms"],
            feedback_score, feedback_text
        )
        log_prediction(
            "v2", "improved", st.session_state["summary"],
            st.session_state["new_pred"], st.session_state["latency_ms"],
            feedback_score, feedback_text
        )
        st.success("Logged to monitoring_logs.csv!")
