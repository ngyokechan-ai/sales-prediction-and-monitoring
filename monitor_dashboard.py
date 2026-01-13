# monitoring_dashboard.py
import os
import pandas as pd
import streamlit as st
from log_utils import LOG_PATH

st.set_page_config(page_title="Model Monitoring & Feedback", layout="wide")
st.title("Model Monitoring & Feedback Dashboard")

# ---------------- Load Logs ----------------
@st.cache_data
def load_logs():
    if not os.path.exists(LOG_PATH):
        # Return empty dataframe if logs do not exist yet
        return pd.DataFrame(columns=[
            "timestamp", "model_version", "model_type", "input_summary",
            "prediction", "latency_ms", "feedback_score", "feedback_text"
        ])
    df = pd.read_csv(LOG_PATH, parse_dates=["timestamp"])
    return df.sort_values("timestamp")

logs = load_logs()

# Handle empty logs
if logs.empty:
    st.warning(
        "No monitoring logs found yet. "
        "Please run the prediction app, submit feedback at least once, and then refresh this page."
    )
    st.stop()

# ---------------- Sidebar Filters ----------------
st.sidebar.header("Filters")
models = ["All"] + sorted(logs["model_version"].unique().tolist())
selected_model = st.sidebar.selectbox("Model version", models)

if selected_model == "All":
    filtered = logs
else:
    filtered = logs[logs["model_version"] == selected_model]

# ---------------- Key Metrics ----------------
st.subheader("Key Metrics")
col1, col2, col3 = st.columns(3)

col1.metric("Total Predictions", len(filtered))
col2.metric(
    "Avg Feedback Score",
    f"{filtered['feedback_score'].mean():.2f}" if filtered["feedback_score"].notna().any() else "N/A"
)
col3.metric(
    "Avg Latency (ms)",
    f"{filtered['latency_ms'].mean():.1f}" if filtered["latency_ms"].notna().any() else "N/A"
)

st.markdown("---")

# ---------------- Tabs ----------------
tab1, tab2, tab3 = st.tabs(["📊 Model Comparison", "💬 Feedback Analysis", "📄 Raw Logs"])

# ---- Tab 1: Model Comparison ----
with tab1:
    st.subheader("Model Version Comparison (Aggregated)")
    summary = logs.groupby("model_version").agg({
        "feedback_score": "mean",
        "latency_ms": "mean",
    }).rename(columns={
        "feedback_score": "avg_feedback_score",
        "latency_ms": "avg_latency_ms",
    })
    st.dataframe(summary.style.format({
        "avg_feedback_score": "{:.2f}",
        "avg_latency_ms": "{:.1f}",
    }))

# ---- Tab 2: Feedback Analysis ----
with tab2:
    st.subheader("Average Feedback Score by Model Version")
    fb = logs.groupby("model_version")["feedback_score"].mean().reset_index()
    fb_chart = fb.set_index("model_version")
    st.bar_chart(fb_chart)

    st.subheader("Recent Comments")
    comments = logs[logs["feedback_text"].astype(str).str.strip() != ""]
    comments = comments.sort_values("timestamp", ascending=False).head(10)

    if comments.empty:
        st.info("No qualitative comments yet.")
    else:
        for _, row in comments.iterrows():
            st.write(f"**[{row['timestamp']}] {row['model_version']} – Score: {row['feedback_score']}**")
            st.write(row["feedback_text"])
            st.markdown("---")

# ---- Tab 3: Raw Logs ----
with tab3:
    st.subheader("Raw Monitoring Logs")
    st.dataframe(filtered)
