import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Machine Failure Predictor",
    page_icon="⚙️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: #0b0f14;
    }

    [data-testid="stHeader"] {
        background: rgba(11, 15, 20, 0.85);
    }

    .block-container {
        max-width: 980px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }

    .hero {
        background: linear-gradient(135deg, #151b24 0%, #10151d 100%);
        border: 1px solid #29313d;
        border-radius: 22px;
        padding: 30px 32px 26px 32px;
        margin-bottom: 22px;
        box-shadow: 0 16px 45px rgba(0, 0, 0, 0.24);
    }

    .hero-icon {
        font-size: 2.3rem;
        margin-bottom: 4px;
    }

    .hero h1 {
        margin: 0;
        font-size: 2.35rem;
        line-height: 1.15;
        letter-spacing: -0.8px;
    }

    .hero p {
        margin: 10px 0 0 0;
        color: #aeb8c6;
        font-size: 1.02rem;
    }

    .section-title {
        font-size: 1.12rem;
        font-weight: 700;
        margin: 8px 0 3px 2px;
    }

    .section-subtitle {
        color: #8e99a8;
        font-size: 0.88rem;
        margin: 0 0 16px 2px;
    }

    .info-card {
        background: #111720;
        border: 1px solid #252e3a;
        border-radius: 16px;
        padding: 15px 17px;
        margin: 4px 0 18px 0;
        color: #b7c1ce;
        font-size: 0.88rem;
    }

    .result-card {
        border-radius: 18px;
        padding: 22px 24px;
        margin-top: 20px;
        border: 1px solid #303947;
        background: #111720;
    }

    .result-label {
        color: #8e99a8;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-size: 0.72rem;
        font-weight: 700;
    }

    .result-title {
        font-size: 1.45rem;
        font-weight: 750;
        margin-top: 5px;
    }

    .probability {
        font-size: 2.2rem;
        font-weight: 800;
        margin-top: 4px;
    }

    .metric-card {
        background: #111720;
        border: 1px solid #252e3a;
        border-radius: 14px;
        padding: 13px 15px;
        text-align: center;
        min-height: 78px;
    }

    .metric-label {
        color: #8e99a8;
        font-size: 0.74rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    .metric-value {
        color: #f3f6fa;
        font-size: 1.05rem;
        font-weight: 700;
        margin-top: 4px;
    }

    .footer {
        text-align: center;
        color: #687484;
        font-size: 0.76rem;
        margin-top: 30px;
    }

    div[data-testid="stForm"] {
        background: #10161f;
        border: 1px solid #252e3a;
        border-radius: 18px;
        padding: 22px 22px 8px 22px;
    }

    div[data-testid="stFormSubmitButton"] button {
        height: 3rem;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.98rem;
    }

    div[data-testid="stNumberInput"] input,
    div[data-testid="stSelectbox"] > div {
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Load model
# -----------------------------
MODEL_PATH = Path(__file__).resolve().parent / "models" / "machine_failure_model.joblib"
model = joblib.load(MODEL_PATH)

# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-icon">⚙️</div>
        <h1>Industrial Machine<br>Failure Predictor</h1>
        <p>Estimate machine failure risk from current operating conditions.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-title">Operating Conditions</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Enter the machine readings below and run the trained ML model.</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="info-card">💡 <b>Tip:</b> Use readings from the same measurement conditions used by the model. The result is a predictive prototype, not an engineering diagnosis.</div>',
    unsafe_allow_html=True,
)

# -----------------------------
# Inputs
# -----------------------------
with st.form("prediction_form"):
    col1, col2 = st.columns(2, gap="large")

    with col1:
        ptype = st.selectbox(
            "Product Type",
            ["L", "M", "H"],
            help="Product quality/type category from the AI4I 2020 dataset.",
        )
        air = st.number_input(
            "Air Temperature [K]",
            min_value=295.0,
            max_value=305.0,
            value=300.0,
            step=0.1,
            format="%.1f",
        )
        process = st.number_input(
            "Process Temperature [K]",
            min_value=305.0,
            max_value=314.0,
            value=310.0,
            step=0.1,
            format="%.1f",
        )

    with col2:
        rpm = st.number_input(
            "Rotational Speed [rpm]",
            min_value=1100,
            max_value=3000,
            value=1500,
            step=1,
        )
        torque = st.number_input(
            "Torque [Nm]",
            min_value=0.0,
            max_value=80.0,
            value=40.0,
            step=0.1,
            format="%.1f",
        )
        wear = st.number_input(
            "Tool Wear [min]",
            min_value=0,
            max_value=260,
            value=100,
            step=1,
        )

    submitted = st.form_submit_button("🔍  Predict Machine Failure", type="primary", use_container_width=True)

# -----------------------------
# Prediction
# -----------------------------
if submitted:
    sample = pd.DataFrame(
        [
            {
                "Type": ptype,
                "Air temperature [K]": air,
                "Process temperature [K]": process,
                "Rotational speed [rpm]": rpm,
                "Torque [Nm]": torque,
                "Tool wear [min]": wear,
            }
        ]
    )

    pred = int(model.predict(sample)[0])
    prob = float(model.predict_proba(sample)[0, 1])

    if pred:
        title = "⚠️ Failure Risk Detected"
        description = "The model predicts a machine failure under these operating conditions."
    else:
        title = "✅ No Failure Predicted"
        description = "The model does not predict a machine failure under these operating conditions."

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">Model Prediction</div>
            <div class="result-title">{title}</div>
            <div style="color:#aeb8c6; margin-top:6px;">{description}</div>
            <div class="probability">{prob:.1%}</div>
            <div style="color:#8e99a8; font-size:0.84rem;">Estimated probability of machine failure</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.progress(prob, text=f"Failure probability: {prob:.1%}")

    st.markdown("### Input Summary")
    m1, m2, m3 = st.columns(3)
    m4, m5, m6 = st.columns(3)

    metrics = [
        (m1, "Product", ptype),
        (m2, "Air Temp", f"{air:.1f} K"),
        (m3, "Process Temp", f"{process:.1f} K"),
        (m4, "Speed", f"{rpm:,} rpm"),
        (m5, "Torque", f"{torque:.1f} Nm"),
        (m6, "Tool Wear", f"{wear} min"),
    ]

    for col, label, value in metrics:
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>',
                unsafe_allow_html=True,
            )

st.markdown(
    '<div class="footer">AI4I 2020 Predictive Maintenance Dataset · Random Forest prototype · For educational/demo use</div>',
    unsafe_allow_html=True,
)
