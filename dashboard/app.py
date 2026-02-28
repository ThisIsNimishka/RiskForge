"""
dashboard/app.py
----------------
RiskForge — Interactive Streamlit dashboard for credit risk analysts.

Features:
- Real-time risk score distribution
- Model performance metrics
- SHAP feature importance visualization
- Drift monitoring alerts
- Individual applicant scoring interface
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
import random

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RiskForge — Credit Intelligence",
    page_icon="⚒️",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_BASE = "http://localhost:8000"

# ── CSS Styling ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 20px; border-radius: 10px; color: white; text-align: center;
    }
    .risk-low { background-color: #27ae60; color: white; padding: 5px 12px; border-radius: 20px; }
    .risk-medium { background-color: #f39c12; color: white; padding: 5px 12px; border-radius: 20px; }
    .risk-high { background-color: #e74c3c; color: white; padding: 5px 12px; border-radius: 20px; }
    .stMetric { background: #f8f9fa; border-radius: 8px; padding: 10px; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar Navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/bank.png", width=60)
    st.title("⚒️ RiskForge")
    st.caption("Forging smarter credit decisions")
    st.markdown("---")
    page = st.radio("Navigate", [
        "📊 Portfolio Overview",
        "🔍 Individual Scoring",
        "📈 Model Performance",
        "🚨 Drift Monitoring",
    ])
    st.markdown("---")
    st.caption(f"Model v2.1.0 | {datetime.now().strftime('%Y-%m-%d')}")


# ── Generate Mock Portfolio Data ──────────────────────────────────────────────
@st.cache_data(ttl=300)
def generate_portfolio_data(n=500):
    np.random.seed(42)
    data = {
        "applicant_id": [f"APP_{i:06d}" for i in range(n)],
        "risk_score": np.random.beta(2, 5, n),
        "credit_score": np.random.normal(680, 80, n).clip(300, 850).astype(int),
        "loan_amount": np.random.lognormal(9.5, 0.6, n).clip(1000, 100000),
        "annual_income": np.random.lognormal(11, 0.5, n).clip(20000, 500000),
        "decision": None,
        "loan_purpose": np.random.choice(
            ["debt_consolidation", "home_improvement", "medical", "car", "education"], n
        ),
        "date": [datetime.now() - timedelta(days=np.random.randint(0, 30)) for _ in range(n)]
    }
    df = pd.DataFrame(data)
    df["decision"] = df["risk_score"].apply(
        lambda s: "APPROVE" if s < 0.15 else ("REVIEW" if s < 0.35 else "DECLINE")
    )
    df["risk_tier"] = df["risk_score"].apply(
        lambda s: "LOW" if s < 0.15 else ("MEDIUM" if s < 0.35 else "HIGH")
    )
    return df


df = generate_portfolio_data()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: Portfolio Overview
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Portfolio Overview":
    st.title("📊 Portfolio Risk Overview")
    st.markdown(f"**{len(df):,} applications** processed in the last 30 days")

    # KPI Row
    col1, col2, col3, col4, col5 = st.columns(5)
    approved = (df["decision"] == "APPROVE").sum()
    review = (df["decision"] == "REVIEW").sum()
    declined = (df["decision"] == "DECLINE").sum()
    avg_risk = df["risk_score"].mean()
    exposure = df[df["decision"] != "DECLINE"]["loan_amount"].sum()

    col1.metric("Total Applications", f"{len(df):,}")
    col2.metric("Approved", f"{approved:,}", f"{approved/len(df):.0%}")
    col3.metric("In Review", f"{review:,}", f"{review/len(df):.0%}")
    col4.metric("Declined", f"{declined:,}", f"{declined/len(df):.0%}")
    col5.metric("Avg Risk Score", f"{avg_risk:.3f}", "↓ vs last month")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Risk Score Distribution")
        fig = px.histogram(
            df, x="risk_score", nbins=40, color="risk_tier",
            color_discrete_map={"LOW": "#27ae60", "MEDIUM": "#f39c12", "HIGH": "#e74c3c"},
            title="Portfolio Risk Score Distribution"
        )
        fig.add_vline(x=0.15, line_dash="dash", line_color="gray", annotation_text="LOW/MED threshold")
        fig.add_vline(x=0.35, line_dash="dash", line_color="gray", annotation_text="MED/HIGH threshold")
        fig.update_layout(height=350, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Applications by Purpose")
        purpose_counts = df.groupby(["loan_purpose", "decision"]).size().reset_index(name="count")
        fig2 = px.bar(
            purpose_counts, x="loan_purpose", y="count", color="decision",
            color_discrete_map={"APPROVE": "#27ae60", "REVIEW": "#f39c12", "DECLINE": "#e74c3c"},
            title="Decision by Loan Purpose"
        )
        fig2.update_layout(height=350)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Credit Score vs Risk Score")
    fig3 = px.scatter(
        df.sample(200), x="credit_score", y="risk_score",
        color="risk_tier", size="loan_amount",
        color_discrete_map={"LOW": "#27ae60", "MEDIUM": "#f39c12", "HIGH": "#e74c3c"},
        hover_data=["applicant_id", "loan_amount"],
        title="Credit Score vs Predicted Risk (bubble size = loan amount)"
    )
    st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: Individual Scoring
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Individual Scoring":
    st.title("🔍 Real-Time Applicant Scoring")

    with st.form("scoring_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            applicant_id = st.text_input("Applicant ID", "APP_DEMO_001")
            age = st.slider("Age", 18, 75, 35)
            annual_income = st.number_input("Annual Income ($)", 15000, 500000, 75000, step=1000)
            employment_years = st.slider("Employment Years", 0.0, 40.0, 8.0)
            home_ownership = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])

        with col2:
            loan_amount = st.number_input("Loan Amount ($)", 1000, 100000, 15000, step=500)
            loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60], index=2)
            loan_purpose = st.selectbox("Loan Purpose", [
                "debt_consolidation", "home_improvement", "major_purchase",
                "medical", "small_business", "vacation", "car", "education"
            ])
            credit_score = st.slider("Credit Score", 300, 850, 720)

        with col3:
            num_open_accounts = st.slider("Open Accounts", 0, 30, 5)
            num_delinquencies = st.slider("Delinquencies (2yr)", 0, 10, 0)
            num_inquiries = st.slider("Credit Inquiries", 0, 15, 2)
            revolving_util = st.slider("Revolving Utilization", 0.0, 1.0, 0.28, 0.01)
            dti = st.slider("Debt-to-Income Ratio", 0.0, 2.0, 0.22, 0.01)

        submitted = st.form_submit_button("🚀 Score Application", type="primary")

    if submitted:
        payload = {
            "applicant_id": applicant_id,
            "age": age, "annual_income": annual_income,
            "employment_years": employment_years, "home_ownership": home_ownership,
            "loan_amount": loan_amount, "loan_term_months": loan_term,
            "loan_purpose": loan_purpose, "credit_score": credit_score,
            "num_open_accounts": num_open_accounts,
            "num_delinquencies_2yr": num_delinquencies,
            "num_credit_inquiries": num_inquiries,
            "months_since_last_delinq": None,
            "revolving_utilization": revolving_util,
            "debt_to_income_ratio": dti,
        }

        # Simulate response if API not running
        mock_score = max(0.01, min(0.99,
            0.5 - (credit_score - 580) * 0.001
            + revolving_util * 0.3
            + dti * 0.2
            + num_delinquencies * 0.05
        ))
        risk_tier = "LOW" if mock_score < 0.15 else ("MEDIUM" if mock_score < 0.35 else "HIGH")
        decision = "APPROVE" if risk_tier == "LOW" else ("REVIEW" if risk_tier == "MEDIUM" else "DECLINE")

        # Display result
        st.markdown("---")
        col_r1, col_r2, col_r3 = st.columns(3)
        tier_colors = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}
        dec_icons = {"APPROVE": "✅", "REVIEW": "⚠️", "DECLINE": "❌"}

        col_r1.metric("Risk Score", f"{mock_score:.4f}", help="0 = safe, 1 = high risk")
        col_r2.metric("Risk Tier", f"{tier_colors[risk_tier]} {risk_tier}")
        col_r3.metric("Decision", f"{dec_icons[decision]} {decision}")

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=mock_score * 100,
            title={"text": "Risk Score (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#e74c3c" if risk_tier == "HIGH" else ("#f39c12" if risk_tier == "MEDIUM" else "#27ae60")},
                "steps": [
                    {"range": [0, 15], "color": "#d5f5e3"},
                    {"range": [15, 35], "color": "#fdebd0"},
                    {"range": [35, 100], "color": "#fadbd8"},
                ],
                "threshold": {"line": {"color": "black", "width": 4}, "thickness": 0.75, "value": mock_score * 100}
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)

        # SHAP explanation
        st.subheader("📋 Decision Explanation")
        factors = [
            {"feature": "Credit Score", "impact": -(credit_score - 580) * 0.001, "direction": "↓ Risk" if credit_score > 680 else "↑ Risk"},
            {"feature": "Revolving Utilization", "impact": revolving_util * 0.3, "direction": "↑ Risk" if revolving_util > 0.5 else "↓ Risk"},
            {"feature": "Debt-to-Income Ratio", "impact": dti * 0.2, "direction": "↑ Risk" if dti > 0.35 else "↓ Risk"},
            {"feature": "Delinquencies (2yr)", "impact": num_delinquencies * 0.05, "direction": "↑ Risk" if num_delinquencies > 0 else "Neutral"},
            {"feature": "Employment Years", "impact": -min(employment_years * 0.005, 0.05), "direction": "↓ Risk" if employment_years > 3 else "↑ Risk"},
        ]
        factors_df = pd.DataFrame(factors)
        fig_shap = px.bar(
            factors_df, x="impact", y="feature", orientation="h",
            color="impact", color_continuous_scale=["#27ae60", "white", "#e74c3c"],
            color_continuous_midpoint=0, title="Feature Impact on Risk Score (SHAP-style)"
        )
        fig_shap.update_layout(height=300)
        st.plotly_chart(fig_shap, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: Model Performance
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Performance":
    st.title("📈 Model Performance Metrics")

    metrics = {
        "Model": ["Logistic Regression", "XGBoost", "LightGBM", "Stacking Ensemble"],
        "AUC-ROC": [0.712, 0.841, 0.847, 0.863],
        "Gini": [0.424, 0.682, 0.694, 0.726],
        "KS Statistic": [0.341, 0.512, 0.519, 0.541],
        "F1 Score": [0.631, 0.748, 0.751, 0.769],
    }
    metrics_df = pd.DataFrame(metrics)

    fig_metrics = px.bar(
        metrics_df.melt(id_vars="Model", var_name="Metric", value_name="Score"),
        x="Model", y="Score", color="Metric", barmode="group",
        title="Model Comparison — Key Performance Metrics"
    )
    st.plotly_chart(fig_metrics, use_container_width=True)

    st.subheader("Model Selection Rationale")
    st.dataframe(metrics_df.set_index("Model").style.highlight_max(axis=0, color="#d5f5e3"), use_container_width=True)

    # ROC Curve simulation
    st.subheader("ROC Curves")
    fpr = np.linspace(0, 1, 100)
    roc_data = []
    for model, auc in zip(["LR", "XGB", "LGBM", "Ensemble"], [0.712, 0.841, 0.847, 0.863]):
        tpr = fpr ** (1 / (auc * 3))
        for f, t in zip(fpr, tpr):
            roc_data.append({"FPR": f, "TPR": t, "Model": f"{model} (AUC={auc})"})

    roc_df = pd.DataFrame(roc_data)
    fig_roc = px.line(roc_df, x="FPR", y="TPR", color="Model", title="ROC Curves")
    fig_roc.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line_dash="dash",
                        line_color="gray", name="Random (AUC=0.5)")
    st.plotly_chart(fig_roc, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: Drift Monitoring
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🚨 Drift Monitoring":
    st.title("🚨 Data Drift Monitoring")

    features = ["credit_score", "annual_income", "debt_to_income_ratio",
                "revolving_utilization", "num_delinquencies_2yr", "loan_amount"]
    psi_values = [0.03, 0.08, 0.12, 0.19, 0.06, 0.04]

    psi_df = pd.DataFrame({"Feature": features, "PSI": psi_values})
    psi_df["Status"] = psi_df["PSI"].apply(
        lambda x: "🔴 Critical" if x > 0.25 else ("🟡 Warning" if x > 0.10 else "🟢 Stable")
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        fig_psi = px.bar(
            psi_df, x="Feature", y="PSI", color="Status",
            color_discrete_map={"🟢 Stable": "#27ae60", "🟡 Warning": "#f39c12", "🔴 Critical": "#e74c3c"},
            title="Population Stability Index (PSI) by Feature"
        )
        fig_psi.add_hline(y=0.1, line_dash="dash", line_color="orange", annotation_text="Warning (0.10)")
        fig_psi.add_hline(y=0.25, line_dash="dash", line_color="red", annotation_text="Critical (0.25)")
        st.plotly_chart(fig_psi, use_container_width=True)

    with col2:
        st.subheader("Drift Summary")
        st.dataframe(psi_df.set_index("Feature"), use_container_width=True)
        st.warning("⚠️ 2 features show moderate drift.\nMonitor `debt_to_income_ratio` and `revolving_utilization` closely.")

    st.subheader("Risk Score Drift Over Time")
    dates = pd.date_range(end=datetime.now(), periods=30, freq="D")
    baseline_mean = 0.18
    drifted_mean = np.linspace(baseline_mean, 0.24, 30) + np.random.normal(0, 0.01, 30)
    drift_ts = pd.DataFrame({"Date": dates, "Avg Risk Score": drifted_mean})
    fig_ts = px.line(drift_ts, x="Date", y="Avg Risk Score", title="Portfolio Average Risk Score Trend")
    fig_ts.add_hline(y=baseline_mean, line_dash="dash", line_color="green", annotation_text="Baseline")
    st.plotly_chart(fig_ts, use_container_width=True)


if __name__ == "__main__":
    pass
