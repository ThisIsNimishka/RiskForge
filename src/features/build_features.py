"""
build_features.py
-----------------
Production-grade feature engineering pipeline for credit risk scoring.

Techniques used:
- Weight of Evidence (WOE) encoding for categorical variables
- Interaction feature generation
- Winsorization for outlier handling
- Missing value imputation with business logic
- Feature versioning for reproducibility
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
from loguru import logger


# ── Weight of Evidence Encoder ────────────────────────────────────────────────

class WOEEncoder(BaseEstimator, TransformerMixin):
    """
    Weight of Evidence encoder for categorical features.
    Captures monotonic relationship between category and target.
    Standard in credit risk modeling (Basel II/III compliant).
    """

    def __init__(self, smoothing: float = 0.5):
        self.smoothing = smoothing
        self.woe_maps_ = {}
        self.iv_scores_ = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WOEEncoder":
        total_events = y.sum()
        total_non_events = (1 - y).sum()

        for col in X.columns:
            stats = pd.DataFrame({
                "y": y.values,
                "x": X[col].values
            }).groupby("x")["y"].agg(["sum", "count"])
            stats.columns = ["events", "count"]
            stats["non_events"] = stats["count"] - stats["events"]

            # Smooth to avoid log(0)
            stats["dist_events"] = (stats["events"] + self.smoothing) / (total_events + self.smoothing * len(stats))
            stats["dist_non_events"] = (stats["non_events"] + self.smoothing) / (total_non_events + self.smoothing * len(stats))

            stats["woe"] = np.log(stats["dist_events"] / stats["dist_non_events"])
            stats["iv"] = (stats["dist_events"] - stats["dist_non_events"]) * stats["woe"]

            self.woe_maps_[col] = stats["woe"].to_dict()
            self.iv_scores_[col] = stats["iv"].sum()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        for col in X.columns:
            X_out[col] = X[col].map(self.woe_maps_.get(col, {})).fillna(0)
        return X_out


# ── Custom Feature Engineering ────────────────────────────────────────────────

class CreditFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Domain-driven feature engineering for credit risk.
    All features have clear business interpretation.
    """

    def fit(self, X: pd.DataFrame, y=None) -> "CreditFeatureEngineer":
        self.income_quantiles_ = X["annual_income"].quantile([0.25, 0.5, 0.75]).to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # ── Ratio features ────────────────────────────────────────────
        df["loan_to_income"] = df["loan_amount"] / (df["annual_income"] + 1)
        df["monthly_payment_to_income"] = df["monthly_payment"] / (df["annual_income"] / 12 + 1)

        # ── Credit health score (composite) ──────────────────────────
        df["credit_health_score"] = (
            (df["credit_score"] - 300) / 550 * 40
            - df["revolving_utilization"] * 20
            - df["num_delinquencies_2yr"] * 10
            + np.log1p(df["employment_years"]) * 5
            - df["num_credit_inquiries"] * 2
        ).clip(0, 100)

        # ── Delinquency recency flag ──────────────────────────────────
        df["has_recent_delinquency"] = (df["num_delinquencies_2yr"] > 0).astype(int)
        df["months_since_delinq_imputed"] = df["months_since_last_delinq"].fillna(99)

        # ── Income tier ───────────────────────────────────────────────
        df["income_tier"] = pd.cut(
            df["annual_income"],
            bins=[0, 40_000, 80_000, 150_000, float("inf")],
            labels=[0, 1, 2, 3]
        ).astype(float)

        # ── Risk interaction terms ─────────────────────────────────────
        df["dti_x_revolving"] = df["debt_to_income_ratio"] * df["revolving_utilization"]
        df["credit_score_x_dti"] = df["credit_score"] * (1 - df["debt_to_income_ratio"])

        # ── Log transforms for skewed distributions ───────────────────
        df["log_annual_income"] = np.log1p(df["annual_income"])
        df["log_loan_amount"] = np.log1p(df["loan_amount"])

        return df


# ── Winsorizer ────────────────────────────────────────────────────────────────

class Winsorizer(BaseEstimator, TransformerMixin):
    """Cap outliers at specified percentiles to improve model stability."""

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper
        self.bounds_ = {}

    def fit(self, X: pd.DataFrame, y=None) -> "Winsorizer":
        for col in X.select_dtypes(include=[np.number]).columns:
            self.bounds_[col] = (
                X[col].quantile(self.lower),
                X[col].quantile(self.upper)
            )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        for col, (low, high) in self.bounds_.items():
            if col in df.columns:
                df[col] = df[col].clip(low, high)
        return df


# ── Full Pipeline ─────────────────────────────────────────────────────────────

CATEGORICAL_FEATURES = ["home_ownership", "loan_purpose"]
DROP_COLUMNS = ["applicant_id", "default", "monthly_payment"]


def build_feature_pipeline(train_df: pd.DataFrame) -> dict:
    """
    Build and fit the full feature engineering pipeline.
    Returns fitted transformers and engineered feature dataframe.
    """
    logger.info("Building feature engineering pipeline...")

    y = train_df["default"]
    X = train_df.drop(columns=DROP_COLUMNS, errors="ignore")

    # Step 1: Custom feature engineering
    engineer = CreditFeatureEngineer()
    X_eng = engineer.fit_transform(X, y)

    # Step 2: WOE encode categoricals
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in X_eng.columns]
    woe_encoder = WOEEncoder()
    X_eng[cat_cols] = woe_encoder.fit_transform(X_eng[cat_cols], y)

    # Step 3: Winsorize
    winsorizer = Winsorizer()
    X_winsorized = winsorizer.fit_transform(X_eng.select_dtypes(include=[np.number]))

    # Step 4: Scale (for logistic regression)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_winsorized),
        columns=X_winsorized.columns,
        index=X_winsorized.index
    )

    logger.info(f"Feature matrix shape: {X_scaled.shape}")
    logger.info(f"WOE IV Scores: {woe_encoder.iv_scores_}")

    return {
        "X_processed": X_winsorized,  # For tree models (don't need scaling)
        "X_scaled": X_scaled,          # For linear models
        "y": y,
        "engineer": engineer,
        "woe_encoder": woe_encoder,
        "winsorizer": winsorizer,
        "scaler": scaler,
        "feature_names": list(X_winsorized.columns)
    }


def save_pipeline(artifacts: dict, output_dir: str = "data/processed") -> None:
    """Persist fitted transformers for serving."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    joblib.dump(artifacts["engineer"], out / "feature_engineer.pkl")
    joblib.dump(artifacts["woe_encoder"], out / "woe_encoder.pkl")
    joblib.dump(artifacts["winsorizer"], out / "winsorizer.pkl")
    joblib.dump(artifacts["scaler"], out / "scaler.pkl")

    artifacts["X_processed"].to_csv(out / "X_train.csv", index=False)
    artifacts["y"].to_csv(out / "y_train.csv", index=False)

    logger.success(f"Pipeline artifacts saved to {out}/")


if __name__ == "__main__":
    logger.info("Loading training data...")
    train_df = pd.read_csv("data/raw/train.csv")

    artifacts = build_feature_pipeline(train_df)
    save_pipeline(artifacts)
    logger.success("Feature engineering complete!")
