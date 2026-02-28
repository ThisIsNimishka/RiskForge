"""
explain.py
----------
SHAP-based model explainability for regulatory compliance.
Generates per-applicant explanations and global feature importance.

Regulatory relevance:
- GDPR Article 22: Right to explanation for automated decisions
- Equal Credit Opportunity Act (ECOA): Adverse action notices
- Fair Housing Act: Non-discriminatory lending reasons
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from loguru import logger
from typing import Optional


class CreditRiskExplainer:
    """
    Wraps SHAP TreeExplainer for the stacking ensemble.
    Provides individual applicant explanations and global importance.
    """

    def __init__(self, model, feature_names: list):
        self.model = model
        self.feature_names = feature_names
        self._explainer = None
        self._background_data = None

    def fit_explainer(self, X_background: pd.DataFrame, n_background: int = 100) -> None:
        """
        Fit SHAP explainer using background dataset for kernel approximation.
        Uses the final estimator (meta-learner) for stacking models.
        """
        logger.info("Fitting SHAP explainer...")

        # Use kmeans summary of background data for efficiency
        background = shap.sample(X_background, n_background, random_state=42)
        self._background_data = background

        # For stacking ensemble, use the model's predict_proba directly
        self._explainer = shap.KernelExplainer(
            lambda x: self.model.predict_proba(x)[:, 1],
            background,
            link="logit"
        )
        logger.info("SHAP explainer ready.")

    def explain_instance(self, X_instance: pd.DataFrame, n_top: int = 5) -> dict:
        """
        Explain a single prediction with SHAP values.
        Returns human-readable factor labels for adverse action notices.
        """
        if self._explainer is None:
            raise RuntimeError("Call fit_explainer() first.")

        shap_values = self._explainer.shap_values(X_instance, nsamples=100)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        shap_df = pd.DataFrame({
            "feature": self.feature_names,
            "shap_value": shap_values[0],
            "feature_value": X_instance.values[0]
        }).sort_values("shap_value", key=abs, ascending=False)

        # Generate human-readable labels
        def make_label(row):
            feature_labels = {
                "credit_score": ("High credit score reduces risk", "Low credit score increases risk"),
                "revolving_utilization": ("Low card utilization is positive", "High card utilization is risky"),
                "debt_to_income_ratio": ("Manageable debt load", "High debt-to-income ratio"),
                "employment_years": ("Stable employment history", "Limited employment history"),
                "num_delinquencies_2yr": ("Clean payment record", "Recent payment delinquencies"),
                "annual_income": ("Strong income level", "Income may not support loan"),
                "num_credit_inquiries": ("Few recent inquiries", "Multiple recent credit inquiries"),
                "credit_health_score": ("Strong overall credit profile", "Credit profile needs improvement"),
            }
            pos_label, neg_label = feature_labels.get(
                row["feature"],
                (f"{row['feature']} favorable", f"{row['feature']} unfavorable")
            )
            return pos_label if row["shap_value"] < 0 else neg_label

        shap_df["label"] = shap_df.apply(make_label, axis=1)

        top_positive = shap_df[shap_df["shap_value"] < 0].head(n_top)
        top_negative = shap_df[shap_df["shap_value"] > 0].head(n_top)

        return {
            "base_value": float(self._explainer.expected_value),
            "top_positive_factors": top_positive[["feature", "shap_value", "label"]].to_dict("records"),
            "top_negative_factors": top_negative[["feature", "shap_value", "label"]].to_dict("records"),
            "all_shap_values": shap_df[["feature", "shap_value"]].to_dict("records"),
        }

    def plot_waterfall(self, X_instance: pd.DataFrame, save_path: Optional[str] = None) -> None:
        """Generate SHAP waterfall plot for a single prediction."""
        shap_values = self._explainer.shap_values(X_instance, nsamples=200)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=self._explainer.expected_value,
                data=X_instance.values[0],
                feature_names=self.feature_names
            )
        )
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()

    def plot_global_importance(self, X_sample: pd.DataFrame, save_path: Optional[str] = None) -> None:
        """Generate global SHAP summary plot."""
        logger.info("Computing global SHAP values (this may take a few minutes)...")
        shap_values = self._explainer.shap_values(X_sample, nsamples=50)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=self.feature_names, show=False)
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
        logger.info("Global importance plot saved.")


def load_explainer(model_path: str = "models/stacking_ensemble.pkl",
                   features_path: str = "models/feature_names.pkl") -> CreditRiskExplainer:
    """Load fitted model and return explainer instance."""
    model = joblib.load(model_path)
    feature_names = joblib.load(features_path)
    return CreditRiskExplainer(model, feature_names)
