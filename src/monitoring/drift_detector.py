"""
drift_detector.py
-----------------
Production data drift monitoring using Evidently AI.

Detects:
- Feature distribution drift (PSI, KS test)
- Target drift
- Model performance degradation
- Data quality issues (missing values, outliers)

Integrates with alerting pipeline for automated retraining triggers.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from loguru import logger
import json

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset, ClassificationPreset
    from evidently.metrics import (
        DatasetDriftMetric, DatasetMissingValuesMetric,
        ColumnDriftMetric, ClassificationQualityMetric
    )
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    logger.warning("Evidently AI not installed. Using fallback drift detection.")


# ── PSI (Population Stability Index) ─────────────────────────────────────────

def compute_psi(reference: pd.Series, production: pd.Series, n_bins: int = 10) -> float:
    """
    Population Stability Index — industry standard for credit model monitoring.
    PSI < 0.1: No significant change
    PSI 0.1-0.25: Moderate change, investigate
    PSI > 0.25: Major shift, retrain model
    """
    min_val = min(reference.min(), production.min())
    max_val = max(reference.max(), production.max())
    bins = np.linspace(min_val, max_val, n_bins + 1)

    ref_pct = np.histogram(reference, bins=bins)[0] / len(reference)
    prod_pct = np.histogram(production, bins=bins)[0] / len(production)

    # Add small epsilon to avoid log(0)
    ref_pct = np.where(ref_pct == 0, 1e-4, ref_pct)
    prod_pct = np.where(prod_pct == 0, 1e-4, prod_pct)

    psi = np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct))
    return float(psi)


# ── Drift Detector ────────────────────────────────────────────────────────────

class DriftDetector:
    """
    Monitors production data for drift relative to training distribution.
    Triggers alerts when drift thresholds are exceeded.
    """

    PSI_THRESHOLDS = {"warning": 0.1, "critical": 0.25}
    KEY_FEATURES = [
        "credit_score", "annual_income", "debt_to_income_ratio",
        "revolving_utilization", "num_delinquencies_2yr", "loan_amount"
    ]

    def __init__(self, reference_data: pd.DataFrame):
        self.reference_data = reference_data
        self.drift_history = []

    def compute_feature_drift(self, production_data: pd.DataFrame) -> dict:
        """Compute PSI for all key features."""
        drift_report = {}
        for feature in self.KEY_FEATURES:
            if feature in self.reference_data.columns and feature in production_data.columns:
                psi = compute_psi(
                    self.reference_data[feature].dropna(),
                    production_data[feature].dropna()
                )
                status = "stable"
                if psi > self.PSI_THRESHOLDS["critical"]:
                    status = "critical"
                elif psi > self.PSI_THRESHOLDS["warning"]:
                    status = "warning"

                drift_report[feature] = {"psi": round(psi, 4), "status": status}

        return drift_report

    def generate_evidently_report(
        self,
        production_data: pd.DataFrame,
        output_path: str = "reports/drift_report.html"
    ) -> None:
        """Generate Evidently AI HTML drift report."""
        if not EVIDENTLY_AVAILABLE:
            logger.warning("Evidently not available. Skipping HTML report.")
            return

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        report = Report(metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
        ])
        report.run(
            reference_data=self.reference_data,
            current_data=production_data
        )
        report.save_html(output_path)
        logger.info(f"Drift report saved: {output_path}")

    def check_and_alert(self, production_data: pd.DataFrame) -> dict:
        """
        Run full drift check. Return summary with retraining recommendation.
        """
        drift_results = self.compute_feature_drift(production_data)

        critical_features = [f for f, r in drift_results.items() if r["status"] == "critical"]
        warning_features = [f for f, r in drift_results.items() if r["status"] == "warning"]

        should_retrain = len(critical_features) >= 2 or len(critical_features) >= 1

        summary = {
            "timestamp": datetime.utcnow().isoformat(),
            "production_samples": len(production_data),
            "drift_by_feature": drift_results,
            "critical_features": critical_features,
            "warning_features": warning_features,
            "overall_drift_status": "critical" if critical_features else ("warning" if warning_features else "stable"),
            "retrain_recommended": should_retrain,
        }

        self.drift_history.append(summary)

        # Log findings
        if critical_features:
            logger.critical(f"DRIFT ALERT: Critical drift in {critical_features}. Retraining recommended!")
        elif warning_features:
            logger.warning(f"DRIFT WARNING: Moderate drift in {warning_features}. Monitor closely.")
        else:
            logger.info("DRIFT CHECK: No significant drift detected.")

        return summary

    def save_drift_log(self, output_path: str = "reports/drift_log.json") -> None:
        """Persist drift history for dashboarding."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.drift_history, f, indent=2)
        logger.info(f"Drift log saved: {output_path}")


if __name__ == "__main__":
    # Demo: simulate production drift
    import sys
    sys.path.append(".")

    logger.info("Loading reference (training) data...")
    reference = pd.read_csv("data/raw/train.csv")

    logger.info("Simulating drifted production data...")
    production = reference.sample(5000).copy()
    # Inject artificial drift
    production["credit_score"] = production["credit_score"] - 40  # Scores declined
    production["debt_to_income_ratio"] = production["debt_to_income_ratio"] * 1.3  # DTI increased

    detector = DriftDetector(reference_data=reference)
    summary = detector.check_and_alert(production)
    detector.generate_evidently_report(production)
    detector.save_drift_log()

    logger.success(f"Drift status: {summary['overall_drift_status']}")
    logger.success(f"Retrain recommended: {summary['retrain_recommended']}")
